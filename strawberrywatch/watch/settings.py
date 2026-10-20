"""
The settings someone can change: checking them, and describing changes for the
review page and History.

Settings are one plain dict, stored as JSON:

    {"dont_flag": [site, ...],          sites still checked but not flagged
     "rain": True,                      allow for rain
     "repeat_hours": 6.0,               ignore repeats of the same thing for this long
     "models": {name: {"on": bool,
                       "scales": {rule: multiple of the tested line},
                       "rain_multiplier": float,
                       "leave_out": [site, ...],
                       "support": [module, ...]}}}
"""

from __future__ import annotations

import copy
import math

# Decided 2026-10-12. Balanced is the tested line everywhere.
PRESETS = (
    ("fewer", "Fewer", 1.25, "Only large, lasting changes."),
    ("balanced", "Balanced", 1.0, "The level the models were tested at."),
    ("more", "More", 0.8, "Smaller changes too, with more false alarms."),
)
# Allowed range for every number someone can type. The forms enforce these, and
# the watcher checks them again when it reads the settings.
RANGES = {
    "scale": (0.75, 1.5),
    "rain_multiplier": (1.0, 3.0),
    "repeat_hours": (0.25, 72.0),
}
SCALE_RANGE = RANGES["scale"]
RAIN_RANGE = RANGES["rain_multiplier"]
REPEAT_RANGE = RANGES["repeat_hours"]

INTERVAL_MINUTES = 15
LOOKBACK_DAYS = 2
# A check has to finish well inside the interval. Another model can't be
# turned on once the last check took more than half of it
CHECK_BUDGET_SECONDS = INTERVAL_MINUTES * 60 / 2
# Cobble Shoal judges a moment once half its sensors report. Leave-outs that
# bring it this close get a warning, and below half they're refused.
COVERAGE_WARN = 0.6

RULE_WORDS = {
    "forecast_residual": ("Forecast threshold", "How far a reading is from the model's forecast."),
    "level_shift": ("Sudden-change threshold", "A sudden change that lasts."),
    "combined_fisher": (
        "Threshold",
        "How unusual each sensor's last 6 hours are, with all its tests combined.",
    ),
    "hampel_spike": ("Spike test", "Spikes: how far a reading is from the readings around it."),
    "cusum_shift": (
        "Shift test",
        "Lasting shifts: a running total of how far readings are from normal.",
    ),
    "ewma_drift": (
        "Drift test",
        "Slow drift: a moving average of how far readings are from normal.",
    ),
}

# Whole-model false alarm rates for Fewer, Balanced and More, the middle three
# of Water_Strider.STOPS. Balanced is exactly the calibrated rate.
PRESET_STOPS = {"fewer": 3e-5, "balanced": 1e-4, "more": 3e-4}

# Allowed on with this note: Cobble Shoal from 2026-10-12, Flame Skimmer from 2026-10-16
NOT_REVALIDATED = {
    "cobble_shoal": (
        "Cobble Shoal hasn't been retrained or re-tested since the node mask change, "
        "so expect some false alerts."
    ),
    "flame_skimmer": (
        "Flame Skimmer is experimental. Only one version has been trained, and its "
        "uncertainty estimate failed its own test."
    ),
}

# The model class behind each serving detector. check_collision reads
# BUILTIN_SUPPORT off the model, and the serving detectors don't carry it.
MODEL_CLASSES = {
    "dusk_crayfish": ("strawberrywatch.models.Dusk_Crayfish", "DuskCrayfish"),
    "cobble_shoal": ("strawberrywatch.models.Cobble_Shoal", "CobbleShoal"),
    "flame_skimmer": ("strawberrywatch.models.Flame_Skimmer", "FlameSkimmer"),
}


# Cautious defaults for whatever the website: block of settings.yaml leaves out
LIMIT_DEFAULTS = {"check_timeout_minutes": 7.0, "min_free_memory_mb": 1024.0, "max_models_on": 2}


def website_limits():
    """
    The website: block of settings.yaml. Read here and not through Config,
    which imports torch. Night Heron's process can read it cheaply.
    """
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parent.parent / "settings.yaml"
    try:
        block = (yaml.safe_load(path.read_text()) or {}).get("website") or {}
    except (OSError, yaml.YAMLError):
        block = {}
    out = {}
    for key, default in LIMIT_DEFAULTS.items():
        try:
            out[key] = type(default)(block.get(key, default))
        except (TypeError, ValueError):
            out[key] = default
    return out


def website_address():
    """Where the website mounts the dashboard, from the same block, for links in emails."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parent.parent / "settings.yaml"
    try:
        block = (yaml.safe_load(path.read_text()) or {}).get("website") or {}
    except (OSError, yaml.YAMLError):
        block = {}
    return str(block.get("address") or "https://www.strawberrycreek.org/strawberrywatch").rstrip(
        "/"
    )


def free_memory_mb():
    """Memory free for a new process now, in MB, or None when it can't be read."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**20
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def recent_costs(checks):
    """
    {model: the most memory one check of it took}, from checks' recorded costs.
    The most, not the latest, since a quiet check can hide what a busy one needs.
    """
    out = {}
    for check in checks:
        for name, mb in (check.get("durations") or {}).get("memory_mb", {}).items():
            out[name] = max(out.get(name, 0.0), float(mb))
    return out


def model_class(name):
    import importlib

    from strawberrywatch.serving import detector_class

    if name in MODEL_CLASSES:
        module, attr = MODEL_CLASSES[name]
        return getattr(importlib.import_module(module), attr)
    return detector_class(name)


def defaults(models):
    """A first run: Dusk Crayfish on with its two explainers, everything at the tested line."""
    explainers = ["newtwork_run", "trial_bed"]
    return {
        "dont_flag": [],
        "rain": True,
        "repeat_hours": 6.0,
        "models": {m.name: model_defaults(m, explainers) for m in models},
    }


def model_defaults(model, explainers=("newtwork_run", "trial_bed")):
    """One model's first settings. A calibrated model starts at its calibrated rate."""
    spec = {
        "on": model.name == "dusk_crayfish",
        "scales": {rule: 1.0 for rule in model.rules},
        "rain_multiplier": model.rain_default,
        "leave_out": [],
        "support": list(explainers) if model.name == "dusk_crayfish" else [],
    }
    if model.sensitivity == "rate":
        spec["rates"] = {rule: _calibrated_rate() for rule in model.rules}
        spec["thresholds"] = {}
    return spec


def with_new_models(settings, models):
    """Saved settings with a default entry, switched off, for any model registered since they were saved."""
    if settings is None:
        return None
    out = copy.deepcopy(settings)
    for model in models:
        if model.name not in out.get("models", {}):
            out.setdefault("models", {})[model.name] = {**model_defaults(model, ()), "on": False}
    return out


def _calibrated_rate():
    from strawberrywatch.models import Water_Strider as wsm

    return wsm.Q_TOTAL


def stops():
    from strawberrywatch.models import Water_Strider as wsm

    return wsm.STOPS


def stop_label(rate):
    for _key, label, value in stops():
        if abs(value - rate) <= value * 1e-6:
            return label
    return f"{rate:g}"


def custom_sensitivity(spec):
    """Whether a calibrated model's settings have moved off what was calibrated."""
    rates = spec.get("rates") or {}
    calibrated = _calibrated_rate()
    return bool(spec.get("thresholds")) or any(abs(r - calibrated) > 1e-15 for r in rates.values())


def site_name(site, inventory=None):
    """What the website calls a site, the inventory name up to its comma: South Fork 1."""
    from strawberrywatch import inventory as inv

    try:
        name = (inventory or inv.load()).site(site).name
    except Exception:
        return site
    return name.split(",")[0].strip() or site


def site_place(site, inventory=None):
    """The landmark after the comma, when the name has one: Stephens Hall."""
    from strawberrywatch import inventory as inv

    try:
        name = (inventory or inv.load()).site(site).name
    except Exception:
        return ""
    return name.split(",", 1)[1].strip() if "," in name else ""


def sensor_words(key):
    """south_fork_1.conductivity.hampel_spike as "the spike test for conductivity at South Fork 1"."""
    site, variable, rule = key.split(".")
    return (
        f"the {RULE_WORDS[rule][0].lower()} for {variable.replace('_', ' ')} at {site_name(site)}"
    )


def alerts_choice(settings):
    """
    fewer, balanced or more if every model sits on that preset (calibrated ones
    at its stop, with no threshold typed by hand). Otherwise custom.
    """
    specs = settings["models"].values()
    scales = {round(v, 4) for spec in specs if "rates" not in spec for v in spec["scales"].values()}
    rates = {round(v, 12) for spec in specs if "rates" in spec for v in spec["rates"].values()}
    typed = any(spec.get("thresholds") for spec in specs)
    for key, _, value, _ in PRESETS:
        if scales <= {value} and rates <= {round(PRESET_STOPS[key], 12)} and not typed:
            return key
    return "custom"


def apply_choice(settings, choice):
    """Set every model to one preset, on or off. Turning a model on later keeps the choice."""
    value = next(v for key, _, v, _ in PRESETS if key == choice)
    out = copy.deepcopy(settings)
    for spec in out["models"].values():
        if "rates" in spec:
            spec["rates"] = {rule: PRESET_STOPS[choice] for rule in spec["rates"]}
            spec["thresholds"] = {}
        else:
            spec["scales"] = {rule: value for rule in spec["scales"]}
    return out


def range_hint(name, unit=""):
    """Between 0.75 and 1.5, said next to a field."""
    lo, hi = RANGES[name]
    return f"Between {lo:g} and {hi:g}{' ' + unit if unit else ''}."


def _number_in(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    lo, hi = RANGES[name]
    return math.isfinite(value) and lo <= value <= hi


def out_of_range(settings):
    """
    Type and range problems in saved settings, for the watcher. Unknown or
    missing models are fine here. A newly registered model can't stop the
    checks.
    """
    if not isinstance(settings, dict):
        return ["The settings aren't a set of values."]
    problems = []
    if not isinstance(settings.get("dont_flag", []), list):
        problems.append("Which sites are never flagged isn't a list.")
    if not _number_in(settings.get("repeat_hours"), "repeat_hours"):
        problems.append("Ignore repeats for is outside its range.")
    models = settings.get("models")
    if not isinstance(models, dict):
        return [*problems, "There are no model settings."]
    for name, spec in models.items():
        if not isinstance(spec, dict):
            problems.append(f"{_title(name)}'s settings aren't a set of values.")
            continue
        for rule, value in (spec.get("scales") or {}).items():
            if not _number_in(value, "scale"):
                problems.append(f"{_title(name)}'s {rule} line is outside its range.")
        if "rain_multiplier" in spec and not _number_in(spec["rain_multiplier"], "rain_multiplier"):
            problems.append(f"{_title(name)}'s rain multiplier is outside its range.")
        for key in ("leave_out", "support"):
            if not isinstance(spec.get(key, []), list):
                problems.append(f"{_title(name)}'s {key.replace('_', ' ')} isn't a list.")
        for rule, rate in (spec.get("rates") or {}).items():
            if not isinstance(rate, int | float) or not 1e-7 <= rate <= 0.1:
                problems.append(f"{_title(name)}'s {rule} rate is outside its range.")
        for key, value in (spec.get("thresholds") or {}).items():
            if not isinstance(value, int | float) or not value > 0:
                problems.append(f"{_title(name)}'s threshold for {key} isn't a number above 0.")
    return problems


def merge(base, mine, theirs):
    """
    Lays my draft over what they applied. Values I changed win, everything else
    stays theirs. Used when someone applied while my draft was open.
    """
    if isinstance(base, dict) and isinstance(mine, dict) and isinstance(theirs, dict):
        out = {}
        for key in {*theirs, *mine}:
            if key not in mine:
                if key in theirs and key not in base:
                    out[key] = theirs[key]
                continue
            if key not in theirs:
                out[key] = mine[key]
                continue
            out[key] = merge(base.get(key), mine[key], theirs[key])
        return out
    return theirs if mine == base else mine


def coverage(model, leave_out):
    """Share of a model's sensors left once some sites are left out."""
    if not model.nodes:
        return 1.0
    kept = [node for node in model.nodes if node[0] not in set(leave_out)]
    return len(kept) / len(model.nodes)


def check(raw, models, sites, applied=None, last_check_seconds=None, limits=None, memory=None):
    """
    Cleans a draft, or says what's wrong with it. Returns (settings, problems,
    warnings). Problems block Apply, warnings don't.

    sites is the inventory's list, and a site name that isn't in it is an
    error. On the website, limits is website_limits() and memory is free
    memory plus each model's recent use. We refuse a change that could run the
    server out of memory.
    """
    from strawberrywatch.anomalies.rain_gate import RainGateConfigError
    from strawberrywatch.serving import UnknownDetector, detector_class
    from strawberrywatch.support_modules import SupportError, registry

    problems, warnings = [], []
    raw = raw if isinstance(raw, dict) else {}
    known_sites = list(sites)
    by_name = {m.name: m for m in models}

    def site_list(values, where):
        out = []
        for value in values or []:
            if value in known_sites:
                if value not in out:
                    out.append(value)
            else:
                problems.append(f"{where}: there's no site called {value!r}")
        return [s for s in known_sites if s in out]

    clean = {
        "dont_flag": site_list(raw.get("dont_flag"), "Which sites to watch"),
        "rain": bool(raw.get("rain", True)),
        "models": {},
    }

    try:
        repeat = float(raw.get("repeat_hours"))
        if not math.isfinite(repeat) or not REPEAT_RANGE[0] <= repeat <= REPEAT_RANGE[1]:
            raise ValueError
        clean["repeat_hours"] = repeat
    except (TypeError, ValueError):
        problems.append(
            f"Ignore repeat alerts for must be between {REPEAT_RANGE[0]:g} and {REPEAT_RANGE[1]:g} hours"
        )
        clean["repeat_hours"] = 6.0

    for name, spec in (raw.get("models") or {}).items():
        try:
            detector_class(name)
        except UnknownDetector as exc:
            problems.append(str(exc))
            continue
        model = by_name[name]
        spec = spec if isinstance(spec, dict) else {}
        out = {"on": bool(spec.get("on")), "scales": {}, "leave_out": [], "support": []}

        for rule in model.rules:
            value = (spec.get("scales") or {}).get(rule, 1.0)
            try:
                value = round(float(value), 4)
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                value = math.nan
            lo, hi = SCALE_RANGE
            if not lo <= value <= hi:
                problems.append(
                    f"{model.title} {RULE_WORDS.get(rule, (rule,))[0].lower()} must be "
                    f"between {lo:g} and {hi:g} times the tested value"
                )
                value = 1.0
            out["scales"][rule] = value

        try:
            multiplier = float(spec.get("rain_multiplier", model.rain_default))
            if not RAIN_RANGE[0] <= multiplier <= RAIN_RANGE[1]:
                raise RainGateConfigError(
                    f"must be between {RAIN_RANGE[0]:g} and {RAIN_RANGE[1]:g}"
                )
            model.gate(True, multiplier)
        except (TypeError, ValueError) as exc:
            problems.append(f"{model.title} rain multiplier {exc}")
            multiplier = model.rain_default
        out["rain_multiplier"] = multiplier

        out["leave_out"] = site_list(spec.get("leave_out"), f"{model.title}, leave a site out")
        if out["leave_out"]:
            if "leave_out" not in model.site_choice:
                problems.append(
                    f"{model.title} can't leave sites out. It needs all "
                    f"{len(model.sites)} of its sites."
                )
            else:
                share = coverage(model, out["leave_out"])
                floor = model.min_coverage or 0.0
                if share < floor:
                    problems.append(
                        f"Leaving out {', '.join(site_name(x) for x in out['leave_out'])} would "
                        f"leave {model.title} with under half its sensors. It stops checking below half."
                    )
                elif share < COVERAGE_WARN:
                    warnings.append(
                        f"{model.title} would keep {share:.0%} of its sensors. It stops checking "
                        f"below half, so one more offline site would stop it."
                    )

        if model.sensitivity == "rate":
            out["scales"] = {rule: 1.0 for rule in model.rules}
            out["rates"], out["thresholds"] = _check_rates(model, spec, problems, warnings)

        support = list(spec.get("support") or [])
        try:
            # support_class, validate and check_collision, in the order main.py
            # runs them, against the model class
            registry.load(support, model_class(name))
            out["support"] = sorted(set(support))
        except SupportError as exc:
            problems.append(str(exc))
        clean["models"][name] = out

    for model in models:
        if model.name not in clean["models"]:
            problems.append(f"No settings for {model.title}")

    before = applied or {"models": {}}
    turning_on = [
        name
        for name, spec in clean["models"].items()
        if spec["on"] and not before["models"].get(name, {}).get("on")
    ]
    if turning_on:
        if turning_on and last_check_seconds and last_check_seconds > CHECK_BUDGET_SECONDS:
            problems.append(
                f"The last check took {last_check_seconds / 60:.0f} minutes, more than half the "
                f"{INTERVAL_MINUTES} minutes between checks, so no more models can be turned on."
            )
        for name in turning_on:
            if name in NOT_REVALIDATED:
                warnings.append(NOT_REVALIDATED[name])
    if limits:
        problems += _over_limits(clean, by_name, turning_on, limits, memory or {}, warnings)
    return clean, problems, warnings


def _check_rates(model, spec, problems, warnings):
    """A calibrated model's slider stops and hand-typed thresholds, cleaned."""
    from strawberrywatch.models import Water_Strider as wsm

    allowed = [value for _, _, value in stops()]
    rates = {}
    for rule in model.rules:
        value = (spec.get("rates") or {}).get(rule, _calibrated_rate())
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        match = next(
            (a for a in allowed if math.isfinite(value) and abs(a - value) <= a * 1e-6), None
        )
        if match is None:
            problems.append(
                f"{model.title} {RULE_WORDS[rule][0].lower()} must be one of the five slider settings"
            )
            match = _calibrated_rate()
        rates[rule] = match
    thresholds = {}
    for key, value in (spec.get("thresholds") or {}).items():
        parts = str(key).split(".")
        if len(parts) != 3 or ".".join(parts[:2]) not in model.tails or parts[2] not in model.rules:
            problems.append(f"{model.title} has no sensor and test called {key!r}")
            continue
        try:
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            problems.append(f"Threshold for {sensor_words(key)} must be a number above 0")
            continue
        thresholds[key] = value
        tail = model.tails[".".join(parts[:2])][parts[2]]
        implied = wsm.rate_for(tail, value)
        lo, hi = wsm.RATE_WINDOW
        if value < tail["t"]:
            warnings.append(
                f"Threshold {value:g} for {sensor_words(key)} is below {tail['t']:.3g}, the lowest "
                f"value the calibration covers, so its false alarm rate is unknown."
            )
        elif not lo <= implied <= hi:
            warnings.append(
                f"Threshold {value:g} for {sensor_words(key)} means a false alarm rate of "
                f"{implied:.1e} per reading, outside the calibrated range ({lo:.0e} to {hi:g})."
            )
    return rates, thresholds


def _over_limits(clean, by_name, turning_on, limits, memory, warnings):
    """The website's load limits, which editors can't raise from the dashboard."""
    problems = []
    on = [name for name, spec in clean["models"].items() if spec["on"]]
    cap = int(limits["max_models_on"])
    if len(on) > cap:
        problems.append(
            f"At most {cap} model{'s' if cap != 1 else ''} can be on at once, to protect memory. "
            "Turn one off first. This limit is set in settings.yaml."
        )
    free, costs = memory.get("free_mb"), memory.get("costs") or {}
    floor = float(limits["min_free_memory_mb"])
    if free is not None and turning_on:
        unknown = [n for n in turning_on if n not in costs]
        extra = sum(costs.get(n, 0.0) for n in turning_on)
        if free - extra < floor:
            titles = ", ".join(by_name[n].title for n in turning_on)
            problems.append(
                f"Not enough memory to turn on {titles}. Recent checks suggest it would leave "
                f"less than {floor:,.0f} MB free. There's {free:,.0f} MB free now."
            )
        elif unknown:
            warnings.append(
                f"Memory use for {', '.join(by_name[n].title for n in unknown)} hasn't been "
                "measured here yet. Run scripts/load_test.py to measure it."
            )
    return problems


def _onoff(value):
    return "On" if value else "Off"


def _list(values):
    return ", ".join(values) if values else "None"


def _sites(values):
    return _list([site_name(v) for v in values])


def _title(name):
    return name.replace("_", " ").title()


def _describe_rates(row, title, model, before, after, choice_after):
    """A calibrated model's stops and typed thresholds, as review rows and History lines."""
    for rule in model.rules:
        rb = (before.get("rates") or {}).get(rule, _calibrated_rate())
        ra = (after.get("rates") or {}).get(rule, _calibrated_rate())
        # Covered by the How many alerts row when every model moved together
        if abs(rb - ra) > 1e-15 and choice_after == "custom":
            words = RULE_WORDS[rule][0].lower()
            row(
                f"{title}, {words}",
                stop_label(rb),
                stop_label(ra),
                f"{title} {words} set to {stop_label(ra).lower()} false alarms.",
            )
    tb, ta = before.get("thresholds") or {}, after.get("thresholds") or {}
    for key in sorted(set(tb) | set(ta)):
        if tb.get(key) == ta.get(key):
            continue
        old = "Calibrated" if key not in tb else f"{tb[key]:g}"
        new = "Calibrated" if key not in ta else f"{ta[key]:g}"
        sentence = (
            f"Threshold for {sensor_words(key)} reset to calibrated."
            if key not in ta
            else f"Threshold for {sensor_words(key)} set to {ta[key]:g}."
        )
        row(f"{title}, {sensor_words(key)}", old, new, sentence)


def describe(before, after, models):
    """
    Every difference between two settings, as rows for the review page:
    [{"setting", "before", "after", "sentence"}]. sentence is the History line.
    """
    rows = []

    def row(setting, old, new, sentence):
        rows.append({"setting": setting, "before": old, "after": new, "sentence": sentence})

    names = {key: label for key, label, _, _ in PRESETS}
    names["custom"] = "Custom"
    choice_before, choice_after = alerts_choice(before), alerts_choice(after)
    if choice_before != choice_after:
        label = names[choice_after]
        sentence = (
            "Alert sensitivity set to custom values in Advanced settings."
            if choice_after == "custom"
            else f"Alert sensitivity set to {label}."
        )
        row("Alert sensitivity", names[choice_before], label, sentence)

    if before["dont_flag"] != after["dont_flag"]:
        sentence = (
            f"Alerts turned off for {_sites(after['dont_flag'])}."
            if after["dont_flag"]
            else "Alerts turned back on for all sites."
        )
        row(
            "Sites with alerts off",
            _sites(before["dont_flag"]),
            _sites(after["dont_flag"]),
            sentence,
        )
    if before["rain"] != after["rain"]:
        row(
            "Rain allowance",
            _onoff(before["rain"]),
            _onoff(after["rain"]),
            f"Rain allowance turned {'on' if after['rain'] else 'off'}.",
        )

    for model in models:
        b = before["models"].get(model.name, {})
        a = after["models"].get(model.name, {})
        title = model.title
        if b.get("on") != a.get("on"):
            row(
                title,
                _onoff(b.get("on")),
                _onoff(a.get("on")),
                f"{title} turned {'on' if a.get('on') else 'off'}.",
            )
        if model.sensitivity == "rate":
            _describe_rates(row, title, model, b, a, choice_after)
        for rule in model.rules if model.sensitivity == "scale" else ():
            sb = (b.get("scales") or {}).get(rule, 1.0)
            sa = (a.get("scales") or {}).get(rule, 1.0)
            # Covered by the How many alerts row when every line moved together
            if abs(sb - sa) > 1e-9 and choice_after == "custom":
                words = RULE_WORDS.get(rule, (rule,))[0]
                row(
                    f"{title}, {words.lower()}",
                    f"{sb:.2f} times",
                    f"{sa:.2f} times",
                    f"{title} {words.lower()} set to {sa:.2f} times the tested value.",
                )
        if b.get("rain_multiplier") != a.get("rain_multiplier"):
            row(
                f"{title}, rain multiplier",
                f"{b.get('rain_multiplier', model.rain_default):g}",
                f"{a.get('rain_multiplier'):g}",
                f"{title} rain multiplier set to {a.get('rain_multiplier'):g}.",
            )
        if b.get("leave_out", []) != a.get("leave_out", []):
            row(
                f"{title}, sites left out",
                _sites(b.get("leave_out", [])),
                _sites(a.get("leave_out", [])),
                f"{title} now leaves out {_sites(a.get('leave_out', []))}."
                if a.get("leave_out")
                else f"{title} now uses all its sites.",
            )
        if b.get("support", []) != a.get("support", []):
            row(
                f"{title}, support modules",
                _list([_title(s) for s in b.get("support", [])]),
                _list([_title(s) for s in a.get("support", [])]),
                f"{title} support modules set to {_list([_title(s) for s in a.get('support', [])])}.",
            )

    if abs(float(before["repeat_hours"]) - float(after["repeat_hours"])) > 1e-9:
        row(
            "Ignore repeat alerts for",
            f"{before['repeat_hours']:g} hours",
            f"{after['repeat_hours']:g} hours",
            f"Repeat alerts ignored for {after['repeat_hours']:g} hours.",
        )
    return rows

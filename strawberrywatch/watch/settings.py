"""
What someone can change, checked with StrawberryWatch's own code, and described
in plain words for the review page and History.

Settings are one plain dict, stored as JSON:

    {"dont_flag": [site, ...],          sites still checked but never flagged
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
    ("fewer", "Fewer", 1.25, "Only clear, lasting changes."),
    ("balanced", "Balanced", 1.0, "What the models were tested at."),
    ("more", "More", 0.8, "Catches smaller changes, with more false alarms."),
)
SCALE_RANGE = (0.75, 1.5)
RAIN_RANGE = (1.0, 3.0)
REPEAT_RANGE = (0.25, 72.0)

INTERVAL_MINUTES = 15
LOOKBACK_DAYS = 2
# A check has to finish well inside the interval, so another model can't be
# turned on once the last check took more than half of it
CHECK_BUDGET_SECONDS = INTERVAL_MINUTES * 60 / 2
# Cobble Shoal judges a moment once half its sensors report. Leave-outs that
# bring it this close get a warning, and below half they're refused.
COVERAGE_WARN = 0.6

RULE_WORDS = {
    "forecast_residual": (
        "Forecast threshold",
        "How far a reading is from what the model expected.",
    ),
    "level_shift": ("Sudden-jump threshold", "A sudden jump that stays."),
    "combined_fisher": (
        "Threshold",
        "How unusual each sensor's last six hours look, with all its checks combined.",
    ),
}

# Cobble Shoal can be turned on (decided 2026-10-12), with this said where it is.
NOT_REVALIDATED = {
    "cobble_shoal": (
        "Cobble Shoal hasn't been retrained with the node mask and checked again "
        "since, so it may notice things that aren't there."
    )
}

# The model class behind each serving detector. check_collision reads
# BUILTIN_SUPPORT off the model, and the serving detectors don't carry it.
MODEL_CLASSES = {
    "dusk_crayfish": ("strawberrywatch.models.Dusk_Crayfish", "DuskCrayfish"),
    "cobble_shoal": ("strawberrywatch.models.Cobble_Shoal", "CobbleShoal"),
}


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
        "models": {
            m.name: {
                "on": m.name == "dusk_crayfish",
                "scales": {rule: 1.0 for rule in m.rules},
                "rain_multiplier": m.rain_default,
                "leave_out": [],
                "support": explainers if m.name == "dusk_crayfish" else [],
            }
            for m in models
        },
    }


def alerts_choice(settings):
    """fewer, balanced or more when every model's every line sits on that preset, otherwise custom."""
    scales = {round(v, 4) for spec in settings["models"].values() for v in spec["scales"].values()}
    for key, _, value, _ in PRESETS:
        if scales == {value}:
            return key
    return "custom"


def apply_choice(settings, choice):
    """Set every line of every model to one preset, on or off, so turning a model on later keeps the choice."""
    value = next(v for key, _, v, _ in PRESETS if key == choice)
    out = copy.deepcopy(settings)
    for spec in out["models"].values():
        spec["scales"] = {rule: value for rule in spec["scales"]}
    return out


def coverage(model, leave_out):
    """Share of a model's sensors left once some sites are left out."""
    if not model.nodes:
        return 1.0
    kept = [node for node in model.nodes if node[0] not in set(leave_out)]
    return len(kept) / len(model.nodes)


def check(raw, models, sites, applied=None, last_check_seconds=None):
    """
    Clean a draft, or say what's wrong with it.

    sites is the inventory's list. Site names are only ever taken from it: a
    name in the draft that isn't there is an error, never stored. Returns
    (settings, problems, warnings). Problems stop Apply. Warnings are worth
    reading but don't.
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
            f"Ignore repeats for has to be between {REPEAT_RANGE[0]:g} and {REPEAT_RANGE[1]:g} hours"
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
                    f"{model.title}'s {RULE_WORDS.get(rule, (rule,))[0].lower()} has to be "
                    f"between {lo:g} and {hi:g} times the tested value"
                )
                value = 1.0
            out["scales"][rule] = value

        try:
            multiplier = float(spec.get("rain_multiplier", model.rain_default))
            if not RAIN_RANGE[0] <= multiplier <= RAIN_RANGE[1]:
                raise RainGateConfigError(
                    f"has to be between {RAIN_RANGE[0]:g} and {RAIN_RANGE[1]:g}"
                )
            model.gate(True, multiplier)
        except (TypeError, ValueError) as exc:
            problems.append(f"{model.title}'s rain multiplier {exc}")
            multiplier = model.rain_default
        out["rain_multiplier"] = multiplier

        out["leave_out"] = site_list(spec.get("leave_out"), f"{model.title}, leave a site out")
        if out["leave_out"]:
            if "leave_out" not in model.site_choice:
                problems.append(
                    f"{model.title} can't leave sites out. It needs all "
                    f"{len(model.sites)} sites until it's retrained with sites dropped."
                )
            else:
                share = coverage(model, out["leave_out"])
                floor = model.min_coverage or 0.0
                if share < floor:
                    problems.append(
                        f"Leaving out {', '.join(out['leave_out'])} would leave {model.title} "
                        f"under half its sensors, and it doesn't judge anything below half."
                    )
                elif share < COVERAGE_WARN:
                    warnings.append(
                        f"{model.title} would keep {share:.0%} of its sensors. Below half it "
                        f"stops judging, so one more quiet site would stop it."
                    )

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
                f"The last check took {last_check_seconds / 60:.0f} minutes, more than half of "
                f"the {INTERVAL_MINUTES} between checks, so another model can't be turned on."
            )
        for name in turning_on:
            if name in NOT_REVALIDATED:
                warnings.append(NOT_REVALIDATED[name])
    return clean, problems, warnings


def _onoff(value):
    return "On" if value else "Off"


def _list(values):
    return ", ".join(values) if values else "None"


def _title(name):
    return name.replace("_", " ").title()


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
            "You changed sensitivity to custom values in Advanced settings."
            if choice_after == "custom"
            else f"You changed sensitivity to {label}{' alerts' if label != 'Balanced' else ''}."
        )
        row("How many alerts", names[choice_before], label, sentence)

    if before["dont_flag"] != after["dont_flag"]:
        sentence = (
            f"You stopped flagging {_list(after['dont_flag'])}."
            if after["dont_flag"]
            else "You set every site to be flagged again."
        )
        row(
            "Sites that are never flagged",
            _list(before["dont_flag"]),
            _list(after["dont_flag"]),
            sentence,
        )
    if before["rain"] != after["rain"]:
        row(
            "Allow for rain",
            _onoff(before["rain"]),
            _onoff(after["rain"]),
            f"You turned {'on' if after['rain'] else 'off'} allowing for rain.",
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
                f"You turned {title} {'on' if a.get('on') else 'off'}.",
            )
        for rule in model.rules:
            sb = (b.get("scales") or {}).get(rule, 1.0)
            sa = (a.get("scales") or {}).get(rule, 1.0)
            # Already said as How many alerts when every line moved together
            if abs(sb - sa) > 1e-9 and choice_after == "custom":
                words = RULE_WORDS.get(rule, (rule,))[0]
                row(
                    f"{title}, {words.lower()}",
                    f"{sb:.2f} times",
                    f"{sa:.2f} times",
                    f"You changed {title}'s {words.lower()} to {sa:.2f} times the tested value.",
                )
        if b.get("rain_multiplier") != a.get("rain_multiplier"):
            row(
                f"{title}, rain multiplier",
                f"{b.get('rain_multiplier', model.rain_default):g}",
                f"{a.get('rain_multiplier'):g}",
                f"You changed {title}'s rain multiplier to {a.get('rain_multiplier'):g}.",
            )
        if b.get("leave_out", []) != a.get("leave_out", []):
            row(
                f"{title}, sites left out",
                _list(b.get("leave_out", [])),
                _list(a.get("leave_out", [])),
                f"You changed which sites {title} leaves out to {_list(a.get('leave_out', []))}.",
            )
        if b.get("support", []) != a.get("support", []):
            row(
                f"{title}, extras",
                _list([_title(s) for s in b.get("support", [])]),
                _list([_title(s) for s in a.get("support", [])]),
                f"You changed {title}'s extras to {_list([_title(s) for s in a.get('support', [])])}.",
            )

    if abs(float(before["repeat_hours"]) - float(after["repeat_hours"])) > 1e-9:
        row(
            "Ignore repeats of the same thing for",
            f"{before['repeat_hours']:g} hours",
            f"{after['repeat_hours']:g} hours",
            f"You changed how long repeats are ignored to {after['repeat_hours']:g} hours.",
        )
    return rows

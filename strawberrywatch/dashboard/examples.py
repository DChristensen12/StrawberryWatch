"""
Made-up alerts for the Alert examples page, one for every sentence an alert can
use, for checking the wording everywhere it shows up. This module doesn't touch
the store.
"""

from __future__ import annotations

import math

from strawberrywatch.dashboard import view

H = 3600
STEP = view.STEP

# (what it shows, model, site, sensor, rule, direction, when, support module output)
SAMPLES = (
    (
        "Sudden change, up",
        "dusk_crayfish",
        "south_fork_1",
        "conductivity",
        "level_shift",
        "up",
        "today",
        {
            "newtwork_run": {
                "verdict": "consistent_with_transport",
                "order": ["south_fork_1", "south_fork_2", "south_fork_3"],
            },
            "trial_bed": {"verdict": "diagnosed", "cause": "sewage"},
        },
    ),
    (
        "Sudden change, down",
        "dusk_crayfish",
        "oxford",
        "conductivity",
        "level_shift",
        "down",
        "yesterday",
        {
            "newtwork_run": {
                "verdict": "consistent_with_transport",
                "order": ["south_fork_3", "oxford"],
            }
        },
    ),
    (
        "Sudden change, direction unclear",
        "dusk_crayfish",
        "kingman_hall",
        "conductivity",
        "level_shift",
        None,
        "this year",
        {"newtwork_run": {"verdict": "catchment_wide", "order": []}},
    ),
    (
        "Forecast",
        "dusk_crayfish",
        "north_fork_0",
        "conductivity",
        "forecast_residual",
        None,
        "last year",
        {"newtwork_run": {"verdict": "not_transport", "order": []}},
    ),
    (
        "Cobble Shoal",
        "cobble_shoal",
        "university_house",
        "temperature",
        "combined_fisher",
        None,
        "today",
        {
            "newtwork_run": {"verdict": "local_to_this_site", "order": []},
            "trial_bed": {"verdict": "possible_new_type", "cause": None},
        },
    ),
    (
        "Flame Skimmer",
        "flame_skimmer",
        "scnf010",
        "dissolved_oxygen",
        "combined_fisher",
        None,
        "today",
        {
            "trial_bed": {"verdict": "cannot_evaluate", "cause": None},
            "settling_pool": {
                "verdict": "held_back",
                "explanation": "held back 3 readings at scnf010 it didn't trust, "
                "before the model read them",
            },
        },
    ),
    ("Spike", "water_strider", "south_fork_2", "depth", "hampel_spike", None, "today", {}),
    (
        "Shift, up",
        "water_strider",
        "south_fork_3",
        "conductivity",
        "cusum_shift",
        "up",
        "today",
        {},
    ),
    (
        "Shift, down",
        "water_strider",
        "south_fork_0",
        "temperature",
        "cusum_shift",
        "down",
        "yesterday",
        {},
    ),
    (
        "Shift, direction unclear",
        "water_strider",
        "botanical_garden",
        "conductivity",
        "cusum_shift",
        None,
        "this year",
        {},
    ),
    ("Drift", "water_strider", "codornices", "conductivity", "ewma_drift", None, "today", {}),
    (
        "Any other rule",
        "dusk_crayfish",
        "south_fork_1",
        "temperature",
        "a_rule_with_no_sentence",
        None,
        "today",
        {},
    ),
)

# Which samples get the weather note, since it can turn up on any of them
NO_WEATHER = ("Spike",)


def moments(ref):
    """A time for each of the four ways a date is written, before ref."""
    today = view.midnight(view._local(ref).date())
    return {
        "today": max(today + 60, ref - 2 * H),
        "yesterday": today - 10 * H + 15 * 60,
        "this year": today - 6 * 86400 + 10 * H + 30 * 60,
        "last year": ref - 400 * 86400,
    }


def items(ref):
    """Every sample as {name, proposed, view}, view being what the pages get for a real alert."""
    when = moments(ref)
    out = []
    for n, (name, model, site, variable, rule, direction, at, said) in enumerate(SAMPLES, 1):
        first = when[at] - when[at] % STEP
        item = {
            "id": n,
            "model": model,
            "site": site,
            "variable": variable,
            "rule": rule,
            "first_over": first,
            "last_over": first + 3 * H,
            "explanations": said,
            "check_weather_ok": 0 if name in NO_WEATHER else 1,
            "check_source": "site",
            "checked_at": first + 4 * H,
        }
        out.append(
            {
                "name": name,
                "proposed": rule in view.PROPOSED_RULES,
                "view": view.item_view(None, item, ref, direction),
            }
        )
    return out


def reading_chart(item):
    """A day of a reading with a lasting jump, the jump picked out like a real alert's."""
    start, end = item["first_over"] - 24 * H, item["last_over"] + 6 * H
    xs = list(range(int(start), int(end) + 1, STEP))
    ys, unusual = [], []
    for t in xs:
        value = 310 + 12 * math.sin(2 * math.pi * (t % 86400) / 86400)
        if t >= item["first_over"]:
            value += 140
        ys.append(round(value, 1))
        unusual.append(ys[-1] if item["first_over"] <= t <= item["last_over"] else None)
    label = view.LABELS.get(item["variable"], item["variable"]).capitalize()
    return {
        "x": xs,
        "y": ys,
        "unusual": unusual,
        "label": f"{label}, {view.UNITS[item['variable']]}",
    }


def score_chart(item):
    """A score that crosses its line during a little rain, 36 hours before to 12 after."""
    start, end = item["first_over"] - 36 * H, item["last_over"] + 12 * H
    xs = list(range(int(start), int(end) + 1, STEP))
    score, line, dry, rain = [], [], [], []
    for t in xs:
        hours = (t - item["first_over"]) / H
        wet = -20 <= hours <= -14
        rain.append(0.6 if wet else 0.0)
        dry.append(1.0)
        line.append(1.6 if -20 <= hours <= -8 else 1.0)
        bump = 2.2 * math.exp(-((hours - 1.5) ** 2) / 3) if -1 <= hours <= 6 else 0.0
        score.append(round(0.35 + 0.15 * math.sin(t / 7000) + bump, 3))
    return {
        "x": xs,
        "score": score,
        "line": line,
        "dry": dry,
        "rain": rain,
        "flag": item["first_over"],
        "label": view.LABELS.get(item["variable"], item["variable"]),
        "about": "",
    }


def overview_boxes(ref):
    """Every state the Overview box can be in, with made-up sites."""
    return (
        ("Waiting for the first check", {"waiting": True}),
        ("One site with alerts", {"flagged": ["south_fork_1"]}),
        ("Several sites with alerts", {"flagged": ["south_fork_1", "oxford"]}),
        ("Alerts, and a site with no recent data", {"flagged": ["oxford"], "late": ["codornices"]}),
        ("No recent data, no alerts", {"late": ["kingman_hall", "codornices"]}),
        ("Nothing current", {}),
        (
            "Checks have stopped",
            {
                "stale": True,
                "stale_text": f"No check has finished since {view.clock(ref - 2 * H)}.",
            },
        ),
    )


def _said(site, explanations, weather_ok=1):
    return view.extras_said(
        {"site": site, "explanations": explanations, "check_weather_ok": weather_ok}
    )[0]


def sentences(ref, samples):
    """Every sentence as (part, variant, text, proposed), in the order an alert reads."""
    rows = [
        ("When", name.capitalize(), view.when_words(t, ref), False)
        for name, t in moments(ref).items()
    ]
    first = samples[0]["view"]
    rows.append(
        (
            "Heading",
            "Every alert",
            f"{first['label'].capitalize()} alert at {first['site_name']}",
            False,
        )
    )
    rows += [("What happened", s["name"], s["view"]["happened"], s["proposed"]) for s in samples]
    rows.append(("Who", "Every alert", f"Flagged by {first['title']}.", False))
    run = "newtwork_run"
    rows += [
        (
            "Newtwork Run",
            "Started here",
            _said(
                "south_fork_1",
                {
                    run: {
                        "verdict": "consistent_with_transport",
                        "order": ["south_fork_1", "south_fork_2", "south_fork_3"],
                    }
                },
            ),
            False,
        ),
        (
            "Newtwork Run",
            "Reached here after other sites",
            _said(
                "oxford",
                {
                    run: {
                        "verdict": "consistent_with_transport",
                        "order": ["south_fork_3", "oxford"],
                    }
                },
            ),
            False,
        ),
        (
            "Newtwork Run",
            "Creek-wide",
            _said("oxford", {run: {"verdict": "catchment_wide"}}),
            False,
        ),
        (
            "Newtwork Run",
            "Another creek too",
            _said("oxford", {run: {"verdict": "not_transport"}}),
            False,
        ),
        (
            "Newtwork Run",
            "Only this site",
            _said("oxford", {run: {"verdict": "local_to_this_site"}}),
            False,
        ),
        (
            "Trial Bed",
            "A match",
            _said("oxford", {"trial_bed": {"verdict": "diagnosed", "cause": "sewage"}}),
            False,
        ),
        (
            "Trial Bed",
            "No match",
            _said("oxford", {"trial_bed": {"verdict": "possible_new_type"}}),
            False,
        ),
        (
            "Trial Bed",
            "Too few sensors",
            _said("oxford", {"trial_bed": {"verdict": "cannot_evaluate"}}),
            False,
        ),
        (
            "Settling Pool",
            "Held readings back",
            _said(
                "oxford",
                {
                    "settling_pool": {
                        "verdict": "held_back",
                        "explanation": "held back 3 readings at oxford it didn't trust, before the model read them",
                    }
                },
            ),
            False,
        ),
        ("Weather", "No weather data", _said("oxford", {}, weather_ok=0), False),
    ]
    return rows

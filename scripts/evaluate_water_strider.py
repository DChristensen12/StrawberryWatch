"""
Scores Water Strider against the same labelled events and clean data as the
other methods, counting whole events rather than single readings.

    python scripts/evaluate_water_strider.py

Methods: Water Strider in full, each of its three tests alone, naive X84 (a
fixed 5.2 raw MADs, no daily profile, no rain), Dusk Crayfish and Cobble Shoal.

Reports which events got flagged and when. The event folders have no start
time, and timing is relative to the earliest flag from any method. False flags
are counted per week on the held-back clean months. Then X84's k is tuned to
match Water Strider's false flag rate, and the two are compared at that rate.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate_water_strider as cw  # noqa: E402

from strawberrywatch import paths  # noqa: E402
from strawberrywatch.models import Water_Strider as wsm  # noqa: E402
from strawberrywatch.watch import sources  # noqa: E402

GRAPH_MODELS = ("dusk_crayfish", "cobble_shoal")
METHODS = ("water_strider", *wsm.RULES, "naive_x84", *GRAPH_MODELS)
X84_K = 5.2


def event_flags(event, detectors, x84_k):
    """{method: first flag time at each site's conductivity} for one replayed event."""
    replay = sources.ReplaySource(event["folder"])
    start, end = replay.window(None)
    weather = replay.weather(start, end)
    tables = replay.tables(list(replay._csv._all()), start, end)
    out = {}
    found = detectors["water_strider"].findings(tables, weather)
    out["water_strider"] = _first(f for f in found if f.variable == "conductivity")
    for rule in wsm.RULES:
        out[rule] = _first(f for f in found if f.variable == "conductivity" and f.rule == rule)
    out["naive_x84"] = {}
    for site, table in tables.items():
        frame = sources.tidy(table)
        if "conductivity" not in frame.columns:
            continue
        values = wsm.on_grid(frame["conductivity"])
        over = wsm.x84(values, x84_k) > 1.0
        if over.any():
            out["naive_x84"][site] = values.index[np.flatnonzero(over)[0]]
    for name in GRAPH_MODELS:
        detector = detectors.get(name)
        if detector is None:
            out[name] = None
            continue
        mine = {n: tables[n] for n in detector.table_names if n in tables}
        out[name] = _first(
            f for f in detector.findings(mine, weather) if f.variable == "conductivity"
        )
    return out


def _first(found):
    first = {}
    for f in found:
        if f.site not in first or f.when < first[f.site]:
            first[f.site] = f.when
    return first


def held_out_flags(calibration, k_values):
    """
    False flags per week on the held-back months' clean steps: Water Strider
    and each test at the calibrated thresholds, and naive X84 at each k.
    """
    start, end = (pd.Timestamp(d, tz="UTC") for d in calibration["span"])
    params, gap = calibration["params"], int(calibration["params"]["decluster_steps"])
    warm = start - pd.Timedelta(days=params["profile_days"] + 2)
    inventory, tables, rain = cw.load(warm, end)
    alarms = {name: 0 for name in ("water_strider", *wsm.RULES)}
    x84 = dict.fromkeys(k_values, 0)
    steps = 0
    for entry in calibration["sensors"].values():
        values = tables[entry["site"]][entry["variable"]]
        values = wsm.on_grid(values[(values.index >= warm) & (values.index < end)])
        _, held = cw.split(
            values.index, cw.clean_mask(values.index, rain), calibration["split"], start, start, end
        )
        prepared = cw.prepare(values, entry["variable"], held | ~held, params)
        union = np.zeros(len(values), dtype=bool)
        for rule in wsm.RULES:
            statistic = prepared["stats"][rule]
            over = held & np.isfinite(statistic) & (statistic > entry["tests"][rule]["threshold"])
            if rule == "hampel_spike":
                over &= np.concatenate([[False], over[:-1]])
            alarms[rule] += cw.alarms(over, gap)
            union |= over
        alarms["water_strider"] += cw.alarms(union, gap)
        steps += int((held & np.isfinite(prepared["stats"]["ewma_drift"])).sum())
        if entry["variable"] == "conductivity":
            ratio_at_one = wsm.x84(values, 1.0)
            for k in k_values:
                over = held & np.isfinite(ratio_at_one) & (ratio_at_one > k)
                x84[k] += cw.alarms(over, gap)
    weeks = steps / wsm.STEPS_PER_DAY / 7
    return (
        {name: count / weeks for name, count in alarms.items()},
        {k: c / weeks for k, c in x84.items()},
        weeks,
    )


def main(argv=None):
    from strawberrywatch.serving import detector_class

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-graph-models", action="store_true")
    parser.add_argument("--calibration", default=None, help="a folder holding another calibration")
    args = parser.parse_args(argv)
    warnings.simplefilter("ignore")

    mine = Path(args.calibration) if args.calibration else paths.model_dir("water_strider")
    calibration = json.loads((mine / "water_strider_calibration.json").read_text())
    detectors = {"water_strider": detector_class("water_strider").load(mine)}
    if not args.no_graph_models:
        for name in GRAPH_MODELS:
            detectors[name] = detector_class(name).load(paths.model_dir(name))

    print("False flags per week on the held-back months' clean data, network wide")
    # Wide enough that X84 can come down to Water Strider's rate, however strict it is
    k_values = [round(float(k), 1) for k in np.geomspace(5.2, 2000, 120)]
    rates, x84_rates, weeks = held_out_flags(calibration, k_values)
    budget = len(calibration["sensors"]) * wsm.STEPS_PER_DAY * 7 * calibration["q_total"]
    print(
        f"  {weeks:.1f} sensor-weeks of clean data; the budget allows about {budget / len(calibration['sensors']):.2f} a sensor-week"
    )
    for name, rate in rates.items():
        print(f"  {name:<14} {rate:7.3f} a sensor-week")
    print(f"  {'naive_x84':<14} {x84_rates[X84_K]:7.3f} a sensor-week at k = {X84_K}")
    matched_k = min(k_values, key=lambda k: (abs(x84_rates[k] - rates["water_strider"]), k))
    print(
        f"  naive X84 matches Water Strider's rate at k = {matched_k} ({x84_rates[matched_k]:.3f})"
    )

    print("\nLabelled events, conductivity at the labelled site")
    rows = []
    for event in sources.replay_events():
        flags = event_flags(event, detectors, X84_K)
        matched = event_flags(event, {"water_strider": detectors["water_strider"]}, matched_k)[
            "naive_x84"
        ]
        labels = event.get("labels") or {}
        sites = list(labels) or ["(rain only)"]
        for site in sites:
            label = labels.get(
                site, "rain only" if "rain" in event["title"].lower() else "unlabelled"
            )
            row = {"event": event["title"], "site": site, "label": label}
            for name in METHODS:
                got = flags.get(name)
                row[name] = None if got is None else (got.get(site) if site in labels else len(got))
            row["naive_x84_matched"] = matched.get(site) if site in labels else len(matched)
            rows.append(row)
    for row in rows:
        print(f"\n  {row['event']} [{row['site']}, {row['label']}]")
        times = [row[n] for n in METHODS if isinstance(row[n], pd.Timestamp)]
        earliest = min(times) if times else None
        for name in (*METHODS, "naive_x84_matched"):
            value = row[name]
            if row["label"] == "rain only":
                text = "not run" if value is None else f"{value} site(s) flagged"
            elif value is None and name in GRAPH_MODELS and args.no_graph_models:
                text = "not run"
            elif isinstance(value, pd.Timestamp):
                lag = (value - earliest).total_seconds() / 3600
                local = value.tz_convert("America/Los_Angeles").strftime("%d %b %H:%M")
                text = f"flagged {local} Pacific, {lag:.1f} h after the earliest"
            else:
                text = "missed"
            print(f"    {name:<18} {text}")

    graded = [r for r in rows if r["label"] == "anomaly"]
    rain = [r for r in rows if r["label"] == "rain only"]
    caught = sum(isinstance(r["water_strider"], pd.Timestamp) for r in graded)
    caught_x84 = sum(isinstance(r["naive_x84_matched"], pd.Timestamp) for r in graded)
    print("\nPass criteria")
    print(
        f"  [{'x' if calibration['passed'] else ' '}] held-out false flags within twice the budget"
    )
    spill = [r for r in graded if "spill" in r["event"].lower()]
    print(
        f"  [{'x' if spill and all(isinstance(r['water_strider'], pd.Timestamp) for r in spill) else ' '}] catches the labelled spill events ({caught} of {len(graded)} labelled anomalies)"
    )
    print(
        f"  [{'x' if rain and all(r['water_strider'] == 0 for r in rain) else ' '}] doesn't flag the labelled rain-only events"
    )
    print(
        f"  [{'x' if caught > caught_x84 else ' '}] beats naive X84 at the same false-flag rate ({caught} against {caught_x84})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

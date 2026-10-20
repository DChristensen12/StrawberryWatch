"""
Builds water_strider_calibration.json from the archive in data/raw_data.

    python scripts/calibrate_water_strider.py
    python scripts/calibrate_water_strider.py --out /some/version/folder

Statistics run over the full span. CUSUM and EWMA carry their sums like a
live check. Only clean steps get fitted: none inside a labelled event, and
none while RainGate is still raising the bar. Even months are fitted and odd
months held back. Both halves get wet and dry weather.

Each test gets a peaks-over-threshold tail, solved for its third of the false
alarm budget. Sensors with too few clusters share a pooled tail with others of
their kind. It passes if the held-back months stay within twice the budget.

Writes to checkpoints/ by default. If you replace the shipped file, update its
line in strawberrywatch/watch/model_files.sha256.json too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_cobble_shoal_real import drop_implausible  # noqa: E402

from strawberrywatch.models import Water_Strider as wsm  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402

SPAN_START = "2025-06-04"
SPAN_END = "2026-04-29"
# Only used by --split chronological
FIT_END = "2026-02-01"
# A sensor needs this many days of clean readings in the fit months to be judged at all
MIN_CLEAN_DAYS = 14


def load(start, end):
    """The archive's tables under inventory names, out-of-range readings dropped, and the rain."""
    from strawberrywatch import inventory as inv
    from strawberrywatch.ingest import weather_archive
    from strawberrywatch.ingest.raw_data_loader import load_archive_by_table

    inventory = inv.load()
    tables = load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    tables, _ = drop_implausible(tables)
    rain = weather_archive.load_weather(start, end)["rain_mm"]
    return inventory, tables, pd.to_numeric(rain, errors="coerce").resample("15min").sum()


def clean_mask(index, rain):
    """True where a step is outside every labelled event and RainGate wouldn't be raising the bar."""
    from strawberrywatch.anomalies.rain_gate import RainGate

    clean = np.ones(len(index), dtype=bool)
    for lo, hi in nw.EVENT_SPANS:
        clean &= ~((index >= pd.Timestamp(lo, tz="UTC")) & (index < pd.Timestamp(hi, tz="UTC")))
    wet = rain.reindex(index).fillna(0.0).to_numpy(dtype=float)
    gate = RainGate(base_threshold=1.0)
    clean &= gate.multipliers(wet, index.tz_localize(None).to_numpy()) <= 1.0 + 1e-9
    return clean


def split(index, clean, how, start, fit_end, end):
    """
    Returns (fit, held) masks over the clean steps. blocked alternates whole
    months, and both sides get wet and dry weather. chronological holds back the
    late months, which on this archive puts all the rain on the held-out side.
    """
    inside = (index >= start) & (index < end)
    if how == "chronological":
        return clean & inside & (index < fit_end), clean & inside & (index >= fit_end)
    month = (index.year * 12 + index.month).to_numpy()
    return clean & inside & (month % 2 == 0), clean & inside & (month % 2 == 1)


def prepare(values, variable, fit, params):
    """One sensor's residuals, spread, floor, autocorrelation, statistics and live profile."""
    raw = values.to_numpy(dtype=float)
    above = raw - wsm.level(raw, params)
    per_step, latest = wsm.rolling_profiles(values.index, above, params)
    resid = above - per_step
    fitted = fit & np.isfinite(resid)
    if fitted.sum() < MIN_CLEAN_DAYS * wsm.STEPS_PER_DAY:
        raise ValueError(
            f"only {fitted.sum() / wsm.STEPS_PER_DAY:.1f} clean days in the fit months"
        )
    middle = np.median(resid[fitted])
    spread = float(np.median(np.abs(resid[fitted] - middle)) * wsm.MAD_SCALE)
    floor = wsm.floor_for(variable, spread, params)
    rho = wsm.autocorrelation(resid, fitted)
    return {
        "spread": spread,
        "floor": floor,
        "rho": rho,
        "profile": latest,
        "stats": wsm.statistics_of(resid, spread, floor, params, rho),
        "clean_days": round(float(fitted.sum()) / wsm.STEPS_PER_DAY, 1),
    }


def alarms(over, gap):
    """How many runs over a line, closer than gap steps counted as one."""
    return wsm.clusters(over.astype(float), over, 0.5, gap).size


def main(argv=None):
    from strawberrywatch import paths

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=SPAN_START)
    parser.add_argument("--end", default=SPAN_END)
    parser.add_argument("--split", choices=("blocked", "chronological"), default="blocked")
    parser.add_argument("--fit-end", default=FIT_END, help="only with --split chronological")
    parser.add_argument("--q-total", type=float, default=wsm.Q_TOTAL)
    parser.add_argument("--out", default=None, help="folder to write into, checkpoints/ by default")
    parser.add_argument(
        "--pool",
        choices=("thin", "all"),
        default="thin",
        help="share a tail across sensors of a kind for only those too thin to fit alone, or for all",
    )
    parser.add_argument("--floor-share", type=float, default=wsm.PARAMS["mad_floor_share"])
    parser.add_argument("--hampel-window", type=int, default=wsm.PARAMS["hampel_window"])
    parser.add_argument("--profile-days", type=int, default=wsm.PARAMS["profile_days"])
    parser.add_argument(
        "--long-run-cusum", action="store_true", help="scale CUSUM's input for slow wander"
    )
    args = parser.parse_args(argv)

    began = time.time()
    start, fit_end, end = (pd.Timestamp(d, tz="UTC") for d in (args.start, args.fit_end, args.end))
    params = {
        **wsm.PARAMS,
        "mad_floor_share": args.floor_share,
        "hampel_window": args.hampel_window,
        "profile_days": args.profile_days,
        "cusum_long_run": args.long_run_cusum,
    }
    gap = int(params["decluster_steps"])
    q_test = wsm.per_test(args.q_total)
    # The level needs half a day and the profile two weeks behind the first fitted step
    warm = start - pd.Timedelta(days=params["profile_days"] + 2)
    inventory, tables, rain = load(warm, end)

    # First all sensors' statistics. A sensor too thin to fit alone can then share a tail
    prepared, skipped, digest = {}, {}, hashlib.sha256()
    for site in sorted(inventory.sites):
        frame = tables.get(site)
        if frame is None or frame.empty:
            continue
        for variable in inventory.site(site).variables:
            if variable not in wsm.RESOLUTION or variable not in frame.columns:
                continue
            key = f"{site}.{variable}"
            values = frame[variable]
            values = values[(values.index >= warm) & (values.index < end)]
            sensor = inventory.site(site).sensor_for(variable)
            # Readings from before this probe went in aren't its own
            if sensor is not None and sensor.install is not None:
                installed = pd.Timestamp(sensor.install)
                installed = installed.tz_localize("UTC") if installed.tz is None else installed
                values = values[values.index >= installed]
            values = wsm.on_grid(values)
            if values.dropna().size < wsm.STEPS_PER_DAY * MIN_CLEAN_DAYS:
                skipped[key] = "too few readings"
                continue
            fit, held = split(
                values.index, clean_mask(values.index, rain), args.split, start, fit_end, end
            )
            try:
                entry = prepare(values, variable, fit, params)
            except ValueError as exc:
                skipped[key] = str(exc)
                continue
            prepared[key] = {"site": site, "variable": variable, "fit": fit, "held": held, **entry}
            digest.update(key.encode())
            digest.update(np.nan_to_num(values[fit].to_numpy(dtype=float), nan=-1.0).tobytes())

    # Then the tails: a sensor's own where it has enough clusters, its kind's otherwise
    tails = {}
    if args.pool == "thin":
        for key, entry in prepared.items():
            for rule, statistic in entry["stats"].items():
                try:
                    tails[(key, rule)] = {
                        **wsm.fit_tail(statistic, entry["fit"], gap),
                        "pooled": False,
                    }
                except ValueError:
                    pass
    for variable in sorted({e["variable"] for e in prepared.values()}):
        for rule in wsm.RULES:
            thin = [
                k
                for k, e in prepared.items()
                if e["variable"] == variable and (k, rule) not in tails
            ]
            if not thin:
                continue
            kind = [k for k, e in prepared.items() if e["variable"] == variable]
            statistic, mask = wsm.pooled(
                [prepared[k]["stats"][rule] for k in kind], [prepared[k]["fit"] for k in kind], gap
            )
            try:
                shared = {**wsm.fit_tail(statistic, mask, gap), "pooled": True}
            except ValueError as exc:
                for k in thin:
                    skipped[k] = f"{rule}: {exc}"
                continue
            for k in thin:
                tails[(k, rule)] = dict(shared)

    # Last, the held-out alarms at those thresholds
    sensors = {}
    counts = {rule: [0, 0] for rule in (*wsm.RULES, "any_test")}
    for key, entry in prepared.items():
        if any((key, rule) not in tails for rule in wsm.RULES):
            continue
        tests, union = {}, np.zeros(len(entry["held"]), dtype=bool)
        held = entry["held"]
        for rule in wsm.RULES:
            tail = tails[(key, rule)]
            tail["threshold"] = wsm.threshold_for(tail, q_test)
            tests[rule] = tail
            statistic = entry["stats"][rule]
            over = held & np.isfinite(statistic) & (statistic > tail["threshold"])
            if rule == "hampel_spike":
                over &= np.concatenate([[False], over[:-1]])
            counts[rule][0] += alarms(over, gap)
            counts[rule][1] += int((held & np.isfinite(statistic)).sum())
            union |= over
        judged = held & np.isfinite(entry["stats"]["ewma_drift"])
        counts["any_test"][0] += alarms(union & judged, gap)
        counts["any_test"][1] += int(judged.sum())
        sensors[key] = {
            "site": entry["site"],
            "variable": entry["variable"],
            "spread": entry["spread"],
            "floor": entry["floor"],
            "rho": entry["rho"],
            "profile": [round(float(v), 6) for v in entry["profile"]],
            "tests": tests,
            "clean_days": entry["clean_days"],
        }

    held_out = {}
    for rule, (over, judged) in counts.items():
        allowed = 2 * (args.q_total if rule == "any_test" else q_test)
        rate = over / judged if judged else None
        held_out[rule] = {
            "alarms": over,
            "steps": judged,
            "rate": rate,
            "allowed": allowed,
            "passed": rate is not None and rate <= allowed,
        }
    calibration = {
        "model": "water_strider",
        "q_total": args.q_total,
        "q_per_test": q_test,
        "params": params,
        "span": [args.start, args.end],
        "split": args.split,
        "pool": args.pool,
        "fit_months": "even months" if args.split == "blocked" else [args.start, args.fit_end],
        "held_out_months": "odd months" if args.split == "blocked" else [args.fit_end, args.end],
        "alarms_are": "runs over the line, closer than decluster_steps counted as one",
        "data_sha256": digest.hexdigest(),
        "made": time.strftime("%Y-%m-%d", time.gmtime()),
        "rain": "steps RainGate would raise the bar for were left out of the fit",
        "events_left_out": [list(span) for span in nw.EVENT_SPANS],
        "held_out": held_out,
        "passed": all(entry["passed"] for entry in held_out.values()),
        "skipped": skipped,
        "sensors": sensors,
    }
    out = Path(args.out) if args.out else paths.checkpoints_dir()
    path = out / "water_strider_calibration.json"
    path.write_text(json.dumps(calibration, indent=1, sort_keys=True))

    pooled = sum(1 for s in sensors.values() for t in s["tests"].values() if t["pooled"])
    print(
        f"{len(sensors)} sensors calibrated ({pooled} of {3 * len(sensors)} tails shared with "
        f"their kind), {len(skipped)} skipped, in {time.time() - began:.0f} s"
    )
    for key, why in skipped.items():
        print(f"  skipped {key}: {why}")
    for rule, entry in held_out.items():
        shown = (
            "none judged"
            if entry["rate"] is None
            else f"{entry['rate']:.2e} ({entry['alarms']} alarms)"
        )
        verdict = "passes" if entry["passed"] else "FAILS"
        print(f"  held out {rule:<13} {shown} against {entry['allowed']:.2e}  {verdict}")
    print(f"wrote {path}")
    return 0 if calibration["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

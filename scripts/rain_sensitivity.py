"""
How much a weather-context Cobble Shoal depends on its rain channels being right.

The question this answers is not "is Open-Meteo accurate", it is "what happens
when the rain source is wrong, and in which direction does that hurt". On
2026-09-03 a storm fell on the catchment that most weather products did not
report at all, so the zeroed variant below is not a thought experiment, it is
what those products would have handed the model that day.

Three context variants, same weights, same windows, same threshold. Only the
three rain channels change:

    openmeteo   what the model trained on, the baseline
    zeroed      rain_mm, rain_6h, rain_72h flat zero, the missed-storm case
    doubled     rain scaled x2, a source that over-reports

    python scripts/rain_sensitivity.py --model weather
    python scripts/rain_sensitivity.py --model weather_blocked

Only meaningful for a checkpoint that was trained with weather context. A clock
model has no rain channels and every column will read the same.

Reads the labelled fixtures under data/anomalies/ directly rather than the test
catalog, because the rain events are the whole point here and they are no longer
listed in tests/events.yaml (see future_work.md). Writes nothing.

Caveat on the thresholds. Counts here are each model's own z_q, not a matched
false-alarm rate, so the columns compare against themselves and not against each
other. The chronological weather model's held-out exceedance rate misses its
nominal 1e-3 by about 22x, which means its z_q does not mean what it claims;
compare within a column, not across models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_audit_comparison import event_tables  # noqa: E402
from train_cobble_shoal_real import (  # noqa: E402
    attach_context,
    hold_offline,
    load_temp_model,
    weather_frame,
)

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.paths import anomalies_dir  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402

WINDOW = 24
RAIN_COLS = ["rain_mm", "rain_6h", "rain_72h"]

# Two wet and three dry, so a column that only moves on the wet rows is visible
# as such. The two rain fixtures are still on disk after being delisted.
FOLDERS = [
    ("apr26_rainfall", "anomaly_2026_04_01_rainfall"),
    ("nov25_rain", "anomaly_2025_11_13_rain_nf1"),
    ("jun25_spill", "anomaly_2025_06_12_spill_sf"),
    ("mar26_hydrant", "anomaly_2026_03_20_hydrant_nf0"),
    ("sep25_overnight", "anomaly_2025_09_10_overnight_sf"),
]

VARIANTS = {
    "openmeteo": lambda wf: wf,
    "zeroed": lambda wf: wf.assign(**{c: 0.0 for c in RAIN_COLS}),
    "doubled": lambda wf: wf.assign(**{c: wf[c] * 2.0 for c in RAIN_COLS}),
}


def event_span(folder):
    """The grid an event's CSVs cover, or None if nothing is readable."""
    tables = event_tables(folder)
    if not tables:
        return None
    spans = [df.index for df in tables.values() if len(df)]
    if not spans:
        return None
    return (
        tables,
        min(s.min() for s in spans).floor("15min"),
        max(s.max() for s in spans).ceil("15min"),
    )


def make_prepare(temp, mutate):
    """temp.prepare, with the weather frame passed through mutate on the way in."""

    def prepare(win):
        if temp.ctx_scaler.columns:
            win = attach_context(win, mutate(weather_frame(win["grid"])), temp.ctx_scaler)
        idx = [k for k, node in enumerate(win["nodes"]) if node.key in temp.offline]
        return hold_offline(win, idx) if idx else win

    return prepare


def score(span, temp, inventory, mutate):
    """Peak score and how many anchors fired, over one event under one variant."""
    tables, start, end = span
    win = nw.build_window(
        tables, start, end, inventory=inventory, scaler=temp.node_scaler, as_of=end
    )
    win = make_prepare(temp, mutate)(win)

    peaks, over = [], []
    for anchor in range(WINDOW, len(win["grid"]) - 1):
        s = np.asarray(temp.net.score(nw.to_batch(win, anchor, WINDOW), temp.nulls)).ravel()
        peaks.append(np.nanmax(s) if np.isfinite(s).any() else np.nan)
        over.append(bool((s > temp.z_q).any()))
    if not peaks:
        return None
    return {"peak": float(np.nanmax(peaks)), "over": int(sum(over)), "n": len(peaks)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weather", help="temp checkpoint tag, e.g. weather_blocked")
    args = ap.parse_args(argv)

    inventory = inv.load()
    temp = load_temp_model(args.model)
    print(f"cobble_shoal_temp_{args.model}.pt")
    print(f"  context : {', '.join(temp.features)}")
    print(f"  z_q     : {temp.z_q:.4f}   operating q: {temp.operating_q:g}")
    if not any(c in temp.features for c in RAIN_COLS):
        print("\n  no rain channels in this checkpoint's context. Every column will match.")
    print()

    head = f"{'event':18s}{'anchors':>8s}{'rain mm':>9s}  " + "".join(f"{v:>20s}" for v in VARIANTS)
    print(head)
    print("-" * len(head))

    for label, folder in FOLDERS:
        if not (anomalies_dir() / folder).exists():
            print(f"{label:18s}fixture not on disk, skipped")
            continue
        span = event_span(folder)
        if span is None:
            print(f"{label:18s}no readable site CSVs, skipped")
            continue

        rain_mm = weather_frame(pd.date_range(span[1], span[2], freq=nw.GRID_FREQ, tz="UTC"))[
            "rain_mm"
        ].sum()

        cells, anchors = "", None
        for mutate in VARIANTS.values():
            r = score(span, temp, inventory, mutate)
            if r is None:
                cells += f"{'-':>20s}"
                continue
            anchors = r["n"]
            cells += f"{r['peak']:>12.1f}{r['over']:>5d}/{r['n']:<3d}"
        print(f"{label:18s}{str(anchors):>8s}{rain_mm:>9.1f}  {cells}")

    print("\npeak is the highest node score over the window, over is how many anchors")
    print("had any node above z_q. rain mm is the Open-Meteo total for the window.")


if __name__ == "__main__":
    main()

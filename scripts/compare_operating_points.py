"""
Every Cobble Shoal artifact at a matched false-alarm rate.

run_audit_comparison scores each model at its own calibrated z_q, which is right
when you trust the calibration. The retrained weather variant does not survive
its own held-out check (2.24e-2 measured against a nominal 1e-3), so its extra
detections might be discrimination or might just be a lower bar. Comparing at a
nominal q cannot tell those apart.

So set every model's threshold from the same measured exceedance rate on one
common fault-free span, then see what each one catches at the same cost.

    python scripts/compare_operating_points.py

The span is March 2026. It sits in validation for the chronological split and in
validation for the blocked one, and the shipped weights never saw real data at
all, so nothing here is scored on its own training data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from run_audit_comparison import WINDOW, event_tables, load_cobble, real_calibration  # noqa: E402
from train_cobble_shoal_real import drop_implausible, load_temp_model  # noqa: E402

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.ingest.raw_data_loader import load_archive_by_table  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402
from tests import event_catalog  # noqa: E402

# Held out by both splits. March is validation under chronological (cut
# 2026-02-01) and validation under blocked (odd months), which is what makes one
# threshold comparable across all five artifacts.
COMMON_START = "2026-03-01"
COMMON_END = "2026-04-01"

# Two operating points rather than one. 1e-2 has enough exceedances in a month
# of windows to be a measurement; 1e-3 is nearer the alerting budget and is
# reported knowing it rests on a couple of dozen points.
RATES = (1e-2, 1e-3)

MODELS = ("shipped", "clock", "weather", "clock_blocked", "weather_blocked")


class Bundle:
    """A model, its null, and how to put a window in its space."""

    def __init__(self, name, net, nulls, scaler, prepare, nominal_z):
        self.name = name
        self.net = net
        self.nulls = nulls
        self.scaler = scaler
        self.prepare = prepare
        self.nominal_z = nominal_z

    def window(self, tables, start, end, inventory, as_of):
        win = nw.build_window(
            tables, start, end, inventory=inventory, scaler=self.scaler, as_of=as_of
        )
        return self.prepare(win) if self.prepare else win

    def scores(self, win, anchors):
        """(n_anchors, n_nodes) combined Fisher, one row per anchor."""
        out = []
        for anchor in anchors:
            batch = nw.to_batch(win, anchor, WINDOW)
            out.append(np.asarray(self.net.score(batch, self.nulls)).ravel())
        return np.array(out) if out else np.zeros((0, len(win["nodes"])))


def load(name):
    if name == "shipped":
        cal = real_calibration()
        return Bundle(name, load_cobble(cal.seed), cal.nulls, cal.window_scaler(), None, cal.z_q)
    temp = load_temp_model(name)
    return Bundle(name, temp.net, temp.nulls, temp.node_scaler, temp.prepare, temp.z_q)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--null-windows", type=int, default=1500)
    args = ap.parse_args(argv)

    inventory = inv.load()
    tables, _ = drop_implausible(
        load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    )
    as_of = max(f.index.max() for f in tables.values() if len(f))

    bundles, thresholds, measured = {}, {}, {}
    for name in args.models:
        try:
            bundles[name] = load(name)
        except FileNotFoundError as exc:
            print(f"{name}: {exc}")
            continue
        b = bundles[name]
        t0 = time.time()
        win = b.window(
            tables,
            pd.Timestamp(COMMON_START, tz="UTC"),
            pd.Timestamp(COMMON_END, tz="UTC"),
            inventory,
            as_of,
        )
        picks = nw.fault_free_anchors(win, args.null_windows, WINDOW)
        flat = b.scores(win, picks).ravel()
        flat = flat[np.isfinite(flat)]
        thresholds[name] = {r: float(np.quantile(flat, 1.0 - r)) for r in RATES}
        measured[name] = (len(picks), flat.size)
        bar = "  ".join(f"@{r:g}={thresholds[name][r]:7.2f}" for r in RATES)
        print(
            f"{name:17} nominal z_q={b.nominal_z:7.2f}   matched {bar}   "
            f"({len(picks)} windows, {flat.size} node-scores, {time.time() - t0:.0f}s)"
        )
    print()

    rows = {}
    for folder, label, sites, expect in event_catalog.by_folder():
        tabs = event_tables(folder)
        if not tabs:
            continue
        spans = [df.index for df in tabs.values() if len(df)]
        start = min(s.min() for s in spans).floor(nw.GRID_FREQ)
        end = max(s.max() for s in spans).ceil(nw.GRID_FREQ)
        rows[label] = {"expect": expect, "sites": sites}
        for name, b in bundles.items():
            win = b.window(tabs, start, end, inventory, end)
            anchors = range(WINDOW, len(win["grid"]) - 1)
            s = b.scores(win, anchors)
            keys = [n.key for n in win["nodes"]]
            rows[label][name] = {
                "n": s.shape[0],
                "peak": float(np.nanmax(s)) if s.size else float("nan"),
                **{
                    r: {
                        "steps": int(np.sum((s > thresholds[name][r]).any(axis=1))),
                        "top": _top(s > thresholds[name][r], keys),
                    }
                    for r in RATES
                },
            }
        print(f"scored {label}")
    print()

    for r in RATES:
        print(f"### steps with any node over the matched {r:g} threshold")
        head = f"{'event':<18}{'label':<15}" + "".join(f"{n:>18}" for n in bundles)
        print(head)
        print("-" * len(head))
        for label, row in rows.items():
            line = f"{label:<18}{row['expect']:<15}"
            for name in bundles:
                m = row[name]
                line += f"{m[r]['steps']:>8d}/{m['n']:<9d}"
            print(line)
        print()

    print("### top firing nodes at the matched 1e-3 threshold, against the catalog's sites")
    for label, row in rows.items():
        print(f"{label}  ({row['expect']}, catalog: {', '.join(row['sites'])})")
        for name in bundles:
            top = ", ".join(f"{k} x{v}" for k, v in row[name][1e-3]["top"]) or "-"
            print(f"   {name:17} {top}")
        print()
    return 0


def _top(over, keys, n=3):
    counts = over.sum(axis=0)
    return [(keys[i], int(counts[i])) for i in np.argsort(-counts)[:n] if counts[i] > 0]


if __name__ == "__main__":
    raise SystemExit(main())

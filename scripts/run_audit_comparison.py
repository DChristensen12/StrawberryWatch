"""
Cobble Shoal and the two control charts on the labelled audit events.

All three reach this corpus through preprocessing/node_windows and read the
same batches in the same normalization, which is the only way the three
thresholds mean the same thing. Dusk Crayfish is not here: it takes a different
contract and different features, so it is graded by tests/test_anomaly_detection.py
instead.

    python scripts/run_audit_comparison.py                  # shipped weights
    python scripts/run_audit_comparison.py --model weather  # retrained, Open-Meteo context
    python scripts/run_audit_comparison.py --model clock    # retrained, clock only

--model picks which Cobble Shoal artifact to grade. Anything but "shipped" is a
tag written by scripts/train_cobble_shoal_real.py, and that script's loader
supplies the context and the offline nodes those weights were trained with.

The event list is tests/events.yaml, read through tests/event_catalog.py. It
used to be a second copy of the catalog kept here, which could and did drift
from the one the tests grade against.

No training, no checkpoint written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from train_cobble_shoal_real import drop_implausible, load_temp_model  # noqa: E402

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.anomalies import baselines as bl  # noqa: E402
from strawberrywatch.anomalies import cobble_calibration  # noqa: E402
from strawberrywatch.paths import anomalies_dir  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402
from tests import event_catalog  # noqa: E402  (the catalog, not the tests that grade it)

WINDOW = 24

# The event CSVs carry raw logger column names; the adapter speaks the inventory's.
EVENT_COLUMNS = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
    "AtlasSci_DO": "dissolved_oxygen",
    "AtlasSci_FloatCond": "floating_conductivity",
}


def event_tables(folder):
    """{inventory table: DataFrame} out of one event folder."""
    tables = {}
    for site in nw.SITE_ORDER:
        table = nw.SITE_TO_TABLE.get(site, site)
        for stem in (site, table):
            path = anomalies_dir() / folder / f"{stem}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path).rename(columns=EVENT_COLUMNS)
            time_col = next((c for c in ("DateTimeUTC", "datetime", "timestamp") if c in df), None)
            if time_col is None:
                continue
            df["datetime"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
            df = df[~df.index.duplicated(keep="first")]
            keep = [c for c in EVENT_COLUMNS.values() if c in df.columns]
            for c in keep:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            tables[table] = df[keep]
            break
    return drop_implausible(tables)[0]


def load_cobble(seed):
    from strawberrywatch.models.Cobble_Shoal import CobbleShoal

    net = CobbleShoal.from_metadata({"seed": seed, "window": WINDOW})
    blob = torch.load(cobble_calibration.weights_path(), map_location="cpu", weights_only=True)
    net.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    net.eval()
    return net


def real_calibration():
    """
    The real-corpus calibration, or stop.

    No fallback to the synthetic artifact. The two are not interchangeable
    (see cobble_calibration's docstring for the measured score scales), and a
    report that quietly swapped one for the other would still print a number
    for every event.
    """
    return cobble_calibration.load_calibration(cobble_calibration.REAL)


def score_event(folder, detectors, inventory, scaler, prepare=None):
    """Every anchor in one event window, scored by every detector."""
    tables = event_tables(folder)
    if not tables:
        return None
    spans = [df.index for df in tables.values() if len(df)]
    start, end = (
        min(s.min() for s in spans).floor("15min"),
        max(s.max() for s in spans).ceil("15min"),
    )
    win = nw.build_window(tables, start, end, inventory=inventory, scaler=scaler, as_of=end)
    if prepare is not None:
        win = prepare(win)

    anchors = range(WINDOW, len(win["grid"]) - 1)
    batches = [nw.to_batch(win, a, WINDOW) for a in anchors]
    out = {"n_steps": len(batches), "nodes": [n.key for n in win["nodes"]], "grid": win["grid"]}
    for name, (det, nulls, threshold) in detectors.items():
        peaks, over = [], []
        for batch in batches:
            s = np.asarray(det.score(batch, nulls)).ravel()
            peaks.append(np.nanmax(s) if np.isfinite(s).any() else np.nan)
            over.append(s > threshold)
        out[name] = {
            "threshold": threshold,
            "peak": float(np.nanmax(peaks)) if peaks else float("nan"),
            "n_over": int(np.sum([o.any() for o in over])),
            "node_steps_over": int(np.sum(over)),
            "top_nodes": _top_nodes(over, out["nodes"]),
        }
    return out


def _top_nodes(over, nodes):
    counts = np.sum(over, axis=0) if over else np.zeros(len(nodes))
    order = np.argsort(-counts)
    return [(nodes[i], int(counts[i])) for i in order[:3] if counts[i] > 0]


def resolve_model(name):
    """
    Which Cobble Shoal to grade, and how to put a window into its space.

    "shipped" is the checked-in weights against the real-corpus calibration.
    Anything else is a tag from scripts/train_cobble_shoal_real.py, which knows
    its own context features and which nodes it held offline. Those travel with
    the weights because a window missing either is scored by a model that was
    never shown one like it.
    """
    if name == "shipped":
        cal = real_calibration()
        return {
            "label": f"shipped weights, calibration {cal.filename}",
            "net": load_cobble(cal.seed),
            "nulls": cal.nulls,
            "z_q": cal.z_q,
            "q": cal.operating_q,
            "scaler": cal.window_scaler(),
            "prepare": None,
            "corpus": cal.corpus,
        }
    temp = load_temp_model(name)
    return {
        "label": f"cobble_shoal_temp_{name}.pt, context {', '.join(temp.features)}",
        "net": temp.net,
        "nulls": temp.nulls,
        "z_q": temp.z_q,
        "q": temp.operating_q,
        "scaler": temp.node_scaler,
        "prepare": temp.prepare,
        "corpus": temp.calibration["corpus"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="shipped", help='"shipped", or a temp checkpoint tag')
    args = ap.parse_args(argv)

    inventory = inv.load()
    model = resolve_model(args.model)
    scaler = model["scaler"]
    corpus_span = model["corpus"]
    print(
        f"Cobble Shoal: {model['label']}\n"
        f"  z_q={model['z_q']:.4f}, q={model['q']:g}, "
        f"corpus {corpus_span.get('start')} .. {corpus_span.get('end')}\n"
    )

    # charts get the same fault-free real windows the model's refitted null came
    # from, or the three thresholds do not mean the same thing. The scaler is
    # the calibration's own, never one fitted here: nulls only mean something
    # applied to data in the space they were fitted in.
    from strawberrywatch.ingest.raw_data_loader import load_archive_by_table

    tables, dropped = drop_implausible(
        load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    )
    if dropped:
        print(
            f"out-of-range readings dropped: {sum(dropped.values())} across {len(dropped)} series"
        )
    as_of = max(f.index.max() for f in tables.values() if len(f))
    corpus = nw.corpus_window(tables, inventory=inventory, as_of=as_of, scaler=scaler)
    if model["prepare"] is not None:
        corpus = model["prepare"](corpus)
    picks = nw.fault_free_anchors(corpus, 200, WINDOW)
    clean = [nw.to_batch(corpus, a, WINDOW) for a in picks]
    print(f"chart nulls fitted on {len(clean)} fault-free real windows")

    detectors = {"cobble_shoal": (model["net"], model["nulls"], model["z_q"])}
    for name, cls in bl.BASELINES.items():
        det = cls()
        nulls, pot = bl.calibrate(det, clean, q=model["q"])
        detectors[name] = (det, nulls, pot["threshold"])
        print(f"{name}: z_q={pot['threshold']:.4f}")
    print()

    rows = {}
    for folder, label, sites, expect in event_catalog.by_folder():
        res = score_event(folder, detectors, inventory, scaler=scaler, prepare=model["prepare"])
        if res is None:
            print(f"{label}: no readable site CSVs, skipped")
            continue
        rows[label] = res
        print(
            f"=== {label}  ({expect}, catalog sites {', '.join(sites)}, {res['n_steps']} anchors)"
        )
        for name in ("cobble_shoal", "cusum", "ewma"):
            m = res[name]
            top = ", ".join(f"{k}x{v}" for k, v in m["top_nodes"]) or "-"
            print(
                f"   {name:13} peak={m['peak']:8.2f}  base={m['threshold']:7.2f}  "
                f"steps_over={m['n_over']:4d}/{res['n_steps']:<4d}  top: {top}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Refit Cobble Shoal's channel nulls and POT threshold against the real corpus.

The shipped artifact was fitted on synthetic windows because that was the only
corpus the model could read. The adapter changed that, and a null describing
data the detector never sees describes nothing. The weights do not move here.
A calibration is the fault-free distribution of the scores they produce.

    python scripts/refit_cobble_calibration.py

Writes cobble_calibration.REAL beside the synthetic one, which stays put so the
two can be compared. The NodeScaler goes into the artifact with the nulls:
they were fitted in that space and only mean anything applied in it.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from strawberrywatch import inventory as inv
from strawberrywatch.anomalies import channel_scoring as scoring
from strawberrywatch.anomalies import cobble_calibration
from strawberrywatch.ingest.raw_data_loader import load_archive_by_table
from strawberrywatch.paths import checkpoints_dir
from strawberrywatch.preprocessing import node_windows as nw

WINDOW = 24


# fault-free means fault-free. a null fitted over a known spill has the spill
# sitting in its own tail
def load_model(seed):
    """Cobble Shoal sized as its checkpoint was, shipped weights loaded."""
    from strawberrywatch.models.Cobble_Shoal import CobbleShoal

    net = CobbleShoal.from_metadata({"seed": seed, "window": WINDOW})
    blob = torch.load(cobble_calibration.weights_path(), map_location="cpu", weights_only=True)
    net.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    net.eval()
    return net


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=400)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    args = ap.parse_args(argv)

    shipped = cobble_calibration.load_calibration(cobble_calibration.SYNTHETIC)
    q = shipped.operating_q  # not a choice made here
    inventory = inv.load()
    tables = load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    as_of = max(f.index.max() for f in tables.values() if len(f))

    print(f"fitting against the real corpus, q={q:g}, seed={shipped.seed}")
    win = nw.corpus_window(tables, inventory=inventory, as_of=as_of)
    print(f"corpus window: {win['grid'][0]} .. {win['grid'][-1]} ({len(win['grid'])} steps)")

    picks = nw.fault_free_anchors(win, args.max_windows, WINDOW, args.min_coverage)
    print(f"fault-free anchors: {len(picks)}")
    if picks:
        print(f"  spanning {win['grid'][picks[0]]} .. {win['grid'][picks[-1]]}")
        cover = win["target_mask"][picks].mean()
        print(f"  mean node coverage at anchor: {cover:.3f}")

    net = load_model(shipped.seed)
    t0 = time.time()
    per_channel = {name: [] for name in net.channel_names}
    for i, anchor in enumerate(picks):
        batch = nw.to_batch(win, anchor, WINDOW)
        for name, value in net.channels(batch).items():
            per_channel[name].append(np.asarray(value).ravel())
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(picks)}  [{time.time() - t0:.0f}s]")

    nulls = scoring.ChannelNulls().fit(
        {name: np.concatenate(vals) for name, vals in per_channel.items()}
    )

    combined = []
    for anchor in picks:
        combined.append(np.asarray(net.score(nw.to_batch(win, anchor, WINDOW), nulls)).ravel())
    combined = np.concatenate(combined)
    combined = combined[np.isfinite(combined)]

    pot = {}
    for rate in (1e-3, 1e-4, 1e-5):
        pot[f"combined_fisher@{rate:g}"] = scoring.pot_diagnostics(combined, q=rate)
    z_q = float(pot[f"combined_fisher@{q:g}"]["threshold"])

    blob = {
        "model": "cobble_shoal",
        "source_checkpoint": shipped.source_checkpoint,
        "seed": shipped.seed,
        "channel_nulls": nulls.to_dict(),
        "pot": pot,
        "operating_q": q,
        "z_q": z_q,
        "n_null_windows": len(picks),
        "node_scaler": win["scaler"].to_dict(),
        "corpus": {
            "start": str(win["grid"][0]),
            "end": str(win["grid"][-1]),
            "min_coverage": args.min_coverage,
            "excluded_spans": list(nw.EVENT_SPANS),
        },
        "note": (
            "Channel nulls and POT refitted on data/raw_data through "
            "preprocessing/node_windows.build_window. Weights unchanged; this "
            "artifact only describes the fault-free distribution of their scores."
        ),
    }
    path = checkpoints_dir() / cobble_calibration.REAL
    path.write_text(json.dumps(blob, indent=1))

    print(f"\nsynthetic z_q = {shipped.z_q:.4f}")
    print(f"real      z_q = {z_q:.4f}")
    for rate in (1e-3, 1e-4, 1e-5):
        key = f"combined_fisher@{rate:g}"
        old = shipped.pot.get(key, {}).get("threshold", float("nan"))
        new = pot[key]["threshold"]
        print(f"  q={rate:<7g} synthetic={old:8.4f}  real={new:8.4f}  fitted={pot[key]['fitted']}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

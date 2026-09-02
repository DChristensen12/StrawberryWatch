"""
Cobble Shoal and Dusk Crayfish on the same events, in one joinable table.

TEMP. Not wired into anything, nothing here is imported by production.

The two models are not comparable as they stand and this file is the three
pieces of work that make them so.

1. Separate threshold calibration. Cobble Shoal scores a combined Fisher
   statistic against a POT z_q; Dusk Crayfish scores a robust normalised
   conductivity residual against a per node threshold out of its metadata
   pickle. Those live on unrelated scales and neither number transfers. The
   only currency both speak is a measured false alarm rate, so every threshold
   here is refitted as an empirical quantile of that model's own scores on one
   common fault free span. The pickle's node_thresholds are printed for
   reference and never used, because they were fitted on production's grid
   rather than on this adapter's.

2. A model_id column, and long format output. Two CSVs rather than one wide
   table: scores keyed by (model_id, event, timestamp, site) and thresholds
   keyed by (model_id, site, rate). Anything you want later is a join, and
   adding a fourth model does not change either schema.

3. A second graph adapter. Cobble Shoal's nodes are (site, variable) pairs,
   17 of them, and node_windows.build_window already builds that. Dusk
   Crayfish's nodes are whole sites with the variables as feature channels,
   4 of them, and nothing built that from the archive. site_window() does.

    python scripts/temp_cross_model_compare.py

What lines up and what does not

Both models score grid step a from the 24 steps before it, so a Cobble anchor
and a Dusk target index are the same instant and the rows join on timestamp.

They do not read the same array, and forcing them to would be worse. Dusk
Crayfish trained on exact aligned grids with zero fill plus a node mask; Cobble
Shoal trained on carry forward values with a staleness channel. Each is driven
the way it was trained, and the false alarm rate is what is held equal.

Rosters differ too. Dusk Crayfish has no footbridge, so the joinable site set is
the intersection, and footbridge is reported as Cobble Shoal only rather than
quietly dropped.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from compare_operating_points import COMMON_END, COMMON_START  # noqa: E402
from run_audit_comparison import WINDOW, event_tables, resolve_model  # noqa: E402
from train_cobble_shoal_real import drop_implausible  # noqa: E402

from strawberrywatch import inventory as inv  # noqa: E402
from strawberrywatch.config import Config  # noqa: E402
from strawberrywatch.ingest import weather_archive  # noqa: E402
from strawberrywatch.ingest.raw_data_loader import load_archive_by_table  # noqa: E402
from strawberrywatch.models import model_calls  # noqa: E402
from strawberrywatch.paths import checkpoints_dir, reports_dir  # noqa: E402
from strawberrywatch.preprocessing import node_windows as nw  # noqa: E402
from tests import event_catalog  # noqa: E402

RATES = (1e-2, 1e-3)

# Cobble Shoal's variable names on the left, Dusk Crayfish's feature names on
# the right. depth_dev and depth are the same reading under two names; nothing
# is rescaled crossing this line.
DUSK_FEATURE_FROM_VAR = {
    "conductivity": "conductivity",
    "depth_dev": "depth",
    "temperature": "temperature",
}

# What the archive has to be masked down to before Dusk Crayfish sees it. Its
# own three sensor channels, at its own four sites.
DUSK_VARS = list(DUSK_FEATURE_FROM_VAR)

WEATHER_FEATURES = ("rain_mm", "air_temp_c", "shortwave_radiation")


def dusk_metadata():
    import pickle

    path = checkpoints_dir() / "dusk_crayfish_metadata.pkl"
    if not path.exists():
        raise FileNotFoundError(f"no {path.name} in {path.parent}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dusk(metadata):
    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

    weights = checkpoints_dir() / "dusk_crayfish_weights.pt"
    model = model_calls.build_from_metadata(DuskCrayfish, metadata, device=Config.DEVICE)
    model.load_state_dict(torch.load(weights, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model


def _time_features(grid):
    """The four clock channels, matching what data_loader appends in production."""
    # np.asarray first: grid.hour is an Index, and trig on an Index stays an
    # Index, which will not broadcast into the (T, sites, features) array.
    hour = 2 * np.pi * (np.asarray(grid.hour) + np.asarray(grid.minute) / 60.0) / 24.0
    doy = 2 * np.pi * (np.asarray(grid.dayofyear) - 1) / 365.0
    return {
        "hour_sin": np.sin(hour),
        "hour_cos": np.cos(hour),
        "dayofyear_sin": np.sin(doy),
        "dayofyear_cos": np.cos(doy),
    }


def site_window(tables, grid, metadata, inventory=None, as_of=None):
    """
    The second graph adapter: (T, sites, features) the way Dusk Crayfish reads it.

    Sourced through node_windows.site_frames_from_archive so the inventory
    masking, the sentinel handling and the plausibility filter are the same ones
    Cobble Shoal gets. Only the reshape differs, which is the whole point: a
    site is one node here and three nodes there.

    Observations are floored to the 15 minute grid rather than exact matched.
    The event fixtures sit on clean boundaries but the archive does not
    (north_fork_0 reports at :17), and exact matching silently drops those sites
    to nothing.

    Returns (values, node_mask) with values already through the trained scaler,
    missing cells zero filled, and node_mask True where conductivity is real.
    """
    inventory = inventory or inv.load()
    features = list(metadata["feature_cols"])
    sites = list(metadata["location_to_idx"])
    cond = features.index("conductivity")

    roster = {s: list(DUSK_VARS) for s in sites}
    frames = nw.site_frames_from_archive(tables, inventory, as_of=as_of, roster=roster)

    values = np.full((len(grid), len(sites), len(features)), np.nan)
    for site, node in metadata["location_to_idx"].items():
        frame = frames.get(site)
        if frame is None or frame.empty:
            continue
        binned = frame.copy()
        binned.index = binned.index.floor(nw.GRID_FREQ)
        binned = binned[~binned.index.duplicated(keep="last")].reindex(grid)
        for var, feature in DUSK_FEATURE_FROM_VAR.items():
            if var in binned.columns and feature in features:
                values[:, node, features.index(feature)] = binned[var].to_numpy(dtype=float)

    # Weather and clock are global, identical at every site, never missing.
    weather = weather_archive.load_weather(grid[0], grid[-1]).reindex(grid).ffill().bfill()
    shared = {name: weather[name].to_numpy(dtype=float) for name in WEATHER_FEATURES}
    shared.update(_time_features(grid))
    for name, column in shared.items():
        if name in features:
            values[:, :, features.index(name)] = column[:, None]

    node_mask = ~np.isnan(values[:, :, cond])
    values = np.nan_to_num(values, nan=0.0)

    scaler = metadata["scaler"]
    for node in range(len(sites)):
        values[:, node, :] = scaler.transform(values[:, node, :])
    return values.astype(np.float32), node_mask


class DuskDetector:
    """
    Dusk Crayfish scoring one conductivity residual per site per step.

    The robust normalisation (per site error median and IQR) stays, because it
    is part of how this model defines a score. The threshold does not: that gets
    refitted here, see the module docstring.
    """

    model_id = "dusk_crayfish"
    family = "sequence_tensor"

    def __init__(self, metadata, model, edge_index, sites=None):
        self.metadata = metadata
        self.model = model
        self.edge_index = edge_index
        # sites is the output order, which is the shared roster and not this
        # model's own. Its node indices are not 0..3 in that order (the pickle
        # has oxford at 3), so the mapping is kept explicitly rather than
        # assumed from position.
        self.sites = list(sites or metadata["location_to_idx"])
        self.columns = [metadata["location_to_idx"][s] for s in self.sites]
        self.features = list(metadata["feature_cols"])
        self.cond = self.features.index("conductivity")

    def build(self, tables, grid, inventory, as_of):
        return site_window(tables, grid, self.metadata, inventory, as_of)

    def scores(self, window, anchors):
        """(n_anchors, n_sites) robust normalised residual, plus the live mask."""
        values, node_mask = window
        median = self.metadata["error_median"]
        iqr = self.metadata["error_iqr"]
        out = np.full((len(anchors), len(self.sites)), np.nan)
        live = np.zeros_like(out, dtype=bool)
        with torch.no_grad():
            for row, anchor in enumerate(anchors):
                sequence = values[anchor - WINDOW : anchor]
                target = values[anchor]
                seq_t = torch.from_numpy(sequence).unsqueeze(0).to(Config.DEVICE)
                mask_t = (
                    torch.from_numpy(node_mask[anchor - WINDOW : anchor])
                    .unsqueeze(0)
                    .to(Config.DEVICE)
                )
                pred = model_calls.run_sequence_model(self.model, seq_t, self.edge_index, mask_t)
                err = torch.abs(
                    pred[0, :, self.cond] - torch.from_numpy(target[:, self.cond]).to(Config.DEVICE)
                )
                raw = err.cpu().numpy()
                for node, (site, column) in enumerate(zip(self.sites, self.columns, strict=True)):
                    out[row, node] = (raw[column] - median[site]) / iqr[site]
                live[row] = node_mask[anchor][self.columns]
        return out, live


class CobbleDetector:
    """
    Cobble Shoal reduced to one score per site, so the two tables can be joined.

    The score at a site is the one at its conductivity node. That throws away
    the other variables, which is a real loss and the price of comparing against
    a model whose nodes are whole sites.
    """

    family = "nested_node_batch"

    def __init__(self, tag, bundle, sites):
        self.model_id = f"cobble_shoal:{tag}"
        self.bundle = bundle
        self.sites = list(sites)

    def build(self, tables, grid, inventory, as_of):
        win = nw.build_window(
            tables,
            grid[0],
            grid[-1],
            inventory=inventory,
            scaler=self.bundle["scaler"],
            as_of=as_of,
        )
        prepare = self.bundle["prepare"]
        return prepare(win) if prepare else win

    def scores(self, window, anchors):
        keys = [n.key for n in window["nodes"]]
        columns = [keys.index(f"{s}.conductivity") for s in self.sites]
        out = np.full((len(anchors), len(self.sites)), np.nan)
        live = np.zeros_like(out, dtype=bool)
        for row, anchor in enumerate(anchors):
            batch = nw.to_batch(window, anchor, WINDOW)
            s = np.asarray(self.bundle["net"].score(batch, self.bundle["nulls"])).ravel()
            out[row] = s[columns]
            live[row] = window["target_mask"][anchor][columns]
        return out, live


def calibrate(detector, tables, inventory, as_of, anchors, grid):
    """
    One threshold per (site, rate) from this detector's own fault free scores.

    Per site rather than one pooled number, because Dusk Crayfish's whole design
    is a per node threshold and pooling it would flatten exactly the thing that
    makes a hot site readable. The pooled figure goes out too, under site
    __all__, so a join can pick either.
    """
    window = detector.build(tables, grid, inventory, as_of)
    scores, live = detector.scores(window, anchors)
    rows = []
    for node, site in enumerate(detector.sites):
        column = scores[live[:, node], node]
        column = column[np.isfinite(column)]
        for rate in RATES:
            rows.append(
                {
                    "model_id": detector.model_id,
                    "site": site,
                    "rate": rate,
                    "threshold": float(np.quantile(column, 1.0 - rate)) if column.size else np.nan,
                    "n_null_scores": int(column.size),
                }
            )
    pooled = scores[live]
    pooled = pooled[np.isfinite(pooled)]
    for rate in RATES:
        rows.append(
            {
                "model_id": detector.model_id,
                "site": "__all__",
                "rate": rate,
                "threshold": float(np.quantile(pooled, 1.0 - rate)) if pooled.size else np.nan,
                "n_null_scores": int(pooled.size),
            }
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cobble", nargs="*", default=["shipped", "clock", "weather"])
    ap.add_argument("--null-windows", type=int, default=800)
    ap.add_argument("--out", default=None, help="directory for the two CSVs")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = inv.load()
    tables, _ = drop_implausible(
        load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    )
    as_of = max(f.index.max() for f in tables.values() if len(f))

    metadata = dusk_metadata()
    from strawberrywatch.utils.graph_utils import create_graph_topology

    edge_index, _locations, _idx = create_graph_topology()

    from strawberrywatch.models.Cobble_Shoal import SITE_INVENTORY

    cobble_sites = list(SITE_INVENTORY)
    dusk_sites = list(metadata["location_to_idx"])
    shared = [s for s in cobble_sites if s in dusk_sites]
    cobble_only = [s for s in cobble_sites if s not in dusk_sites]
    print(f"joinable sites: {', '.join(shared)}")
    print(f"cobble shoal only, not in the comparison: {', '.join(cobble_only) or 'none'}")
    print(f"pickle node_thresholds (reference only, not used): {metadata['node_thresholds']}\n")

    detectors = [DuskDetector(metadata, load_dusk(metadata), edge_index, shared)]
    for tag in args.cobble:
        try:
            detectors.append(CobbleDetector(tag, resolve_model(tag), shared))
        except Exception as exc:  # noqa: BLE001
            print(f"cobble_shoal:{tag} unavailable, skipped ({exc})")

    grid = pd.date_range(COMMON_START, COMMON_END, freq=nw.GRID_FREQ, tz="UTC")
    reference = nw.build_window(tables, grid[0], grid[-1], inventory=inventory, as_of=as_of)
    anchors = nw.fault_free_anchors(reference, args.null_windows, WINDOW)
    print(
        f"calibrating on {len(anchors)} fault-free anchors "
        f"in {COMMON_START} .. {COMMON_END}, one threshold per model per site\n"
    )

    thresholds = []
    for det in detectors:
        t0 = time.time()
        rows = calibrate(det, tables, inventory, as_of, anchors, grid)
        thresholds.extend(rows)
        shown = {r["site"]: round(r["threshold"], 3) for r in rows if r["rate"] == 1e-3}
        print(f"{det.model_id:26} @1e-3 {shown}  [{time.time() - t0:.0f}s]")
    print()

    scores = []
    for folder, label, catalog_sites, expect in event_catalog.by_folder():
        tabs = event_tables(folder)
        if not tabs:
            print(f"{label}: no readable site CSVs, skipped")
            continue
        spans = [df.index for df in tabs.values() if len(df)]
        start = min(s.min() for s in spans).floor(nw.GRID_FREQ)
        end = max(s.max() for s in spans).ceil(nw.GRID_FREQ)
        event_grid = pd.date_range(start, end, freq=nw.GRID_FREQ, tz="UTC")
        event_anchors = list(range(WINDOW, len(event_grid) - 1))
        for det in detectors:
            window = det.build(tabs, event_grid, inventory, end)
            value, live = det.scores(window, event_anchors)
            for row, anchor in enumerate(event_anchors):
                for node, site in enumerate(det.sites):
                    scores.append(
                        {
                            "model_id": det.model_id,
                            "model_family": det.family,
                            "event": label,
                            "label": expect,
                            "catalog_site": site in catalog_sites,
                            "timestamp": event_grid[anchor].isoformat(),
                            "site": site,
                            "score": float(value[row, node]),
                            "live": bool(live[row, node]),
                        }
                    )
        print(f"scored {label} ({len(event_anchors)} steps)")

    score_path, threshold_path = (
        out_dir / "cross_model_scores.csv",
        out_dir / "cross_model_thresholds.csv",
    )
    _write(score_path, scores)
    _write(threshold_path, thresholds)
    print(f"\nwrote {score_path} ({len(scores)} rows)")
    print(f"wrote {threshold_path} ({len(thresholds)} rows)")
    _summary(scores, thresholds)
    return 0


def _write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(scores, thresholds):
    """The join the two tables exist to make, printed once so the run says something."""
    s = pd.DataFrame(scores)
    t = pd.DataFrame(thresholds)
    if s.empty or t.empty:
        return
    joined = s[s["live"]].merge(t[t["rate"] == 1e-3], on=["model_id", "site"], how="inner")
    joined["over"] = joined["score"] > joined["threshold"]
    # Only the sites the catalog names for that event. A firing somewhere else
    # is a different question from whether the event was caught.
    at_target = joined[joined["catalog_site"]]
    table = at_target.groupby(["event", "label", "model_id"])["over"].sum().unstack(fill_value=0)
    print("\n### steps over the matched 1e-3 threshold, at the catalog's own sites")
    print(table.to_string())


if __name__ == "__main__":
    raise SystemExit(main())

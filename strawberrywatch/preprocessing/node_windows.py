"""
The window pipeline the real corpus and the synthetic generator both run through.

Cobble Shoal reads per-node series where a node is a (site, variable) pair, and
prepare_sequences_normalized produces site-by-feature windows, so the model was
never reachable from the package pipeline. build_window() is the missing half.

NodeScaler, regrid_to_nodes and add_context_features moved here out of
tests/synthetic/creek_synthetic.py so the generator and the adapter share one regrid
instead of two that agree by inspection.

Sensor state goes through the mask, never through filling. A probe that was not
installed comes out with target_mask False and staleness at its ceiling. Zero
filling it would hand the model a reading it treats as real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch import inventory as inventory_module
from strawberrywatch.models.Cobble_Shoal import (
    CANONICAL_VARS,
    CONTEXT_FEATURES,
    GRID_FREQ,
    GRID_MINUTES,
    SITE_ORDER,
    TARGET_TOLERANCE,
    build_node_registry,
)

# inventory names on the left, Cobble Shoal names on the right. ph has no
# canonical variable so it gets dropped
VARIABLE_MAP = {
    "conductivity": "conductivity",
    "temperature": "temperature",
    "depth": "depth_dev",
    "dissolved_oxygen": "do_pct",
    "floating_conductivity": "float_cond",
}

# scnf010 is Wickson Footbridge under its old site code. the inventory speaks
# table names, the model speaks site names, and the inventory is not moving
TABLE_TO_SITE = {"scnf010": "footbridge"}
SITE_TO_TABLE = {v: k for k, v in TABLE_TO_SITE.items()}


class NodeScaler:
    """
    Per node z-score, fitted on observed values only and saved alongside the
    weights. Carried forward values are excluded from the fit because a node
    that reports rarely would otherwise have its own stale readings dominate
    its own statistics.

    Refitting this at inference is the failure that silently shifted the
    conductivity mean out of training space last time. Fit once, save, load.
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, values, observed, min_std=1e-3):
        n = values.shape[1]
        self.mean = np.zeros(n, dtype=np.float32)
        self.std = np.ones(n, dtype=np.float32)
        for i in range(n):
            v = values[observed[:, i], i]
            if v.size >= 2:
                self.mean[i] = float(np.mean(v))
                self.std[i] = max(float(np.std(v)), min_std)
        return self

    def transform(self, values):
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse(self, scaled):
        return scaled * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def regrid_to_nodes(site_frames, grid, nodes):
    """
    Build node level arrays. Three outputs where there used to be two, because
    "can this step be a model input" and "can this step be a scoring target" are
    different questions:

      values       (T, N) last observation carried forward, backward only, so
                          the model never sees the future
      staleness    (T, N) grid steps since that observation
      target_val   (T, N) nearest observation within half a grid step, NaN
                          otherwise, which is what a prediction is scored against
      target_mask  (T, N) whether target_val exists

    Scoring against the carried forward value instead of the real one would let
    a model score perfectly by copying its own anchor.
    """
    t, n = len(grid), len(nodes)
    values = np.zeros((t, n), dtype=np.float32)
    staleness = np.full((t, n), float(t), dtype=np.float32)
    target_val = np.full((t, n), np.nan, dtype=np.float32)
    target_mask = np.zeros((t, n), dtype=bool)

    gf = pd.DataFrame({"grid_time": grid}).sort_values("grid_time")

    for i, node in enumerate(nodes):
        df = site_frames.get(node.site)
        if df is None or node.var not in df.columns:
            continue
        obs = df[[node.var]].dropna().reset_index()
        if obs.empty:
            continue
        obs = obs.rename(columns={"datetime": "obs_time"}).sort_values("obs_time")

        back = pd.merge_asof(
            gf, obs, left_on="grid_time", right_on="obs_time", direction="backward"
        )
        have = back["obs_time"].notna().to_numpy()
        age = (
            (back["grid_time"] - back["obs_time"]).dt.total_seconds() / 60.0 / GRID_MINUTES
        ).to_numpy()
        vals = back[node.var].to_numpy(dtype=float)
        fill = float(np.nanmean(vals[have])) if have.any() else 0.0
        values[:, i] = np.where(have, vals, fill)
        staleness[:, i] = np.where(have, age, float(t))

        near = pd.merge_asof(
            gf,
            obs,
            left_on="grid_time",
            right_on="obs_time",
            direction="nearest",
            tolerance=TARGET_TOLERANCE,
        )
        hit = near["obs_time"].notna().to_numpy()
        target_mask[:, i] = hit
        target_val[hit, i] = near.loc[hit, node.var].to_numpy()

    return values, staleness, target_val, target_mask


def add_context_features(grid):
    """
    Shared exogenous channels, present at every step, so no mask.

    Two features, both clock. The weather arguments this function used to take
    are gone: rain handling moved to the web application, so rain, air
    temperature and shortwave radiation are not available to the model at
    inference and must not be available to it in training either.
    """
    ctx = pd.DataFrame(index=grid, columns=CONTEXT_FEATURES, dtype=float)
    ha = 2 * np.pi * (grid.hour + grid.minute / 60.0) / 24.0
    ctx["hour_sin"], ctx["hour_cos"] = np.sin(ha), np.cos(ha)
    return ctx.ffill().fillna(0.0)


def roster_from_inventory(inventory=None, site_order=SITE_ORDER):
    """
    {site: [canonical vars]} for the sites this model covers.

    Off the inventory, so adding a probe is one edit to inventory.yaml. A sensor
    the grid marks "-" is not fitted and yields no node. Whether a fitted probe
    is switched on is a per-timestep question the mask answers.
    """
    inventory = inventory or inventory_module.load()
    roster = {}
    for table in inventory.tables:
        site = TABLE_TO_SITE.get(table, table)
        if site not in site_order:
            continue
        variables = []
        for name in sorted(inventory.sites[table].sensors):
            sensor = inventory.sites[table].sensors[name]
            if not sensor.fitted:
                continue
            for variable in sensor.variables:
                canonical = VARIABLE_MAP.get(variable)
                if canonical in CANONICAL_VARS and canonical not in variables:
                    variables.append(canonical)
        roster[site] = [v for v in CANONICAL_VARS if v in variables]
    return {s: roster[s] for s in site_order if s in roster}


class RosterMismatch(ValueError):
    """The inventory describes a creek the shipped weights are not sized for."""


def check_roster(roster, expected):
    """
    Refuse a roster the checkpoint cannot take, naming what moved.

    Node count is baked into the weights, so a different roster either dies deep
    in a matmul or, worse, loads and scores the wrong node.
    """
    for site in sorted(set(roster) | set(expected)):
        got, want = roster.get(site), expected.get(site)
        if got != want:
            raise RosterMismatch(
                f"the inventory gives {site} {got}, but the shipped weights were "
                f"built for {want}. Retrain, or correct the inventory. "
                f"Node counts: inventory {sum(len(v) for v in roster.values())}, "
                f"weights {sum(len(v) for v in expected.values())}."
            )
    return roster


def _install_masked(frame, table, variable, inventory, as_of):
    """
    One column with everything that is not a measurement set to NaN.

    Sentinels first. raw_data_loader keeps them on purpose, since naming a
    sentinel is a quality test's job, but this is a model input path and -9999
    is not a conductivity. Left in, 258 of them at north_fork_0 put that node's
    temperature sd at 306 degrees and flattened every real excursion.

    Then whatever resolve_series calls NOT_INSTALLED: not yet in the creek,
    removed, switched off, or inside a Balance site's unretrievable window.
    """
    clean = frame.mask(frame.isin(list(inventory.sentinel_values)))
    states = inventory.resolve_series(table, variable, clean, as_of=as_of)
    return clean.where(states.to_numpy() != inventory_module.NOT_INSTALLED)


def site_frames_from_archive(tables, inventory=None, as_of=None, roster=None):
    """
    {site: DataFrame} in the model's vocabulary, masked by the inventory.

    depth maps straight onto depth_dev with no datum removed. Subtracting the
    site median looked tidy and was lookahead, since the median of the whole
    series is a function of the future and a reading in April moved the scaled
    value in March. NodeScaler centres each node on the corpus once instead.
    """
    inventory = inventory or inventory_module.load()
    roster = roster or roster_from_inventory(inventory)
    frames = {}
    for site, variables in roster.items():
        table = SITE_TO_TABLE.get(site, site)
        source = tables.get(table)
        if source is None or source.empty:
            frames[site] = pd.DataFrame(columns=variables).rename_axis("datetime")
            continue
        columns = {}
        for canonical in variables:
            raw = next(k for k, v in VARIABLE_MAP.items() if v == canonical)
            if raw not in source.columns:
                continue
            columns[canonical] = _install_masked(source[raw], table, raw, inventory, as_of)
        frames[site] = pd.DataFrame(columns, index=source.index).rename_axis("datetime")
    return frames


def build_window(tables, start, end, inventory=None, scaler=None, as_of=None):
    """
    A real corpus window in exactly the shape make_synthetic_creek returns.

    Same keys, same dtypes, same regrid, so anything scoring a synthetic window
    scores a real one without a branch. scaler=None fits one here; pass a fitted
    one on the serving path, where fitting at inference is the whole defect.
    """
    inventory = inventory or inventory_module.load()
    roster = check_roster(roster_from_inventory(inventory), _expected_roster())
    nodes, node_site, node_var = build_node_registry(roster, list(roster))

    grid = pd.date_range(start=start, end=end, freq=GRID_FREQ, tz="UTC")
    frames = site_frames_from_archive(tables, inventory, as_of=as_of, roster=roster)
    values, staleness, target_val, target_mask = regrid_to_nodes(frames, grid, nodes)

    if scaler is None:
        scaler = NodeScaler().fit(values, target_mask)

    return {
        "values": scaler.transform(values),
        "staleness": staleness,
        "target_val": np.nan_to_num(scaler.transform(target_val)).astype(np.float32),
        "target_mask": target_mask,
        "context": add_context_features(grid).to_numpy(dtype=np.float32),
        "grid": grid,
        "nodes": nodes,
        "node_site": node_site,
        "node_var": node_var,
        "scaler": scaler,
        "roster": roster,
    }


# Spans the audit catalog labels as events, plus the botanical actuator period
# the catalog dropped. Fault-free has to mean fault-free: a null fitted over a
# known spill has the spill in its own tail and moves the threshold past it.
EVENT_SPANS = (
    ("2025-06-10", "2025-06-15"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-08", "2025-09-13"),
    ("2025-11-03", "2025-11-08"),
    ("2025-11-11", "2025-11-16"),
    ("2026-01-01", "2026-02-25"),
    ("2026-03-18", "2026-03-23"),
    ("2026-03-30", "2026-04-03"),
)


def corpus_window(tables, inventory=None, as_of=None, scaler=None):
    """One window spanning the whole archive, and the scaler fitted on it."""
    spans = [f.index for f in tables.values() if len(f)]
    start = min(s.min() for s in spans).floor(GRID_FREQ)
    end = max(s.max() for s in spans).ceil(GRID_FREQ)
    return build_window(tables, start, end, inventory=inventory, scaler=scaler, as_of=as_of)


def _labelled(grid, spans=EVENT_SPANS):
    bad = np.zeros(len(grid), dtype=bool)
    for lo, hi in spans:
        bad |= (grid >= pd.Timestamp(lo, tz="UTC")) & (grid < pd.Timestamp(hi, tz="UTC"))
    return bad


def fault_free_anchors(win, count, window=24, min_coverage=0.5, spans=EVENT_SPANS):
    """
    Anchor steps for fitting a null: clear of the labelled events, and reporting.

    Spread over the corpus, not taken from the front. The archive opens in May
    2021 with only scnf010 in it, that being the inferred install date the
    inventory flags as most likely wrong, so the first four years are one node
    and sixteen holes.

    min_coverage is the fraction of nodes needing a real observation at the
    anchor. Half is a judgment call that drops the single-node prologue.
    """
    bad = _labelled(win["grid"], spans)
    mask = win["target_mask"]
    n_nodes = mask.shape[1]
    usable = [
        anchor
        for anchor in range(window, len(win["grid"]) - 1)
        if not bad[anchor - window : anchor + 1].any()
        and mask[anchor].sum() >= min_coverage * n_nodes
    ]
    if len(usable) <= count:
        return usable
    take = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[i] for i in sorted(set(take))]


def to_batch(win, anchor, window):
    """
    One (B=1) encoder batch cut out of a window dict, real or synthetic.

    Graph tensors come from the window's own roster so a batch cannot describe a
    different creek from the values in it. predict_all_masked asserts B == 1.
    """
    import torch

    from strawberrywatch.models.Cobble_Shoal import (
        build_site_matrix,
        build_variable_adjacency,
        inventory_matrix,
    )

    roster = win.get("roster") or _expected_roster()
    sites = list(roster)
    node_site = win["node_site"]
    lo, hi = anchor - window, anchor

    stale = torch.tensor(win["staleness"][lo:hi], dtype=torch.float32).unsqueeze(0)
    return {
        "values": torch.tensor(win["values"][lo:hi], dtype=torch.float32).unsqueeze(0),
        "staleness": stale,
        "context": torch.tensor(win["context"][lo:hi], dtype=torch.float32).unsqueeze(0),
        "target": torch.tensor(win["target_val"][hi], dtype=torch.float32).unsqueeze(0),
        "target_mask": torch.tensor(win["target_mask"][hi], dtype=torch.bool).unsqueeze(0),
        "obs_mask": stale < 1.0,
        "node_site": node_site,
        "node_var": win["node_var"],
        "site_inventory": inventory_matrix(sites, roster),
        "a_var": build_variable_adjacency(node_site),
        "site_matrix": build_site_matrix(node_site, len(sites)),
        "nodes": win["nodes"],
    }


def _expected_roster():
    """What the shipped weights were built for. Late import, circular otherwise."""
    from strawberrywatch.models.Cobble_Shoal import SITE_INVENTORY

    return {s: list(v) for s, v in SITE_INVENTORY.items()}


__all__ = [
    "EVENT_SPANS",
    "GRID_FREQ",
    "GRID_MINUTES",
    "SITE_TO_TABLE",
    "TABLE_TO_SITE",
    "VARIABLE_MAP",
    "NodeScaler",
    "RosterMismatch",
    "add_context_features",
    "build_window",
    "check_roster",
    "corpus_window",
    "fault_free_anchors",
    "regrid_to_nodes",
    "roster_from_inventory",
    "site_frames_from_archive",
    "to_batch",
]

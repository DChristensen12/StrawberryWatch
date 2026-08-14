"""
Synthetic creek generator with known ground truth: the project's only test set.

There are no labelled anomalies in the real data and there never will be, so
injected faults with known truth are the whole basis for measuring detection.
That makes this test infrastructure rather than scratch, which is why it lives
under tests/ and not beside the prototypes it was written next to.

Faults: stuck, stale, partial, drift, spike, decouple, slow_all.

The window pipeline pieces below (NodeScaler, regrid_to_nodes,
add_context_features) moved with it. Generated series are pushed through the
same regridding the real ingestion path uses, so cadence and dropout exercise
that path rather than bypassing it.

Graph structure and canonical variables are imported from the model, so the
generator and the model cannot disagree about the creek.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch.models.Riffle_Darner import (
    CONTEXT_FEATURES,
    FLOW_EDGES,
    GRID_FREQ,
    GRID_MINUTES,
    SITE_INVENTORY,
    SITE_ORDER,
    TARGET_TOLERANCE,
    build_node_registry,
)

# Window pipeline, moved with the generator


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


# Fault generator

EPOCH = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")

# Steps of travel per flow edge. One hop is an hour at a 15 minute grid.
TRAVEL_STEPS = 4

# Per site baseline, roughly the levels the real network sits at, so the scaler
# has something site-specific to find.
BASELINE = {
    "north_fork_0": {
        "conductivity": 600.0,
        "depth_dev": 0.0,
        "temperature": 15.0,
        "do_pct": 90.0,
        "float_cond": 600.0,
    },
    "footbridge": {
        "conductivity": 370.0,
        "depth_dev": 0.0,
        "temperature": 14.0,
        "do_pct": 88.0,
        "float_cond": 370.0,
    },
    "south_fork_1": {
        "conductivity": 820.0,
        "depth_dev": 0.0,
        "temperature": 16.0,
        "do_pct": 85.0,
        "float_cond": 820.0,
    },
    "south_fork_2": {
        "conductivity": 780.0,
        "depth_dev": 0.0,
        "temperature": 16.5,
        "do_pct": 86.0,
        "float_cond": 780.0,
    },
    "oxford": {
        "conductivity": 390.0,
        "depth_dev": 0.0,
        "temperature": 15.5,
        "do_pct": 92.0,
        "float_cond": 390.0,
    },
}

DIURNAL_AMP = {
    "conductivity": 12.0,
    "depth_dev": 4.0,
    "temperature": 2.5,
    "do_pct": 3.0,
    "float_cond": 12.0,
}
NOISE_SD = {
    "conductivity": 1.5,
    "depth_dev": 0.8,
    "temperature": 0.15,
    "do_pct": 0.5,
    "float_cond": 1.8,
}

# Raw-unit fault sizes per variable, scaled by the sweep multipliers.
BASE_MAG = {
    "conductivity": 120.0,
    "depth_dev": 60.0,
    "temperature": 8.0,
    "do_pct": 15.0,
    "float_cond": 120.0,
}
MULTS = (0.5, 1.0, 2.0)

# Residual-noise fractions for the partial-freeze sweep, as a fraction of the
# node's own normal variance over the fault window.
PARTIAL_FRACS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)

SHAPES = ("stuck", "stale", "partial", "drift", "spike", "decouple", "slow_all")

# Fault window and the anchor step scored inside it. The scored window is
# [anchor - WIN, anchor), which sits strictly inside [F_LO, F_HI).
F_LO, F_HI, F_ANCHOR = 300, 360, 340


def flow_weights(edges=FLOW_EDGES):
    """
    Generator-side edge weights. A destination fed by several upstreams splits
    evenly between them, which is the same even split the model starts from but
    is here a fact about the data rather than a parameter.
    """
    inflow = {}
    for _src, dst in edges:
        inflow[dst] = inflow.get(dst, 0) + 1
    return {(s, d): 1.0 / inflow[d] for s, d in edges}


def downstream_paths(source, edges=FLOW_EDGES):
    """
    Every site reachable downstream of source, with the product of the edge
    weights along the way and the hop count. Flow-unconnected sites are simply
    absent from the result, which is what makes the isolation test meaningful.
    """
    w = flow_weights(edges)
    out, frontier = {}, [(source, 1.0, 0)]
    while frontier:
        node, att, hops = frontier.pop()
        for src, dst in edges:
            if src != node:
                continue
            a, h = att * w[(src, dst)], hops + 1
            if dst not in out or a > out[dst][0]:
                out[dst] = (a, h)
            frontier.append((dst, a, h))
    return out


def unconnected_pairs(sites=SITE_ORDER, edges=FLOW_EDGES):
    """Ordered pairs with no directed flow path, enumerated from the edges."""
    pairs = []
    for a in sites:
        reach = downstream_paths(a, edges)
        for b in sites:
            if a != b and b not in reach:
                pairs.append((a, b))
    return pairs


def _series(site, var, times, rng, spill, rain):
    """Ground truth value for one (site, var) at arbitrary times."""
    hours = (times - EPOCH).total_seconds().to_numpy() / 3600.0
    base = BASELINE[site][var]
    v = base + DIURNAL_AMP[var] * np.sin(2 * np.pi * hours / 24.0)
    v = v + rng.normal(0.0, NOISE_SD[var], size=len(times))

    steps = hours * 60.0 / GRID_MINUTES

    if spill and var in ("conductivity", "float_cond"):
        reach = downstream_paths(spill["site"])
        if site == spill["site"]:
            att, lag = 1.0, 0
        elif site in reach:
            att, lag = reach[site][0], reach[site][1] * TRAVEL_STEPS
        else:
            att, lag = 0.0, 0
        if att > 0:
            lo = spill["start"] + lag
            hi = lo + spill["duration"]
            v = v + np.where((steps >= lo) & (steps < hi), spill["magnitude"] * att, 0.0)

    if rain and var in ("conductivity", "float_cond"):
        lo, hi = rain["start"], rain["start"] + rain["duration"]
        # No lag and no attenuation: rain lands on the whole catchment at once.
        # It no longer has an exogenous signature the model can read, because
        # rain handling moved to the web application and the weather channels
        # left the context, so this is now purely a coherent network-wide event.
        v = v + np.where((steps >= lo) & (steps < hi), rain["magnitude"], 0.0)

    return v


def _obs_times(n_steps, cadence_min):
    """Observation stamps for one site. Off-grid cadence lands between grid points."""
    span = n_steps * GRID_MINUTES
    k = int(span // cadence_min) + 1
    return pd.DatetimeIndex([EPOCH + pd.Timedelta(minutes=cadence_min * i) for i in range(k)])


def _drop(times, windows):
    """Boolean keep-mask over observation times, dropping [start, end) in grid steps."""
    keep = np.ones(len(times), dtype=bool)
    steps = (times - EPOCH).total_seconds().to_numpy() / 60.0 / GRID_MINUTES
    for lo, hi in windows:
        keep &= ~((steps >= lo) & (steps < hi))
    return keep


def make_synthetic_creek(
    n_steps,
    seed,
    sites=None,
    inventory=None,
    spill=None,
    rain=None,
    dropout=None,
    cadence=None,
    scaler=None,
):
    """
    Build one fault-free window with known ground truth.

    spill:   {"site": str, "start": int, "duration": int, "magnitude": float}
    rain:    {"start": int, "duration": int, "magnitude": float}
    dropout: {site or "site.var": [(start, end), ...]} in grid steps
    cadence: {site: minutes between observations}, default 15 and on-grid

    Returns the dict build_window returns, plus "truth". Note that no weather is
    written into the context: it is two clock features and nothing else.
    """
    sites = list(sites or SITE_ORDER)
    inventory = inventory or SITE_INVENTORY
    dropout = dropout or {}
    cadence = cadence or {}
    rng = np.random.default_rng(seed)

    grid = pd.date_range(start=EPOCH, periods=n_steps, freq=GRID_FREQ, tz="UTC")
    nodes, node_site, node_var = build_node_registry(inventory, sites)

    frames = {}
    for site in sites:
        times = _obs_times(n_steps, cadence.get(site, GRID_MINUTES))
        cols = {}
        for var in inventory.get(site, []):
            vals = _series(site, var, times, rng, spill, rain)
            keep = _drop(times, dropout.get(site, []))
            keep &= _drop(times, dropout.get(f"{site}.{var}", []))
            cols[var] = np.where(keep, vals, np.nan)
        frames[site] = pd.DataFrame(cols, index=times).rename_axis("datetime")

    values, staleness, target_val, target_mask = regrid_to_nodes(frames, grid, nodes)

    if scaler is None:
        scaler = NodeScaler().fit(values, target_mask)

    truth = {
        "spill": dict(spill) if spill else None,
        "spill_sites": (
            {spill["site"]: (1.0, 0), **downstream_paths(spill["site"])} if spill else {}
        ),
        "rain": dict(rain) if rain else None,
        "unconnected_pairs": unconnected_pairs(sites),
        "flow_weights": flow_weights(),
        "travel_steps": TRAVEL_STEPS,
        "raw_values": values.copy(),
    }

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
        "truth": truth,
    }


def node_magnitude(win, node_index, mult=1.0):
    """Raw-unit fault size for this node's variable, in that node's scaled units."""
    var = win["nodes"][node_index].var
    return BASE_MAG[var] * mult / float(win["scaler"].std[node_index])


def inject(win, node_index, shape, mag=0.0, frac=0.0, seed=0, lo=F_LO, hi=F_HI, anchor=F_ANCHOR):
    """
    Fault one node (or, for slow_all, every node) over [lo, hi) of a fault-free
    window. Returns (values, target, truth) where values is the full modified
    (T, N) scaled series, target is the modified (N,) observation at the anchor
    step, and truth names what was done.

    Nothing here reads the model, and the returned truth is the only thing the
    scoring is allowed to compare against.
    """
    v = win["values"].copy()
    t = win["target_val"][anchor].copy()
    rng = np.random.default_rng(seed)
    faulted = [node_index]

    if shape == "stuck":
        v[lo:hi, node_index] = v[lo, node_index]
        t[node_index] = v[lo, node_index]
    elif shape == "stale":
        v[lo:hi, node_index] = v[lo, node_index]
    elif shape == "partial":
        # The node's own normal variance over this stretch, before freezing it.
        sd = float(np.std(win["values"][lo:hi, node_index]))
        noise = rng.normal(0.0, sd * frac, size=hi - lo)
        v[lo:hi, node_index] = v[lo, node_index] + noise
        t[node_index] = v[anchor, node_index]
    elif shape == "drift":
        ramp = np.linspace(0.0, mag, hi - lo)
        v[lo:hi, node_index] += ramp
        t[node_index] += ramp[anchor - lo]
    elif shape == "spike":
        # Onset at the anchor: the window the model conditions on is clean and
        # the step change lands on the observation being predicted.
        v[anchor:hi, node_index] += mag
        t[node_index] += mag
    elif shape == "decouple":
        # Reflect about the window mean. Mean and variance are preserved
        # exactly, so the marginal distribution is untouched and only the
        # correlation with the node's siblings is inverted. This is calibration
        # drift, and it is the case reconstruction is claimed to catch when
        # forecast and leave-one-out cannot.
        seg = v[lo:hi, node_index]
        v[lo:hi, node_index] = 2.0 * seg.mean() - seg
        t[node_index] = v[anchor, node_index]
    elif shape == "slow_all":
        # Every node drifts together. The forecast sees a tiny per-step delta
        # and the leave-one-out channel sees neighbours that agree, so both
        # should be weak; the window shape is the only thing that changed.
        faulted = list(range(v.shape[1]))
        for i in faulted:
            ramp = np.linspace(0.0, node_magnitude(win, i, mag), hi - lo)
            v[lo:hi, i] += ramp
            t[i] += ramp[anchor - lo]
    else:
        raise ValueError(f"unknown fault shape {shape!r}")

    truth = {
        "shape": shape,
        "nodes": faulted,
        "magnitude": float(mag),
        "frac": float(frac),
        "window": (lo, hi),
        "anchor": anchor,
        "node_keys": [win["nodes"][i].key for i in faulted],
    }
    return v, t, truth


def fault_cases(win, shapes=SHAPES, mults=MULTS, fracs=PARTIAL_FRACS, seed=0):
    """
    Every fault case in the sweep: shape, node, magnitude where applicable.

    Yields (values, target, truth). slow_all is emitted once per magnitude
    rather than once per node, because it faults every node at once.
    """
    n = win["values"].shape[1]
    for shape in shapes:
        if shape == "slow_all":
            for mult in mults:
                yield inject(win, 0, shape, mag=mult, seed=seed)
        elif shape == "partial":
            for i in range(n):
                for frac in fracs:
                    yield inject(win, i, shape, frac=frac, seed=seed + i)
        elif shape in ("stuck", "stale", "decouple"):
            for i in range(n):
                yield inject(win, i, shape, seed=seed + i)
        else:
            for i in range(n):
                for mult in mults:
                    yield inject(win, i, shape, mag=node_magnitude(win, i, mult), seed=seed + i)

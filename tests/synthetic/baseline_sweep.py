"""
The fault sweep, run over any detector that exposes score(batch, nulls).

Lives beside creek_synthetic because it is the same kind of thing. Models and
control charts go through one code path here. A comparison whose two halves are
scored by two functions is a comparison of the two functions.

Dusk Crayfish is absent by construction. Its feature_cols want rain_mm,
air_temp_c and shortwave_radiation, and creek_synthetic produces no weather, so
there is nothing to feed it. It gets compared on the real audit events instead.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from strawberrywatch.models.Cobble_Shoal import (
    SITE_ORDER,
    build_node_registry,
    build_site_matrix,
    build_variable_adjacency,
    inventory_matrix,
)
from tests.synthetic import creek_synthetic as cs
from tests.synthetic import nested_batch

# n_steps has to clear F_HI or inject indexes past the end of the series
ANCHOR = cs.F_ANCHOR
WINDOW = nested_batch.WINDOW
N_STEPS = ANCHOR + WINDOW

CALIBRATION_SEEDS = tuple(range(1000, 1200))
SWEEP_SEEDS = (0, 1, 2, 3, 4)


@functools.cache
def _graph():
    """Node registry and adjacency, built once. They do not depend on the window."""
    nodes, node_site, node_var = build_node_registry()
    return {
        "nodes": nodes,
        "node_site": node_site,
        "node_var": node_var,
        "site_inventory": inventory_matrix(),
        "a_var": build_variable_adjacency(node_site),
        "site_matrix": build_site_matrix(node_site, len(SITE_ORDER)),
    }


@functools.lru_cache(maxsize=64)
def _window(seed):
    """
    One generated creek per seed, cached.

    build_batch regenerates the whole series every call. Fine for a handful of
    batches, quadratic misery over 1375 cases, since the regrid alone is a
    merge_asof per node.
    """
    return cs.make_synthetic_creek(n_steps=N_STEPS, seed=seed)


def _batch(win, values=None, target=None):
    """The encoder's batch dict off an already generated window."""
    graph = _graph()
    v = win["values"] if values is None else values
    lo, hi = ANCHOR - WINDOW, ANCHOR

    stale = torch.tensor(win["staleness"][lo:hi], dtype=torch.float32).unsqueeze(0)
    tgt = win["target_val"][hi] if target is None else target
    return {
        "values": torch.tensor(v[lo:hi], dtype=torch.float32).unsqueeze(0),
        "staleness": stale,
        "context": torch.tensor(win["context"][lo:hi], dtype=torch.float32).unsqueeze(0),
        "target": torch.tensor(tgt, dtype=torch.float32).unsqueeze(0),
        "target_mask": torch.tensor(win["target_mask"][hi], dtype=torch.bool).unsqueeze(0),
        "obs_mask": stale < 1.0,
        **graph,
    }


def clean_batch(seed):
    """One fault-free window, cut at the same anchor the faulted ones are."""
    return _batch(_window(seed))


def calibration_batches(seeds=CALIBRATION_SEEDS):
    """Fault-free windows for fitting a null. Disjoint seeds from the sweep."""
    return [clean_batch(s) for s in seeds]


def faulted_batch(seed, values, target):
    """A faulted window on the same graph tensors as the clean one it came from."""
    return _batch(_window(seed), values=values, target=target)


def cases(seed, shapes=cs.SHAPES):
    """Every fault case at one seed, as (batch, truth)."""
    win = _window(seed)
    for values, target, truth in cs.fault_cases(win, shapes=shapes, seed=seed):
        yield _batch(win, values=values, target=target), truth


def rank_of_truth(scores, faulted):
    """
    1-based rank of the best placed true node. 1 means it topped the list.

    Best placed rather than a mean, since slow_all faults every node and any of
    them coming first is a correct localisation.
    """
    order = np.argsort(-np.asarray(scores).ravel(), kind="stable")
    positions = {int(node): i + 1 for i, node in enumerate(order)}
    return min(positions[i] for i in faulted)


def evaluate(detector, nulls, threshold, seeds=SWEEP_SEEDS, shapes=cs.SHAPES):
    """
    Score every case in the sweep. One record per case, no aggregation.

    That is the caller's job. A mean across shapes hides what this exists to find.
    """
    records = []
    for seed in seeds:
        for batch, truth in cases(seed, shapes=shapes):
            scores = np.asarray(detector.score(batch, nulls)).ravel()
            faulted = truth["nodes"]
            finite = scores[np.isfinite(scores)]
            records.append(
                {
                    "seed": seed,
                    "shape": truth["shape"],
                    "magnitude": truth["magnitude"],
                    "frac": truth["frac"],
                    "n_faulted": len(faulted),
                    "fired": bool(finite.size and finite.max() > threshold),
                    "top1": bool(int(np.argmax(np.nan_to_num(scores, nan=-np.inf))) in faulted),
                    "rank": rank_of_truth(scores, faulted),
                    "peak": float(finite.max()) if finite.size else float("nan"),
                    "truth_score": float(np.nanmax(scores[faulted])),
                }
            )
    return records


def by_shape(records):
    """Detection, top-1 and mean rank per fault shape."""
    out = {}
    for shape in sorted({r["shape"] for r in records}):
        rows = [r for r in records if r["shape"] == shape]
        out[shape] = {
            "n": len(rows),
            "detection": float(np.mean([r["fired"] for r in rows])),
            "top1": float(np.mean([r["top1"] for r in rows])),
            "mean_rank": float(np.mean([r["rank"] for r in rows])),
        }
    return out


def false_alarms(detector, nulls, threshold, seeds, calmest_decile=False):
    """
    Rate on fault-free windows, over windows and over node-steps.

    calmest_decile takes the tenth with the least movement, by median over nodes
    of the window sd. A detector that only behaves when the creek is busy is not
    calibrated.
    """
    batches = [clean_batch(s) for s in seeds]
    if calmest_decile:
        activity = [float(np.median(np.std(np.asarray(b["values"])[0], axis=0))) for b in batches]
        keep = np.argsort(activity)[: max(1, len(batches) // 10)]
        batches = [batches[i] for i in keep]

    per_window, per_node = [], []
    for batch in batches:
        scores = np.asarray(detector.score(batch, nulls)).ravel()
        finite = scores[np.isfinite(scores)]
        per_window.append(bool(finite.size and finite.max() > threshold))
        per_node.append(finite > threshold)
    return {
        "n_windows": len(batches),
        "window_rate": float(np.mean(per_window)) if per_window else float("nan"),
        "node_rate": float(np.mean(np.concatenate(per_node))) if per_node else float("nan"),
    }


__all__ = [
    "ANCHOR",
    "CALIBRATION_SEEDS",
    "SWEEP_SEEDS",
    "WINDOW",
    "by_shape",
    "calibration_batches",
    "cases",
    "clean_batch",
    "evaluate",
    "false_alarms",
    "rank_of_truth",
]

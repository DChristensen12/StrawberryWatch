"""
Builds the nested-node batch the per-node models read.

The generator makes (T, N) grids, the encoder wants a batched dict plus the
graph tensors. Two callers need it, so it lives here instead of being copied.
Same seed, window and anchor every time, so scores through it are reproducible.
"""

from __future__ import annotations

import torch

from strawberrywatch.models.Cobble_Shoal import (
    SITE_ORDER,
    build_node_registry,
    build_site_matrix,
    build_variable_adjacency,
    inventory_matrix,
)
from tests import creek_synthetic

# Window length the reconstruction head was built for. A batch whose time axis
# disagrees with it does not fail loudly, it fails at the matmul, so it is
# stated here once.
WINDOW = 24

# Anchor step inside the generated series. Far enough in that every node has a
# populated history behind it.
ANCHOR = 120


def build_batch(seed=0, window=WINDOW, anchor=ANCHOR, values=None, target=None):
    """
    One (B=1) window as the encoder wants it.

    values/target override the generated arrays, which is how a faulted window
    is scored against the same graph tensors as the clean one it came from.
    predict_all_masked asserts B == 1, so the batch axis stays singleton.
    """
    win = creek_synthetic.make_synthetic_creek(n_steps=anchor + window, seed=seed)
    nodes, node_site, node_var = build_node_registry()

    v = win["values"] if values is None else values
    lo, hi = anchor - window, anchor

    val = torch.tensor(v[lo:hi], dtype=torch.float32).unsqueeze(0)
    stale = torch.tensor(win["staleness"][lo:hi], dtype=torch.float32).unsqueeze(0)
    ctx = torch.tensor(win["context"][lo:hi], dtype=torch.float32).unsqueeze(0)

    tgt = win["target_val"][hi] if target is None else target
    tgt = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)
    tgt_mask = torch.tensor(win["target_mask"][hi], dtype=torch.bool).unsqueeze(0)

    # A step counts as observed when the reading landed on it rather than being
    # carried forward into it. Dispersion needs the difference: a node that is
    # merely offline is flat for the same arithmetic reason a frozen one is.
    obs_mask = stale < 1.0

    return {
        "values": val,
        "staleness": stale,
        "context": ctx,
        "node_site": node_site,
        "node_var": node_var,
        "site_inventory": inventory_matrix(),
        "a_var": build_variable_adjacency(node_site),
        "site_matrix": build_site_matrix(node_site, len(SITE_ORDER)),
        "target": tgt,
        "target_mask": tgt_mask,
        "obs_mask": obs_mask,
        "nodes": nodes,
    }


def build_model(metadata=None):
    """Cobble Shoal sized the way its checkpoint was, weights left to the caller."""
    from strawberrywatch.models.Cobble_Shoal import CobbleShoal

    return CobbleShoal.from_metadata(metadata or {"seed": 20260806, "window": WINDOW})


def node_keys():
    """The node identifiers, in the order the score columns come out in."""
    return [n.key for n in build_node_registry()[0]]


__all__ = ["ANCHOR", "WINDOW", "build_batch", "build_model", "node_keys"]

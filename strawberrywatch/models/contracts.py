"""
What a model declares about the data it will accept.

INPUT_CONTRACT says what shape the model reads and lives in model_calls.py.
FLAG_POLICY says which Settling Pool flag levels it refuses, and lives here
beside it. Both are declarations. Neither one enforces anything: enforcement is
the consumer's job. A test knows nothing about the stores that read it.
"""

from __future__ import annotations

from strawberrywatch.models.model_calls import (
    CONTRACTS,
    NESTED_NODE_BATCH,
    SEQUENCE_TENSOR,
    input_contract,
)
from strawberrywatch.support_modules.qc_tests import FAIL, GOOD, MISSING, SUSPECT, UNKNOWN

# Exclude the readings that are not measurements, keep the ones that are merely
# doubtful. SUSPECT stays in with its flag recorded, which is the whole reason
# the levels are not a boolean.
DEFAULT_FLAG_POLICY = (FAIL, MISSING)

# Per contract rather than per model name, so adding a third model that speaks
# an existing contract needs no edit here. A model wanting something else
# declares its own FLAG_POLICY next to its INPUT_CONTRACT.
FLAG_POLICY = {
    SEQUENCE_TENSOR: DEFAULT_FLAG_POLICY,
    NESTED_NODE_BATCH: DEFAULT_FLAG_POLICY,
}


class FlagPolicyError(ValueError):
    """A model declared a policy that is not a set of flag levels."""


_VALID = frozenset((GOOD, UNKNOWN, SUSPECT, FAIL, MISSING))


def flag_policy(model):
    """Which flag levels this model excludes, as a sorted tuple."""
    declared = getattr(model, "FLAG_POLICY", None)
    if declared is None:
        declared = FLAG_POLICY[input_contract(model)]

    levels = tuple(sorted(set(declared)))
    unknown = [level for level in levels if level not in _VALID]
    if unknown:
        raise FlagPolicyError(
            f"{getattr(model, '__name__', type(model).__name__)} excludes flag levels "
            f"{unknown}, which are not QARTOD levels {sorted(_VALID)}"
        )
    if GOOD in levels:
        raise FlagPolicyError(
            f"{getattr(model, '__name__', type(model).__name__)} excludes GOOD, which "
            f"would leave the model nothing to read"
        )
    return levels


def excluded_mask(flags, model):
    """Bool array, True where this model's policy says do not read the cell."""
    import numpy as np

    flags = np.asarray(flags, dtype=np.uint8)
    mask = np.zeros(flags.shape, dtype=bool)
    for level in flag_policy(model):
        mask |= flags == level
    return mask


def admit_grid(flags, nodes, n_steps, model=None):
    """
    Collapse per (site, variable) flags into the (T, N) grid a run reads.

    The reshape Settling Pool deliberately does not do. A node is out at a step
    if any of its variables carries a level the model excludes.
    """
    import numpy as np

    policy = DEFAULT_FLAG_POLICY if model is None else flag_policy(model)
    grid = np.ones((n_steps, len(nodes)), dtype=bool)

    for j, node in enumerate(nodes):
        for (site, _variable), levels in flags.items():
            if site != node:
                continue
            levels = np.asarray(levels, dtype=np.uint8)
            if len(levels) != n_steps:
                raise FlagPolicyError(f"{site} flags are {len(levels)} long, expected {n_steps}")
            for level in policy:
                grid[:, j] &= levels != level
    return grid


__all__ = [
    "CONTRACTS",
    "DEFAULT_FLAG_POLICY",
    "FLAG_POLICY",
    "NESTED_NODE_BATCH",
    "SEQUENCE_TENSOR",
    "FlagPolicyError",
    "admit_grid",
    "excluded_mask",
    "flag_policy",
]

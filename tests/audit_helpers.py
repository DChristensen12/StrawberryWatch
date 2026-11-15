"""
Shared assertions for the audit tests.
"""

import numpy as np


def differs(a, b, what):
    """Asserts two inputs give different answers. +inf counts as one, only all-NaN fails."""
    # hours_since_wet returns +inf before any rain, its documented value
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert not np.isnan(a).all() and not np.isnan(b).all(), f"{what}: all NaN"
    assert not np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99)), (
        f"{what}: identical for two inputs that should differ, so it is pinned"
    )

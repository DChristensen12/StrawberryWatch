"""
Shared assertions for the audit tests.
"""

import numpy as np


def differs(a, b, what):
    """
    Two inputs that should give different answers, and do.

    Infinity counts as a real answer. hours_since_wet returns +inf before any
    rain has fallen and that is the documented value, so only an all-NaN result
    means the statistic could not be computed.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert not np.isnan(a).all() and not np.isnan(b).all(), f"{what}: all NaN"
    assert not np.array_equal(np.nan_to_num(a, nan=-9e99), np.nan_to_num(b, nan=-9e99)), (
        f"{what}: identical for two inputs that should differ, so it is pinned"
    )

"""
Quality tests. Each one takes numbers and returns flags.

Nothing here imports the config, the inventory or the storage layer, and no
function applies, corrects or cleans anything. That split is deliberate: tests
know about arrays, callers know where the arrays came from.
"""

from __future__ import annotations

import numpy as np

# Flag values
GOOD = 1
UNKNOWN = 2
SUSPECT = 3
FAIL = 4
MISSING = 9

FLAGS = (GOOD, UNKNOWN, SUSPECT, FAIL, MISSING)

FLAG_NAMES = {
    GOOD: "GOOD",
    UNKNOWN: "UNKNOWN",
    SUSPECT: "SUSPECT",
    FAIL: "FAIL",
    MISSING: "MISSING",
}


def _array(values):
    return np.asarray(values, dtype=float)


def _base(values):
    """Start every test at UNKNOWN, with the gaps already marked MISSING."""
    v = _array(values)
    flags = np.empty(v.shape, dtype=np.uint8)
    flags.fill(UNKNOWN)
    missing = np.isnan(v)
    flags[missing] = MISSING
    return v, flags


def gross_range(values, fail_range=None, suspect_range=None):
    """Flag readings outside the operator's ranges. Fail wins over suspect."""
    v, flags = _base(values)
    ok = ~np.isnan(v)

    if suspect_range is not None:
        low, high = suspect_range
        flags[ok & ((v < low) | (v > high))] = SUSPECT
    if fail_range is not None:
        low, high = fail_range
        flags[ok & ((v < low) | (v > high))] = FAIL

    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def _run_lengths(v, tolerance):
    """How many steps back each value has been sitting within tolerance."""
    n = len(v)
    runs = np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isnan(v[i]) or np.isnan(v[i - 1]):
            runs[i] = 0
        elif abs(v[i] - v[i - 1]) <= tolerance:
            runs[i] = runs[i - 1] + 1
    return runs


def flat_line(values, suspect_count, fail_count, tolerance=0.0):
    """Flag a reading that has stopped changing for too many steps."""
    v, flags = _base(values)
    ok = ~np.isnan(v)
    runs = _run_lengths(v, tolerance)

    flags[ok & (runs >= suspect_count)] = SUSPECT
    flags[ok & (runs >= fail_count)] = FAIL
    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def attenuated_signal(values, window, min_std=None, min_range=None):
    """
    Flag a probe whose variability has collapsed without going flat.

    The fouled or debris wrapped case. A flat line test misses it because no two
    readings are actually equal.
    """
    v, flags = _base(values)
    n = len(v)
    ok = ~np.isnan(v)

    for i in range(n):
        if not ok[i] or i + 1 < window:
            continue
        chunk = v[i + 1 - window : i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if len(chunk) < 2:
            continue
        if min_std is not None and np.std(chunk, ddof=1) < min_std:
            flags[i] = SUSPECT
        if min_range is not None and (chunk.max() - chunk.min()) < min_range:
            flags[i] = SUSPECT

    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def spike(values, threshold, fail_threshold=None):
    """Flag a single step excursion. Compares against both neighbours."""
    v, flags = _base(values)
    n = len(v)
    ok = ~np.isnan(v)

    for i in range(1, n - 1):
        if not (ok[i] and ok[i - 1] and ok[i + 1]):
            continue
        excursion = abs(v[i] - 0.5 * (v[i - 1] + v[i + 1]))
        if fail_threshold is not None and excursion > fail_threshold:
            flags[i] = FAIL
        elif excursion > threshold:
            flags[i] = SUSPECT

    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def rate_of_change(values, threshold, window):
    """Flag change sustained across a window faster than the creek can move."""
    v, flags = _base(values)
    n = len(v)
    ok = ~np.isnan(v)

    for i in range(n):
        if not ok[i] or i < window:
            continue
        prior = v[i - window]
        if np.isnan(prior):
            continue
        if abs(v[i] - prior) > threshold:
            flags[i] = SUSPECT

    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def sentinel(values, sentinels=(-9999,)):
    """
    Flag the sentinel readings. A -9999 in a CTD channel means the SDI-12
    address is wrong, which is a field fix rather than noise.
    """
    v, flags = _base(values)
    ok = ~np.isnan(v)
    hit = np.zeros(v.shape, dtype=bool)
    for value in sentinels:
        hit |= v == value

    flags[ok & hit] = FAIL
    flags[ok & ~hit] = GOOD
    return flags


def duplicate_feed(values, other, match_fraction, window=None):
    """
    Flag a series reporting the same numbers as another site.

    One matching reading between two creek sites is a coincidence. A quarter of
    them is a wiring problem.
    """
    a, flags = _base(values)
    b = _array(other)
    if len(b) != len(a):
        raise ValueError(f"duplicate_feed got {len(a)} and {len(b)} readings")

    both = ~np.isnan(a) & ~np.isnan(b)
    if window is None:
        matched = (a[both] == b[both]).mean() if both.any() else 0.0
        verdict = SUSPECT if matched > match_fraction else GOOD
        flags[~np.isnan(a)] = verdict
        return flags

    n = len(a)
    for i in range(n):
        if np.isnan(a[i]) or i + 1 < window:
            continue
        lo = i + 1 - window
        pair = both[lo : i + 1]
        if not pair.any():
            continue
        matched = (a[lo : i + 1][pair] == b[lo : i + 1][pair]).mean()
        flags[i] = SUSPECT if matched > match_fraction else GOOD

    flags[~np.isnan(a) & (flags == UNKNOWN)] = GOOD
    return flags


def staleness(timestamps, present, max_gap_seconds):
    """
    Flag steps whose newest reading is older than the allowed gap.

    present is the per step bool of whether a value arrived. Timestamps come in
    as epoch seconds so this stays free of pandas.
    """
    t = np.asarray(timestamps, dtype=float)
    present = np.asarray(present, dtype=bool)
    flags = np.full(t.shape, GOOD, dtype=np.uint8)

    last = np.nan
    for i in range(len(t)):
        if present[i]:
            last = t[i]
            flags[i] = GOOD
            continue
        if np.isnan(last):
            flags[i] = MISSING
        elif t[i] - last > max_gap_seconds:
            flags[i] = FAIL
        else:
            flags[i] = SUSPECT
    return flags


def worst(*flag_arrays):
    """
    Combine flags by severity, not by value. UNKNOWN never beats a real verdict.

    MISSING outranks everything, then FAIL, SUSPECT, GOOD, UNKNOWN last.
    """
    order = {UNKNOWN: 0, GOOD: 1, SUSPECT: 2, FAIL: 3, MISSING: 4}
    arrays = [np.asarray(f, dtype=np.uint8) for f in flag_arrays]
    if not arrays:
        raise ValueError("worst() needs at least one flag array")

    ranked = np.zeros(arrays[0].shape, dtype=np.uint8)
    for flags in arrays:
        if flags.shape != ranked.shape:
            raise ValueError(f"flag arrays disagree: {flags.shape} and {ranked.shape}")
        current = np.vectorize(order.__getitem__)(flags).astype(np.uint8)
        ranked = np.maximum(ranked, current)

    back = {v: k for k, v in order.items()}
    return np.vectorize(back.__getitem__)(ranked).astype(np.uint8)


__all__ = [
    "FAIL",
    "FLAGS",
    "FLAG_NAMES",
    "GOOD",
    "MISSING",
    "SUSPECT",
    "UNKNOWN",
    "attenuated_signal",
    "duplicate_feed",
    "flat_line",
    "gross_range",
    "rate_of_change",
    "sentinel",
    "spike",
    "staleness",
    "worst",
]

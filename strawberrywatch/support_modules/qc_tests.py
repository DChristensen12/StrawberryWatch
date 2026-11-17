"""
Quality tests. Each one takes numbers and returns flags.

This module doesn't import the config, the inventory or the storage layer, and
no function applies, corrects or cleans data. Tests know about arrays, and
callers know where the arrays came from.
"""

from __future__ import annotations

import time

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


def window_count(timestamps, seconds, default):
    """How many readings span this many seconds, at the series' own median spacing."""
    # The conversion ioos_qc's flat_line_test makes from its time thresholds,
    # rounded to the nearest reading
    # https://github.com/ioos/ioos_qc/blob/53246ba5c3d1c06ece3b74c5a5c9e9a0c1f655bc/ioos_qc/qartod.py
    t = np.asarray(timestamps, dtype=float)
    t = t[~np.isnan(t)]
    if t.size < 2:
        return default
    interval = np.median(np.diff(t))
    if not interval > 0:
        return default
    return max(1, int(round(seconds / interval)))


def rate_of_change(values, threshold, window, timestamps=None, span=None):
    """Flag change sustained across a window faster than the creek can move."""
    v, flags = _base(values)
    n = len(v)
    ok = ~np.isnan(v)
    t = None if timestamps is None or span is None else np.asarray(timestamps, dtype=float)

    for i in range(n):
        if not ok[i] or i < window:
            continue
        prior = v[i - window]
        if np.isnan(prior):
            continue
        # threshold is the change allowed over span seconds, held to the same rate
        # for readings further apart or closer, as ioos_qc's test does
        limit = threshold
        if t is not None and t[i] - t[i - window] != span:
            limit = threshold * (t[i] - t[i - window]) / span
        if abs(v[i] - prior) > limit:
            flags[i] = SUSPECT

    flags[ok & (flags == UNKNOWN)] = GOOD
    return flags


def sentinel(values, sentinels=(-9999,)):
    """
    Flag the sentinel readings. A -9999 in a CTD channel means the SDI-12
    address is wrong. That's a field fix, not noise.
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


# The three time tests are adapted from ioos_qc (Apache 2.0), on epoch seconds
# https://github.com/ioos/ioos_qc/blob/53246ba5c3d1c06ece3b74c5a5c9e9a0c1f655bc/ioos_qc/qartod.py


def _time_base(timestamps):
    """Start every stamp at GOOD, with the unreadable ones marked MISSING."""
    t = np.asarray(timestamps, dtype=float)
    flags = np.full(t.shape, GOOD, dtype=np.uint8)
    valid = ~np.isnan(t)
    flags[~valid] = MISSING
    return t, flags, valid


def impossible_date(timestamps, fail_span=None, now=None):
    """Flag stamps in the future, or outside fail_span when one is given."""
    t, flags, valid = _time_base(timestamps)
    now = time.time() if now is None else now

    # Check for time travelers
    flags[valid & (t > now)] = FAIL

    if fail_span is not None:
        low, high = fail_span
        flags[valid & ((t < low) | (t > high))] = FAIL
    return flags


def data_reception(timestamps, fail_span=6, from_time=None):
    """Flag stamps more than fail_span hours older than from_time, which defaults to now."""
    t, flags, valid = _time_base(timestamps)
    from_time = time.time() if from_time is None else from_time

    flags[valid & (from_time - t > fail_span * 3600)] = FAIL
    return flags


def time_gap(timestamps, fail_span=2):
    """Flag the reading that follows a gap longer than fail_span hours."""
    t, flags, valid = _time_base(timestamps)
    if t.size < 2:
        return flags

    diff = np.zeros(t.shape)
    diff[1:] = t[1:] - t[:-1]

    # the previous stamp has to be readable too
    pair_valid = valid & np.r_[False, valid[:-1]]
    flags[pair_valid & (diff > fail_span * 3600)] = FAIL
    flags[valid & np.isnan(diff)] = UNKNOWN

    # The first point takes the second's flag, unless the second is unreadable
    if not valid[1] and valid[0]:
        flags[0] = UNKNOWN
    else:
        flags[0] = flags[1]
    return flags


def worst(*flag_arrays):
    """
    Combine flags by severity, not by value. UNKNOWN loses to any real verdict.

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
    "data_reception",
    "duplicate_feed",
    "flat_line",
    "gross_range",
    "impossible_date",
    "rate_of_change",
    "sentinel",
    "spike",
    "staleness",
    "time_gap",
    "window_count",
    "worst",
]

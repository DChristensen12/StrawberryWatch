"""Data checks, the typical day, the histogram and the drift check."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strawberrywatch.workbench import text

# 8 identical readings in a row is 2 hours
STUCK_RUN = 8
# A day either side at 15 minutes
HAMPEL_WINDOW = 96
HAMPEL_K = 3.5
MAD_SCALE = 1.4826

# Evidently's default for large numerical columns: Wasserstein distance over the reference spread
DRIFT_RATIO = 0.1
DRIFT_MIN = 1000
REFERENCE_SIZE = 5000
DRIFT_DAYS = 30


def expected(start, end):
    """Readings a site should send between two UTC times, at 96 a day."""
    return int((end - start) / pd.Timedelta(minutes=15)) + 1


def missing(values, start, end):
    """(missing, expected) for one reading's 15 minute values over [start, end]."""
    n_expected = expected(start, end)
    present = values.dropna()
    present = present[(present.index >= start) & (present.index <= end)]
    return max(n_expected - len(present), 0), n_expected


def stuck(values):
    """Stretches of at least STUCK_RUN identical readings in a row."""
    clean = values.dropna().to_numpy()
    if clean.size == 0:
        return 0
    starts = np.concatenate([[True], clean[1:] != clean[:-1]])
    lengths = np.diff(np.concatenate([np.flatnonzero(starts), [clean.size]]))
    return int((lengths >= STUCK_RUN).sum())


def far(values):
    """True where a reading is a Hampel outlier against the readings around it."""
    clean = values.dropna()
    if clean.empty:
        return pd.Series(dtype=bool)
    rolling = clean.rolling(HAMPEL_WINDOW, center=True, min_periods=1)
    median = rolling.median()
    mad = (clean - median).abs().rolling(HAMPEL_WINDOW, center=True, min_periods=1).median()
    return (clean - median).abs() > HAMPEL_K * MAD_SCALE * mad


def typical_day(values):
    """(hours 0 to 23, median, 25th, 75th) by local hour of day."""
    clean = values.dropna()
    hours = clean.index.tz_convert(text.LOCAL).hour
    grouped = clean.groupby(hours)
    median = grouped.median()
    quartiles = grouped.quantile([0.25, 0.75]).unstack()
    index = range(24)
    median = median.reindex(index)
    quartiles = quartiles.reindex(index)
    return list(index), median.tolist(), quartiles[0.25].tolist(), quartiles[0.75].tolist()


def histogram(values, width):
    """(edges, counts) with edges at multiples of width covering the readings."""
    clean = values.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return [], []
    lo = math.floor(clean.min() / width) * width
    hi = math.ceil(clean.max() / width) * width
    if hi <= lo:
        hi = lo + width
    n_bins = int(round((hi - lo) / width))
    edges = lo + width * np.arange(n_bins + 1)
    counts, edges = np.histogram(clean, bins=edges)
    return edges.tolist(), counts.tolist()


def clean_nan(values):
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else v for v in values]


# Drift


def reference_sample(values):
    """Up to REFERENCE_SIZE readings, the same ones every time."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size <= REFERENCE_SIZE:
        return clean
    rng = np.random.default_rng(0)
    return np.sort(rng.choice(clean, REFERENCE_SIZE, replace=False))


def drifted(current, reference):
    """(drifted, ratio) for one site and reading, or None with too few readings on either side."""
    from scipy.stats import wasserstein_distance

    current = np.asarray(current, dtype=float)
    reference = np.asarray(reference, dtype=float)
    current = current[np.isfinite(current)]
    reference = reference[np.isfinite(reference)]
    if current.size < DRIFT_MIN or reference.size < DRIFT_MIN:
        return None
    spread = reference.std()
    if spread == 0:
        return None
    ratio = float(wasserstein_distance(current, reference) / spread)
    return ratio > DRIFT_RATIO, ratio

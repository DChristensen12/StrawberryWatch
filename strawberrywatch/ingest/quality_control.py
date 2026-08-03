"""
Reusable data QC, modeled on the HadISD test suite: flag first, decide later.
Every check here returns flags (a mask, or a list of flagged pairs/ranges) and
never mutates or drops rows. Both the corpus build and live ingest call these
and decide what to do with the flags themselves.

Expects a long-format frame: a 'location' column plus sensor columns
(conductivity, depth, temperature), indexed by datetime.
"""

import itertools

import numpy as np
import pandas as pd

_DEFAULT_SENSOR_COLS = ("conductivity", "depth", "temperature")


def _sensor_cols_present(df, sensor_cols):
    if sensor_cols is not None:
        return [c for c in sensor_cols if c in df.columns]
    return [c for c in _DEFAULT_SENSOR_COLS if c in df.columns]


def _group_contiguous_ranges(times, gap_multiplier=3):
    """
    Groups a sorted DatetimeIndex into (start, end) ranges, tolerating small
    gaps so one missing coincident timestamp doesn't split a run in two.
    Gap tolerance scales off the run's own median spacing since coincident
    timestamps between two sites can be irregular (footbridge in particular).
    """
    if len(times) == 0:
        return []
    if len(times) == 1:
        return [(times[0], times[0])]

    diffs = times.to_series().diff().dropna()
    median_gap = diffs.median()
    if pd.isna(median_gap) or median_gap <= pd.Timedelta(0):
        median_gap = pd.Timedelta(minutes=15)
    max_gap = median_gap * gap_multiplier

    ranges = []
    run_start = times[0]
    prev = times[0]
    for t in times[1:]:
        if t - prev > max_gap:
            ranges.append((run_start, prev))
            run_start = t
        prev = t
    ranges.append((run_start, prev))
    return ranges


def inter_site_duplicate_check(
    df, sensor_col="conductivity", min_points=1000, match_frac=0.25, return_all=False
):
    """
    For every pair of sites, over timestamps where both report a value for
    sensor_col, counts exact matches. Flags pairs with more than min_points
    coincident observations where more than match_frac of those are exact
    matches (one feed duplicating another).

    A pair whose duplication is confined to one contiguous stretch can still
    fall under match_frac once diluted against a much longer non-duplicated
    history, so pass return_all=True to get every pair with more than
    min_points coincident readings regardless of match_frac, each tagged with
    a "flagged" bool. Useful for a QC report that wants to show near-misses,
    not just the ones that cleared the bar.

    Returns a list of dicts: site_a, site_b, n_coincident, n_matched,
    match_frac, flagged, ranges (list of (start, end) tuples covering the
    matched stretches, contiguous with small-gap tolerance).
    """
    if "location" not in df.columns or sensor_col not in df.columns:
        return []

    sites = sorted(df["location"].dropna().unique())
    results = []

    for site_a, site_b in itertools.combinations(sites, 2):
        series_a = df.loc[df["location"] == site_a, sensor_col].dropna()
        series_b = df.loc[df["location"] == site_b, sensor_col].dropna()
        series_a = series_a[~series_a.index.duplicated(keep="first")]
        series_b = series_b[~series_b.index.duplicated(keep="first")]

        common_idx = series_a.index.intersection(series_b.index).sort_values()
        n_coincident = len(common_idx)
        if n_coincident <= min_points:
            continue

        is_match = series_a.loc[common_idx].values == series_b.loc[common_idx].values
        n_matched = int(is_match.sum())
        frac = n_matched / n_coincident
        is_flagged = frac > match_frac
        if not is_flagged and not return_all:
            continue

        matched_times = common_idx[is_match]
        ranges = _group_contiguous_ranges(matched_times)

        results.append(
            {
                "site_a": site_a,
                "site_b": site_b,
                "n_coincident": n_coincident,
                "n_matched": n_matched,
                "match_frac": frac,
                "flagged": is_flagged,
                "ranges": ranges,
            }
        )

    return results


def sentinel_check(df, sensor_cols=None, sentinels=(-9999, -999, -99)):
    """
    Flags rows where any sensor column equals a sentinel value, or where
    conductivity, depth, or temperature is negative (physically impossible
    for these sensors). Returns a boolean Series aligned to df.index, frame
    untouched.
    """
    cols = _sensor_cols_present(df, sensor_cols)
    mask = pd.Series(False, index=df.index)

    for col in cols:
        mask |= df[col].isin(sentinels)
        mask |= df[col] < 0

    return mask


def repeated_streak_check(df, sensor_cols=None, max_streak=96):
    """
    Flags runs where a sensor holds an identical value for more than
    max_streak consecutive timesteps, per (location, sensor). Stuck sensors
    read as perfectly valid data to a reconstruction model, so this needs to
    be caught explicitly. Returns a boolean Series aligned to df.index.
    """
    cols = _sensor_cols_present(df, sensor_cols)
    mask = pd.Series(False, index=df.index)
    if "location" not in df.columns:
        return mask

    for location, group in df.groupby("location"):
        group = group.sort_index()
        for col in cols:
            vals = group[col]
            valid = vals.notna()
            new_run = ~vals.eq(vals.shift())
            run_id = new_run.cumsum()
            run_size = vals.groupby(run_id).transform("size")
            flagged = valid & (run_size > max_streak)
            mask.loc[group.index[flagged]] = True

    return mask

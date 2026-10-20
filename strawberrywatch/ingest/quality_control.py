"""
Reusable data QC, modeled on the HadISD test suite: flag first, decide later.
Every check here returns flags (a mask, or a list of flagged pairs/ranges) and
leaves the rows alone. Both the corpus build and live ingest call these
and decide what to do with the flags themselves.

Expects a long-format frame: a 'location' column plus sensor columns
(conductivity, depth, temperature), indexed by datetime.
"""

import itertools

import pandas as pd

_DEFAULT_SENSOR_COLS = ("conductivity", "depth", "temperature")


def _sensor_cols_present(df, sensor_cols):
    if sensor_cols is not None:
        return [c for c in sensor_cols if c in df.columns]
    return [c for c in _DEFAULT_SENSOR_COLS if c in df.columns]


def _group_contiguous_ranges(times, gap_multiplier=3):
    """
    Group a sorted DatetimeIndex into (start, end) ranges, tolerating small
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
    For every pair of sites, counts exact matches in sensor_col over the
    timestamps both report. Flags pairs with more than min_points coincident
    readings where over match_frac match exactly (one feed duplicating another).

    A duplicated stretch can fall under match_frac once diluted by a long clean
    history. return_all=True returns every pair over min_points with a
    "flagged" bool, for reports that show near-misses.

    Returns dicts with site_a, site_b, n_coincident, n_matched, match_frac,
    flagged and ranges (the matched stretches as (start, end)).
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
    Flag rows where any sensor column equals a sentinel value, or where
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
    Flag runs where a sensor holds an identical value for more than
    max_streak consecutive timesteps, per (location, sensor). Stuck sensors
    read as perfectly valid data to a reconstruction model, and we have to
    catch them here. Returns a boolean Series aligned to df.index.
    """
    cols = _sensor_cols_present(df, sensor_cols)
    mask = pd.Series(False, index=df.index)
    if "location" not in df.columns:
        return mask

    for _location, group in df.groupby("location"):
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

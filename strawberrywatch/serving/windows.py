"""
Turning a plain readings frame into the windows the model was trained on.

This mirrors preprocessing/data_processor.py's prepare_sequences_normalized,
minus everything that only matters while training: no scaler fitting, no QC
valid column, no train/validation split. It is a separate function rather than
a call into that one because that module imports Config and the ingest package,
which drags our API client and our file paths along with it. A caller feeding us
their own data should not inherit any of that.

Keeping the two in step matters. If you change how training builds a window,
change it here too, or the model will be scored on inputs shaped differently
from the ones it learned on and nothing will raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class WindowError(ValueError):
    """The frame we were handed cannot be turned into windows for this model."""


def _check_frame(frame, feature_cols, sites):
    if not isinstance(frame, pd.DataFrame):
        raise WindowError(f"expected a DataFrame, got {type(frame).__name__}")
    if "location" not in frame.columns:
        raise WindowError(
            "frame has no 'location' column, so there is no way to tell which site "
            f"a row belongs to. Columns present: {sorted(frame.columns)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise WindowError(
            f"frame index is {type(frame.index).__name__}, expected a DatetimeIndex of "
            f"UTC timestamps. Set the reading time as the index before calling."
        )
    if frame.index.tz is None:
        raise WindowError("frame index is timezone naive, localize it to UTC first")

    missing = [c for c in feature_cols if c not in frame.columns]
    if missing:
        raise WindowError(
            f"frame is missing {missing}, which this model was trained on. Every "
            f"trained feature needs a column, even one that is entirely NaN, because "
            f"the model reads them by position. Expected: {feature_cols}"
        )

    present = set(frame["location"].unique())
    unknown = sorted(present - set(sites))
    if unknown:
        raise WindowError(
            f"frame carries sites the model does not know: {unknown}. Known sites are "
            f"{sites}. Rename or drop them before calling, since node identity is "
            f"positional and a stray site would shift the others."
        )


def add_time_features(frame):
    """
    Add the four clock features the model expects, derived from the index.

    hour and day of year go in as sine and cosine pairs so 23:00 and 00:00 sit
    next to each other instead of at opposite ends of a number line. These come
    from timestamps, never from a sensor, so they are never missing. Safe to call
    on a frame that already has them.
    """
    idx = frame.index
    hour = 2 * np.pi * (idx.hour + idx.minute / 60.0) / 24.0
    doy = 2 * np.pi * (idx.dayofyear - 1) / 365.0
    out = frame.copy()
    out["hour_sin"] = np.sin(hour)
    out["hour_cos"] = np.cos(hour)
    out["dayofyear_sin"] = np.sin(doy)
    out["dayofyear_cos"] = np.cos(doy)
    return out


def _impute_short_gaps(frame, feature_cols, limit_hours):
    """
    Interpolate over gaps shorter than limit_hours, per site.

    Longer gaps stay NaN on purpose. A four hour hole in conductivity is not
    something we get to invent a value for, and downstream the absence is what
    tells the mask that node was not reporting.
    """
    stamps = pd.DatetimeIndex(sorted(frame.index.unique()))
    if len(stamps) < 2:
        return frame

    step = pd.Series(stamps).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return frame

    limit_rows = max(1, int(limit_hours * (pd.Timedelta("1h") / step)))
    out = frame.copy()
    for site in out["location"].unique():
        rows = out["location"] == site
        # pandas interpolate falls over when the limit is longer than the series
        if rows.sum() <= 2 or limit_rows >= rows.sum():
            continue
        out.loc[rows, feature_cols] = out.loc[rows, feature_cols].interpolate(
            method="time", limit=limit_rows, limit_area="inside"
        )
    return out


def cadence(frame):
    """Median spacing between distinct timestamps, or None if we cannot tell."""
    stamps = pd.DatetimeIndex(sorted(frame.index.unique()))
    if len(stamps) < 2:
        return None
    step = pd.Series(stamps).diff().median()
    return None if pd.isna(step) or step <= pd.Timedelta(0) else step


def build_windows(
    frame,
    feature_cols,
    location_to_idx,
    normalization,
    window,
    imputation_limit_hours=3.0,
):
    """
    Slice a long frame into (sequences, targets, timestamps, node_mask).

    frame is one row per (timestamp, location) with a DatetimeIndex, a 'location'
    column, and a column per trained feature. Everything comes back normalized
    with the training statistics, because refitting on a live window moves the
    goalposts: our live conductivity averages about 100 uS/cm above what the
    model trained on, so a fresh scaler would put every input somewhere the model
    has never seen.

    node_mask is True where a site actually reported conductivity at that step.
    Missing cells go in as zero, which is the normalized mean, and the mask is
    what stops the model treating that zero as a real reading.

    Returns empty arrays when the frame has no run of consecutive usable steps
    long enough to fill a window. That is a normal outcome on a sparse feed, not
    an error, so the caller decides what to do about it.
    """
    sites = [s for s, _ in sorted(location_to_idx.items(), key=lambda kv: kv[1])]
    _check_frame(frame, feature_cols, sites)

    frame = _impute_short_gaps(frame.sort_index(), feature_cols, imputation_limit_hours)

    means = np.array([normalization[c][0] for c in feature_cols], dtype=float)
    scales = np.array([normalization[c][1] for c in feature_cols], dtype=float)
    # A feature that never varied in training has scale 0 and would divide to inf
    scales = np.where(scales == 0, 1.0, scales)

    stamps = pd.DatetimeIndex(sorted(frame.index.unique()))
    n_nodes, n_feat = len(sites), len(feature_cols)
    grid = np.full((len(stamps), n_nodes, n_feat), np.nan)

    position = {ts: i for i, ts in enumerate(stamps)}
    values = frame[feature_cols].to_numpy(dtype=float)
    rows = frame.index.map(position).to_numpy()
    cols = frame["location"].map(location_to_idx).to_numpy()
    grid[rows, cols, :] = (values - means) / scales

    cond = feature_cols.index("conductivity") if "conductivity" in feature_cols else None
    reported = ~np.isnan(grid[:, :, cond]) if cond is not None else ~np.isnan(grid).all(axis=2)

    filled = np.nan_to_num(grid, nan=0.0)
    # A step nobody reported at is not worth putting in a window
    usable = ~np.isnan(grid).all(axis=(1, 2))

    sequences, targets, target_stamps, masks = [], [], [], []
    for i in range(len(stamps) - window):
        if not usable[i : i + window + 1].all():
            continue
        sequences.append(filled[i : i + window])
        targets.append(filled[i + window])
        masks.append(reported[i : i + window])
        target_stamps.append(stamps[i + window])

    if not sequences:
        empty = np.empty((0, window, n_nodes, n_feat), dtype=np.float32)
        return (
            empty,
            np.empty((0, n_nodes, n_feat), dtype=np.float32),
            [],
            np.empty((0, window, n_nodes), dtype=bool),
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        target_stamps,
        np.asarray(masks, dtype=bool),
    )

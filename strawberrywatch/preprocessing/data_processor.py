import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from strawberrywatch.config import Config
from strawberrywatch.ingest.data_loader import _NON_FEATURE_COLUMNS


def _impute_short_gaps(df, feature_cols, limit_hours):
    """
    For each (location, feature) pair, linearly interpolate over gaps shorter
    than limit_hours. Longer gaps stay NaN, and the absence tracking
    downstream treats them as missing instead of made up.
    limit_area='inside' prevents extrapolation beyond first/last real obs.
    """
    timestamps = pd.DatetimeIndex(sorted(df.index.unique()))
    if len(timestamps) < 2:
        return df

    median_interval = pd.Series(timestamps).diff().median()
    if pd.isna(median_interval) or median_interval <= pd.Timedelta(0):
        print("couldn't determine sampling cadence, skipping imputation.")
        return df

    rows_per_hour = pd.Timedelta("1h") / median_interval
    limit_rows = max(1, int(limit_hours * rows_per_hour))

    # Safety: pandas' interpolation breaks if limit_rows >= per-site array length.
    min_site_rows = df.groupby("location").size().min()
    max_safe_limit = max(1, min_site_rows - 2)
    if limit_rows > max_safe_limit:
        print(
            f"capping interpolation limit to {max_safe_limit} rows "
            f"(site has only {min_site_rows} rows)."
        )
        limit_rows = max_safe_limit

    print(f"imputing gaps up to {limit_hours}h ({limit_rows} rows)...")
    df = df.copy()
    total_filled = 0

    for location in df["location"].unique():
        mask = df["location"] == location
        before = int(df.loc[mask, feature_cols].isna().sum().sum())

        site_row_count = mask.sum()
        if site_row_count <= 2:
            print(f"  [{location}] only {site_row_count} rows, skipping interpolation")
            continue

        df.loc[mask, feature_cols] = df.loc[mask, feature_cols].interpolate(
            method="time", limit=limit_rows, limit_area="inside"
        )

        filled = before - int(df.loc[mask, feature_cols].isna().sum().sum())
        if filled > 0:
            print(f"  [{location}] filled {filled} missing values")
        total_filled += filled

    print(f"filled {total_filled} values total.\n")
    return df


def prepare_sequences_normalized(
    df_featured,
    location_to_idx,
    sequence_length=Config.SEQUENCE_LENGTH,
    return_node_mask=False,
    scaler=None,
    scaler_feature_cols=None,
):
    """
    Normalizes df_featured and slices it into (sequences, targets).

    Missing data comes three ways: permanently absent (no valid value all
    window, like an offline site or a sensor a site didn't have), transiently
    absent (NaN at some steps), and present (real, or filled by short-gap
    imputation). All three are zero-filled, and a timestep's validity is only
    checked against present cells.

    A 'valid' column from the wide training corpus is a different signal: the
    number looks real but QC doesn't trust it. Those rows stay out of the scaler
    fit, and return_node_mask=True also returns node masks built from them.

    At inference pass the trained scaler and its feature order instead of
    fitting a fresh one (see the reuse branch). Training leaves both None.
    """
    exclude_cols = _NON_FEATURE_COLUMNS | {"location"}
    feature_cols = [
        col
        for col in df_featured.select_dtypes(include=[np.number]).columns.tolist()
        if col not in exclude_cols
    ]
    print(f"using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Fill short gaps first. A cell still NaN after this sits in a long gap,
    # and we don't guess its value.
    df_featured = _impute_short_gaps(df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS)

    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)

    # A cell is permanently absent if it has zero non-null values across the
    # loaded window. "Sensor not installed" and "sensor offline all window"
    # look the same in the data.
    print("checking for permanently absent channels...")
    permanent_absent = set()  # {(node_idx, feat_idx), ...}
    for location, node_idx in location_to_idx.items():
        loc_mask = df_featured["location"] == location
        for feat_idx, feat in enumerate(feature_cols):
            if not loc_mask.any() or df_featured.loc[loc_mask, feat].isna().all():
                permanent_absent.add((node_idx, feat_idx))
                print(f"  {location}/{feat}: no data, permanently absent")
    if not permanent_absent:
        print("  (none)")
    print()

    has_valid_col = "valid" in df_featured.columns

    reuse_scaler = scaler is not None
    if reuse_scaler:
        # Refitting on a live window moves the goalposts. Live conductivity
        # averages ~602 against the trained ~497, which puts each input ~105
        # uS/cm outside the space the model learned.
        #
        # Lined up by name, not position: air_temp_c and rain_mm swap depending
        # on which weather path filled them in.
        src_cols = list(scaler_feature_cols) if scaler_feature_cols else list(feature_cols)
        stats = {
            name: (float(scaler.mean_[i]), float(scaler.scale_[i]))
            for i, name in enumerate(src_cols)
        }
        unknown = [c for c in feature_cols if c not in stats]
        means = np.array([stats.get(c, (0.0, 1.0))[0] for c in feature_cols])
        scales = np.array([stats.get(c, (0.0, 1.0))[1] for c in feature_cols])
        print(f"scaler: reusing trained stats ({len(stats)} features), not refitting")
        if unknown:
            print(
                f"  no trained stats for {unknown}, left unscaled, "
                f"feature alignment drops them before the model sees them"
            )
    else:
        # Fit only on fully valid rows. Outages and QC-flagged readings (like
        # oxford during the university_house duplication) would skew the mean and
        # scale. Flagged rows aren't NaN, and the outage check alone misses them.
        all_data = []
        all_qc_valid = []
        for location in location_to_idx.keys():
            loc_df = df_featured[df_featured["location"] == location]
            all_data.append(loc_df[feature_cols].values)
            if has_valid_col:
                all_qc_valid.append(loc_df["valid"].values.astype(bool))
            else:
                all_qc_valid.append(np.ones(len(loc_df), dtype=bool))
        all_data = np.vstack(all_data)
        all_qc_valid = np.concatenate(all_qc_valid) if all_qc_valid else np.zeros(0, dtype=bool)

        valid_rows = ~np.isnan(all_data).any(axis=1)
        if has_valid_col:
            n_qc_excluded = int(valid_rows.sum() - (valid_rows & all_qc_valid).sum())
            print(f"scaler fit: excluding {n_qc_excluded:,} QC-invalid rows on top of NaN outages")
            valid_rows &= all_qc_valid
        scaler = StandardScaler()
        scaler.fit(all_data[valid_rows])

    df_normalized = df_featured.copy()
    for location in location_to_idx.keys():
        loc_mask = df_featured["location"] == location
        if not loc_mask.any():
            print(f"  [{location}] no rows in this window, skipping normalization")
            continue
        vals = df_featured.loc[loc_mask, feature_cols].values
        if reuse_scaler:
            df_normalized.loc[loc_mask, feature_cols] = (vals - means) / scales
        else:
            df_normalized.loc[loc_mask, feature_cols] = scaler.transform(vals)

    print("building 3D array...")
    timestamps_all = sorted(df_normalized.index.unique())
    data_3d = np.full((len(timestamps_all), num_nodes, num_features), np.nan)
    # QC verdicts only. Absence and "not flagged" are different things and this
    # array carries only the second. It can't be the mask by itself. Presence
    # gets ANDed in once nan_mask exists.
    qc_valid_3d = np.ones((len(timestamps_all), num_nodes), dtype=bool)

    # One scatter over the frame. A duplicate (timestamp, location) still
    # resolves to its last row, since the rows keep the frame's order.
    row_idx = pd.DatetimeIndex(timestamps_all).get_indexer(df_normalized.index)
    node_idx = df_normalized["location"].map(location_to_idx)
    if node_idx.isna().any():
        # An unknown location fails here, by name
        unknown = sorted(set(df_normalized["location"][node_idx.isna()]))
        raise KeyError(f"no node index for location(s) {unknown}")
    node_idx = node_idx.to_numpy(dtype=np.intp)
    data_3d[row_idx, node_idx, :] = df_normalized[feature_cols].to_numpy(dtype=float)
    if has_valid_col:
        qc_valid_3d[row_idx, node_idx] = df_normalized["valid"].to_numpy(dtype=bool)

    # transient_absent_mask[t, n, f] = True if (node n, feature f) is NaN at
    # timestep t and NOT permanently absent. Real outages longer than the
    # imputation limit but the sensor isn't gone for good.
    nan_mask = np.isnan(data_3d)
    permanent_mask = np.zeros((num_nodes, num_features), dtype=bool)
    for node_idx, feat_idx in permanent_absent:
        permanent_mask[node_idx, feat_idx] = True

    # A node counts only if it reported and QC didn't flag it. A live pull has
    # no QC column, and a missing node has no row to flag. Using QC alone once
    # let 11,709 made-up zeros (10.05% of target cells) reach the loss.
    if "conductivity" in feature_cols:
        cond_idx = feature_cols.index("conductivity")
        present_3d = ~nan_mask[:, :, cond_idx]
        valid_3d = present_3d & qc_valid_3d if has_valid_col else present_3d
        source = (
            "real conductivity presence AND the QC 'valid' column"
            if has_valid_col
            else "real conductivity presence (no 'valid' column, live data)"
        )
        print(f"node_mask source: {source}")
    else:
        valid_3d = qc_valid_3d
        print(
            "node_mask source: no conductivity feature to check presence against, QC column only"
            if has_valid_col
            else "node_mask source: nothing to derive presence from, falling back to all-True"
        )

    transient_absent_mask = nan_mask & ~permanent_mask[np.newaxis, :, :]
    n_transient = int(transient_absent_mask.sum())
    n_total_cells = int(nan_mask.size)
    print(
        f"transient absences: {n_transient:,} cells "
        f"({100 * n_transient / n_total_cells:.1f}% of all (t, node, feature) cells)\n"
    )

    # A timestep is valid when each NaN is a known absence. We ask this of the
    # array in one go
    unaccounted = nan_mask & ~permanent_mask[np.newaxis, :, :] & ~transient_absent_mask
    valid_mask = ~unaccounted.any(axis=(1, 2))
    print(f"valid timesteps: {valid_mask.sum():,} / {len(valid_mask):,}")

    # Optional safety: a timestep with EVERY node absent is not useful even
    # if it technically passes the check. Require at least one node to have
    # real (non-absent) data at each timestep.
    real_data_per_timestep = ~(nan_mask | permanent_mask[np.newaxis, :, :])
    has_any_real_node = real_data_per_timestep.any(axis=(1, 2))
    valid_mask &= has_any_real_node
    print(f"after node filter: {valid_mask.sum():,} / {len(valid_mask):,} valid timesteps")

    print(f"creating sequences (length {sequence_length})...")

    # Zero the absences once over the corpus, not once per window
    cleaned = data_3d.copy()
    cleaned[:, permanent_mask] = 0
    cleaned[transient_absent_mask] = 0

    # Which windows are fully valid, from one cumulative sum instead of a
    # rescan of sequence_length+1 flags per candidate start.
    n_starts = max(len(timestamps_all) - sequence_length, 0)
    run = np.concatenate([[0], np.cumsum(valid_mask, dtype=np.int64)])
    offsets = np.arange(n_starts)
    starts = np.flatnonzero(
        run[offsets + sequence_length + 1] - run[offsets] == sequence_length + 1
    )

    if starts.size:
        # Gathered straight into the returned array to avoid holding the corpus twice
        window_idx = starts[:, None] + np.arange(sequence_length)
        sequences = cleaned[window_idx]
        targets = cleaned[starts + sequence_length]
        sequence_timestamps = [timestamps_all[i + sequence_length] for i in starts]
        node_mask_sequences = valid_3d[window_idx] if return_node_mask else np.array([])
        target_node_mask = valid_3d[starts + sequence_length] if return_node_mask else np.array([])
    else:
        sequences = targets = np.array([])
        node_mask_sequences = target_node_mask = np.array([])
        sequence_timestamps = []

    print(f"done. {len(sequence_timestamps):,} sequences total")

    if return_node_mask:
        return (
            sequences,
            targets,
            sequence_timestamps,
            scaler,
            feature_cols,
            node_mask_sequences,
            target_node_mask,
        )
    return sequences, targets, sequence_timestamps, scaler, feature_cols

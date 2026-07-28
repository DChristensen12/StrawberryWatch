import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from config.config import Config
from src.ingest.data_loader import _NON_FEATURE_COLUMNS


def _impute_short_gaps(df, feature_cols, limit_hours):
    """
    For each (location, feature) pair, linearly interpolate over gaps shorter
    than limit_hours. Longer gaps are left as NaN so the absence-tracking
    logic downstream can treat them as missing rather than fabricated.
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

        df.loc[mask, feature_cols] = (
            df.loc[mask, feature_cols]
            .interpolate(method="time", limit=limit_rows, limit_area="inside")
        )

        filled = before - int(df.loc[mask, feature_cols].isna().sum().sum())
        if filled > 0:
            print(f"  [{location}] filled {filled} missing values")
        total_filled += filled

    print(f"filled {total_filled} values total.\n")
    return df


def prepare_sequences_normalized(
    df_featured, location_to_idx, sequence_length=Config.SEQUENCE_LENGTH,
    return_node_mask=False, scaler=None, scaler_feature_cols=None,
):
    """
    Normalizes and slices df_featured into (sequences, targets) for training.

    Missing data comes in three flavors:
      1. Permanently absent: an (node, feature) cell has no valid values across
         the whole loaded window. Covers fully-offline sensors and sites that
         never had a given feature (e.g. footbridge fully down, or oxford
         lacking a sensor installed elsewhere).
      2. Transiently absent: cell is NaN at some timesteps but valid at others.
         Includes sites that came online mid-window, calibration blackouts,
         and outages longer than the short-gap imputation limit.
      3. Present: a real value, or one filled in by short-gap imputation.

    All three are zero-filled before the model sees them. Timestep validity is
    checked only against cells that are actually present.

    If df_featured has a 'valid' column (from a wide training corpus, see
    scripts/build_training_corpus.py), that's a DIFFERENT signal from
    absence: a masked cell still has a real-looking number in it, we just
    don't trust it (QC-flagged sentinel/streak/duplicate). It's excluded from
    the scaler fit here. Pass return_node_mask=True to also get back
    per-sequence node_mask arrays derived from it, for the model's feature
    propagation / masked pooling and for loss masking. Without a 'valid'
    column, or with return_node_mask=False, this behaves exactly as before.

    scaler / scaler_feature_cols: pass the trained scaler (and the feature
    order it was fit on) to normalize with it instead of fitting a new one.
    Inference has to do this, see the reuse branch below for why. Leave both
    None to fit fresh, which is what training does.
    """
    exclude_cols = _NON_FEATURE_COLUMNS | {"location"}
    feature_cols = [
        col for col in df_featured.select_dtypes(include=[np.number]).columns.tolist()
        if col not in exclude_cols
    ]
    print(f"using {len(feature_cols)} features: {', '.join(feature_cols)}")

    # Fill short gaps before anything else. Anything still NaN after this is
    # a long enough gap that we won't pretend we know its value.
    df_featured = _impute_short_gaps(df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS)

    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)

    # A cell is permanently absent if it has zero non-null values across the
    # whole loaded window. "Sensor never installed" and "sensor offline all
    # window" are indistinguishable from data alone.
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
        # Inference has to normalize with the exact stats the model trained on.
        # Refitting on a live window silently moves the goalposts: live
        # conductivity averages ~602 vs the trained ~497, so every input lands
        # ~105 uS/cm outside the space the model learned, and Rule 2 ends up
        # comparing live-space levels against a trained-space cond_median.
        #
        # Lined up BY NAME rather than position, because the live feature order
        # isn't guaranteed to match the trained one (air_temp_c and rain_mm
        # swap depending on which weather path filled them in).
        src_cols = list(scaler_feature_cols) if scaler_feature_cols else list(feature_cols)
        stats = {name: (float(scaler.mean_[i]), float(scaler.scale_[i]))
                 for i, name in enumerate(src_cols)}
        unknown = [c for c in feature_cols if c not in stats]
        means = np.array([stats.get(c, (0.0, 1.0))[0] for c in feature_cols])
        scales = np.array([stats.get(c, (0.0, 1.0))[1] for c in feature_cols])
        print(f"scaler: reusing trained stats ({len(stats)} features), not refitting")
        if unknown:
            print(f"  no trained stats for {unknown}, left unscaled -- "
                  f"feature alignment drops them before the model sees them")
    else:
        # Fit only on fully-valid rows so long outages don't corrupt the scaler stats.
        # QC-invalid rows (masked oxford during the university_house duplication, a
        # stuck sensor's streak, etc.) get excluded here too -- they're not NaN, so
        # the outage check alone wouldn't catch them, but they're not real readings
        # either and would otherwise shift the mean and scale.
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
    # node_mask[t, n] = True where node n is trustworthy at timestep t. Starts
    # all-True, the QC loop below fills it in when a 'valid' column exists.
    # If there's no 'valid' column this gets replaced wholesale further down,
    # see the disambiguation right after nan_mask is built.
    valid_3d = np.ones((len(timestamps_all), num_nodes), dtype=bool)

    for t_idx, timestamp in enumerate(tqdm(timestamps_all, desc="Pivoting data")):
        t_data = df_normalized.loc[timestamp]
        if isinstance(t_data, pd.Series):
            t_data = t_data.to_frame().T
        for _, row in t_data.iterrows():
            node_idx = location_to_idx[row["location"]]
            data_3d[t_idx, node_idx, :] = row[feature_cols].values
            if has_valid_col:
                valid_3d[t_idx, node_idx] = bool(row["valid"])

    # transient_absent_mask[t, n, f] = True if (node n, feature f) is NaN at
    # timestep t and NOT permanently absent. Real outages longer than the
    # imputation limit but the sensor isn't gone for good.
    nan_mask = np.isnan(data_3d)
    permanent_mask = np.zeros((num_nodes, num_features), dtype=bool)
    for node_idx, feat_idx in permanent_absent:
        permanent_mask[node_idx, feat_idx] = True

    # node_mask disambiguation. The wide training corpus carries a real QC
    # 'valid' column (sentinel/streak/duplicate checks from
    # build_training_corpus.py) -- use it, unchanged from before. A live API
    # pull never has one, so valid_3d was defaulting to all-True for
    # everything, including a node with zero rows this window, which meant
    # masked pooling and feature propagation never actually engaged live, and
    # a fully offline node's zero-filled values got fed to its graph
    # neighbors as if real. Same notion as anomaly_detector.py's
    # _extract_real_mask (real conductivity presence = trustworthy), computed
    # locally here off nan_mask instead of re-deriving it from df_original,
    # since nan_mask already has everything this needs.
    if has_valid_col:
        print("node_mask source: QC 'valid' column (training-corpus data)")
    elif "conductivity" in feature_cols:
        cond_idx = feature_cols.index("conductivity")
        valid_3d = ~nan_mask[:, :, cond_idx]
        print("node_mask source: no 'valid' column (live data) -- derived from "
              "real conductivity presence instead")
    else:
        print("node_mask source: no 'valid' column and no conductivity feature, "
              "falling back to all-True")

    transient_absent_mask = nan_mask & ~permanent_mask[np.newaxis, :, :]
    n_transient = int(transient_absent_mask.sum())
    n_total_cells = int(nan_mask.size)
    print(
        f"transient absences: {n_transient:,} cells "
        f"({100 * n_transient / n_total_cells:.1f}% of all (t, node, feature) cells)\n"
    )

    # A timestep is valid if zeroing out all known absences leaves no NaN.
    # That means every NaN is accounted for, and at least one cell had real data.
    def is_valid_timestep(t_idx):
        t_data = data_3d[t_idx].copy()
        for node_idx, feat_idx in permanent_absent:
            t_data[node_idx, feat_idx] = 0
        t_data[transient_absent_mask[t_idx]] = 0
        return not np.isnan(t_data).any()

    valid_mask = np.array([is_valid_timestep(i) for i in range(len(timestamps_all))])
    print(f"valid timesteps: {valid_mask.sum():,} / {len(valid_mask):,}")

    # Optional safety: a timestep with EVERY node absent is not useful even
    # if it technically passes the check. Require at least one node to have
    # real (non-absent) data at each timestep.
    real_data_per_timestep = ~(
        nan_mask | permanent_mask[np.newaxis, :, :]
    )
    has_any_real_node = real_data_per_timestep.any(axis=(1, 2))
    valid_mask &= has_any_real_node
    print(f"after node filter: {valid_mask.sum():,} / {len(valid_mask):,} valid timesteps")

    print(f"creating sequences (length {sequence_length})...")
    sequences = []
    targets = []
    sequence_timestamps = []
    node_mask_sequences = []
    target_node_mask = []

    for i in tqdm(range(len(timestamps_all) - sequence_length), desc="Sliding window"):
        if valid_mask[i:i+sequence_length+1].all():
            seq = data_3d[i:i+sequence_length].copy()
            target = data_3d[i+sequence_length].copy()

            # Zero permanent absences across the whole window + target
            for node_idx, feat_idx in permanent_absent:
                seq[:, node_idx, feat_idx] = 0
                target[node_idx, feat_idx] = 0

            for step_offset in range(sequence_length):
                t_idx = i + step_offset
                seq[step_offset][transient_absent_mask[t_idx]] = 0
            target[transient_absent_mask[i + sequence_length]] = 0

            sequences.append(seq)
            targets.append(target)
            sequence_timestamps.append(timestamps_all[i + sequence_length])

            if return_node_mask:
                node_mask_sequences.append(valid_3d[i:i+sequence_length].copy())
                target_node_mask.append(valid_3d[i+sequence_length].copy())

    print(f"done. {len(sequences):,} sequences total")

    if return_node_mask:
        return (
            np.array(sequences), np.array(targets), sequence_timestamps, scaler, feature_cols,
            np.array(node_mask_sequences), np.array(target_node_mask),
        )
    return np.array(sequences), np.array(targets), sequence_timestamps, scaler, feature_cols

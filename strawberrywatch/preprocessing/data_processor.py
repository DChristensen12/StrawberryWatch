import numpy as np
from sklearn.preprocessing import StandardScaler

from strawberrywatch.config import Config
from strawberrywatch.ingest.data_loader import _NON_FEATURE_COLUMNS
from strawberrywatch.serving import windows


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

    # Short gaps filled ahead of the scaler fit. A cell still NaN sits in a long gap
    df_featured = windows.impute_short_gaps(
        df_featured, feature_cols, Config.IMPUTATION_LIMIT_HOURS
    )

    # Not installed and offline all window look the same, and both go in as zeros
    print("checking for permanently absent channels...")
    absent = 0
    for location in location_to_idx:
        loc_mask = df_featured["location"] == location
        for feat in feature_cols:
            if not loc_mask.any() or df_featured.loc[loc_mask, feat].isna().all():
                absent += 1
                print(f"  {location}/{feat}: no data, permanently absent")
    if not absent:
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

    if not reuse_scaler:
        means, scales = scaler.mean_, scaler.scale_
    normalization = {
        name: (float(mean), float(scale))
        for name, mean, scale in zip(feature_cols, means, scales, strict=True)
    }

    # The same window builder serving calls. The corpus is on its grid already,
    # and its gaps were filled above
    sequences, targets, sequence_timestamps, node_mask_sequences, target_node_mask = (
        windows.build_windows(
            df_featured,
            feature_cols,
            location_to_idx,
            normalization,
            sequence_length,
            imputation_limit_hours=None,
            grid=None,
            valid="valid" if has_valid_col else None,
            dtype=np.float64,
            target_mask=True,
        )
    )
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

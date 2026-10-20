from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from strawberrywatch.anomalies import anomaly_detector as _ad
from strawberrywatch.anomalies.anomaly_detector import _rain_multipliers
from strawberrywatch.config import Config
from strawberrywatch.models import model_calls
from tests import event_catalog

ROOT = Path(__file__).parent.parent
ANOMALY_DIR = ROOT / "data" / "anomalies"

# Raw sensor column names to the internal feature names the model trained on.
COLUMN_MAP = {
    "DateTimeUTC": "datetime",
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
}

# One CSV per site in each event folder, named the way the model names the site.
# footbridge is scnf010, renamed when the fixtures were built.

# A node counts as live once this many readings land on the shared grid. Raw
# row counts mislead, since footbridge's broken cadence gives lots of rows that
# barely align.
MIN_ALIGNED_ROWS = 30

# Imported from production, not copied. These tests grade the rule that runs
MIN_TIMESTEPS_TO_JUDGE = _ad.MIN_TIMESTEPS_TO_JUDGE
MIN_TIMESTEPS_OVER_THRESHOLD = _ad.MIN_TIMESTEPS_OVER_THRESHOLD


# (event_folder, target_site, label, group) from tests/events.yaml. "anomaly"
# must flag and "true_negative" mustn't. "relative_only" has no test yet.
EVENT_CATALOG = [e.as_tuple() for e in event_catalog.load()]


def _load_threshold(model_metadata, target_site):
    """Per-node threshold, median and IQR, since some sites just run hotter or colder."""
    node_thresholds = model_metadata.get("node_thresholds")
    error_median = model_metadata.get("error_median")
    error_iqr = model_metadata.get("error_iqr")
    if (
        not node_thresholds
        or not error_median
        or not error_iqr
        or target_site not in node_thresholds
    ):
        pytest.skip(
            f"No per-node threshold for '{target_site}' in model metadata. "
            "Retrain with 'python main.py --mode train' so node_thresholds, "
            "error_median, and error_iqr get saved."
        )
    return (
        float(node_thresholds[target_site]),
        float(error_median[target_site]),
        float(error_iqr[target_site]),
    )


def _add_time_features(index):
    """The four cyclical time features data_loader adds in production, as {column: array}."""
    idx = pd.DatetimeIndex(index)
    hour_angle = 2 * np.pi * (idx.hour + idx.minute / 60.0) / 24.0
    doy_angle = 2 * np.pi * (idx.dayofyear - 1) / 365.0
    return {
        "hour_sin": np.sin(hour_angle),
        "hour_cos": np.cos(hour_angle),
        "dayofyear_sin": np.sin(doy_angle),
        "dayofyear_cos": np.cos(doy_angle),
    }


def _load_event_grid(event_folder, feature_cols, location_to_idx):
    """
    Reads an event folder onto one shared 15-minute grid. Returns (data_3d,
    node_mask, time_index, live_sites, rain_series), where data_3d is
    (time, node, feature) with gaps as 0 and node_mask is True where a node has
    real conductivity. A site that barely lands on the grid, like footbridge,
    ends up masked as missing.
    """
    folder = ANOMALY_DIR / event_folder
    if not folder.exists():
        pytest.skip(f"Event folder {event_folder} not found.")

    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)
    cond_idx = feature_cols.index("conductivity")

    # Read each core site we have a node for, renamed to internal columns.
    per_site = {}
    for site in location_to_idx:
        csv = folder / f"{site}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv).rename(columns=COLUMN_MAP)
        if "datetime" not in df.columns or "conductivity" not in df.columns:
            continue
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        per_site[site] = df

    if not per_site:
        pytest.skip(f"{event_folder}: no readable core-site CSVs.")

    # The grid comes from sites whose median gap is 15 minutes. Footbridge isn't
    # one, but it can still align where it happens to land.
    grid_sources = []
    for df in per_site.values():
        if len(df) < 2:
            continue
        gap = pd.Series(df.index).diff().median()
        if gap == pd.Timedelta(minutes=15):
            grid_sources.append(df.index)
    if not grid_sources:
        pytest.skip(f"{event_folder}: no site on a regular 15-minute cadence.")

    grid = grid_sources[0]
    for other in grid_sources[1:]:
        grid = grid.union(other)
    grid = pd.DatetimeIndex(sorted(grid))

    # All NaN to start, then each site's aligned rows
    data_3d = np.full((len(grid), num_nodes, num_features), np.nan)
    live_sites = []

    for site, df in per_site.items():
        node_idx = location_to_idx[site]
        aligned = df.reindex(grid)  # exact join, non-matching rows become NaN
        real_cond = aligned["conductivity"].notna().sum()
        if real_cond < MIN_ALIGNED_ROWS:
            # Too few readings fall on the grid. Treat as absent.
            continue
        live_sites.append(site)
        # Real weather too. The model trained with these three channels, and
        # zero-filling them threw off every event, not just rainy ones.
        for feat in [
            "conductivity",
            "depth",
            "temperature",
            "air_temp_c",
            "rain_mm",
            "shortwave_radiation",
        ]:
            if feat in aligned.columns and feat in feature_cols:
                data_3d[:, node_idx, feature_cols.index(feat)] = aligned[feat].values

    if not live_sites:
        pytest.skip(f"{event_folder}: no core site cleared {MIN_ALIGNED_ROWS} aligned rows.")

    # Time features are global: identical across nodes and always present.
    time_feats = _add_time_features(grid)
    for fname, vals in time_feats.items():
        if fname in feature_cols:
            fi = feature_cols.index(fname)
            for node_idx in range(num_nodes):
                data_3d[:, node_idx, fi] = vals

    # node_mask: a node is real at a step if its conductivity is real there.
    node_mask = ~np.isnan(data_3d[:, :, cond_idx])

    # Rain is the same across the creek. Take it from the first live site that
    # has it
    rain_series = None
    rain_idx = feature_cols.index("rain_mm") if "rain_mm" in feature_cols else None
    if rain_idx is not None and live_sites:
        first_live_node = location_to_idx[live_sites[0]]
        rain_vals = data_3d[:, first_live_node, rain_idx]
        if not np.isnan(rain_vals).all():
            rain_series = pd.Series(np.nan_to_num(rain_vals, nan=0.0), index=grid)

    # 0 is the neutral z-score. The mask says what was real.
    data_3d = np.nan_to_num(data_3d, nan=0.0)

    return data_3d, node_mask, grid, live_sites, rain_series


def _normalize(data_3d, scaler, feature_cols, location_to_idx):
    """Apply the trained scaler per node, matching production normalization."""
    num_nodes = data_3d.shape[1]
    out = data_3d.copy()
    for node_idx in range(num_nodes):
        out[:, node_idx, :] = scaler.transform(data_3d[:, node_idx, :])
    return out


def _rain_adjusted_thresholds(base_threshold, timestamps, rain_series):
    """
    Threshold per scored step, raised during and after rain by production's own
    _rain_multipliers. The test can't drift from what runs. rain_series is
    per-15-minute rain on the grid, or None.
    """
    multipliers, _ = _rain_multipliers(
        timestamps,
        rain_series,
        Config.RAIN_WINDOW_HOURS,
        Config.RAIN_THRESHOLD_MULTIPLIER,
        Config.RAIN_AMOUNT_THRESHOLD,
        Config.POST_RAIN_DECAY_HOURS,
    )
    return base_threshold * multipliers


def _reconstruction_errors(
    event_folder, target_site, model, model_metadata, edge_index, use_mask=False
):
    """
    Runs an event through the model. Returns the target node's normalized
    conductivity errors, their timestamps and the rain. use_mask passes the
    node mask to models that take one.
    """
    feature_cols = model_metadata["feature_cols"]
    scaler = model_metadata["scaler"]
    location_to_idx = model_metadata["location_to_idx"]

    if "conductivity" not in feature_cols:
        pytest.skip("conductivity not in feature_cols.")
    cond_idx = feature_cols.index("conductivity")

    if target_site not in location_to_idx:
        pytest.skip(
            f"Target '{target_site}' is not a node in the trained model "
            f"(graph has {list(location_to_idx)}). Needs the expanded graph."
        )
    node_idx = location_to_idx[target_site]

    # Its own median and IQR. A site that runs hotter shouldn't look anomalous
    error_median = model_metadata["error_median"][target_site]
    error_iqr = model_metadata["error_iqr"][target_site]

    data_3d, node_mask, grid, live_sites, rain_series = _load_event_grid(
        event_folder, feature_cols, location_to_idx
    )

    # The target node must itself be live, or there's no score.
    if target_site not in live_sites:
        pytest.skip(
            f"{event_folder}: target '{target_site}' has too little data this "
            f"window (live sites: {live_sites})."
        )

    normalized = _normalize(data_3d, scaler, feature_cols, location_to_idx)

    seq_len = Config.SEQUENCE_LENGTH
    if len(normalized) <= seq_len:
        pytest.skip(
            f"{event_folder}: only {len(normalized)} aligned steps, need more "
            f"than {seq_len} for one sequence."
        )

    errors = []
    target_times = []
    target_real = []
    with torch.no_grad():
        for i in range(len(normalized) - seq_len):
            seq = normalized[i : i + seq_len]  # (seq_len, nodes, feat)
            target = normalized[i + seq_len]  # (nodes, feat)
            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(Config.DEVICE)

            mask_t = None
            if use_mask:
                mask_seq = node_mask[i : i + seq_len]  # (seq_len, nodes)
                mask_t = torch.BoolTensor(mask_seq).unsqueeze(0).to(Config.DEVICE)
            pred = model_calls.run_sequence_model(model, seq_t, edge_index, mask_t)

            err = torch.abs(
                pred[0, node_idx] - torch.FloatTensor(target[node_idx]).to(Config.DEVICE)
            )
            raw_err = err[cond_idx].item()
            # Same reason, a site that runs high shouldn't look anomalous
            errors.append((raw_err - error_median) / error_iqr)
            target_times.append(grid[i + seq_len])
            target_real.append(bool(node_mask[i + seq_len, node_idx]))

    errors = np.array(errors)
    target_times = pd.DatetimeIndex(target_times)
    real = np.array(target_real, dtype=bool)

    # Only steps where the target really reported, same as production's
    # _extract_real_mask. Scoring the zero-filled gaps once made most of the
    # June 2025 spill's "detection" an absent sensor.
    return errors[real], target_times[real], rain_series


def _curve_shape(errors, n_buckets=10):
    if len(errors) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    bucket = max(1, len(errors) // n_buckets)
    buckets = [errors[i : i + bucket].mean() for i in range(0, len(errors), bucket)]
    lo, hi = min(buckets), max(buckets)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((b - lo) / span * 7))] for b in buckets)


def _report(name, errors, thresholds):
    # Rain only raises the line. The lowest one is the base
    over = int((errors > thresholds).sum())
    base = float(np.min(thresholds))
    print(
        f"\n  [{name}] points={len(errors)} peak={errors.max():.3f} "
        f"mean={errors.mean():.3f} over_thresh={over}/{len(errors)} "
        f"base_thresh={base:.3f}"
    )
    print(f"    curve: {_curve_shape(errors)}")


def _is_flagged(errors, thresholds):
    """
    Production's Rule 1 verdict. The errors and thresholds are already
    normalized and rain-adjusted. _rule1_forecast_residual gets an identity
    median, IQR and multiplier and only does the counting.
    """
    return bool(
        _ad._rule1_forecast_residual(
            np.asarray(errors, dtype=float),
            0.0,
            1.0,
            np.asarray(thresholds, dtype=float),
            1.0,
        )["flagged"]
    )


ANOMALOUS_CASES = [(e, s, g) for e, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"]
TRUE_NEGATIVE_CASES = [(e, s, g) for e, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"]


@pytest.mark.parametrize(
    "event,target,group",
    ANOMALOUS_CASES,
    ids=[g + "/" + e for e, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"],
)
def test_anomaly_detected(event, target, group, model_bundle, edge_index):
    """Conductivity error at the target node must cross threshold for a known anomaly."""
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    base_threshold, node_median, node_iqr = _load_threshold(metadata, target)
    errors, target_times, rain_series = _reconstruction_errors(
        event,
        target,
        model,
        metadata,
        edge_index,
        use_mask=model_bundle["use_mask"],
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(f"{event}: only {len(errors)} points, too few to judge.")
    thresholds = _rain_adjusted_thresholds(base_threshold, target_times, rain_series)
    _report(f"{model_bundle['name']}/{group}/{target}", errors, thresholds)
    assert _is_flagged(errors, thresholds), (
        f"[{model_bundle['name']}] [{group}] {event}/{target}: expected an anomaly but only "
        f"{int((errors > thresholds).sum())} points crossed the threshold. "
        f"Peak {errors.max():.4f}, base {base_threshold:.4f} "
        f"(node median {node_median:.4f}, iqr {node_iqr:.4f})."
    )


@pytest.mark.parametrize(
    "event,target,group",
    TRUE_NEGATIVE_CASES,
    ids=[g + "/" + e for e, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"],
)
def test_true_negative_not_flagged(event, target, group, model_bundle, edge_index):
    """Conductivity error at the target node must stay below threshold for a normal event."""
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    base_threshold, node_median, node_iqr = _load_threshold(metadata, target)
    errors, target_times, rain_series = _reconstruction_errors(
        event,
        target,
        model,
        metadata,
        edge_index,
        use_mask=model_bundle["use_mask"],
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(f"{event}: only {len(errors)} points, too few to judge.")
    thresholds = _rain_adjusted_thresholds(base_threshold, target_times, rain_series)
    _report(f"{model_bundle['name']}/{group}/{target}", errors, thresholds)
    assert not _is_flagged(errors, thresholds), (
        f"[{model_bundle['name']}] [{group}] {event}/{target}: expected no anomaly but "
        f"{int((errors > thresholds).sum())} points crossed the threshold. "
        f"Peak {errors.max():.4f}, base {base_threshold:.4f} "
        f"(node median {node_median:.4f}, iqr {node_iqr:.4f})."
    )

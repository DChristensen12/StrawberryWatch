import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.config import Config

ANOMALY_DIR = ROOT / "data" / "anomalies"

# Raw sensor column names to the internal feature names the model trained on.
COLUMN_MAP = {
    "DateTimeUTC":          "datetime",
    "Meter_Hydros21_Cond":  "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp":  "temperature",
}

# The raw event folders carry one CSV per site, named by the model's site name.
# footbridge is the file the raw data calls north_fork_1 (scnf010), already
# renamed at fixture-build time, so the filename matches the model's location.

# A node counts as live for an event only if it has at least this many readings
# that actually land on the shared 15-minute grid. Raw row count is not enough:
# footbridge reports on a broken irregular cadence, so it can show a high raw
# count yet contribute almost nothing once aligned. Aligning first then counting
# is what correctly drops it.
MIN_ALIGNED_ROWS = 30

# A window needs at least this many scored error points to be judged.
MIN_TIMESTEPS_TO_JUDGE = 30

# How many points over threshold before we call it an event.
MIN_TIMESTEPS_OVER_THRESHOLD = 3


# (event_folder, target_site, label)
# label: "anomaly" must flag, "true_negative" must not flag, "relative_only"
# judged by comparison rather than absolute threshold. target_site is the node
# whose conductivity error we score, named in the model's vocabulary.
#
# Heads up: nothing consumes "relative_only". Only test_anomaly_detected and
# test_true_negative_not_flagged parametrize off this list, and they filter to
# their own labels, so the three relative_only rows below are documentation of
# events we know about, not coverage. Writing the comparison test would turn
# them back into real cases.
EVENT_CATALOG = [
    # June 2025 south-fork spill, propagating downstream. Confirmed real by SCMG.
    ("anomaly_2025_06_12_spill_sf",        "south_fork_1", "anomaly",       "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf",        "south_fork_2", "relative_only", "jun25_spill"),
    # September overnight conductivity spike across the south fork.
    ("anomaly_2025_09_10_overnight_sf",    "south_fork_1", "anomaly",       "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf",    "south_fork_2", "anomaly",       "sep25_overnight"),
    # November foam event. Footbridge is the labeled site but its sensor is
    # broken for this window, so the target falls back to north_fork_0 as the
    # nearest live north-fork node, judged relative since nf0 was the contrast.
    ("anomaly_2025_11_05_foam_nf1",        "north_fork_0", "relative_only", "nov25_foam"),
    # November storm, same footbridge problem, nf0 as the live north stand-in.
    ("anomaly_2025_11_13_rain_nf1",        "north_fork_0", "relative_only", "nov25_rain"),
    # April rainfall, confirmed heavy rain. True negatives under rain-adjusted
    # threshold. sf1 is absent this window so it is not listed.
    ("anomaly_2026_04_01_rainfall",        "north_fork_0", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall",        "south_fork_2", "true_negative", "apr26_rainfall"),
    # Botanical actuator malfunction Jan to Feb 2026. Confirmed real. Botanical
    # itself is off-graph, so we score the downstream live core instead.
    ("anomaly_2026_01_botanical_actuator", "oxford",       "anomaly",       "jan26_actuator"),
    # Fire-hydrant spill at north fork 0. Confirmed real. sf1 absent this window.
    ("anomaly_2026_03_20_hydrant_nf0",     "north_fork_0", "anomaly",       "mar26_hydrant"),
]


def _load_threshold(model_metadata, target_site):
    """
    Per-node threshold, median, and IQR for target_site. One global threshold
    doesn't work once you have sites that just run hotter or colder than the
    network average, each node needs to be judged against its own spread.
    """
    node_thresholds = model_metadata.get("node_thresholds")
    error_median = model_metadata.get("error_median")
    error_iqr = model_metadata.get("error_iqr")
    if not node_thresholds or not error_median or not error_iqr or target_site not in node_thresholds:
        pytest.skip(
            f"No per-node threshold for '{target_site}' in model metadata. "
            "Retrain with 'python main.py --mode train' so node_thresholds, "
            "error_median, and error_iqr get saved."
        )
    return float(node_thresholds[target_site]), float(error_median[target_site]), float(error_iqr[target_site])


def _add_time_features(index):
    """
    Build the four cyclical time features from a datetime index, matching what
    data_loader appends in production. Returns a dict of column name to array.
    These are never missing since every timestep has a clock.
    """
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
    Read every site CSV in an event folder, align them onto one shared 15-minute
    timestamp grid, and build the model's (time, node, feature) array plus a
    node_mask marking which nodes have a real conductivity reading at each step.

    The grid is the union of timestamps from sites that sit on the clean 15-min
    cadence. Each site is exact-joined onto it. A site whose readings do not land
    on the grid (footbridge's broken cadence) or that has too few aligned rows
    contributes mostly NaN and ends up masked missing, which is the honest result.

    Returns (data_3d, node_mask, time_index, live_sites, rain_series):
      data_3d:    (T, num_nodes, num_features) raw values, missing filled with 0
      node_mask:  (T, num_nodes) bool, True where that node has real conductivity
      time_index: the shared DatetimeIndex
      live_sites: list of site names that cleared MIN_ALIGNED_ROWS
      rain_series: raw per-15-min rain indexed by the grid, or None if unavailable
    """
    folder = ANOMALY_DIR / event_folder
    if not folder.exists():
        pytest.skip(f"Event folder {event_folder} not found.")

    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)
    cond_idx = feature_cols.index("conductivity")

    # Read each core site we have a node for, renamed to internal columns.
    per_site = {}
    for site, node_idx in location_to_idx.items():
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

    # Build the shared grid from sites on a regular 15-min cadence. A site is
    # "regular" if its median gap is 15 min. Footbridge fails this and does not
    # define the grid, though it can still align onto it where it happens to land.
    grid_sources = []
    for site, df in per_site.items():
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

    # Assemble the 3D array. Start all-NaN, fill each site's aligned rows.
    data_3d = np.full((len(grid), num_nodes, num_features), np.nan)
    live_sites = []

    for site, df in per_site.items():
        node_idx = location_to_idx[site]
        aligned = df.reindex(grid)  # exact join, non-matching rows become NaN
        real_cond = aligned["conductivity"].notna().sum()
        if real_cond < MIN_ALIGNED_ROWS:
            # Too few readings actually land on the grid. Treat as absent.
            continue
        live_sites.append(site)
        # Sensor channels plus the weather the fixtures now carry. Weather columns
        # already use the model's internal names (rain_mm, air_temp_c,
        # shortwave_radiation) so no rename is needed. Feeding real weather here
        # matters: the model trained with these three channels, so zero-filling
        # them made it predict from blanks and threw off every event, not just rain.
        for feat in ["conductivity", "depth", "temperature",
                     "air_temp_c", "rain_mm", "shortwave_radiation"]:
            if feat in aligned.columns and feat in feature_cols:
                data_3d[:, node_idx, feature_cols.index(feat)] = aligned[feat].values

    if not live_sites:
        pytest.skip(
            f"{event_folder}: no core site cleared {MIN_ALIGNED_ROWS} aligned rows."
        )

    # Time features are global, identical across nodes, never missing.
    time_feats = _add_time_features(grid)
    for fname, vals in time_feats.items():
        if fname in feature_cols:
            fi = feature_cols.index(fname)
            for node_idx in range(num_nodes):
                data_3d[:, node_idx, fi] = vals

    # node_mask: a node is real at a step if its conductivity is real there.
    node_mask = ~np.isnan(data_3d[:, :, cond_idx])

    # Rain series aligned to the grid, for the rain-adjusted threshold. Rain is
    # global (same weather over the whole small creek), so pull it from the first
    # live site that has it. Values are the per-15-min disaggregated rain the
    # fixtures carry, indexed by the grid timestamps.
    rain_series = None
    rain_idx = feature_cols.index("rain_mm") if "rain_mm" in feature_cols else None
    if rain_idx is not None and live_sites:
        first_live_node = location_to_idx[live_sites[0]]
        rain_vals = data_3d[:, first_live_node, rain_idx]
        if not np.isnan(rain_vals).all():
            rain_series = pd.Series(np.nan_to_num(rain_vals, nan=0.0), index=grid)

    # Fill the remaining NaNs with 0 (the post-z-score neutral value) so the
    # array is clean. The mask records what was real; the model decides what to
    # do with the zeros (ignore them if a mask is passed, average them if not).
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
    Build a per-timestep threshold that rises during and after rain. The bar
    scales with how much rain actually fell in the lookback window, not just
    whether any rain happened at all, so trace drizzle barely moves it and a
    real storm pushes it hard. After rain it tapers from wherever it peaked back to base over
    the decay window. Without a rain series it just returns the flat base
    everywhere, same as the no-rain path in production.

    timestamps is the DatetimeIndex for the scored errors (the target timesteps).
    rain_series is raw per-15-min rain indexed by the grid, or None.
    """
    n = len(timestamps)
    if rain_series is None or rain_series.empty:
        return np.full(n, base_threshold)

    window_h = Config.RAIN_WINDOW_HOURS
    mult = Config.RAIN_THRESHOLD_MULTIPLIER
    amount = Config.RAIN_AMOUNT_THRESHOLD
    saturation_mm = Config.RAIN_SATURATION_MM
    decay_h = Config.POST_RAIN_DECAY_HOURS

    multipliers = np.ones(n, dtype=float)
    ridx = rain_series.index

    def _scale(rain_sum):
        # amount is the floor where rain starts to count at all. saturation_mm
        # is the point where the full multiplier kicks in. anything between
        # scales linearly, so a trace amount just above the floor barely
        # moves the threshold, and a real storm climbs toward the full mult.
        frac = (rain_sum - amount) / (saturation_mm - amount)
        frac = min(max(frac, 0.0), 1.0)
        return 1.0 + (mult - 1.0) * frac

    for i, ts in enumerate(timestamps):
        lookback_start = ts - pd.Timedelta(hours=window_h)
        recent = rain_series[(ridx >= lookback_start) & (ridx <= ts)]
        recent_sum = recent.sum()
        if recent_sum > amount:
            multipliers[i] = _scale(recent_sum)
            continue
        decay_start = ts - pd.Timedelta(hours=window_h + decay_h)
        prior = rain_series[(ridx >= decay_start) & (ridx < lookback_start)]
        wet = prior.index[prior > amount]
        if len(wet) > 0:
            peak_mult = _scale(prior[prior > amount].max())
            hours_since = (ts - wet.max()).total_seconds() / 3600.0 - window_h
            frac = min(max(hours_since / decay_h, 0.0), 1.0)
            multipliers[i] = 1.0 + (peak_mult - 1.0) * (1.0 - frac)
    return base_threshold * multipliers


def _reconstruction_errors(event_folder, target_site, model, model_metadata,
                           edge_index, use_mask=False):
    """
    Run an event folder through the model. Returns the per-timestep robust-
    normalized conductivity errors for the target node, the timestamps those
    errors land on, and the rain series for the window (used to build the
    rain-adjusted threshold).

    use_mask=True passes the node_mask to the model, which turns on feature
    propagation and masked pooling. A model that does not accept the argument
    is called without it.
    """
    feature_cols    = model_metadata["feature_cols"]
    scaler          = model_metadata["scaler"]
    location_to_idx = model_metadata["location_to_idx"]
    num_nodes       = len(location_to_idx)
    num_features    = len(feature_cols)

    if "conductivity" not in feature_cols:
        pytest.skip("conductivity not in feature_cols.")
    cond_idx = feature_cols.index("conductivity")

    if target_site not in location_to_idx:
        pytest.skip(
            f"Target '{target_site}' is not a node in the trained model "
            f"(graph has {list(location_to_idx)}). Needs the expanded graph."
        )
    node_idx = location_to_idx[target_site]

    # this node's own error median/IQR, so a site that just runs hotter than
    # everyone else doesn't look permanently anomalous next to the rest
    error_median = model_metadata["error_median"][target_site]
    error_iqr = model_metadata["error_iqr"][target_site]

    data_3d, node_mask, grid, live_sites, rain_series = _load_event_grid(
        event_folder, feature_cols, location_to_idx
    )

    # The target node must itself be live, or there is nothing to score.
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
    with torch.no_grad():
        for i in range(len(normalized) - seq_len):
            seq = normalized[i : i + seq_len]          # (seq_len, nodes, feat)
            target = normalized[i + seq_len]            # (nodes, feat)
            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(Config.DEVICE)

            if use_mask:
                mask_seq = node_mask[i : i + seq_len]    # (seq_len, nodes)
                mask_t = torch.BoolTensor(mask_seq).unsqueeze(0).to(Config.DEVICE)
                pred = model(seq_t, edge_index, batch_size=1,
                             num_nodes=num_nodes, node_mask=mask_t)
            else:
                pred = model(seq_t, edge_index, batch_size=1, num_nodes=num_nodes)

            err = torch.abs(
                pred[0, node_idx] -
                torch.FloatTensor(target[node_idx]).to(Config.DEVICE)
            )
            raw_err = err[cond_idx].item()
            # robust-normalized so a site that naturally runs high doesn't
            # look permanently anomalous next to sites that run low
            errors.append((raw_err - error_median) / error_iqr)
            target_times.append(grid[i + seq_len])

    return np.array(errors), pd.DatetimeIndex(target_times), rain_series


def _curve_shape(errors, n_buckets=10):
    if len(errors) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    bucket = max(1, len(errors) // n_buckets)
    buckets = [errors[i:i + bucket].mean() for i in range(0, len(errors), bucket)]
    lo, hi = min(buckets), max(buckets)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((b - lo) / span * 7))] for b in buckets)


def _report(name, errors, thresholds):
    # thresholds is a per-timestep array (rain-adjusted). Report the base for
    # readability, which is the min since rain only raises it.
    over = int((errors > thresholds).sum())
    base = float(np.min(thresholds))
    print(
        f"\n  [{name}] points={len(errors)} peak={errors.max():.3f} "
        f"mean={errors.mean():.3f} over_thresh={over}/{len(errors)} "
        f"base_thresh={base:.3f}"
    )
    print(f"    curve: {_curve_shape(errors)}")


def _is_flagged(errors, thresholds):
    return int((errors > thresholds).sum()) >= MIN_TIMESTEPS_OVER_THRESHOLD


ANOMALOUS_CASES     = [(e, s, g) for e, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"]
TRUE_NEGATIVE_CASES = [(e, s, g) for e, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"]


@pytest.mark.parametrize(
    "event,target,group", ANOMALOUS_CASES,
    ids=[g + "/" + e for e, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"],
)
def test_anomaly_detected(event, target, group, model_bundle, edge_index):
    """Conductivity error at the target node must cross threshold for a known anomaly."""
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    base_threshold, node_median, node_iqr = _load_threshold(metadata, target)
    errors, target_times, rain_series = _reconstruction_errors(
        event, target, model, metadata, edge_index,
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
    "event,target,group", TRUE_NEGATIVE_CASES,
    ids=[g + "/" + e for e, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"],
)
def test_true_negative_not_flagged(event, target, group, model_bundle, edge_index):
    """Conductivity error at the target node must stay below threshold for a normal event."""
    model = model_bundle["model"]
    metadata = model_bundle["metadata"]
    base_threshold, node_median, node_iqr = _load_threshold(metadata, target)
    errors, target_times, rain_series = _reconstruction_errors(
        event, target, model, metadata, edge_index,
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
    
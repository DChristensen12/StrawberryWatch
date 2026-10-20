import numpy as np
import pandas as pd
import torch

from strawberrywatch.config import Config
from strawberrywatch.models import model_calls

# Scored steps over the line a rule needs in the window, not necessarily in a
# row. One noisy point can't flag on its own. Applies to both rules.
MIN_TIMESTEPS_OVER_THRESHOLD = 3

# Rule 2: robust deviations from a node's normal level before it counts as
# shifted. A starting guess, not calibrated yet.
LEVEL_SHIFT_K = 4.0

# Real readings a node needs in the window to be judged at all (see _extract_real_mask)
MIN_TIMESTEPS_TO_JUDGE = 30


def run_model_over_sequences(model, sequences, node_mask, edge_index, device):
    """
    One forward pass over a batch, keeping per-node predictions. node_mask only
    goes to models that take it. sequences is (T, seq_len, nodes, features),
    normalized, and predictions come back as (T, nodes, features).
    """
    model.eval()
    model = model.to(device)
    edge_index = edge_index.to(device)

    with torch.no_grad():
        seq_tensor = torch.FloatTensor(sequences).to(device)
        mask_tensor = None if node_mask is None else torch.BoolTensor(node_mask).to(device)
        predictions = model_calls.run_sequence_model(model, seq_tensor, edge_index, mask_tensor)

    return predictions.cpu().numpy()


def _extract_rain_series(df_original, locations):
    """
    Rain from the first site that has it, since it's the same across the
    creek. A rain_mm Series by time, or None.
    """
    has_rain = (
        df_original is not None
        and "rain_mm" in df_original.columns
        and not df_original["rain_mm"].isna().all()
        and "location" in df_original.columns
        and len(locations) > 0
        and (df_original["location"] == locations[0]).any()
    )
    if not has_rain:
        return None
    site_rows = df_original[df_original["location"] == locations[0]]
    return site_rows["rain_mm"]


def _rain_multipliers(
    timestamps,
    rain_series,
    rain_window_hours,
    rain_threshold_multiplier,
    rain_amount_threshold,
    post_rain_decay_hours,
):
    """
    Rule 1's rain adjustment. The full multiplier once lookback rain clears the
    floor, then a straight taper back to 1 over post_rain_decay_hours after it
    stops. Returns (multipliers, rain_flags), one per timestamp.
    """
    n = len(timestamps)
    multipliers = np.ones(n, dtype=float)
    rain_flags = np.zeros(n, dtype=bool)
    if rain_series is None or rain_series.empty:
        return multipliers, rain_flags

    for i, ts in enumerate(timestamps):
        lookback_start = ts - pd.Timedelta(hours=rain_window_hours)
        recent = rain_series[(rain_series.index >= lookback_start) & (rain_series.index <= ts)]
        if recent.sum() > rain_amount_threshold:
            rain_flags[i] = True
            multipliers[i] = rain_threshold_multiplier
            continue

        decay_start = ts - pd.Timedelta(hours=rain_window_hours + post_rain_decay_hours)
        prior = rain_series[
            (rain_series.index >= decay_start) & (rain_series.index < lookback_start)
        ]
        wet = prior.index[prior > rain_amount_threshold]
        if len(wet) > 0:
            most_recent_wet = wet.max()
            hours_since = (ts - most_recent_wet).total_seconds() / 3600.0 - rain_window_hours
            fraction = min(max(hours_since / post_rain_decay_hours, 0.0), 1.0)
            multipliers[i] = 1.0 + (rain_threshold_multiplier - 1.0) * (1.0 - fraction)
            if multipliers[i] > 1.0:
                rain_flags[i] = True

    return multipliers, rain_flags


def _extract_real_mask(df_original, timestamps, location_to_idx):
    """
    (T, nodes) bool, True where a node really reported conductivity, read from
    the raw data instead of node_mask.

    node_mask only means something when the data has a QC valid column. A live
    API pull doesn't, and there it's all True even for a node with no rows.
    Scoring that node's zero-filled target against a real prediction makes a
    huge fake error. This checks whether a sensor reported at all.
    """
    num_nodes = len(location_to_idx)
    real_mask = np.zeros((len(timestamps), num_nodes), dtype=bool)
    if (
        df_original is None
        or "location" not in df_original.columns
        or "conductivity" not in df_original.columns
    ):
        return (
            real_mask  # no raw data to check against, treat all of it as unreal instead of guessing
        )
    for site, node_idx in location_to_idx.items():
        site_rows = df_original[df_original["location"] == site]
        if site_rows.empty:
            continue
        real_mask[:, node_idx] = site_rows["conductivity"].reindex(timestamps).notna().to_numpy()
    return real_mask


def _rain_adjust_level_k(k, timestamps, rain_series):
    """
    Hook for Rule 2's rain handling, a no-op for now. Rain raises the forecast
    error and the actual level for different reasons, and borrowing Rule 1's
    multiplier isn't obviously right. The deployment battery will say
    whether Rule 2 needs anything here.
    """
    return np.full(len(timestamps), float(k))


def _longest_run(bool_array):
    """Longest stretch of consecutive True, for reporting how long an exceedance lasted."""
    longest = current = 0
    for v in bool_array:
        current = current + 1 if v else 0
        longest = max(longest, current)
    return longest


def _rule1_forecast_residual(
    errors_node,
    error_median,
    error_iqr,
    node_threshold,
    rain_multipliers,
    min_timesteps_over=MIN_TIMESTEPS_OVER_THRESHOLD,
):
    """
    Rule 1: robust-normalized forecast error against a per-node, rain-adjusted
    threshold. Catches onsets and ramps. A clean step goes quiet after a step or
    two because the model absorbs it, which is the gap Rule 2 covers.
    """
    normalized = (errors_node - error_median) / error_iqr
    thresholds = node_threshold * rain_multipliers
    over = normalized > thresholds
    n_over = int(over.sum())
    return {
        "flagged": n_over >= min_timesteps_over,
        "n_over_threshold": n_over,
        "longest_run": _longest_run(over),
        "peak_deviation": float(normalized.max()) if len(normalized) else float("nan"),
        # Which real steps went over, for whatever points at the readings
        "over": over,
        # Step-by-step scores, for serving to hand out with the verdict
        "score": normalized,
        "threshold": thresholds,
    }


def _rule2_level_shift(
    level_node, cond_median, cond_iqr, k, min_timesteps_over=MIN_TIMESTEPS_OVER_THRESHOLD
):
    """
    Rule 2: how far a node's conductivity level sits from its normal level, in
    robust deviations, sustained. It anchors on the level and not the model's
    error, and stays flagged for as long as the water does.
    """
    deviation = (level_node - cond_median) / cond_iqr
    over = np.abs(deviation) > k
    n_over = int(over.sum())
    return {
        "flagged": n_over >= min_timesteps_over,
        "n_over_threshold": n_over,
        "longest_run": _longest_run(over),
        "peak_deviation": float(np.abs(deviation).max()) if len(deviation) else float("nan"),
        "over": over,
        "score": np.abs(deviation),
        "threshold": np.broadcast_to(np.asarray(k, dtype=float), deviation.shape),
    }


def detect_anomalies(
    model,
    sequences,
    targets,
    timestamps,
    node_mask,
    edge_index,
    metadata,
    df_original=None,
    locations=None,
    device=None,
    operating_point=None,
    rain_params=None,
):
    """
    Production detection, per node, without averaging. Runs the model with
    node_mask and scores each node against both rules. A node is flagged if
    either fires, and the result says which, since an onset (Rule 1) and a
    lasting level (Rule 2) mean different things on the ground.

    metadata needs feature_cols, location_to_idx, error_median, error_iqr and
    node_thresholds for Rule 1, plus cond_median and cond_iqr for Rule 2. A node
    without a threshold is skipped, not guessed at.

    Pass df_original. It's how real readings are told apart from zero-filled
    gaps (see _extract_real_mask). Without it we skip every node, the safe way
    round.

    operating_point and rain_params override our fitted constants, for anyone
    running this on a different feed.

    Returns {site: result}. An unjudged node is {"judged": False, "reason": ...},
    a judged one has flagged, rules_fired, rule1, rule2 and n_real. rain_flags
    says, per step, whether rain was raising Rule 1's line.
    """
    if device is None:
        device = Config.DEVICE
    op = {
        "min_timesteps_over": MIN_TIMESTEPS_OVER_THRESHOLD,
        "min_timesteps_to_judge": MIN_TIMESTEPS_TO_JUDGE,
        "level_shift_k": LEVEL_SHIFT_K,
        **(operating_point or {}),
    }
    rain = {
        "rain_window_hours": Config.RAIN_WINDOW_HOURS,
        "rain_threshold_multiplier": Config.RAIN_THRESHOLD_MULTIPLIER,
        "rain_amount_threshold": Config.RAIN_AMOUNT_THRESHOLD,
        "post_rain_decay_hours": Config.POST_RAIN_DECAY_HOURS,
        **(rain_params or {}),
    }

    feature_cols = metadata["feature_cols"]
    location_to_idx = metadata["location_to_idx"]
    error_median = metadata.get("error_median", {})
    error_iqr = metadata.get("error_iqr", {})
    node_thresholds = metadata.get("node_thresholds", {})
    cond_median = metadata.get("cond_median", {})
    cond_iqr = metadata.get("cond_iqr", {})

    if "conductivity" not in feature_cols:
        raise ValueError(
            "detect_anomalies needs conductivity in feature_cols, nothing else is scored."
        )
    cond_idx = feature_cols.index("conductivity")

    idx_to_location = {idx: loc for loc, idx in location_to_idx.items()}
    # A DatetimeIndex, for timestamps[real]. The plain list it arrives as can't do that.
    timestamps = pd.DatetimeIndex(timestamps)

    predictions = run_model_over_sequences(model, sequences, node_mask, edge_index, device)
    errors = np.abs(predictions[:, :, cond_idx] - targets[:, :, cond_idx])  # (T, num_nodes)
    levels = targets[:, :, cond_idx]  # (T, num_nodes), actual normalized conductivity
    real_mask = _extract_real_mask(df_original, timestamps, location_to_idx)  # (T, num_nodes)

    rain_series = _extract_rain_series(df_original, locations) if df_original is not None else None
    rain_multipliers, rain_flags = _rain_multipliers(timestamps, rain_series, **rain)

    results = {}
    for node_idx in range(errors.shape[1]):
        site = idx_to_location.get(node_idx)
        if (
            site is None
            or site not in node_thresholds
            or site not in error_median
            or site not in error_iqr
        ):
            continue  # not calibrated for this node, no threshold to judge against

        real = real_mask[:, node_idx]
        n_real = int(real.sum())
        if n_real < op["min_timesteps_to_judge"]:
            results[site] = {"judged": False, "reason": "insufficient_real_data", "n_real": n_real}
            continue

        real_ts = timestamps[real]
        rain_mult_real = rain_multipliers[real]
        level_k_real = _rain_adjust_level_k(op["level_shift_k"], real_ts, rain_series)

        rule1 = _rule1_forecast_residual(
            errors[real, node_idx],
            error_median[site],
            error_iqr[site],
            node_thresholds[site],
            rain_mult_real,
            min_timesteps_over=op["min_timesteps_over"],
        )

        rule2 = {
            "flagged": False,
            "n_over_threshold": 0,
            "longest_run": 0,
            "peak_deviation": float("nan"),
            "over": np.zeros(n_real, dtype=bool),
            "score": np.full(n_real, np.nan),
            "threshold": np.full(n_real, np.nan),
        }
        if cond_median.get(site) is not None and cond_iqr.get(site) is not None:
            rule2 = _rule2_level_shift(
                levels[real, node_idx],
                cond_median[site],
                cond_iqr[site],
                level_k_real,
                min_timesteps_over=op["min_timesteps_over"],
            )

        rules_fired = []
        if rule1["flagged"]:
            rules_fired.append("forecast_residual")
        if rule2["flagged"]:
            rules_fired.append("level_shift")

        # Scattered back onto the full time axis, because the rules only saw
        # this node's real timesteps and a caller holds every timestep.
        over_timesteps = np.zeros(len(timestamps), dtype=bool)
        over_timesteps[real] = rule1["over"] | rule2["over"]

        # Same scatter per rule. An alert can then say when each one crossed,
        # not only when either did.
        for rule in (rule1, rule2):
            per_rule = np.zeros(len(timestamps), dtype=bool)
            per_rule[real] = rule["over"]
            rule["over_timesteps"] = per_rule

        results[site] = {
            "judged": True,
            "flagged": bool(rules_fired),
            "rules_fired": rules_fired,
            "rule1": rule1,
            "rule2": rule2,
            "n_real": n_real,
            "over_timesteps": over_timesteps,
            # Which timesteps the rules' per-step arrays line up with
            "real_timesteps": real,
        }

    return results, rain_flags

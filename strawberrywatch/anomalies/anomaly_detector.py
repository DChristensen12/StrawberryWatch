import inspect

import numpy as np
import pandas as pd
import torch

from strawberrywatch.config import Config

# Same sustained-crossing bar tests/test_anomaly_detection.py uses, ported so
# a lone noisy point doesn't flag but a real event does. Applies to both rules
# below, "sustained" means at least this many scored timesteps over threshold
# in the window, not necessarily consecutive -- that's what the test path
# means by it and what the deployment battery was validated against.
MIN_TIMESTEPS_OVER_THRESHOLD = 3

# Rule 2: how many robust deviations from a node's normal conductivity level
# before it counts as shifted. Starting guess, not yet calibrated -- Stage 3
# is what tells us if this is too loose or too tight.
LEVEL_SHIFT_K = 4.0

# Same "don't judge on scraps" bar tests/test_anomaly_detection.py uses. A node
# with fewer than this many REAL (non-fabricated) readings in the window
# doesn't get scored at all -- see _extract_real_mask below for why that's
# not optional.
MIN_TIMESTEPS_TO_JUDGE = 30


def run_model_over_sequences(model, sequences, node_mask, edge_index, device):
    """
    One forward pass over a batch of sequences, per-node predictions kept
    (never averaged). Passes node_mask only if the loaded model class accepts
    it, the same check trainer.py uses, so this works no matter which model the
    registry handed us.

    sequences: (T, seq_len, num_nodes, num_features) normalized
    node_mask: (T, seq_len, num_nodes) bool, or None
    Returns predictions, (T, num_nodes, num_features) normalized.
    """
    model.eval()
    model = model.to(device)
    edge_index = edge_index.to(device)

    supports_node_mask = "node_mask" in inspect.signature(model.forward).parameters

    with torch.no_grad():
        seq_tensor = torch.FloatTensor(sequences).to(device)
        kwargs = dict(batch_size=len(sequences), num_nodes=sequences.shape[2])
        if supports_node_mask and node_mask is not None:
            kwargs["node_mask"] = torch.BoolTensor(node_mask).to(device)
        predictions = model(seq_tensor, edge_index, **kwargs)

    return predictions.cpu().numpy()


def _extract_rain_series(df_original, locations):
    """
    Pulls rain from the first known location, because rain is one shared
    weather signal over the whole small creek, not a per-node quantity, so one
    site's rain_mm column speaks for all of them. Returns a rain_mm Series
    indexed by datetime, or None if rain data isn't available this run.
    """
    has_rain = (
        df_original is not None
        and 'rain_mm' in df_original.columns
        and not df_original['rain_mm'].isna().all()
        and 'location' in df_original.columns
        and len(locations) > 0
        and (df_original['location'] == locations[0]).any()
    )
    if not has_rain:
        return None
    site_rows = df_original[df_original['location'] == locations[0]]
    return site_rows['rain_mm']


def _rain_multipliers(timestamps, rain_series, rain_window_hours, rain_threshold_multiplier,
                       rain_amount_threshold, post_rain_decay_hours):
    """
    Rule 1's rain adjustment: full multiplier the moment lookback rain clears
    the floor, then a linear taper back to 1.0 over post_rain_decay_hours once
    it stops. Note this is a step, not the saturation-scaled ramp
    tests/test_anomaly_detection.py's _rain_adjusted_thresholds uses. The two
    rain formulas in this codebase disagree and production uses this one. Worth
    reconciling in the rain tuning pass.

    Returns (multipliers, rain_flags), both length len(timestamps).
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
        prior = rain_series[(rain_series.index >= decay_start) & (rain_series.index < lookback_start)]
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
    (T, num_nodes) bool, True where that node has an actual conductivity
    reading at that timestep, straight from the raw data, not from node_mask.

    node_mask/target_mask (from prepare_sequences_normalized) only carry
    signal when the source data has a QC 'valid' column, which only exists
    for the wide training-corpus format (scripts/build_training_corpus.py). A
    live API pull has no such column, so node_mask defaults to all-True
    everywhere, including for a node with zero rows this window -- it was
    never told "this node is absent," only ever told "nothing flagged this as
    QC-bad." Scoring a node whose target is the zero-fill placeholder
    (permanently-absent cells get zeroed before this function ever sees them)
    against the model's real prediction produces a huge, completely fake
    error. This is a second, independent check: not "is this QC-trustworthy"
    but "did a sensor actually report here at all."
    """
    num_nodes = len(location_to_idx)
    real_mask = np.zeros((len(timestamps), num_nodes), dtype=bool)
    if df_original is None or 'location' not in df_original.columns or 'conductivity' not in df_original.columns:
        return real_mask  # no raw data to check against, treat everything as unreal rather than guess
    for site, node_idx in location_to_idx.items():
        site_rows = df_original[df_original['location'] == site]
        if site_rows.empty:
            continue
        real_mask[:, node_idx] = site_rows['conductivity'].reindex(timestamps).notna().to_numpy()
    return real_mask


def _rain_adjust_level_k(k, timestamps, rain_series):
    """
    Hook for Rule 2's rain behavior. Currently a flat pass-through, k unchanged
    at every timestep, deliberately not wired to Rule 1's multiplier. Rain
    raises the model's forecast error (what Rule 1 tunes for) for a different
    reason than it raises the water's actual level (what Rule 2 looks at), so
    inheriting Rule 1's multiplier here isn't obviously correct, just
    convenient. Leaving this as an explicit no-op hook rather than guessing,
    the deployment battery is what tells us whether Rule 2 even needs it.
    """
    return np.full(len(timestamps), float(k))


def _longest_run(bool_array):
    """Longest stretch of consecutive True, for reporting how long an exceedance lasted."""
    longest = current = 0
    for v in bool_array:
        current = current + 1 if v else 0
        longest = max(longest, current)
    return longest


def _rule1_forecast_residual(errors_node, error_median, error_iqr, node_threshold, rain_multipliers):
    """
    Ported from tests/test_anomaly_detection.py: robust-normalized reconstruction
    error against a per-node, rain-adjusted threshold. Catches onsets and ramps,
    anything that keeps surprising the model step after step. Structurally blind
    to a clean step/spike past the first timestep or two, since the model
    absorbs a level shift into its own forecast almost immediately -- that gap
    is exactly what Rule 2 exists to cover.
    """
    normalized = (errors_node - error_median) / error_iqr
    thresholds = node_threshold * rain_multipliers
    over = normalized > thresholds
    n_over = int(over.sum())
    return {
        "flagged": n_over >= MIN_TIMESTEPS_OVER_THRESHOLD,
        "n_over_threshold": n_over,
        "longest_run": _longest_run(over),
        "peak_deviation": float(normalized.max()) if len(normalized) else float("nan"),
    }


def _rule2_level_shift(level_node, cond_median, cond_iqr, k):
    """
    Level-shift rule: how far the node's actual normalized conductivity sits
    from where it normally sits, in robust deviations, sustained over multiple
    timesteps. cond_median/cond_iqr anchor on the LEVEL itself, not the model's
    error, which is what lets this stay flagged for as long as the water
    actually stays elevated, even after Rule 1's residual has gone quiet.
    """
    deviation = (level_node - cond_median) / cond_iqr
    over = np.abs(deviation) > k
    n_over = int(over.sum())
    return {
        "flagged": n_over >= MIN_TIMESTEPS_OVER_THRESHOLD,
        "n_over_threshold": n_over,
        "longest_run": _longest_run(over),
        "peak_deviation": float(np.abs(deviation).max()) if len(deviation) else float("nan"),
    }


def detect_anomalies(model, sequences, targets, timestamps, node_mask, edge_index, metadata,
                      df_original=None, locations=None, device=Config.DEVICE):
    """
    Production detection, per node, never averaged. Runs the model with
    node_mask so feature propagation and masked pooling actually happen live,
    then scores each node against both rules independently. A node is
    anomalous if either rule fires; which one fired is in the result, since an
    onset (Rule 1) and a sustained abnormal level (Rule 2) mean different
    things operationally.

    metadata needs feature_cols, location_to_idx, error_median, error_iqr,
    node_thresholds (Rule 1), and cond_median, cond_iqr (Rule 2) -- all written
    by trainer.py/main.py's train path. A node missing from node_thresholds
    (never calibrated, e.g. not enough real data at train time) is skipped
    rather than guessed at.

    df_original is also what tells this function which timesteps are REAL for
    each node, as opposed to the zero-fill placeholder permanently-absent
    cells get. Without that check a fully offline node (no rows in df_original
    at all) still gets scored against its fabricated zero target, which reads
    as a huge fake error/level deviation and flags a node that reported
    nothing. See _extract_real_mask. Pass df_original or this will silently
    treat every node as having zero real data and skip all of them, not treat
    everything as real -- the unsafe direction is assuming real, not the other
    way around.

    Returns a dict keyed by site name. A node that never got real data this
    window: {"judged": False, "reason": "insufficient_real_data", ...}. A node
    that did: {"judged": True, "flagged": bool, "rules_fired": [...],
    "rule1": {...}, "rule2": {...}, "n_real": int}. Plus rain_flags, the
    per-timestep bool array of whether rain was pushing Rule 1's threshold up.
    """
    feature_cols = metadata["feature_cols"]
    location_to_idx = metadata["location_to_idx"]
    error_median = metadata.get("error_median", {})
    error_iqr = metadata.get("error_iqr", {})
    node_thresholds = metadata.get("node_thresholds", {})
    cond_median = metadata.get("cond_median", {})
    cond_iqr = metadata.get("cond_iqr", {})

    if "conductivity" not in feature_cols:
        raise ValueError("detect_anomalies needs conductivity in feature_cols, nothing else is scored.")
    cond_idx = feature_cols.index("conductivity")

    idx_to_location = {idx: loc for loc, idx in location_to_idx.items()}
    # needs to support boolean fancy-indexing below (timestamps[real]), a plain
    # list (what prepare_sequences_normalized actually returns) doesn't
    timestamps = pd.DatetimeIndex(timestamps)

    predictions = run_model_over_sequences(model, sequences, node_mask, edge_index, device)
    errors = np.abs(predictions[:, :, cond_idx] - targets[:, :, cond_idx])  # (T, num_nodes)
    levels = targets[:, :, cond_idx]  # (T, num_nodes), actual normalized conductivity
    real_mask = _extract_real_mask(df_original, timestamps, location_to_idx)  # (T, num_nodes)

    rain_series = _extract_rain_series(df_original, locations) if df_original is not None else None
    rain_multipliers, rain_flags = _rain_multipliers(
        timestamps, rain_series,
        Config.RAIN_WINDOW_HOURS, Config.RAIN_THRESHOLD_MULTIPLIER,
        Config.RAIN_AMOUNT_THRESHOLD, Config.POST_RAIN_DECAY_HOURS,
    )

    results = {}
    for node_idx in range(errors.shape[1]):
        site = idx_to_location.get(node_idx)
        if site is None or site not in node_thresholds or site not in error_median or site not in error_iqr:
            continue  # never calibrated for this node, nothing to judge against

        real = real_mask[:, node_idx]
        n_real = int(real.sum())
        if n_real < MIN_TIMESTEPS_TO_JUDGE:
            results[site] = {"judged": False, "reason": "insufficient_real_data", "n_real": n_real}
            continue

        real_ts = timestamps[real]
        rain_mult_real = rain_multipliers[real]
        level_k_real = _rain_adjust_level_k(LEVEL_SHIFT_K, real_ts, rain_series)

        rule1 = _rule1_forecast_residual(
            errors[real, node_idx], error_median[site], error_iqr[site], node_thresholds[site], rain_mult_real
        )

        rule2 = {"flagged": False, "n_over_threshold": 0, "longest_run": 0, "peak_deviation": float("nan")}
        if cond_median.get(site) is not None and cond_iqr.get(site) is not None:
            rule2 = _rule2_level_shift(levels[real, node_idx], cond_median[site], cond_iqr[site], level_k_real)

        rules_fired = []
        if rule1["flagged"]:
            rules_fired.append("forecast_residual")
        if rule2["flagged"]:
            rules_fired.append("level_shift")

        results[site] = {
            "judged": True,
            "flagged": bool(rules_fired),
            "rules_fired": rules_fired,
            "rule1": rule1,
            "rule2": rule2,
            "n_real": n_real,
        }

    return results, rain_flags

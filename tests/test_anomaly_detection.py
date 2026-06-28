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
NORMAL_DIR = ROOT / "data" / "normal"

STATION_MAP = {
    "nf0": "north_fork_0",
    "nf1": "north_fork_1",
    "sf0": "south_fork_0",
    "sf1": "south_fork_1",
    "sf2": "south_fork_2",
}

COLUMN_MAP = {
    "DateTimeUTC":                   "datetime",
    "Meter_Hydros21_Cond":           "conductivity",
    "Meter_Hydros21_Depth":          "depth",
    "Meter_Hydros21_Temp":           "temperature",
    "TE_TR_525USW_Precip_5minTotal": "rain_mm",
    "Sensirion_SHT40_Temperature":   "air_temp_c",
}

# (filename, station_suffix_or_None, label, event_group)
# label: "anomaly" = must flag, "true_negative" = must not flag,
#        "relative_only" = relative comparison only (seasonal drift or mislabels)
EVENT_CATALOG = [
    # June 2025 mystery spill, propagating downstream across all south-fork sensors.
    # Confirmed real by SCMG: multi-day deviation, no rain, supervisors unaware of cause.
    ("anomaly_2025_06_12_spill_sf0.csv",        "sf0", "anomaly",       "jun25_spill"),
    ("anomaly_2025_06_12_spill_sf1.csv",        "sf1", "anomaly",       "jun25_spill"),
    # sf2 signal is weaker here than sf1; doesn't reliably clear the absolute threshold.
    ("anomaly_2025_06_12_spill_sf2.csv",        "sf2", "relative_only", "jun25_spill"),
    # September 2025 overnight conductivity spike across south fork.
    ("anomaly_2025_09_10_overnight_sf0.csv",    "sf0", "anomaly",       "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf1.csv",    "sf1", "anomaly",       "sep25_overnight"),
    ("anomaly_2025_09_10_overnight_sf2.csv",    "sf2", "anomaly",       "sep25_overnight"),
    # November 2025 foam/chemical event, nf1 anomalous, nf0 contrast.
    ("anomaly_2025_11_05_foam_nf1.csv",         "nf1", "anomaly",       "nov25_foam"),
    ("anomaly_2025_11_05_foam_nf0.csv",         "nf0", "true_negative", "nov25_foam"),
    # November 2025 storm. nf1 shows an anomalous conductivity spike. nf0 was
    # originally labeled a clean rain response, but SCMG notes describe the
    # North Fork behaving anomalously during this storm, so nf0 is relative-only.
    ("anomaly_2025_11_13_rain_nf1.csv",         "nf1", "anomaly",       "nov25_rain"),
    ("anomaly_2025_11_13_rain_nf0.csv",         "nf0", "relative_only", "nov25_rain"),
    # April 2026 rainfall. Confirmed heavy rain 04/01-04/02 per SCMG. These are
    # true negatives judged with the rain-adjusted threshold, since a first-flush
    # conductivity bump during confirmed rain is what rain-aware thresholding
    # is designed to absorb.
    ("anomaly_2026_04_01_rainfall_nf0.csv",     "nf0", "relative_only", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_nf1.csv",     "nf1", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf0.csv",     "sf0", "true_negative", "apr26_rainfall"),
    ("anomaly_2026_04_01_rainfall_sf2.csv",     "sf2", "true_negative", "apr26_rainfall"),
    # Standalone true-negative rain event.
    ("anomaly_2025_05_12_rain_nf1.csv",         "nf1", "true_negative", "may25_rain_tn"),
    # Botanical actuator malfunction (Jan/Feb 2026). Confirmed real by SCMG.
    ("anomaly_2026_01_botanical_actuator.csv",  None,  "anomaly",       "jan26_actuator"),
    # Botanical normal baseline (January 2026). Model trained on April-May, so
    # this winter window reads mildly elevated from seasonal shift, not an event.
    # Relative-only: judged against the actuator window, not absolute. In data/normal/.
    ("normal_2026_01_botanical_baseline.csv",   None,  "relative_only", "jan26_actuator_baseline"),
    # Fire-hydrant spill at north fork 0 (03/20 Euclid Ave). Confirmed real by SCMG.
    ("anomaly_2026_03_20_hydrant_nf0.csv",      "nf0", "anomaly",       "mar26_hydrant"),
]

# A window needs at least this many error points to be judged. Shorter files
# are skipped rather than failed.
MIN_TIMESTEPS_TO_JUDGE = 30

# How many consecutive timesteps must exceed the threshold to call it an event.
MIN_TIMESTEPS_OVER_THRESHOLD = 3


def _load_threshold(model_metadata) -> float:
    threshold = model_metadata.get("threshold")
    if threshold is None:
        pytest.skip(
            "No trained threshold in model metadata. Retrain with the current "
            "pipeline (python main.py --mode train) so the threshold is saved."
        )
    return float(threshold)


def _rain_window_periods() -> int:
    """Number of 15-min timesteps in the rain look-back window (4 per hour)."""
    hours = getattr(Config, "RAIN_WINDOW_HOURS", 12)
    return int(hours * 4)


def _per_timestep_rain_threshold(errors, rain_series, base_threshold):
    """
    Builds a per-timestep threshold array matching production's rain-adjustment
    logic in anomaly_detector.detect_spills_with_rain_adjustment.

    Sums rain_mm over the preceding RAIN_WINDOW_HOURS for each error timestep.
    If the sum exceeds RAIN_AMOUNT_THRESHOLD, that timestep's threshold is
    base_threshold * RAIN_THRESHOLD_MULTIPLIER (same as production).

    If rain_series is None or all zero (most labeled files have no rain column),
    returns a flat array at base_threshold, matching production's no-rain behavior.
    """
    multiplier      = getattr(Config, "RAIN_THRESHOLD_MULTIPLIER", 2.0)
    amount_threshold = getattr(Config, "RAIN_AMOUNT_THRESHOLD", 0.1)
    look_back       = _rain_window_periods()
    seq_len         = Config.SEQUENCE_LENGTH

    thresholds = np.full(len(errors), base_threshold, dtype=float)

    if rain_series is None or np.all(np.nan_to_num(rain_series) <= 0):
        return thresholds

    rain = np.nan_to_num(np.asarray(rain_series, dtype=float))

    for i in range(len(errors)):
        center = i + seq_len  # the creek timestep this error scores
        lo = max(0, center - look_back)
        window = rain[lo:center + 1]
        if window.size and window.sum() > amount_threshold:
            thresholds[i] = base_threshold * multiplier

    return thresholds


def _reconstruction_errors(csv_path, station_suffix, model, model_metadata, edge_index,
                           return_rain=False):
    """
    Loads a labeled CSV, runs it through the model, and returns per-timestep
    conductivity reconstruction errors for the target node.

    Set return_rain=True to also get the raw rain_mm series for building a
    rain-adjusted threshold. Returns errors or (errors, rain_series).
    """
    feature_cols    = model_metadata["feature_cols"]
    scaler          = model_metadata["scaler"]
    location_to_idx = model_metadata["location_to_idx"]
    num_nodes       = len(location_to_idx)
    num_features    = len(feature_cols)

    if "conductivity" not in feature_cols:
        pytest.skip("conductivity not in feature_cols, cannot score the way production does")
    cond_idx = feature_cols.index("conductivity")

    station_name = STATION_MAP.get(station_suffix) if station_suffix else None
    if station_name is not None and station_name not in location_to_idx:
        pytest.skip(
            f"Station '{station_name}' (suffix '{station_suffix}') is not in the "
            f"trained model's graph. Add it to Config.LOCATIONS and retrain."
        )
    node_idx = location_to_idx.get(station_name, 0)

    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_MAP)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    feature_matrix = pd.DataFrame(0.0, index=df.index, columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            feature_matrix[col] = df[col]

    # Pull rain before normalisation so the threshold logic sees real mm
    rain_series = None
    if "rain_mm" in df.columns:
        rain_series = df["rain_mm"].fillna(0.0).values

    normalised = scaler.transform(feature_matrix.fillna(0.0).values)

    seq_len = Config.SEQUENCE_LENGTH
    if len(normalised) <= seq_len:
        pytest.skip(
            f"{csv_path.name} has only {len(normalised)} rows, need more than "
            f"{seq_len} for a single sequence. File too short to test."
        )

    errors = []
    with torch.no_grad():
        for i in range(len(normalised) - seq_len):
            seq    = np.zeros((seq_len, num_nodes, num_features))
            target = np.zeros((num_nodes, num_features))

            seq[:, node_idx, :] = normalised[i : i + seq_len]
            target[node_idx, :] = normalised[i + seq_len]

            seq_t = torch.FloatTensor(seq).unsqueeze(0).to(Config.DEVICE)
            pred  = model(seq_t, edge_index, batch_size=1, num_nodes=num_nodes)

            err = torch.abs(
                pred[0, node_idx] -
                torch.FloatTensor(target[node_idx]).to(Config.DEVICE)
            )
            errors.append(err[cond_idx].item())

    errors = np.array(errors)
    if return_rain:
        return errors, rain_series
    return errors


def _curve_shape(errors: np.ndarray, n_buckets: int = 10) -> str:
    if len(errors) == 0:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    bucket_size = max(1, len(errors) // n_buckets)
    buckets = [
        errors[i:i + bucket_size].mean()
        for i in range(0, len(errors), bucket_size)
    ]
    lo, hi = min(buckets), max(buckets)
    span = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((b - lo) / span * 7))] for b in buckets)


def _report(name, errors, thresholds):
    """Print per-case diagnostics. thresholds may be scalar or per-timestep array."""
    thr_arr = np.full(len(errors), thresholds) if np.isscalar(thresholds) else thresholds
    over = int((errors > thr_arr).sum())
    base = float(thr_arr.min())
    rained = (not np.isscalar(thresholds)) and (thr_arr.max() > thr_arr.min())
    rain_note = " (rain-adjusted in places)" if rained else ""
    print(
        f"\n  [{name}] points={len(errors)} peak={errors.max():.3f} "
        f"mean={errors.mean():.3f} over_thresh={over}/{len(errors)} "
        f"base_thresh={base:.3f}{rain_note}"
    )
    print(f"    curve: {_curve_shape(errors)}")


def _is_flagged(errors: np.ndarray, thresholds) -> bool:
    """
    True if at least MIN_TIMESTEPS_OVER_THRESHOLD points exceed their threshold.
    thresholds may be scalar or a per-timestep array (rain-adjusted).
    """
    thr_arr = np.full(len(errors), thresholds) if np.isscalar(thresholds) else thresholds
    return int((errors > thr_arr).sum()) >= MIN_TIMESTEPS_OVER_THRESHOLD


ANOMALOUS_CASES     = [(f, s, g) for f, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"]
TRUE_NEGATIVE_CASES = [(f, s, g) for f, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"]


@pytest.mark.parametrize(
    "filename,station,group", ANOMALOUS_CASES,
    ids=[g + "/" + f.replace(".csv", "")
         for f, s, lbl, g in EVENT_CATALOG if lbl == "anomaly"],
)
def test_anomaly_detected(filename, station, group, trained_model, model_metadata, edge_index):
    """Conductivity error must cross the (rain-adjusted) threshold for known anomalies."""
    base_threshold = _load_threshold(model_metadata)
    errors, rain = _reconstruction_errors(
        ANOMALY_DIR / filename, station, trained_model, model_metadata, edge_index,
        return_rain=True,
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(
            f"{filename}: only {len(errors)} error points "
            f"(need {MIN_TIMESTEPS_TO_JUDGE}). Too short to judge reliably."
        )
    thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
    _report(f"{group}/{filename}", errors, thresholds)
    assert _is_flagged(errors, thresholds), (
        f"[{group}] {filename}: expected an anomaly but only "
        f"{int((errors > thresholds).sum())} timesteps crossed threshold "
        f"(base {base_threshold:.4f}, need {MIN_TIMESTEPS_OVER_THRESHOLD}). "
        f"Peak error {errors.max():.4f}."
    )


@pytest.mark.parametrize(
    "filename,station,group", TRUE_NEGATIVE_CASES,
    ids=[g + "/" + f.replace(".csv", "")
         for f, s, lbl, g in EVENT_CATALOG if lbl == "true_negative"],
)
def test_true_negative_not_flagged(filename, station, group, trained_model, model_metadata, edge_index):
    """Conductivity error must stay below the (rain-adjusted) threshold for normal events."""
    base_threshold = _load_threshold(model_metadata)
    data_dir = NORMAL_DIR if filename.startswith("normal_") else ANOMALY_DIR
    errors, rain = _reconstruction_errors(
        data_dir / filename, station, trained_model, model_metadata, edge_index,
        return_rain=True,
    )
    if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
        pytest.skip(
            f"{filename}: only {len(errors)} error points "
            f"(need {MIN_TIMESTEPS_TO_JUDGE}). Too short to judge reliably."
        )
    thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
    _report(f"{group}/{filename}", errors, thresholds)
    assert not _is_flagged(errors, thresholds), (
        f"[{group}] {filename}: expected no anomaly but "
        f"{int((errors > thresholds).sum())} timesteps crossed threshold "
        f"(base {base_threshold:.4f}). Peak error {errors.max():.4f}. "
        f"Model may be over-flagging this case. If this file carries no rain "
        f"column the rain-adjustment is a no-op; check whether real rain data "
        f"would have suppressed it (see per-case output for whether rain engaged)."
    )


# Within-group relative tests. Threshold-independent, so robust to seasonal drift.

def test_foam_event_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """Nov 2025 foam: nf1 (anomalous) peak should exceed nf0 (contrast)."""
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_05_foam_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    print(f"\n  foam: nf1_peak={err_nf1.max():.3f} nf0_peak={err_nf0.max():.3f}")
    assert err_nf1.max() > err_nf0.max(), (
        f"Foam event: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_rain_storm_nf1_more_anomalous_than_nf0(trained_model, model_metadata, edge_index):
    """
    Nov 2025 storm: nf1 (anomalous conductivity spike) peak should exceed nf0.
    Both forks behaved oddly per SCMG notes, so this is relative, not a true-negative on nf0.
    """
    err_nf1 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf1.csv",
        "nf1", trained_model, model_metadata, edge_index
    )
    err_nf0 = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2025_11_13_rain_nf0.csv",
        "nf0", trained_model, model_metadata, edge_index
    )
    print(f"\n  storm: nf1_peak={err_nf1.max():.3f} nf0_peak={err_nf0.max():.3f}")
    assert err_nf1.max() > err_nf0.max(), (
        f"Nov storm: nf1 peak ({err_nf1.max():.4f}) should exceed "
        f"nf0 peak ({err_nf0.max():.4f})."
    )


def test_spill_propagation_all_south_fork_flagged(trained_model, model_metadata, edge_index):
    """
    June 2025 spill: all judged south-fork sensors should show sustained
    conductivity error over threshold. No rain during this event per SCMG notes.
    Sensors too short to judge are skipped.
    """
    base_threshold = _load_threshold(model_metadata)
    results = {}
    for suffix, fname in [("sf0", "anomaly_2025_06_12_spill_sf0.csv"),
                          ("sf1", "anomaly_2025_06_12_spill_sf1.csv"),
                          ("sf2", "anomaly_2025_06_12_spill_sf2.csv")]:
        errors, rain = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index,
            return_rain=True,
        )
        if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
            continue
        thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
        _report(f"jun25_spill/{suffix}", errors, thresholds)
        results[suffix] = (errors, thresholds)

    if not results:
        pytest.skip("All south-fork spill files too short to judge.")

    failures = [s for s, (e, t) in results.items() if not _is_flagged(e, t)]
    assert not failures, (
        f"Jun 2025 spill: expected all judged sf sensors flagged, "
        f"but {failures} did not cross threshold."
    )


def test_botanical_actuator_more_anomalous_than_baseline(trained_model, model_metadata, edge_index):
    """
    Jan 2026 actuator malfunction should score higher than the same sensor's
    normal baseline window. The relative comparison cancels seasonal offset.
    """
    err_actuator = _reconstruction_errors(
        ANOMALY_DIR / "anomaly_2026_01_botanical_actuator.csv",
        None, trained_model, model_metadata, edge_index
    )
    err_baseline = _reconstruction_errors(
        NORMAL_DIR / "normal_2026_01_botanical_baseline.csv",
        None, trained_model, model_metadata, edge_index
    )
    print(f"\n  actuator: anomaly_peak={err_actuator.max():.3f} baseline_peak={err_baseline.max():.3f}")
    assert err_actuator.max() > err_baseline.max(), (
        f"Botanical actuator: anomaly peak ({err_actuator.max():.4f}) should "
        f"exceed baseline peak ({err_baseline.max():.4f})."
    )


def test_april_rainfall_no_false_positives(trained_model, model_metadata, edge_index):
    """
    April 2026 rain event: no sensor should cross the rain-adjusted threshold.
    Checks that rain-aware thresholding absorbs first-flush conductivity bumps.

    If the CSVs have no rain_mm column the adjustment is a no-op; the per-case
    output shows whether rain engaged, so you can tell if a flag is a real gap.
    """
    base_threshold = _load_threshold(model_metadata)
    files = [
        ("nf0", "anomaly_2026_04_01_rainfall_nf0.csv"),
        ("nf1", "anomaly_2026_04_01_rainfall_nf1.csv"),
        ("sf0", "anomaly_2026_04_01_rainfall_sf0.csv"),
        ("sf2", "anomaly_2026_04_01_rainfall_sf2.csv"),
    ]
    false_positives = []
    judged = 0
    for suffix, fname in files:
        errors, rain = _reconstruction_errors(
            ANOMALY_DIR / fname, suffix, trained_model, model_metadata, edge_index,
            return_rain=True,
        )
        if len(errors) < MIN_TIMESTEPS_TO_JUDGE:
            continue
        judged += 1
        thresholds = _per_timestep_rain_threshold(errors, rain, base_threshold)
        _report(f"apr26_rainfall/{suffix}", errors, thresholds)
        if _is_flagged(errors, thresholds):
            false_positives.append(suffix)

    if judged == 0:
        pytest.skip("All April rainfall files too short to judge.")

    assert not false_positives, (
        f"Apr 2026 rainfall: sensors {false_positives} crossed the rain-adjusted "
        f"threshold but should be true negatives. If these files lack a rain_mm "
        f"column the adjustment was a no-op; see per-case output."
    )

import os
import sys
import pickle
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config.config import Config
from src.ingest.data_loader import load_and_preprocess_data
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.training.trainer import train_temporal_gnn
from src.anomalies.anomaly_detector import compute_anomaly_scores, detect_spills_with_rain_adjustment
from src.models.Dusk_Crayfish import DuskCrayfish
from src.anomalies.metrics import classify_event, format_classification


# How many hours before a flagged event to use as the baseline for classification.
# 24h captures a full daily cycle while staying recent enough to be meaningful.
_BASELINE_HOURS = 24


def _compute_system_scores(errors, feature_cols):
    """
    Conductivity-only scoring, averaged across nodes, same as main.py.
    Kept identical so this script detects exactly the same anomalies the production path would.
    """
    if "conductivity" in feature_cols:
        cond_idx = feature_cols.index("conductivity")
        return np.mean(errors[:, :, cond_idx], axis=1)
    return np.mean(errors, axis=(1, 2))


def _find_flagged_runs(spill_flags):
    """
    Groups per-timestep flags into contiguous runs. Returns a list of
    (start_idx, end_idx_inclusive) pairs, one per continuous event.
    """
    runs = []
    start = None
    for i, flagged in enumerate(spill_flags):
        if flagged and start is None:
            start = i
        elif not flagged and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(spill_flags) - 1))
    return runs


def _affected_site(errors, run, feature_cols, location_to_idx):
    """
    Finds the site that drove a flagged run: the node with the highest mean
    conductivity error across it. That's the location we slice for classification.
    """
    if "conductivity" not in feature_cols:
        return None
    cond_idx = feature_cols.index("conductivity")
    start, end = run
    per_node = errors[start:end + 1, :, cond_idx].mean(axis=0)
    node_idx = int(np.argmax(per_node))
    for loc, idx in location_to_idx.items():
        if idx == node_idx:
            return loc
    return None


def _slice_windows(df_original, location, event_start_ts, event_end_ts):
    """
    Pulls the event window and the pre-event baseline from the unnormalized dataframe.

    Baseline is _BASELINE_HOURS before the event start. Returns (baseline_df, event_df)
    for the given site, or (None, None) if there's not enough data.
    """
    if "location" in df_original.columns:
        site = df_original[df_original["location"] == location].copy()
    else:
        site = df_original.copy()

    if site.empty:
        return None, None

    # datetime may be the index or a column depending on how df_original arrived.
    if not isinstance(site.index, pd.DatetimeIndex):
        if "datetime" in site.columns:
            site = site.set_index("datetime")
        else:
            return None, None
    site = site.sort_index()

    baseline_start = event_start_ts - pd.Timedelta(hours=_BASELINE_HOURS)
    baseline_df = site.loc[(site.index >= baseline_start) & (site.index < event_start_ts)]
    event_df = site.loc[(site.index >= event_start_ts) & (site.index <= event_end_ts)]

    if baseline_df.empty or event_df.empty:
        return None, None
    return baseline_df, event_df


def main(mode, data_source):
    """Runs detection and classification in the requested mode."""
    model_dir = os.path.join(_REPO_ROOT, "models")
    model_path = os.path.join(model_dir, "dusk_crayfish_weights.pt")
    metadata_path = os.path.join(model_dir, "dusk_crayfish_metadata.pkl")
    os.makedirs(model_dir, exist_ok=True)

    print("--- Dusk Crayfish with Metrics (detection + spill classification) ---")
    print(f"Mode: {mode.upper()}   Source: {data_source}   Device: {Config.DEVICE}\n")

    days = 2 if mode == "inference" else 30
    df_featured, df_original, locations = load_and_preprocess_data(
        force_download=True, days=days, data_source=data_source
    )

    edge_index, _, location_to_idx = create_graph_topology()

    sequences, targets, timestamps, scaler, feature_cols = prepare_sequences_normalized(
        df_featured, location_to_idx, Config.SEQUENCE_LENGTH
    )
    if len(sequences) == 0:
        print("ERROR: no valid sequences built from this window.")
        sys.exit(1)

    have_weights = os.path.exists(model_path)
    have_metadata = os.path.exists(metadata_path)

    resolved_mode = mode
    if mode in ["update", "inference"] and not have_weights:
        print("No weights found, switching to fresh train.")
        resolved_mode = "train"

    loading_existing = resolved_mode in ["update", "inference"] and have_weights
    if loading_existing:
        if not have_metadata:
            print("ERROR: weights exist but metadata is missing. Retrain with --mode train.")
            sys.exit(1)
        with open(metadata_path, "rb") as f:
            saved = pickle.load(f)
        trained_feature_cols = saved.get("feature_cols")
        if feature_cols != trained_feature_cols:
            # Size to the trained feature set, same approach as main.py.
            print(f"[INFO] feature sets differ; using trained set {trained_feature_cols}")
        feature_cols = list(trained_feature_cols)

    num_node_features = len(feature_cols)
    model = DuskCrayfish(num_node_features=num_node_features).to(Config.DEVICE)
    if loading_existing:
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))

    mode = resolved_mode

    if mode == "inference":
        test_seq, test_tgt, test_timestamps = sequences, targets, timestamps
    else:
        split = int(len(sequences) * Config.TRAIN_SPLIT)
        train_seq, test_seq = sequences[:split], sequences[split:]
        train_tgt, test_tgt = targets[:split], targets[split:]
        test_timestamps = timestamps[split:]

    trained_threshold = None
    if mode in ["train", "update"]:
        print("Training...")
        _, _, trained_threshold = train_temporal_gnn(
            model, train_seq, train_tgt, edge_index,
            val_sequences=test_seq, val_targets=test_tgt, feature_cols=feature_cols,
        )
        torch.save(model.state_dict(), model_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({
                "scaler": scaler, "feature_cols": feature_cols,
                "location_to_idx": location_to_idx, "threshold": trained_threshold,
                "threshold_percentile": Config.THRESHOLD_PERCENTILE,
            }, f)

    model.eval()
    errors, _ = compute_anomaly_scores(model, test_seq, test_tgt, edge_index, Config.DEVICE)
    system_scores = _compute_system_scores(errors, feature_cols)

    base_threshold = trained_threshold
    if base_threshold is None and have_metadata:
        with open(metadata_path, "rb") as f:
            base_threshold = pickle.load(f).get("threshold")
    if base_threshold is None:
        base_threshold = float(np.percentile(system_scores, Config.THRESHOLD_PERCENTILE))

    spill_flags, rain_flags, adjusted_thresholds = detect_spills_with_rain_adjustment(
        system_anomaly_scores=system_scores, timestamps=test_timestamps,
        df_original=df_original, locations=locations, base_threshold=base_threshold,
    )

    spill_count = int(np.sum(spill_flags))
    print(f"\nDetection finished. Anomalies: {spill_count}, threshold {base_threshold:.4f}")

    if spill_count == 0:
        print("No anomalies to classify.")
        return

    runs = _find_flagged_runs(spill_flags)
    print(f"\n{len(runs)} contiguous event(s) to classify.\n")

    for n, run in enumerate(runs, 1):
        start, end = run
        event_start_ts = pd.to_datetime(test_timestamps[start], utc=True)
        event_end_ts = pd.to_datetime(test_timestamps[end], utc=True)
        site = _affected_site(errors, run, feature_cols, location_to_idx)

        print(f"=== Event {n}: {event_start_ts} to {event_end_ts} ===")
        print(f"Most affected site: {site}")

        if site is None:
            print("Could not determine affected site; skipping classification.\n")
            continue

        baseline_df, event_df = _slice_windows(df_original, site, event_start_ts, event_end_ts)
        if baseline_df is None:
            print(f"Not enough surrounding data to classify (need {_BASELINE_HOURS}h baseline).\n")
            continue

        result = classify_event(baseline_df, event_df)
        print(format_classification(result))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuskCrayfish detection plus spill classification")
    parser.add_argument("--mode", default="train", choices=["train", "update", "inference"])
    parser.add_argument("--data-source", default="api", choices=["api", "sql"])
    args = parser.parse_args()
    main(mode=args.mode, data_source=args.data_source)

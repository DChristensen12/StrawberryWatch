import torch
import numpy as np
import os
import sys
import pickle

from config.config import Config
from src.ingest.data_loader import load_and_preprocess_data
from src.utils.graph_utils import create_graph_topology
from src.preprocessing.data_processor import prepare_sequences_normalized
from src.training.trainer import train_temporal_gnn
from src.anomalies.anomaly_detector import compute_anomaly_scores, detect_spills_with_rain_adjustment


# Maps --model names to classes. Add new models here; nothing else in main.py changes.
from src.models.Dusk_Crayfish import DuskCrayfish
# from src.models.Flame_Skimmer import FlameSkimmer    # not yet implemented
# from src.models.Water_Strider import WaterStrider    # not yet implemented

_MODEL_REGISTRY = {
    "dusk_crayfish":  DuskCrayfish,
    # "flame_skimmer": FlameSkimmer,
    # "water_strider": WaterStrider,
}


def _align_to_trained_features(sequences, targets, current_feature_cols, trained_feature_cols):
    """
    Makes the freshly-built sequences match the feature set the model was trained on.

    Fix for size-mismatch crashes at load time. Training might have seen 6 features,
    but the current run only got 4 because a weather fetch failed. We build a
    zero-filled array sized to the trained feature set, copy over whatever matches,
    and drop anything extra. Zeros mean "no signal," same as everywhere else.

    Returns aligned_sequences, aligned_targets, and a short report of what moved.
    """
    if current_feature_cols == trained_feature_cols:
        return sequences, targets, "exact match (no alignment needed)"

    n_seq, seq_len, n_nodes, _ = sequences.shape
    n_trained = len(trained_feature_cols)

    aligned_seq = np.zeros((n_seq, seq_len, n_nodes, n_trained), dtype=sequences.dtype)
    aligned_tgt = np.zeros((targets.shape[0], n_nodes, n_trained), dtype=targets.dtype)

    current_idx = {name: i for i, name in enumerate(current_feature_cols)}

    filled = []
    zero_filled = []
    for trained_pos, feat in enumerate(trained_feature_cols):
        if feat in current_idx:
            src = current_idx[feat]
            aligned_seq[:, :, :, trained_pos] = sequences[:, :, :, src]
            aligned_tgt[:, :, trained_pos] = targets[:, :, src]
            filled.append(feat)
        else:
            # Slot stays zero: feature the model expects but this data doesn't have.
            zero_filled.append(feat)

    dropped = [f for f in current_feature_cols if f not in trained_feature_cols]

    report_parts = []
    if zero_filled:
        report_parts.append(f"zero-filled absent: {zero_filled}")
    if dropped:
        report_parts.append(f"dropped extra: {dropped}")
    report = "; ".join(report_parts) if report_parts else "reordered only"

    return aligned_seq, aligned_tgt, report


def _compute_system_scores(errors, feature_cols):
    """
    Collapses per-(sequence, node, feature) errors down to one score per timestep.

    Scores only on conductivity because that's where spills show up. The model
    predicts all features to stay grounded in physics, but rolling everything into
    the anomaly score just adds noise. Averaging conductivity across nodes means a
    network-wide deviation scores higher than a single-site blip.
    """
    if "conductivity" in feature_cols:
        cond_idx = feature_cols.index("conductivity")
        print(f"[INFO] Scoring on conductivity error (feature index {cond_idx}), averaged across nodes")
        return np.mean(errors[:, :, cond_idx], axis=1)
    else:
        # Fallback for backward compat or unusual feature sets.
        print("[WARN] 'conductivity' not in feature_cols — falling back to mean across all features.")
        return np.mean(errors, axis=(1, 2))


def main(mode="update", data_source="api", model_name="dusk_crayfish", visualize=False):
    """
    Runs the GNN anomaly detection pipeline in train, update, or inference mode.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, f"{model_name}_weights.pt")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("--- SCMG Anomaly Detection System ---")
    print(f"Execution Mode: {mode.upper()}")
    print(f"Model:          {model_name}")
    print(f"Device:         {Config.DEVICE}")

    if mode == "inference":
        # Pull only 2 days for speed during live monitoring
        df_featured, df_original, locations = load_and_preprocess_data(
            force_download=True, days=2, data_source=data_source
        )
    else:
        # Pull 30 days for training/updating
        df_featured, df_original, locations = load_and_preprocess_data(
            force_download=True, days=30, data_source=data_source
        )

    edge_index, _, location_to_idx = create_graph_topology()

    sequences, targets, timestamps, scaler, feature_cols = prepare_sequences_normalized(
        df_featured,
        location_to_idx,
        Config.SEQUENCE_LENGTH,
    )

    if len(sequences) == 0:
        print("ERROR: No valid sequences could be built from this data window.")
        print("       Try a longer time window (use --mode train for 30 days)")
        print("       or check sensor health.")
        sys.exit(1)

    if model_name not in _MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
        sys.exit(1)
    ModelClass = _MODEL_REGISTRY[model_name]

    # Resolve mode before sizing the model: inference falls back to train if no
    # weights exist, and that affects whether we size from metadata or fresh data.
    have_weights = os.path.exists(model_path)
    have_metadata = os.path.exists(metadata_path)

    resolved_mode = mode
    if mode in ["update", "inference"] and not have_weights:
        print(f"No weight file found at {model_path}. Switching to fresh train.")
        if mode == "inference":
            print("WARNING: training on a 2-day window will overfit. Consider")
            print("         running --mode train first to train on 30 days.")
        resolved_mode = "train"

    # Loading an existing model: size to the trained feature set from metadata,
    # then align current data to match. Training fresh: current data defines the set.
    loading_existing = resolved_mode in ["update", "inference"] and have_weights

    if loading_existing:
        if not have_metadata:
            print(f"ERROR: weights exist at {model_path} but metadata is missing at "
                  f"{metadata_path}. Cannot determine the trained feature set. "
                  f"Retrain with --mode train.")
            sys.exit(1)
        with open(metadata_path, "rb") as f:
            saved_metadata = pickle.load(f)
        trained_feature_cols = saved_metadata.get("feature_cols")
        if not trained_feature_cols:
            print("ERROR: metadata has no feature_cols. Retrain with --mode train.")
            sys.exit(1)

        sequences, targets, align_report = _align_to_trained_features(
            sequences, targets, feature_cols, trained_feature_cols
        )
        print(f"[INFO] Feature alignment: current {feature_cols} -> "
              f"trained {trained_feature_cols} ({align_report})")
        # From here on, the active feature set IS the trained one.
        feature_cols = list(trained_feature_cols)
    else:
        # Fresh train: the current data defines the feature set.
        trained_feature_cols = list(feature_cols)

    num_node_features = len(feature_cols)
    model = ModelClass(num_node_features=num_node_features).to(Config.DEVICE)

    if loading_existing:
        print(f"Loading weights from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
        )

    mode = resolved_mode

    if mode == "inference":
        train_seq, train_tgt = None, None
        test_seq, test_tgt = sequences, targets
        test_timestamps = timestamps
    else:
        split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]
        test_timestamps = timestamps[split_idx:]

    trained_threshold = None
    if mode in ["train", "update"]:
        print("Commencing model optimization...")
        _, _, trained_threshold = train_temporal_gnn(
            model,
            train_seq,
            train_tgt,
            edge_index,
            val_sequences=test_seq,
            val_targets=test_tgt,
            feature_cols=feature_cols,
        )
        torch.save(model.state_dict(), model_path)
        print(f"Optimization complete. Weights saved to {model_path}")

        with open(metadata_path, "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "feature_cols": feature_cols,
                "location_to_idx": location_to_idx,
                "threshold": trained_threshold,
                "threshold_percentile": Config.THRESHOLD_PERCENTILE,
            }, f)
        print(f"Model metadata saved to {metadata_path}")
    else:
        print("Skipping training phase. Entering evaluation mode.")

    model.eval()
    errors, predictions = compute_anomaly_scores(
        model,
        test_seq,
        test_tgt,
        edge_index,
        Config.DEVICE,
    )

    system_scores = _compute_system_scores(errors, feature_cols)

    # Priority: this run's computed threshold, then saved metadata, then P99 fallback.
    # The fallback is the old buggy behavior; retrain to get a stable one.
    base_threshold = trained_threshold
    if base_threshold is None and have_metadata:
        with open(metadata_path, "rb") as f:
            saved_metadata = pickle.load(f)
        base_threshold = saved_metadata.get("threshold")

    if base_threshold is None:
        print(
            "[WARN] No trained threshold available — falling back to "
            f"P{Config.THRESHOLD_PERCENTILE} of current run's scores. "
            "This is the OLD buggy behavior; retrain to get a stable threshold."
        )
        base_threshold = np.percentile(system_scores, Config.THRESHOLD_PERCENTILE)
    else:
        print(f"[INFO] Using trained threshold: {base_threshold:.6f}")

    spill_flags, rain_flags, adjusted_thresholds = detect_spills_with_rain_adjustment(
        system_anomaly_scores=system_scores,
        timestamps=test_timestamps,
        df_original=df_original,
        locations=locations,
        base_threshold=base_threshold,
    )

    spill_count = np.sum(spill_flags)
    print(f"Detection cycle finished. Anomalies identified: {spill_count}")

    if visualize:
        from src.utils.visualizations import plot_static_dashboard, plot_interactive_plotly
        plot_static_dashboard(
            timestamps=test_timestamps,
            system_anomaly_scores=system_scores,
            normalized_anomaly_scores=errors,
            adjusted_thresholds=adjusted_thresholds,
            base_threshold=base_threshold,
            spill_flags=spill_flags,
            rain_flags=rain_flags,
            df_original=df_original,
            locations=locations,
            threshold_percentile=Config.THRESHOLD_PERCENTILE,
        )
        if mode != "inference":
            plot_interactive_plotly(
                timestamps=test_timestamps,
                system_anomaly_scores=system_scores,
                adjusted_thresholds=adjusted_thresholds,
                base_threshold=base_threshold,
                spill_flags=spill_flags,
                rain_flags=rain_flags,
                rain_threshold_multiplier=Config.RAIN_THRESHOLD_MULTIPLIER,
                rain_window_hours=Config.RAIN_WINDOW_HOURS,
                threshold_percentile=Config.THRESHOLD_PERCENTILE,
            )

    if mode == "inference" and spill_count > 0:
        try:
            from src.utils.notifier import send_spill_alert
            # spill_flags is 1D per-timestep, so any flagged timestep means the whole
            # network alerted. Per-location attribution needs per-node scores, which
            # we don't compute here.
            send_spill_alert(int(spill_count), locations)
        except Exception as e:
            print(f"Alerting failed: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SCMG GNN Pipeline")
    parser.add_argument(
        "--mode", type=str, default="update",
        choices=["train", "update", "inference"],
    )
    parser.add_argument(
        "--data-source", type=str, default="api", choices=["api", "sql"],
        help="Where to pull data from: REST API (default) or SQL database",
    )
    parser.add_argument(
        "--model", type=str, default="dusk_crayfish",
        choices=list(_MODEL_REGISTRY.keys()),
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate static and interactive plots after detection",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        data_source=args.data_source,
        model_name=args.model,
        visualize=args.visualize,
    )

import torch
import numpy as np
import os
import sys
import pickle
import inspect
from strawberrywatch import paths
from strawberrywatch.config import Config
from strawberrywatch.ingest.data_loader import load_and_preprocess_data
from strawberrywatch.utils.graph_utils import create_graph_topology
from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized
from strawberrywatch.training.trainer import train_temporal_gnn
from strawberrywatch.anomalies.anomaly_detector import detect_anomalies
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

# Maps --model names to classes. Add new models here; nothing else in main.py changes.
_MODEL_REGISTRY = {
    "dusk_crayfish": DuskCrayfish,
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


def main(
    mode="update", data_source="api", model_name="dusk_crayfish", visualize=False, data_file=None
):
    """
    Runs the GNN anomaly detection pipeline in train, update, or inference mode.
    """
    model_dir = paths.checkpoints_dir()
    model_path = os.path.join(model_dir, f"{model_name}_weights.pt")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")

    file_path = data_file or Config.data_file()

    print("SCMG Anomaly Detection System")
    print(f"Execution Mode: {mode.upper()}")
    print(f"Model:          {model_name}")
    print(f"Device:         {Config.DEVICE}")
    print(f"Data file:      {file_path}")

    if mode == "inference":
        # Pull only 2 days for speed during live monitoring
        df_featured, df_original, locations = load_and_preprocess_data(
            file_path=file_path, force_download=True, days=2, data_source=data_source
        )
    else:
        # Pull 30 days for training/updating
        df_featured, df_original, locations = load_and_preprocess_data(
            file_path=file_path, force_download=True, days=30, data_source=data_source
        )

    edge_index, _, location_to_idx = create_graph_topology()

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

    # Metadata has to be read BEFORE sequences get built, because the scaler in
    # it is what normalization needs to use. Fitting a fresh one on the live
    # window instead puts the model's inputs in a different space than it
    # trained on.
    saved_metadata = None
    trained_feature_cols = None
    if loading_existing:
        if not have_metadata:
            print(
                f"ERROR: weights exist at {model_path} but metadata is missing at "
                f"{metadata_path}. Cannot determine the trained feature set. "
                f"Retrain with --mode train."
            )
            sys.exit(1)
        with open(metadata_path, "rb") as f:
            saved_metadata = pickle.load(f)
        trained_feature_cols = saved_metadata.get("feature_cols")
        if not trained_feature_cols:
            print("ERROR: metadata has no feature_cols. Retrain with --mode train.")
            sys.exit(1)

    sequences, targets, timestamps, scaler, feature_cols, node_mask_seq, target_mask = (
        prepare_sequences_normalized(
            df_featured,
            location_to_idx,
            Config.SEQUENCE_LENGTH,
            return_node_mask=True,
            scaler=saved_metadata["scaler"] if loading_existing else None,
            scaler_feature_cols=trained_feature_cols if loading_existing else None,
        )
    )

    if len(sequences) == 0:
        print("ERROR: No valid sequences could be built from this data window.")
        print("       Try a longer time window (use --mode train for 30 days)")
        print("       or check sensor health.")
        sys.exit(1)

    if loading_existing:
        sequences, targets, align_report = _align_to_trained_features(
            sequences, targets, feature_cols, trained_feature_cols
        )
        print(
            f"feature alignment: current {feature_cols} -> "
            f"trained {trained_feature_cols} ({align_report})"
        )
        # From here on, the active feature set IS the trained one.
        feature_cols = list(trained_feature_cols)
    else:
        # Fresh train: the current data defines the feature set.
        trained_feature_cols = list(feature_cols)

    num_node_features = len(feature_cols)

    # Only pass num_nodes to models that ask for it in their __init__ signature,
    # so a model that sizes itself purely from the feature count still loads.
    model_kwargs = {"num_node_features": num_node_features}
    if "num_nodes" in inspect.signature(ModelClass.__init__).parameters:
        model_kwargs["num_nodes"] = len(location_to_idx)
    model = ModelClass(**model_kwargs).to(Config.DEVICE)

    if loading_existing:
        print(f"Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))

    mode = resolved_mode

    train_node_mask = val_node_mask = None
    train_target_mask = val_target_mask = None

    if mode == "inference":
        train_seq, train_tgt = None, None
        test_seq, test_tgt = sequences, targets
        test_timestamps = timestamps
        # inference doesn't split, so the whole mask applies to the whole test set.
        # this was missing before the rewrite, which is why node_mask never made
        # it into a live forward pass -- there was nothing here to pass.
        test_node_mask = node_mask_seq
    else:
        split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
        train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]
        train_tgt, test_tgt = targets[:split_idx], targets[split_idx:]
        test_timestamps = timestamps[split_idx:]
        train_node_mask, val_node_mask = node_mask_seq[:split_idx], node_mask_seq[split_idx:]
        train_target_mask, val_target_mask = target_mask[:split_idx], target_mask[split_idx:]
        test_node_mask = val_node_mask

    trained_threshold = None
    if mode in ["train", "update"]:
        print("Commencing model optimization...")
        _, _, trained_threshold, node_error_stats = train_temporal_gnn(
            model,
            train_seq,
            train_tgt,
            edge_index,
            val_sequences=test_seq,
            val_targets=test_tgt,
            feature_cols=feature_cols,
            train_node_mask=train_node_mask,
            val_node_mask=val_node_mask,
            train_target_mask=train_target_mask,
            val_target_mask=val_target_mask,
        )
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Optimization complete. Weights saved to {model_path}")

        # trainer only knows node index, not site name, so translate here where
        # location_to_idx is available
        idx_to_location = {idx: loc for loc, idx in location_to_idx.items()}
        if node_error_stats:
            error_median = {idx_to_location[i]: s["median"] for i, s in node_error_stats.items()}
            error_iqr = {idx_to_location[i]: s["iqr"] for i, s in node_error_stats.items()}
            node_thresholds = {
                idx_to_location[i]: s["threshold"] for i, s in node_error_stats.items()
            }
            # normal conductivity LEVEL per node, not error -- the anchor the
            # level-shift rule needs, see trainer.py's node_error_stats docstring
            cond_median = {
                idx_to_location[i]: s["cond_median"] for i, s in node_error_stats.items()
            }
            cond_iqr = {idx_to_location[i]: s["cond_iqr"] for i, s in node_error_stats.items()}
        else:
            error_median, error_iqr, node_thresholds = {}, {}, {}
            cond_median, cond_iqr = {}, {}

        with open(metadata_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "feature_cols": feature_cols,
                    "location_to_idx": location_to_idx,
                    # old scalar threshold, kept so nothing depending on it hard
                    # crashes, but detection should use node_thresholds now
                    "threshold": trained_threshold,
                    "threshold_percentile": Config.THRESHOLD_PERCENTILE,
                    "error_median": error_median,
                    "error_iqr": error_iqr,
                    "node_thresholds": node_thresholds,
                    "cond_median": cond_median,
                    "cond_iqr": cond_iqr,
                },
                f,
            )
        print(f"Model metadata saved to {metadata_path}")
        detection_metadata = {
            "feature_cols": feature_cols,
            "location_to_idx": location_to_idx,
            "error_median": error_median,
            "error_iqr": error_iqr,
            "node_thresholds": node_thresholds,
            "cond_median": cond_median,
            "cond_iqr": cond_iqr,
        }
    else:
        print("Skipping training phase. Entering evaluation mode.")
        # saved_metadata was already loaded above for feature alignment, and
        # loading_existing being true (the only way to reach this branch) means
        # it has everything detect_anomalies needs -- no need to read it twice.
        detection_metadata = saved_metadata

    model.eval()
    node_results, rain_flags = detect_anomalies(
        model,
        test_seq,
        test_tgt,
        test_timestamps,
        test_node_mask,
        edge_index,
        detection_metadata,
        df_original=df_original,
        locations=locations,
        device=Config.DEVICE,
    )

    judged_sites = [site for site, r in node_results.items() if r["judged"]]
    unjudged_sites = [site for site, r in node_results.items() if not r["judged"]]
    flagged_sites = [site for site in judged_sites if node_results[site]["flagged"]]
    print(
        f"Detection cycle finished. Nodes judged: {len(judged_sites)}/{len(node_results)}. "
        f"Nodes flagged: {len(flagged_sites)}."
    )
    for site in unjudged_sites:
        r = node_results[site]
        print(f"  {site}: not judged, {r['reason']} (n_real={r['n_real']})")
    for site in flagged_sites:
        r = node_results[site]
        for rule_name in r["rules_fired"]:
            rule = r["rule1"] if rule_name == "forecast_residual" else r["rule2"]
            print(
                f"  {site}: {rule_name}, peak_deviation={rule['peak_deviation']:.2f}, "
                f"duration={rule['longest_run']} timesteps ({rule['n_over_threshold']} total over threshold)"
            )

    if visualize:
        print(
            "--visualize is not yet ported to the per-node detection rewrite -- "
            "plot_static_dashboard/plot_interactive_plotly still expect the old "
            "collapsed-scalar shape (system_anomaly_scores, spill_flags, etc). "
            "Skipping plots this run."
        )

    if mode == "inference" and flagged_sites:
        try:
            from strawberrywatch.utils.notifier import send_spill_alert

            send_spill_alert(len(flagged_sites), flagged_sites)
        except Exception as e:
            print(f"Alerting failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCMG GNN Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        default="update",
        choices=["train", "update", "inference"],
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="api",
        choices=["api", "sql"],
        help="Where to pull data from: REST API (default) or SQL database",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dusk_crayfish",
        choices=list(_MODEL_REGISTRY.keys()),
        help="Which model architecture to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate static and interactive plots after detection",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to the data CSV, overriding the default cache. Accepts either "
        "the long-format rolling cache or a wide training-corpus CSV (see "
        "scripts/build_training_corpus.py; auto-detected by *_valid columns). "
        "Omit to use the default cache, unchanged.",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        data_source=args.data_source,
        model_name=args.model,
        visualize=args.visualize,
        data_file=args.data_file,
    )

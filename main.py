import argparse
import inspect
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

from strawberrywatch import paths
from strawberrywatch.anomalies.anomaly_detector import LEVEL_SHIFT_K, detect_anomalies
from strawberrywatch.config import Config
from strawberrywatch.ingest.data_loader import load_and_preprocess_data
from strawberrywatch.models import model_calls
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized
from strawberrywatch.support_modules import DEFAULT_Q_TOTAL, SUPPORT_REGISTRY, SupportStack
from strawberrywatch.training.trainer import train_temporal_gnn
from strawberrywatch.utils.graph_utils import create_graph_topology

# --model names to classes. A new model goes here and nowhere else in main.py.
# dusk_crayfish stays exactly as trained, since the audit comparison rests on it.
_MODEL_REGISTRY = {
    "dusk_crayfish": DuskCrayfish,
    "cobble_shoal": CobbleShoal,
}

# What --operating-point may override, and the type each is read as.
# detect_anomalies ignores any other key, and a typo would have no effect. We
# catch it here.
_OPERATING_POINT = {
    "level_shift_k": float,
    "min_timesteps_over": int,
    "min_timesteps_to_judge": int,
}


def _operating_point_item(text):
    """One KEY=VALUE off --operating-point, checked before anything downloads."""
    key, _, value = text.partition("=")
    key = key.strip()
    cast = _OPERATING_POINT.get(key)
    if cast is None:
        raise argparse.ArgumentTypeError(
            f"{text!r}: expected KEY=VALUE with KEY one of {', '.join(_OPERATING_POINT)}"
        )
    try:
        number = cast(value)
    except ValueError:
        kind = "a whole number" if cast is int else "a number"
        raise argparse.ArgumentTypeError(f"{text!r}: {key} needs {kind}") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"{text!r}: {key} has to be positive")
    return key, number


def _align_to_trained_features(sequences, targets, current_feature_cols, trained_feature_cols):
    """
    Matches the sequences to the feature set the model trained on. Training might
    have seen 6 features while this run got 4 because a weather fetch failed, so
    missing ones go in as zeros (no signal) and extras are dropped. Returns
    aligned_sequences, aligned_targets and a short report of what moved.
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


def _support_window(sequences, targets, timestamps, node_mask, df_original, model=None):
    """
    What a support module gets as its window. A dict, since the kinds read
    different parts of it. model is the class, and a screen reads its
    FLAG_POLICY.
    """
    return {
        "sequences": sequences,
        "targets": targets,
        "timestamps": timestamps,
        "node_mask": node_mask,
        "raw": df_original,
        "model": model,
    }


def _fired_grid(node_results, site_order, shape):
    """
    Which node-timesteps the primary flagged, as the grid an explainer reads.

    A node that was never judged, or judged and not flagged, is all False. The
    primary didn't flag it, and an explainer has nothing to explain.
    """
    grid = np.zeros(shape, dtype=bool)
    for j, site in enumerate(site_order):
        result = node_results.get(site, {})
        if result.get("judged") and result.get("flagged"):
            grid[:, j] = result["over_timesteps"]
    return grid


def _anomaly_events(
    node_results, timestamps, metadata, accounts, site_order, level_shift_k=LEVEL_SHIFT_K
):
    """
    One alert event per (site, rule) that fired, shaped the way notifier wants.

    Rule 1 and Rule 2 go out separately. An onset and a lasting level mean
    different things to whoever reads the email. Trial Bed's account comes
    along when that explainer ran. level_shift_k is the k Rule 2 was held to,
    which lets an --operating-point run report its own bar.
    """
    node_thresholds = metadata.get("node_thresholds", {})
    trial_bed = accounts.get("trial_bed", [])
    site_pos = {site: i for i, site in enumerate(site_order)}

    events = []
    for site, result in node_results.items():
        if not result.get("judged") or not result.get("flagged"):
            continue

        pos = site_pos.get(site)
        classification = trial_bed[pos] if pos is not None and pos < len(trial_bed) else None

        for rule_name in result["rules_fired"]:
            rule = result["rule1"] if rule_name == "forecast_residual" else result["rule2"]
            # Rule 1's bar is the node's calibrated threshold, Rule 2's the flat k.
            # Rain moves Rule 1's per timestep, and we report the calibrated one.
            threshold = (
                node_thresholds.get(site, float("nan"))
                if rule_name == "forecast_residual"
                else level_shift_k
            )
            crossed = rule.get("over_timesteps")
            # First crossing, since that's when the thing started, not when we noticed.
            when = (
                timestamps[int(np.argmax(crossed))]
                if crossed is not None and crossed.any()
                else timestamps[-1]
            )
            events.append(
                {
                    "location": site,
                    "rule": rule_name,
                    "score": rule["peak_deviation"],
                    "threshold": threshold,
                    "event_time": when,
                    "classification": classification,
                }
            )
    return events


def _dispatch_alerts(events, alert_sites=None, dry_run=False, send=None):
    """
    Email what fired and stamp each event with a status saying what happened.

    alert_sites None means every site. A muted site is still scored. We keep its
    event, marked "muted", and the results file can show what muting cost. send
    is for tests, and is normally notifier.fire_anomaly_alerts.
    """
    outgoing = []
    for ev in events:
        if alert_sites is not None and ev["location"] not in alert_sites:
            ev["status"] = "muted"
            print(f"  muted: {ev['location']} {ev['rule']}")
        elif dry_run:
            ev["status"] = "dry_run"
            print(
                f"  would email: {ev['location']} {ev['rule']}, "
                f"peak {ev['score']:.2f} against {ev['threshold']:.2f}"
            )
        else:
            outgoing.append(ev)
    if not outgoing:
        return events

    try:
        if send is None:
            from strawberrywatch.utils.notifier import fire_anomaly_alerts as send
        delivered = list(send(outgoing) or [])
    except Exception as e:
        print(f"Alerting failed: {e}")
        delivered = []
    # notifier skips without a word when .env has no credentials. "emailed"
    # comes from what it reports as sent.
    for i, ev in enumerate(outgoing):
        ev["status"] = "emailed" if i < len(delivered) and delivered[i] else "not_emailed"
    return events


def _plain(value):
    """
    Something json can write. NaN and inf become null, since JSON has neither
    and a browser's JSON.parse refuses the file outright if they sneak in.
    """
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="s")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _write_results(path, header, node_results, accounts, site_order, events):
    """
    What the run found, for anything that wants it without parsing the log.

    Scalars only. Nobody reading this needs 192 booleans a rule.
    """
    sites = {}
    for site, result in node_results.items():
        entry = {"judged": bool(result.get("judged")), "n_real": result.get("n_real")}
        if entry["judged"]:
            entry["flagged"] = bool(result.get("flagged"))
            entry["rules_fired"] = list(result.get("rules_fired", []))
            for key in ("rule1", "rule2"):
                rule = result.get(key) or {}
                entry[key] = {
                    k: rule.get(k) for k in ("peak_deviation", "longest_run", "n_over_threshold")
                }
        else:
            entry["reason"] = result.get("reason")
        sites[site] = entry

    explanations = {}
    for name, per_node in accounts.items():
        for site, account in zip(site_order, per_node, strict=True):
            if account.get("verdict") == "nothing_to_explain":
                continue
            explanations.setdefault(name, {})[site] = {
                k: account.get(k) for k in ("verdict", "cause", "confidence", "explanation")
            }

    alerts = []
    for ev in events:
        classification = ev.get("classification") or {}
        alerts.append(
            {
                "site": ev["location"],
                "rule": ev["rule"],
                "score": ev["score"],
                "threshold": ev["threshold"],
                "event_time": ev["event_time"],
                "status": ev.get("status"),
                "diagnosis": classification.get("explanation"),
                "diagnosis_verdict": classification.get("verdict"),
                "diagnosis_cause": classification.get("cause"),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        payload = {**header, "sites": sites, "explanations": explanations, "alerts": alerts}
        json.dump(_plain(payload), f, indent=2, allow_nan=False)
    print(f"Results written to {path}")


def main(
    mode="update",
    data_source="api",
    model_name="dusk_crayfish",
    data_file=None,
    support=None,
    q_total=DEFAULT_Q_TOTAL,
    operating_point=None,
    alert_sites=None,
    dry_run=False,
    results_json=None,
):
    """
    Run the GNN anomaly detection pipeline in train, update, or inference mode.

    support is a list of names from support_modules.SUPPORT_REGISTRY, or None
    for a run with no support attached. None and [] take the same path as three
    modules do; there is no separate no-support branch anywhere below.

    The last four default to the usual behavior. operating_point overrides
    detect_anomalies' constants, alert_sites limits who gets emailed about
    (every site is still scored), dry_run prints the alerts instead of sending
    them, and results_json writes what the run found to that path.
    """
    requested_mode = mode
    model_dir = paths.checkpoints_dir()
    model_path = os.path.join(model_dir, f"{model_name}_weights.pt")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.pkl")

    file_path = data_file or Config.data_file()

    if model_name not in _MODEL_REGISTRY:
        print(f"ERROR: Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
        sys.exit(1)
    ModelClass = _MODEL_REGISTRY[model_name]

    # Everything below is the sequence-tensor pipeline. A nested-node model
    # shares none of it and would only fail with a TypeError after the download,
    # so it's refused up front with the path that does work.
    if model_calls.input_contract(ModelClass) != model_calls.SEQUENCE_TENSOR:
        print(
            f"ERROR: '{model_name}' speaks the "
            f"{model_calls.input_contract(ModelClass)} contract, which main.py "
            f"does not drive. This pipeline builds sequence tensors; that model "
            f"needs per-node windows from preprocessing.node_windows."
        )
        print()
        print("       Score it on the labelled events with:")
        print("           python scripts/run_audit_comparison.py")
        print("       Refit its calibration against the archive with:")
        print("           python scripts/refit_cobble_calibration.py")
        sys.exit(1)

    # Same reason as the support check below: a typo should fail before a
    # thirty day download, not after it.
    unknown = sorted(set(operating_point or {}) - set(_OPERATING_POINT))
    if unknown:
        print(f"ERROR: unknown operating point keys {unknown}. Valid: {list(_OPERATING_POINT)}")
        sys.exit(1)
    off_roster = sorted(set(alert_sites or []) - set(Config.LOCATIONS))
    if off_roster:
        print(f"ERROR: alert sites {off_roster} are not on the roster {Config.LOCATIONS}")
        sys.exit(1)

    # Resolved before anything downloads. An unknown name, a module that can't
    # attach, a collision or an impossible budget split fail with nothing done.
    stack = SupportStack.load(support, ModelClass, q_total=q_total)
    split = stack.budget()

    print("SCMG Anomaly Detection System")
    print(f"Execution Mode: {mode.upper()}")
    print(f"Model:          {model_name}")
    print(f"Device:         {Config.DEVICE}")
    print(f"Data file:      {file_path}")
    print(f"Support:        {', '.join(stack.names) if stack.names else 'none'}")
    for line in stack.describe():
        print(f"  {line}")
    print(f"False alarm budget: q_total={split['q_total']:g}, primary={split['primary']:g}")
    for name, q in sorted(split["detectors"].items()):
        print(f"  {name}: q={q:g}")
    if Config.SETTINGS_OVERRIDES:
        print(f"Settings:       settings.yaml + {Config.SETTINGS_OVERRIDES}")
    if operating_point:
        pairs = ", ".join(f"{k}={v:g}" for k, v in sorted(operating_point.items()))
        print(f"Operating point: {pairs}")
    if alert_sites is not None:
        print(f"Alerting on:    {', '.join(alert_sites) if alert_sites else 'no sites'}")
    if dry_run:
        print("Dry run:        alerts get printed, nothing is emailed")

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
            saved_metadata = pickle.load(f)  # nosec B301: our own checkpoint, written by train
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

    # Node names in score-column order, handed to every support module with its window
    site_order = [site for site, _ in sorted(location_to_idx.items(), key=lambda kv: kv[1])]
    support_window = _support_window(
        sequences, targets, timestamps, node_mask_seq, df_original, model=ModelClass
    )

    # Screens run here, before the model sees anything. node_mask_seq carries
    # "don't read this" through the rest of the pipeline, and a screen narrows
    # that mask instead of editing readings.
    if stack.screens:
        admitted = stack.admit(support_window, site_order, (len(sequences), len(site_order)))
        withheld = int((~admitted).sum())
        node_mask_seq = node_mask_seq & admitted[:, None, :]
        print(
            f"screens withheld {withheld} of {admitted.size} node-timesteps "
            f"({100 * withheld / admitted.size:.2f}%)"
        )

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
        # Inference doesn't split. All of the mask applies to the test set
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

        # trainer only knows node indices. We translate here, where location_to_idx is.
        idx_to_location = {idx: loc for loc, idx in location_to_idx.items()}
        if node_error_stats:
            error_median = {idx_to_location[i]: s["median"] for i, s in node_error_stats.items()}
            error_iqr = {idx_to_location[i]: s["iqr"] for i, s in node_error_stats.items()}
            node_thresholds = {
                idx_to_location[i]: s["threshold"] for i, s in node_error_stats.items()
            }
            # normal conductivity LEVEL per node, not error, the anchor the
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
                    # So a checkpoint says what model to rebuild, instead of the
                    # loader guessing from whatever settings.yaml says later
                    "architecture": getattr(model, "architecture", None),
                    # The old scalar threshold, kept for older readers. Detection
                    # uses node_thresholds.
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
        # Already loaded above, with everything detect_anomalies needs
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
        operating_point=operating_point,
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

    # Modulators and detectors run after the primary has judged: a modulator
    # moves the bar the primary was compared against, a detector adds its own
    # test beside it. Both read the same window the model read.
    shape = (len(test_seq), len(site_order))
    if stack.modulators or stack.detectors:
        stack.multipliers(support_window, site_order, shape)
        stack.detector_results(support_window, site_order)

    # Explainers run last, on firings the primary already made. They add no test
    # and take no budget.
    accounts = {}
    if stack.explainers:
        fired_grid = _fired_grid(node_results, site_order, shape)
        accounts = stack.explanations(support_window, site_order, fired_grid)
        for name, per_node in sorted(accounts.items()):
            for site, account in zip(site_order, per_node, strict=True):
                if account["verdict"] == "nothing_to_explain":
                    continue
                print(f"  {name} at {site}: {account['explanation']}")
                print(f"    confidence: {account['confidence']}")

    # Every anomaly emails out, whatever the mode: one email per rule per site
    # plus a batch summary, all to ALERT_EMAIL_RECEIVER
    level_k = (operating_point or {}).get("level_shift_k", LEVEL_SHIFT_K)
    events = _anomaly_events(
        node_results, test_timestamps, detection_metadata, accounts, site_order, level_k
    )
    _dispatch_alerts(events, alert_sites, dry_run)

    if results_json:
        header = {
            "model": model_name,
            "requested_mode": requested_mode,
            "mode": mode,
            "support": stack.names,
            "q_total": q_total,
            "operating_point": operating_point or {},
            "alert_sites": alert_sites,
            "dry_run": dry_run,
            "settings_overrides": Config.SETTINGS_OVERRIDES,
            "window": [test_timestamps[0], test_timestamps[-1]] if len(test_timestamps) else [],
        }
        _write_results(results_json, header, node_results, accounts, site_order, events)


if __name__ == "__main__":
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
        "--support",
        type=str,
        nargs="+",
        default=None,
        choices=sorted(SUPPORT_REGISTRY),
        metavar="NAME",
        help="Support modules to attach, space separated. Order does not "
        "matter. Omit for a run with no support. Valid names: "
        + ", ".join(sorted(SUPPORT_REGISTRY)),
    )
    parser.add_argument(
        "--q-total",
        type=float,
        default=DEFAULT_Q_TOTAL,
        help="Total false alarm rate for the whole run, divided across the "
        "primary and every attached detector rather than handed to each. "
        f"Default {DEFAULT_Q_TOTAL:g}.",
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
    parser.add_argument(
        "--operating-point",
        type=_operating_point_item,
        nargs="+",
        default=None,
        metavar="KEY=VALUE",
        help="Override detection constants for this run, space separated. Keys: "
        + ", ".join(_OPERATING_POINT)
        + ". Omit to use anomaly_detector's own values.",
    )
    parser.add_argument(
        "--alert-sites",
        nargs="+",
        default=None,
        choices=Config.LOCATIONS,
        metavar="SITE",
        help="Only email about these sites. Every site is still scored, the rest "
        "just stay quiet. Omit to alert on all of them. Valid names: "
        + ", ".join(Config.LOCATIONS),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the alerts this run would email and send nothing.",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        metavar="PATH",
        help="Also write what the run found to PATH as JSON: per site verdicts, "
        "explainer accounts, and every alert with whether it was sent.",
    )
    args = parser.parse_args()
    main(
        mode=args.mode,
        data_source=args.data_source,
        model_name=args.model,
        data_file=args.data_file,
        support=args.support,
        q_total=args.q_total,
        operating_point=dict(args.operating_point) if args.operating_point else None,
        alert_sites=args.alert_sites,
        dry_run=args.dry_run,
        results_json=args.results_json,
    )

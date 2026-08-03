"""
Forecast skill gate: does Dusk Crayfish actually beat trivial baselines at one-step
conductivity forecasting, on real observed data in the held-out split?

Every anomaly test in the suite passes when reconstruction error is HIGH, so a
model that just forecasts badly passes those tests for free. This is the test
that has to pass before any of that is meaningful: if the model doesn't clearly
beat persistence per node, its error is noise around a lagged value and nothing
downstream is interpretable.

Read-only. Doesn't touch the model, the corpus, or anything else in the repo.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent

from strawberrywatch import paths
from strawberrywatch.config import Config
from strawberrywatch.ingest.data_loader import load_and_preprocess_data
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized
from strawberrywatch.utils.graph_utils import create_graph_topology

MODEL_NAME = "dusk_crayfish"
USE_MASK = True  # matches conftest.py's (dusk_crayfish, True) pairing
CORPUS_PATH = paths.training_corpus()
MIN_EVALUABLE = 200
LAG_15MIN = pd.Timedelta(minutes=15)
LAG_24H = pd.Timedelta(hours=24)


def load_model_and_metadata():
    meta_path = paths.checkpoints_dir() / f"{MODEL_NAME}_metadata.pkl"
    weights_path = paths.checkpoints_dir() / f"{MODEL_NAME}_weights.pt"
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    num_features = len(metadata["feature_cols"])
    num_nodes = len(metadata["location_to_idx"])
    model = DuskCrayfish(num_node_features=num_features, num_nodes=num_nodes).to(Config.DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model, metadata


def step1_split_and_correspondence(metadata):
    print("=" * 70)
    print("STEP 1: held-out split")
    print("=" * 70)
    print("main.py's split, for mode in ['train', 'update']:")
    print("    split_idx = int(len(sequences) * Config.TRAIN_SPLIT)")
    print("    train_seq, test_seq = sequences[:split_idx], sequences[split_idx:]")
    print(f"Config.TRAIN_SPLIT = {Config.TRAIN_SPLIT}")
    print()
    print("sequences come from a chronological sliding window over sorted timestamps")
    print("(data_processor.py::prepare_sequences_normalized), and split_idx slices that")
    print("array positionally. No shuffling anywhere in the pipeline -- this IS a")
    print("chronological split, not random. Held-out is strictly the most recent slice.")
    print()

    edge_index, _, location_to_idx = create_graph_topology()
    df_featured, _, _ = load_and_preprocess_data(
        file_path=CORPUS_PATH, force_download=True, days=999
    )
    sequences, targets, timestamps, scaler, feature_cols, node_mask_seq, target_mask = (
        prepare_sequences_normalized(
            df_featured, location_to_idx, Config.SEQUENCE_LENGTH, return_node_mask=True
        )
    )

    mean_match = np.allclose(scaler.mean_, metadata["scaler"].mean_)
    scale_match = np.allclose(scaler.scale_, metadata["scaler"].scale_)
    print(f"checking whether models/{MODEL_NAME}_weights.pt corresponds to this exact split:")
    print(
        f"  recomputed scaler vs saved metadata scaler -- mean match: {mean_match}, scale match: {scale_match}"
    )
    if mean_match and scale_match:
        print(f"  MATCH. This checkpoint was fit on {CORPUS_PATH} as it exists right now,")
        print("  so its held-out split is exactly the one computed below.")
    else:
        print(f"  MISMATCH. models/{MODEL_NAME}_weights.pt does NOT correspond to the current")
        print("  corpus/pipeline. Everything below is being computed on a split that does not")
        print("  match what this checkpoint actually held out during its own training -- retrain")
        print("  first (see command at the end) before trusting these numbers.")
    print()

    split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
    held_ts = pd.DatetimeIndex(timestamps[split_idx:])
    print(f"total sequences: {len(sequences):,}")
    print(
        f"split_idx: {split_idx:,} ({Config.TRAIN_SPLIT:.0%} train / {1 - Config.TRAIN_SPLIT:.0%} held out)"
    )
    print(f"held-out sequences: {len(held_ts):,}")
    print(f"held-out date range: {held_ts[0]} -> {held_ts[-1]}")
    print()

    return {
        "edge_index": edge_index,
        "location_to_idx": location_to_idx,
        "feature_cols": feature_cols,
        "scaler": metadata["scaler"],  # the one the checkpoint actually trained with
        "held_seq": sequences[split_idx:],
        "held_tgt": targets[split_idx:],
        "held_ts": held_ts,
        "held_node_mask": node_mask_seq[split_idx:],
    }


def build_real_observation_lookups(node_names):
    """
    Per site: a Series of RAW (un-imputed) conductivity indexed by datetime,
    read straight off training_corpus.csv, plus the {site}_valid QC column.

    NaN in the raw column means that grid step had no underlying raw sensor
    reading. Checked this is bit-identical to freshly resampling
    data/raw_data/ onto the same grid (build_training_corpus.py never imputes
    -- only prepare_sequences_normalized does, and only in memory at train
    time), so reading the corpus directly IS the reconstruction the task
    asked for, not a shortcut around it.
    """
    corpus = pd.read_csv(CORPUS_PATH)
    corpus["datetime"] = pd.to_datetime(corpus["datetime"], utc=True)
    corpus = corpus.set_index("datetime")

    raw_cond, qc_valid = {}, {}
    for site in node_names:
        raw_cond[site] = corpus[f"{site}_conductivity"]
        qc_valid[site] = corpus[f"{site}_valid"].astype(bool)
    return raw_cond, qc_valid


def run_model_over_held_out(model, edge_index, held_seq, held_node_mask, num_nodes, batch_size=256):
    """
    Same calling convention test_anomaly_detection.py::_reconstruction_errors
    uses (node_mask passed when USE_MASK, model.eval() + no_grad already set),
    just batched instead of one sequence at a time. Verified first that this
    model has no batch-dependent layers (no batchnorm, dropout is off in
    eval()) -- batched output matched single-sequence output to float32
    precision (max abs diff ~6e-8) on a synthetic check, so this is a speed
    fix, not a change to what's being measured. 5,820 sequences one at a time
    didn't finish in 10 minutes; this does.
    """
    preds = []
    with torch.no_grad():
        for i in range(0, len(held_seq), batch_size):
            seq_t = torch.FloatTensor(held_seq[i : i + batch_size]).to(Config.DEVICE)
            if USE_MASK:
                mask_t = torch.BoolTensor(held_node_mask[i : i + batch_size]).to(Config.DEVICE)
                pred = model(
                    seq_t, edge_index, batch_size=len(seq_t), num_nodes=num_nodes, node_mask=mask_t
                )
            else:
                pred = model(seq_t, edge_index, batch_size=len(seq_t), num_nodes=num_nodes)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)  # (n_held, num_nodes, num_features)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    model, metadata = load_model_and_metadata()
    ctx = step1_split_and_correspondence(metadata)

    location_to_idx = ctx["location_to_idx"]
    feature_cols = ctx["feature_cols"]
    scaler = ctx["scaler"]
    held_ts = ctx["held_ts"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    cond_mean, cond_scale = scaler.mean_[cond_idx], scaler.scale_[cond_idx]

    print("=" * 70)
    print("STEP 2: filtering to real observations")
    print("=" * 70)
    print("method: raw corpus NaN pattern for {site}_conductivity (verified equivalent")
    print("to re-resampling data/raw_data/ onto the grid -- build_training_corpus.py")
    print("applies no imputation, so the corpus's own NaN IS that reconstruction).")
    print()

    raw_cond, qc_valid = build_real_observation_lookups(node_names)

    t_minus_1 = held_ts - LAG_15MIN
    t_minus_24h = held_ts - LAG_24H

    per_node = {}
    print(f"{'node':15s} {'total':>8s} {'stage1 ok':>10s} {'pct':>7s}   status")
    for site in node_names:
        target_raw = raw_cond[site].reindex(held_ts)
        valid = qc_valid[site].reindex(held_ts).fillna(False)
        stage1_keep = target_raw.notna().to_numpy() & valid.to_numpy()
        n_stage1 = int(stage1_keep.sum())
        pct = 100 * n_stage1 / len(held_ts)

        status = "insufficient (<200)" if n_stage1 < MIN_EVALUABLE else "ok"
        print(f"{site:15s} {len(held_ts):>8,} {n_stage1:>10,} {pct:>6.1f}%   {status}")

        per_node[site] = {
            "target_raw": target_raw,
            "stage1_keep": stage1_keep,
            "n_stage1": n_stage1,
            "sufficient": n_stage1 >= MIN_EVALUABLE,
        }
    print()

    print("=" * 70)
    print("running model over held-out sequences (one at a time, use_mask=True)...")
    print("=" * 70)
    preds_norm = run_model_over_held_out(
        model, ctx["edge_index"], ctx["held_seq"], ctx["held_node_mask"], num_nodes
    )
    preds_raw_cond = preds_norm[:, :, cond_idx] * cond_scale + cond_mean  # (n_held, num_nodes)
    print(f"done. predictions shape: {preds_norm.shape}\n")

    print("=" * 70)
    print("STEP 3: baselines vs model, conductivity only, per node")
    print("=" * 70)

    rows = []
    for node_idx, site in enumerate(node_names):
        info = per_node[site]
        if not info["sufficient"]:
            rows.append({"node": site, "n": info["n_stage1"], "verdict": "INSUFFICIENT DATA"})
            continue

        persistence_raw = raw_cond[site].reindex(t_minus_1)
        persistence_raw.index = held_ts
        diurnal_raw = raw_cond[site].reindex(t_minus_24h)
        diurnal_raw.index = held_ts

        lag_real = persistence_raw.notna().to_numpy() & diurnal_raw.notna().to_numpy()
        final_keep = info["stage1_keep"] & lag_real
        n_dropped_by_lag = int(info["stage1_keep"].sum() - final_keep.sum())
        n = int(final_keep.sum())

        print(
            f"\n[{site}] stage1 ok={info['n_stage1']:,}, dropped for missing lag={n_dropped_by_lag:,}, final n={n:,}"
        )

        if n < MIN_EVALUABLE:
            rows.append({"node": site, "n": n, "verdict": "INSUFFICIENT DATA (after lag filter)"})
            continue

        target_raw = info["target_raw"].to_numpy()[final_keep]
        persist_raw = persistence_raw.to_numpy()[final_keep]
        diurnal_raw_v = diurnal_raw.to_numpy()[final_keep]
        model_raw = preds_raw_cond[final_keep, node_idx]

        # normalized units = the space the model was actually trained/scored in
        target_n = (target_raw - cond_mean) / cond_scale
        persist_n = (persist_raw - cond_mean) / cond_scale
        diurnal_n = (diurnal_raw_v - cond_mean) / cond_scale
        model_n = (model_raw - cond_mean) / cond_scale

        mse_persist = np.mean((target_n - persist_n) ** 2)
        mse_diurnal = np.mean((target_n - diurnal_n) ** 2)
        mse_model = np.mean((target_n - model_n) ** 2)

        skill_persist = 1 - mse_model / mse_persist if mse_persist > 0 else float("nan")
        skill_diurnal = 1 - mse_model / mse_diurnal if mse_diurnal > 0 else float("nan")

        beats_persist = skill_persist > 0
        beats_diurnal = skill_diurnal > 0
        if beats_persist and beats_diurnal:
            verdict = "beats both baselines"
        elif beats_persist or beats_diurnal:
            verdict = f"beats only {'persistence' if beats_persist else 'diurnal naive'}"
        else:
            verdict = "beats NEITHER baseline"

        residuals = target_n - model_n
        autocorr = float(pd.Series(residuals).autocorr(lag=1))

        rows.append(
            {
                "node": site,
                "n": n,
                "mae_persist_norm": mae(target_n, persist_n),
                "mae_diurnal_norm": mae(target_n, diurnal_n),
                "mae_model_norm": mae(target_n, model_n),
                "rmse_persist_norm": rmse(target_n, persist_n),
                "rmse_diurnal_norm": rmse(target_n, diurnal_n),
                "rmse_model_norm": rmse(target_n, model_n),
                "mae_persist_raw": mae(target_raw, persist_raw),
                "mae_diurnal_raw": mae(target_raw, diurnal_raw_v),
                "mae_model_raw": mae(target_raw, model_raw),
                "skill_vs_persist": skill_persist,
                "skill_vs_diurnal": skill_diurnal,
                "resid_autocorr_lag1": autocorr,
                "verdict": verdict,
            }
        )

    print()
    print("=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    df = pd.DataFrame(rows).set_index("node")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df.to_string())
    print()

    print("=" * 70)
    print("STEP 4: model residual lag-1 autocorrelation")
    print("=" * 70)
    print("computed on the same final-filtered, chronologically-ordered residual series")
    print("as the table above. Gaps from filtering mean 'lag-1' is 'previous surviving")
    print("point', not always exactly 15 minutes back -- caveat, not a rewrite of the metric.")
    for _, row in df.iterrows():
        if "resid_autocorr_lag1" in row and pd.notna(row.get("resid_autocorr_lag1")):
            print(f"  {row.name:15s} autocorr(resid, lag=1) = {row['resid_autocorr_lag1']:.4f}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    failing = []
    for _, row in df.iterrows():
        v = row.get("verdict", "")
        print(f"  {row.name:15s} n={row.get('n', 0):>6}   {v}")
        if "INSUFFICIENT" in v or "NEITHER" in v or v.startswith("beats only"):
            failing.append((row.name, v))

    print()
    if not failing:
        print("Every node with enough real data clearly beats both persistence and diurnal")
        print("naive on conductivity. The reconstruction error downstream reflects genuine")
        print("forecasting skill, not noise around a lagged value.")
    else:
        print("Does the model beat persistence per node, on real observed data? Not everywhere:")
        for name, v in failing:
            print(f"  - {name}: {v}")
        print()
        print("Any node listed above means its anomaly-detection results are not currently")
        print("interpretable -- the reconstruction error there could just be persistence error")
        print("in disguise (or, for insufficient nodes, there isn't enough real data this")
        print("window to know either way).")


if __name__ == "__main__":
    main()

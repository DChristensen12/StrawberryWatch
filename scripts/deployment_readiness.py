"""
Deployment readiness gate for Dusk Crayfish. Four questions: does it generalize
past the training window, does it stay quiet on clean data, does it actually
detect and localize a real perturbation, and what's the smallest spill it
reliably catches. The skill baseline test only answered "is the forecast
better than a lagged copy" -- this is the next gate up, whether the thing is
safe to point at production.

Read-only. Doesn't touch the model, the corpus, or existing files. Reuses the
imputation-detection method from forecast_skill_baseline.py and the real
detection path (robust-normalized error, rain-adjusted threshold, _is_flagged)
from tests/test_anomaly_detection.py rather than re-deriving either.

conductivity only, per node (never averaged), real observations only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strawberrywatch.config import Config
from strawberrywatch.utils.graph_utils import create_graph_topology
from strawberrywatch.ingest.data_loader import load_and_preprocess_data
from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized

from forecast_skill_baseline import (
    load_model_and_metadata,
    build_real_observation_lookups,
    run_model_over_held_out,
    mae,
    CORPUS_PATH,
    MIN_EVALUABLE,
    LAG_15MIN,
)
import tests.test_anomaly_detection as tad  # noqa: E402  (real detection path: _is_flagged, _rain_adjusted_thresholds, _normalize, the MIN_* constants)

ANOMALY_DIR = ROOT / "data" / "anomalies"
ANOMALY_PAD = pd.Timedelta(hours=12)  # same padding build_training_corpus.py uses
WINDOW_STEPS = 192  # 2 days at 15-min cadence, the Section 2/3 window size
MIN_WINDOW_STEPS = Config.SEQUENCE_LENGTH + tad.MIN_TIMESTEPS_TO_JUDGE

INJECTION_SHAPES = ("step", "ramp", "spike")
RAMP_STEPS = 8    # 2 hours at 15-min cadence
SPIKE_STEPS = 2   # 30 minutes at 15-min cadence
CONFIRM_MAGNITUDE = 150.0  # uS/cm, Section 3's "does the mechanism work at all" magnitude
N_SECTION3_WINDOWS = 10    # a spread subset of the clean windows, not all 30 -- compute tractability

SECTION4_NODES = ("north_fork_0", "south_fork_2")  # the two nodes with enough real data to sweep meaningfully
SECTION4_MAGNITUDES = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 500.0, 2000.0)  # 500/2000 exist to settle
# whether "not reached" at spill-scale magnitudes is a calibration problem or a structural one -- see 4b
RELIABLE_DETECTION_RATE = 0.5  # "reliably detected" = flagged in most windows, per the spec

FP_RATE_CAVEAT = 0.2       # false positive rate above this gets flagged in the verdict
DETECT_RATE_CAVEAT = 0.5   # detection rate below this (at CONFIRM_MAGNITUDE) gets flagged
LOCALIZE_RATE_CAVEAT = 0.2  # neighbor false-flag rate above this gets flagged


def load_corpus():
    corpus = pd.read_csv(ROOT / CORPUS_PATH)
    corpus["datetime"] = pd.to_datetime(corpus["datetime"], utc=True)
    return corpus.set_index("datetime")


def held_out_range(corpus):
    """
    Same boundary main.py's split produces, derived directly from the corpus
    index instead of re-running the full sequence build (which takes a couple
    minutes just to pivot). Every timestep in this corpus is a valid sequence
    position -- Section 1 confirmed valid_mask is all-True, 29,099 sequences
    from 29,123 timesteps, exactly len - SEQUENCE_LENGTH -- so the sequence
    split index maps onto the timestamp index with a fixed offset.
    """
    all_ts = corpus.index.sort_values().unique()
    n_sequences = len(all_ts) - Config.SEQUENCE_LENGTH
    split_idx = int(n_sequences * Config.TRAIN_SPLIT)
    held_start = all_ts[Config.SEQUENCE_LENGTH + split_idx]
    held_end = all_ts[-1]
    return held_start, held_end


def all_anomaly_windows(pad=ANOMALY_PAD):
    """
    Every folder under data/anomalies/, min/max timestamp padded 12h each
    side -- same derivation build_training_corpus.py uses, EXCEPT this
    doesn't skip botanical_actuator. That event stays in the training corpus
    on purpose (it's being dropped from the catalog, treated as ordinary wet
    season data), but it's still a real labeled event, and calling its window
    "clean" for a false-positive test would be circular.
    """
    windows = []
    for folder in sorted(ANOMALY_DIR.glob("*")):
        if not folder.is_dir():
            continue
        starts, ends = [], []
        for f in folder.glob("*.csv"):
            df = pd.read_csv(f, usecols=lambda c: c in ("DateTimeUTC", "timestamp", "datetime"))
            tcol = next((c for c in ("DateTimeUTC", "timestamp", "datetime") if c in df.columns), None)
            if tcol is None:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dropna()
            if ts.empty:
                continue
            starts.append(ts.min())
            ends.append(ts.max())
        if starts:
            windows.append((folder.name, min(starts) - pad, max(ends) + pad))
    return windows


def build_clean_windows(corpus, held_start, held_end, window_steps=WINDOW_STEPS, min_steps=MIN_WINDOW_STEPS):
    """
    Contiguous 15-min-grid stretches inside the held-out span that (a) exist
    as real rows in the corpus -- anomaly-window rows were already stripped
    when the corpus was built, so most of the "does this overlap a known
    anomaly" work is already done for us -- and (b) don't fall inside ANY
    known anomaly window including botanical_actuator (see
    all_anomaly_windows). Chops each contiguous stretch into non-overlapping
    window_steps chunks, keeping a final remainder if it still clears
    min_steps.

    A sequence built across a gap (say, from just before the hydrant_nf0
    exclusion to just after it) would silently jump 2 days between two
    "consecutive" array positions without anything flagging it, so windows
    have to come from genuinely wall-clock-contiguous stretches, not just
    whatever rows happen to exist.
    """
    full_grid = pd.date_range(corpus.index.min(), corpus.index.max(), freq="15min")
    present = pd.Series(full_grid.isin(corpus.index), index=full_grid)

    clean = present.copy()
    for _, start, end in all_anomaly_windows():
        clean &= ~((full_grid >= start) & (full_grid <= end))

    in_span = (full_grid >= held_start) & (full_grid <= held_end)
    clean = clean[in_span]

    # group consecutive True runs
    runs = []
    run_start = None
    prev_ok, prev_ts = False, None
    for ts, ok in clean.items():
        if ok and not prev_ok:
            run_start = ts
        if not ok and prev_ok:
            runs.append((run_start, prev_ts))
        prev_ok, prev_ts = ok, ts
    if prev_ok:
        runs.append((run_start, prev_ts))

    windows = []
    for run_start, run_end in runs:
        run_idx = full_grid[(full_grid >= run_start) & (full_grid <= run_end)]
        n_chunks = len(run_idx) // window_steps
        for c in range(n_chunks):
            chunk = run_idx[c * window_steps:(c + 1) * window_steps]
            if len(chunk) >= min_steps:
                windows.append(chunk)
        remainder = run_idx[n_chunks * window_steps:]
        if len(remainder) >= min_steps:
            windows.append(remainder)

    return windows


def build_window_arrays(corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid):
    """
    (T, num_nodes, num_features) raw array for one clean window, straight off
    the corpus -- no imputation, these windows were chosen specifically
    because the underlying rows are real. node_mask combines "conductivity is
    actually present" with "{site}_valid": a node is only trusted if BOTH
    hold. QC-valid alone isn't enough -- a node can be un-flagged by QC and
    still have a plain data gap, and if node_mask said "trust this" for a gap,
    feature propagation would never kick in to fix it, and the model would
    see a fabricated raw zero as if it were a real reading.
    """
    num_nodes = len(location_to_idx)
    num_features = len(feature_cols)
    T = len(window_ts)
    data_3d = np.full((T, num_nodes, num_features), np.nan)
    node_mask = np.zeros((T, num_nodes), dtype=bool)

    for site, node_idx in location_to_idx.items():
        for feat in ("conductivity", "depth", "temperature"):
            if feat in feature_cols:
                col = f"{site}_{feat}"
                if col in corpus.columns:
                    data_3d[:, node_idx, feature_cols.index(feat)] = corpus.loc[window_ts, col].to_numpy()
        for feat in ("rain_mm", "air_temp_c", "shortwave_radiation"):
            if feat in feature_cols and feat in corpus.columns:
                data_3d[:, node_idx, feature_cols.index(feat)] = corpus.loc[window_ts, feat].to_numpy()

        real = raw_cond[site].reindex(window_ts).notna().to_numpy()
        valid = qc_valid[site].reindex(window_ts).fillna(False).to_numpy()
        node_mask[:, node_idx] = real & valid

    time_feats = tad._add_time_features(window_ts)
    for fname, vals in time_feats.items():
        if fname in feature_cols:
            fi = feature_cols.index(fname)
            for node_idx in range(num_nodes):
                data_3d[:, node_idx, fi] = vals

    rain_series = corpus.loc[window_ts, "rain_mm"] if "rain_mm" in corpus.columns else None

    data_3d = np.nan_to_num(data_3d, nan=0.0)
    return data_3d, node_mask, rain_series


def run_window_model(model, edge_index, normalized, node_mask, num_nodes, batch_size=256):
    """
    Batched forward pass over one window's sliding sequences. Returns
    (predictions, targets), both (n_positions, num_nodes, num_features)
    normalized, or (None, None) if the window is too short for one sequence.
    Every node's predictions come out of the same pass, so this only needs to
    run once per window, not once per target node.
    """
    seq_len = Config.SEQUENCE_LENGTH
    n_positions = len(normalized) - seq_len
    if n_positions <= 0:
        return None, None

    seqs = np.stack([normalized[i:i + seq_len] for i in range(n_positions)])
    masks = np.stack([node_mask[i:i + seq_len] for i in range(n_positions)])
    targets = np.stack([normalized[i + seq_len] for i in range(n_positions)])

    preds = []
    with torch.no_grad():
        for i in range(0, n_positions, batch_size):
            seq_t = torch.FloatTensor(seqs[i:i + batch_size]).to(Config.DEVICE)
            mask_t = torch.BoolTensor(masks[i:i + batch_size]).to(Config.DEVICE)
            pred = model(seq_t, edge_index, batch_size=len(seq_t), num_nodes=num_nodes, node_mask=mask_t)
            preds.append(pred.cpu().numpy())
    predictions = np.concatenate(preds, axis=0)
    return predictions, targets


def robust_errors_for_node(predictions, targets, node_idx, cond_idx, error_median, error_iqr):
    raw_err = np.abs(predictions[:, node_idx, cond_idx] - targets[:, node_idx, cond_idx])
    return (raw_err - error_median) / error_iqr


def flagged_for_site(predictions, targets, site, node_idx, cond_idx, target_ts, rain_series,
                      node_thresholds, error_median, error_iqr):
    """Real detection path for one node on one window: robust error, rain-adjusted threshold, _is_flagged."""
    errs = robust_errors_for_node(predictions, targets, node_idx, cond_idx, error_median[site], error_iqr[site])
    thresholds = tad._rain_adjusted_thresholds(node_thresholds[site], target_ts, rain_series)
    return tad._is_flagged(errs, thresholds)


def node_fully_real(target_ts, site, raw_cond, qc_valid):
    """
    True if site has a real, QC-valid conductivity reading at EVERY scored
    position in this window. Required for the injected node (so the
    perturbation lands on genuine readings, not a fabricated placeholder) and
    for any node we're checking for false localization (a node that was going
    to be diffused/propagated the whole time regardless of the injection
    can't tell us anything about smearing).
    """
    real = raw_cond[site].reindex(target_ts).notna().to_numpy()
    valid = qc_valid[site].reindex(target_ts).fillna(False).to_numpy()
    return bool((real & valid).all())


def apply_injection(data_3d, node_idx, cond_idx, start_pos, shape, magnitude):
    """
    Adds a synthetic perturbation to data_3d[:, node_idx, cond_idx] in RAW
    conductivity units, from start_pos through the end of the window. Doesn't
    mutate the input. Injected into the same array used to build both the
    model's input sequences and its targets, so the target the model gets
    scored against is the perturbed (i.e. "actual sensor reading now") value
    -- exactly what a real spill would look like: clean history in, anomalous
    reading out, error is the gap between what the model expected and what's
    actually there.
    """
    out = data_3d.copy()
    T = out.shape[0]
    delta = np.zeros(T)

    if shape == "step":
        delta[start_pos:] = magnitude
    elif shape == "ramp":
        ramp_end = min(start_pos + RAMP_STEPS, T)
        n_ramp = ramp_end - start_pos
        if n_ramp > 0:
            delta[start_pos:ramp_end] = magnitude * np.arange(1, n_ramp + 1) / RAMP_STEPS
        delta[ramp_end:] = magnitude
    elif shape == "spike":
        spike_end = min(start_pos + SPIKE_STEPS, T)
        delta[start_pos:spike_end] = magnitude
    else:
        raise ValueError(f"unknown injection shape: {shape}")

    out[:, node_idx, cond_idx] = out[:, node_idx, cond_idx] + delta
    return out


def load_full_split():
    """
    Same corpus load + sequence build as forecast_skill_baseline.py, but keeps
    both the train and held-out portions instead of just held-out -- Section 1
    needs both to compare.
    """
    edge_index, _, location_to_idx = create_graph_topology()
    df_featured, _, _ = load_and_preprocess_data(
        file_path=CORPUS_PATH, force_download=True, days=999
    )
    sequences, targets, timestamps, scaler, feature_cols, node_mask_seq, target_mask = (
        prepare_sequences_normalized(
            df_featured, location_to_idx, Config.SEQUENCE_LENGTH, return_node_mask=True
        )
    )
    split_idx = int(len(sequences) * Config.TRAIN_SPLIT)
    ts = pd.DatetimeIndex(timestamps)

    return {
        "edge_index": edge_index,
        "location_to_idx": location_to_idx,
        "feature_cols": feature_cols,
        "split_idx": split_idx,
        "train_seq": sequences[:split_idx],
        "train_ts": ts[:split_idx],
        "train_node_mask": node_mask_seq[:split_idx],
        "held_seq": sequences[split_idx:],
        "held_ts": ts[split_idx:],
        "held_node_mask": node_mask_seq[split_idx:],
    }


def portion_skill_vs_persistence(site, portion_ts, preds_raw_cond, node_idx, raw_cond, qc_valid, cond_mean, cond_scale):
    """
    conductivity MAE (raw uS/cm) and skill vs persistence for one node, on one
    portion (train or held-out), real observations only. Same filtering
    methodology as forecast_skill_baseline.py: target must be a real reading
    and QC-valid, and the t-1 lag persistence depends on must also be real, so
    persistence and the model are compared on identical points.
    """
    target_raw = raw_cond[site].reindex(portion_ts)
    valid = qc_valid[site].reindex(portion_ts).fillna(False)
    stage1_keep = target_raw.notna().to_numpy() & valid.to_numpy()

    persistence_raw = raw_cond[site].reindex(portion_ts - LAG_15MIN)
    persistence_raw.index = portion_ts
    lag_real = persistence_raw.notna().to_numpy()

    final_keep = stage1_keep & lag_real
    n = int(final_keep.sum())
    if n < MIN_EVALUABLE:
        return {"n": n, "sufficient": False}

    tgt = target_raw.to_numpy()[final_keep]
    pers = persistence_raw.to_numpy()[final_keep]
    model = preds_raw_cond[final_keep, node_idx]

    tgt_n = (tgt - cond_mean) / cond_scale
    pers_n = (pers - cond_mean) / cond_scale
    model_n = (model - cond_mean) / cond_scale

    mse_persist = np.mean((tgt_n - pers_n) ** 2)
    mse_model = np.mean((tgt_n - model_n) ** 2)
    skill = 1 - mse_model / mse_persist if mse_persist > 0 else float("nan")

    return {
        "n": n,
        "sufficient": True,
        "mae_model_raw": mae(tgt, model),
        "mae_persist_raw": mae(tgt, pers),
        "skill_vs_persist": skill,
    }


def section1(model, metadata):
    print("=" * 70)
    print("SECTION 1: overfitting vs generalization")
    print("=" * 70)

    ctx = load_full_split()
    location_to_idx = ctx["location_to_idx"]
    feature_cols = ctx["feature_cols"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    scaler = metadata["scaler"]
    cond_mean, cond_scale = scaler.mean_[cond_idx], scaler.scale_[cond_idx]

    print("\n1a. split check")
    print("main.py: split_idx = int(len(sequences) * Config.TRAIN_SPLIT); positional slice,")
    print("no shuffling anywhere in the pipeline. Chronological, confirmed (same as the skill")
    print("baseline script established).")
    print(f"  train:     {ctx['train_ts'][0]} -> {ctx['train_ts'][-1]}  ({len(ctx['train_ts']):,} sequences)")
    print(f"  held-out:  {ctx['held_ts'][0]} -> {ctx['held_ts'][-1]}  ({len(ctx['held_ts']):,} sequences)")
    print(
        "  note: the corpus itself only starts 2025-06-04 (bounded to when the last graph node\n"
        "  came online, see build_training_corpus.py), so train contains ZERO prior spring data.\n"
        "  held-out's Mar-May window is the only spring this model has ever seen, in either split."
    )

    print("\nrunning model over train sequences (this is the big one, ~4x held-out)...")
    train_preds_norm = run_model_over_held_out(
        model, ctx["edge_index"], ctx["train_seq"], ctx["train_node_mask"], num_nodes
    )
    train_preds_raw_cond = train_preds_norm[:, :, cond_idx] * cond_scale + cond_mean
    print(f"done. {train_preds_raw_cond.shape}")

    print("\nrunning model over held-out sequences...")
    held_preds_norm = run_model_over_held_out(
        model, ctx["edge_index"], ctx["held_seq"], ctx["held_node_mask"], num_nodes
    )
    held_preds_raw_cond = held_preds_norm[:, :, cond_idx] * cond_scale + cond_mean
    print(f"done. {held_preds_raw_cond.shape}\n")

    raw_cond, qc_valid = build_real_observation_lookups(node_names)

    print("1b. conductivity MAE (raw uS/cm) and skill vs persistence, train vs held-out")
    print("-" * 100)
    header = f"{'node':15s} {'train n':>9s} {'train MAE':>10s} {'train skill':>12s}   {'held n':>8s} {'held MAE':>10s} {'held skill':>11s}   {'MAE gap':>8s}"
    print(header)

    rows = []
    for node_idx, site in enumerate(node_names):
        train_r = portion_skill_vs_persistence(
            site, ctx["train_ts"], train_preds_raw_cond, node_idx, raw_cond, qc_valid, cond_mean, cond_scale
        )
        held_r = portion_skill_vs_persistence(
            site, ctx["held_ts"], held_preds_raw_cond, node_idx, raw_cond, qc_valid, cond_mean, cond_scale
        )

        if not train_r["sufficient"] or not held_r["sufficient"]:
            print(f"{site:15s} {'insufficient data (train n=' + str(train_r['n']) + ', held n=' + str(held_r['n']) + ')':>90s}")
            rows.append({"node": site, "train": train_r, "held": held_r, "status": "insufficient"})
            continue

        gap = held_r["mae_model_raw"] - train_r["mae_model_raw"]
        print(
            f"{site:15s} {train_r['n']:>9,} {train_r['mae_model_raw']:>10.3f} {train_r['skill_vs_persist']:>12.4f}   "
            f"{held_r['n']:>8,} {held_r['mae_model_raw']:>10.3f} {held_r['skill_vs_persist']:>11.4f}   {gap:>8.3f}"
        )
        rows.append({"node": site, "train": train_r, "held": held_r, "status": "ok", "gap": gap})

    print("\n1c. interpretation")
    for row in rows:
        site = row["node"]
        if row["status"] == "insufficient":
            print(f"  {site}: insufficient real data on one or both portions, can't assess generalization here.")
            continue
        train_r, held_r, gap = row["train"], row["held"], row["gap"]
        ratio = held_r["mae_model_raw"] / train_r["mae_model_raw"] if train_r["mae_model_raw"] > 0 else float("inf")
        print(f"  {site}: train MAE {train_r['mae_model_raw']:.3f}, held MAE {held_r['mae_model_raw']:.3f} ({ratio:.1f}x)")
        print(
            f"    train skill vs persistence {train_r['skill_vs_persist']:.4f}, "
            f"held skill vs persistence {held_r['skill_vs_persist']:.4f}"
        )

    print()
    print("Held-out covers a season (spring, post-wet-season drawdown) that literally does not")
    print("exist anywhere in the training portion, because the corpus itself starts June 2025 --")
    print("there's no prior spring to have included. That's a structural argument for distribution")
    print("shift being at least a real contributor to any gap above, for every node, before even")
    print("looking at the per-node numbers. Whether it's ALSO overfitting can't be fully separated")
    print("from that with a single chronological split and no independent same-season holdout --")
    print("said plainly: this data cannot distinguish 'overfit' from 'never seen this season' on")
    print("its own. A same-season holdout (e.g. holding out last spring instead of last calendar")
    print("chunk) would be needed to separate them cleanly.")

    return ctx, rows


def section2(model, metadata):
    print("=" * 70)
    print("SECTION 2: false positive rate on clean data")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_thresholds = metadata["node_thresholds"]
    error_median = metadata["error_median"]
    error_iqr = metadata["error_iqr"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    edge_index, _, _ = create_graph_topology()

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)

    held_start, held_end = held_out_range(corpus)
    print(f"held-out span: {held_start} -> {held_end}")

    print("\n2a. clean window extraction")
    print(f"window size: {WINDOW_STEPS} steps (2 days), minimum viable: {MIN_WINDOW_STEPS} steps "
          f"(SEQUENCE_LENGTH={Config.SEQUENCE_LENGTH} + MIN_TIMESTEPS_TO_JUDGE={tad.MIN_TIMESTEPS_TO_JUDGE})")
    windows = build_clean_windows(corpus, held_start, held_end)
    print(f"clean windows found: {len(windows)} (aiming for >=15)")
    for i, w in enumerate(windows):
        print(f"  window {i:2d}: {w[0]} -> {w[-1]}  ({len(w)} steps)")

    print("\n2b/2c. running the real detection path per window per node")
    print("(robust-normalized error, per-node rain-adjusted threshold, _is_flagged"
          f" with MIN_TIMESTEPS_OVER_THRESHOLD={tad.MIN_TIMESTEPS_OVER_THRESHOLD})")

    results = {
        site: {"n_eval": 0, "n_flagged": 0, "n_rain": 0, "n_flagged_rain": 0}
        for site in node_names
    }

    for window_ts in windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        normalized = tad._normalize(data_3d, scaler, feature_cols, location_to_idx)
        predictions, targets = run_window_model(model, edge_index, normalized, node_mask, num_nodes)
        if predictions is None:
            continue
        target_ts = window_ts[Config.SEQUENCE_LENGTH:]

        did_rain = bool(
            rain_series is not None
            and (rain_series.reindex(target_ts).fillna(0) > Config.RAIN_AMOUNT_THRESHOLD).any()
        )

        for site in node_names:
            if site not in node_thresholds:
                continue
            node_idx = location_to_idx[site]
            errs_all = robust_errors_for_node(
                predictions, targets, node_idx, cond_idx, error_median[site], error_iqr[site]
            )

            real = (
                raw_cond[site].reindex(target_ts).notna().to_numpy()
                & qc_valid[site].reindex(target_ts).fillna(False).to_numpy()
            )
            n_real = int(real.sum())
            if n_real < tad.MIN_TIMESTEPS_TO_JUDGE:
                continue

            errs_real = errs_all[real]
            real_ts = target_ts[real]
            thresholds = tad._rain_adjusted_thresholds(node_thresholds[site], real_ts, rain_series)
            flagged = tad._is_flagged(errs_real, thresholds)

            results[site]["n_eval"] += 1
            if did_rain:
                results[site]["n_rain"] += 1
            if flagged:
                results[site]["n_flagged"] += 1
                if did_rain:
                    results[site]["n_flagged_rain"] += 1

    print()
    print("2c. false positive rate per node")
    print("-" * 90)
    print(f"{'node':15s} {'evaluated':>10s} {'flagged':>8s} {'FP rate':>9s}   {'rain wins':>10s} {'flag@rain':>10s} {'flag@dry':>9s}")
    for site in node_names:
        r = results[site]
        if r["n_eval"] == 0:
            print(f"{site:15s} {'insufficient clean data to evaluate':>60s}")
            continue
        fp_rate = r["n_flagged"] / r["n_eval"]
        n_dry = r["n_eval"] - r["n_rain"]
        n_flagged_dry = r["n_flagged"] - r["n_flagged_rain"]
        print(
            f"{site:15s} {r['n_eval']:>10d} {r['n_flagged']:>8d} {fp_rate:>9.1%}   "
            f"{r['n_rain']:>10d} {r['n_flagged_rain']:>10d} {n_flagged_dry:>9d}"
        )
        if r["n_rain"] > 0 and r["n_flagged"] > 0:
            rain_flag_rate = r["n_flagged_rain"] / r["n_rain"]
            dry_flag_rate = n_flagged_dry / n_dry if n_dry > 0 else float("nan")
            print(f"    flag rate during rain windows: {rain_flag_rate:.1%}  vs  dry windows: {dry_flag_rate:.1%}")

    return results


def section3(model, metadata, magnitude=CONFIRM_MAGNITUDE, n_windows=N_SECTION3_WINDOWS):
    print("=" * 70)
    print(f"SECTION 3: synthetic injection, detection and localization (X={magnitude:.0f} uS/cm)")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_thresholds = metadata["node_thresholds"]
    error_median = metadata["error_median"]
    error_iqr = metadata["error_iqr"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    edge_index, _, _ = create_graph_topology()

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)
    held_start, held_end = held_out_range(corpus)
    all_windows = build_clean_windows(corpus, held_start, held_end)
    test_windows = all_windows[::max(1, len(all_windows) // n_windows)][:n_windows]
    print(f"using {len(test_windows)} of {len(all_windows)} clean windows (spread across the held-out span)")
    print(f"injection starts at the first scored position in each window (index {Config.SEQUENCE_LENGTH}),")
    print(f"ramp climbs over {RAMP_STEPS * 15} min, spike holds for {SPIKE_STEPS * 15} min\n")

    detection = {}       # (site, shape) -> {"tested", "detected"}
    localization = {}    # site -> {"injections", "neighbor_checks", "neighbor_false_flags"}
    skipped = {"target_not_real": 0, "no_usable_neighbors": 0}

    for window_ts in test_windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        target_ts = window_ts[Config.SEQUENCE_LENGTH:]
        start_pos = Config.SEQUENCE_LENGTH

        fully_real = {site: node_fully_real(target_ts, site, raw_cond, qc_valid) for site in node_names}

        for target_site in node_names:
            if not fully_real[target_site]:
                skipped["target_not_real"] += 1
                continue
            neighbor_sites = [s for s in node_names if s != target_site and fully_real[s]]
            if not neighbor_sites:
                skipped["no_usable_neighbors"] += 1
                continue

            node_idx = location_to_idx[target_site]

            for shape in INJECTION_SHAPES:
                injected_3d = apply_injection(data_3d, node_idx, cond_idx, start_pos, shape, magnitude)
                normalized = tad._normalize(injected_3d, scaler, feature_cols, location_to_idx)
                predictions, targets = run_window_model(model, edge_index, normalized, node_mask, num_nodes)
                if predictions is None:
                    continue

                flagged_target = flagged_for_site(
                    predictions, targets, target_site, node_idx, cond_idx, target_ts, rain_series,
                    node_thresholds, error_median, error_iqr
                )

                key = (target_site, shape)
                d = detection.setdefault(key, {"tested": 0, "detected": 0})
                d["tested"] += 1
                if flagged_target:
                    d["detected"] += 1

                loc = localization.setdefault(target_site, {"injections": 0, "neighbor_checks": 0, "neighbor_false_flags": 0})
                loc["injections"] += 1
                for nb_site in neighbor_sites:
                    nb_idx = location_to_idx[nb_site]
                    flagged_nb = flagged_for_site(
                        predictions, targets, nb_site, nb_idx, cond_idx, target_ts, rain_series,
                        node_thresholds, error_median, error_iqr
                    )
                    loc["neighbor_checks"] += 1
                    if flagged_nb:
                        loc["neighbor_false_flags"] += 1

    print(f"skipped: target node not fully real in window = {skipped['target_not_real']}, "
          f"no usable neighbors to check = {skipped['no_usable_neighbors']}\n")

    print("3c. DETECTION: did the injected node flag, per node per shape")
    print("-" * 60)
    print(f"{'node':15s} {'step':>15s} {'ramp':>15s} {'spike':>15s}")
    for site in node_names:
        cells = []
        for shape in INJECTION_SHAPES:
            d = detection.get((site, shape))
            if d is None or d["tested"] == 0:
                cells.append("no data")
            else:
                rate = d["detected"] / d["tested"]
                cells.append(f"{d['detected']}/{d['tested']} ({rate:.0%})")
        print(f"{site:15s} {cells[0]:>15s} {cells[1]:>15s} {cells[2]:>15s}")

    print("\n3d. LOCALIZATION: did a clean neighbor false-flag when a DIFFERENT node was perturbed")
    print("-" * 60)
    print(f"{'injected node':15s} {'injections':>11s} {'neighbor checks':>16s} {'false flags':>12s} {'rate':>7s}")
    for site in node_names:
        loc = localization.get(site)
        if loc is None or loc["neighbor_checks"] == 0:
            continue
        rate = loc["neighbor_false_flags"] / loc["neighbor_checks"]
        print(
            f"{site:15s} {loc['injections']:>11d} {loc['neighbor_checks']:>16d} "
            f"{loc['neighbor_false_flags']:>12d} {rate:>6.1%}"
        )

    return detection, localization


def section4(model, metadata, nodes=SECTION4_NODES, magnitudes=SECTION4_MAGNITUDES, n_windows=N_SECTION3_WINDOWS):
    """
    Step injection only, one node at a time, sweeping magnitude instead of
    shape. Same clean windows as Section 3 for consistency. Restricted to
    north_fork_0 and south_fork_2 because those are the only two nodes with
    enough fully-real windows for a sweep to mean anything -- oxford's
    duplication gaps and footbridge's sparse cadence make node_fully_real
    rare enough that a 6-point sweep would mostly be "no data".
    """
    print("=" * 70)
    print("SECTION 4: sensitivity curve (step injection, magnitude sweep)")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_thresholds = metadata["node_thresholds"]
    error_median = metadata["error_median"]
    error_iqr = metadata["error_iqr"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    edge_index, _, _ = create_graph_topology()

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)
    held_start, held_end = held_out_range(corpus)
    all_windows = build_clean_windows(corpus, held_start, held_end)
    test_windows = all_windows[::max(1, len(all_windows) // n_windows)][:n_windows]
    print(f"using {len(test_windows)} of {len(all_windows)} clean windows, nodes={nodes}\n")

    results = {site: {} for site in nodes}  # site -> magnitude -> {"tested", "detected", "n_over_sum"}

    for window_ts in test_windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        target_ts = window_ts[Config.SEQUENCE_LENGTH:]
        start_pos = Config.SEQUENCE_LENGTH

        for site in nodes:
            if site not in location_to_idx or not node_fully_real(target_ts, site, raw_cond, qc_valid):
                continue
            node_idx = location_to_idx[site]

            for magnitude in magnitudes:
                injected_3d = apply_injection(data_3d, node_idx, cond_idx, start_pos, "step", magnitude)
                normalized = tad._normalize(injected_3d, scaler, feature_cols, location_to_idx)
                predictions, targets = run_window_model(model, edge_index, normalized, node_mask, num_nodes)
                if predictions is None:
                    continue
                errs = robust_errors_for_node(predictions, targets, node_idx, cond_idx, error_median[site], error_iqr[site])
                thresholds = tad._rain_adjusted_thresholds(node_thresholds[site], target_ts, rain_series)
                n_over = int((errs > thresholds).sum())
                d = results[site].setdefault(magnitude, {"tested": 0, "detected": 0, "n_over_sum": 0})
                d["tested"] += 1
                d["n_over_sum"] += n_over
                if n_over >= tad.MIN_TIMESTEPS_OVER_THRESHOLD:
                    d["detected"] += 1

    print("detection rate vs magnitude, per node (avg elevated timesteps per window in parens)")
    print(f"flagging needs >= {tad.MIN_TIMESTEPS_OVER_THRESHOLD} elevated timesteps in one window")
    print("-" * 70)
    print(f"{'magnitude (uS/cm)':>18s}" + "".join(f"{site:>26s}" for site in nodes))

    min_reliable = {}
    for magnitude in magnitudes:
        row = f"{magnitude:>18.0f}"
        for site in nodes:
            d = results[site].get(magnitude)
            if d is None or d["tested"] == 0:
                row += f"{'no data':>26s}"
                continue
            rate = d["detected"] / d["tested"]
            avg_over = d["n_over_sum"] / d["tested"]
            cell = f"{d['detected']}/{d['tested']} ({rate:.0%}, avg {avg_over:.1f})"
            row += f"{cell:>26s}"
            if rate > RELIABLE_DETECTION_RATE and site not in min_reliable:
                min_reliable[site] = magnitude
        print(row)

    print(f"\nsmallest magnitude reliably detected (flagged in >{RELIABLE_DETECTION_RATE:.0%} of windows):")
    for site in nodes:
        if site in min_reliable:
            print(f"  {site}: {min_reliable[site]:.0f} uS/cm -- the honest deployment claim for this node")
        else:
            avg_over_top = results[site].get(magnitudes[-1], {}).get("n_over_sum", 0) / max(
                results[site].get(magnitudes[-1], {}).get("tested", 1), 1
            )
            print(
                f"  {site}: not reached anywhere in the tested range {magnitudes} "
                f"(avg elevated timesteps still only {avg_over_top:.1f} at {magnitudes[-1]:.0f} uS/cm)"
            )

    print(
        "\n4c. why this caps out instead of climbing with magnitude: a step injection is a single\n"
        "level shift. The model predicts one step ahead, so as soon as the elevated reading enters\n"
        "its own input history (the very next timestep), it starts predicting the NEW value as\n"
        "normal -- the residual collapses back down immediately. That caps a step at roughly 1\n"
        "elevated timestep and a spike (up then back down) at roughly 2, no matter how large the\n"
        "jump is -- confirmed above up to 2000 uS/cm, 4x the entire conductivity scale. Since\n"
        f"_is_flagged needs >= {tad.MIN_TIMESTEPS_OVER_THRESHOLD} elevated timesteps, a clean step/spike on an\n"
        "otherwise-quiet window structurally can't cross that bar at these nodes, regardless of X.\n"
        "Ramp shapes don't have this problem, because each of the RAMP_STEPS climbing steps is its\n"
        "own fresh surprise -- Section 3's ramp numbers (which use total magnitude, not per-step\n"
        "rate) are the more honest read on real detectability here, not this sweep."
    )

    return results, min_reliable


def final_verdict(node_names, section1_rows, section2_results, section3_detection, section3_localization, section4_min_reliable):
    """
    Pulls the four sections into one scoped call per node instead of a single
    yes/no. Thresholds below (FP_RATE_CAVEAT, DETECT_RATE_CAVEAT,
    LOCALIZE_RATE_CAVEAT) are judgment calls, not something the model or data
    dictate -- printed alongside the table so the reasoning is checkable, not
    just the label.
    """
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print(f"thresholds used: FP rate > {FP_RATE_CAVEAT:.0%} caveat, detection < {DETECT_RATE_CAVEAT:.0%} caveat, "
          f"neighbor false-flag > {LOCALIZE_RATE_CAVEAT:.0%} caveat\n")

    skill_by_node = {r["node"]: r for r in section1_rows}

    print(f"{'node':15s} {'held skill':>10s} {'FP rate':>8s} {'detect avg':>10s} {'localize':>9s} {'min reliable':>13s}")
    print("-" * 75)

    verdicts = {}
    for site in node_names:
        s1 = skill_by_node.get(site)
        skill = s1["held"]["skill_vs_persist"] if s1 and s1["status"] == "ok" else None

        fp = section2_results.get(site)
        fp_rate = fp["n_flagged"] / fp["n_eval"] if fp and fp["n_eval"] > 0 else None

        shape_rates = {}
        for shape in INJECTION_SHAPES:
            d = section3_detection.get((site, shape))
            if d and d["tested"] > 0:
                shape_rates[shape] = d["detected"] / d["tested"]
        detect_avg = sum(shape_rates.values()) / len(shape_rates) if shape_rates else None
        ramp_rate = shape_rates.get("ramp")
        step_spike_rates = [r for s, r in shape_rates.items() if s != "ramp"]
        step_spike_avg = sum(step_spike_rates) / len(step_spike_rates) if step_spike_rates else None

        loc = section3_localization.get(site)
        loc_rate = loc["neighbor_false_flags"] / loc["neighbor_checks"] if loc and loc["neighbor_checks"] > 0 else None

        min_reliable = section4_min_reliable.get(site)

        skill_str = f"{skill:.3f}" if skill is not None else "n/a"
        fp_str = f"{fp_rate:.0%}" if fp_rate is not None else "n/a"
        detect_str = f"{detect_avg:.0%}" if detect_avg is not None else "n/a"
        loc_str = f"{loc_rate:.0%}" if loc_rate is not None else "n/a"
        min_str = f"{min_reliable:.0f}" if min_reliable is not None else "n/a"
        print(f"{site:15s} {skill_str:>10s} {fp_str:>8s} {detect_str:>10s} {loc_str:>9s} {min_str:>13s}")

        caveats = []
        if skill is None:
            verdict = "not deployable"
            caveats.append("insufficient real held-out data to assess forecast skill at all")
        elif skill < 0:
            verdict = "not deployable"
            caveats.append(f"forecast doesn't beat persistence (skill={skill:.3f})")
        elif detect_avg is None:
            verdict = "not deployable"
            caveats.append("no fully-real injection windows -- can't confirm detection works on this node")
        else:
            verdict = "deployable"
            if fp_rate is not None and fp_rate > FP_RATE_CAVEAT:
                verdict = "deployable with caveats"
                caveats.append(f"false-positive rate {fp_rate:.0%} on clean data")
            if detect_avg < DETECT_RATE_CAVEAT:
                verdict = "deployable with caveats"
                if ramp_rate is not None and ramp_rate >= DETECT_RATE_CAVEAT and step_spike_avg is not None and step_spike_avg < DETECT_RATE_CAVEAT:
                    caveats.append(
                        f"sustained-change (ramp) events are caught ({ramp_rate:.0%}), but a clean "
                        f"instantaneous jump (step/spike, {step_spike_avg:.0%}) mostly isn't -- see Section 4c, "
                        "this is a rule limitation (needs 3+ elevated timesteps), not a per-node calibration gap"
                    )
                else:
                    caveats.append(f"detection rate only {detect_avg:.0%} at {CONFIRM_MAGNITUDE:.0f} uS/cm")
            if loc_rate is not None and loc_rate > LOCALIZE_RATE_CAVEAT:
                verdict = "deployable with caveats"
                caveats.append(f"neighbors false-flag {loc_rate:.0%} of the time when this node is perturbed")

        verdicts[site] = {"verdict": verdict, "caveats": caveats}

    print()
    for site in node_names:
        v = verdicts[site]
        print(f"{site}: {v['verdict']}")
        for c in v["caveats"]:
            print(f"  - {c}")

    return verdicts


def main():
    model, metadata = load_model_and_metadata()
    model_metadata_pair = (model, metadata)
    _, s1_rows = section1(*model_metadata_pair)
    s2_results = section2(*model_metadata_pair)
    s3_detection, s3_localization = section3(*model_metadata_pair)
    _, s4_min_reliable = section4(*model_metadata_pair)
    node_names = list(metadata["location_to_idx"].keys())
    final_verdict(node_names, s1_rows, s2_results, s3_detection, s3_localization, s4_min_reliable)


if __name__ == "__main__":
    main()

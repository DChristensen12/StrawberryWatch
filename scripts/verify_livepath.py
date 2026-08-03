"""
Stage 3: does the anomaly_detector.py rewrite actually work, tested against the
REAL production functions (Rule 1 + Rule 2 from src/anomalies/anomaly_detector.py),
not a re-derivation of them and not the test-path _is_flagged this repo's other
battery (deployment_readiness.py) normally borrows.

Reuses deployment_readiness.py's window building, clean-window extraction, and
injection machinery completely unchanged -- only the scoring/flagging step is
swapped from tad._is_flagged to the actual ad._rule1_forecast_residual /
ad._rule2_level_shift a live run would call. tad._normalize is still used, that's
just scaler.transform per node, not a detection decision, same thing
prepare_sequences_normalized does in production.

Read-only, no retrain, no pytest.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from strawberrywatch.config import Config
from strawberrywatch.utils.graph_utils import create_graph_topology
import strawberrywatch.anomalies.anomaly_detector as ad
import tests.test_anomaly_detection as tad  # noqa: E402  (only _normalize + MIN_TIMESTEPS_TO_JUDGE, infra not detection logic)

from forecast_skill_baseline import load_model_and_metadata, build_real_observation_lookups
from deployment_readiness import (
    load_corpus,
    held_out_range,
    build_clean_windows,
    build_window_arrays,
    run_window_model,
    apply_injection,
    node_fully_real,
    CONFIRM_MAGNITUDE,
    N_SECTION3_WINDOWS,
    INJECTION_SHAPES,
    SECTION4_NODES,
    SECTION4_MAGNITUDES,
)

# prior numbers from the test-path deployment battery, for the before/after tables
PRIOR_FP_RATE = {
    "north_fork_0": 0.267,
    "south_fork_2": 0.167,
    "south_fork_1": 0.312,
    "oxford": 0.143,
}
PRIOR_STEP_DETECTION_150 = {
    "north_fork_0": 0.20,
    "south_fork_2": 0.17,
    "south_fork_1": 0.00,
    "oxford": 0.17,
}


def _score_node(predictions, targets, node_idx, cond_idx, site, real_ts, rain_series, metadata):
    """
    Runs the actual production rules (Rule 1 + Rule 2) on one node, filtered to
    real observation positions only, same filtering deployment_readiness.py's
    test-path scoring always used. Returns (flagged, rule1_result, rule2_result).
    """
    error_median = metadata["error_median"]
    error_iqr = metadata["error_iqr"]
    node_thresholds = metadata["node_thresholds"]
    cond_median = metadata["cond_median"]
    cond_iqr = metadata["cond_iqr"]

    errors_node = np.abs(predictions[:, node_idx, cond_idx] - targets[:, node_idx, cond_idx])
    levels_node = targets[:, node_idx, cond_idx]

    rain_mult, _ = ad._rain_multipliers(
        real_ts,
        rain_series,
        Config.RAIN_WINDOW_HOURS,
        Config.RAIN_THRESHOLD_MULTIPLIER,
        Config.RAIN_AMOUNT_THRESHOLD,
        Config.POST_RAIN_DECAY_HOURS,
    )
    rule1 = ad._rule1_forecast_residual(
        errors_node, error_median[site], error_iqr[site], node_thresholds[site], rain_mult
    )

    rule2 = {
        "flagged": False,
        "n_over_threshold": 0,
        "longest_run": 0,
        "peak_deviation": float("nan"),
    }
    if cond_median.get(site) is not None and cond_iqr.get(site) is not None:
        level_k = ad._rain_adjust_level_k(ad.LEVEL_SHIFT_K, real_ts, rain_series)
        rule2 = ad._rule2_level_shift(levels_node, cond_median[site], cond_iqr[site], level_k)

    flagged = rule1["flagged"] or rule2["flagged"]
    return flagged, rule1, rule2


def section2_livepath(model, metadata, edge_index):
    print("=" * 70)
    print("SECTION 2 (live path): false positive rate on clean data")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_thresholds = metadata["node_thresholds"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")
    edge_index_local = edge_index

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)
    held_start, held_end = held_out_range(corpus)
    windows = build_clean_windows(corpus, held_start, held_end)
    print(f"clean windows: {len(windows)}\n")

    results = {
        site: {"n_eval": 0, "n_flagged": 0, "n_flagged_rule1": 0, "n_flagged_rule2": 0}
        for site in node_names
    }

    for window_ts in windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        normalized = tad._normalize(data_3d, scaler, feature_cols, location_to_idx)
        predictions, targets = run_window_model(
            model, edge_index_local, normalized, node_mask, num_nodes
        )
        if predictions is None:
            continue
        target_ts = window_ts[Config.SEQUENCE_LENGTH :]

        for site in node_names:
            if site not in node_thresholds:
                continue
            node_idx = location_to_idx[site]
            real = (
                raw_cond[site].reindex(target_ts).notna().to_numpy()
                & qc_valid[site].reindex(target_ts).fillna(False).to_numpy()
            )
            n_real = int(real.sum())
            if n_real < tad.MIN_TIMESTEPS_TO_JUDGE:
                continue
            real_ts = target_ts[real]

            preds_real = predictions[real]
            tgts_real = targets[real]
            flagged, rule1, rule2 = _score_node(
                preds_real, tgts_real, node_idx, cond_idx, site, real_ts, rain_series, metadata
            )

            results[site]["n_eval"] += 1
            if flagged:
                results[site]["n_flagged"] += 1
            if rule1["flagged"]:
                results[site]["n_flagged_rule1"] += 1
            if rule2["flagged"]:
                results[site]["n_flagged_rule2"] += 1

    print(
        f"{'node':15s} {'evaluated':>10s} {'flagged':>8s} {'FP rate':>9s} {'rule1':>7s} {'rule2':>7s}"
    )
    for site in node_names:
        r = results[site]
        if r["n_eval"] == 0:
            print(f"{site:15s} {'insufficient clean data':>50s}")
            continue
        fp_rate = r["n_flagged"] / r["n_eval"]
        print(
            f"{site:15s} {r['n_eval']:>10d} {r['n_flagged']:>8d} {fp_rate:>9.1%} "
            f"{r['n_flagged_rule1']:>7d} {r['n_flagged_rule2']:>7d}"
        )
    return results


def section3_livepath(
    model, metadata, edge_index, magnitude=CONFIRM_MAGNITUDE, n_windows=N_SECTION3_WINDOWS
):
    print("=" * 70)
    print(f"SECTION 3 (live path): injection detection + localization (X={magnitude:.0f} uS/cm)")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)
    held_start, held_end = held_out_range(corpus)
    all_windows = build_clean_windows(corpus, held_start, held_end)
    test_windows = all_windows[:: max(1, len(all_windows) // n_windows)][:n_windows]
    print(f"using {len(test_windows)} of {len(all_windows)} clean windows\n")

    detection = {}  # (site, shape) -> {"tested", "flagged", "rule1", "rule2"}
    localization = {}  # site -> {"rule2_injections", "rule2_neighbor_checks", "rule2_neighbor_false_flags",
    #          "any_injections", "any_neighbor_checks", "any_neighbor_false_flags"}

    for window_ts in test_windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        target_ts = window_ts[Config.SEQUENCE_LENGTH :]
        start_pos = Config.SEQUENCE_LENGTH
        fully_real = {
            site: node_fully_real(target_ts, site, raw_cond, qc_valid) for site in node_names
        }

        for target_site in node_names:
            if not fully_real[target_site]:
                continue
            neighbor_sites = [s for s in node_names if s != target_site and fully_real[s]]
            if not neighbor_sites:
                continue
            node_idx = location_to_idx[target_site]

            for shape in INJECTION_SHAPES:
                injected_3d = apply_injection(
                    data_3d, node_idx, cond_idx, start_pos, shape, magnitude
                )
                normalized = tad._normalize(injected_3d, scaler, feature_cols, location_to_idx)
                predictions, targets = run_window_model(
                    model, edge_index, normalized, node_mask, num_nodes
                )
                if predictions is None:
                    continue

                flagged, rule1, rule2 = _score_node(
                    predictions,
                    targets,
                    node_idx,
                    cond_idx,
                    target_site,
                    target_ts,
                    rain_series,
                    metadata,
                )
                key = (target_site, shape)
                d = detection.setdefault(key, {"tested": 0, "flagged": 0, "rule1": 0, "rule2": 0})
                d["tested"] += 1
                if flagged:
                    d["flagged"] += 1
                if rule1["flagged"]:
                    d["rule1"] += 1
                if rule2["flagged"]:
                    d["rule2"] += 1

                loc = localization.setdefault(
                    target_site,
                    {
                        "any_injections": 0,
                        "any_neighbor_checks": 0,
                        "any_neighbor_false_flags": 0,
                        "rule2_injections": 0,
                        "rule2_neighbor_checks": 0,
                        "rule2_neighbor_false_flags": 0,
                    },
                )
                loc["any_injections"] += 1
                if rule2["flagged"]:
                    loc["rule2_injections"] += 1
                for nb_site in neighbor_sites:
                    nb_idx = location_to_idx[nb_site]
                    nb_flagged, nb_rule1, nb_rule2 = _score_node(
                        predictions,
                        targets,
                        nb_idx,
                        cond_idx,
                        nb_site,
                        target_ts,
                        rain_series,
                        metadata,
                    )
                    loc["any_neighbor_checks"] += 1
                    if nb_flagged:
                        loc["any_neighbor_false_flags"] += 1
                    if rule2[
                        "flagged"
                    ]:  # only counts toward the rule2-specific stat when rule2 fired on the target
                        loc["rule2_neighbor_checks"] += 1
                        if nb_flagged:
                            loc["rule2_neighbor_false_flags"] += 1

    print(
        "3c. DETECTION: combined (Rule1 OR Rule2), per node per shape -- rule breakdown in parens"
    )
    print("-" * 80)
    print(f"{'node':15s} {'step':>22s} {'ramp':>22s} {'spike':>22s}")
    for site in node_names:
        cells = []
        for shape in INJECTION_SHAPES:
            d = detection.get((site, shape))
            if d is None or d["tested"] == 0:
                cells.append("no data")
            else:
                rate = d["flagged"] / d["tested"]
                cells.append(
                    f"{d['flagged']}/{d['tested']} ({rate:.0%}, r1={d['rule1']} r2={d['rule2']})"
                )
        print(f"{site:15s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")

    print(
        "\n3d. LOCALIZATION: clean neighbor false-flag rate, any injection vs specifically when Rule 2 fired"
    )
    print("-" * 80)
    print(
        f"{'injected node':15s} {'any: checks':>12s} {'any: false':>11s} {'any rate':>9s}   "
        f"{'rule2: checks':>13s} {'rule2: false':>13s} {'rule2 rate':>10s}"
    )
    for site in node_names:
        loc = localization.get(site)
        if loc is None or loc["any_neighbor_checks"] == 0:
            continue
        any_rate = loc["any_neighbor_false_flags"] / loc["any_neighbor_checks"]
        r2_rate = (
            loc["rule2_neighbor_false_flags"] / loc["rule2_neighbor_checks"]
            if loc["rule2_neighbor_checks"] > 0
            else float("nan")
        )
        r2_checks_str = (
            str(loc["rule2_neighbor_checks"])
            if loc["rule2_neighbor_checks"] > 0
            else "0 (rule2 never fired)"
        )
        print(
            f"{site:15s} {loc['any_neighbor_checks']:>12d} {loc['any_neighbor_false_flags']:>11d} {any_rate:>9.1%}   "
            f"{r2_checks_str:>13s} {loc['rule2_neighbor_false_flags']:>13d} "
            f"{(f'{r2_rate:.1%}' if loc['rule2_neighbor_checks'] > 0 else 'n/a'):>10s}"
        )

    return detection, localization


def section4_livepath(
    model,
    metadata,
    edge_index,
    nodes=SECTION4_NODES,
    magnitudes=SECTION4_MAGNITUDES,
    n_windows=N_SECTION3_WINDOWS,
):
    print("=" * 70)
    print("SECTION 4 (live path): step-shape sensitivity sweep, Rule 1 vs Rule 2 vs combined")
    print("=" * 70)

    location_to_idx = metadata["location_to_idx"]
    feature_cols = metadata["feature_cols"]
    scaler = metadata["scaler"]
    node_names = list(location_to_idx.keys())
    num_nodes = len(node_names)
    cond_idx = feature_cols.index("conductivity")

    corpus = load_corpus()
    raw_cond, qc_valid = build_real_observation_lookups(node_names)
    held_start, held_end = held_out_range(corpus)
    all_windows = build_clean_windows(corpus, held_start, held_end)
    test_windows = all_windows[:: max(1, len(all_windows) // n_windows)][:n_windows]
    print(f"using {len(test_windows)} of {len(all_windows)} clean windows, nodes={nodes}\n")

    results = {site: {} for site in nodes}

    for window_ts in test_windows:
        data_3d, node_mask, rain_series = build_window_arrays(
            corpus, window_ts, feature_cols, location_to_idx, raw_cond, qc_valid
        )
        target_ts = window_ts[Config.SEQUENCE_LENGTH :]
        start_pos = Config.SEQUENCE_LENGTH

        for site in nodes:
            if site not in location_to_idx or not node_fully_real(
                target_ts, site, raw_cond, qc_valid
            ):
                continue
            node_idx = location_to_idx[site]

            for magnitude in magnitudes:
                injected_3d = apply_injection(
                    data_3d, node_idx, cond_idx, start_pos, "step", magnitude
                )
                normalized = tad._normalize(injected_3d, scaler, feature_cols, location_to_idx)
                predictions, targets = run_window_model(
                    model, edge_index, normalized, node_mask, num_nodes
                )
                if predictions is None:
                    continue
                flagged, rule1, rule2 = _score_node(
                    predictions, targets, node_idx, cond_idx, site, target_ts, rain_series, metadata
                )
                d = results[site].setdefault(
                    magnitude, {"tested": 0, "flagged": 0, "rule1": 0, "rule2": 0}
                )
                d["tested"] += 1
                if flagged:
                    d["flagged"] += 1
                if rule1["flagged"]:
                    d["rule1"] += 1
                if rule2["flagged"]:
                    d["rule2"] += 1

    print(f"{'magnitude (uS/cm)':>18s}" + "".join(f"{site:>28s}" for site in nodes))
    for magnitude in magnitudes:
        row = f"{magnitude:>18.0f}"
        for site in nodes:
            d = results[site].get(magnitude)
            if d is None or d["tested"] == 0:
                row += f"{'no data':>28s}"
                continue
            rate = d["flagged"] / d["tested"]
            cell = f"{d['flagged']}/{d['tested']} ({rate:.0%}, r1={d['rule1']} r2={d['rule2']})"
            row += f"{cell:>28s}"
        print(row)

    return results


def before_after(section2_results, section3_detection):
    print("=" * 70)
    print("BEFORE / AFTER")
    print("=" * 70)

    print("\nFalse positive rate on clean data")
    print(f"{'node':15s} {'before (test path)':>20s} {'after (live path)':>20s} {'delta':>8s}")
    for site in ("north_fork_0", "south_fork_2", "south_fork_1", "oxford"):
        before = PRIOR_FP_RATE.get(site)
        r = section2_results.get(site)
        after = r["n_flagged"] / r["n_eval"] if r and r["n_eval"] > 0 else None
        before_str = f"{before:.1%}" if before is not None else "n/a"
        after_str = f"{after:.1%}" if after is not None else "n/a"
        delta_str = (
            f"{(after - before):+.1%}" if before is not None and after is not None else "n/a"
        )
        print(f"{site:15s} {before_str:>20s} {after_str:>20s} {delta_str:>8s}")

    print("\nStep detection at 150 uS/cm")
    print(f"{'node':15s} {'before (Rule1 only)':>20s} {'after (Rule1 OR Rule2)':>24s}")
    for site in ("north_fork_0", "south_fork_2", "south_fork_1", "oxford"):
        before = PRIOR_STEP_DETECTION_150.get(site)
        d = section3_detection.get((site, "step"))
        after = d["flagged"] / d["tested"] if d and d["tested"] > 0 else None
        before_str = f"{before:.0%}" if before is not None else "n/a"
        after_str = f"{after:.0%}" if after is not None else "n/a"
        print(f"{site:15s} {before_str:>20s} {after_str:>24s}")


def main():
    model, metadata = load_model_and_metadata()
    edge_index, _, _ = create_graph_topology()
    s2 = section2_livepath(model, metadata, edge_index)
    s3_detection, s3_localization = section3_livepath(model, metadata, edge_index)
    section4_livepath(model, metadata, edge_index)
    before_after(s2, s3_detection)


if __name__ == "__main__":
    main()

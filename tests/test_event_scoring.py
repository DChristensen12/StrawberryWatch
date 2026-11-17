"""Event scoring, the fixture fingerprints, and one window builder for training and serving."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import paths
from strawberrywatch.anomalies import event_scoring as es
from strawberrywatch.ingest import fixtures


def _steps(n=96):
    return pd.date_range("2026-06-01", periods=n, freq="15min", tz="UTC")


# Scoring


def test_events_numbers_each_run_and_merges_close_ones():
    flags = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert es.events(flags).tolist() == [0, 1, 1, 0, 0, 2, 0, 3, 0, 0, 0, 4]
    assert es.events(flags, gap=1).tolist() == [0, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 3]
    assert es.events(np.zeros(4, dtype=bool)).tolist() == [0, 0, 0, 0]


def test_each_event_counts_once_however_long_it_is():
    labelled = np.zeros(40, dtype=bool)
    labelled[5:25] = True
    labelled[30:33] = True
    detected = np.zeros(40, dtype=bool)
    detected[10] = True
    detected[36:38] = True
    classes = es.compare_events(es.events(labelled), es.events(detected))
    counts = es.event_metrics(classes)
    assert (counts["true_positives"], counts["false_positives"], counts["false_negatives"]) == (
        1,
        1,
        1,
    )
    assert counts["precision"] == 0.5 and counts["recall"] == 0.5


def test_one_detection_does_not_mark_the_whole_event_detected():
    """No point adjustment: range recall stays low when one step of a long event is caught."""
    labelled = np.zeros(100, dtype=bool)
    labelled[20:60] = True
    detected = np.zeros(100, dtype=bool)
    detected[30] = True
    precision, recall = es.range_scores(labelled, detected)
    assert precision == 1.0 and recall == pytest.approx(1 / 40)
    assert np.isnan(es.range_scores(labelled, np.zeros(100, dtype=bool))[1])


def test_caught_says_how_long_it_took():
    times = _steps()
    over = np.zeros(96, dtype=bool)
    over[[2, 30, 31]] = True
    start, end = times[24], times[48]
    assert es.caught(times, over, start, end) == (True, 90.0)
    assert es.caught(times, over, times[60], times[70]) == (False, None)


def test_false_alarms_are_events_per_clean_week():
    week = 7 * 96
    over = np.zeros(2 * week, dtype=bool)
    over[[10, 11, 12, 500, 900]] = True
    clean = np.ones(2 * week, dtype=bool)
    assert es.false_alarms_per_week(over, clean) == 1.5
    # an alarm inside a labelled stretch isn't a false one
    clean[890:910] = False
    assert es.false_alarms_per_week(over, clean) == pytest.approx(2 / (2 - 20 / week))
    assert np.isnan(es.false_alarms_per_week(over, np.zeros(2 * week, dtype=bool)))


def test_score_puts_it_together():
    times = _steps(200)
    over = np.zeros(200, dtype=bool)
    over[[52, 53, 150]] = True
    out = es.score(times, over, [(times[50], times[70]), (times[100], times[110])])
    assert [e["caught"] for e in out["events"]] == [True, False]
    assert out["events"][0]["minutes"] == 30.0
    assert (out["true_positives"], out["false_positives"], out["false_negatives"]) == (1, 1, 1)
    assert out["false_alarms_per_week"] > 0 and 0 < out["range_recall"] < 1


# Fixture fingerprints


def test_no_fixture_changed_without_its_fingerprint():
    assert fixtures.changed() == [], (
        "these fixtures differ from data/anomalies/fixtures.sha256.json. If the change is "
        "meant, run python -m strawberrywatch.ingest.fixtures"
    )


def test_the_replay_copies_are_the_same_files():
    """The dashboard replays copies of the fixtures, and they carry the same fingerprints."""
    from strawberrywatch.watch import sources

    assert fixtures.fingerprints(sources.EVENTS_DIR) == fixtures.recorded()


def test_a_changed_fixture_is_named_until_it_is_recorded(tmp_path):
    folder = tmp_path / "anomaly_x"
    folder.mkdir()
    (folder / "oxford.csv").write_text("DateTimeUTC,Meter_Hydros21_Cond\n2026-01-01 00:00:00,350\n")
    assert fixtures.changed(tmp_path) == ["anomaly_x/oxford.csv"]
    assert (
        fixtures.record(tmp_path) == ["anomaly_x/oxford.csv"] and fixtures.changed(tmp_path) == []
    )
    (folder / "oxford.csv").write_text("DateTimeUTC,Meter_Hydros21_Cond\n2026-01-01 00:00:00,351\n")
    assert fixtures.changed(tmp_path) == ["anomaly_x/oxford.csv"]
    assert set(json.loads((tmp_path / fixtures.MANIFEST).read_text())) == {"anomaly_x/oxford.csv"}


# One window builder


def _raw_fixture():
    """The June 2025 spill as raw tables, and its weather."""
    from strawberrywatch.watch import sources

    replay = sources.ReplaySource("anomaly_2025_06_12_spill_sf")
    start, end = replay.window(None)
    return replay.tables(list(replay._csv._all()), start, end), replay.weather(start, end)


def test_training_and_serving_build_identical_windows_from_one_fixture():
    """Breck et al. 2017: training and serving features compute the same values."""
    from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized
    from strawberrywatch.serving import detector as det
    from strawberrywatch.serving import detector_class, windows

    folder = paths.model_dir("dusk_crayfish")
    metadata_path = folder / "dusk_crayfish_metadata.pkl"
    if not metadata_path.exists():
        pytest.skip("no dusk_crayfish_metadata.pkl, so there is no trained scaler to reuse")
    detector = detector_class("dusk_crayfish").load(folder)
    tables, weather = _raw_fixture()

    # The frame serving builds: stacked, weather merged, absent features filled, clock added
    frame = detector._frame({s: tables[s] for s in detector.sites if s in tables})
    frame, _ = det._merge_weather(frame, detector.features, weather)
    frame = det._fill_absent(
        frame[[c for c in detector.features if c in frame.columns] + ["location"]],
        detector.features,
        detector._normalization,
    )
    frame = windows.add_time_features(frame)
    serving = windows.build_windows(
        frame,
        detector.checkpoint.feature_cols,
        detector.checkpoint.location_to_idx,
        detector._normalization,
        detector.window,
        detector.imputation_limit_hours,
        detector.grid,
    )

    # Training's path over the same readings, on the grid the corpus is built on
    gridded = windows.regrid(frame, detector.checkpoint.feature_cols, "15min")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    sequences, targets, stamps, _, cols, node_mask, _ = prepare_sequences_normalized(
        gridded,
        detector.checkpoint.location_to_idx,
        detector.window,
        return_node_mask=True,
        scaler=metadata["scaler"],
        scaler_feature_cols=metadata["feature_cols"],
    )
    assert cols == list(detector.checkpoint.feature_cols)
    assert len(stamps) > 100 and list(stamps) == list(serving[2])
    assert np.array_equal(sequences.astype(np.float32), serving[0])
    assert np.array_equal(targets.astype(np.float32), serving[1])
    assert np.array_equal(node_mask, serving[3])


def test_the_training_corpus_arrays_are_the_ones_the_shipped_model_trained_on():
    """Fingerprints of prepare_sequences_normalized on the shipped corpus, taken before the merge."""
    import contextlib
    import hashlib
    import io

    from strawberrywatch.config import Config
    from strawberrywatch.ingest.data_loader import load_and_preprocess_data
    from strawberrywatch.preprocessing.data_processor import prepare_sequences_normalized
    from strawberrywatch.utils.graph_utils import create_graph_topology

    corpus = paths.training_corpus()
    if not corpus.is_file():
        pytest.skip("no training_corpus.csv here")
    with contextlib.redirect_stdout(io.StringIO()):
        df_featured, _, _ = load_and_preprocess_data(file_path=str(corpus))
        _, _, location_to_idx = create_graph_topology(announce=False)
        sequences, targets, stamps, _, _, node_mask, target_mask = prepare_sequences_normalized(
            df_featured, location_to_idx, Config.SEQUENCE_LENGTH, return_node_mask=True
        )

    def digest(array):
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]

    assert sequences.shape == (29099, 24, 4, 10) and sequences.dtype == np.float64
    assert digest(sequences) == "898fb20c1ae5e524"
    assert digest(targets) == "d12b046d61e48140"
    assert digest(node_mask) == "4562c0a65e1bb79f"
    assert digest(target_mask) == "650c635fbc972a7e"

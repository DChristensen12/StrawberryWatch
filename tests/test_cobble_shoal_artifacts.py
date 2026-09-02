"""
Cobble Shoal's shipped artifacts, loaded and scored.

The rename moved four filenames at once and nothing in the package loads the
weights, so a broken artifact name would otherwise surface late. Skips when the
checkpoints are absent, same as conftest.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from strawberrywatch.anomalies import cobble_calibration
from tests.synthetic import nested_batch


@pytest.fixture(scope="module")
def calibration():
    try:
        return cobble_calibration.load_calibration(cobble_calibration.SYNTHETIC)
    except cobble_calibration.CalibrationError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def real_calibration():
    try:
        return cobble_calibration.load_calibration(cobble_calibration.REAL)
    except cobble_calibration.CalibrationError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def model(calibration):
    path = cobble_calibration.weights_path()
    if not path.exists():
        pytest.skip(f"no weights at {path}")
    net = nested_batch.build_model({"seed": calibration.seed, "window": nested_batch.WINDOW})
    blob = torch.load(path, map_location="cpu", weights_only=True)
    # The training harness saved {"state_dict", "meta"} rather than a bare
    # state dict, and nothing in the package unwraps it yet.
    net.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
    net.eval()
    return net


def test_the_weights_file_matches_the_class(model):
    """load_state_dict is strict, so reaching here means no key moved."""
    assert model.window == nested_batch.WINDOW
    assert model.channel_names == ("excursion", "loo", "reconstruction", "dispersion")


def test_calibration_carries_a_usable_threshold(calibration):
    assert np.isfinite(calibration.z_q)
    assert calibration.threshold_at(calibration.operating_q) == pytest.approx(calibration.z_q)


def test_scoring_gives_one_finite_number_per_node(model, calibration):
    """
    The whole contract the alerting path needs: a score per node, and no
    channel visible from here.
    """
    batch = nested_batch.build_batch()
    keys = nested_batch.node_keys()

    combined = model.score(batch, calibration.nulls)
    assert combined.shape == (1, len(keys))
    assert np.all(np.isfinite(combined)), "a node scored NaN on a clean window"

    combined_again, channels = model.score(batch, calibration.nulls, return_channels=True)
    assert combined_again.tobytes() == combined.tobytes(), "score is not deterministic"
    assert set(channels) == set(model.channel_names)


def test_a_faulted_node_outscores_the_clean_window(model, calibration):
    """
    Not a detection claim, just evidence the score responds to the input at
    all. A frozen node against the same window it was frozen out of.
    """
    from tests.synthetic import creek_synthetic

    clean = nested_batch.build_batch()
    win = creek_synthetic.make_synthetic_creek(
        n_steps=nested_batch.ANCHOR + nested_batch.WINDOW, seed=0
    )
    values, target, truth = creek_synthetic.inject(
        win, 0, "stuck", lo=80, hi=nested_batch.ANCHOR + 1, anchor=nested_batch.ANCHOR
    )
    faulted = nested_batch.build_batch(values=values, target=target)

    before = model.score(clean, calibration.nulls).ravel()[truth["nodes"][0]]
    after = model.score(faulted, calibration.nulls).ravel()[truth["nodes"][0]]
    assert after > before, f"freezing {truth['node_keys'][0]} did not raise its score"


def test_the_real_calibration_carries_the_scaler_its_nulls_were_fitted_in(real_calibration):
    """
    The whole point of recording it. A null is fitted in a normalization space
    and only means anything applied in that space, so the space has to travel
    with the null rather than being refitted by whoever scores next.
    """
    scaler = real_calibration.window_scaler()
    assert len(scaler.mean) == len(nested_batch.node_keys())
    assert len(scaler.std) == len(nested_batch.node_keys())
    assert np.all(scaler.std > 0), "a zero scale would divide every reading to infinity"
    assert real_calibration.is_real


def test_the_synthetic_calibration_refuses_to_supply_a_scaler(calibration):
    """
    It never recorded one, and returning None would let build_window fit a
    fresh scaler and score real data against nulls from another space. That
    failure produces finite, plausible numbers, so it has to raise.
    """
    assert calibration.node_scaler is None
    assert not calibration.is_real
    with pytest.raises(cobble_calibration.CalibrationError, match="no node_scaler"):
        calibration.window_scaler()


def test_the_two_calibrations_are_not_interchangeable(calibration, real_calibration, model):
    """
    Close thresholds, different score scales. This is why load_calibration
    takes the filename rather than picking one, and why nothing falls back
    from one to the other.
    """
    batch = nested_batch.build_batch()
    synthetic = model.score(batch, calibration.nulls).ravel()
    real = model.score(batch, real_calibration.nulls).ravel()

    assert abs(calibration.z_q - real_calibration.z_q) < 5.0, "thresholds drifted apart"
    assert synthetic.max() > 5 * real.max(), (
        "the two nulls now agree about scale; if that is genuine, the warning in "
        "cobble_calibration's docstring needs rewriting"
    )


def test_loading_an_absent_calibration_names_the_file(tmp_path):
    with pytest.raises(cobble_calibration.CalibrationError, match="no calibration artifact"):
        cobble_calibration.load_calibration(cobble_calibration.REAL, checkpoint_dir=tmp_path)

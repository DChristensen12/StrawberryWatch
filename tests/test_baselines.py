"""
The model-free baselines, and the properties the comparison rests on.

These detectors are a fair floor, so most of the tests are about fairness: same
input, same step, same null path, no lookahead. The two blind spots, a frozen
node and a reflected one, get asserted rather than described. They are the
finding, and a quiet regression into "it catches everything" would erase it.
"""

from __future__ import annotations

import numpy as np
import pytest

from strawberrywatch.anomalies import baselines as bl
from strawberrywatch.anomalies import channel_scoring as scoring
from tests.synthetic import baseline_sweep as bs
from tests.synthetic import creek_synthetic as cs


@pytest.fixture(scope="module")
def calibrated():
    """One null and threshold per detector, fitted on fault-free windows only."""
    batches = [bs.clean_batch(s) for s in range(1000, 1120)]
    out = {}
    for name, cls in bl.BASELINES.items():
        det = cls()
        nulls, pot = bl.calibrate(det, batches, q=1e-3)
        out[name] = (det, nulls, pot)
    return out


# The chart arithmetic


def test_the_rolling_median_is_trailing_and_never_reads_ahead():
    v = np.zeros((12, 1))
    v[7:] = 100.0
    med = bl.rolling_median(v, window=4)
    assert np.all(med[:7] == 0.0), "a step before the jump already sees it"
    assert med[-1] == 100.0, "the median never caught up to the new level"


def test_the_median_window_is_shorter_than_the_scored_window():
    """Otherwise it is an expanding median with no history behind the early steps."""
    assert bl.MEDIAN_WINDOW < bs.WINDOW


def test_an_unobserved_step_contributes_nothing():
    # A one-step excursion at the end, so the trailing median has not absorbed
    # it. A sustained shift would read zero either way once the median catches
    # up, which is the chart working, not the mask.
    v = np.zeros((12, 1))
    v[3] = 1.0
    v[-1] = 50.0
    seen = np.ones((12, 1), dtype=bool)
    blind = seen.copy()
    blind[-1] = False
    assert bl.standardised_residual(v, seen, window=4)[-1, 0] != 0.0
    assert bl.standardised_residual(v, blind, window=4)[-1, 0] == 0.0


def test_cusum_accumulates_a_sustained_shift_and_ewma_tracks_it():
    x = np.zeros((40, 1))
    x[20:] = 2.0
    cusum = bl.cusum_path(x)
    ewma = bl.ewma_path(x)
    assert cusum[19, 0] == 0.0 and cusum[-1, 0] > cusum[25, 0] > 0.0, "CUSUM did not accumulate"
    assert ewma[-1, 0] > ewma[19, 0], "EWMA did not respond"


def test_cusum_is_two_sided():
    up, down = np.zeros((30, 1)), np.zeros((30, 1))
    up[10:] = 2.0
    down[10:] = -2.0
    assert bl.cusum_path(up)[-1, 0] == pytest.approx(bl.cusum_path(down)[-1, 0])


def test_ewma_early_steps_are_not_inflated_by_a_small_denominator():
    """Exact time-varying variance, not the asymptote."""
    x = np.random.default_rng(0).normal(size=(200, 1))
    path = bl.ewma_path(x)
    assert path[0, 0] < 4.0, "step one is already off the chart"


# Fairness of the comparison


def test_the_chart_scores_the_same_step_the_models_score():
    """batch['target'], where the spike lands, rather than the window's last row."""
    batch = bs.clean_batch(0)
    values, observed = bl.series_with_target(batch)
    assert values.shape[0] == np.asarray(batch["values"]).shape[1] + 1
    assert observed.shape == values.shape
    np.testing.assert_array_equal(observed[-1], np.asarray(batch["target_mask"])[0])


def test_no_target_carries_the_last_value_forward_rather_than_a_nan():
    batch = dict(bs.clean_batch(0))
    mask = np.asarray(batch["target_mask"]).copy()
    mask[:] = False
    batch["target_mask"] = mask
    values, observed = bl.series_with_target(batch)
    assert np.all(np.isfinite(values)), "a NaN target poisoned the series"
    assert not observed[-1].any()


def test_the_detector_interface_matches_the_models():
    """Duck typed on purpose, so one harness scores a chart and a network alike."""
    from strawberrywatch.models.Cobble_Shoal import CobbleShoal

    for cls in bl.BASELINES.values():
        for name in ("score", "channels", "channel_names", "INPUT_CONTRACT"):
            assert hasattr(cls, name), f"{cls.__name__} is missing {name}"
        assert cls.INPUT_CONTRACT == CobbleShoal.INPUT_CONTRACT


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_scoring_gives_one_finite_number_per_node(calibrated, name):
    det, nulls, _pot = calibrated[name]
    batch = bs.clean_batch(7)
    scores = det.score(batch, nulls)
    assert scores.shape == (1, len(batch["nodes"]))
    assert np.all(np.isfinite(scores))
    again = det.score(batch, nulls)
    assert again.tobytes() == scores.tobytes(), "score is not deterministic"


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_calibration_fits_a_real_tail(calibrated, name):
    _det, _nulls, pot = calibrated[name]
    assert pot["fitted"], "the GPD fit failed, so the threshold is a fallback"
    assert np.isfinite(pot["threshold"]) and pot["threshold"] > pot["u"]


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_nothing_is_fitted_at_scoring_time(calibrated, name):
    """A detector that refits on the window it judges has no null."""
    det, nulls, _pot = calibrated[name]
    before = [n.samples.copy() for n in nulls.nulls.values()]
    det.score(bs.clean_batch(11), nulls)
    for null, snapshot in zip(nulls.nulls.values(), before, strict=True):
        np.testing.assert_array_equal(null.samples, snapshot)


# What they catch, and what they cannot


def _faulted(seed, node, shape, **kw):
    win = bs._window(seed)
    values, target, truth = cs.inject(win, node, shape, seed=seed, **kw)
    return bs.faulted_batch(seed, values, target), truth


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_a_spike_on_the_scored_step_outscores_the_clean_window(calibrated, name):
    det, nulls, _pot = calibrated[name]
    node = 0
    win = bs._window(0)
    clean = det.score(bs.clean_batch(0), nulls).ravel()[node]
    batch, _truth = _faulted(0, node, "spike", mag=cs.node_magnitude(win, node, 2.0))
    assert det.score(batch, nulls).ravel()[node] > clean


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_a_frozen_node_is_invisible_to_a_residual_chart(calibrated, name):
    """
    The finding, asserted. A stuck sensor has no residual against its own
    trailing median, so a residual chart scores it below a healthy node instead
    of above one. Dispersion catches this and a chart has no dispersion channel.
    If this starts failing, the baseline grew a second channel.
    """
    det, nulls, _pot = calibrated[name]
    node = 0
    clean = det.score(bs.clean_batch(0), nulls).ravel()[node]
    batch, _truth = _faulted(0, node, "stuck")
    assert det.score(batch, nulls).ravel()[node] <= clean


@pytest.mark.parametrize("name", sorted(bl.BASELINES))
def test_a_reflected_node_is_invisible_too(calibrated, name):
    """
    decouple preserves mean and variance and only inverts the correlation with
    the node's siblings, which a marginal chart never looks at.
    """
    det, nulls, _pot = calibrated[name]
    node = 0
    batch, _truth = _faulted(0, node, "decouple")
    scores = det.score(batch, nulls).ravel()
    assert bs.rank_of_truth(scores, [node]) > 1


# The harness itself


def test_rank_of_truth_is_one_based_and_takes_the_best_placed_node():
    scores = np.array([5.0, 9.0, 1.0])
    assert bs.rank_of_truth(scores, [1]) == 1
    assert bs.rank_of_truth(scores, [0]) == 2
    assert bs.rank_of_truth(scores, [2]) == 3
    assert bs.rank_of_truth(scores, [0, 2]) == 2


def test_the_sweep_covers_every_shape_and_every_node():
    truths = [t for _batch, t in bs.cases(0)]
    assert {t["shape"] for t in truths} == set(cs.SHAPES)
    single = {t["nodes"][0] for t in truths if t["shape"] == "stuck"}
    assert single == set(range(17)), "not every node gets faulted"


def test_the_cached_window_matches_a_freshly_generated_one():
    """The cache is a speedup. Same creek."""
    fresh = cs.make_synthetic_creek(n_steps=bs.N_STEPS, seed=3)
    np.testing.assert_array_equal(fresh["values"], bs._window(3)["values"])


def test_the_batch_matches_the_shared_builder():
    """baseline_sweep builds its own batch for speed. Has to be the same batch."""
    import torch

    from tests.synthetic import nested_batch

    mine = bs.clean_batch(0)
    theirs = nested_batch.build_batch(seed=0, window=bs.WINDOW, anchor=bs.ANCHOR)
    for key in ("values", "staleness", "context", "target", "target_mask", "obs_mask"):
        assert torch.equal(mine[key], theirs[key]), key


def test_channel_scores_go_through_the_same_combine_the_models_use(calibrated):
    det, nulls, _pot = calibrated["cusum"]
    batch = bs.clean_batch(0)
    chans = det.channels(batch)
    expected = scoring.combine(pvalues=nulls.pvalues(chans), rule="fisher")
    np.testing.assert_array_equal(det.score(batch, nulls), expected)

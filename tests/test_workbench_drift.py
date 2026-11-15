"""The drift rule: Wasserstein distance over the reference spread, at least 1,000 readings a side."""

import numpy as np

from strawberrywatch.workbench import checks


def test_same_distribution_is_no_drift():
    rng = np.random.default_rng(0)
    reference = rng.normal(500, 20, 5000)
    current = rng.normal(500, 20, 2880)
    moved, ratio = checks.drifted(current, reference)
    assert not moved and ratio < checks.DRIFT_RATIO


def test_half_a_standard_deviation_is_drift():
    rng = np.random.default_rng(0)
    reference = rng.normal(500, 20, 5000)
    current = rng.normal(510, 20, 2880)
    moved, ratio = checks.drifted(current, reference)
    assert moved and ratio > checks.DRIFT_RATIO


def test_too_few_readings_gives_no_result():
    rng = np.random.default_rng(0)
    assert checks.drifted(rng.normal(0, 1, 999), rng.normal(0, 1, 5000)) is None
    assert checks.drifted(rng.normal(0, 1, 5000), rng.normal(0, 1, 999)) is None


def test_reference_sample_is_capped_and_repeatable():
    values = np.arange(20000, dtype=float)
    first = checks.reference_sample(values)
    assert first.size == checks.REFERENCE_SIZE
    np.testing.assert_array_equal(first, checks.reference_sample(values))
    assert checks.reference_sample(values[:100]).size == 100

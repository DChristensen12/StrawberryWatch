"""
Detection channels and their nulls: does each statistic depend on its input,
and does each mask actually mask.

Both classes have bitten here before. A statistic that returns the same number
whatever you feed it looks like a working measurement for weeks, and the
dispersion channel once scored ordinary sensor outages as frozen sensors
because its mask was optional and nobody passed it.
"""

import numpy as np

from strawberrywatch.anomalies import channel_scoring as scoring
from tests.audit_helpers import differs


def test_excursion_varies():
    tgt = np.array([[1.0, 2.0]])
    pred_a = np.array([[1.0, 2.0]])
    pred_b = np.array([[3.0, 2.0]])
    scale = np.ones((1, 2))
    m = np.ones((1, 2), bool)
    differs(
        scoring.excursion(pred_a, scale, tgt, m, m),
        scoring.excursion(pred_b, scale, tgt, m, m),
        "excursion vs prediction",
    )
    differs(
        scoring.excursion(pred_b, scale, tgt, m, m),
        scoring.excursion(pred_b, scale * 4.0, tgt, m, m),
        "excursion vs scale",
    )


def test_loo_varies():
    tgt = np.array([[1.0, 2.0]])
    m = np.ones((1, 2), bool)
    differs(
        scoring.loo(np.array([[1.0, 2.0]]), np.ones((1, 2)), tgt, m),
        scoring.loo(np.array([[5.0, 2.0]]), np.ones((1, 2)), tgt, m),
        "loo",
    )


def test_reconstruction_varies():
    vals = np.zeros((1, 4, 2))
    m = np.ones((1, 4, 2), bool)
    xh_a = np.zeros((1, 2, 4))
    xh_b = np.ones((1, 2, 4)) * 3.0
    sc = np.ones((1, 2, 4))
    differs(
        scoring.reconstruction(xh_a, sc, vals, m),
        scoring.reconstruction(xh_b, sc, vals, m),
        "reconstruction",
    )


def test_dispersion_varies():
    flat = np.ones((1, 8, 2))
    wiggly = np.random.default_rng(0).normal(size=(1, 8, 2)) * 5.0
    scale = np.ones((1, 2))
    differs(
        scoring.dispersion(scale, flat),
        scoring.dispersion(scale, wiggly),
        "dispersion vs observed movement",
    )
    differs(
        scoring.dispersion(scale, flat),
        scoring.dispersion(scale * 9.0, flat),
        "dispersion vs predicted scale",
    )


def test_gpd_and_pot_vary():
    rng = np.random.default_rng(1)
    light = rng.exponential(1.0, 4000)
    heavy = rng.exponential(3.0, 4000)
    differs([scoring.fit_gpd(light)[0]], [scoring.fit_gpd(heavy)[0]], "fit_gpd sigma")
    differs([scoring.fit_pot(light, q=1e-3)], [scoring.fit_pot(heavy, q=1e-3)], "fit_pot threshold")
    differs(
        scoring.gpd_sf(np.array([2.0]), 1.0, 1.0, 0.0),
        scoring.gpd_sf(np.array([2.0]), 1.0, 2.0, 0.0),
        "gpd_sf vs sigma",
    )


def test_channel_null_pvalues_vary():
    null = scoring.ChannelNull().fit(np.random.default_rng(2).normal(size=3000))
    differs(
        null.to_pvalue(np.array([0.0])), null.to_pvalue(np.array([3.0])), "ChannelNull.to_pvalue"
    )


def test_combine_varies():
    pv_a = {"x": np.array([0.5]), "y": np.array([0.5])}
    pv_b = {"x": np.array([1e-6]), "y": np.array([0.5])}
    differs(
        scoring.combine(pvalues=pv_a, rule="fisher"),
        scoring.combine(pvalues=pv_b, rule="fisher"),
        "combine fisher",
    )
    differs(
        scoring.combine(raw={"x": np.array([1.0])}, rule="max"),
        scoring.combine(raw={"x": np.array([7.0])}, rule="max"),
        "combine max",
    )


def test_chi2_sf_varies():
    differs(
        scoring.chi2_sf_even(np.array([1.0]), 4),
        scoring.chi2_sf_even(np.array([20.0]), 4),
        "chi2_sf_even vs stat",
    )
    differs(
        scoring.chi2_sf_even(np.array([5.0]), 2),
        scoring.chi2_sf_even(np.array([5.0]), 8),
        "chi2_sf_even vs dof",
    )


def test_scoring_masks_are_live():
    tgt = np.array([[1.0, 2.0]])
    pred = np.array([[9.0, 9.0]])
    scale = np.ones((1, 2))
    on = np.ones((1, 2), bool)
    off = np.zeros((1, 2), bool)

    # excursion honours target_mask AND anchor_valid independently
    full = scoring.excursion(pred, scale, tgt, on, on)
    assert np.isfinite(full).all()
    assert np.isnan(scoring.excursion(pred, scale, tgt, off, on)).all(), (
        "excursion ignores target_mask"
    )
    assert np.isnan(scoring.excursion(pred, scale, tgt, on, off)).all(), (
        "excursion ignores anchor_valid"
    )

    assert np.isnan(scoring.loo(pred, scale, tgt, off)).all(), "loo ignores target_mask"


def test_reconstruction_mask_is_live():
    vals = np.zeros((1, 4, 2))
    xh = np.ones((1, 2, 4)) * 3.0
    sc = np.ones((1, 2, 4))
    on = np.ones((1, 4, 2), bool)
    off = np.zeros((1, 4, 2), bool)
    assert np.isfinite(scoring.reconstruction(xh, sc, vals, on)).all()
    assert np.isnan(scoring.reconstruction(xh, sc, vals, off)).all(), (
        "reconstruction ignores observed_mask"
    )
    # Partial masking must actually change the mean. The unmasked timesteps
    # carry a different error from the masked ones, so averaging over a subset
    # has to move the answer; if it does not, the mask is being ignored.
    half = on.copy()
    half[:, :2, :] = False
    mixed = np.zeros((1, 4, 2))
    mixed[:, :2, :] = 3.0  # only the masked-out timesteps differ
    differs(
        scoring.reconstruction(xh, sc, mixed, on),
        scoring.reconstruction(xh, sc, mixed, half),
        "reconstruction partial mask",
    )


def test_dispersion_mask_is_live():
    flat = np.ones((1, 8, 2))
    scale = np.ones((1, 2))
    on = np.ones((1, 8, 2), bool)
    off = np.zeros((1, 8, 2), bool)
    assert np.isfinite(scoring.dispersion(scale, flat, on)).all()
    assert np.isnan(scoring.dispersion(scale, flat, off)).all(), (
        "dispersion ignores obs_mask, the outage-reads-as-freeze bug"
    )
    # the min_obs boundary must actually bite
    half = on.copy()
    half[:, :5, :] = False  # 3 of 8 observed, below window//2 = 4
    assert np.isnan(scoring.dispersion(scale, flat, half)).all(), (
        "dispersion min_obs threshold is inert"
    )

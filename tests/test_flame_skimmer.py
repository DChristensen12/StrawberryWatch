# Unit tests for Flame Skimmer

import math

import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from strawberrywatch.anomalies import channel_scoring as scoring
from strawberrywatch.models.Flame_Skimmer import (
    SERVING_SAMPLES,
    BayesLinear,
    FlameSkimmer,
)
from tests.synthetic import nested_batch

SEED = 20260806


def _model(**metadata):
    return FlameSkimmer.from_metadata(
        {"seed": SEED, "window": nested_batch.WINDOW, "n_train": 1000, **metadata}
    )


@pytest.fixture(scope="module")
def batch():
    return nested_batch.build_batch(seed=0)


def _set_rho(model, rho):
    with torch.no_grad():
        for layer in model.head.layers:
            layer.rho_weight.fill_(rho)
            layer.rho_bias.fill_(rho)


def test_kl_equals_summed_torch_kl_divergence():
    """kl() against torch.distributions, summed over every weight and bias in the head."""
    model = _model(prior_sigma=0.7)
    with torch.no_grad():
        for layer in model.head.layers:
            # spread rho out so the check is not only at the flat init
            layer.rho_weight.uniform_(-6.0, 1.0)
            layer.rho_bias.uniform_(-6.0, 1.0)
    expected = 0.0
    for layer in model.head.layers:
        prior = Normal(0.0, 0.7)
        expected += kl_divergence(Normal(layer.mu_weight, layer.sigma_weight), prior).sum()
        expected += kl_divergence(Normal(layer.mu_bias, layer.sigma_bias), prior).sum()
    torch.testing.assert_close(model.kl(), expected, rtol=1e-5, atol=1e-3)


def test_kl_is_zero_at_the_prior():
    """mu = 0 and sigma = prior_sigma is the prior itself."""
    layer = BayesLinear(7, 5, prior_sigma=1.3)
    with torch.no_grad():
        layer.mu_weight.zero_()
        layer.mu_bias.zero_()
        # inverse softplus. sigma comes out at exactly prior_sigma
        layer.rho_weight.fill_(math.log(math.expm1(1.3)))
        layer.rho_bias.fill_(math.log(math.expm1(1.3)))
    assert abs(float(layer.kl().detach())) < 1e-5


def test_encoder_matches_cobble_shoal_at_same_seed():
    """Same seed, byte-identical encoder init. A difference in results is the head's."""
    flame = _model()
    cobble = nested_batch.build_model({"seed": SEED, "window": nested_batch.WINDOW})
    flame_state = flame.encoder.state_dict()
    cobble_state = cobble.encoder.state_dict()
    assert flame_state.keys() == cobble_state.keys()
    for key in flame_state:
        assert torch.equal(flame_state[key], cobble_state[key]), key


def test_total_is_aleatoric_plus_epistemic(batch):
    model = _model().eval()
    with torch.no_grad():
        pred = model.predictive(batch)
    assert torch.allclose(pred.total, pred.aleatoric + pred.epistemic, atol=1e-6)
    assert bool((pred.epistemic >= 0).all())


def test_predictive_is_reproducible_with_same_generator(batch):
    model = _model().eval()
    with torch.no_grad():
        first = model.predictive(batch, samples=16, generator=torch.Generator().manual_seed(3))
        second = model.predictive(batch, samples=16, generator=torch.Generator().manual_seed(3))
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_serving_noise_is_fixed_across_calls_and_instances(batch):
    """No generator means the fixed serving noise. The same window always scores the same."""
    first = _model().eval()
    second = _model().eval()
    with torch.no_grad():
        a = first.predictive(batch)
        b = first.predictive(batch)
        c = second.predictive(batch)
    for x, y, z in zip(a, b, c, strict=True):
        assert torch.equal(x, y)
        assert torch.equal(x, z)
    with pytest.raises(ValueError):
        first.predictive(batch, samples=SERVING_SAMPLES + 1)


def test_window_score_does_not_depend_on_batch_neighbours(batch):
    """Weight space sampling at prediction. Stacking windows doesn't change any one window."""
    model = _model().eval()
    other = nested_batch.build_batch(seed=1)
    stacked = {
        k: torch.cat([batch[k], other[k]])
        if k in ("values", "staleness", "context", "target", "target_mask", "obs_mask")
        else batch[k]
        for k in batch
    }
    with torch.no_grad():
        alone = model.predictive(batch)
        together = model.predictive(stacked)
    for a, b in zip(alone, together, strict=True):
        torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)


def test_tiny_rho_reduces_to_deterministic_heteroscedastic_forecaster(batch):
    """With sigma near zero each sample is the mean network, and epistemic is ~0."""
    model = _model().eval()
    _set_rho(model, -20.0)
    with torch.no_grad():
        pred = model.predictive(batch)
        h, anchor = model.encode(batch)
        x = h
        for i, layer in enumerate(model.head.layers):
            x = torch.nn.functional.linear(x, layer.mu_weight, layer.mu_bias)
            if i < len(model.head.layers) - 1:
                x = torch.tanh(x)
        mu = anchor + x[..., 0]
        scale = torch.nn.functional.softplus(x[..., 1]) + 1e-2
    assert float(pred.epistemic.max()) < 1e-10
    torch.testing.assert_close(pred.mean, mu, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pred.aleatoric, scale**2, rtol=1e-5, atol=1e-6)


def test_loss_is_nll_plus_scaled_kl(batch):
    model = _model().train()
    total, parts = model.loss(batch, beta=0.25, generator=torch.Generator().manual_seed(0))
    assert parts["beta"] == 0.25
    assert math.isclose(float(total), parts["nll"] + 0.25 * parts["kl"] / 1000, rel_tol=1e-5)
    total.backward()
    grads = [layer.rho_weight.grad for layer in model.head.layers]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_forward_channels_and_score_run_on_synthetic_batch(batch):
    model = _model()
    chans = model.channels(batch)
    assert set(chans) == set(model.channel_names)
    n = len(nested_batch.node_keys())
    for values in chans.values():
        assert values.shape == (1, n)
    finite = {k: np.asarray(v)[np.isfinite(v)] for k, v in chans.items()}
    nulls = scoring.ChannelNulls().fit(
        {k: np.concatenate([v, np.linspace(0.0, 3.0, 50)]) for k, v in finite.items()}
    )
    combined = model.score(batch, nulls)
    assert np.asarray(combined).shape == (1, n)
    assert model.training, "channels() should put the model back the way it found it"


# The served model on the real checkpoint. Skipped without the weights, like the conformance tests.


@pytest.fixture(scope="module")
def served():
    import pandas as pd

    from strawberrywatch import paths
    from strawberrywatch.serving.flame import FlameSkimmerDetector

    folder = paths.checkpoints_dir()
    for name in FlameSkimmerDetector.FILES:
        if not (folder / name).exists():
            pytest.skip(f"no {name} in checkpoints/")
    # Every anchor, not just the last 2 hours, to compare with the evaluation's peaks
    return FlameSkimmerDetector.load(folder, recent=pd.Timedelta(days=365))


def _event(detector, event):
    import pandas as pd

    from strawberrywatch.watch import sources

    start, end = pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2030-01-01", tz="UTC")
    return detector.score(sources.ReplaySource(event).tables(detector.table_names, start, end))


def test_served_it_flags_the_september_spike_at_the_documents_peak(served):
    result = _event(served, "anomaly_2025_09_10_overnight_sf")
    j = [n.key for n in result["nodes"]].index("south_fork_1.conductivity")
    # Table 3 of Math_and_Design_of_FlameSkimmer: 23.740 against a line of 16.271
    assert np.nanmax(result["scores"][:, j]) == pytest.approx(23.740, abs=1e-3)
    assert result["fired"][:, j].any()


def test_served_it_never_scores_the_sensors_training_never_saw(served):
    from strawberrywatch.watch import models as wm

    result = _event(served, "anomaly_2025_09_10_overnight_sf")
    for j, node in enumerate(result["nodes"]):
        if node.key in ("footbridge.do_pct", "footbridge.float_cond"):
            assert np.isnan(result["scores"][:, j]).all() and not result["fired"][:, j].any()
    info = next(m for m in wm.registered() if m.name == "flame_skimmer")
    assert ("scnf010", "dissolved_oxygen") not in info.nodes
    assert ("scnf010", "floating_conductivity") not in info.nodes
    assert ("scnf010", "conductivity") in info.nodes

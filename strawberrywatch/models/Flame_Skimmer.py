# Flame Skimmer is a bayesian neural network

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from strawberrywatch.anomalies import channel_scoring as scoring
from strawberrywatch.models import model_calls
from strawberrywatch.models.Cobble_Shoal import (
    CANONICAL_VARS,
    CONTEXT_FEATURES,
    ENCODER_KEYS,
    HEAD_SEED_OFFSET,
    MAX_ANCHOR_STALENESS_STEPS,
    MIN_SCALE,
    SITE_ORDER,
    NestedEncoder,
    build_node_registry,
    gaussian_nll,
)

# Head weight samples used when scoring. The serving noise is drawn once with
# this many samples, so it is also the most predictive() can take without an
# explicit generator.
SERVING_SAMPLES = 64

# Seed for the serving noise. Fixed, so the same window always gets the same
# score and an alert cannot flap between two runs over the same data.
SERVING_NOISE_SEED = 20261013


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0, rho_init=-5.0):
        """
        Linear layer with a mean-field Gaussian posterior over its weight and bias.

        Parameters:
            in_features: int -> size of each input sample,
            out_features: int -> size of each output sample,
            prior_sigma: float -> standard deviation of the N(0, prior_sigma^2) prior on every weight and bias,
            rho_init: float -> initial rho for every weight, sigma = softplus(rho). -5 starts sigma near 0.0067,
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)
        self.rho_init = float(rho_init)

        self.mu_weight = Parameter(torch.empty(out_features, in_features))
        self.rho_weight = Parameter(torch.empty(out_features, in_features))
        self.mu_bias = Parameter(torch.empty(out_features))
        self.rho_bias = Parameter(torch.empty(out_features))

        # filled by fix_noise, left out of the state_dict on purpose
        self.register_buffer("fixed_eps_weight", None, persistent=False)
        self.register_buffer("fixed_eps_bias", None, persistent=False)

        self.init_parameters()

    def init_parameters(self):
        """mu starts like nn.Linear's default weights, rho starts flat at rho_init."""
        nn.init.kaiming_uniform_(self.mu_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        nn.init.uniform_(self.mu_bias, -bound, bound)
        nn.init.constant_(self.rho_weight, self.rho_init)
        nn.init.constant_(self.rho_bias, self.rho_init)

    @property
    def sigma_weight(self):
        # softplus is log1p(exp(rho)) without the overflow for large rho
        return F.softplus(self.rho_weight)

    @property
    def sigma_bias(self):
        return F.softplus(self.rho_bias)

    def kl(self):
        """
        Closed form KL(q || N(0, prior_sigma^2)), summed over every weight and bias, not averaged.

        returns torch.Tensor of shape ()
        """
        total = 0.0
        for mu, sigma in ((self.mu_weight, self.sigma_weight), (self.mu_bias, self.sigma_bias)):
            total = (
                total
                + (
                    math.log(self.prior_sigma)
                    - torch.log(sigma)
                    + (sigma**2 + mu**2) / (2.0 * self.prior_sigma**2)
                    - 0.5
                ).sum()
            )
        return total

    def forward(self, input, generator=None):
        """
        Training forward with the local reparameterization trick, one epsilon per output element.

        Parameters:
            input: torch.Tensor -> (..., in_features),
            generator: torch.Generator -> optional, for reproducible noise,

        returns torch.Tensor of shape (..., out_features)
        """
        mean = F.linear(input, self.mu_weight, self.mu_bias)
        var = F.linear(input**2, self.sigma_weight**2, self.sigma_bias**2)
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + torch.sqrt(var) * eps

    def draw_noise(self, samples, generator=None):
        """
        Standard normal noise for `samples` whole weight matrices. Noise rather than weights, so it outlives training.

        returns (eps_weight (samples, out, in), eps_bias (samples, out))
        """
        dev, dt = self.mu_weight.device, self.mu_weight.dtype
        eps_weight = torch.randn(
            samples, self.out_features, self.in_features, generator=generator, device=dev, dtype=dt
        )
        eps_bias = torch.randn(
            samples, self.out_features, generator=generator, device=dev, dtype=dt
        )
        return eps_weight, eps_bias

    def fix_noise(self, samples, generator):
        """Draw the serving noise once and keep it, so scoring is deterministic."""
        # drawn on cpu so the values do not depend on which device the model sits on
        eps_weight = torch.randn(samples, self.out_features, self.in_features, generator=generator)
        eps_bias = torch.randn(samples, self.out_features, generator=generator)
        dev, dt = self.mu_weight.device, self.mu_weight.dtype
        self.fixed_eps_weight = eps_weight.to(device=dev, dtype=dt)
        self.fixed_eps_bias = eps_bias.to(device=dev, dtype=dt)

    def forward_sampled(self, input, eps_weight, eps_bias, shared_input=False):
        """
        Prediction forward with one whole weight matrix per sample, so a window ignores its batch neighbours.

        Parameters:
            input: torch.Tensor -> (S, ..., in_features), or (..., in_features) when shared_input,
            eps_weight: torch.Tensor -> (S, out_features, in_features) standard normal noise,
            eps_bias: torch.Tensor -> (S, out_features) standard normal noise,
            shared_input: bool -> input has no sample axis and every sample reads it,

        returns torch.Tensor of shape (S, ..., out_features)
        """
        samples = eps_weight.shape[0]
        weight = self.mu_weight + self.sigma_weight * eps_weight
        bias = self.mu_bias + self.sigma_bias * eps_bias
        # told, not guessed from the shape, since a batch of 64 looks like 64 samples
        if shared_input:
            input = input.unsqueeze(0).expand(samples, *input.shape)
        lead = input.shape[1:-1]
        flat = input.reshape(samples, -1, self.in_features)
        out = torch.baddbmm(bias.unsqueeze(1), flat, weight.transpose(1, 2))
        return out.reshape(samples, *lead, self.out_features)


class BayesianHead(nn.Module):
    def __init__(self, in_dim, width=32, prior_sigma=1.0, rho_init=-5.0):
        """
        Two Bayesian hidden layers and a Bayesian output layer giving delta and raw scale.

        Parameters:
            in_dim: int -> width of the encoder output the head reads,
            width: int -> units in each hidden layer,
            prior_sigma: float -> prior standard deviation, passed to every BayesLinear,
            rho_init: float -> initial rho, passed to every BayesLinear,
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BayesLinear(in_dim, width, prior_sigma, rho_init),
                BayesLinear(width, width, prior_sigma, rho_init),
                BayesLinear(width, 2, prior_sigma, rho_init),
            ]
        )

    def kl(self):
        return sum(layer.kl() for layer in self.layers)

    def forward(self, h, generator=None):
        """
        One local reparameterization draw per datapoint, for the training loss.

        returns (delta (...), raw_scale (...))
        """
        x = h
        for i, layer in enumerate(self.layers):
            x = layer(x, generator=generator)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x[..., 0], x[..., 1]

    def draw_noise(self, samples, generator=None):
        return [layer.draw_noise(samples, generator) for layer in self.layers]

    def fixed_noise(self, samples):
        """The first `samples` draws of the serving noise."""
        available = self.layers[0].fixed_eps_weight.shape[0]
        if samples > available:
            raise ValueError(
                f"asked for {samples} samples but the serving noise holds {available}; "
                f"pass a generator to draw fresh noise instead"
            )
        return [
            (layer.fixed_eps_weight[:samples], layer.fixed_eps_bias[:samples])
            for layer in self.layers
        ]

    def forward_sampled(self, h, noise):
        """
        The head under each sampled set of weights. h goes in once, unsampled.

        returns (delta (S, ...), raw_scale (S, ...))
        """
        x = h
        for i, (layer, (eps_weight, eps_bias)) in enumerate(zip(self.layers, noise, strict=True)):
            x = layer.forward_sampled(x, eps_weight, eps_bias, shared_input=i == 0)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
        return x[..., 0], x[..., 1]


class Predictive(NamedTuple):
    """
    mean       (B, N)  forecast, mean over head samples
    aleatoric  (B, N)  mean of the predicted variance, the sensor noise
    epistemic  (B, N)  variance of the forecast across head samples
    total      (B, N)  aleatoric + epistemic
    """

    mean: torch.Tensor
    aleatoric: torch.Tensor
    epistemic: torch.Tensor
    total: torch.Tensor


class FlameSkimmer(nn.Module):
    """
    Cobble Shoal's NestedEncoder plus a Bayesian heteroscedastic forecast head.

    Channels: excursion, dispersion.
    """

    INPUT_CONTRACT = model_calls.NESTED_NODE_BATCH

    # rain is handled outside the model by anomalies/rain_gate.py
    BUILTIN_SUPPORT = ()

    channel_names = ("excursion", "dispersion")

    @classmethod
    def from_metadata(cls, metadata):
        """
        Rebuild from a trained metadata blob, sized from the creek structure like CobbleShoal.

        Parameters:
            metadata: dict -> seed and window as for CobbleShoal, plus prior_sigma, head_width and n_train,
        """
        return cls(
            len(SITE_ORDER),
            len(CANONICAL_VARS),
            len(CONTEXT_FEATURES),
            len(build_node_registry()[0]),
            int(metadata.get("seed", 0)),
            window=int(metadata.get("window", 24)),
            head_width=int(metadata.get("head_width", 32)),
            prior_sigma=float(metadata.get("prior_sigma", 1.0)),
            n_train=int(metadata.get("n_train", 1)),
        )

    def __init__(
        self,
        num_sites,
        num_vars,
        num_context,
        num_nodes,
        seed,
        hidden=64,
        var_emb=8,
        site_emb=12,
        window=24,
        head_width=32,
        prior_sigma=1.0,
        rho_init=-5.0,
        n_train=1,
    ):
        """
        Parameters:
            num_sites, num_vars, num_context, num_nodes, seed, hidden, var_emb, site_emb, window -> as for CobbleShoal,
            head_width: int -> units in each Bayesian hidden layer,
            prior_sigma: float -> prior standard deviation on every head weight,
            rho_init: float -> initial rho on every head weight,
            n_train: int -> supervised (anchor, node) targets in the training set, divides the KL,
        """
        super().__init__()
        if n_train < 1:
            raise ValueError(f"n_train must be at least 1, got {n_train}")
        self.hidden = hidden
        self.window = window
        self.seed = seed
        self.head_width = head_width
        self.prior_sigma = float(prior_sigma)
        self.n_train = int(n_train)

        torch.manual_seed(seed)
        self.encoder = NestedEncoder(
            num_sites,
            num_vars,
            num_context,
            num_nodes,
            hidden=hidden,
            var_emb=var_emb,
            site_emb=site_emb,
        )
        torch.manual_seed(seed + HEAD_SEED_OFFSET)
        self.head = BayesianHead(hidden * 2, head_width, prior_sigma, rho_init)

        # own generator, so drawing the serving noise leaves the global RNG alone
        noise_gen = torch.Generator().manual_seed(SERVING_NOISE_SEED)
        for layer in self.head.layers:
            layer.fix_noise(SERVING_SAMPLES, noise_gen)

    def encode(self, batch):
        out = self.encoder(**{k: batch[k] for k in ENCODER_KEYS})
        return torch.cat([out.h, out.site_ctx], dim=-1), out.anchor

    @staticmethod
    def anchor_fresh(batch):
        return batch["staleness"][:, -1] <= MAX_ANCHOR_STALENESS_STEPS

    def kl(self):
        return self.head.kl()

    def loss(self, batch, beta=1.0, generator=None):
        """
        Negative ELBO per supervised target: masked Gaussian NLL plus beta * KL / n_train.

        Parameters:
            batch: dict -> a nested node batch, see model_calls.NESTED_NODE_BATCH,
            beta: float -> KL weight, warmed from 0 to 1 by the training script,
            generator: torch.Generator -> optional, for reproducible head noise,

        returns (total loss tensor, {"nll", "kl", "beta"})
        """
        h, anchor = self.encode(batch)
        delta, raw_scale = self.head(h, generator=generator)
        mu = anchor + delta
        scale = F.softplus(raw_scale) + MIN_SCALE
        mask = batch["target_mask"] & self.anchor_fresh(batch)
        nll = gaussian_nll(mu, scale, batch["target"], mask)
        kl = self.kl()
        total = nll + beta * kl / self.n_train
        return total, {"nll": float(nll.detach()), "kl": float(kl.detach()), "beta": float(beta)}

    def predictive(self, batch, samples=SERVING_SAMPLES, generator=None):
        """
        Moment matched predictive distribution. Encoder once, head `samples` times.

        Parameters:
            batch: dict -> a nested node batch,
            samples: int -> head weight samples,
            generator: torch.Generator -> optional, draws fresh noise instead of the serving noise,

        returns Predictive
        """
        h, anchor = self.encode(batch)
        if generator is None:
            noise = self.head.fixed_noise(samples)
        else:
            noise = self.head.draw_noise(samples, generator)
        delta, raw_scale = self.head.forward_sampled(h, noise)
        mu = anchor.unsqueeze(0) + delta
        sigma = F.softplus(raw_scale) + MIN_SCALE
        mean = mu.mean(dim=0)
        aleatoric = (sigma**2).mean(dim=0)
        # population variance, which is what the moment matched mixture uses
        epistemic = mu.var(dim=0, unbiased=False)
        return Predictive(mean, aleatoric, epistemic, aleatoric + epistemic)

    @torch.no_grad()
    def channels(self, batch):
        """
        Raw per-channel scores, (B, N) each, NaN where a score cannot be computed.

        Excursion reads the total standard deviation, dispersion the aleatoric part only.
        """
        was_training = self.training
        self.eval()
        pred = self.predictive(batch)
        chans = {
            "excursion": scoring.excursion(
                pred.mean,
                torch.sqrt(pred.total),
                batch["target"],
                batch["target_mask"],
                self.anchor_fresh(batch),
            ),
            "dispersion": scoring.dispersion(
                torch.sqrt(pred.aleatoric), batch["values"], obs_mask=batch["obs_mask"]
            ),
        }
        if was_training:
            self.train()
        return chans

    @torch.no_grad()
    def score(self, batch, nulls, return_channels=False):
        """
        One number per node, the combined Fisher statistic. Larger is more anomalous.

        Parameters:
            batch: dict -> a nested node batch,
            nulls: ChannelNulls -> fitted at calibration time, nothing is fitted here,
            return_channels: bool -> also return the raw channel scores, for diagnostics,
        """
        chans = self.channels(batch)
        combined = scoring.combine(pvalues=nulls.pvalues(chans), rule="fisher")
        return (combined, chans) if return_channels else combined


def build_flame_skimmer(num_sites, num_vars, num_context, num_nodes, seed, window, **head_args):
    """Same argument order as build_cobble_shoal, so the training scripts read alike."""
    return FlameSkimmer(
        num_sites, num_vars, num_context, num_nodes, seed, window=window, **head_args
    )

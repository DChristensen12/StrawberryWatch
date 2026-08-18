"""
Detection channels, per-channel nulls, p-value combination and the POT
threshold, for the Cobble Shoal detector.

Ported from the comparison harness that selected this detection objective. The
arithmetic is deliberately unchanged: the calibrated artifacts shipped beside
the weights (channel nulls, POT threshold z_q) were fitted against exactly
these functions, so an edit here silently invalidates them.

The organising constraint is that no labelled anomalies exist and none will, so
nothing here may be tuned against a fault. Every number a channel produces is
turned into a right-tail probability against that channel's own fault-free
distribution, and the alerting threshold comes from Extreme Value Theory at a
false alarm rate chosen a priori.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# Window used for the observed dispersion statistic. Long enough that ordinary
# noise averages out, short enough that a sensor stuck for an hour shows up.
# This is the only definition; common.py used to carry an identical unread copy,
# which is the kind of duplicate that silently diverges.
DISPERSION_WINDOW = 8

# Below this many exceedances a two-parameter GPD fit is noise. The empirical
# tail is used instead and the caller is told the fit was skipped.
MIN_EXCEEDANCES = 20


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# Raw channels. Each returns (B, N) with NaN where the score cannot honestly be
# computed, never a zero standing in for "unknown".


def excursion(pred, scale, target, target_mask, anchor_valid):
    """
    Standardised one-step forecast residual. Catches a reading that moved when
    it should not have.

    Structurally blind to a frozen sensor: the prediction is anchored on the
    frozen reading itself, so predicting and observing the same frozen number
    gives zero residual. That blindness is the reason the other channels exist.
    """
    usable = _np(target_mask).astype(bool) & _np(anchor_valid).astype(bool)
    z = np.abs(_np(target) - _np(pred)) / _np(scale)
    return np.where(usable, z, np.nan)


def loo(loo_pred, loo_scale, target, target_mask):
    """
    Standardised residual between a node's leave-one-out prediction and what it
    actually reported.

    With the node's own input removed the prediction comes from its siblings,
    its flow neighbours and the clock, all of which are still moving, so a
    frozen reading stands out by exactly as much as the creek has moved since it
    froze. Anchor freshness is not a condition here: the leave-one-out
    prediction does not use the node's anchor at all.
    """
    usable = _np(target_mask).astype(bool)
    z = np.abs(_np(target) - _np(loo_pred)) / _np(loo_scale)
    return np.where(usable, z, np.nan)


def reconstruction(x_hat, rec_scale, values, observed_mask):
    """
    Mean standardised reconstruction error over the input window, per node.

    x_hat and rec_scale are (B, N, T); values and observed_mask are (B, T, N).

    Anchored on neither the last value nor the neighbours, which is what makes
    it the only channel that can see a window whose shape is wrong while its
    level is right and the network agrees.
    """
    xh, sc = _np(x_hat), _np(rec_scale)
    v = np.transpose(_np(values), (0, 2, 1))
    m = np.transpose(_np(observed_mask), (0, 2, 1)).astype(bool)
    z = np.abs(v - xh) / sc
    n_obs = m.sum(axis=-1)
    tot = np.where(m, z, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_obs > 0, tot / np.maximum(n_obs, 1), np.nan)


def dispersion(scale, values, obs_mask=None, window=DISPERSION_WINDOW, floor=1e-3, min_obs=None):
    """
    How much flatter the sensor is than the model expected, in log units, zero
    when it is at least as variable as predicted.

    Nearly free, no head of its own, and it is what a bit-exact freeze trips:
    the residual goes to zero, which is what blinds the excursion channel, but
    the observed dispersion goes to zero too while the predicted scale does not.

    obs_mask is not optional in practice, and leaving it out is a measured
    mistake rather than a hypothetical one. Values are last-observation-carried-
    forward, so a node that is simply offline has a flat series for exactly the
    same arithmetic reason a frozen sensor does. Without the mask, ordinary
    outages sit in the fault-free null of this channel at the same magnitude as
    the faults, the null's tail swallows them, and detection at the POT
    threshold on stuck faults measured 0.0% while the same channel was still
    ranking the frozen node first. Requiring real observations in the window is
    also the same refusal the decay cell makes: a flat battery must not read as
    an anomaly.
    """
    obs = _np(values)[:, -window:, :].std(axis=1)
    sc = _np(scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.clip(np.log(np.maximum(sc, floor) / np.maximum(obs, floor)), 0.0, None)
    if obs_mask is None:
        return ratio
    need = window // 2 if min_obs is None else min_obs
    n_obs = _np(obs_mask)[:, -window:, :].astype(bool).sum(axis=1)
    return np.where(n_obs >= need, ratio, np.nan)


# Nulls and p-values


def fit_gpd(excess):
    """
    Two-parameter Generalised Pareto fit by maximum likelihood, on excesses over
    a threshold. Returns (sigma, gamma).

    Profile parameterisation, following Grimshaw: with t = gamma/sigma the
    likelihood collapses to one dimension,

        gamma(t) = mean(log(1 + t*y))
        sigma(t) = gamma(t)/t
        l(t)     = n*log(t/gamma(t)) - n*(gamma(t) + 1)

    maximised over t > -1/max(y), t != 0. Grid then golden section, rather than
    an optimiser dependency; the exponential limit t -> 0 is evaluated
    separately and wins if it is genuinely better, which is what makes the
    gamma-near-zero case a fitted answer rather than a special case.
    """
    y = np.asarray(excess, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    n = y.size
    if n < 2:
        return float("nan"), float("nan")

    y_max, y_mean = float(y.max()), float(y.mean())

    def loglik(t):
        if abs(t) < 1e-12:
            return -n * math.log(y_mean) - n
        g = float(np.mean(np.log1p(t * y)))
        if g <= 1e-12 or (g / t) <= 0:
            return -np.inf
        return n * math.log(t / g) - n * (g + 1.0)

    lo = -1.0 / y_max + 1e-9
    span = np.exp(np.linspace(math.log(1e-6 / y_mean), math.log(1e3 / y_mean), 400))
    grid = np.concatenate([np.clip(-span[::-1], lo, None), span])
    grid = np.unique(grid[(grid > lo) & (np.abs(grid) > 1e-12)])

    ll = np.array([loglik(t) for t in grid])
    best = int(np.argmax(ll))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, grid.size - 1)]

    # Golden section on the bracketing interval. The profile is unimodal there
    # for every sample we have looked at; if it is not, the grid answer stands.
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    for _ in range(80):
        if loglik(c) > loglik(d):
            b, d = d, c
            c = b - phi * (b - a)
        else:
            a, c = c, d
            d = a + phi * (b - a)
    t = 0.5 * (a + b)

    if loglik(0.0) >= loglik(t):
        return y_mean, 0.0
    gamma = float(np.mean(np.log1p(t * y)))
    return float(gamma / t), float(gamma)


def gpd_sf(x, u, sigma, gamma):
    """P(X > u + x) / P(X > u) under the fitted tail, with the gamma -> 0 limit."""
    z = np.asarray(x, dtype=float) - u
    z = np.maximum(z, 0.0)
    if abs(gamma) < 1e-8:
        return np.exp(-z / sigma)
    arg = 1.0 + gamma * z / sigma
    # gamma < 0 gives a finite upper endpoint; beyond it the tail probability is
    # zero, which is reported as zero rather than as a complex number.
    return np.where(arg > 0, np.power(np.maximum(arg, 1e-300), -1.0 / gamma), 0.0)


class ChannelNull:
    """
    The fault-free distribution of one channel's score.

    Raw channel scores are not comparable across channels: an excursion of 3 and
    a dispersion of 3 mean different things, which is why a max over raw
    magnitudes lets an uninformative channel outrank an informative one. A tail
    probability against the channel's own null is comparable by construction.

    Below the 95th percentile the empirical CDF is used directly. Above it the
    tail is a fitted GPD, so two scores that both sit past every observed null
    sample still order against each other instead of both saturating at the
    1/(n+1) floor the empirical estimate can never go below.

    The tail used for p-values clamps gamma at zero, and that is deliberate. A
    negative gamma is a finite upper endpoint, the MLE puts that endpoint at the
    largest observed excess, and every score past it collapses to p = 0. That is
    not a small p-value, it is a tie: the largest fault-free sample and a sensor
    frozen for six hours both come out at the floor, the combined score
    saturates for both, and no threshold can separate them. Measured, it drove
    detection at the POT threshold to 0.0% on stuck faults the dispersion
    channel was ranking first. Clamping to the exponential limit keeps the tail
    unbounded, keeps the ordering the combination rule needs, and is
    conservative: it assigns a larger p than the unconstrained fit. The
    unconstrained gamma is still stored and reported.
    """

    def __init__(self, samples=None, u=None, sigma=None, gamma=None, tail_p=None):
        self.samples = None if samples is None else np.sort(np.asarray(samples, float))
        self.u = u
        self.sigma = sigma
        self.gamma = gamma
        self.tail_p = tail_p

    @property
    def tail_gamma(self):
        return max(self.gamma, 0.0) if self.gamma is not None else None

    def fit(self, scores, tail_pct=95.0):
        s = np.asarray(scores, dtype=float).ravel()
        s = s[np.isfinite(s)]
        self.samples = np.sort(s)
        n = self.samples.size
        self.u = self.sigma = self.gamma = self.tail_p = None
        if n >= 50:
            u = float(np.percentile(self.samples, tail_pct))
            excess = self.samples[self.samples > u] - u
            if excess.size >= MIN_EXCEEDANCES:
                sigma, gamma = fit_gpd(excess)
                if np.isfinite(sigma) and sigma > 0 and np.isfinite(gamma):
                    self.u, self.sigma, self.gamma = u, sigma, gamma
                    self.tail_p = float(excess.size) / n
        return self

    def to_pvalue(self, score):
        """Right-tail probability, NaN in and NaN out."""
        s = np.asarray(score, dtype=float)
        n = self.samples.size
        ge = n - np.searchsorted(self.samples, s, side="left")
        # The +1 keeps a score above every null sample from reading as exactly
        # zero, which would dominate any combination rule outright.
        p = (1.0 + ge) / (1.0 + n)
        if self.u is not None:
            tail = self.tail_p * gpd_sf(s, self.u, self.sigma, self.tail_gamma)
            p = np.where(s > self.u, np.maximum(tail, 1e-300), p)
        return np.where(np.isfinite(s), p, np.nan)

    def to_dict(self):
        return {
            "samples": self.samples.tolist(),
            "u": self.u,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "tail_p": self.tail_p,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["samples"], float), d["u"], d["sigma"], d["gamma"], d["tail_p"])


class ChannelNulls:
    """One ChannelNull per channel, saved alongside the weights."""

    def __init__(self, nulls=None):
        self.nulls = nulls or {}

    def fit(self, per_channel):
        self.nulls = {k: ChannelNull().fit(v) for k, v in per_channel.items()}
        return self

    def pvalues(self, scores):
        return {k: self.nulls[k].to_pvalue(v) for k, v in scores.items() if k in self.nulls}

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.nulls.items()}

    @classmethod
    def from_dict(cls, d):
        return cls({k: ChannelNull.from_dict(v) for k, v in d.items()})


def chi2_sf_even(stat, dof):
    """
    Survival function of chi-squared with even degrees of freedom, in closed
    form: exp(-x/2) * sum_{i<k} (x/2)^i / i! for dof = 2k.

    Fisher's statistic always has even dof, so this covers every case here and
    keeps the folder free of a scipy dependency.
    """
    k = dof // 2
    x = np.asarray(stat, dtype=float) / 2.0
    term = np.ones_like(x)
    total = np.ones_like(x)
    for i in range(1, k):
        term = term * x / i
        total = total + term
    return np.clip(np.exp(-x) * total, 0.0, 1.0)


def combine(pvalues=None, raw=None, rule="fisher"):
    """
    One score per node from several channels. Larger is more anomalous.

    "fisher" is -2 * sum(log p) over the channels' tail probabilities. It is
    reported as the rule in use everywhere below.

    "max" is the naive path, the maximum over raw channel magnitudes with no
    null and no calibration. It is kept because a previous run measured it at
    68.6% combined where a single channel alone scored 92.2%: raw magnitudes are
    not comparable, so the loudest channel wins whether or not it carries any
    information. That regression has to stay visible in the results table rather
    than be assumed away.

    Fisher assumes independent channels, which these are not: dispersion and
    leave-one-out both fire on a frozen sensor, so an agreeing pair overstates
    its significance. The statistic is still monotone in the evidence and the
    degrees of freedom are constant across nodes for a given model, so ranking
    is unaffected; only the absolute p-value is optimistic, which is why the
    alerting threshold comes from POT on the statistic's own null rather than
    from the chi-squared tail.
    """
    if rule == "max":
        stack = np.stack([np.asarray(v, float) for v in raw.values()])
        with np.errstate(invalid="ignore"):
            out = np.nanmax(stack, axis=0)
        return np.where(np.all(~np.isfinite(stack), axis=0), np.nan, out)
    if rule != "fisher":
        raise ValueError(f"unknown combination rule {rule!r}")
    stack = np.stack([np.asarray(v, float) for v in pvalues.values()])
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = -2.0 * np.nansum(np.log(np.clip(stack, 1e-300, 1.0)), axis=0)
    return np.where(np.all(~np.isfinite(stack), axis=0), np.nan, stat)


def fisher_pvalue(stat, k):
    """
    The chi-squared tail for a Fisher statistic over k channels.

    ILLUSTRATIVE ONLY. Nothing in the alerting path may use this. Fisher's
    chi2(2k) reference distribution assumes the combined p-values are
    independent, and these channels share an encoder, so an agreeing pair
    overstates its significance and this number is optimistic. See combine()
    above for the same warning from the other side.

    The calibrated quantity is the right-tail probability of the combined score
    against its OWN fault-free distribution, which is what ChannelNull.fit /
    to_pvalue gives and what the POT threshold is built from. Use that wherever
    a p-value is shown next to a threshold decision.
    """
    return chi2_sf_even(stat, 2 * k)


# Peaks over threshold


def fit_pot(scores, q=1e-4, u_pct=98.0):
    """
    Fit the tail of the fault-free scores and return the alerting threshold.

    An initial threshold u sits at the u_pct percentile of the fault-free
    scores, and the exceedances above it are what the tail is fitted to. A
    Generalised Pareto fitted to those by maximum likelihood gives sigma and
    gamma, from which z_q = u + (sigma/gamma) * ((q*n/N_u)**(-gamma) - 1).
    Gamma near zero takes the exponential limit, z_q = u - sigma*log(q*n/N_u).

    q is chosen a priori and the false alarm rate follows by construction, with
    no labels anywhere. Returns the threshold; pot_diagnostics returns the fit.
    """
    return pot_diagnostics(scores, q, u_pct)["threshold"]


def pot_diagnostics(scores, q=1e-4, u_pct=98.0, held_out=None):
    """
    The POT fit and everything needed to judge whether to trust it: sigma,
    gamma, u, N_u, and the empirical exceedance rate at the returned threshold,
    measured on held_out if given and on the fitting sample otherwise.

    An empirical rate that diverges from nominal q by more than about an order
    of magnitude means the GPD fit is bad and the threshold is not trustworthy.
    """
    s = np.asarray(scores, dtype=float).ravel()
    s = s[np.isfinite(s)]
    n = s.size
    out = {
        "q": q,
        "n": int(n),
        "u": float("nan"),
        "n_u": 0,
        "sigma": float("nan"),
        "gamma": float("nan"),
        "threshold": float("nan"),
        "empirical_rate": float("nan"),
        "fitted": False,
        "note": "",
    }
    if n < 50:
        out["note"] = "too few fault-free scores to fit"
        return out

    u = float(np.percentile(s, u_pct))
    excess = s[s > u] - u
    out["u"], out["n_u"] = u, int(excess.size)
    if excess.size < MIN_EXCEEDANCES:
        # Fall back to the empirical quantile and say so, rather than reporting
        # a GPD fit that was never made.
        out["threshold"] = float(np.quantile(s, 1.0 - q)) if q < 1 else u
        out["note"] = f"only {excess.size} exceedances, empirical quantile used"
    else:
        sigma, gamma = fit_gpd(excess)
        out["sigma"], out["gamma"] = sigma, gamma
        ratio = q * n / max(excess.size, 1)
        if not np.isfinite(sigma) or sigma <= 0:
            out["threshold"] = float(np.quantile(s, 1.0 - q))
            out["note"] = "GPD fit failed, empirical quantile used"
        elif abs(gamma) < 1e-8:
            out["threshold"] = u - sigma * math.log(ratio)
            out["fitted"] = True
        else:
            out["threshold"] = u + (sigma / gamma) * (ratio ** (-gamma) - 1.0)
            out["fitted"] = True

    ref = s if held_out is None else np.asarray(held_out, float).ravel()
    ref = ref[np.isfinite(ref)]
    if ref.size:
        out["empirical_rate"] = float((ref > out["threshold"]).mean())
        out["n_eval"] = int(ref.size)
    return out

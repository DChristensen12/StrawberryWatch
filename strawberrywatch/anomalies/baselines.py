"""
Two control charts on residuals, so the models have a floor to beat.

CUSUM accumulates signed departures until the running sum clears a slack value.
EWMA is an exponentially weighted mean of the same residuals. Both expose
score(batch, nulls) like a model does and go through ChannelNull and POT
unchanged, so the numbers are comparable.
"""

from __future__ import annotations

import numpy as np

from strawberrywatch.anomalies import channel_scoring as scoring

# has to be shorter than the 24 step scored window or the median is an
# expanding one with no history behind the early steps
MEDIAN_WINDOW = 8

CUSUM_K = 0.5  # slack in sigma, half the shift we care about
EWMA_LAMBDA = 0.2

# stops a node that never moves scoring infinite
MIN_SCALE = 1e-2

# 1 / Phi^-1(0.75), turns a MAD into a sigma
MAD_TO_SIGMA = 1.4826


def rolling_median(values, window=MEDIAN_WINDOW):
    """Trailing median per column, expanding until the window fills."""
    v = np.asarray(values, dtype=float)
    out = np.empty_like(v)
    for t in range(v.shape[0]):
        lo = max(0, t - window + 1)
        out[t] = np.median(v[lo : t + 1], axis=0)
    return out


def robust_scale(residual, floor=MIN_SCALE):
    """MAD based sigma per column. Faults don't get to inflate their own scale."""
    r = np.asarray(residual, dtype=float)
    mad = np.median(np.abs(r - np.median(r, axis=0)), axis=0)
    return np.maximum(MAD_TO_SIGMA * mad, floor)


def series_with_target(batch):
    """
    The window plus the target observation as one more step, and its mask.

    The models score the target, so a chart reading only batch["values"] sits a
    step behind them and misses any fault whose onset is the observation being
    predicted. That is where creek_synthetic puts a spike.
    """
    values = np.asarray(batch["values"])[0]
    target = np.asarray(batch["target"])[0]
    observed = (
        np.asarray(batch["obs_mask"])[0]
        if "obs_mask" in batch
        else np.ones_like(values, dtype=bool)
    )
    target_mask = np.asarray(batch["target_mask"])[0]

    # no target means no new reading. a NaN here would poison the node's median
    filled = np.where(target_mask, np.nan_to_num(target, nan=0.0), values[-1])
    return (
        np.vstack([values, filled[None, :]]),
        np.vstack([observed, target_mask[None, :]]),
    )


def standardised_residual(values, observed=None, window=MEDIAN_WINDOW):
    """
    (T, N) residual against the trailing median, in sigma.

    Carried forward steps contribute zero. They are not new evidence, and
    letting them accumulate drifts every offline node into an alarm.
    """
    v = np.asarray(values, dtype=float)
    residual = v - rolling_median(v, window)
    x = residual / robust_scale(residual)
    if observed is not None:
        x = np.where(np.asarray(observed, dtype=bool), x, 0.0)
    return x


def cusum_path(x, k=CUSUM_K):
    """Two-sided CUSUM, (T, N) of max(S+, S-). Window score is the last row."""
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    hi = np.zeros(n)
    lo = np.zeros(n)
    out = np.zeros((t, n))
    for i in range(t):
        hi = np.maximum(0.0, hi + x[i] - k)
        lo = np.maximum(0.0, lo - x[i] - k)
        out[i] = np.maximum(hi, lo)
    return out


def ewma_path(x, lam=EWMA_LAMBDA):
    """
    Standardised two-sided EWMA over the same residuals.

    Var(z_t) is lam/(2-lam) * (1 - (1-lam)^(2t)) sigma^2, used exactly and not
    at its asymptote, or step 1 reads as anomalous on a small denominator.
    """
    x = np.asarray(x, dtype=float)
    t, n = x.shape
    z = np.zeros(n)
    out = np.zeros((t, n))
    for i in range(t):
        z = lam * x[i] + (1.0 - lam) * z
        var = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * (i + 1)))
        out[i] = np.abs(z) / np.sqrt(max(var, 1e-12))
    return out


class _ResidualChart:
    """
    A control chart wearing the model's detector interface.

    score(batch, nulls) is the whole contract the alerting path needs, so these
    drop into any harness that scores a model and the comparison stays on one
    code path.
    """

    channel_names = ()
    INPUT_CONTRACT = "nested_node_batch"
    BUILTIN_SUPPORT = ()

    def __init__(self, window=MEDIAN_WINDOW):
        self.window = window

    def _statistic(self, x):
        raise NotImplementedError

    def path(self, batch):
        """(T+1, N) statistic per step, last row is the target step."""
        values, observed = series_with_target(batch)
        return self._statistic(standardised_residual(values, observed, self.window))

    def channels(self, batch):
        terminal = self.path(batch)[-1]
        return {self.channel_names[0]: terminal[None, :]}

    def score(self, batch, nulls, return_channels=False):
        """One number per node, same Fisher and null path a model uses."""
        chans = self.channels(batch)
        combined = scoring.combine(pvalues=nulls.pvalues(chans), rule="fisher")
        return (combined, chans) if return_channels else combined


class CusumBaseline(_ResidualChart):
    """Catches a sustained shift. Blind to a frozen sensor."""

    channel_names = ("cusum",)

    def _statistic(self, x):
        return cusum_path(x)


class EwmaBaseline(_ResidualChart):
    """Faster on a step, slower on a slow drift, same blind spot."""

    channel_names = ("ewma",)

    def _statistic(self, x):
        return ewma_path(x)


BASELINES = {"cusum": CusumBaseline, "ewma": EwmaBaseline}


def calibrate(detector, batches, q, u_pct=98.0):
    """
    Fit the channel null and the POT threshold on fault-free windows.

    Null first so scores become tail probabilities, then POT on the combined
    statistic. q comes from whoever owns the alerting budget.
    """
    per_channel = {name: [] for name in detector.channel_names}
    for batch in batches:
        for name, value in detector.channels(batch).items():
            per_channel[name].append(np.asarray(value).ravel())

    nulls = scoring.ChannelNulls().fit(
        {name: np.concatenate(vals) for name, vals in per_channel.items()}
    )

    combined = np.concatenate(
        [np.asarray(detector.score(batch, nulls)).ravel() for batch in batches]
    )
    combined = combined[np.isfinite(combined)]
    diagnostics = scoring.pot_diagnostics(combined, q=q, u_pct=u_pct)
    return nulls, diagnostics


__all__ = [
    "BASELINES",
    "CUSUM_K",
    "EWMA_LAMBDA",
    "MEDIAN_WINDOW",
    "CusumBaseline",
    "EwmaBaseline",
    "calibrate",
    "cusum_path",
    "ewma_path",
    "robust_scale",
    "rolling_median",
    "series_with_target",
    "standardised_residual",
]

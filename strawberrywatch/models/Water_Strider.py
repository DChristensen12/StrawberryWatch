"""
Water Strider: a calibrated statistical detector. Each sensor is judged on its
own readings with three tests that each catch what the other two miss: a
rolling Hampel test for spikes, a two-sided CUSUM for shifts that hold, and an
EWMA for slow drift. Nothing is trained by gradient descent. Every number is a
median, a deviation or a fitted tail, so each alert can be said in one sentence.

All three run on residuals: the reading minus the sensor's normal day, divided
by a robust spread. Every threshold is fitted on clean data with peaks over
threshold (Siffer et al., KDD 2017) to the test's share of one false alarm
budget, so the rate somebody chose is the rate they get, rather than whatever a
fixed multiplier happens to give.

The calibration lives in water_strider_calibration.json beside the other
checkpoints. It keeps each fitted tail, not just its threshold, which is what
lets the sensitivity slider move without recalibrating anything.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

RULES = ("hampel_spike", "cusum_shift", "ewma_drift")
RULE_WORDS = {
    "hampel_spike": "a sudden spike",
    "cusum_shift": "a jump that stayed",
    "ewma_drift": "a slow drift",
}

# A raw MAD times this is about one standard deviation for normal data. Every
# spread below is a scaled MAD, and the names say so, because 5.2 raw MADs and
# 5.2 scaled ones are 3.5 and 5.2 standard deviations.
MAD_SCALE = 1.4826

STEP = pd.Timedelta(minutes=15)
STEPS_PER_DAY = 96

# The settings the tests run with. A calibration records the ones it was fitted
# at, since a threshold means nothing at another window length.
PARAMS = {
    # Three hours either side of a spike is long enough that one bad reading
    # can't drag the median, short enough that a shift ends up in it within
    # hours, which is when CUSUM has taken over
    "hampel_window": 13,
    # Half a spread of drift allowed per step before the sums start to grow,
    # the usual choice for catching a shift of about one spread
    "cusum_allowance": 0.5,
    # NIST's handbook suggests 0.2 to 0.3
    "ewma_lambda": 0.2,
    # The level the day's rhythm sits on: the median of the last 12 hours.
    # Measured against 6 and 24: at 24 the shift test ran over its budget on
    # held-out months, at 6 a drift is soaked into the level too quickly
    "level_hours": 12,
    # How many days of clean data the daily profile is the median of
    "profile_days": 14,
    # A flat sensor's spread never goes below this share of its long-run spread
    "mad_floor_share": 0.10,
    # A window is judged only when this share of its readings are real ones
    "min_real": 0.75,
    # A gap longer than this restarts CUSUM and EWMA rather than carrying sums over it
    "gap_reset_steps": 8,
    # Exceedances closer together than this are one cluster, so one alarm
    "decluster_steps": 8,
    # Whether CUSUM's input is scaled for the readings' slow wander (see
    # long_run_factor). Off: it held the false alarms down but shrank a real
    # shift below the allowance, so the shift test hardly fired at all
    "cusum_long_run": False,
}

# The smallest step each probe reports, the other half of the spread's floor
RESOLUTION = {
    "conductivity": 1.0,
    "temperature": 0.1,
    "depth": 1.0,
    "dissolved_oxygen": 0.01,
    "floating_conductivity": 0.1,
}

# One false alarm budget for the whole model, split evenly across the three tests.
# The same default the support modules share.
Q_TOTAL = 1e-4
START_PERCENTILES = (98.0, 95.0, 90.0, 85.0, 80.0)
# Below this many exceedances a tail fit is noise
MIN_EXCEEDANCES = 20
# |gamma| under this uses the exponential limit, never a division by nearly zero
GAMMA_TOLERANCE = 1e-6

# The slider's stops, as the whole model's rate. Each test gets a third.
STOPS = (
    ("far_fewer", "Far fewer", 1e-5),
    ("fewer", "Fewer", 3e-5),
    ("calibrated", "Calibrated", 1e-4),
    ("more", "More", 3e-4),
    ("far_more", "Far more", 1e-3),
)
# Past this a test's threshold falls below the 98th percentile the tail fit
# starts from, and below the lowest stop it lands far beyond the data
RATE_WINDOW = (1e-5 / len(RULES), 0.02)


def per_test(q_total):
    return float(q_total) / len(RULES)


# The daily profile and the residuals


def on_grid(values):
    """One sensor's readings on the 15 minute grid, NaN where none came."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return values
    values.index = values.index.floor("15min")
    values = values.groupby(level=0).mean()
    grid = pd.date_range(values.index.min(), values.index.max(), freq="15min", tz="UTC")
    return values.reindex(grid)


def _trailing(values, width, min_real, reduce):
    """reduce() over each step's previous `width` values, NaN where too few were real."""
    out = np.full(values.size, np.nan)
    if values.size <= width:
        return out
    padded = np.concatenate([np.full(width, np.nan), values])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)[: values.size]
    real = np.isfinite(windows).sum(axis=1)
    ok = real >= math.ceil(min_real * width)
    if ok.any():
        with np.errstate(all="ignore"):
            out[ok] = reduce(windows[ok])
    return out


def level(values, params=PARAMS):
    """The median of each step's previous day, which the daily rhythm sits on. Not the step itself."""
    width = int(params["level_hours"] * 4)
    return _trailing(values, width, params["min_real"], lambda w: np.nanmedian(w, axis=1))


def slot_of(index):
    """Which quarter hour of the Pacific day each stamp is, 0 to 95."""
    local = index.tz_convert("America/Los_Angeles")
    return (local.hour * 4 + local.minute // 15).to_numpy()


def rolling_profiles(index, above_level, params=PARAMS):
    """
    The daily profile in force on each day: the median of (reading minus level)
    at each quarter hour over the previous profile_days days. Medians keep a
    past event from bending it. Returns (profile per step, the latest profile).
    """
    # Calendar dates at the creek, so a day stays a day across a clock change
    days = np.array(index.tz_convert("America/Los_Angeles").date)
    slots = slot_of(index)
    frame = pd.DataFrame({"day": days, "slot": slots, "y": above_level})
    table = frame.pivot_table(index="day", columns="slot", values="y", aggfunc="median")
    table = table.reindex(columns=range(STEPS_PER_DAY))
    every_day = pd.date_range(min(days), max(days), freq="D").date
    table = table.reindex(every_day)
    width = int(params["profile_days"])
    matrix = table.to_numpy(dtype=float)
    padded = np.vstack([np.full((width, STEPS_PER_DAY), np.nan), matrix])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)[: len(matrix)]
    profiles = _median_where_seen(windows, axis=2)
    row = {day: i for i, day in enumerate(table.index)}
    per_step = profiles[[row[d] for d in days], slots]
    latest = _median_where_seen(matrix[-width:].T[None], axis=2)[0]
    return per_step, latest


def _median_where_seen(values, axis):
    """Median along an axis, and 0 where fewer than three values were there to take it from."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        middle = np.nanmedian(values, axis=axis)
    counts = np.isfinite(values).sum(axis=axis)
    # A quarter hour seen on fewer than three of those days has no rhythm to remove yet
    return np.where(counts >= 3, middle, 0.0)


def residuals(values, profile, params=PARAMS):
    """The reading minus its level and its quarter hour's place in the day, on the grid."""
    above = values.to_numpy(dtype=float) - level(values.to_numpy(dtype=float), params)
    shape = np.asarray(profile, dtype=float)
    if shape.ndim == 1:
        shape = shape[slot_of(values.index)]
    return above - shape


# The three tests


def hampel(resid, floor, params=PARAMS):
    """
    Distance from the trailing window's median, in scaled MADs. The window
    holds the step itself, so a spike is judged against the readings around it
    and can't hide in its own median.
    """
    width = int(params["hampel_window"])
    out = np.full(resid.size, np.nan)
    if resid.size < width:
        return out
    padded = np.concatenate([np.full(width - 1, np.nan), resid])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    real = np.isfinite(windows).sum(axis=1)
    ok = np.isfinite(resid) & (real >= math.ceil(params["min_real"] * width))
    if not ok.any():
        return out
    with np.errstate(all="ignore"):
        median = np.nanmedian(windows[ok], axis=1)
        mad_raw = np.nanmedian(np.abs(windows[ok] - median[:, None]), axis=1)
    mad_scaled = np.maximum(mad_raw * MAD_SCALE, floor)
    out[ok] = np.abs(resid[ok] - median) / mad_scaled
    return out


def cusum(z, params=PARAMS):
    """
    Two-sided CUSUM on standardised residuals: running sums of what lies beyond
    a small allowance, each held at zero from below, as in detecta's
    detect_cusum. Reports the larger sum. A shift that holds keeps adding.
    """
    k, reset = float(params["cusum_allowance"]), int(params["gap_reset_steps"])
    out = np.full(z.size, np.nan)
    high = low = 0.0
    missing = 0
    for i, value in enumerate(z):
        if not np.isfinite(value):
            missing += 1
            if missing > reset:
                high = low = 0.0
            continue
        missing = 0
        high = max(0.0, high + value - k)
        low = max(0.0, low - value - k)
        out[i] = max(high, low)
    return out


def ewma(z, params=PARAMS):
    """
    An exponentially weighted average of standardised residuals, divided by its
    own spread, sqrt(lambda / (2 - lambda)), so it reads in standard deviations
    of the average rather than of a single reading. A slow drift moves it.
    """
    lam, reset = float(params["ewma_lambda"]), int(params["gap_reset_steps"])
    spread = math.sqrt(lam / (2.0 - lam))
    out = np.full(z.size, np.nan)
    average, missing = 0.0, 0
    for i, value in enumerate(z):
        if not np.isfinite(value):
            missing += 1
            if missing > reset:
                average = 0.0
            continue
        missing = 0
        average = lam * value + (1.0 - lam) * average
        out[i] = abs(average) / spread
    return out


def long_run_factor(rho):
    """
    How much wider a running sum spreads than independent readings would let
    it, sqrt((1 + rho) / (1 - rho)) for lag-one autocorrelation rho. Readings
    here wander slowly, rho near 0.96, so without this CUSUM's allowance of half
    a spread is far smaller than the wander it adds up, and it never stops firing.
    """
    rho = min(max(float(rho), 0.0), 0.995)
    return math.sqrt((1.0 + rho) / (1.0 - rho))


def autocorrelation(resid, mask):
    """Lag-one autocorrelation of the residuals over the masked steps."""
    kept = np.where(mask, resid, np.nan)
    return float(pd.Series(kept).autocorr(1)) if np.isfinite(kept).sum() > 2 else 0.0


def statistics(values, profile, spread, floor, params=PARAMS, rho=0.0):
    """All three tests' statistics at every step of one sensor, as {rule: array}."""
    return statistics_of(residuals(values, profile, params), spread, floor, params, rho)


def statistics_of(resid, spread, floor, params=PARAMS, rho=0.0):
    """The same, from residuals already worked out, for calibration's sake."""
    z = resid / max(spread, floor)
    return {
        "hampel_spike": hampel(resid, floor, params),
        "cusum_shift": cusum(
            z / (long_run_factor(rho) if params.get("cusum_long_run") else 1.0), params
        ),
        "ewma_drift": ewma(z, params),
    }


def floor_for(variable, spread, params=PARAMS):
    """The larger of the probe's resolution and a share of the sensor's long-run spread."""
    return max(RESOLUTION.get(variable, 0.0), params["mad_floor_share"] * spread)


# Tails and thresholds


def clusters(statistic, mask, level, gap=8):
    """
    The peak of each run of steps over level among the masked ones, with runs
    closer than gap steps counted as one. CUSUM and EWMA stay over a line for
    many steps once they cross it, and every one of those steps is the same
    alarm, so this is what gets counted, both for the fit and for a false flag.
    """
    hits = np.flatnonzero(mask & np.isfinite(statistic) & (statistic > level))
    if hits.size == 0:
        return np.array([])
    breaks = np.flatnonzero(np.diff(hits) > gap) + 1
    return np.array([statistic[run].max() for run in np.split(hits, breaks)])


def pooled(arrays, masks, gap=8):
    """
    Several sensors' statistic arrays end to end, with a gap between each that
    no cluster can cross, for one tail shared by every sensor of a kind.
    """
    spacer = np.full(gap + 1, np.nan)
    statistic = np.concatenate([part for a in arrays for part in (a, spacer)])
    mask = np.concatenate([part for m in masks for part in (m, np.zeros(gap + 1, dtype=bool))])
    return statistic, mask


def fit_tail(statistic, mask, gap=8):
    """
    The generalised Pareto tail of a statistic over its clean steps, fitted
    with SciPy's genpareto to the peak of each cluster above the 98th
    percentile. Declustering is the usual remedy for a series whose steps
    aren't independent (Coles, An Introduction to Statistical Modeling of
    Extreme Values, chapter 5), so n_t counts clusters and q is alarms per step.
    Returns {t, sigma, gamma, n, n_t}.
    """
    from scipy.stats import genpareto

    statistic = np.asarray(statistic, dtype=float)
    judged = np.asarray(mask, dtype=bool) & np.isfinite(statistic)
    n = int(judged.sum())
    if n == 0:
        raise ValueError("no clean values to fit")
    # The 98th percentile as the paper has it, unless CUSUM or EWMA spend that
    # top 2% in so few long runs that there aren't enough clusters to fit, in
    # which case the start comes down until there are
    for percentile in START_PERCENTILES:
        t = float(np.percentile(statistic[judged], percentile))
        peaks = clusters(statistic, judged, t, gap)
        if peaks.size >= MIN_EXCEEDANCES:
            break
    else:
        raise ValueError(f"only {peaks.size} clusters even from the {percentile:g}th percentile")
    gamma, _, sigma = genpareto.fit(peaks - t, floc=0.0)
    return {
        "t": t,
        "sigma": float(sigma),
        "gamma": float(gamma),
        "n": n,
        "n_t": int(peaks.size),
        "start_percentile": percentile,
    }


def threshold_for(tail, q):
    """The threshold crossed with probability q, from a fitted tail."""
    t, sigma, gamma = tail["t"], tail["sigma"], tail["gamma"]
    ratio = q * tail["n"] / tail["n_t"]
    if abs(gamma) < GAMMA_TOLERANCE:
        return t - sigma * math.log(ratio)
    return t + sigma / gamma * (ratio ** (-gamma) - 1.0)


def rate_for(tail, z):
    """The same formula backwards: how often a typed threshold z would be crossed."""
    t, sigma, gamma = tail["t"], tail["sigma"], tail["gamma"]
    share = tail["n_t"] / tail["n"]
    if abs(gamma) < GAMMA_TOLERANCE:
        return share * math.exp(-(z - t) / sigma)
    base = 1.0 + gamma * (z - t) / sigma
    # Past a negative shape's finite end the tail says it never happens
    if base <= 0.0:
        return 0.0
    return share * base ** (-1.0 / gamma)


# Turning statistics into flags


def corroborated_spikes(ratios, siblings=None):
    """
    The Hampel statistic as a share of its threshold, kept only where a second
    reading agrees: the one before it, or another sensor at the same site at
    the same step. A single bad reading is Settling Pool's job, not this one's.
    """
    before = np.concatenate([[np.nan], ratios[:-1]])
    agree = np.fmax(before, siblings) if siblings is not None else before
    return np.fmin(ratios, agree)


def x84(values, k=5.2, params=PARAMS):
    """
    Naive X84, for the comparison only: distance from a trailing window's
    median in raw MADs, with no daily profile, no rain and a fixed k.
    """
    raw = values.to_numpy(dtype=float)
    width = int(params["hampel_window"])
    out = np.full(raw.size, np.nan)
    if raw.size < width:
        return out
    padded = np.concatenate([np.full(width - 1, np.nan), raw])
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    ok = np.isfinite(raw)
    with np.errstate(all="ignore"):
        median = np.nanmedian(windows[ok], axis=1)
        mad_raw = np.nanmedian(np.abs(windows[ok] - median[:, None]), axis=1)
    out[ok] = np.abs(raw[ok] - median) / np.maximum(mad_raw, 1e-9) / k
    return out

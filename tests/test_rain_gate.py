"""
Rain gate: threshold adjustment, never score adjustment.

The gate raises the alerting bar during and after rain. POT calibrated z_q
against the fault-free score distribution, so scaling a score would invalidate
that fit and every p-value under it, while scaling the threshold leaves both
alone. The no-contamination test at the bottom is what holds that line.

The detection-cost and false-alarm numbers that justified the design were
measured against the comparison harness corpus, which is gone. They are
recorded in the rain_gate module docstring and cannot be recomputed here
without that corpus.
"""

import numpy as np
import pytest

from strawberrywatch.anomalies import channel_scoring as scoring
from strawberrywatch.anomalies.cobble_calibration import CalibrationError, load_calibration
from strawberrywatch.anomalies.rain_gate import RainGate, RainGateConfigError, fired

N_CHANNELS = 4


@pytest.fixture(scope="module")
def z_q():
    """The calibrated threshold, read from the shipped artifact."""
    try:
        return load_calibration().z_q
    except CalibrationError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def combined_null():
    """
    A fault-free distribution for the combined score, so the audit trail shows
    a p-value from the same distribution the threshold came from rather than
    the optimistic chi-squared tail. Fitted here because this is a test, not a
    serving path; production loads its nulls from the calibration artifact.
    """
    rng = np.random.default_rng(0)
    return scoring.ChannelNull().fit(rng.gamma(shape=4.0, scale=2.0, size=20000))


def synthetic_rain(n, sample_hours=1.0, events=((30, 8, 1.4), (150, 5, 0.9))):
    """Hourly rain with two storms, as (start, duration_hours, mm/h)."""
    rain = np.zeros(n)
    for start, dur, rate in events:
        rain[start : start + int(dur / sample_hours)] = rate
    return rain


def test_off_mode_is_the_identity(z_q):
    rain = synthetic_rain(400)
    tau = RainGate(base_threshold=z_q, mode="off", multiplier=2.0).thresholds(rain)
    expected = np.full(rain.size, z_q)
    assert tau.tobytes() == expected.tobytes(), "off mode moved the threshold"


def test_off_mode_reproduces_the_ungated_decision(z_q, combined_null):
    """Gate off must give the same fired/not-fired as a bare comparison."""
    rng = np.random.default_rng(1)
    scores = rng.gamma(shape=4.0, scale=2.0, size=(400, 17)) * 4.0
    rain = synthetic_rain(400)
    nodes = list(range(17))
    tau = RainGate(base_threshold=z_q, mode="off").thresholds(rain, nodes=nodes)

    gated = fired(scores, tau)
    ungated = np.isfinite(scores) & (scores > z_q)
    assert np.array_equal(gated, ungated)
    assert gated.any(), "test is vacuous if nothing fires at all"


def test_step_matches_a_hand_computation(z_q):
    """
    multiplier 2.0, lookback 12 h, wet_mm 0.1, hourly samples. Rain of
    0.5 mm/h at hours 5, 6, 7 and nothing else.

    The window at hour t covers [t-12, t], so the last hour whose window still
    holds hour 7 is t = 19. The bar is doubled for hours 5 to 19 inclusive.
    """
    n = 40
    rain = np.zeros(n)
    rain[5:8] = 0.5
    gate = RainGate(
        base_threshold=z_q,
        mode="step",
        multiplier=2.0,
        lookback_hours=12.0,
        wet_mm=0.1,
        sample_hours=1.0,
    )
    hand = np.full(n, z_q)
    hand[5:20] = 2.0 * z_q
    assert np.array_equal(gate.thresholds(rain), hand)


def test_deployed_mode_matches_the_production_formula(z_q):
    """
    "deployed" reproduces anomaly_detector._rain_multipliers: hold the full
    multiplier while the lookback total is wet, then taper over decay_hours
    measured from when the lookback window closes.
    """
    n = 80
    rain = np.zeros(n)
    rain[5:8] = 0.5
    m, look, decay = 2.0, 12.0, 36.0
    got = (
        RainGate(
            base_threshold=z_q,
            mode="deployed",
            multiplier=m,
            lookback_hours=look,
            decay_hours=decay,
            wet_mm=0.1,
            sample_hours=1.0,
        ).thresholds(rain)
        / z_q
    )

    want = np.ones(n)
    for t in range(n):
        if rain[max(0, t - int(look)) : t + 1].sum() > 0.1:
            want[t] = m
            continue
        wet_idx = [i for i in range(t + 1) if rain[i] > 0.1]
        if wet_idx:
            frac = min(max(((t - wet_idx[-1]) - look) / decay, 0.0), 1.0)
            want[t] = 1.0 + (m - 1.0) * (1.0 - frac)
    assert np.allclose(got, want)


def test_decay_shape(z_q):
    m, decay, n = 2.0, 36.0, 120
    rain = np.zeros(n)
    rain[10:20] = 1.0  # stops after hour 19
    gate = RainGate(
        base_threshold=z_q,
        mode="decay",
        multiplier=m,
        decay_hours=decay,
        wet_mm=0.1,
        sample_hours=1.0,
    )
    tau = gate.thresholds(rain)
    hours = gate.hours_since_wet(rain)

    stop = 19
    assert tau[stop] == m * z_q and hours[stop] == 0.0
    assert np.all(np.diff(tau[stop:]) <= 0), "decay is not monotone"

    back = stop + int(decay)
    assert tau[back] == pytest.approx(z_q), "did not return to base at decay_hours"
    assert tau[back - 1] > z_q, "returned to base early"
    assert np.all(tau >= z_q - 1e-12) and np.all(tau <= m * z_q + 1e-12)


def test_multiplier_one_is_the_identity_in_every_mode(z_q):
    rain = synthetic_rain(200)
    for mode in ("off", "step", "decay", "deployed"):
        tau = RainGate(base_threshold=z_q, mode=mode, multiplier=1.0, sample_hours=1.0).thresholds(
            rain
        )
        assert np.array_equal(tau, np.full(rain.size, z_q)), f"multiplier=1 moved tau in {mode}"


def test_config_round_trips_and_rejects_nonsense(z_q):
    cfg = {
        "base_threshold": z_q,
        "mode": "decay",
        "multiplier": 2.5,
        "decay_hours": 24.0,
        "wet_mm": 0.2,
        "lookback_hours": 6.0,
        "sample_hours": 1.0,
        "per_node": {"node_7": 3.0},
    }
    gate = RainGate.from_dict(cfg)
    assert RainGate.from_dict(gate.to_dict()).to_dict() == gate.to_dict()
    assert RainGate.from_yaml(gate.to_yaml()).to_dict() == gate.to_dict()

    for bad in (
        {"base_threshold": z_q, "multiplier": 0.5},
        {"base_threshold": z_q, "decay_hours": 0.0},
        {"base_threshold": z_q, "wet_mm": -1.0},
        {"base_threshold": z_q, "mode": "sometimes"},
        {"base_threshold": z_q, "mutliplier": 2.0},
        {"multiplier": 2.0},
        {"base_threshold": z_q, "per_node": {"n": 0.2}},
    ):
        with pytest.raises(RainGateConfigError):
            RainGate.from_dict(bad)


def test_per_node_multiplier_only_moves_the_named_node(z_q):
    rain = np.zeros(60)
    rain[10:14] = 1.0
    gate = RainGate(
        base_threshold=z_q,
        mode="decay",
        multiplier=2.0,
        per_node={"n3": 4.0},
        decay_hours=36.0,
        sample_hours=1.0,
    )
    tau = gate.thresholds(rain, nodes=["n1", "n3"])
    assert tau[0, 0] == z_q and tau[0, 1] == z_q, "per-node differs while dry"
    assert tau[13, 0] == 2.0 * z_q
    assert tau[13, 1] == 4.0 * z_q


def test_delayed_first_flush_separates_the_modes(z_q):
    """
    Rain, then a conductivity pulse 21 h after it stops: outside the 12 h
    lookback, inside the 36 h decay. step is back at base by then and flags the
    pulse at any magnitude; decay and deployed still hold the bar up. Neither
    is asserted correct, the point is that they differ.
    """
    n, pulse_t = 80, 36
    rain = np.zeros(n)
    rain[10:16] = 1.2
    gates = {
        mode: RainGate(
            base_threshold=z_q,
            mode=mode,
            multiplier=2.0,
            lookback_hours=12.0,
            decay_hours=36.0,
            wet_mm=0.1,
            sample_hours=1.0,
        )
        for mode in ("off", "step", "decay", "deployed")
    }
    tau = {m: g.thresholds(rain)[pulse_t] / z_q for m, g in gates.items()}
    assert tau["step"] == pytest.approx(1.0), "step should be back at base once lookback closes"
    assert tau["decay"] > 1.0 and tau["deployed"] > tau["decay"]

    score = np.full(n, 0.3 * z_q)
    score[pulse_t] = 1.25 * z_q
    assert fired(score, gates["step"].thresholds(rain))[pulse_t]
    assert not fired(score, gates["decay"].thresholds(rain))[pulse_t]


def test_scores_and_pvalues_are_identical_across_modes(z_q, combined_null):
    """
    The gate may move the threshold and nothing else. Bit-identical, because
    "close enough" here would mean the detector was being touched.
    """
    rng = np.random.default_rng(2)
    scores = rng.gamma(shape=4.0, scale=2.0, size=(300, 17)) * 4.0
    rain = synthetic_rain(300)
    nodes = list(range(17))
    reference = scores.copy()
    ref_p = combined_null.to_pvalue(reference).ravel()

    seen_scores, seen_p, taus = {}, {}, {}
    for mode in ("off", "step", "decay", "deployed"):
        gate = RainGate(
            base_threshold=z_q,
            mode=mode,
            multiplier=2.0,
            lookback_hours=12.0,
            decay_hours=36.0,
            wet_mm=0.1,
            sample_hours=1.0,
        )
        taus[mode] = gate.thresholds(rain, nodes=nodes)
        records = gate.decisions(scores, combined_null.to_pvalue(scores), rain, nodes=nodes)
        seen_scores[mode] = np.array([r["score"] for r in records])
        seen_p[mode] = np.array([r["pvalue"] for r in records])
        assert scores.tobytes() == reference.tobytes(), f"{mode} mutated the scores in place"

    for mode in seen_scores:
        assert seen_scores[mode].tobytes() == reference.ravel().tobytes()
        assert seen_p[mode].tobytes() == ref_p.tobytes()
    assert not np.array_equal(taus["off"], taus["decay"]), "no mode moved the threshold"


def test_audit_record_carries_the_whole_decision(z_q, combined_null):
    rain = np.zeros(30)
    rain[5:9] = 1.0
    scores = np.full((30, 2), 1.5 * z_q)
    gate = RainGate(
        base_threshold=z_q, mode="decay", multiplier=2.0, decay_hours=36.0, sample_hours=1.0
    )
    records = gate.decisions(scores, combined_null.to_pvalue(scores), rain, nodes=["n1", "n2"])
    assert len(records) == 60

    required = {
        "score",
        "pvalue",
        "base_threshold",
        "multiplier",
        "threshold",
        "hours_since_wet",
        "fired",
        "node",
        "timestep",
        "mode",
        "scorable",
    }
    assert required <= set(records[0])

    raining = records[2 * 8]  # hour 8, still raining, node n1
    assert raining["multiplier"] == pytest.approx(2.0)
    assert raining["fired"] is False
    assert raining["hours_since_wet"] == 0.0


def test_nan_never_fires_in_either_path(z_q, combined_null):
    """
    The array path and the audit path must agree, including on NaN. They used
    to state the comparison separately, so a change to fired() left decisions()
    untouched.
    """
    tau = np.full((4, 1), z_q)
    scores = np.array([[np.nan], [z_q * 2], [z_q * 0.5], [np.nan]])
    gate = RainGate(base_threshold=z_q, mode="off", sample_hours=1.0)

    assert list(fired(scores, tau).ravel()) == [False, True, False, False]
    records = gate.decisions(scores, combined_null.to_pvalue(scores), np.zeros(4), nodes=["n"])
    assert [r["fired"] for r in records] == [False, True, False, False]
    assert [r["scorable"] for r in records] == [False, True, True, False]

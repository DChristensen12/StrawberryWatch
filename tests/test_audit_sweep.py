"""
Standing audit sweep over the package. One test per recurring defect class.

These classes have each cost real time in this project, so they are checked
mechanically rather than by inspection:

  metrics that cannot vary   a residual correlation sat pinned at -1/(n-1)
                             for weeks before anyone noticed
  masks that do not mask     a node_mask once existed and did nothing live
  duplicated rules           three rain formulas; two copies of the alerting
                             constants; decisions() carrying its own fired()
  fit-at-inference           a scaler refitted on a serving path
  zip length mismatch        silently truncating to the shorter sequence

    .venv/bin/python -m pytest tests/test_audit_sweep.py -q
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch.anomalies import anomaly_detector as ad
from strawberrywatch.config import Config
from tests.audit_helpers import differs

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRS = ("strawberrywatch", "tests", "scripts")


def _grid(n=200):
    return pd.date_range("2026-04-01", periods=n, freq="15min")


# Metrics that cannot vary


def test_rain_multipliers_respond_to_rain_and_to_every_knob():
    g = _grid()
    dry = pd.Series(np.zeros(len(g)), index=g)
    wet = dry.copy()
    wet.iloc[20:40] = 1.0
    args = (12, 2.0, 0.1, 36)
    m_dry, f_dry = ad._rain_multipliers(g, dry, *args)
    m_wet, f_wet = ad._rain_multipliers(g, wet, *args)
    differs(m_dry, m_wet, "_rain_multipliers vs rain")
    differs(f_dry.astype(float), f_wet.astype(float), "_rain_multipliers rain_flags")
    differs(m_wet, ad._rain_multipliers(g, wet, 12, 4.0, 0.1, 36)[0], "vs multiplier")
    differs(m_wet, ad._rain_multipliers(g, wet, 3, 2.0, 0.1, 36)[0], "vs window_hours")
    differs(m_wet, ad._rain_multipliers(g, wet, 12, 2.0, 99.0, 36)[0], "vs amount")
    differs(m_wet, ad._rain_multipliers(g, wet, 12, 2.0, 0.1, 4)[0], "vs decay_hours")

    # decay_hours sets the taper slope as well as how far back the decay branch
    # looks for the last wet reading. Comparing 36 against 72 keeps the rain
    # inside both lookbacks, so only the slope can move the answer, which is
    # what a mutation to the fraction alone has to trip.
    slow = ad._rain_multipliers(g, wet, 12, 2.0, 0.1, 36)[0]
    slower = ad._rain_multipliers(g, wet, 12, 2.0, 0.1, 72)[0]
    tapering = (slow > 1.0) & (slow < 2.0)
    assert tapering.any(), "no timestep is in the taper, so the check is vacuous"
    differs(slow[tapering], slower[tapering], "decay taper slope")


def test_longest_run_varies():
    differs(
        [ad._longest_run(np.array([True, False, True]))],
        [ad._longest_run(np.array([True, True, True]))],
        "_longest_run",
    )


def test_both_rules_vary_with_signal_and_with_their_threshold():
    err = np.zeros(50)
    err[10:20] = 100.0
    quiet = np.zeros(50)
    a = ad._rule1_forecast_residual(quiet, 0.0, 1.0, 5.0, np.ones(50))
    b = ad._rule1_forecast_residual(err, 0.0, 1.0, 5.0, np.ones(50))
    differs([a["n_over_threshold"]], [b["n_over_threshold"]], "_rule1 vs errors")
    c = ad._rule1_forecast_residual(err, 0.0, 1.0, 5.0, np.full(50, 40.0))
    differs([b["n_over_threshold"]], [c["n_over_threshold"]], "_rule1 vs rain multiplier")

    lvl = np.zeros(50)
    lvl[10:20] = 50.0
    d = ad._rule2_level_shift(np.zeros(50), 0.0, 1.0, 3.0)
    e = ad._rule2_level_shift(lvl, 0.0, 1.0, 3.0)
    differs([d["n_over_threshold"]], [e["n_over_threshold"]], "_rule2 vs level")
    f = ad._rule2_level_shift(lvl, 0.0, 1.0, 999.0)
    differs([e["n_over_threshold"]], [f["n_over_threshold"]], "_rule2 vs k")


def test_rain_adjust_level_k_is_a_documented_no_op():
    """
    This one IS constant, deliberately: an explicit pass-through hook, left
    unwired because inheriting Rule 1's multiplier is convenient rather than
    justified. Asserted so that wiring it up trips this test.
    """
    g = _grid(50)
    dry = pd.Series(np.zeros(len(g)), index=g)
    wet = dry.copy()
    wet.iloc[10:30] = 5.0
    assert np.array_equal(
        ad._rain_adjust_level_k(3.0, g, dry), ad._rain_adjust_level_k(3.0, g, wet)
    ), "_rain_adjust_level_k is no longer the documented no-op"
    differs(
        ad._rain_adjust_level_k(3.0, g, dry),
        ad._rain_adjust_level_k(9.0, g, dry),
        "_rain_adjust_level_k vs k",
    )


# Masks that do not mask


def test_real_mask_responds_and_fails_closed():
    ts = _grid(10)
    loc = {"site_a": 0}
    df = pd.DataFrame({"location": ["site_a"] * 10, "conductivity": [1.0] * 10}, index=ts)
    full = ad._extract_real_mask(df, ts, loc)
    assert full.all()

    gappy = df.copy()
    gappy.loc[gappy.index[3:7], "conductivity"] = np.nan
    partial = ad._extract_real_mask(gappy, ts, loc)
    differs(full.astype(float), partial.astype(float), "_extract_real_mask")
    assert partial.sum() == 6

    assert not ad._extract_real_mask(None, ts, loc).any(), (
        "_extract_real_mask should treat unknown as unreal, not as real"
    )
    no_cols = pd.DataFrame({"nothing": [1.0] * 10}, index=ts)
    assert not ad._extract_real_mask(no_cols, ts, loc).any(), "missing columns should fail closed"


# Duplicated rules


def test_alerting_constants_have_one_definition():
    """
    The test module must not carry its own copy of the alerting thresholds.
    It used to, and a test grading against its own copy of a rule stops being
    evidence about production the moment either side moves.
    """
    import tests.test_anomaly_detection as t

    assert t.MIN_TIMESTEPS_OVER_THRESHOLD is ad.MIN_TIMESTEPS_OVER_THRESHOLD
    assert t.MIN_TIMESTEPS_TO_JUDGE is ad.MIN_TIMESTEPS_TO_JUDGE


def test_rain_rule_has_one_definition():
    """
    There were three rain formulas: production's, a different one in this test
    suite, and a third in the design document. The test suite's copy is gone
    and its helper now delegates.
    """
    import tests.test_anomaly_detection as t

    src = inspect.getsource(t._rain_adjusted_thresholds)
    assert "_rain_multipliers(" in src, "the test rain rule stopped delegating"

    # Code only. The docstring explains the history and is allowed to name the
    # arithmetic it replaced; the body is not allowed to perform it.
    fn = ast.parse(textwrap.dedent(src)).body[0]
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    code = "\n".join(ast.unparse(stmt) for stmt in body)
    for restated in ("Timedelta", "saturation", "RAIN_SATURATION_MM", "clip"):
        assert restated not in code, (
            f"the test rain rule is restating production arithmetic ({restated})"
        )


def test_flagged_verdict_has_one_definition():
    import tests.test_anomaly_detection as t

    src = inspect.getsource(t._is_flagged)
    assert "_rule1_forecast_residual" in src, "_is_flagged restates production's verdict"


# Fit-at-inference


def test_serving_path_reuses_the_trained_scaler():
    """
    prepare_sequences_normalized refits only when no scaler is handed to it.
    Every caller that scores against a trained checkpoint must hand one over,
    or the model is evaluated in a different normalised space from the one it
    was trained in.
    """
    from strawberrywatch.preprocessing import data_processor

    src = inspect.getsource(data_processor.prepare_sequences_normalized)
    assert "reuse_scaler" in src, "the reuse branch is gone"

    tree = ast.parse((ROOT / "main.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "prepare_sequences_normalized"
    ]
    assert calls, "main.py no longer calls prepare_sequences_normalized"
    for call in calls:
        assert any(kw.arg == "scaler" for kw in call.keywords), (
            "main.py must pass scaler= so the serving path cannot refit"
        )

    tree = ast.parse((ROOT / "scripts" / "deployment_readiness.py").read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "prepare_sequences_normalized"
    ]
    for call in calls:
        assert any(kw.arg == "scaler" for kw in call.keywords), (
            "deployment_readiness scores a trained model against a refitted scaler"
        )


# zip() length mismatch


def test_no_unguarded_zip_anywhere_editable():
    """
    Every zip over sequences that must be the same length needs strict=True,
    or a length mismatch truncates silently instead of raising.
    """
    offenders = []
    for d in PACKAGE_DIRS:
        for path in (ROOT / d).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "zip"):
                    continue
                if len(node.args) < 2:
                    continue  # zip of one iterable cannot mismatch
                if not any(kw.arg == "strict" for kw in node.keywords):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders, "zip() without strict=: " + ", ".join(offenders)


def test_rain_config_has_no_orphan_keys():
    """
    RAIN_SATURATION_MM was the upper end of an amount-scaled ramp nothing
    implemented. Every rain knob left on Config must have a reader.
    """
    readers = "\n".join(p.read_text() for d in PACKAGE_DIRS for p in (ROOT / d).rglob("*.py"))
    for name in [n for n in dir(Config) if n.startswith("RAIN_") or "RAIN" in n]:
        assert f"Config.{name}" in readers or f"{name}" in readers, f"Config.{name} has no reader"
    assert not hasattr(Config, "RAIN_SATURATION_MM"), "RAIN_SATURATION_MM is back without a reader"


def test_synthetic_generator_never_refits_a_supplied_scaler():
    """
    creek_synthetic fits a NodeScaler only when the caller omits one. Fault
    windows must reuse the training scaler: normalising them with statistics
    computed from data containing the fault shrinks the signal the sweep is
    there to measure.

    Object identity, so a silent refit cannot pass as merely equal.
    """
    from tests import creek_synthetic

    train = creek_synthetic.make_synthetic_creek(n_steps=120, seed=1)
    assert train["scaler"] is not None
    for seed in (2, 3):
        got = creek_synthetic.make_synthetic_creek(n_steps=120, seed=seed, scaler=train["scaler"])
        assert got["scaler"] is train["scaler"], "generator refitted a supplied scaler"


def test_calibration_is_loaded_not_fitted():
    """
    The serving path reads its nulls and threshold off disk. If this module
    ever grows a fit call, the artifact stops being what production runs on.
    """
    import inspect

    from strawberrywatch.anomalies import cobble_calibration

    src = inspect.getsource(cobble_calibration)
    assert ".fit(" not in src, "cobble_calibration started fitting something"
    assert "json.loads" in src, "calibration is no longer read from the artifact"

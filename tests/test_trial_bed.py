"""
What Trial Bed says, and that saying it costs the detector nothing.

The table is the hard part, so the match tests are built out of the table
itself rather than out of a copy of it. A row edited by the Water Quality team
changes what these assert, which is the point: a test carrying its own copy of
the directions would keep passing after the table stopped agreeing with it.
"""

from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pandas as pd
import pytest

from strawberrywatch.anomalies.rain_gate import RainGate
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.support_modules import SupportStack, base
from strawberrywatch.support_modules import spill_signatures as sig
from strawberrywatch.support_modules.trial_bed import (
    CANNOT_EVALUATE,
    DIAGNOSED,
    NOTHING_TO_EXPLAIN,
    TrialBed,
    classify,
    observed_direction,
)

BASE_THRESHOLD = 39.13047989749433
NODES = ["n1", "n2", "n3"]
T = 12
EVENT_STEPS = (8, 9, 10, 11)

# What the deployed network measures. Config.SCORED_TARGET_FEATURES is the same
# three, and depth is not in the signature table at all.
DEPLOYED_CHANNELS = ("conductivity", "depth", "temperature")

# The opposite of every direction the table asserts, for the row that must not
# match. FLAT has no opposite, so it moves.
OPPOSITE = {sig.UP: sig.DOWN, sig.DOWN: sig.UP, sig.FLAT: sig.UP}


def a_gate():
    return RainGate(base_threshold=BASE_THRESHOLD, mode="off", sample_hours=1.0)


def a_run():
    """Scores, p-values and rain where n1 fires over EVENT_STEPS and nothing else does."""
    scores = np.full((T, len(NODES)), 0.5 * BASE_THRESHOLD)
    for i in EVENT_STEPS:
        scores[i, 0] = 2.0 * BASE_THRESHOLD
    return scores, np.full_like(scores, 0.5), np.zeros(T)


def a_window(channels=None, event_node="n1"):
    """
    A raw frame where one node carries a sewage pattern over EVENT_STEPS.

    channels limits it to a subset, which is how the deployed network's three
    columns are put in front of a table that wants five.
    """
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-04-01", periods=T, freq="h", tz="UTC")

    # Baseline level then event level, for the sewage row: temperature up,
    # dissolved oxygen down, conductivity up, floating conductivity up.
    levels = {
        "conductivity": (300.0, 400.0),
        "floating_conductivity": (300.0, 400.0),
        "temperature": (15.0, 17.0),
        "dissolved_oxygen": (8.0, 4.0),
        "ph": (7.5, 7.5),
        "depth": (0.4, 0.4),
    }
    wanted = list(levels) if channels is None else list(channels)

    rows = []
    for node in NODES:
        for i, when in enumerate(times):
            row = {"location": node}
            during = node == event_node and i in EVENT_STEPS
            for name in wanted:
                base_level, event_level = levels[name]
                centre = event_level if during else base_level
                row[name] = centre + rng.normal(0.0, abs(centre) * 0.01 + 1e-3)
            rows.append((when, row))

    raw = pd.DataFrame([r for _, r in rows], index=pd.DatetimeIndex([t for t, _ in rows]))
    return {"raw": raw, "timestamps": list(times)}


def firings(records):
    """The set of node-timesteps that came out as FIRED."""
    return {(r["timestep"], r["node"]) for r in records if r["verdict"] == base.FIRED}


# Where metrics.py went


def test_metrics_is_gone_and_nothing_imports_it():
    """
    metrics.py was stranded: nothing on the live path built a classification,
    so there was no behaviour to preserve and no output to pin the port
    against. Its logic is here now, so the module must not still be importable
    beside it, or the table would exist in two places.
    """
    assert importlib.util.find_spec("strawberrywatch.anomalies.metrics") is None


# Matching, built out of the table


@pytest.mark.parametrize("pollutant", sorted(sig.SIGNATURES))
def test_a_signature_matches_its_own_pattern(pollutant):
    """Feed a row exactly what it asserts and it has to name itself."""
    observed = {p: sig.SIGNATURES[pollutant][p][0] for p in sig.required_parameters(pollutant)}
    result = classify(observed)

    assert result["verdict"] == DIAGNOSED
    assert result["cause"] == pollutant
    assert result["ranked"][0]["pollutant"] == pollutant
    assert result["ranked"][0]["score"] == pytest.approx(1.0)
    assert f"consistent with {pollutant}" in result["explanation"]


@pytest.mark.parametrize("pollutant", sorted(sig.SIGNATURES))
def test_a_signature_does_not_match_its_own_opposite(pollutant):
    """Reverse every direction a row asserts and that row must score zero."""
    observed = {
        p: OPPOSITE[sig.SIGNATURES[pollutant][p][0]] for p in sig.required_parameters(pollutant)
    }
    result = classify(observed)

    assert result["cause"] != pollutant
    by_name = {r["pollutant"]: r for r in result["ranked"]}
    assert by_name[pollutant]["score"] == pytest.approx(0.0)


# Degrading honestly


@pytest.mark.parametrize("pollutant", sorted(sig.SIGNATURES))
def test_a_row_short_of_one_parameter_cannot_be_evaluated(pollutant):
    """
    Hold back one parameter a row needs and the row drops out, naming what was
    missing. Scoring it on the rest would report a partial signature at the
    same confidence as a whole one.
    """
    required = sig.required_parameters(pollutant)
    withheld = required[-1]
    observed = {p: sig.SIGNATURES[pollutant][p][0] for p in required[:-1]}

    result = classify(observed)
    assert pollutant not in [r["pollutant"] for r in result["ranked"]]
    assert result["unevaluable"][pollutant] == [withheld]
    assert result["cause"] != pollutant


def test_three_parameters_of_a_row_that_needs_four_cannot_be_evaluated():
    """The whole verdict, not just the row: two of sewage's four is not a match."""
    required = sig.required_parameters("sewage")
    assert len(required) >= 3, "the table changed shape; this case needs a longer row"

    observed = {p: sig.SIGNATURES["sewage"][p][0] for p in required[:2]}
    result = classify(observed)

    assert result["verdict"] == CANNOT_EVALUATE
    assert result["cause"] is None
    assert result["ranked"] == []
    assert "cannot evaluate" in result["explanation"]
    assert set(sig.missing_parameters("sewage", observed)) == set(required[2:])


def test_no_signature_is_evaluable_on_the_deployed_channels():
    """
    Conductivity, depth and temperature is what the network measures today.
    Every row needs dissolved oxygen and floating conductivity, and depth is
    not in the table at all, so nothing can be judged. Pinned as a number so a
    table edit that changes it has to come past this test.
    """
    assert sig.evaluable_on(DEPLOYED_CHANNELS) == []
    assert len(sig.SIGNATURES) == 5

    for pollutant in sig.SIGNATURES:
        missing = set(sig.required_parameters(pollutant)) - set(DEPLOYED_CHANNELS)
        assert missing, f"{pollutant} became evaluable without the group editing the table"


def test_the_deployed_channels_produce_cannot_evaluate_not_a_guess():
    """End to end on three columns: it declines, and says which channels are absent."""
    s, p, rain = a_run()
    window = a_window(channels=DEPLOYED_CHANNELS)
    stack = SupportStack([TrialBed()])
    records = stack.decisions(a_gate(), s, p, rain, NODES, window=window)

    entry = next(
        e for r in records if r["node"] == "n1" for e in r["support"] if e["name"] == "trial_bed"
    )
    assert entry["cause"] is None
    assert "cannot evaluate" in entry["explanation"]
    assert "dissolved oxygen not measured" in entry["explanation"]
    assert entry["confidence"].startswith("none")


def test_a_node_that_never_fired_has_nothing_to_explain():
    """No firing is not an all-clear on the pollutant type, it is no question asked."""
    window = a_window()
    fired = np.zeros((T, len(NODES)), dtype=bool)
    fired[list(EVENT_STEPS), 0] = True

    accounts = TrialBed().explain(window, NODES, fired)
    assert accounts[0]["verdict"] == DIAGNOSED
    for account in accounts[1:]:
        assert account["verdict"] == NOTHING_TO_EXPLAIN
        assert account["cause"] is None


# The table is actually read


def test_flipping_one_direction_in_the_table_changes_the_answer():
    """
    A matcher that had drifted off the table would keep naming sewage after
    sewage stopped saying what it says. Mutated in a copy, so the shipped table
    is untouched either way.
    """
    observed = {p: sig.SIGNATURES["sewage"][p][0] for p in sig.required_parameters("sewage")}
    assert classify(observed)["cause"] == "sewage"

    mutated = copy.deepcopy(sig.SIGNATURES)
    direction, magnitude = mutated["sewage"]["conductivity"]
    mutated["sewage"]["conductivity"] = (OPPOSITE[direction], magnitude)

    after = classify(observed, table=mutated)
    assert after["cause"] != "sewage" or after["ranked"][0]["score"] < 1.0
    assert classify(observed)["cause"] == "sewage", "the shipped table was mutated"


def test_a_direction_is_read_off_the_baselines_own_spread():
    """A move under one baseline standard deviation is not a move."""
    baseline = [10.0, 10.5, 9.5, 10.2, 9.8]
    assert observed_direction(baseline, [10.1, 10.0]) == sig.FLAT
    assert observed_direction(baseline, [14.0, 14.2]) == sig.UP
    assert observed_direction(baseline, [6.0, 5.8]) == sig.DOWN
    assert observed_direction([10.0], [14.0]) is None


# It spends no false alarm budget


def test_attaching_trial_bed_changes_no_firing():
    """
    The same window, run with the explainer and without it. Every threshold,
    every primary decision and the set of firing node-timesteps must come out
    identical, or an explainer has quietly become a test.
    """
    s, p, rain = a_run()
    window = a_window()

    without = SupportStack([]).decisions(a_gate(), s, p, rain, NODES, window=window)
    with_bed = SupportStack([TrialBed()]).decisions(a_gate(), s, p, rain, NODES, window=window)

    assert firings(without) == firings(with_bed)
    assert firings(without), "vacuous if nothing fires at all"

    for a, b in zip(without, with_bed, strict=True):
        for key in ("timestep", "node", "score", "threshold", "fired", "scorable"):
            assert repr(a[key]) == repr(b[key]), f"{key} moved when the explainer attached"
        assert repr(a["final_threshold"]) == repr(b["final_threshold"])
        assert a["primary_fired"] is b["primary_fired"]
        assert a["verdict"] == b["verdict"]

    # And it did have something to say, so the comparison above was not vacuous.
    entry = next(e for e in with_bed[0]["support"] if e["name"] == "trial_bed")
    assert entry["budget_q"] is None and entry["fired"] is None


def test_trial_bed_leaves_the_whole_budget_to_the_primary():
    split = SupportStack([TrialBed()]).budget()
    assert split["detectors"] == {}
    assert split["primary"] == pytest.approx(split["q_total"])


# Model agnostic


@pytest.mark.parametrize(
    "model", [DuskCrayfish, CobbleShoal], ids=["dusk_crayfish", "cobble_shoal"]
)
def test_the_same_readings_give_the_same_account_on_both_models(model):
    """
    Nothing model-specific is threaded through. It reads the raw frame and the
    timestamps, which main.py builds the same way whichever model ran.
    """
    s, p, rain = a_run()
    window = a_window()
    stack = SupportStack.load(["trial_bed"], model)
    records = stack.decisions(a_gate(), s, p, rain, NODES, window=window)

    entry = next(e for e in records[0]["support"] if e["name"] == "trial_bed")
    assert entry["cause"] == "sewage"
    assert entry["evidence"]["conductivity"] == sig.UP
    assert entry["evidence"]["dissolved_oxygen"] == sig.DOWN
    assert "consistent with sewage" in entry["explanation"]


def test_the_account_names_the_cause_the_evidence_and_the_confidence():
    """One line an operator can act on, not a label."""
    s, p, rain = a_run()
    window = a_window()
    records = SupportStack([TrialBed()]).decisions(a_gate(), s, p, rain, NODES, window=window)

    entry = next(e for e in records[0]["support"] if e["name"] == "trial_bed")
    assert entry["cause"] == "sewage"
    assert entry["confidence"].startswith("good")
    for phrase in ("conductivity up", "temperature up", "dissolved oxygen down"):
        assert phrase in entry["explanation"]

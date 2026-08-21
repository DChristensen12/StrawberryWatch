"""
Newtwork Run: reading transport off firing order, and the control that refutes it.

Ordering only. Nothing here supplies a travel time, because nothing in the repo
has measured one, and the module has to be useful without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.support_modules import SupportStack, base
from strawberrywatch.support_modules.newtwork_run import (
    CATCHMENT_WIDE,
    LOCAL,
    NOT_TRANSPORT,
    NOTHING_TO_EXPLAIN,
    TRANSPORT,
    TRAVEL_TIMES,
    NewtworkRun,
    travel_window,
)

T = 24
SOUTH = ["south_fork_1", "south_fork_2", "south_fork_3", "oxford"]


def grid(nodes, firings):
    """firings is {node: timestep}."""
    out = np.zeros((T, len(nodes)), dtype=bool)
    for node, step in firings.items():
        out[step, nodes.index(node)] = True
    return out


def verdicts(nodes, firings):
    accounts = NewtworkRun().explain({}, nodes, grid(nodes, firings))
    return dict(zip(nodes, [a["verdict"] for a in accounts], strict=True))


def test_it_is_an_explainer_and_takes_no_budget():
    run = NewtworkRun()
    assert run.kind == "explainer"
    assert isinstance(run, base.SupportExplainer)
    assert not hasattr(run, "score")
    assert not hasattr(run, "multipliers")
    assert not hasattr(run, "admit")

    stack = SupportStack.load(["newtwork_run"], CobbleShoal)
    assert stack.budget()["detectors"] == {}
    assert stack.budget()["primary"] == pytest.approx(stack.q_total)


def test_upstream_before_downstream_reads_as_transport():
    seen = verdicts(SOUTH, {"south_fork_1": 2, "south_fork_2": 5, "oxford": 9})
    assert seen["south_fork_1"] == TRANSPORT
    assert seen["south_fork_2"] == TRANSPORT
    assert seen["oxford"] == TRANSPORT
    assert seen["south_fork_3"] == NOTHING_TO_EXPLAIN


def test_everything_at_once_reads_as_catchment_wide():
    seen = verdicts(SOUTH, {site: 4 for site in SOUTH})
    assert set(seen.values()) == {CATCHMENT_WIDE}


def test_one_site_alone_reads_as_local():
    seen = verdicts(SOUTH, {"south_fork_2": 6})
    assert seen["south_fork_2"] == LOCAL


def test_the_order_is_what_decides_it():
    """Reverse the order and the same sites stop looking like transport."""
    forward = verdicts(SOUTH, {"south_fork_1": 2, "oxford": 8})
    simultaneous = verdicts(SOUTH, {"south_fork_1": 2, "oxford": 2})

    assert forward["oxford"] == TRANSPORT
    assert simultaneous["oxford"] == CATCHMENT_WIDE
    assert forward != simultaneous


def test_codornices_firing_at_the_same_step_rules_out_transport():
    """
    Item 23. Codornices is a separate watershed, so a signal at codornices and
    at a Strawberry Creek site at the same timestep cannot have travelled.
    """
    nodes = [*SOUTH, "codornices"]
    seen = verdicts(nodes, {"south_fork_1": 3, "south_fork_2": 3, "codornices": 3})

    assert seen["south_fork_1"] == NOT_TRANSPORT
    assert seen["south_fork_2"] == NOT_TRANSPORT
    assert "codornices" in seen


def test_the_control_only_bites_when_it_fires_together():
    nodes = [*SOUTH, "codornices"]
    together = verdicts(nodes, {"south_fork_1": 3, "south_fork_2": 3, "codornices": 3})
    apart = verdicts(nodes, {"south_fork_1": 3, "south_fork_2": 3, "codornices": 11})

    assert together["south_fork_1"] == NOT_TRANSPORT
    assert apart["south_fork_1"] != NOT_TRANSPORT


def test_the_control_never_accuses_itself():
    nodes = ["codornices"]
    seen = verdicts(nodes, {"codornices": 5})
    assert seen["codornices"] == LOCAL


# The travel time seam


def test_no_travel_times_are_supplied_and_the_module_says_so():
    """Item 24. The seam is documented and empty, never filled with a guess."""
    assert TRAVEL_TIMES == {}
    assert travel_window("south_fork_1", "south_fork_2") is None

    accounts = NewtworkRun().explain({}, SOUTH, grid(SOUTH, {"south_fork_1": 1, "oxford": 5}))
    assert all(
        "ordering only" in a["confidence"] for a in accounts if a["verdict"] != NOTHING_TO_EXPLAIN
    )


def test_a_supplied_travel_time_is_picked_up_without_a_code_change(monkeypatch):
    monkeypatch.setitem(
        TRAVEL_TIMES,
        ("south_fork_1", "south_fork_2"),
        {"unknown": {"minutes_to_peak": 30, "tolerance_minutes": 10}},
    )
    assert travel_window("south_fork_1", "south_fork_2")["minutes_to_peak"] == 30

    accounts = NewtworkRun().explain({}, SOUTH, grid(SOUTH, {"south_fork_1": 1, "south_fork_2": 5}))
    sf2 = accounts[SOUTH.index("south_fork_2")]
    assert "measured travel times" in sf2["confidence"]


# It cannot move a detection


@pytest.mark.parametrize(
    "model", [DuskCrayfish, CobbleShoal], ids=["dusk_crayfish", "cobble_shoal"]
)
def test_it_spends_no_false_alarm_budget_on_either_model(model):
    """
    Item 25. The set of firing node-timesteps has to be identical with the
    module attached and without it.
    """
    from strawberrywatch.anomalies.rain_gate import RainGate

    nodes = ["south_fork_1", "south_fork_2", "oxford"]
    rng = np.random.default_rng(7)
    scores = rng.normal(40, 5, (T, len(nodes)))
    pvalues = rng.uniform(0, 1, (T, len(nodes)))
    rain = np.zeros(T)

    gate = RainGate(base_threshold=39.13047989749433)
    without = SupportStack([]).decisions(gate, scores, pvalues, rain, nodes, window={})
    with_module = SupportStack.load(["newtwork_run"], model).decisions(
        gate, scores, pvalues, rain, nodes, window={}
    )

    fired_without = {(r["timestep"], r["node"]) for r in without if r["verdict"] == base.FIRED}
    fired_with = {(r["timestep"], r["node"]) for r in with_module if r["verdict"] == base.FIRED}
    assert fired_without == fired_with

    # and the record says explicitly that switching it off changes nothing
    for record in with_module:
        assert record["verdict_without"]["newtwork_run"] == record["verdict"]


def test_a_site_the_inventory_does_not_describe_says_so_rather_than_guessing():
    nodes = ["south_fork_1", "made_up_creek"]
    accounts = NewtworkRun().explain({}, nodes, grid(nodes, {"made_up_creek": 3}))
    assert accounts[1]["verdict"] == "cannot_evaluate"
    assert "inventory" in accounts[1]["explanation"]


def test_the_flow_graph_it_reads_is_the_inventory_and_not_a_hardcoded_list():
    """Swap the inventory and the verdicts follow it."""
    import copy as copylib

    swapped = copylib.deepcopy(inv.load())
    # cut south_fork_1 off from the chain, and its lead stops meaning transport
    swapped.sites["south_fork_1"].downstream = "oxford"
    swapped.sites["south_fork_2"].downstream = "oxford"

    default = NewtworkRun()
    rewired = NewtworkRun(inventory=swapped)
    firings = grid(SOUTH, {"south_fork_1": 2, "south_fork_3": 6})

    a = [x["evidence"].get("upstream_before") for x in default.explain({}, SOUTH, firings)]
    b = [x["evidence"].get("upstream_before") for x in rewired.explain({}, SOUTH, firings)]
    assert a != b

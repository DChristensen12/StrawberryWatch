"""
Settling Pool: the tests it runs, the flags it emits, and the layering it keeps.

The layering test is the load bearing one. The split works because the tests
know nothing about where the data came from, and a single import of the config
into qc_tests.py would quietly undo that.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.models import contracts
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
from strawberrywatch.support_modules import qc_tests as qc
from strawberrywatch.support_modules.settling_pool import SettlingPool, flag_one

SITES = ["south_fork_1", "south_fork_2", "oxford"]
T = 48


def index(n=T):
    return pd.date_range("2026-03-01", periods=n, freq="15min", tz="UTC")


def frame(site, variable, values, idx=None):
    idx = index(len(values)) if idx is None else idx
    return pd.DataFrame({variable: values, "location": site}, index=idx)


# Layering


def test_the_tests_import_nothing_from_the_config_or_the_storage_layer():
    """
    Item 9, read off the parsed tree rather than the text so a comment naming
    the config does not trip it.
    """
    tree = ast.parse(inspect.getsource(qc))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    for module in imported:
        assert not module.startswith("strawberrywatch"), (
            f"qc_tests imports {module}; the test layer must not reach the config, "
            f"the inventory or the storage layer"
        )


def test_no_test_function_corrects_or_cleans_anything():
    """Every public test returns flags and leaves its input alone."""
    values = np.array([1.0, 2.0, -9999.0, 4.0, 5.0])
    original = values.copy()

    for name in ("gross_range", "sentinel", "flat_line", "spike", "rate_of_change"):
        fn = getattr(qc, name)
        if name == "gross_range":
            out = fn(values, (0, 100), (0, 10))
        elif name == "sentinel":
            out = fn(values)
        elif name == "flat_line":
            out = fn(values, 2, 4)
        elif name == "spike":
            out = fn(values, 1.0)
        else:
            out = fn(values, 1.0, 1)
        assert out.dtype == np.uint8
        assert set(np.unique(out)) <= set(qc.FLAGS)
        np.testing.assert_array_equal(values, original)


# The tests themselves, each against its failure mode


def test_gross_range_separates_suspect_from_fail():
    values = np.array([5.0, 50.0, 500.0])
    flags = qc.gross_range(values, fail_range=(0, 100), suspect_range=(0, 10))
    assert list(flags) == [qc.GOOD, qc.SUSPECT, qc.FAIL]


def test_flat_line_catches_a_sensor_that_stopped_changing():
    flags = qc.flat_line(np.full(12, 3.3), suspect_count=3, fail_count=6)
    assert flags[0] == qc.GOOD
    assert qc.SUSPECT in flags and qc.FAIL in flags


def test_attenuated_signal_catches_the_near_but_not_flat_case():
    """
    A fouled probe still moves. It moves by far less than the creek, which is
    what a flat line test cannot see.
    """
    rng = np.random.default_rng(0)
    lively = rng.normal(0, 5.0, 64)
    fouled = rng.normal(0, 0.01, 64)

    assert qc.SUSPECT not in qc.attenuated_signal(lively, 24, min_std=0.5)
    assert qc.SUSPECT in qc.attenuated_signal(fouled, 24, min_std=0.5)
    # and it is not just the flat line test wearing a hat
    assert qc.SUSPECT not in qc.flat_line(fouled, 24, 96)


def test_spike_catches_a_single_step_excursion():
    values = np.array([10.0, 10.1, 90.0, 10.2, 10.1])
    flags = qc.spike(values, threshold=5.0)
    assert flags[2] in (qc.SUSPECT, qc.FAIL)


def test_rate_of_change_catches_sustained_movement():
    slow = np.linspace(0, 40, 40)
    fast = np.linspace(0, 4000, 40)
    assert qc.SUSPECT not in qc.rate_of_change(slow, threshold=50.0, window=4)
    assert qc.SUSPECT in qc.rate_of_change(fast, threshold=50.0, window=4)


def test_sentinel_names_the_wrong_sdi12_address():
    flags = qc.sentinel(np.array([120.0, -9999.0, 118.0]))
    assert list(flags) == [qc.GOOD, qc.FAIL, qc.GOOD]


def test_duplicate_feed_catches_one_site_reporting_another():
    a = np.arange(100, dtype=float)
    same = a.copy()
    different = a + 0.5

    assert qc.SUSPECT in qc.duplicate_feed(a, same, match_fraction=0.25)
    assert qc.SUSPECT not in qc.duplicate_feed(a, different, match_fraction=0.25)


def test_staleness_uses_the_gap_it_is_given():
    stamps = np.arange(10, dtype=float) * 900.0
    present = np.zeros(10, dtype=bool)
    present[0] = True

    flags = qc.staleness(stamps, present, max_gap_seconds=2 * 3600)
    assert flags[0] == qc.GOOD
    assert flags[8] == qc.SUSPECT
    assert flags[9] == qc.FAIL


def test_worst_ranks_by_severity_not_by_flag_value():
    good = np.array([qc.GOOD] * 3, dtype=np.uint8)
    unknown = np.array([qc.UNKNOWN] * 3, dtype=np.uint8)
    # UNKNOWN is 2 and GOOD is 1, so a plain maximum would pick the wrong one
    assert list(qc.worst(good, unknown)) == [qc.GOOD] * 3


# Reading the inventory


def test_thresholds_come_from_the_inventory_and_not_from_code():
    """Move a threshold in a copy of the inventory and the flags move with it."""
    import copy as copylib

    # varying, so the flat line test stays out of the comparison
    rng = np.random.default_rng(1)
    values = 500.0 + rng.normal(0, 3.0, T)
    idx = index()

    loose_inv = inv.load()
    loose, per_test = flag_one(values, idx, "south_fork_1", "conductivity", loose_inv)

    tightened = copylib.deepcopy(loose_inv)
    tightened.site("south_fork_1").thresholds["conductivity"]["suspect_range"] = [0, 10]
    tight, tight_per_test = flag_one(values, idx, "south_fork_1", "conductivity", tightened)

    assert not np.array_equal(loose, tight)
    assert qc.SUSPECT not in per_test["gross_range"]
    assert set(tight_per_test["gross_range"]) == {qc.SUSPECT}


def test_a_never_installed_sensor_is_unknown_rather_than_missing():
    """
    QARTOD 2 means not evaluated, which is what a probe that was never in the
    creek deserves. Calling it MISSING would report a fault nobody can fix.
    """
    # south_fork_3 declares a pH probe with a null install date, so the
    # inventory knows the channel exists and knows nothing was ever fitted
    pool = SettlingPool()
    flags = pool.flags({"raw": None, "timestamps": index()}, ["south_fork_3"])
    assert set(flags[("south_fork_3", "ph")]) == {qc.UNKNOWN}
    assert qc.MISSING not in flags[("south_fork_3", "ph")]


def test_flags_are_keyed_by_site_and_variable_and_nothing_else():
    """Item 13. Neither model's tensor layout may reach this module."""
    pool = SettlingPool()
    flags = pool.flags({"raw": None, "timestamps": index()}, SITES)
    for key in flags:
        assert isinstance(key, tuple) and len(key) == 2
        site, variable = key
        assert site in SITES
        assert variable in inv.load().site(site).variables


# The policy


def test_both_models_declare_the_same_default_policy():
    assert contracts.flag_policy(DuskCrayfish) == (qc.FAIL, qc.MISSING)
    assert contracts.flag_policy(CobbleShoal) == (qc.FAIL, qc.MISSING)


def test_a_policy_that_excludes_good_is_refused():
    class Blind:
        INPUT_CONTRACT = contracts.SEQUENCE_TENSOR
        FLAG_POLICY = (qc.GOOD, qc.FAIL)

    with pytest.raises(contracts.FlagPolicyError, match="excludes GOOD"):
        contracts.flag_policy(Blind)


def test_enforcing_the_policy_removes_exactly_the_flagged_cells_and_nothing_else():
    """
    Item 15. Run the same window with and without enforcement and the
    difference has to be the flagged cells, cell for cell.
    """
    flags = {
        ("south_fork_1", "conductivity"): np.full(T, qc.GOOD, dtype=np.uint8),
        ("south_fork_2", "conductivity"): np.full(T, qc.GOOD, dtype=np.uint8),
        ("oxford", "conductivity"): np.full(T, qc.GOOD, dtype=np.uint8),
    }
    flags[("south_fork_2", "conductivity")][5] = qc.FAIL
    flags[("oxford", "conductivity")][9] = qc.MISSING
    flags[("south_fork_1", "conductivity")][11] = qc.SUSPECT

    enforced = contracts.admit_grid(flags, SITES, T, DuskCrayfish)

    expected = np.ones((T, len(SITES)), dtype=bool)
    expected[5, 1] = False
    expected[9, 2] = False

    np.testing.assert_array_equal(enforced, expected)
    # SUSPECT survives on purpose: usable, with the flag recorded
    assert enforced[11, 0]


def test_flipping_a_flag_moves_the_admitted_grid():
    """Item 27, on the mask this module hands the model."""
    clean = {("south_fork_1", "conductivity"): np.full(T, qc.GOOD, dtype=np.uint8)}
    dirty = {("south_fork_1", "conductivity"): np.full(T, qc.GOOD, dtype=np.uint8)}
    dirty[("south_fork_1", "conductivity")][3] = qc.FAIL

    before = contracts.admit_grid(clean, ["south_fork_1"], T, DuskCrayfish)
    after = contracts.admit_grid(dirty, ["south_fork_1"], T, DuskCrayfish)

    assert before.all()
    assert not after.all()
    assert (before != after).sum() == 1


def test_the_screen_takes_no_budget():
    from strawberrywatch.support_modules import SupportStack

    stack = SupportStack.load(["settling_pool"], CobbleShoal)
    assert stack.budget()["detectors"] == {}
    assert stack.budget()["primary"] == pytest.approx(stack.q_total)

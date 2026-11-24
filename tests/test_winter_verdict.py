"""The winter verdict's scoring on a small made-up record: a win, a loss and too few storms."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import winter_scoring as W

ROOT = Path(__file__).resolve().parent.parent
# The only code the storm-mode check keeps: the winter verdict and its module
WINTER_CODE = {"scripts/winter_verdict.py", "scripts/winter_scoring.py"}
FROZEN = W.load_frozen()
Z = FROZEN["switch"]
C_PRE, D_PRE, D_TOP = 500.0, 100.0, 200.0
# The mixing method's fit, as frozen for the three core sites, and one that expects the wrong dilution
FITTED = {"c_ro": 89.0, "beta": 2.35, "cutoff": 0.01}
WRONG = {"c_ro": 89.0, "beta": 0.5, "cutoff": 0.01}


def _record(n_storms, days_apart=7):
    """Core sites at baseflow, and a storm every days_apart days in which conductivity follows the mixing curve."""
    start = pd.Timestamp("2026-12-01", tz="UTC")
    index = pd.date_range(start, periods=(n_storms * days_apart + 8) * 96, freq="15min")
    depth = np.full(len(index), D_PRE)
    rain = pd.Series(0.0, index)
    for k in range(n_storms):
        t0 = 8 * 96 + k * days_apart * 96
        rise, fall = np.linspace(D_PRE, D_TOP, 9), np.linspace(D_TOP, D_PRE, 49)
        depth[t0 : t0 + 9] = rise
        depth[t0 + 9 : t0 + 58] = fall
        rain.iloc[t0 - 8 : t0 + 4] = 1.0
    cond = FITTED["c_ro"] + (C_PRE - FITTED["c_ro"]) * (D_PRE / depth) ** FITTED["beta"]
    frame = pd.DataFrame({W.COND: cond, W.DEPTH: depth, W.TEMP: 12.0}, index=index)
    tables = {s: frame.copy() for s in W.CORE}
    return tables, rain, index


def _scores(tables, index, place):
    """Dusk Crayfish's made-up scores: always under its line, and over its dry line but not its rain line at a fault's site."""
    clean = pd.concat(
        [pd.DataFrame({"site": s, "time": index, "score": 0.2, "line": 1.0}) for s in tables],
        ignore_index=True,
    )
    runs = {}
    for row in place.itertuples():
        faulted, marks = W.inject(
            tables, row.node, row.start, row.shape, row.size, row.sign, seed=0
        )
        if faulted is None:
            continue
        sites = {
            s: pd.DataFrame({"time": list(on), "score": 2.5, "line": 1.0})
            for s, on in marks.items()
        }
        runs[row.id] = {"marks": {s: list(on) for s, on in marks.items()}, "sites": sites}
    return clean, runs


def _water_strider(index, groups):
    """One Water Strider alert in every other storm group, at north_fork_0."""
    ws = pd.DataFrame(
        {
            "site": "north_fork_0",
            "variable": "conductivity",
            "rule": "r",
            "time": index,
            "score": 0.5,
            "line": 1.0,
        }
    )
    for g in groups[::2]:
        ws.loc[ws["time"] == g["answers"][0], "score"] = 2.0
    return ws


def _verdict(n_storms, fit):
    tables, rain, index = _record(n_storms)
    lo, hi = index[0], index[-1]
    groups = W.storm_groups(tables, [], Z, lo, hi)
    place = W.placements(tables, groups, [], rain >= W.WET)
    clean, runs = _scores(tables, index, place)
    ws = _water_strider(index, groups)
    ok = {s: pd.Series(True, index) for s in W.CORE}
    scales = {"s_current": {"all": 2.55}, "s_storm": {"all": FROZEN["s_storm"]}}
    results = {}
    for method in ("current", "mixing"):
        fits = {s: {"all": fit} for s in W.CORE} if method == "mixing" else {}
        results[method] = W.evaluate(
            method,
            clean,
            {"placements": place, "runs": runs},
            tables,
            rain,
            groups,
            [],
            ws,
            ok,
            Z,
            fits,
            scales,
        )
    sums = {}
    for ox in ("with", "without"):
        current = {s: results["current"] for s in W.SEEDS}
        sums[("mixing", ox)] = W.summary({s: results["mixing"] for s in W.SEEDS}, current)
        # the depth fit is left out here, so it never wins
        sums[("depth_fit", ox)] = {**sums[("mixing", ox)], "met": {k: False for k in range(1, 6)}}
    return W.decide(sums, len(groups))[0], groups, results


@pytest.fixture(scope="module")
def won():
    return _verdict(10, FITTED)


def test_a_storm_every_week_makes_one_storm_group_each(won):
    _, groups, results = won
    assert len(groups) == 10
    assert results["mixing"]["storm_placements"] == 180 and results["mixing"]["dry_placements"] > 0


def test_storm_mode_that_follows_the_storm_wins(won):
    verdict, _, results = won
    m, current = results["mixing"], results["current"]
    assert verdict == "mixing"
    assert m["rain_fa"] == 0 and m["rain_only_alerted"] == 0
    assert (
        m["storm_detect"] > 0.75 * m["dry_detect"]
        and current["storm_detect"] < 0.75 * current["dry_detect"]
    )


def test_storm_mode_that_expects_the_wrong_dilution_loses():
    verdict, _, results = _verdict(10, WRONG)
    assert verdict == "current model"
    assert results["mixing"]["rain_fa"] > results["mixing"]["ws_rain_fa"]


def test_fewer_than_ten_storm_groups_is_inconclusive():
    verdict, groups, _ = _verdict(3, FITTED)
    assert len(groups) == 3 and verdict == "inconclusive"


def test_a_tie_goes_to_a():
    row = {
        "ratio": 1.2,
        "ratio_sd": 0.05,
        "ratio_current": 0.2,
        "ratio_current_sd": 0.05,
        "met": {k: True for k in range(1, 6)},
    }
    sums = {
        (m, ox): dict(row, ratio=1.2 if m == "depth_fit" else 1.23)
        for m in ("depth_fit", "mixing")
        for ox in ("with", "without")
    }
    assert W.decide(sums, 12)[0] == "depth_fit"
    sums = {
        (m, ox): dict(row, ratio=1.2 if m == "depth_fit" else 1.5)
        for m in ("depth_fit", "mixing")
        for ox in ("with", "without")
    }
    assert W.decide(sums, 12)[0] == "mixing"


def test_the_winter_verdict_is_one_script_and_its_module():
    found = {str(p.relative_to(ROOT)) for p in (ROOT / "scripts").glob("winter_*.py")}
    assert found == WINTER_CODE
    module = re.compile(r"\bwinter_scoring\b|\bwinter_verdict\b|storm_mode/|[\"']storm_mode[\"']")
    for path in (ROOT / "scripts").glob("*.py"):
        if str(path.relative_to(ROOT)) not in WINTER_CODE:
            assert not module.search(path.read_text()), path
    for path in (ROOT / "strawberrywatch").rglob("*.py"):
        assert not module.search(path.read_text()), path
    for model in ("dusk_crayfish", "water_strider"):
        assert not list((ROOT / "checkpoints" / model / "storm_mode").rglob("*.py"))

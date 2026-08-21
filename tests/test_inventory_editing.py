"""
The inventory as a thing a non-coder edits: the grid, and the guard rails.

Every validation rule gets its bad value written and its message checked, because
a message that does not name the site and the sensor is a message that sends a
student back to someone who writes Python.
"""

from __future__ import annotations

import re
import shutil

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import inventory as inv

YAML = inv._yaml_path()

SITES = [
    "botanical_garden",
    "south_fork_0",
    "south_fork_1",
    "south_fork_2",
    "south_fork_3",
    "kingman_hall",
    "north_fork_0",
    "scnf010",
    "university_house",
    "oxford",
    "codornices",
]


@pytest.fixture
def text():
    with open(YAML) as handle:
        return handle.read()


@pytest.fixture
def sandbox(tmp_path):
    """A copy of the real file, so a test that writes cannot touch the repo."""
    copy = tmp_path / "inventory.yaml"
    shutil.copy(YAML, copy)
    return copy


def grid_block(text):
    """The grid rows as written, in file order."""
    block = text.split("\nin_service:\n", 1)[1].split("\n\n", 1)[0]
    return block.splitlines()


# The grid


def test_the_grid_is_the_first_thing_in_the_file(text):
    body = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert body[0] == "in_service:", f"the file opens with {body[0]!r}"


def test_the_grid_holds_every_site_and_nothing_else(text):
    inventory = inv.load()
    rows = [ln.split(":", 1)[0].strip() for ln in grid_block(text)]
    assert rows == SITES
    assert set(rows) == set(inventory.sites)


def test_every_cell_says_yes_no_or_dash(text):
    for line in grid_block(text):
        cells = re.findall(r"(\w+): ([^,}]+)", line.split("{", 1)[1])
        assert cells, f"no cells parsed out of {line!r}"
        for sensor, value in cells:
            assert value.strip() in ("yes", "no", "-"), f"{sensor} reads {value.strip()!r}"


def test_every_row_carries_the_whole_column_set_for_its_kind(text):
    inventory = inv.load()
    for table, cells in inventory.grid_rows():
        expected = inventory.sensor_columns[inventory.sites[table].source]
        assert list(cells) == list(expected), f"{table} columns are {list(cells)}"


def test_the_columns_line_up_so_the_grid_reads_as_a_grid(text):
    """The whole point of the layout. A ragged grid is not a grid."""
    lines = grid_block(text)
    mayfly = [ln for ln in lines if "ctd:" in ln]
    balance = [ln for ln in lines if "stage:" in ln]

    for group in (mayfly, balance):
        assert len(group) > 1
        columns = [{m.start() for m in re.finditer(r"\w+: ", ln)} for ln in group]
        assert all(c == columns[0] for c in columns), "cells do not start at the same column"


def test_the_instructions_sit_above_the_grid(text):
    header = text.split("\nin_service:", 1)[0].lower()
    # the three spellings, explained where a person will read them
    assert "yes =" in header and "no =" in header and "- =" in header
    assert "out of service" in header
    assert "python -m strawberrywatch.inventory" in header


def test_the_header_says_how_to_add_a_newly_installed_probe(text):
    header = text.split("\nin_service:", 1)[0].lower()
    assert "new probe" in header
    assert "install date" in header


def test_the_file_explains_the_difference_between_installed_and_in_service(text):
    prose = text.split("\nsites:\n", 1)[0].lower()
    assert "in service is not the same as installed" in prose
    # both examples, so neither direction is left to be worked out
    assert "cleaning" in prose
    assert "never fitted" in prose
    # and what switching one off actually does
    assert "train the models" in prose


def test_there_is_no_second_place_a_switch_could_live(text):
    """One source of truth. A stray in_service line below would contradict the grid."""
    below = text.split("\nsites:\n", 1)[1]
    assert "in_service" not in below


def test_the_comments_stay_short_enough_not_to_intimidate(text):
    """A wall of text is a file nobody edits."""
    header = text.split("\nin_service:", 1)[0]
    assert len(header.splitlines()) <= 8, "the first thing an intern sees is too long"
    # and no comment anywhere runs on past a screen width
    for line in text.splitlines():
        if line.strip().startswith("#"):
            assert len(line) <= 88, f"comment runs long: {line!r}"


def test_the_scnf010_date_is_flagged_as_the_one_to_fix_first(text):
    detail = text.split("\nsites:\n", 1)[1]
    block = detail.split("  scnf010:\n", 1)[1].split("    name:", 1)[0]
    assert "LIKELY WRONG" in block
    assert "117,882" in block


def test_every_guessed_date_is_marked_in_the_file(text):
    inventory = inv.load()
    detail = text.split("\nsites:\n", 1)[1]
    for site, sensor, _date, _evidence in inventory.inferred_dates():
        block = detail.split(f"  {site}:\n", 1)[1].split(f"      {sensor}:\n", 1)[1]
        head = block.split("install:", 1)[0]
        assert "GUESSED" in head, f"{site}.{sensor} has no guessed marker above its date"


# Round trip


def test_writing_a_switch_leaves_every_other_line_alone(sandbox):
    before = sandbox.read_text()
    inv.set_in_service("scnf010", "do", False, path=sandbox)
    after = sandbox.read_text()

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    assert len(before_lines) == len(after_lines)

    changed = [i for i, (a, b) in enumerate(zip(before_lines, after_lines, strict=True)) if a != b]
    assert len(changed) == 1
    rewritten = after_lines[changed[0]]
    assert rewritten.startswith("  scnf010:")
    assert "do: no" in rewritten
    assert "ctd: yes" in rewritten


def test_a_rewritten_row_keeps_the_grid_lined_up(sandbox):
    inv.set_in_service("scnf010", "do", False, path=sandbox)
    lines = grid_block(sandbox.read_text())
    mayfly = [ln for ln in lines if "ctd:" in ln]
    columns = [{m.start() for m in re.finditer(r"\w+: ", ln)} for ln in mayfly]
    assert all(c == columns[0] for c in columns), "switching a sensor off broke the alignment"


def test_load_write_load_round_trips_and_the_comments_survive(sandbox):
    original = sandbox.read_text()
    first = inv.load(path=sandbox, reload=True)

    # write the value it already has, so the document must come back identical
    inv.set_in_service(
        "south_fork_3", "do", first.sites["south_fork_3"].sensors["do"].in_service, path=sandbox
    )
    assert sandbox.read_text() == original

    # and a real change round trips through the loader
    inv.set_in_service("south_fork_3", "do", False, path=sandbox)
    second = inv.load(path=sandbox, reload=True)
    assert second.sites["south_fork_3"].sensors["do"].in_service is False

    inv.set_in_service("south_fork_3", "do", True, path=sandbox)
    assert sandbox.read_text() == original
    third = inv.load(path=sandbox, reload=True)
    assert third.sites["south_fork_3"].sensors["do"].in_service is True

    comments = [ln for ln in original.splitlines() if ln.strip().startswith("#")]
    kept = [ln for ln in sandbox.read_text().splitlines() if ln.strip().startswith("#")]
    assert comments == kept and len(comments) > 40


def test_taking_a_whole_site_out_round_trips_too(sandbox):
    original = sandbox.read_text()
    inv.set_site_in_service("south_fork_3", False, path=sandbox)
    off = inv.load(path=sandbox, reload=True)
    assert off.sites["south_fork_3"].in_service is False

    inv.set_site_in_service("south_fork_3", True, path=sandbox)
    assert sandbox.read_text() == original


def test_a_dash_cannot_be_switched_on_from_code(sandbox):
    """Turning a - into a yes needs an install date, which the grid cannot carry."""
    with pytest.raises(inv.InventoryError, match="no such probe"):
        inv.set_in_service("south_fork_0", "ph", True, path=sandbox)


# Toggling


def index(days=30):
    return pd.date_range("2026-04-01", periods=days * 96, freq="15min", tz="UTC")


def readings(idx):
    return pd.Series(np.linspace(400, 500, len(idx)), index=idx)


def frames_for(idx, variables=("conductivity", "dissolved_oxygen")):
    return {
        t: pd.DataFrame({v: readings(idx) for v in variables}, index=idx) for t in inv.load().tables
    }


def test_switching_one_sensor_off_leaves_the_others_alone(sandbox):
    """south_fork_3 carries four probes, so one going off must move only one."""
    idx = index()
    frames = frames_for(idx)

    on = inv.load(path=sandbox, reload=True)
    before = on.state_counts(frames, as_of=idx[-1])
    assert "dissolved_oxygen" in set(before[before["site"] == "south_fork_3"]["variable"])

    inv.set_in_service("south_fork_3", "do", False, path=sandbox)
    off = inv.load(path=sandbox, reload=True)
    after = off.state_counts(frames, as_of=idx[-1])

    site_rows = after[after["site"] == "south_fork_3"]
    assert "dissolved_oxygen" not in set(site_rows["variable"]), "the probe is still in the table"
    assert "conductivity" in set(site_rows["variable"]), "its neighbours moved with it"
    assert "floating_conductivity" in set(site_rows["variable"])

    # the site itself is still in service, because its other probes are
    assert "south_fork_3" in off.scored_sites()
    assert off.out_of_service() == []
    assert off.switched_off() == [("south_fork_3", "do")]

    # and it reads as NOT_INSTALLED everywhere downstream
    states = off.resolve_series("south_fork_3", "dissolved_oxygen", readings(idx), as_of=idx[-1])
    assert set(states) == {inv.NOT_INSTALLED}


def test_switching_one_sensor_off_moves_nothing_else(sandbox):
    idx = index(3)
    frames = frames_for(idx)

    on = inv.load(path=sandbox, reload=True)
    before = on.state_counts(frames, as_of=idx[-1])

    inv.set_in_service("south_fork_3", "do", False, path=sandbox)
    off = inv.load(path=sandbox, reload=True)
    after = off.state_counts(frames, as_of=idx[-1])

    key = ["site", "variable", "state"]
    moved = (before["site"] == "south_fork_3") & (before["variable"] == "dissolved_oxygen")
    shared = before[~moved].set_index(key)["count"]
    still = after.set_index(key)["count"]
    pd.testing.assert_series_equal(shared.sort_index(), still.sort_index())

    assert on.edges() == off.edges()


def test_a_switched_off_sensor_keeps_its_history_for_training(sandbox):
    idx = index()
    inv.set_in_service("south_fork_3", "do", False, path=sandbox)
    off = inv.load(path=sandbox, reload=True)

    training = off.resolve_series(
        "south_fork_3", "dissolved_oxygen", readings(idx), as_of=idx[-1], in_service_only=False
    )
    assert set(training) == {inv.PRESENT}

    live = off.resolve_series("south_fork_3", "dissolved_oxygen", readings(idx), as_of=idx[-1])
    assert set(live) == {inv.NOT_INSTALLED}


def test_a_site_with_every_probe_off_is_out_of_service(sandbox):
    idx = index()
    frames = frames_for(idx)

    inv.set_site_in_service("south_fork_2", False, path=sandbox)
    off = inv.load(path=sandbox, reload=True)

    assert off.sites["south_fork_2"].in_service is False
    assert off.out_of_service() == ["south_fork_2"]
    assert "south_fork_2" not in off.scored_sites()
    assert "south_fork_2" not in set(off.state_counts(frames, as_of=idx[-1])["site"])

    training = off.resolve_series(
        "south_fork_2", "conductivity", readings(idx), as_of=idx[-1], in_service_only=False
    )
    assert set(training) == {inv.PRESENT}


def test_the_screen_withholds_a_site_that_is_out_of_service(sandbox):
    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish
    from strawberrywatch.support_modules.settling_pool import SettlingPool

    idx = pd.date_range("2026-03-01", periods=24, freq="15min", tz="UTC")
    sites = ["south_fork_1", "south_fork_2", "oxford"]

    # readings drawn from each site's own healthy range, so the grid starts
    # fully admitted and the only thing that can empty a column is the switch
    rng = np.random.default_rng(3)
    inventory = inv.load(path=sandbox, reload=True)
    frames = []
    for site in sites:
        entry = inventory.site(site)
        columns = {"location": site}
        for variable in entry.variables:
            low, high = entry.threshold(variable, "suspect_range") or (0.0, 1.0)
            middle = (low + high) / 2
            columns[variable] = middle + rng.normal(0, (high - low) / 100, len(idx))
        frames.append(pd.DataFrame(columns, index=idx))
    raw = pd.concat(frames).sort_index()
    window = {"raw": raw, "timestamps": idx, "model": DuskCrayfish}

    on = SettlingPool(inventory=inv.load(path=sandbox, reload=True)).admit(window, sites)

    inv.set_site_in_service("south_fork_2", False, path=sandbox)
    off = SettlingPool(inventory=inv.load(path=sandbox, reload=True)).admit(window, sites)

    assert on.all()
    assert not off[:, 1].any()
    np.testing.assert_array_equal(on[:, 0], off[:, 0])
    np.testing.assert_array_equal(on[:, 2], off[:, 2])


# Every validation rule, with the bad value actually written


def _minimal(**swaps):
    body = """
in_service:
  upstream:  {ctd: yes, do: -,   fc: -,   ph: -}
  oxford:    {stage: yes, balance_feed: -}

sensor_variables:
  ctd: [conductivity, temperature, depth]
  do: [dissolved_oxygen]
  fc: [floating_conductivity]
  ph: [ph]
  stage: [depth]
  balance_feed: [conductivity, temperature]

sensor_columns:
  mayfly: [ctd, do, fc, ph]
  balance_hydrologics: [stage, balance_feed]

sites:
  upstream:
    name: 'Upstream'
    source: mayfly
    downstream: oxford
    sensors:
      ctd:
        install: '2025-01-01'
        removed: null
      do:
        install: null
        removed: null
  oxford:
    name: 'Oxford Street'
    source: balance_hydrologics
    downstream: null
    sensors:
      stage:
        install: '2025-01-01'
        removed: null
"""
    for old, new in swaps.items():
        body = body.replace(old.replace("__", " "), new)
    return body


MAYFLY_ROW = "{ctd: yes, do: -,   fc: -,   ph: -}"
CTD_BLOCK = "      ctd:\n        install: '2025-01-01'"
DO_BLOCK = "      do:\n        install: null"

BAD_EDITS = [
    ("yes written as true", "{ctd: yes, do: -,", "{ctd: true, do: -,", "upstream", "ctd"),
    ("yes written as True", "{ctd: yes, do: -,", "{ctd: True, do: -,", "upstream", "ctd"),
    ("yes written as on", "{ctd: yes, do: -,", "{ctd: on, do: -,", "upstream", "ctd"),
    ("yes written as 1", "{ctd: yes, do: -,", "{ctd: 1, do: -,", "upstream", "ctd"),
    ("no written as false", "{ctd: yes, do: -,", "{ctd: false, do: -,", "upstream", "ctd"),
    ("no written as 0", "{ctd: yes, do: -,", "{ctd: 0, do: -,", "upstream", "ctd"),
    ("yes in quotes", "{ctd: yes, do: -,", "{ctd: 'yes', do: -,", "upstream", "ctd"),
    ("a column left out", MAYFLY_ROW, "{ctd: yes, do: -,   fc: -}", "upstream", "ph"),
    (
        "a column that cannot exist here",
        MAYFLY_ROW,
        "{ctd: yes, do: -,   fc: -,   ph: -, stage: yes}",
        "upstream",
        "stage",
    ),
    (
        "switched on with no install date",
        "{ctd: yes, do: -,",
        "{ctd: yes, do: yes,",
        "upstream",
        "do",
    ),
    (
        "a dash with an install date below",
        DO_BLOCK,
        "      do:\n        install: '2025-02-02'",
        "upstream",
        "do",
    ),
    (
        "a row with no block",
        "  oxford:    {stage",
        "  sather_gate: {stage: yes, balance_feed: -}\n  oxford:    {stage",
        "sather_gate",
        None,
    ),
    ("a block with no row", f"  upstream:  {MAYFLY_ROW}\n", "", "upstream", None),
    (
        "an unknown sensor name",
        CTD_BLOCK,
        "      turbidty:\n        install: '2025-01-01'",
        "upstream",
        "turbidty",
    ),
    (
        "a switch left in the detail block",
        CTD_BLOCK,
        "      ctd:\n        in_service: true\n        install: '2025-01-01'",
        "upstream",
        "ctd",
    ),
    (
        "a malformed date",
        CTD_BLOCK,
        "      ctd:\n        install: 'last summer'",
        "upstream",
        "ctd",
    ),
    (
        "a removal before its install",
        "        install: '2025-01-01'\n        removed: null\n      do:",
        "        install: '2025-01-01'\n        removed: '2024-01-01'\n      do:",
        "upstream",
        "ctd",
    ),
    (
        "a downstream that does not exist",
        "downstream: oxford",
        "downstream: sather_gate",
        "upstream",
        None,
    ),
    (
        "a cycle in the flow graph",
        "    downstream: null",
        "    downstream: upstream",
        "upstream",
        None,
    ),
    (
        "no downstream and not an outlet",
        "    downstream: oxford",
        "    downstream: null",
        "upstream",
        None,
    ),
    ("a source nobody recognises", "source: mayfly", "source: carrier_pigeon", "upstream", None),
    (
        "a site with no sensors",
        "      ctd:\n        install: '2025-01-01'\n        removed: null\n      do:\n        install: null\n        removed: null\n",
        "",
        "upstream",
        None,
    ),
]


@pytest.mark.parametrize("label,old,new,site,sensor", BAD_EDITS, ids=[b[0] for b in BAD_EDITS])
def test_a_bad_edit_raises_and_the_message_names_what_to_fix(label, old, new, site, sensor):
    good = _minimal()
    assert old in good, f"the fixture no longer contains {old!r}"
    bad = good.replace(old, new, 1)

    inv.from_text(good)
    with pytest.raises(inv.InventoryError) as exc:
        inv.from_text(bad)

    message = str(exc.value)
    assert site in message, f"{label}: message does not name {site}\n{message}"
    if sensor is not None:
        assert sensor in message, f"{label}: message does not name {sensor}\n{message}"
    assert len(message.splitlines()) > 1, f"{label}: message gives no instruction\n{message}"


@pytest.mark.parametrize("label,old,new,site,sensor", BAD_EDITS, ids=[b[0] for b in BAD_EDITS])
def test_a_bad_edit_names_the_line_it_is_on(label, old, new, site, sensor):
    """A student needs the line number, not just the site."""
    bad = _minimal().replace(old, new, 1)
    with pytest.raises(inv.InventoryError) as exc:
        inv.from_text(bad)

    assert re.search(r"line \d+", str(exc.value)), f"{label}: no line number\n{exc.value}"


def test_a_typo_never_silently_drops_a_site():
    """The failure this whole file exists to prevent."""
    good = inv.from_text(_minimal())
    assert set(good.sites) == {"upstream", "oxford"}

    bad = _minimal().replace("  upstream:  {ctd", "  upsteam:  {ctd", 1)
    with pytest.raises(inv.InventoryError) as exc:
        inv.from_text(bad)
    assert "upsteam" in str(exc.value) or "upstream" in str(exc.value)


def test_a_typo_never_silently_drops_a_sensor():
    """The same failure one level down, which the site-only switchboard could not catch."""
    bad = _minimal().replace("{ctd: yes, do: -,", "{ctdd: yes, do: -,", 1)
    with pytest.raises(inv.InventoryError) as exc:
        inv.from_text(bad)
    assert "ctd" in str(exc.value)


# The command line view


def test_the_report_prints_the_grid_in_the_same_layout_as_the_file(text):
    """So a person can hold the two side by side and see their edit reflected."""
    printed = inv.report(check_data=False)
    for line in grid_block(text):
        assert line in printed, f"the report does not show the row as written: {line!r}"


def test_the_report_names_every_site_and_says_which_dates_were_guessed():
    printed = inv.report(check_data=False)
    inventory = inv.load()

    for table in inventory.tables:
        assert table in printed
    assert "GUESSED" in printed
    assert str(len(inventory.inferred_dates())) in printed


def test_the_report_says_which_probes_are_switched_off(sandbox):
    inv.set_in_service("scnf010", "do", False, path=sandbox)
    inventory = inv.load(path=sandbox, reload=True)
    printed = inv.report(inventory, check_data=False)

    assert "scnf010 do" in printed
    assert "set to no" in printed
    assert "still used for training" in printed


def test_the_report_says_when_a_whole_site_has_gone_dark(sandbox):
    inv.set_site_in_service("kingman_hall", False, path=sandbox)
    inventory = inv.load(path=sandbox, reload=True)
    printed = inv.report(inventory, check_data=False)

    assert "kingman_hall" in printed
    assert "out of service" in printed


def test_a_broken_file_makes_the_command_print_the_problem_rather_than_a_traceback(
    tmp_path, capsys
):
    broken = tmp_path / "inventory.yaml"
    broken.write_text(_minimal().replace("{ctd: yes, do: -,", "{ctd: Yes, do: -,", 1))

    from strawberrywatch import inventory

    original = inventory._yaml_path
    inventory._yaml_path = lambda path=None: str(broken)
    try:
        code = inventory.main([])
    finally:
        inventory._yaml_path = original

    assert code == 1
    printed = capsys.readouterr().out
    assert "nothing will run until it is fixed" in printed
    assert "upstream" in printed
    assert "ctd" in printed


def test_the_report_warns_when_the_file_and_the_data_disagree(sandbox):
    """Named, never resolved. The file is not edited by the check."""
    before = sandbox.read_text()
    inv.set_site_in_service("south_fork_1", False, path=sandbox)
    inventory = inv.load(path=sandbox, reload=True)

    printed = inv.report(inventory, check_data=True)
    assert "FILE AGAINST DATA" in printed

    # the check reads the file and does not write it
    after = sandbox.read_text()
    changed = [
        (a, b) for a, b in zip(before.splitlines(), after.splitlines(), strict=True) if a != b
    ]
    assert len(changed) == 1 and changed[0][0].strip().startswith("south_fork_1")


def test_the_clock_reset_row_is_dropped_and_reported():
    from strawberrywatch.ingest.raw_data_loader import load_archive_by_table

    inventory = inv.load()
    kept, dropped = load_archive_by_table(
        earliest=inventory.earliest_plausible_reading, with_report=True
    )

    assert any(d["table"] == "kingman_hall" for d in dropped)
    entry = next(d for d in dropped if d["table"] == "kingman_hall")
    assert entry["rows"] == 1
    assert entry["worst"].startswith("2000-06-17")
    assert "clock reset" in entry["reason"]

    assert kept["kingman_hall"].index.min() >= inventory.earliest_plausible_reading

    # nothing is dropped when no floor is given, so the drop is a choice not a default
    untouched = load_archive_by_table()
    assert untouched["kingman_hall"].index.min() < inventory.earliest_plausible_reading

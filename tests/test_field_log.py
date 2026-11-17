"""The field log and the reviewed level it feeds."""

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import field_log
from strawberrywatch import inventory as inv
from strawberrywatch.ingest import levels
from tests import event_catalog as ec

HEADER = ",".join(field_log.COLUMNS)
COND = "Meter_Hydros21_Cond"


def _log(tmp_path, *lines):
    path = tmp_path / "field_log.csv"
    path.write_text("\n".join([HEADER, *lines]) + "\n")
    return path


def test_the_shipped_log_loads_and_every_line_is_a_known_event():
    log = field_log.load()
    assert set(log["event"]) <= set(field_log.EVENTS)
    assert list(pd.read_csv(field_log.path_for(), nrows=0).columns) == list(field_log.COLUMNS)


def test_a_time_carries_its_offset_and_a_bare_date_is_the_pacific_day(tmp_path):
    path = _log(
        tmp_path,
        "2026-02-09,16:03:00-08:00,south_fork_0,conductivity,bad,2026-04-28,flat,dan,no,,",
        "2026-07-01,,south_fork_0,ctd,cleaned,,wiped the cell,ana,yes,412.5,398",
    )
    log = field_log.load(path)
    bad, cleaned = log.iloc[0], log.iloc[1]
    assert bad["start"] == pd.Timestamp("2026-02-10 00:03", tz="UTC")
    # a day with no time runs to its last minute, and April is on daylight time
    assert bad["end"] == pd.Timestamp("2026-04-29 07:00", tz="UTC")
    assert bad["line"] == 2 and not bad["confirmed"]
    assert bad["key"] == "2026-02-09 south_fork_0 conductivity bad"
    assert cleaned["variables"] == ("conductivity", "temperature", "depth")
    assert cleaned["start"] == pd.Timestamp("2026-07-01 07:00", tz="UTC") and pd.isna(
        cleaned["end"]
    )
    # the readings either side of a cleaning, for a drift correction later
    assert (cleaned["value_before"], cleaned["value_after"]) == (412.5, 398.0)


@pytest.mark.parametrize(
    "line, said, fix",
    [
        ("3/5/2026,,south_fork_0,ctd,bad,,,,no,,", "is not a date", "year-month-day"),
        ("2026-03-05,2pm,south_fork_0,ctd,bad,,,,no,,", "is not a time", "24 hour clock"),
        ("2026-03-05,14:30,south_fork_0,ctd,bad,,,,no,,", "offset from UTC", "14:30:00-08:00"),
        ("2026-03-05,,south_fork_9,ctd,bad,,,,no,,", "inventory doesn't have", "south_fork_0"),
        ("2026-03-05,,south_fork_0,ph_probe,bad,,,,no,,", "doesn't have", "or all"),
        ("2026-03-05,,south_fork_0,ctd,broken,,,,no,,", "not one of the events", "out_of_water"),
        ("2026-03-05,,south_fork_0,ctd,bad,,,,maybe,,", "confirmed is", "Write yes"),
        ("2026-03-05,,south_fork_0,ctd,bad,2026-03-01,,,no,,", "is before", "leave end empty"),
        ("2026-03-05,,south_fork_0,ctd,cleaned,,,,yes,high,", "not a number", "412.5"),
    ],
)
def test_a_bad_line_is_named_with_its_fix(tmp_path, line, said, fix):
    path = _log(tmp_path, "2026-01-01,,south_fork_0,ctd,visit_ok,,,ana,yes,,", line)
    with pytest.raises(field_log.FieldLogError) as caught:
        field_log.load(path)
    message = str(caught.value)
    assert message.startswith("field_log.csv line 3:") and said in message and fix in message


def test_a_missing_column_is_named(tmp_path):
    path = tmp_path / "field_log.csv"
    path.write_text("date,time,site,sensor,event\n")
    with pytest.raises(field_log.FieldLogError, match="first line is missing end, note"):
        field_log.load(path)


def test_empty_lines_and_a_spreadsheets_byte_order_mark_are_fine(tmp_path):
    path = tmp_path / "field_log.csv"
    path.write_text(
        "﻿" + HEADER + "\n,,,,,,,,,,\n2026-01-01,,oxford,all,visit_ok,,,ana,yes,,\n",
        encoding="utf-8",
    )
    assert len(field_log.load(path)) == 1


def _table(start, periods):
    index = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    wave = np.sin(np.arange(periods))
    return pd.DataFrame(
        {COND: 900.0 + 20 * wave, "Meter_Hydros21_Temp": 15.0 + wave / 10}, index=index
    )


def test_the_reviewed_level_takes_a_bad_period_out_and_says_which_line(tmp_path):
    inventory = inv.load()
    path = _log(
        tmp_path,
        "2026-06-01,17:00:00-07:00,south_fork_0,conductivity,bad,2026-06-01T23:00:00-07:00,flat,,no,,",
        "2026-06-01,19:00:00-07:00,south_fork_0,ctd,visit_ok,,looked fine,ana,yes,,",
    )
    log = field_log.load(path, inventory)
    raw = {"south_fork_0": _table("2026-06-01 20:00", 96)}
    raw["south_fork_0"].iloc[90, 0] = -9999.0
    kept, why = levels.reviewed(raw, inventory, log)

    # 17:00 to 23:00 Pacific is 00:00 to 06:00 UTC the next day
    lo, hi = pd.Timestamp("2026-06-02 00:00", tz="UTC"), pd.Timestamp("2026-06-02 06:00", tz="UTC")
    inside = (raw["south_fork_0"].index >= lo) & (raw["south_fork_0"].index <= hi)
    assert kept["south_fork_0"][COND][inside].isna().all()
    assert (why["south_fork_0"][COND][inside] == 2).all()
    # only that sensor, and a visit takes nothing
    assert kept["south_fork_0"]["Meter_Hydros21_Temp"].notna().all()
    assert kept["south_fork_0"][COND][~inside].notna().sum() == (~inside).sum() - 1
    # the sentinel went to the screen, as it does for a live check
    assert why["south_fork_0"][COND].iloc[90] == levels.SCREENED
    # raw is left as it was
    assert raw["south_fork_0"][COND].notna().all()

    rows = levels.removed_by(why)
    assert sorted(map(tuple, rows.to_numpy())) == [
        ("south_fork_0", COND, levels.SCREENED, 1),
        ("south_fork_0", COND, 2, int(inside.sum())),
    ]


def test_everything_after_a_units_change_is_left_out(tmp_path):
    inventory = inv.load()
    path = _log(
        tmp_path, "2026-04-01,14:07:00-07:00,south_fork_3,dissolved_oxygen,units_changed,,,,no,,"
    )
    log = field_log.load(path, inventory)
    index = pd.date_range("2026-04-01 18:00", periods=24, freq="15min", tz="UTC")
    values = np.where(index < pd.Timestamp("2026-04-01 21:07", tz="UTC"), 74.0, 9.0)
    raw = {
        "south_fork_3": pd.DataFrame(
            {"dissolved_oxygen": values + np.arange(24) * 0.01}, index=index
        )
    }
    kept, why = levels.reviewed(raw, inventory, log)
    after = index >= pd.Timestamp("2026-04-01 21:07", tz="UTC")
    assert kept["south_fork_3"]["dissolved_oxygen"][after].isna().all()
    assert kept["south_fork_3"]["dissolved_oxygen"][~after].notna().all()
    assert set(why["south_fork_3"]["dissolved_oxygen"][after]) == {2}


def test_events_yaml_can_cite_field_log_lines():
    blob = {
        "events": [
            {
                "folder": "anomaly_2026_03_20_hydrant_nf0",
                "group": "mar26_hydrant",
                "site": "north_fork_0",
                "label": "anomaly",
                "field_log": ["2026-03-19 south_fork_1 ctd stolen"],
            }
        ]
    }
    lines = {"2026-03-19 south_fork_1 ctd stolen"}
    assert ec.parse(blob, known_lines=lines)[0].field_log == ("2026-03-19 south_fork_1 ctd stolen",)
    with pytest.raises(ec.CatalogError, match="isn't a line of field_log.csv"):
        ec.parse(blob, known_lines=set())


def test_visit_ok_lines_are_the_true_negative_candidates(tmp_path, monkeypatch):
    path = _log(
        tmp_path,
        "2026-06-01,10:00:00-07:00,oxford,all,visit_ok,,nothing wrong,ana,yes,,",
        "2026-06-02,,oxford,all,cleaned,,,ana,yes,,",
    )
    monkeypatch.setattr(field_log, "path_for", lambda _path=None: path)
    assert ec.true_negative_candidates() == [
        ("oxford", "all", pd.Timestamp("2026-06-01 17:00", tz="UTC"), "nothing wrong")
    ]


def test_a_utc_time_reads_the_same_as_its_pacific_one(tmp_path):
    path = _log(
        tmp_path,
        "2026-02-09,16:03:00-08:00,south_fork_0,conductivity,bad,,,,no,,",
        "2026-02-10,00:03:00Z,south_fork_0,depth,bad,,,,no,,",
    )
    log = field_log.load(path)
    assert log["start"].iloc[0] == log["start"].iloc[1]


def test_the_seeded_lines_take_out_what_was_agreed():
    log = field_log.load()
    keys = set(log["key"])
    assert "2025-11-20 south_fork_0 ctd stolen" in keys
    assert not log["confirmed"].any()
    flat = pd.DatetimeIndex(["2026-02-10 00:03", "2026-03-15", "2026-04-29 06:59"], tz="UTC")
    fine = pd.DatetimeIndex(["2026-02-09 23:00", "2026-04-29 07:00"], tz="UTC")
    assert (field_log.excluded(log, "south_fork_0", "conductivity", flat) > 0).all()
    assert (field_log.excluded(log, "south_fork_0", "conductivity", fine) == 0).all()
    # the theft takes nothing out, and neither line touches the probe's other readings
    assert (field_log.excluded(log, "south_fork_0", "temperature", flat) == 0).all()
    oxygen = pd.DatetimeIndex(["2026-03-05 19:45", "2026-04-01 21:07"], tz="UTC")
    assert list(field_log.excluded(log, "south_fork_3", "dissolved_oxygen", oxygen) > 0) == [
        False,
        True,
    ]

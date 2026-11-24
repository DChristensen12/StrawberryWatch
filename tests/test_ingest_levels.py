"""The ingest job and the raw and screened levels, on both stores."""

import hashlib
import json
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.ingest import levels
from strawberrywatch.support_modules import qc_tests as qc
from strawberrywatch.watch import ingest, runner, sources
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st
from tests.test_scmg_website import PrefixedStore

COND, DEPTH, TEMP = "Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp"
SITE = "south_fork_1"
STEPS = 25


@pytest.fixture(scope="module")
def models():
    return wm.registered()


@pytest.fixture(params=["sqlite", "sw_ tables"])
def store(request, tmp_path, models):
    """A store reading the API, once as the local file and once through the website's prefixing."""
    folder = tmp_path / "data"
    store = st.Store(folder) if request.param == "sqlite" else PrefixedStore(folder)
    store.set_source(sources.API)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    return store


class Feed:
    """The API and the weather faked: 25 quarter hours of readings at every site."""

    def __init__(self):
        self.end = pd.Timestamp.now(tz="UTC").floor("15min")
        self.stamps = pd.date_range(end=self.end, periods=STEPS, freq="15min")
        # {(site, column, position): value}, laid over the made-up readings
        self.changed = {}
        self.left_out = set()
        self.rain = 0.2
        # NBM's rain, for a checkpoint that trained on it. None leaves the column out
        self.nbm = None
        self.down = False

    def fetch(self, site, _start, _end, variables=None, token=None):
        if self.down:
            raise RuntimeError("the API is down")
        wave = np.sin(np.arange(STEPS))
        frame = pd.DataFrame(
            {
                "timestamp": self.stamps,
                COND: 400.0 + wave,
                DEPTH: 150.0 + wave,
                TEMP: 14.0 + wave / 10,
                "station_id": site,
            }
        )
        for (where, column, position), value in self.changed.items():
            if where == site:
                frame.loc[position, column] = value
        keep = [i for i in range(STEPS) if (site, i) not in self.left_out]
        return frame.iloc[keep]

    def weather(self, start, end):
        if self.rain is None:
            return None
        index = pd.date_range(start.floor("15min"), end, freq="15min")
        frame = pd.DataFrame({"rain_mm": self.rain, "air_temp_c": 12.0}, index=index)
        if self.nbm is not None:
            frame["rain_nbm_mm"] = self.nbm
        return frame


@pytest.fixture
def feed(monkeypatch):
    from strawberrywatch.ingest import api_client

    feed = Feed()
    monkeypatch.setattr(api_client, "fetch_creek_data", feed.fetch)
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: feed.weather(a, b))
    return feed


class Seen:
    """A detector that keeps what a check handed it."""

    wants_weather = True
    rain_params = None
    table_names = [SITE, "north_fork_0"]

    def __init__(self):
        self.tables, self.weather = None, "not called"

    def scores(self, tables, weather, leave_out=None):
        self.tables, self.weather = tables, weather
        return []


def _check(store, models, **kw):
    seen = Seen()
    runner.run_check(store, models, token="t", detectors={"dusk_crayfish": seen}, **kw)
    return seen


# The three time tests


def test_time_gap_flags_the_reading_after_a_gap():
    hours = np.array([0, 1, 2, 6, 7]) * 3600.0
    assert qc.time_gap(hours, fail_span=2).tolist() == [qc.GOOD, qc.GOOD, qc.GOOD, qc.FAIL, qc.GOOD]
    # the first point takes the second's flag
    assert qc.time_gap(np.array([0, 9, 10]) * 3600.0, fail_span=2)[0] == qc.FAIL
    assert qc.time_gap([0.0]).tolist() == [qc.GOOD]


def test_data_reception_fails_a_feed_gone_quiet():
    now = 100 * 3600.0
    stamps = np.array([now - 7 * 3600, now - 3600, np.nan])
    assert qc.data_reception(stamps, fail_span=6, from_time=now).tolist() == [
        qc.FAIL,
        qc.GOOD,
        qc.MISSING,
    ]


def test_impossible_date_fails_the_future_and_the_clock_resets():
    now = pd.Timestamp("2026-11-15", tz="UTC").timestamp()
    earliest = pd.Timestamp("2020-01-01", tz="UTC").timestamp()
    stamps = [
        pd.Timestamp("2000-06-17", tz="UTC").timestamp(),
        now - 3600,
        now + 3600,
    ]
    assert qc.impossible_date(stamps, (earliest, np.inf), now).tolist() == [
        qc.FAIL,
        qc.GOOD,
        qc.FAIL,
    ]


# Raw level


def test_ingest_keeps_saving_while_checks_are_paused(store, models, feed):
    store.set_pref("checks_paused", "yes")
    assert runner.Watcher(store, models).check() is None
    row = ingest.run(store, token="t")
    assert row["ok"] and row["rows_new"] == 3 * STEPS * len(inv.load().sites)
    assert len(store.raw(SITE)) == 3 * STEPS
    # the dashboard's own readings table is kept up too, for the site pages
    assert store.last_readings(live=True)[SITE]["time"] == feed.end.timestamp()


def test_the_ingest_thread_runs_without_a_check(store, feed):
    _, stop = ingest.start(store, token=lambda: "t", interval=0.05)
    deadline = time.time() + 30
    while store.last_ingest() is None and time.time() < deadline:
        time.sleep(0.05)
    stop.set()
    assert store.last_ingest()["ok"] and store.last_check() is None


def test_a_repeat_fetch_adds_nothing(store, feed):
    assert ingest.run(store, token="t")["rows_new"] > 0
    before = len(store.raw(SITE))
    assert ingest.run(store, token="t")["rows_new"] == 0
    assert len(store.raw(SITE)) == before
    when = feed.stamps[3].timestamp()
    assert len(store.raw_versions(SITE, COND, when)) == 1


def test_a_changed_value_adds_a_row_and_keeps_the_old_one(store, feed):
    now = time.time()
    ingest.run(store, token="t", clock=lambda: now)
    when = feed.stamps[3].timestamp()
    old = store.raw_versions(SITE, COND, when)[0]["value"]
    feed.changed[(SITE, COND, 3)] = 512.5
    assert ingest.run(store, token="t", clock=lambda: now + 900)["rows_new"] == 1
    versions = store.raw_versions(SITE, COND, when)
    assert [v["value"] for v in versions] == [old, 512.5]
    assert [v["fetched_at"] for v in versions] == [now, now + 900]
    # readers get the newest
    newest = {(r["col"], r["time"]): r["value"] for r in store.raw(SITE)}
    assert newest[(COND, when)] == 512.5


def test_raw_rows_say_where_and_when_they_came_from(store, feed):
    now = time.time()
    ingest.run(store, token="t", clock=lambda: now)
    row = store.raw(SITE)[0]
    assert row["source"] == "api" and row["fetched_at"] == now
    assert row["col"] in (COND, DEPTH, TEMP)


# Screened level


def test_each_reading_gets_its_flag_and_the_tests_that_did_not_pass(store, feed):
    feed.changed[(SITE, COND, 10)] = -9999.0
    ingest.run(store, token="t")
    when = feed.stamps[10].timestamp()
    rows = {(r["col"], r["time"]): r for r in store.screened(SITE)}
    bad = rows[(COND, when)]
    assert bad["flag"] == qc.FAIL
    assert json.loads(bad["tests"])["sentinel"] == qc.FAIL
    # the raw value is still what the logger sent
    assert store.raw_versions(SITE, COND, when)[0]["value"] == -9999.0
    good = rows[(COND, feed.stamps[5].timestamp())]
    assert good["flag"] == qc.GOOD and good["tests"] is None


def test_a_step_with_no_reading_gets_a_missing_row(store, feed):
    feed.left_out.add((SITE, 20))
    ingest.run(store, token="t")
    when = feed.stamps[20].timestamp()
    rows = {(r["col"], r["time"]): r["flag"] for r in store.screened(SITE)}
    assert {rows[(c, when)] for c in (COND, DEPTH, TEMP)} == {qc.MISSING}
    assert all(r["time"] != when for r in store.raw(SITE))


def test_missing_rows_are_only_for_sensors_in_service(store, feed):
    ingest.run(store, token="t")
    inventory = inv.load()
    # south_fork_3 has a dissolved oxygen probe in service, botanical_garden has none
    assert inventory.site("south_fork_3").sensor_for("dissolved_oxygen").in_service
    assert not inventory.site("botanical_garden").sensor_for("dissolved_oxygen").in_service
    oxygen = [r for r in store.screened("south_fork_3") if r["col"] == "AtlasSci_DO"]
    assert oxygen and {r["flag"] for r in oxygen} == {qc.MISSING}
    assert not [r for r in store.screened("botanical_garden") if r["col"] == "AtlasSci_DO"]


def test_missing_rows_follow_the_sensors_own_cadence():
    inventory = inv.load()
    start = pd.Timestamp("2026-06-01", tz="UTC")
    hourly = pd.date_range(start, periods=24, freq="60min").delete(7)
    table = pd.DataFrame({COND: 400.0 + np.arange(23.0)}, index=hourly)
    flags = levels.screen_table(table, SITE, inventory, start, hourly[-1])[COND]
    missing = flags.index[flags["flag"] == qc.MISSING]
    assert list(missing) == [start + pd.Timedelta(hours=7)]


def test_a_gap_is_recorded_without_failing_the_reading_after_it():
    inventory = inv.load()
    start = pd.Timestamp("2026-06-01", tz="UTC")
    stamps = pd.date_range(start, periods=40, freq="15min").delete(range(10, 30))
    table = pd.DataFrame({COND: 400.0 + np.sin(np.arange(20.0))}, index=stamps)
    flags = levels.screen_table(table, SITE, inventory)[COND]
    after = flags.loc[stamps[10]]
    assert after["time_gap"] == qc.FAIL and after["flag"] != qc.FAIL


# What a check reads


def test_a_sentinel_reaches_no_model(store, models, feed):
    feed.changed[(SITE, COND, 10)] = -9999.0
    feed.changed[(SITE, TEMP, 12)] = 332.0
    seen = _check(store, models)
    table = seen.tables[SITE]
    assert np.isnan(table.loc[feed.stamps[10], COND])
    assert np.isnan(table.loc[feed.stamps[12], TEMP])
    assert not (table == -9999.0).any().any() and table[TEMP].max() < 40
    # its neighbours and the other columns at that step are still there
    assert table.loc[feed.stamps[10], DEPTH] > 0 and table[COND].notna().sum() == STEPS - 1
    assert store.last_check()["ok"]


def test_a_replay_is_screened_the_same_way():
    inventory = inv.load()
    stamps = pd.date_range("2026-06-01", periods=STEPS, freq="15min", tz="UTC")
    raw = {SITE: pd.DataFrame({COND: 400.0 + np.sin(np.arange(STEPS))}, index=stamps)}
    raw[SITE].iloc[10, 0] = -9999.0
    flags = levels.screen(raw, inventory, stamps[0], stamps[-1])
    tables = runner._screened(raw, flags)
    assert np.isnan(tables[SITE].loc[stamps[10], COND])
    assert tables[SITE][COND].notna().sum() == STEPS - 1


def test_a_check_ahead_of_ingest_runs_one_pass_and_the_next_reuses_it(store, models, feed):
    _check(store, models)
    first = store.last_ingest()
    assert first["ok"] and first["rows_new"] > 0
    _check(store, models)
    assert store.last_ingest()["id"] == first["id"]


def test_a_check_fails_as_before_when_the_readings_cannot_be_fetched(store, models, feed):
    feed.down = True
    _check(store, models)
    last = store.last_check()
    assert last["ok"] == 0 and "the API is down" in last["message"]
    assert store.last_ingest()["ok"] == 0


# Weather


def test_a_failed_weather_fetch_is_not_zero_rain(store, models, feed):
    feed.rain = None
    seen = _check(store, models)
    assert store.weather_rows(st.EARLIEST, st.LATEST) == []
    assert seen.weather is None and store.last_check()["weather_ok"] == 0
    assert "Weather fetch failed" in store.last_ingest()["message"]
    # the readings were still saved, and the check itself was fine
    assert store.last_ingest()["ok"] and store.last_check()["ok"]


def test_nbm_rain_is_saved_beside_the_default_and_handed_to_checks(store, models, feed):
    feed.rain, feed.nbm = 1.5, 0.7
    seen = _check(store, models)
    rows = store.weather_rows(st.EARLIEST, st.LATEST)
    assert rows and all(r["rain_mm"] == 1.5 and r["rain_nbm_mm"] == 0.7 for r in rows)
    assert (seen.weather["rain_mm"] == 1.5).all() and (seen.weather["rain_nbm_mm"] == 0.7).all()


def test_weather_without_the_nbm_column_is_saved_and_says_to_run_setup_tables(tmp_path, caplog):
    from strawberrywatch.dashboard import view

    store = st.Store(tmp_path / "updated")
    # an install that hasn't run setup_tables since rain_nbm_mm was added
    with store.db() as db:
        db.execute("DROP TABLE weather")
        db.execute(
            "CREATE TABLE weather (time REAL PRIMARY KEY, rain_mm REAL, air_temp_c REAL, "
            "shortwave_radiation REAL, fetched_at REAL)"
        )
    store._nbm_column = False
    with caplog.at_level("WARNING"):
        store.save_weather([(900.0, 1.5, 10.0, 200.0, 0.7)], fetched_at=1000.0)
    assert "setup_tables" in caplog.text
    row = store.weather_rows(0, 2000)[0]
    assert (row["rain_mm"], row["air_temp_c"], row["fetched_at"]) == (1.5, 10.0, 1000.0)
    assert view.data_health(store, 2000.0)["no_nbm_column"]
    # setup_tables adds it, and the next save clears the warning
    with store.db() as db:
        db.execute("ALTER TABLE weather ADD COLUMN rain_nbm_mm REAL")
    store.save_weather([(900.0, 1.5, 10.0, 200.0, 0.7)], fetched_at=1100.0)
    assert store.weather_rows(0, 2000)[0]["rain_nbm_mm"] == 0.7
    assert not view.data_health(store, 2000.0)["no_nbm_column"]


def test_checks_use_the_last_saved_rain_and_say_when_it_is_old(store, models, feed):
    from strawberrywatch.dashboard import view

    now = time.time()
    feed.rain = 1.5
    seen = _check(store, models, clock=lambda: now)
    assert store.last_check()["weather_ok"] == 1 and (seen.weather["rain_mm"] == 1.5).all()
    assert "air_temp_c" in seen.weather.columns

    feed.rain = None
    later = now + 2 * 3600
    seen = _check(store, models, clock=lambda: later)
    assert (seen.weather["rain_mm"] == 1.5).all()
    assert store.last_check()["weather_ok"] == 0
    item = {"explanations": {}, "check_weather_ok": 0}
    assert view.extras_said(item) == ["Rain data unavailable for this check."]


# Installs from before the levels


def test_an_install_without_the_level_tables_checks_the_old_way(tmp_path, models, feed):
    store = st.Store(tmp_path / "old")
    store.set_source(sources.API)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    with store.db() as db:
        for name in ("raw_readings", "screened", "ingest_runs"):
            db.execute(f"DROP TABLE {name}")
    store._levels = False

    assert not store.has_levels() and ingest.run(store, token="t") is None
    feed.changed[(SITE, COND, 10)] = -9999.0
    seen = _check(store, models)
    assert store.last_check()["ok"] and store.last_check()["weather_ok"] == 1
    # screened in memory, and the sentinel still stays out
    assert np.isnan(seen.tables[SITE].loc[feed.stamps[10], COND])
    assert store.weather(st.EARLIEST, st.LATEST) and store.last_readings(live=True)


# Backfill


def test_backfill_copies_the_archive_and_the_saved_readings_once(store, tmp_path):
    archive = tmp_path / "raw_data"
    archive.mkdir()
    lines = ["uuid\tsite_code\ttimestamp\tMeter_Hydros21_Cond\tEnviroDIY_Mayfly_Batt"]
    stamps = pd.date_range("2026-03-01", periods=8, freq="15min")
    for i, stamp in enumerate(stamps):
        value = "NULL" if i == 2 else ("-9999" if i == 4 else f"{400 + i}")
        lines.append(f"id{i}\t{SITE}\t{stamp:%Y-%m-%d %H:%M:%S}\t{value}\t4.1")
    path = archive / f"{SITE}.csv"
    path.write_text("\n".join(lines) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    saved = pd.Timestamp("2026-11-01", tz="UTC").timestamp()
    store.save_readings([("oxford", saved, 351.0, 2.0, 13.0, None, None, 4.1)], live=True)
    store.save_readings([("oxford", saved + 900, 9.0, 9.0, 9.0, None, None, None)], live=False)

    counts = ingest.backfill(store, raw_dir=archive)
    # seven conductivity values and eight battery ones. The NULL isn't a reading
    assert counts == {"archive": 15, "store": 4}
    rows = store.raw(SITE)
    assert {r["source"] for r in rows} == {"archive"}
    assert {r["col"] for r in rows} == {COND, "EnviroDIY_Mayfly_Batt"}
    assert {r["source"] for r in store.raw("oxford")} == {"store"}
    # a replay's readings aren't the live creek's
    assert all(r["time"] == saved for r in store.raw("oxford"))
    flags = {r["time"]: r["flag"] for r in store.screened(SITE) if r["col"] == COND}
    assert flags[stamps[4].tz_localize("UTC").timestamp()] == qc.FAIL
    assert flags[stamps[2].tz_localize("UTC").timestamp()] == qc.MISSING

    assert ingest.backfill(store, raw_dir=archive) == {"archive": 0, "store": 0}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


# The website's side


def test_the_website_runs_ingest_ahead_of_the_check_whatever_the_check_does(monkeypatch):
    from strawberrywatch.integrations.night_heron import gnn_alerts

    order = []

    def check(_now):
        order.append("check")
        raise RuntimeError("the check fell over")

    monkeypatch.setattr(gnn_alerts, "DASHBOARD", True)
    monkeypatch.setattr(gnn_alerts, "_ingest_in_child", lambda: order.append("ingest"))
    monkeypatch.setattr(gnn_alerts, "_check_in_child", check)
    gnn_alerts._run_pass(None)
    gnn_alerts._run_pass(None)
    assert order == ["ingest", "check", "ingest", "check"]


def test_a_failed_ingest_process_is_logged_and_not_raised():
    from strawberrywatch.integrations.night_heron import gnn_alerts

    def gone(command, timeout=None, **_):
        raise subprocess.TimeoutExpired(command, timeout)

    def failed(command, **_):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="no such table")

    def fine(command, **_):
        assert command[-1] == "strawberrywatch.watch.ingest_once"
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    assert gnn_alerts._ingest_in_child(run=gone) is False
    assert gnn_alerts._ingest_in_child(run=failed) is False
    assert gnn_alerts._ingest_in_child(run=fine) is True


def test_ingest_once_prints_one_line_of_json(store, feed, capsys, monkeypatch):
    from strawberrywatch.ingest import sql_client
    from strawberrywatch.watch import ingest_once

    # The website's source is its own sensor database
    def fetch(site, start, end, credentials=None):
        return feed.fetch(site, start, end)

    monkeypatch.setattr(sql_client, "ping", lambda credentials=None: None)
    monkeypatch.setattr(sql_client, "fetch_creek_data_sql", fetch)
    assert ingest_once.main(store=store, argv=[]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["ok"] and result["rows_new"] > 0
    assert {r["source"] for r in store.raw(SITE)} == {"sql"}


# The dashboard


def test_a_site_with_data_arriving_reads_unknown_while_checks_are_paused(store, feed):
    from strawberrywatch.dashboard import view

    ingest.run(store, token="t")
    inventory, applied = inv.load(), store.applied()[1]
    ref = time.time()
    assert view.site_statuses(store, inventory, applied, ref)[SITE]["text"] == "Normal"
    store.set_pref("checks_paused", "yes")
    status = view.site_statuses(store, inventory, applied, ref)[SITE]
    assert status == {"key": "unknown", "text": "Unknown, checks paused"}
    # with ingest 6 hours behind too, a pause can't tell a quiet site from a stopped feed
    later = view.site_statuses(store, inventory, applied, ref + 6 * 3600)[SITE]
    assert later == {"key": "unknown", "text": "Unknown, checks paused"}
    store.set_pref("checks_paused", "no")
    assert view.site_statuses(store, inventory, applied, ref + 6 * 3600)[SITE]["key"] == "late"
    assert view.collecting(store, ref) is not None
    assert view.collecting(store, ref + 6 * 3600) is None


def test_the_paused_banner_says_readings_are_still_collected(tmp_path, feed, models):
    from tests.test_website_load import _dashboard

    store = PrefixedStore(tmp_path / "site")
    store.set_source(sources.API)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    store.set_pref("checks_paused", "yes")
    page = _dashboard(store).get("/strawberrywatch/").data.decode()
    assert "Model checks paused" in page and "Readings are still being collected" not in page
    ingest.run(store, token="t")
    page = _dashboard(store).get("/strawberrywatch/").data.decode()
    assert "Readings are still being collected. Last collected" in page
    assert "Unknown, checks paused" in page
    assert "sw-square--unknown" in page


def test_a_live_check_through_the_levels_scores_what_a_direct_read_scores(
    tmp_path, models, monkeypatch
):
    """The June 2025 spill read as a replay, and again as live data that went through ingest."""
    from strawberrywatch.ingest import sql_client

    spill = "anomaly_2025_06_12_spill_sf"
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))

    direct = st.Store(tmp_path / "direct")
    direct.set_source(sources.REPLAY, event=spill)
    direct.apply(clean)
    runner.run_check(direct, models)

    replay = sources.ReplaySource(spill)
    start, end = replay.window(None)
    raw = replay.raw_tables(list(inv.load().sites), start, end)

    def fetch(site, _start, _end, credentials=None):
        frame = raw.get(site)
        if frame is None:
            return pd.DataFrame()
        return frame.rename_axis("timestamp").reset_index()

    monkeypatch.setattr(sql_client, "ping", lambda credentials=None: None)
    monkeypatch.setattr(sql_client, "fetch_creek_data_sql", fetch)
    monkeypatch.setattr(sources.SqlSource, "window", lambda _self, _lookback: (start, end))
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: replay.weather(a, b))
    live = st.Store(tmp_path / "live")
    live.set_source(sources.SITE)
    live.apply(clean)
    # The spill is from 2025, and today's prune would take its scores as too old to keep
    live.set_pref("pruned_at", repr(time.time()))
    runner.run_check(live, models)

    assert live.last_check()["ok"] and live.last_check()["source"] == sources.SITE
    assert live.last_ingest()["rows_new"] > 0

    def scored(store):
        keys = ("model", "site", "variable", "rule", "time", "score", "threshold", "base")
        return [tuple(row[k] for k in keys) for row in store.scores()]

    assert scored(direct) and scored(live) == scored(direct)


# Limits by time, for sensors that aren't on 15 minutes


def test_window_count_turns_time_into_readings_at_the_series_own_spacing():
    quarter = np.arange(50) * 900.0
    hourly = np.arange(50) * 3600.0
    assert qc.window_count(quarter, 6 * 3600, 24) == 24
    assert qc.window_count(hourly, 6 * 3600, 24) == 6
    # a second of jitter doesn't cost a step
    assert qc.window_count(np.arange(50) * 901.0, 6 * 3600, 24) == 24
    assert qc.window_count([0.0], 6 * 3600, 24) == 24


def test_rate_of_change_holds_an_hourly_sensor_to_the_same_rate():
    # 60 units over six hours is allowed, whatever the spacing
    quarter = np.arange(30) * 900.0
    hourly = np.arange(30) * 3600.0
    slow = np.arange(30) * 9.0
    assert qc.SUSPECT not in qc.rate_of_change(slow, 60.0, 6, hourly, 6 * 3600)
    assert qc.SUSPECT in qc.rate_of_change(slow * 1.5, 60.0, 6, hourly, 6 * 3600)
    # on the 15 minute grid the answer is the one the step count gave
    steady = np.arange(30) * 2.0
    by_time = qc.rate_of_change(steady, 60.0, 24, quarter, 6 * 3600)
    assert by_time.tolist() == qc.rate_of_change(steady, 60.0, 24).tolist()


def test_a_flat_hourly_sensor_is_flagged_after_six_hours_not_twenty_four():
    inventory = inv.load()
    start = pd.Timestamp("2026-06-01", tz="UTC")
    hourly = pd.date_range(start, periods=40, freq="60min")
    values = 400.0 + np.sin(np.arange(40.0))
    values[10:20] = 412.0
    flags = levels.screen_table(pd.DataFrame({COND: values}, index=hourly), SITE, inventory)[COND]
    flat = flags["flat_line"]
    # six identical hours in, the seventh reading of the run
    assert flat.iloc[15] == qc.GOOD and flat.iloc[16] == qc.SUSPECT


# Fail outside the sensor's span, suspect outside the site's own range


def test_a_reading_outside_the_sites_own_range_is_suspect_and_stays_in():
    inventory = inv.load()
    low, high = inventory.site(SITE).threshold("conductivity", "fail_range")
    stamps = pd.date_range("2026-06-01", periods=STEPS, freq="15min", tz="UTC")
    values = 400.0 + np.sin(np.arange(STEPS))
    values[10] = high + 100
    raw = {SITE: pd.DataFrame({COND: values}, index=stamps)}
    flags = levels.screen(raw, inventory)
    assert flags[SITE][COND]["gross_range"].iloc[10] == qc.SUSPECT
    assert flags[SITE][COND]["flag"].iloc[10] == qc.SUSPECT
    assert runner._screened(raw, flags)[SITE].loc[stamps[10], COND] == high + 100


def test_a_reading_outside_the_hydros_21_span_fails():
    inventory = inv.load()
    assert inventory.span(SITE, "conductivity") == (0.0, 120000.0)
    assert inventory.span(SITE, "temperature") == (-40.0, 60.0)
    assert inventory.span(SITE, "depth") == (-250.0, 10000.0)
    # a Balance Hydrologics stage isn't a Hydros 21, and nobody has given its span
    assert inventory.span("oxford", "depth") is None
    stamps = pd.date_range("2026-06-01", periods=STEPS, freq="15min", tz="UTC")
    wave = np.sin(np.arange(STEPS))
    table = pd.DataFrame(
        {COND: 400.0 + wave, DEPTH: 150.0 + wave, TEMP: 14.0 + wave / 10}, index=stamps
    )
    table.iloc[5, 0], table.iloc[6, 1], table.iloc[7, 2] = 130000.0, 10500.0, 332.0
    flags = levels.screen_table(table, SITE, inventory)
    assert flags[COND]["flag"].iloc[5] == qc.FAIL
    assert flags[DEPTH]["flag"].iloc[6] == qc.FAIL
    assert flags[TEMP]["flag"].iloc[7] == qc.FAIL


def test_a_span_that_is_not_a_pair_is_named_with_its_fix():
    text = open(inv._yaml_path()).read().replace("depth: [-250, 10000]", "depth: [10000]")
    with pytest.raises(inv.InventoryError, match="sensor_span for ctd depth") as caught:
        inv.from_text(text)
    assert "conductivity: [0, 120000]" in str(caught.value)


def test_no_default_path_takes_suspect_readings_out(models):
    """SUSPECT stays in for every model unless someone turns Settling Pool on."""
    settings = ws.defaults(models)
    for model in models:
        assert runner._policy(model.name) == (qc.FAIL, qc.MISSING)
        assert "settling_pool" not in settings["models"][model.name]["support"]
    assert levels.ALWAYS_OUT == (qc.FAIL, qc.MISSING)


def test_the_thirty_day_pull_comes_back_at_the_reviewed_level(tmp_path):
    from strawberrywatch.ingest import data_loader

    stamps = pd.date_range("2026-03-01", periods=STEPS, freq="15min", tz="UTC")
    wave = np.sin(np.arange(STEPS))
    frames = []
    for site in ("south_fork_1", "south_fork_0"):
        frames.append(
            pd.DataFrame(
                {
                    "datetime": stamps,
                    "location": site,
                    "conductivity": 600.0 + wave,
                    "depth": 150.0 + wave,
                    "temperature": 15.0 + wave / 10,
                    "rain_mm": 0.0,
                }
            )
        )
    frames[0].loc[10, "conductivity"] = -9999.0
    cache = tmp_path / "full_creek_gnn.csv"
    pd.concat(frames).to_csv(cache, index=False)
    before = cache.read_bytes()

    _, as_fetched, _ = data_loader.load_and_preprocess_data(file_path=str(cache))
    featured, reviewed, _ = data_loader.load_and_preprocess_data(
        file_path=str(cache), level="reviewed"
    )
    _, screened, _ = data_loader.load_and_preprocess_data(file_path=str(cache), level="screened")
    # the screened level drops the sentinel and keeps south_fork_0's field log period
    assert screened["conductivity"].isna().sum() == 1
    one = reviewed[reviewed["location"] == "south_fork_1"]
    assert (as_fetched["conductivity"] == -9999.0).sum() == 1
    assert (
        np.isnan(one.loc[stamps[10], "conductivity"])
        and one["conductivity"].notna().sum() == STEPS - 1
    )
    # March 2026 is inside south_fork_0's bad line in the field log, for conductivity only
    zero = reviewed[reviewed["location"] == "south_fork_0"]
    assert zero["conductivity"].isna().all() and zero["depth"].notna().all()
    assert (reviewed["rain_mm"] == 0.0).all() and len(reviewed) == len(as_fetched)
    assert featured["conductivity"].isna().sum() == STEPS + 1
    # the cache keeps the readings as fetched
    assert cache.read_bytes() == before


def test_a_depth_just_below_zero_is_good_and_below_the_floor_is_out_of_water():
    """kingman_hall's probe sits near the surface. Within 25 mm of zero is its accuracy."""
    inventory = inv.load()
    stamps = pd.date_range("2026-10-13", periods=STEPS, freq="15min", tz="UTC")
    wave = np.sin(np.arange(STEPS))
    table = pd.DataFrame(
        {COND: 500.0 + wave, DEPTH: -10.0 + wave, TEMP: 15.0 + wave / 10}, index=stamps
    )
    table.iloc[12, 1] = -25.3
    flags = levels.screen_table(table, "kingman_hall", inventory)
    near = [i for i in range(STEPS) if i != 12]
    assert (flags[DEPTH]["gross_range"].iloc[near] == qc.GOOD).all()
    assert (flags[DEPTH]["out_of_water"].iloc[near] == qc.GOOD).all()
    # below the floor: suspect, not failed, with the probe's other readings at that step
    for col in (DEPTH, COND, TEMP):
        assert flags[col]["out_of_water"].iloc[12] == qc.SUSPECT
        assert flags[col]["flag"].iloc[12] == qc.SUSPECT
    assert (flags[COND]["out_of_water"].iloc[near] == qc.GOOD).all()


def test_a_depth_under_the_span_floor_fails_and_marks_nothing_else():
    inventory = inv.load()
    assert inventory.out_of_water_depth == -25.0
    stamps = pd.date_range("2026-10-13", periods=STEPS, freq="15min", tz="UTC")
    wave = np.sin(np.arange(STEPS))
    table = pd.DataFrame(
        {COND: 500.0 + wave, DEPTH: 50.0 + wave, TEMP: 15.0 + wave / 10}, index=stamps
    )
    table.iloc[8, 1], table.iloc[16, 1] = -251.0, -249.0
    flags = levels.screen_table(table, "kingman_hall", inventory)
    assert flags[DEPTH]["flag"].iloc[8] == qc.FAIL
    assert flags[DEPTH]["out_of_water"].iloc[8] == qc.GOOD
    assert flags[COND]["out_of_water"].iloc[8] == qc.GOOD
    assert flags[DEPTH]["flag"].iloc[16] == qc.SUSPECT
    assert flags[COND]["flag"].iloc[16] == qc.SUSPECT


# Pauses


class Broken(Seen):
    """A detector whose scoring always fails."""

    calls = 0

    def scores(self, tables, weather, leave_out=None):
        self.calls += 1
        raise RuntimeError("its weights don't load")


def test_a_model_that_keeps_failing_pauses_alone_and_tries_again_in_an_hour(store, models, feed):
    settings = ws.defaults(models)
    settings["models"]["water_strider"]["on"] = True
    clean, problems, _ = ws.check(settings, models, list(inv.load().sites))
    assert not problems
    store.apply(clean)
    broken, healthy = Broken(), Seen()
    start = time.time()

    def check(at, dusk):
        runner.run_check(
            store,
            models,
            token="t",
            detectors={"dusk_crayfish": dusk, "water_strider": healthy},
            clock=lambda: at,
        )
        runner.count_outcome(store, store.last_check()["id"], at)

    for i in range(runner.PAUSE_AFTER):
        check(start + i * 900, broken)
    at = start + runner.PAUSE_AFTER * 900
    assert runner.model_paused(store, "dusk_crayfish", at)
    # the check itself didn't fail, so nothing else pauses
    assert store.last_check()["ok"] and store.pref("checks_paused") != "yes"
    assert "its weights don't load" in store.pref("model_paused_reason:dusk_crayfish")
    healthy.tables = None
    check(at, broken)
    assert broken.calls == runner.PAUSE_AFTER and healthy.tables is not None
    # an hour on it runs again, and a good turn puts it back
    retry = start + (runner.PAUSE_AFTER - 1) * 900 + runner.RETRY_SECONDS
    check(retry, Seen())
    assert not runner.model_paused(store, "dusk_crayfish", retry + 1)
    assert store.pref("model_failures:dusk_crayfish") == "0"


def test_a_pause_of_every_check_tries_again_each_hour_until_one_works(store):
    start = 1_800_000_000.0
    for _ in range(runner.PAUSE_AFTER):
        runner.count_check(store, False, start)
    assert runner.checks_paused(store, start + 60)
    assert not runner.checks_paused(store, start + runner.RETRY_SECONDS)
    # the retry failed, so another hour
    runner.count_check(store, False, start + runner.RETRY_SECONDS)
    assert runner.checks_paused(store, start + runner.RETRY_SECONDS + 60)
    runner.count_check(store, True, start + 2 * runner.RETRY_SECONDS)
    assert store.pref("checks_paused") == "no" and store.pref("check_failures") == "0"

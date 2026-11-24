"""The data check on each fetch, the daily sensor checks, and sensor alerts."""

import time

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import field_log
from strawberrywatch import inventory as inv
from strawberrywatch.ingest import levels
from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING, load_archive_by_table
from strawberrywatch.watch import health, ingest, runner, sources
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st
from tests.test_ingest_levels import Feed
from tests.test_scmg_website import PrefixedStore

COND, DEPTH, TEMP = "Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp"
DO = "AtlasSci_DO"
NOW = pd.Timestamp("2026-06-02 12:00", tz="UTC").timestamp()


@pytest.fixture(scope="module")
def inventory():
    return inv.load()


@pytest.fixture(scope="module")
def archive(inventory):
    """The archive under the logger's own column names, the way ingest saves it."""
    back = {v: k for k, v in _ALL_CHANNEL_MAPPING.items()}
    tables = load_archive_by_table(earliest=inventory.earliest_plausible_reading)
    return {site: t.rename(columns=back) for site, t in tables.items()}


def _clean(site="south_fork_1", n=48, step="15min", end=NOW):
    stamps = pd.date_range(end=pd.Timestamp(end, unit="s", tz="UTC"), periods=n, freq=step)
    wave = np.sin(np.arange(n))
    return pd.DataFrame(
        {COND: 600.0 + 10 * wave, DEPTH: 150.0 + wave, TEMP: 15.0 + wave / 10}, index=stamps
    )


def _window(table, now):
    """The table as daily() reads it: CURRENT_DAYS plus HISTORY_DAYS up to now."""
    since = pd.Timestamp(now - (health.CURRENT_DAYS + health.HISTORY_DAYS) * health.DAY, unit="s")
    until = pd.Timestamp(now, unit="s", tz="UTC")
    return table[(table.index >= since.tz_localize("UTC")) & (table.index <= until)]


# The data check on each fetch


def test_a_clean_fetch_passes_the_schema(inventory):
    assert health.check_table("south_fork_1", _clean(), inventory, NOW) == []


@pytest.mark.parametrize(
    "spoil, said",
    [
        (
            lambda t: t.assign(**{COND: t[COND].where(t.index != t.index[5], 130000.0)}),
            "outside 0 to 120000",
        ),
        (lambda t: t.iloc[::-1], "times out of order"),
        (lambda t: pd.concat([t, t.iloc[[3]]]), "unique"),
        (lambda t: t.set_axis(t.index + pd.Timedelta(days=2)), "times in the future"),
        (lambda t: t.iloc[::4], "not every 15 minutes"),
        (lambda t: t.set_axis(t.index.tz_convert("America/Los_Angeles")), "times aren't UTC"),
        (lambda t: t.drop(columns=[TEMP]), "no temperature column"),
    ],
)
def test_each_schema_check_trips_on_its_own_fault(inventory, spoil, said):
    found = health.check_table("south_fork_1", spoil(_clean()), inventory, NOW)
    assert found and any(said in p["message"] for p in found), [p["message"] for p in found]
    assert {p["kind"] for p in found} == {health.DATA_CHECK}
    assert all(p["message"].startswith("Data check at South Fork 1") for p in found)


def test_a_balance_site_is_held_to_the_quarter_hours_it_serves(inventory):
    assert inventory.site("oxford").source == inv.BALANCE
    table = _clean("oxford")
    assert health.check_table("oxford", table, inventory, NOW) == []
    assert health.check_table("oxford", table.iloc[::2], inventory, NOW)


# Sensor alerts from the screen


def test_a_failed_reading_is_a_sensor_alert_in_plain_words(inventory):
    table = _clean("south_fork_0")
    table.iloc[30, 0] = -9999.0
    flags = levels.screen_table(table, "south_fork_0", inventory)
    found = health.reading_problems("south_fork_0", table, flags, inventory)
    assert [p["message"] for p in found] == [
        "Sensor problem at South Fork 0 conductivity: reading of -9999"
    ]
    clean = _clean("south_fork_0")
    flags = levels.screen_table(clean, "south_fork_0", inventory)
    assert health.reading_problems("south_fork_0", clean, flags, inventory) == []


def test_a_probe_out_of_the_water_is_one_alert_for_the_site(inventory):
    table = _clean("kingman_hall")
    table[DEPTH] = -10.0 + np.sin(np.arange(48))
    table.iloc[40, 1] = -25.3
    flags = levels.screen_table(table, "kingman_hall", inventory)
    found = health.reading_problems("kingman_hall", table, flags, inventory)
    assert [(p["kind"], p["message"]) for p in found] == [
        (
            health.OUT_OF_WATER,
            "Sensor problem at Kingman Hall Garden: depth of -25.3 mm, possibly out of water",
        )
    ]
    table.iloc[40, 1] = -24.0
    flags = levels.screen_table(table, "kingman_hall", inventory)
    assert health.reading_problems("kingman_hall", table, flags, inventory) == []


# The daily checks


def test_a_dead_feed_is_named_after_staleness_hours(inventory):
    hours = inventory.staleness_hours
    assert health.dead_feed("oxford", COND, NOW - (hours - 0.5) * 3600, inventory, NOW) is None
    found = health.dead_feed("oxford", COND, NOW - (hours + 1) * 3600, inventory, NOW)
    assert found["kind"] == health.DEAD and "Oxford" in found["message"]
    assert "no reading since" in found["message"]
    assert "no readings" in health.dead_feed("oxford", COND, None, inventory, NOW)["message"]


def test_oxford_is_dead_after_it_stopped_on_28_october(inventory):
    """The real last reading, 2026-10-29 05:15 UTC, a day later."""
    last = pd.Timestamp("2026-10-29 05:15", tz="UTC").timestamp()
    assert health.dead_feed("oxford", COND, last, inventory, last + 86400)["kind"] == health.DEAD
    assert health.dead_feed("oxford", COND, last, inventory, last + 900) is None


def _daily(archive, inventory, site, now):
    raw = _window(archive[site], now)
    raw.attrs["site"] = site
    flags = levels.screen_table(raw, site, inventory)
    return health.daily(raw, flags, inventory, field_log.load(inventory=inventory), now)


def test_the_south_fork_3_oxygen_units_change_trips_the_level_check(archive, inventory):
    """Percent saturation (median 88) to mg/L (median 12) on 1 April 2026."""
    now = pd.Timestamp("2026-04-15", tz="UTC").timestamp()
    found = [p for p in _daily(archive, inventory, "south_fork_3", now) if p["col"] == DO]
    assert [p["kind"] for p in found] == [health.LEVEL]
    assert "possibly a unit change or recalibration" in found[0]["message"]
    assert (
        "South Fork 3 dissolved oxygen: level of 12.3 against a usual 88.1" in found[0]["message"]
    )


def test_the_oxygen_before_the_change_does_not_trip_it(archive, inventory):
    now = pd.Timestamp("2026-03-05", tz="UTC").timestamp()
    found = [p for p in _daily(archive, inventory, "south_fork_3", now) if p["col"] == DO]
    assert [p["kind"] for p in found] != [health.LEVEL]


def test_an_ordinary_fortnight_trips_nothing(archive, inventory):
    now = pd.Timestamp("2026-04-20", tz="UTC").timestamp()
    assert _daily(archive, inventory, "south_fork_2", now) == []


def test_south_fork_0s_flat_conductivity_trips_the_daily_range_check(archive, inventory):
    """Daily range 3 to 8 against a usual 52, February to April 2026."""
    now = pd.Timestamp("2026-03-08", tz="UTC").timestamp()
    found = [p for p in _daily(archive, inventory, "south_fork_0", now) if p["col"] == COND]
    assert [p["kind"] for p in found] == [health.MUTED]
    assert "possibly fouled or stuck" in found[0]["message"]


def test_south_fork_0s_conductivity_once_it_recovered_does_not(archive, inventory):
    now = pd.Timestamp("2026-05-06", tz="UTC").timestamp()
    found = [p for p in _daily(archive, inventory, "south_fork_0", now) if p["col"] == COND]
    assert found == []


# Sensor alerts: stored, shown, and emailed at most once a day


@pytest.fixture(params=["sqlite", "sw_ tables"])
def store(request, tmp_path):
    folder = tmp_path / "data"
    store = st.Store(folder) if request.param == "sqlite" else PrefixedStore(folder)
    store.set_source(sources.API)
    models = wm.registered()
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    return store


def _found(message="Sensor problem at South Fork 0 conductivity: reading of -9999"):
    return [health.problem("south_fork_0", COND, health.READING, message, -9999.0)]


def test_a_sensor_problem_is_emailed_when_it_starts_and_when_it_clears(store, monkeypatch):
    from strawberrywatch.dashboard import mail

    sent = []
    monkeypatch.setattr(
        mail,
        "health_summary",
        lambda _s, started, cleared=(), website=False: (
            sent.append(([p["message"] for p in started], [p["message"] for p in cleared])) or True
        ),
    )
    message = _found()[0]["message"]
    # seen every pass for two days: one email, when it started
    for quarter in range(0, 2 * 96):
        when = NOW + quarter * 900
        store.note_problems(_found(), when)
        health.clear(store, when, daily_ran=False)
        health.send_summary(store, when)
    assert sent == [([message], [])]
    # then a day without it: cleared, and one email saying so
    last = NOW + (2 * 96 - 1) * 900
    for quarter in range(1, 100):
        when = last + quarter * 900
        health.clear(store, when, daily_ran=False)
        health.send_summary(store, when)
    assert sent == [([message], []), ([], [message])]
    assert store.sensor_problems() == []
    # back again, it's a new problem
    later = last + 3 * 86400
    store.note_problems(_found(), later)
    health.send_summary(store, later)
    assert sent[-1] == ([message], []) and store.sensor_problems()[0]["first_at"] == later


def test_a_daily_problem_clears_when_the_next_daily_run_misses_it(store):
    dead = health.problem(
        "oxford", COND, health.DEAD, "Sensor problem at Oxford Street: no readings"
    )
    store.note_problems([dead], NOW)
    health.clear(store, NOW + 3600, daily_ran=False)
    assert len(store.sensor_problems()) == 1
    health.clear(store, NOW + 86400, daily_ran=True)
    assert store.sensor_problems() == []


def test_the_email_subject_counts_both():
    one = [{"message": "x"}]
    assert mail_subject(one, []) == "StrawberryWatch: 1 new sensor problem"
    assert mail_subject(one * 2, one) == "StrawberryWatch: 2 new sensor problems, 1 cleared"
    assert mail_subject([], one * 3) == "StrawberryWatch: 3 sensor problems cleared"


def mail_subject(started, cleared):
    from strawberrywatch.dashboard import mail

    return mail.health_subject(started, cleared)


def test_the_data_health_section_lists_the_days_problems(tmp_path):
    from strawberrywatch.dashboard import view
    from tests.test_website_load import _dashboard

    store = PrefixedStore(tmp_path / "site")
    store.set_source(sources.SITE)
    models = wm.registered()
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    page = _dashboard(store).get("/strawberrywatch/").data.decode()
    assert "<h2>Data health</h2>" in page and "No sensor problems." in page
    now = time.time()
    store.note_problems(_found(), now)
    store.set_pref("health_checked_at", repr(now))
    page = _dashboard(store).get("/strawberrywatch/").data.decode()
    assert "Sensor problem at South Fork 0 conductivity: reading of -9999. Since" in page
    assert "Sensors last checked" in page
    health.clear(store, now + 2 * 86400, daily_ran=False)
    assert view.data_health(store, now + 2 * 86400)["problems"] == []


def test_a_sensor_alert_is_never_a_creek_alert(store, monkeypatch):
    """The QC and anomaly boundary. A -9999 makes a sensor alert and no model sees it."""
    from tests.test_ingest_levels import _check

    feed = Feed()
    from strawberrywatch.ingest import api_client

    monkeypatch.setattr(api_client, "fetch_creek_data", feed.fetch)
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: feed.weather(a, b))
    feed.changed[("south_fork_1", COND, 10)] = -9999.0
    seen = _check(store, wm.registered())
    assert np.isnan(seen.tables["south_fork_1"].loc[feed.stamps[10], COND])
    problems = store.sensor_problems()
    assert any(
        p["message"] == "Sensor problem at South Fork 1 conductivity: reading of -9999"
        for p in problems
    )
    assert store.noticed() == []


def _held(monkeypatch, rise):
    """A feed whose South Fork 1 conductivity holds at one value for all its readings."""
    from strawberrywatch.ingest import api_client

    feed = Feed()
    monkeypatch.setattr(api_client, "fetch_creek_data", feed.fetch)
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: feed.weather(a, b))
    for i in range(len(feed.stamps)):
        feed.changed[("south_fork_1", COND, i)] = 412.5
        feed.changed[("south_fork_1", DEPTH, i)] = 150.0 + rise * i / (len(feed.stamps) - 1)
    return feed


def test_conductivity_held_through_a_depth_rise_is_a_sensor_alert(store, monkeypatch):
    _held(monkeypatch, rise=60.0)
    ingest.run(store, token="t")
    stuck = [p for p in store.sensor_problems() if p["kind"] == health.STUCK]
    assert [(p["site"], p["message"]) for p in stuck] == [
        (
            "south_fork_1",
            "Sensor problem at South Fork 1 conductivity: unchanged at 412.5 uS/cm for 6 hours "
            "while depth moved 60 mm, possibly stuck",
        )
    ]
    assert store.noticed() == []


def test_conductivity_held_while_the_creek_is_steady_is_not(store, monkeypatch):
    # 8 mm is under the storm trigger's 10 mm
    _held(monkeypatch, rise=8.0)
    ingest.run(store, token="t")
    assert [p for p in store.sensor_problems() if p["kind"] == health.STUCK] == []


def _hold(n, step, rise):
    """n identical conductivity readings at this spacing while depth rises by rise mm."""
    stamps = pd.date_range("2026-01-05 12:00", periods=n, freq=step, tz="UTC")
    return pd.Series(412.5, stamps), pd.Series(150.0 + np.linspace(0, rise, n), stamps)


@pytest.mark.parametrize(
    "n, step, held",
    [(9, "15min", True), (8, "15min", False), (25, "5min", True), (24, "5min", False)],
)
def test_the_stuck_window_is_two_hours_whatever_the_spacing(inventory, n, step, held):
    cond, depth = _hold(n, step, rise=30.0)
    hits = health.stuck_steps(cond, depth, inventory.stuck_hours, inventory.stuck_min_readings)
    assert bool(hits) == held
    if held:
        assert hits[-1][2] == 2.0 and round(hits[-1][3]) == 30


@pytest.mark.parametrize("n, step, held", [(3, "60min", False), (4, "40min", True)])
def test_the_stuck_window_needs_four_readings(inventory, n, step, held):
    cond, depth = _hold(n, step, rise=30.0)
    hits = health.stuck_steps(cond, depth, inventory.stuck_hours, inventory.stuck_min_readings)
    assert bool(hits) == held


def test_a_failing_health_check_never_stops_ingest(store, monkeypatch):
    from strawberrywatch.ingest import api_client

    feed = Feed()
    monkeypatch.setattr(api_client, "fetch_creek_data", feed.fetch)
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: feed.weather(a, b))

    def broken(*_a, **_k):
        raise RuntimeError("the schema is broken")

    monkeypatch.setattr(health, "check_table", broken)
    row = ingest.run(store, token="t")
    assert row["ok"] and row["rows_new"] > 0 and store.raw("south_fork_1")


def test_a_pass_a_check_starts_sends_no_email(tmp_path, monkeypatch):
    """Only the ingest job emails. A check's own pass on the website's store mustn't touch .env."""
    from strawberrywatch.dashboard import mail
    from strawberrywatch.ingest import api_client

    feed = Feed()
    monkeypatch.setattr(api_client, "fetch_creek_data", feed.fetch)
    monkeypatch.setattr(sources.Source, "weather", lambda _self, a, b: feed.weather(a, b))

    def no(*_a, **_k):
        raise AssertionError("a check's ingest pass reached the mail code")

    monkeypatch.setattr(mail, "mailing_list", no)
    monkeypatch.setattr(mail, "health_summary", no)
    store = PrefixedStore(tmp_path / "site")
    store.set_source(sources.API)
    models = wm.registered()
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    store.set_pref("email_on", "yes")
    store.add_recipient("ana@example.org")
    feed.changed[("south_fork_1", COND, 10)] = -9999.0
    runner.run_check(store, models, token="t", detectors={})
    assert store.last_ingest()["ok"] and store.sensor_problems()
    assert [p["email"] for p in store.recipients()] == ["ana@example.org"]


# The guards every test runs under


def test_no_test_reaches_the_real_env_smtp_or_cloudflare(tmp_path):
    import smtplib

    import requests

    from strawberrywatch.dashboard import mail

    assert mail.env_path() == tmp_path / ".env"
    with pytest.raises(AssertionError, match="real SMTP connection"):
        smtplib.SMTP("smtp.example.org", 587)
    with pytest.raises(AssertionError, match="real SMTP connection"):
        smtplib.SMTP_SSL("smtp.example.org", 465)
    with pytest.raises(AssertionError, match="Cloudflare"):
        requests.Session().post("https://api.cloudflare.com/client/v4/accounts/x/email/send")


def test_on_the_website_sensor_emails_go_only_to_admin_and_scmg(tmp_path, monkeypatch):
    from strawberrywatch.dashboard import mail

    store = PrefixedStore(tmp_path / "site")
    store.set_pref("email_on", "yes")
    for email, label in (
        ("admin@example.org", "Admin"),
        ("scmg@example.org", "SCMG"),
        ("ehs@example.org", "EH&S"),
        ("other@example.org", "Other"),
    ):
        store.add_recipient(email, None, store.label_named(label)["id"])
    store.add_recipient("nolabel@example.org")
    sent = []

    def deliver(people, build):
        sent.extend(build("watch@example.org", p)["to"][0] for p in people)
        return [(p["email"], True, "Sent") for p in people]

    monkeypatch.setattr(mail, "deliver_cloudflare", deliver)
    store.note_problems(_found(), NOW)
    assert mail.health_summary(store, store.sensor_problems(), website=True)
    assert sorted(sent) == ["admin@example.org", "scmg@example.org"]

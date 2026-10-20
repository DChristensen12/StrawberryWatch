"""
Dashboard changes shared by the local app and the website, through Flask's
test client with the clock fixed.
"""

import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.dashboard import create_app, security, view
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import sources
from strawberrywatch.watch import store as st

STATIC = Path(view.__file__).resolve().parent / "static"
# 2:30 pm on Tuesday 13 October 2026 at the creek
NOW = datetime(2026, 10, 13, 14, 30, tzinfo=view.PACIFIC).timestamp()
H = 3600


class FakeWatcher:
    busy = False
    next_at = None

    def __init__(self):
        self.asked = 0

    def check_now(self):
        self.asked += 1


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _store(folder, models, kind=sources.API, **source):
    store = st.Store(folder)
    store.set_source(kind, **source)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    return store


def _noticed(store, kind, last_over, detail=None, site="south_fork_1", finished=True):
    """One flagged item from one check of the given source, its last reading at last_over."""
    check = store.start_check(
        kind, 1, last_over - 6 * H, last_over, started_at=last_over + 60, detail=detail
    )
    if finished:
        store.finish_check(check, True, "", {})
    store.add_noticed(
        check,
        {
            "model": "dusk_crayfish",
            "site": site,
            "variable": "conductivity",
            "rule": "forecast_residual",
            "first_over": last_over - 900,
            "last_over": last_over,
            "peak": 9.0,
            "base": 3.0,
        },
        True,
        {},
        1.0,
    )
    return check


def _client(store, models, clock=lambda: NOW):
    app = create_app(
        store=store, models=models, watcher=FakeWatcher(), login=security.Login("t"), clock=clock
    )
    client = app.test_client()
    assert client.get("/login?token=t").status_code == 302
    return client


# Showing only what's current, in Pacific time, and zoom


def test_an_item_23_hours_old_shows_and_one_25_hours_old_doesnt(models, tmp_path):
    store = _store(tmp_path / "s", models)
    _noticed(store, sources.API, NOW - 25 * H, site="oxford")
    _noticed(store, sources.API, NOW - 23 * H, site="south_fork_1")
    _noticed(store, sources.API, NOW - 30, site="north_fork_0")
    client = _client(store, models)
    overview = client.get("/").data.decode()
    # Two sites with alerts in the last day. The headline says Alerts
    assert "Alerts at" in overview
    assert "South Fork 1" in overview.split("Alerts at")[1].split("</p>")[0]
    assert "Oxford Street" not in overview.split("Alerts at")[1].split("</p>")[0]
    unusual = client.get("/unusual").data.decode()
    assert "In the last 24 hours" in unusual
    assert "Conductivity alert at South Fork 1" in unusual
    assert "Conductivity alert at Oxford Street" not in unusual
    # Still kept, just not current
    assert "Conductivity alert at Oxford Street" in client.get("/history").data.decode()


def test_the_midnight_edge_says_yesterday_not_today(models, tmp_path):
    ten_past_midnight = datetime(2026, 10, 14, 0, 10, tzinfo=view.PACIFIC).timestamp()
    store = _store(tmp_path / "s", models)
    _noticed(store, sources.API, ten_past_midnight - 20 * 60 + 900)
    page = _client(store, models, clock=lambda: ten_past_midnight).get("/unusual").data.decode()
    assert "Yesterday at 11:50 pm" in page
    assert view.when_words(ten_past_midnight - 60, ten_past_midnight) == "Today at 12:09 am"


def test_a_replay_never_shows_on_overview_or_alerts(models, tmp_path):
    store = _store(tmp_path / "s", models, sources.SITE)
    # A replay stamped right now, like load_test.py leaves, still isn't today's alert
    _noticed(store, sources.REPLAY, NOW - H, detail="anomaly_2025_06_12_spill_sf")
    client = _client(store, models)
    assert "Alert at " not in client.get("/").data.decode()
    page = client.get("/unusual").data.decode()
    assert "No alerts in the last 24 hours." in page and "South Fork 1" not in page


def test_times_read_in_pacific_on_a_computer_set_to_utc(models, tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    try:
        assert view.clock(NOW) == "2:30 pm"
        assert view.day(NOW) == "13 Oct"
        assert (
            view.midnight(datetime(2026, 10, 13).date())
            == datetime(2026, 10, 13, tzinfo=view.PACIFIC).timestamp()
        )
        store = _store(tmp_path / "s", models)
        _noticed(store, sources.API, NOW - H)
        page = _client(store, models).get("/unusual").data.decode()
        assert "Today at 1:15 pm" in page
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_a_stopped_watcher_never_reads_as_a_calm_creek(models, tmp_path):
    store = _store(tmp_path / "s", models)
    check = store.start_check(sources.API, 1, NOW - 2 * H, NOW - H, started_at=NOW - 75 * 60)
    store.finish_check(check, True, "", {})
    store.save_readings([(s, NOW - 60, 1.0, 1.0, 1.0, None, None, None) for s in inv.load().sites])
    # finish_check stamps the real time. Pin it to when the check would have finished
    with store.db() as db:
        db.execute("UPDATE checks SET finished_at = ? WHERE id = ?", (NOW - 74 * 60, check))
    overview = _client(store, models).get("/").data.decode()
    assert "No check has finished since 1:16 pm." in overview
    assert "No alerts in the last 24 hours" not in overview
    # Within 45 minutes it's calm again
    with store.db() as db:
        db.execute("UPDATE checks SET finished_at = ? WHERE id = ?", (NOW - 10 * 60, check))
    overview = _client(store, models).get("/").data.decode()
    assert "No alerts in the last 24 hours" in overview and "No check has finished" not in overview


class Score:
    """One stored score for south_fork_1's conductivity, over its line of 2 when score is over 2."""

    def __init__(self, t, score=1.0):
        self.site, self.variable, self.rule = ("south_fork_1", "conductivity", "forecast_residual")
        self.time = datetime.fromtimestamp(t, view.PACIFIC)
        self.score, self.threshold, self.base = score, 2.0, 2.0


def test_every_full_size_chart_gets_the_zoom_buttons(models, tmp_path):
    store = _store(tmp_path / "s", models)
    check = _noticed(store, sources.API, NOW - H)
    store.save_scores(check, "dusk_crayfish", [Score(NOW - k * 900) for k in range(40)])
    item = store.noticed()[0]["id"]
    page = _client(store, models).get(f"/history/{item}").data.decode()
    assert re.search(r'data-sw-chart="score" data-sw-zoom="time"', page)
    script = (STATIC / "app.js").read_text()
    assert "from './zoom.js'" in script
    for words in ("Zoom in", "Zoom out", "Reset view"):
        assert words in script
    # Real buttons, which the keyboard and screen readers can reach
    assert "document.createElement('button')" in script and "button.type = 'button'" in script
    assert "reset.disabled" in script
    zoom = (STATIC / "zoom.js").read_text()
    assert "['Last 6 hours', 6]" in zoom and "['Last 24 hours', 24]" in zoom
    assert "['Last 2 days', 48]" in zoom


@pytest.mark.skipif(shutil.which("node") is None, reason="node isn't installed")
def test_the_zoom_arithmetic(tmp_path):
    """Range, zoom and reset, run in node against the same zoom.js the page loads."""
    shutil.copy(STATIC / "zoom.js", tmp_path / "zoom.mjs")
    (tmp_path / "check.mjs").write_text(
        """
import { clamp, ending, zoom, zoomed, MIN_SPAN } from './zoom.mjs';
const full = { min: 0, max: 48 * 3600 };
const same = (a, b) => { if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(JSON.stringify([a, b])); };
same(ending(6, full), { min: 42 * 3600, max: 48 * 3600 });
same(ending(72, full), full);
same(zoom(full, 0.5, full), { min: 12 * 3600, max: 36 * 3600 });
same(zoom({ min: 0, max: 6 * 3600 }, 2, full), { min: 0, max: 12 * 3600 });
same(zoom(full, 2, full), full);
const tiny = zoom({ min: 1000, max: 1100 }, 0.5, full);
if (tiny.max - tiny.min !== MIN_SPAN) throw new Error('zoomed past two readings');
if (zoomed(full, full) || !zoomed(ending(6, full), full)) throw new Error('reset state');
same(clamp(-100, 3500, full), { min: 0, max: 3600 });
"""
    )
    result = subprocess.run(
        ["node", "check.mjs"], cwd=tmp_path, capture_output=True, text=True, env=os.environ
    )
    assert result.returncode == 0, result.stderr


# Email in .env, the data source and its backup, crash safeguards


def _csrf(page):
    return re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()


def _post(client, path, data):
    return client.post(path, data={"csrf_token": _csrf(client.get(path).data), **data})


class Domain:
    def __init__(self, address, mx=True):
        self.normalized = address
        self.mx = [(10, "mx")] if mx else None


@pytest.fixture
def env(monkeypatch, tmp_path):
    """A .env of the test's own, and email-validator answering without DNS."""
    import email_validator
    from email_validator import EmailUndeliverableError

    from strawberrywatch.dashboard import mail

    path = tmp_path / "account" / ".env"
    path.parent.mkdir()
    path.write_text("# my settings\nALERT_EMAIL_SENDER=watch@example.org\nOTHER=1\n")
    path.chmod(0o664)

    def validate(address, check_deliverability=True, timeout=None):
        domain = address.rsplit("@", 1)[1]
        if domain == "berkely.edu":
            raise EmailUndeliverableError("The domain name berkely.edu does not exist.")
        return Domain(address, mx=domain != "slow.example")

    monkeypatch.setattr(email_validator, "validate_email", validate)
    monkeypatch.setattr(mail, "env_path", lambda: path)
    return path


def test_a_recipient_lands_on_one_env_line_and_nothing_else_changes(models, tmp_path, env):
    store = _store(tmp_path / "s", models)
    client = _client(store, models)
    assert (
        _post(
            client, "/settings/email", {"email": "ana@example.org", "name": "Ana Lee"}
        ).status_code
        == 302
    )
    assert _post(client, "/settings/email", {"email": "ben@example.org"}).status_code == 302
    lines = env.read_text().splitlines()
    assert lines[:3] == ["# my settings", "ALERT_EMAIL_SENDER=watch@example.org", "OTHER=1"]
    assert lines[3] == "ALERT_EMAIL_RECIPIENTS='Ana Lee <ana@example.org>, ben@example.org'"
    assert len(lines) == 4
    assert env.stat().st_mode & 0o777 == 0o600
    backup = env.with_name(".env.bak")
    assert backup.exists() and backup.stat().st_mode & 0o777 == 0o600
    assert "ALERT_EMAIL_RECIPIENTS='Ana Lee <ana@example.org>'" in backup.read_text()
    # The list leaves no trace in the SQLite file
    assert store.recipients() == []


def test_addresses_are_checked_and_explained_plainly(models, tmp_path, env):
    client = _client(_store(tmp_path / "s", models), models)
    assert (
        "Enter an email address with an @"
        in _post(client, "/settings/email", {"email": "ana.example.org"}).data.decode()
    )
    page = _post(client, "/settings/email", {"email": "dan@berkely.edu"}).data.decode()
    assert "berkely.edu doesn&#39;t exist. Check the spelling after the @" in page
    assert "berkely.edu" not in env.read_text()
    # A lookup that can't run still saves, with a note
    _post(client, "/settings/email", {"email": "eve@slow.example"})
    assert "eve@slow.example" in env.read_text()
    assert (
        "Couldn&#39;t confirm that slow.example accepts email"
        in client.get("/settings/email").data.decode()
    )
    assert (
        "already a recipient"
        in _post(client, "/settings/email", {"email": " EVE@slow.example "}).data.decode()
    )


def test_people_on_the_old_list_move_to_env_once(models, tmp_path, env):
    store = _store(tmp_path / "s", models)
    store.add_recipient("old@example.org", "Old Friend")
    page = _client(store, models).get("/settings/email").data.decode()
    assert "old@example.org" in page
    assert "Old Friend <old@example.org>" in env.read_text() and store.recipients() == []


def _database_down(monkeypatch):
    from strawberrywatch.ingest import sql_client

    def down(credentials=None):
        raise OSError("connection refused")

    monkeypatch.setattr(sql_client, "ping", down)


def test_an_unreachable_database_says_so_and_keeps_the_last_results(models, tmp_path, monkeypatch):
    from strawberrywatch.watch import runner

    _database_down(monkeypatch)
    store = _store(tmp_path / "s", models, sources.SITE)
    _noticed(store, sources.SITE, time.time() - H)
    # No API token, no backup
    runner.run_check(store, models)
    last = store.last_check()
    assert last["ok"] == 0 and last["message"] == sources.UNREACHABLE
    page = _client(store, models, clock=time.time).get("/").data.decode()
    assert f"Couldn&#39;t reach the database at {view.clock(last['started_at'])}." in page
    assert "Alert at South Fork 1" in page


def test_with_a_token_the_api_backs_up_an_unreachable_database(models, tmp_path, monkeypatch):
    import pandas as pd

    from strawberrywatch.ingest import api_client
    from strawberrywatch.watch import runner

    _database_down(monkeypatch)
    asked = []

    def fetch(name, start, end, variables=None, token=None):
        asked.append(token)
        return pd.DataFrame()

    monkeypatch.setattr(api_client, "fetch_creek_data", fetch)
    store = _store(tmp_path / "s", models, sources.SITE)
    # With all models off, no model loads and no weather is fetched
    settings = store.applied()[1]
    for spec in settings["models"].values():
        spec["on"] = False
    store.apply(settings)
    runner.run_check(store, models, token="api-token-123")
    last = store.last_check()
    assert last["source"] == sources.API and asked and set(asked) == {"api-token-123"}
    page = _client(store, models, clock=time.time).get("/").data.decode()
    assert "Data source: Strawberry Creek API, as a backup." in page
    assert "api-token-123" not in page


def test_two_editors_applying_at_once(models, tmp_path):
    store = _store(tmp_path / "s", models)
    first = store.applied()[1]
    mine = {**first, "repeat_hours": 12.0}
    theirs = {**first, "rain": False}
    store.save_draft(mine, owner="ana")
    store.save_draft(theirs, owner="ben")
    store.apply(theirs, by="ben", base_id=store.draft_base("ben"), owner="ben")
    with pytest.raises(st.Conflict) as caught:
        store.apply(mine, by="ana", base_id=store.draft_base("ana"), owner="ana")
    assert caught.value.applied_by == "ben"
    merged = ws.merge(first, mine, store.applied()[1])
    assert merged["repeat_hours"] == 12.0 and merged["rain"] is False


def test_apply_after_someone_else_says_so_and_keeps_my_changes(models, tmp_path):
    store = _store(tmp_path / "s", models)
    client = _client(store, models)
    draft = {**store.applied()[1], "repeat_hours": 12.0}
    store.save_draft(draft)
    store.apply({**store.applied()[1], "rain": False}, by="someone", owner="someone")
    response = client.post(
        "/settings/apply", data={"csrf_token": _csrf(client.get("/settings/review").data)}
    )
    assert response.headers["Location"] == "/settings/review"
    page = client.get("/settings/review").data.decode()
    assert "someone saved changes at" in page and "Check them before saving" in page
    assert store.draft()["repeat_hours"] == 12.0 and store.draft()["rain"] is False
    # Now it's based on theirs and goes through
    response = client.post(
        "/settings/apply", data={"csrf_token": _csrf(client.get("/settings/review").data)}
    )
    assert response.headers["Location"] == "/"
    assert store.applied()[1]["repeat_hours"] == 12.0 and store.applied()[1]["rain"] is False


def test_unreadable_settings_fall_back_to_the_last_good_ones(models, tmp_path, monkeypatch):
    from strawberrywatch.watch import runner

    store = _store(tmp_path / "s", models, sources.REPLAY, event="anomaly_2025_06_12_spill_sf")
    good_id, good = runner.read_settings(store)[:2]
    with store.db() as db:
        db.execute(
            "INSERT INTO settings_version (applied_at, settings) VALUES (?, ?)",
            (time.time(), '{"repeat_hours": -4, "models": {}}'),
        )
    settings_id, settings, notice = runner.read_settings(store)
    assert (settings_id, settings) == (good_id, good) and "last ones that could" in notice


def test_the_form_shows_each_range_and_refuses_outside_it(models, tmp_path):
    client = _client(_store(tmp_path / "s", models), models)
    page = client.get("/settings/advanced").data.decode()
    assert "Between 0.25 and 72 hours." in page
    assert "Between 0.75 and 1.5 times the tested value." in page
    assert "Between 1 and 3." in page
    assert ws.out_of_range({"repeat_hours": 0, "models": {}}) == [
        "Ignore repeats for is outside its range."
    ]
    assert ws.out_of_range(
        {"repeat_hours": 6, "models": {"dusk_crayfish": {"scales": {"x": 9}, "rain_multiplier": 0}}}
    ) == [
        "Dusk Crayfish's x line is outside its range.",
        "Dusk Crayfish's rain multiplier is outside its range.",
    ]


def test_a_model_with_missing_files_is_marked_unavailable(models, tmp_path, monkeypatch):
    from strawberrywatch.watch import runner

    def missing(name, folder=None):
        raise runner.ModelFileError("dusk_crayfish_weights.pt isn't in /nowhere")

    monkeypatch.setattr(runner, "load_detector", missing)
    store = _store(tmp_path / "s", models, sources.REPLAY, event="anomaly_2025_06_12_spill_sf")
    runner.run_check(store, models)
    assert store.last_check()["message"] == (
        "Dusk Crayfish is unavailable: its files are missing or changed"
    )


def test_a_check_without_weather_says_so_on_the_item(models, tmp_path, monkeypatch):
    from strawberrywatch.watch import runner

    monkeypatch.setattr(sources.ReplaySource, "weather", lambda *_: None)
    store = _store(tmp_path / "s", models, sources.REPLAY, event="anomaly_2025_06_12_spill_sf")
    check = runner.run_check(store, models)
    assert store.last_check()["weather_ok"] == 0
    store.add_noticed(
        check,
        {
            "model": "dusk_crayfish",
            "site": "south_fork_1",
            "variable": "conductivity",
            "rule": "forecast_residual",
            "first_over": NOW - H,
            "last_over": NOW,
            "peak": 9.0,
            "base": 3.0,
        },
        True,
        {},
        1.0,
    )
    item = store.noticed(flagged=True)[0]
    assert (
        "No weather data for this check, so rain wasn't allowed for."
        in view.item_view(store, item, NOW)["extras"]
    )


def test_a_csv_without_sensor_columns_is_refused_plainly(tmp_path):
    (tmp_path / "oxford.csv").write_text("DateTimeUTC,colour\n2026-01-01 00:00,blue\n")
    from datetime import timedelta

    source = sources.CsvSource(tmp_path)
    with pytest.raises(sources.SourceError, match="none of the sensor columns"):
        source.window(timedelta(days=2))


def test_the_score_charts_key_says_what_each_line_means_with_tick_boxes(models, tmp_path):
    store = _store(tmp_path / "s", models)
    check = _noticed(store, sources.API, NOW - H)
    store.save_scores(check, "dusk_crayfish", [Score(NOW - k * 900) for k in range(40)])
    item = store.noticed()[0]["id"]
    page = _client(store, models).get(f"/history/{item}").data.decode()
    data = re.search(r'<script type="application/json" id="chart-item">(.*?)</script>', page, re.S)
    assert json.loads(data.group(1))["about"] == ws.RULE_WORDS["forecast_residual"][1]
    assert "An alert starts when the score goes above the alert line." in page
    script = (STATIC / "app.js").read_text()
    # uPlot's legend boxes are gone, and the key's are real tick boxes, ticked when shown
    assert "label: 'How unusual'" not in script and "box.type = 'checkbox'" in script
    for words in ("Alert line.", "Untick a line to hide it."):
        assert words in script


def test_unusual_readings_shows_the_reading_with_only_the_unusual_ones_marked(models, tmp_path):
    store = _store(tmp_path / "s", models)
    check = _noticed(store, sources.API, NOW - H)
    over = {NOW - H - 900, NOW - H}
    store.save_scores(
        check,
        "dusk_crayfish",
        [Score(NOW - k * 900, 5.0 if NOW - k * 900 in over else 1.0) for k in range(40)],
    )
    store.save_readings(
        [
            ("south_fork_1", NOW - k * 900, 400.0 + k, None, None, None, None, None)
            for k in range(40)
        ]
    )
    page = _client(store, models).get("/unusual").data.decode()
    assert 'data-sw-chart="reading"' in page
    # None of the score chart's band, rain or zoom buttons
    assert 'data-sw-chart="score"' not in page and "data-sw-zoom" not in page
    assert "rain underneath" not in page
    data = re.search(r'<script type="application/json" id="chart-\d+">(.*?)</script>', page, re.S)
    chart = json.loads(data.group(1))
    assert chart["label"] == "Conductivity, µS/cm"
    marked = {x for x, u in zip(chart["x"], chart["unusual"], strict=True) if u is not None}
    assert marked == over


def test_scores_are_pruned_once_a_day(tmp_path):
    store = st.Store(tmp_path / "s")
    assert store.prune_daily(NOW) is True
    assert store.prune_daily(NOW + 3600) is False
    assert store.prune_daily(NOW + 86400) is True

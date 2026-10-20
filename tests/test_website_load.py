"""
The website's load safeguards. A model check can fail, but it mustn't take
Night Heron's own alerts down with it.
"""

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.integrations.night_heron import gnn_alerts
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import sources
from tests.test_scmg_website import PrefixedStore

NOW = datetime(2026, 10, 13, 21, 30, tzinfo=UTC)


@pytest.fixture
def store(tmp_path, monkeypatch):
    gnn_alerts.reset_state()
    monkeypatch.setattr(ws, "free_memory_mb", lambda: 8000.0)
    store = PrefixedStore(tmp_path / "site")
    store.set_source(sources.SITE)
    models = wm.registered()
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    return store


def _finished(stdout, returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


def test_a_killed_check_process_leaves_the_alert_process_running(store):
    def run(_command, **kwargs):
        # A real child process, killed the way the kernel ends one that's out of memory
        return subprocess.run(
            [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
            **kwargs,
        )

    assert gnn_alerts._check_in_child(NOW, store=store, run=run) == []
    last = store.last_check()
    assert last["ok"] == 0 and last["message"] == (
        "Stopped: the check's process was ended before it finished"
    )
    assert store.pref("check_failures") == "1" and store.pref("checks_paused") != "yes"


def test_a_check_past_the_time_limit_is_stopped(store):
    def run(command, timeout=None, **_):
        raise subprocess.TimeoutExpired(command, timeout)

    assert gnn_alerts._check_in_child(NOW, store=store, run=run) == []
    assert store.last_check()["message"] == "Stopped: took longer than 7 minutes"


def test_below_the_memory_floor_the_check_is_skipped_and_the_page_says_so(store, monkeypatch):
    monkeypatch.setattr(ws, "free_memory_mb", lambda: 300.0)
    ran = []
    gnn_alerts._check_in_child(NOW, store=store, run=lambda *a, **k: ran.append(a))
    assert ran == [] and store.last_check()["message"] == "Skipped: the server was low on memory"
    page = _dashboard(store).get("/strawberrywatch/").data.decode()
    assert "Skipped: the server was low on memory, at" in page


def test_three_failures_pause_the_checks_until_try_again(store):
    def run(command, timeout=None, **_):
        raise subprocess.TimeoutExpired(command, timeout)

    for _ in range(3):
        gnn_alerts._check_in_child(NOW, store=store, run=run)
    assert store.pref("checks_paused") == "yes"
    calls = []
    gnn_alerts._check_in_child(NOW, store=store, run=lambda *a, **k: calls.append(a))
    assert calls == []
    client = _dashboard(store)
    page = client.get("/strawberrywatch/").data
    assert (
        b"Model checks paused after 3 failed checks. Night Heron's threshold alerts still work."
        in page
    )
    token = _csrf(page)
    client.post(
        "/strawberrywatch/checks/resume",
        data={"csrf_token": token},
        headers={"Referer": "https://localhost/strawberrywatch/"},
    )
    assert store.pref("checks_paused") == "no" and store.pref("check_failures") == "0"


def test_night_heron_only_gets_alerts_to_text(store, monkeypatch):
    """StrawberryWatch emails its own list. Night Heron only sends texts."""
    result = {
        "check_id": 1,
        "ok": True,
        "message": "",
        "alerts": [
            {
                "model": "cobble_shoal",
                "site": "south_fork_1",
                "variable": "conductivity",
                "rule": "combined_fisher",
                "first_over": NOW.timestamp(),
                "values": [400, 410, 900],
            }
        ],
        "email_on": True,
    }
    run = lambda *a, **k: _finished(json.dumps(result))  # noqa: E731
    monkeypatch.setenv("GNN_ALERT_PHONES", "")
    assert gnn_alerts._check_in_child(NOW, store=store, run=run) == []
    monkeypatch.setenv("GNN_ALERT_PHONES", "5105550100")
    (alert,) = gnn_alerts._check_in_child(NOW, store=store, run=run)
    assert alert == {
        "values": [400.0, 410.0, 900.0],
        "site": "south_fork_1",
        "sensor": "conductivity",
        "alert_type": "cobble_shoal_combined_fisher",
        "emails": [],
        "phones": ["5105550100"],
    }
    # Cooling down. The same alert a pass later isn't sent again
    assert gnn_alerts._check_in_child(NOW, store=store, run=run) == []


def test_a_restart_doesnt_resend_what_was_alerted(store):
    check = store.start_check(sources.SITE, 1, 0, NOW.timestamp(), started_at=NOW.timestamp() - 600)
    store.finish_check(check, True, "", {})
    store.add_noticed(
        check,
        {
            "model": "dusk_crayfish",
            "site": "oxford",
            "variable": "conductivity",
            "rule": "forecast_residual",
            "first_over": NOW.timestamp() - 1200,
            "last_over": NOW.timestamp() - 600,
            "peak": 9.0,
            "base": 3.0,
        },
        True,
        {},
        1.0,
    )
    gnn_alerts.reset_state()
    result = {
        "ok": True,
        "alerts": [
            {
                "model": "dusk_crayfish",
                "site": "oxford",
                "variable": "conductivity",
                "rule": "forecast_residual",
                "values": [1.0],
            }
        ],
        "emails": [],
    }
    assert (
        gnn_alerts._check_in_child(
            NOW, store=store, run=lambda *a, **k: _finished(json.dumps(result))
        )
        == []
    )


def test_the_model_cap_and_the_memory_floor_refuse_with_plain_words():
    models = wm.registered()
    sites = list(inv.load().sites)
    before = ws.defaults(models)
    both = {**before, "models": {n: {**s, "on": True} for n, s in before["models"].items()}}
    limits = {"max_models_on": 1, "min_free_memory_mb": 1024.0, "check_timeout_minutes": 7.0}
    _, problems, _ = ws.check(both, models, sites, before, limits=limits)
    assert any("At most 1 model can be on at once" in p for p in problems)
    roomy = {**limits, "max_models_on": len(models)}
    _, problems, _ = ws.check(
        both,
        models,
        sites,
        before,
        limits=roomy,
        memory={"free_mb": 1500.0, "costs": {"cobble_shoal": 800.0}},
    )
    assert any("would leave less than 1,024 MB free" in p for p in problems)
    _, problems, warnings = ws.check(
        both, models, sites, before, limits=roomy, memory={"free_mb": 6000.0, "costs": {}}
    )
    assert not problems and any("hasn't been measured here yet" in w for w in warnings)


def test_one_check_in_its_own_process_records_each_models_cost(store, monkeypatch, capsys):
    from strawberrywatch.ingest import sql_client
    from strawberrywatch.watch import check_once

    replay = sources.ReplaySource("anomaly_2025_06_12_spill_sf")
    start, end = replay.window(None)
    tables = replay.tables(list(inv.load().sites), start, end)

    def fetch(site, start_time, end_time, credentials=None):
        frame = tables.get(site)
        if frame is None:
            return pd.DataFrame()
        out = frame.reset_index().rename(columns={frame.index.name or "index": "timestamp"})
        return out

    monkeypatch.setattr(sql_client, "ping", lambda credentials=None: None)
    monkeypatch.setattr(sql_client, "fetch_creek_data_sql", fetch)
    monkeypatch.setattr(sources.SqlSource, "window", lambda self, lookback: (start, end))
    monkeypatch.setattr(sources.Source, "weather", lambda self, a, b: replay.weather(a, b))
    store.set_pref("email_on", "yes")
    store.add_recipient("ana@example.org")
    store.add_recipient("ben@example.org")
    sent = _fake_mail(monkeypatch)
    # The replay's readings are from 2025, and today's prune would take them as too old to keep
    store.set_pref("pruned_at", repr(datetime.now(UTC).timestamp()))
    check_once.main(store=store)
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["ok"], result["message"]
    assert result["email_on"] and result["peak_mb"] > 0
    # StrawberryWatch sent its own emails, one per person with their own unsubscribe link
    assert sorted(to for _, to in sent) == [["ana@example.org"], ["ben@example.org"]]
    tokens = {p["email"]: p["unsubscribe_token"] for p in store.recipients()}
    for message, (to,) in sent:
        assert message["From"] == "StrawberryWatch <watch@example.org>"
        assert message["List-Unsubscribe"] == f"<{ws.website_address()}/unsubscribe/{tokens[to]}>"
        html = message.get_body(("html",)).get_content()
        assert tokens[to] in html and "cid:emblem" in html and "cid:sponsors" in html
        assert {part.get("Content-ID") for part in message.walk()} >= {"<emblem>", "<sponsors>"}
    assert store.emails()[0]["ok"] == 1
    durations = store.last_check()["durations"]
    assert "dusk_crayfish" in durations and durations["memory_mb"]["dusk_crayfish"] >= 0
    assert durations["peak_mb"] > 0
    assert {a["site"] for a in result["alerts"]} >= {"south_fork_1"}
    assert all(a["values"] for a in result["alerts"])


def _fake_mail(monkeypatch):
    """A mail server that keeps what it's sent, and the account settings to reach it."""
    import smtplib

    from strawberrywatch.dashboard import mail

    sent = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            pass

        def login(self, user, secret):
            pass

        def send_message(self, message, from_addr=None, to_addrs=None):
            sent.append((message, to_addrs))

    for name, value in {
        "ALERT_EMAIL_SENDER": "watch@example.org",
        "ALERT_EMAIL_PASSWORD": "not-a-real-one",
        "SMTP_SERVER": "smtp.example.org",
        "SMTP_PORT": "587",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(mail, "load_dotenv", lambda *a, **k: None)
    return sent


def test_the_load_test_measures_and_suggests_the_limits():
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "scripts/load_test.py", "--models", "dusk_crayfish"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "dusk_crayfish" in done.stdout and "Suggested website: block" in done.stdout
    for key in ("check_timeout_minutes: 7.5", "min_free_memory_mb:", "max_models_on:"):
        assert key in done.stdout


def _csrf(page):
    import re

    return re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()


def _dashboard(store):
    """The website dashboard with a fixed signed-in editor, no Django needed."""
    from werkzeug.test import Client

    from strawberrywatch.dashboard import create_app, profiles
    from strawberrywatch.dashboard.security import SiteLogin
    from strawberrywatch.integrations.scmg_website import NoWatcher

    app = create_app(
        store=store,
        models=wm.registered(),
        watcher=NoWatcher(),
        profile=profiles.WEBSITE,
        signing_key="test",
    )

    def signed_in(environ, start_response):
        environ[SiteLogin.USER] = "ana"
        environ[SiteLogin.CAN_EDIT] = True
        environ["SCRIPT_NAME"] = "/strawberrywatch"
        environ["PATH_INFO"] = environ["PATH_INFO"].removeprefix("/strawberrywatch")
        return app(environ, start_response)

    client = Client(signed_in)
    real_get, real_post = client.get, client.post
    client.get = lambda url, **k: real_get(url, base_url="https://localhost", **k)
    client.post = lambda url, **k: real_post(url, base_url="https://localhost", **k)
    return client

"""
The local watcher: settings, sources, decisions, the 7 day preview, the store
and the 15 minute thread. Tests from the build spec's steps 1, 2 and 5 name
their step in the docstring.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import inventory as inv
from strawberrywatch import paths
from strawberrywatch.serving import Score, notice
from strawberrywatch.watch import decide, runner, sources
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

SPILL = "anomaly_2025_06_12_spill_sf"


@pytest.fixture(scope="module")
def models():
    return wm.registered()


@pytest.fixture(scope="module")
def sites():
    return list(inv.load().sites)


def _store(folder, models, sites, kind=sources.REPLAY, event=SPILL, edit=None):
    store = st.Store(folder)
    store.set_source(kind, event=event if kind == sources.REPLAY else None)
    settings = ws.defaults(models)
    if edit:
        edit(settings)
    clean, problems, _ = ws.check(settings, models, sites)
    assert not problems
    store.apply(clean)
    return store


@pytest.fixture(scope="module")
def replayed(tmp_path_factory, models, sites):
    """One replay of the June 2025 spill, checked once with the real models."""
    folder = tmp_path_factory.mktemp("replayed")
    store = _store(folder, models, sites)
    runner.run_check(store, models)
    return folder


def _copy(folder, tmp_path):
    shutil.copytree(folder, tmp_path / "data")
    return st.Store(tmp_path / "data")


# Settings


def test_the_three_choices_are_the_decided_values(models):
    assert [(k, v) for k, _, v, _ in ws.PRESETS] == [
        ("fewer", 1.25),
        ("balanced", 1.0),
        ("more", 0.8),
    ]
    settings = ws.defaults(models)
    assert ws.alerts_choice(settings) == "balanced"
    for key in ("fewer", "more", "balanced"):
        assert ws.alerts_choice(ws.apply_choice(settings, key)) == key
    settings["models"]["dusk_crayfish"]["scales"]["level_shift"] = 1.1
    assert ws.alerts_choice(settings) == "custom"


def test_unknown_names_are_refused_with_strawberrywatchs_message(models, sites):
    settings = ws.defaults(models)
    settings["models"]["dusk_crayfsh"] = settings["models"]["dusk_crayfish"]
    settings["models"]["cobble_shoal"]["support"] = ["trail_bed"]
    settings["dont_flag"] = ["oxford", "../../etc"]
    _, problems, _ = ws.check(settings, models, sites)
    assert any("unknown model 'dusk_crayfsh'" in p for p in problems)
    assert any("unknown support module 'trail_bed'" in p for p in problems)
    assert any("no site called '../../etc'" in p for p in problems)


def test_site_names_only_come_from_the_inventory(models, sites):
    settings = ws.defaults(models)
    settings["dont_flag"] = ["oxford", "not_a_site"]
    clean, _, _ = ws.check(settings, models, sites)
    assert clean["dont_flag"] == ["oxford"]


def test_dusk_crayfish_cant_leave_a_site_out(models, sites):
    settings = ws.defaults(models)
    settings["models"]["dusk_crayfish"]["leave_out"] = ["oxford"]
    _, problems, _ = ws.check(settings, models, sites)
    assert any("Dusk Crayfish can't leave sites out" in p for p in problems)


def test_cobble_shoal_refuses_going_under_half_its_sensors_and_warns_near_it(models, sites):
    """Build order step 5."""
    settings = ws.defaults(models)
    cobble = settings["models"]["cobble_shoal"]
    cobble["leave_out"] = ["scnf010", "north_fork_0", "south_fork_1"]
    _, problems, _ = ws.check(settings, models, sites)
    assert any("under half its sensors" in p for p in problems)
    cobble["leave_out"] = ["scnf010", "oxford"]
    _, problems, warnings = ws.check(settings, models, sites)
    assert not problems and any("one more offline site would stop it" in w for w in warnings)


def test_a_rain_modulator_on_dusk_crayfish_is_refused(models, sites, monkeypatch):
    """check_collision, against the model class rather than the serving detector."""
    from strawberrywatch.support_modules import SUPPORT_REGISTRY, SupportModulator

    class Rain(SupportModulator):
        name = "rain"

        def multipliers(self, window, nodes):
            return np.ones((1, len(nodes)))

    monkeypatch.setitem(SUPPORT_REGISTRY, "rain", Rain)
    settings = ws.defaults(models)
    settings["models"]["dusk_crayfish"]["support"] = ["rain"]
    _, problems, _ = ws.check(settings, models, sites)
    assert any("already handles 'rain' internally" in p for p in problems)


def test_another_model_cant_be_turned_on_once_checks_take_half_the_interval(models, sites):
    applied = ws.defaults(models)
    settings = ws.defaults(models)
    settings["models"]["cobble_shoal"]["on"] = True
    _, problems, warnings = ws.check(settings, models, sites, applied, last_check_seconds=500)
    assert any("no more models can be turned on" in p for p in problems)
    _, problems, warnings = ws.check(settings, models, sites, applied, last_check_seconds=20)
    assert not problems and any("Cobble Shoal hasn't been retrained" in w for w in warnings)


def test_ranges_hold(models, sites):
    settings = ws.defaults(models)
    settings["models"]["dusk_crayfish"]["scales"]["level_shift"] = 0.5
    settings["models"]["dusk_crayfish"]["rain_multiplier"] = 3.5
    settings["repeat_hours"] = 0
    _, problems, _ = ws.check(settings, models, sites)
    assert len(problems) == 3


def test_describe_words_every_change(models):
    before = ws.defaults(models)
    after = ws.apply_choice(before, "fewer")
    after["models"]["cobble_shoal"]["on"] = True
    after["rain"] = False
    after["dont_flag"] = ["oxford"]
    sentences = [row["sentence"] for row in ws.describe(before, after, models)]
    assert "Alert sensitivity set to Fewer." in sentences
    assert "Cobble Shoal turned on." in sentences
    assert "Rain allowance turned off." in sentences
    assert "Alerts turned off for Oxford Street." in sentences
    assert ws.describe(before, before, models) == []


# Decisions


def _scores(rows):
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return [
        Score(site, "conductivity", rule, t0 + pd.Timedelta(minutes=15 * i), value, line, line)
        for site, rule, values, line in rows
        for i, value in enumerate(values)
    ]


def test_decide_says_what_serving_notice_says():
    rng = np.random.default_rng(3)
    rows = []
    for site in ("oxford", "south_fork_1"):
        for rule in ("forecast_residual", "level_shift"):
            rows.append((site, rule, list(rng.gamma(2, 2, 60)), 6.0))
    scored = _scores(rows)
    grouped = decide.series(scored)
    start, end = scored[0].time.timestamp(), max(s.time for s in scored).timestamp()
    for scale in (0.8, 1.0, 1.25):
        for min_steps in (1, 3):
            ours = decide.notice_window(
                grouped, start, end, {"forecast_residual": scale, "level_shift": scale}, min_steps
            )
            theirs = notice(scored, min_steps, scale)
            assert sorted((i["site"], i["rule"], i["first_over"]) for i in ours) == sorted(
                (n.site, n.rule, n.when.timestamp()) for n in theirs
            )


def test_the_same_thing_stays_one_item_and_quick_repeats_are_ignored():
    item = {"first_over": 1000.0}
    assert decide.classify(item, None, 5000, 6) == decide.NEW
    assert decide.classify(item, {"first_over": 1000.0, "checked_at": 0}, 50000, 6) == decide.SAME
    assert (
        decide.classify({"first_over": 9000.0}, {"first_over": 1000.0, "checked_at": 0}, 3600, 6)
        == decide.REPEAT
    )
    assert (
        decide.classify(
            {"first_over": 9000.0}, {"first_over": 1000.0, "checked_at": 0}, 7 * 3600, 6
        )
        == decide.NEW
    )


# Sources and checks


def test_a_replay_notices_what_the_events_own_tests_expect(replayed, tmp_path, models):
    """Build order step 2. tests/events.yaml grades south_fork_1 as an anomaly for this event."""
    store = _copy(replayed, tmp_path)
    labels = next(e for e in sources.replay_events() if e["folder"] == SPILL)["labels"]
    expected = {site for site, label in labels.items() if label == "anomaly"}
    noticed = {(n["model"], n["site"]) for n in store.noticed(flagged=True)}
    assert {("dusk_crayfish", site) for site in expected} <= noticed
    last = store.last_check()
    assert last["ok"] and last["durations"]["dusk_crayfish"] > 0


def test_replays_and_csv_snap_readings_onto_the_quarter_hour():
    """Otherwise a Mayfly stamped 07:00:31 doesn't count as reporting in detect_anomalies."""
    src = sources.ReplaySource(SPILL)
    start, end = src.window(None)
    table = src.tables(["south_fork_1"], start, end)["south_fork_1"]
    assert (table.index == table.index.floor("15min")).all()


def test_csv_files_outside_the_folder_or_too_big_are_not_read(tmp_path, monkeypatch):
    outside = tmp_path / "outside.csv"
    outside.write_text("DateTimeUTC,Meter_Hydros21_Cond\n2026-01-01 00:00:00,400\n")
    folder = tmp_path / "csv"
    folder.mkdir()
    (folder / "oxford.csv").symlink_to(outside)
    source = sources.CsvSource(folder)
    assert (
        source.tables(
            ["oxford", "../outside"], pd.Timestamp("2025", tz="UTC"), pd.Timestamp("2027", tz="UTC")
        )
        == {}
    )
    (folder / "south_fork_1.csv").write_text(
        "DateTimeUTC,Meter_Hydros21_Cond\n2026-01-01 00:00:00,400\n"
    )
    monkeypatch.setattr(sources, "MAX_CSV_BYTES", 10)
    with pytest.raises(sources.SourceError, match="wasn't read"):
        sources.CsvSource(folder).tables(
            ["south_fork_1"], pd.Timestamp("2025", tz="UTC"), pd.Timestamp("2027", tz="UTC")
        )


def test_the_api_source_uses_the_token_its_given(monkeypatch, sites):
    """With the API faked: one table back for every site asked for."""
    from strawberrywatch.ingest import api_client

    seen = []

    def fake(site, start, end, variables=None, token=None):
        seen.append((site, token))
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start, periods=4, freq="15min", tz="UTC"),
                "Meter_Hydros21_Cond": 400.0,
                "station_id": site,
            }
        )

    monkeypatch.setattr(api_client, "fetch_creek_data", fake)
    source = sources.from_config({"kind": sources.API}, token="secret-token-123")
    assert "secret" not in repr(source)
    start, end = source.window(timedelta(days=2))
    tables = source.tables(sites, start, end)
    assert set(tables) == set(sites)
    assert {token for _, token in seen} == {"secret-token-123"}


def test_the_token_never_reaches_the_store(monkeypatch, tmp_path, models, sites):
    from strawberrywatch.ingest import api_client

    monkeypatch.setattr(api_client, "fetch_creek_data", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sources.Source, "weather", lambda *_: None)
    store = _store(tmp_path / "data", models, sites, kind=sources.API)
    runner.run_check(store, models, token="secret-token-456")
    for path in (tmp_path / "data").iterdir():
        assert b"secret-token-456" not in path.read_bytes()


def test_the_seven_day_preview_matches_what_checks_decided(replayed, tmp_path, models):
    """At the applied values the preview replays to what checks recorded."""
    from strawberrywatch.dashboard import view

    store = _copy(replayed, tmp_path)
    runner.run_check(store, models)  # the same thing again, which stays one item
    applied = store.applied()[1]
    counts = view.preview_counts(store, models, applied)
    recorded = len([n for n in store.noticed(flagged=True) if n["model"] == "dusk_crayfish"])
    assert counts["dusk_crayfish"] == {"would": recorded, "did": recorded}
    more = view.preview_counts(store, models, ws.apply_choice(applied, "more"))
    fewer = view.preview_counts(store, models, ws.apply_choice(applied, "fewer"))
    assert more["dusk_crayfish"]["would"] >= recorded >= fewer["dusk_crayfish"]["would"]


def test_a_site_not_flagged_is_still_checked_and_recorded(tmp_path, models, sites):
    def edit(settings):
        settings["dont_flag"] = ["south_fork_1"]

    store = _store(tmp_path / "data", models, sites, edit=edit)
    runner.run_check(store, models)
    assert store.scores(site="south_fork_1")
    everything = [n for n in store.noticed() if n["site"] == "south_fork_1"]
    assert everything and not any(n["flagged"] for n in everything)


def test_model_files_are_checked_against_their_hashes(tmp_path):
    from strawberrywatch.serving import detector_class

    cls = detector_class("dusk_crayfish")
    for name in cls.FILES:
        shutil.copy(paths.checkpoints_dir() / name, tmp_path / name)
    runner.verify_files(cls, tmp_path)
    with open(tmp_path / cls.FILES[0], "ab") as f:
        f.write(b"tampered")
    with pytest.raises(runner.ModelFileError, match="doesn't match"):
        runner.verify_files(cls, tmp_path)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_the_data_folder_is_private(tmp_path):
    store = st.Store(tmp_path / "data")
    assert (os.stat(store.folder).st_mode & 0o777) == 0o700
    assert (os.stat(store.path).st_mode & 0o777) == 0o600


def test_checks_never_overlap_and_stop_with_the_app(tmp_path, models, sites, monkeypatch):
    """Build order step 2, with the check itself stood in for."""
    store = _store(tmp_path / "data", models, sites, kind=sources.SITE)
    running, ran = threading.Event(), []

    def slow(*_args, **_kw):
        running.set()
        time.sleep(0.6)
        ran.append(time.time())
        # A good check. Three in a row don't pause the watcher
        check = store.start_check(sources.SITE, 1, None, None)
        store.finish_check(check, True, "", {})
        return check

    monkeypatch.setattr(runner, "run_check", slow)
    watcher = runner.Watcher(store, models, interval=timedelta(seconds=1))
    first = threading.Thread(target=watcher.check)
    first.start()
    running.wait(2)
    assert watcher.busy and watcher.check() is None
    first.join()
    watcher.start()
    time.sleep(2.6)
    watcher.stop(timeout=5)
    stopped_at = len(ran)
    assert stopped_at >= 3
    time.sleep(1.5)
    assert len(ran) == stopped_at


def test_the_watcher_hands_each_check_on_once_it_is_recorded(tmp_path, models, sites, monkeypatch):
    """How the dashboard's emails hear about a check."""
    store = _store(tmp_path / "data", models, sites, kind=sources.SITE)
    monkeypatch.setattr(runner, "run_check", lambda *_a, **_k: 7)
    heard = []
    watcher = runner.Watcher(store, models, after_check=heard.append)
    assert watcher.check() == 7 and heard == [7]


def test_the_watcher_hands_each_check_the_api_token_and_the_model_cap(
    tmp_path, models, sites, monkeypatch
):
    store = _store(tmp_path / "data", models, sites, kind=sources.SITE)
    seen = {}
    monkeypatch.setattr(runner, "run_check", lambda *_a, **k: seen.update(k))
    runner.Watcher(store, models, api_token=lambda: "api-token-789").check()
    assert seen["token"] == "api-token-789"
    assert seen["max_models"] == int(ws.website_limits()["max_models_on"])


def test_three_failed_checks_pause_the_watcher_until_restarted(
    tmp_path, models, sites, monkeypatch
):
    """As gnn_alerts pauses the website's checks."""
    store = _store(tmp_path / "data", models, sites, kind=sources.SITE)

    def failing(store, *_a, **_k):
        check = store.start_check(sources.SITE, 1, None, None)
        store.finish_check(check, False, sources.UNREACHABLE, {})
        return check

    monkeypatch.setattr(runner, "run_check", failing)
    watcher = runner.Watcher(store, models)
    for _ in range(runner.PAUSE_AFTER):
        assert watcher.check() is not None
    assert store.pref("checks_paused") == "yes" and watcher.check() is None
    # What Restart checks does
    store.set_pref("check_failures", "0")
    store.set_pref("checks_paused", "no")
    assert watcher.check() is not None


def test_nothing_imports_django_or_the_night_heron_hook():
    """Build order step 1."""
    code = (
        "import sys, strawberrywatch.dashboard, strawberrywatch.dashboard.routes, strawberrywatch.cli, "
        "strawberrywatch.watch.runner, strawberrywatch.watch.sources; "
        "print([m for m in sys.modules if m.split('.')[0] == 'django' or 'night_heron' in m])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    for path in Path("strawberrywatch/watch").rglob("*.py"):
        assert "night_heron" not in path.read_text(), path
    for path in Path("strawberrywatch/dashboard").rglob("*.py"):
        assert "night_heron" not in path.read_text(), path


def test_no_email_addresses_in_the_package():
    """Build order step 1."""
    pattern = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    for path in Path("strawberrywatch").rglob("*"):
        if path.is_file() and path.suffix in {
            ".py",
            ".html",
            ".js",
            ".css",
            ".json",
            ".yaml",
            ".md",
            ".txt",
            ".csv",
        }:
            assert not pattern.search(path.read_bytes()), path

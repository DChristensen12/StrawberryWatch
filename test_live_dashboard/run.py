"""
The website's dashboard on this computer, to try looks and buttons before they
go live:

    python test_live_dashboard/run.py

then open http://127.0.0.1:8791/strawberrywatch/

Same app and address as strawberrycreek.org. The only stand-ins are sign-in
(always you), the store (a SQLite file in test_live_dashboard/data) and the
readings (real checks of a past event, moved to today). It can't reach the
website's database or send email.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple
from werkzeug.wrappers import Request, Response

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "site"
# A copy of the installed model files. Model updates writes here and leaves the
# repository's checkpoints alone
MODEL_FILES = HERE / "data" / "checkpoints"
# Stands in for StrawberryWatch's main branch on GitHub
GITHUB = HERE / "data" / "github"
GITHUB_STATES = ("newer", "same", "broken", "offline")
PORT = 8791
# 10 of 11 sites reporting, and one unusual reading 17 hours before the end
EVENT = "anomaly_2025_09_10_overnight_sf"
STEP = 15 * 60
# Cobble Shoal and Flame Skimmer only judge the last 2 hours of a check. A
# check every 2 hours covers what Night Heron's 15 minute checks would
EVERY_HOURS = 2
DAYS = 3
# What a check writes. Moving these forward together keeps the sample check the latest one.
CHECK_TIMES = {
    "checks": ("started_at", "finished_at", "window_start", "window_end"),
    "readings": ("time",),
    "scores": ("time",),
    "weather": ("time",),
    "noticed": ("first_over", "last_over", "created_at"),
}


class Moment:
    """The event as a live check at one moment would have seen it, just the lookback before then."""

    def __init__(self, replay, start, end):
        self.replay, self.start, self.end = replay, start, end
        self.kind, self.detail = replay.kind, replay.detail

    def window(self, _lookback):
        return self.start, self.end

    def tables(self, names, start, end):
        return self.replay.tables(names, start, end)

    def weather(self, start, end):
        return self.replay.weather(start, end)


def run_checks(store, models, event, max_models):
    """
    Runs the checks Night Heron would have run over the event's last days. One
    check of the full event would show no alerts from Cobble Shoal or Flame Skimmer.
    """
    from unittest import mock

    import pandas as pd

    from strawberrywatch.watch import runner, sources
    from strawberrywatch.watch.check_once import peak_mb

    replay = sources.ReplaySource(event)
    first, last = replay.window(runner.LOOKBACK)
    ends = [
        last - pd.Timedelta(hours=h)
        for h in range(DAYS * 24 - EVERY_HOURS, -1, -EVERY_HOURS)
        if last - pd.Timedelta(hours=h) > first
    ]
    for moment in [Moment(replay, max(first, end - runner.LOOKBACK), end) for end in ends]:
        began, started = time.time(), moment.end.timestamp() + 60
        with mock.patch.object(sources, "from_config", lambda *_a, now=moment, **_k: now):
            check = runner.run_check(
                store,
                models,
                clock=lambda at=started: at,
                memory=peak_mb,
                max_models=max_models,
            )
        # Its start is the moment's; its length is how long it really took
        with store.db() as db:
            db.execute(
                "UPDATE checks SET finished_at = started_at + ? WHERE id = ?",
                (time.time() - began, check),
            )


def seed(folder, event, calm=False, on=None):
    """A fresh store: setup_tables' first version, then the checks Night Heron would have run."""
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import models as wm
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch import sources
    from strawberrywatch.watch import store as st

    shutil.rmtree(folder, ignore_errors=True)
    store = st.Store(folder)
    models = wm.registered()
    store.set_source(sources.REPLAY, event=event)
    first = ws.defaults(models)
    if on is not None:
        for m in models:
            first["models"][m.name]["on"] = m.name in on
    clean, problems, _ = ws.check(first, models, list(inv.load().sites))
    if problems:
        sys.exit(" ".join(problems))
    store.apply(clean, by="setup_tables")
    run_checks(store, models, event, int(ws.website_limits()["max_models_on"]))

    now = time.time()
    with store.db() as db:
        check = db.execute("SELECT * FROM checks ORDER BY id DESC LIMIT 1").fetchone()
        if check is None or not check["ok"]:
            sys.exit(f"The sample check didn't work: {check and check['message']}")
        # Last reading on the latest quarter hour and the check finished a minute
        # after, like Night Heron's. Any later and a site shows as late.
        end = (now // STEP) * STEP
        if end + check["finished_at"] - check["window_end"] > now:
            end -= STEP
        delta = end - check["window_end"]
        shift(db, delta)
        db.execute("UPDATE checks SET source = ?, source_detail = NULL", (sources.SITE,))
        db.execute("UPDATE noticed SET created_at = first_over + ?", (STEP,))
        db.execute("UPDATE readings SET live = 1")
        # On the website it's always been the site's database, set up a week back
        start = check["window_start"] + delta
        db.execute(
            "UPDATE source SET kind = ?, event = NULL, changed_at = ?", (sources.SITE, start)
        )
        db.execute("UPDATE settings_version SET applied_at = ?", (start,))
        if calm:
            db.execute("DELETE FROM noticed WHERE last_over >= ?", (now - 24 * 3600,))
    store.set_pref("email_on", "yes")
    labels = {lb["name"]: lb["id"] for lb in store.labels()}
    store.add_recipient("first.person@example.org", "First Person", labels["SCMG"])
    store.add_recipient("second.person@example.org", None, labels["Public"], "Sign-up")
    return store


def shift(db, delta):
    """Every check time moved by delta, through negatives to keep primary keys from colliding."""
    # Table and column names come from CHECK_TIMES, and delta goes in as ?
    for table, columns in CHECK_TIMES.items():
        away = ", ".join(f"{c} = -({c} + ?)" for c in columns)
        db.execute(f"UPDATE {table} SET {away}", [delta] * len(columns))  # nosec B608
        back = ", ".join(f"{c} = -{c}" for c in columns)
        db.execute(f"UPDATE {table} SET {back}")  # nosec B608


def keep_current(store):
    """
    Moves the sample check forward in 15 minute steps. It stays no older than
    a real one would be. Notes, settings and the email list stay put.
    """
    with store.db() as db:
        last = db.execute("SELECT MAX(finished_at) FROM checks").fetchone()[0]
        steps = int((time.time() - last) // STEP) if last else 0
        if steps > 0:
            shift(db, steps * STEP)


class StandInGate:
    """Stands in for the site's gate. It's always the same person."""

    def __init__(self, app, store, username, can_edit):
        self.app, self.store = app, store
        self.username, self.can_edit = username, can_edit

    def __call__(self, environ, start_response):
        from strawberrywatch.dashboard.security import SiteLogin
        from strawberrywatch.integrations.scmg_website.gate import PUBLIC_PATHS

        keep_current(self.store)
        path = environ.get("PATH_INFO", "")
        # The sign-up pages see a visitor, the way the real gate lets them through
        if path != "/signup" and not path.startswith(PUBLIC_PATHS):
            environ[SiteLogin.USER] = self.username
            environ[SiteLogin.CAN_EDIT] = self.can_edit
        return self.app(environ, start_response)


@Request.application
def rest_of_site(request):
    """The parts of strawberrycreek.org the dashboard links to."""
    from strawberrywatch.integrations.scmg_website import MOUNT

    if request.path == "/":
        return Response(status=302, headers={"Location": MOUNT + "/"})
    if request.path == "/logout/":
        page = (
            "<!DOCTYPE html><title>Signed out</title>"
            "<p>On the website this is strawberrycreek.org's own sign-out page.</p>"
            f'<p><a href="{MOUNT}/">Back to StrawberryWatch</a></p>'
        )
        return Response(page, content_type="text/html; charset=utf-8")
    return Response("Not part of the test dashboard.", status=404)


def copy_model_files():
    """The files the models were installed with, the way the website's checkpoints folder holds them."""
    from strawberrywatch.watch.runner import HASHES_PATH

    shutil.rmtree(MODEL_FILES, ignore_errors=True)
    MODEL_FILES.mkdir(parents=True)
    for name in json.loads(HASHES_PATH.read_text()):
        shutil.copy(HERE.parent / "checkpoints" / name, MODEL_FILES / name)


def stand_in_github(state):
    """
    A fake main branch for Model updates. newer has a recalibrated Water
    Strider, broken adds Cobble Shoal weights that won't load, same matches
    what's installed, and offline can't be reached.
    """
    from strawberrywatch.watch import model_updates as mu
    from strawberrywatch.watch.runner import HASHES_PATH

    shutil.rmtree(GITHUB, ignore_errors=True)
    files = GITHUB / mu.FILES_IN_REPO
    files.mkdir(parents=True)
    hashes = json.loads(HASHES_PATH.read_text())
    for name in hashes:
        shutil.copy(HERE.parent / "checkpoints" / name, files / name)
    if state in ("newer", "broken"):
        calibration = json.loads((files / "water_strider_calibration.json").read_text())
        calibration["made"] = time.strftime("%Y-%m-%d")
        (files / "water_strider_calibration.json").write_text(json.dumps(calibration, indent=2))
    if state == "broken":
        weights = files / "cobble_shoal_weights.pt"
        weights.write_bytes(weights.read_bytes()[:4096])
    for name in hashes:
        hashes[name] = hashlib.sha256((files / name).read_bytes()).hexdigest()
    listed = GITHUB / mu.HASHES_IN_REPO
    listed.parent.mkdir(parents=True)
    listed.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    return mu.FolderGitHub(GITHUB, offline=state == "offline")


def build(store, username, can_edit, github, port=PORT):
    """The site with the dashboard mounted, as INSTALL.md's wsgi.py lines do it."""
    from strawberrywatch.dashboard import create_app, profiles
    from strawberrywatch.integrations.scmg_website import MOUNT, NoWatcher
    from strawberrywatch.watch import models as wm
    from strawberrywatch.watch.model_updates import ThreadUpdater
    from strawberrywatch.watch.store_mysql import MysqlStore

    models = wm.registered()
    app = create_app(
        store=store,
        models=models,
        watcher=NoWatcher(),
        profile=profiles.WEBSITE,
        # The website's own lasting key, kept in prefs. A restart won't sign you out
        signing_key=MysqlStore.session_key(store),
        # The website uses a process per update, a thread is fine here
        updater=ThreadUpdater(models, MODEL_FILES, github),
        # Stands in for the addresses on Night Heron's alert rules
        copy_list=lambda: ["third.person@example.org", "first.person@example.org"],
    )
    app.config["SESSION_COOKIE_PATH"] = MOUNT
    # Links in emails and on Alert examples point here, not at the real site
    app.config["PUBLIC_ADDRESS"] = f"http://127.0.0.1:{port}{MOUNT}"
    app.config["SIGN_OUT_URL"] = "/logout/"
    # The website is HTTPS. This is plain http on 127.0.0.1, where a Secure cookie
    # wouldn't come back and each form would fail its CSRF check.
    app.config["SESSION_COOKIE_SECURE"] = False
    # A template edit shows on the next refresh, no restart needed
    app.jinja_env.auto_reload = True
    return DispatcherMiddleware(rest_of_site, {MOUNT: StandInGate(app, store, username, can_edit)})


def main(argv=None):
    from strawberrywatch.serving import available
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch import sources
    from strawberrywatch.watch import store as st

    events = [e["folder"] for e in sources.replay_events()]
    parser = argparse.ArgumentParser(
        description="The website's dashboard on this computer, with sample readings."
    )
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--fresh", action="store_true", help="start over, dropping notes and changes made here"
    )
    parser.add_argument(
        "--event",
        choices=events,
        default=EVENT,
        help="which past event the readings come from (starts over)",
    )
    parser.add_argument(
        "--calm", action="store_true", help="nothing unusual in the last 24 hours (starts over)"
    )
    parser.add_argument(
        "--paused", action="store_true", help="show the banner for checks paused after failures"
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="signed in as someone outside STRAWBERRYWATCH_EDITOR_GROUP",
    )
    parser.add_argument(
        "--models",
        help=f"which models are on and run, comma separated, from {', '.join(available())} "
        "(starts over). Without it, what the website starts with: Dusk Crayfish.",
    )
    parser.add_argument(
        "--github",
        choices=GITHUB_STATES,
        default="newer",
        help="what the stand-in GitHub holds for Model updates",
    )
    parser.add_argument("--user", default="test_user", help="the username changes and notes carry")
    args = parser.parse_args(argv)

    on = None
    if args.models is not None:
        on = [name.strip() for name in args.models.split(",") if name.strip()]
        unknown = [name for name in on if name not in available()]
        if unknown:
            parser.error(f"no model called {', '.join(unknown)}; there's {', '.join(available())}")
        cap = int(ws.website_limits()["max_models_on"])
        if len(on) > cap:
            parser.error(f"the website runs at most {cap} at once (max_models_on in settings.yaml)")

    logging.basicConfig(level=logging.WARNING)
    fresh = args.fresh or args.calm or args.event != EVENT or on is not None
    if fresh or not (DATA / st.DB_NAME).exists():
        print(f"Running the checks Night Heron would have run on {args.event} (under a minute)...")
        store = seed(DATA, args.event, calm=args.calm, on=on)
        copy_model_files()
    else:
        store = st.Store(DATA)
    store.set_pref("checks_paused", "yes" if args.paused else "no")
    if not MODEL_FILES.exists():
        copy_model_files()
    github = stand_in_github(args.github)

    print(f"\nOpen http://127.0.0.1:{args.port}/strawberrywatch/\n")
    run_simple(
        "127.0.0.1",
        args.port,
        build(store, args.user, not args.view_only, github, args.port),
        threaded=True,
    )


if __name__ == "__main__":
    main()

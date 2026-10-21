"""
The SQLite file: settings versions and drafts, checks, the scores and readings
they saw, and the alerts.

Every query is parameterized, and tokens and passwords stay out of here. The
SQL sticks to what SQLite and MySQL both understand (REPLACE INTO, backticks
around `key`, no two-argument MAX), and store_mysql.py reuses it.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import sys
import threading
import time
from pathlib import Path

DB_NAME = "strawberrywatch.db"
SCORE_DAYS = 30
READING_DAYS = 8
PRUNE_EVERY = 86400

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings_version (
    id INTEGER PRIMARY KEY, applied_at REAL NOT NULL, settings TEXT NOT NULL, applied_by TEXT);
CREATE TABLE IF NOT EXISTS draft (
    id INTEGER PRIMARY KEY CHECK (id = 1), settings TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS drafts (
    owner TEXT PRIMARY KEY, settings TEXT NOT NULL, updated_at REAL NOT NULL, base_id INTEGER);
CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY, changed_at REAL NOT NULL, kind TEXT NOT NULL,
    folder TEXT, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL,
    window_start REAL, window_end REAL, source TEXT, settings_id INTEGER,
    ok INTEGER, message TEXT, durations TEXT, source_detail TEXT, weather_ok INTEGER);
CREATE TABLE IF NOT EXISTS scores (
    model TEXT NOT NULL, site TEXT NOT NULL, variable TEXT NOT NULL, rule TEXT NOT NULL,
    time REAL NOT NULL, score REAL, threshold REAL, base REAL, check_id INTEGER,
    PRIMARY KEY (model, site, variable, rule, time));
CREATE TABLE IF NOT EXISTS noticed (
    id INTEGER PRIMARY KEY, check_id INTEGER NOT NULL, model TEXT NOT NULL,
    site TEXT NOT NULL, variable TEXT NOT NULL, rule TEXT NOT NULL,
    first_over REAL NOT NULL, last_over REAL NOT NULL, peak REAL, base REAL,
    scale REAL, flagged INTEGER NOT NULL, explanations TEXT, note TEXT,
    noted_at REAL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS readings (
    site TEXT NOT NULL, time REAL NOT NULL, conductivity REAL, depth REAL,
    temperature REAL, dissolved_oxygen REAL, floating_conductivity REAL, battery REAL,
    live INTEGER, PRIMARY KEY (site, time));
CREATE TABLE IF NOT EXISTS weather (time REAL PRIMARY KEY, rain_mm REAL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, noticed_id INTEGER NOT NULL, written_at REAL NOT NULL,
    author TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS notes_noticed ON notes (noticed_id);
CREATE TABLE IF NOT EXISTS recipients (
    id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT,
    added_at REAL NOT NULL, label_id INTEGER, unsubscribe_token TEXT, added_by TEXT);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY, sent_at REAL NOT NULL, kind TEXT NOT NULL, check_id INTEGER,
    people INTEGER NOT NULL, ok INTEGER NOT NULL, message TEXT);
CREATE TABLE IF NOT EXISTS email_results (
    email TEXT PRIMARY KEY, sent_at REAL NOT NULL, kind TEXT NOT NULL, ok INTEGER NOT NULL,
    message TEXT);
CREATE TABLE IF NOT EXISTS prefs (`key` TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS noticed_key ON noticed (model, site, variable, rule, first_over);
CREATE INDEX IF NOT EXISTS checks_time ON checks (started_at);
"""

# Added since the first release. Older files get them on open, empty, which
# every query reads as "not known".
ADDED_COLUMNS = (
    ("settings_version", "applied_by", "TEXT"),
    ("source", "detail", "TEXT"),
    ("checks", "source_detail", "TEXT"),
    ("checks", "weather_ok", "INTEGER"),
    ("readings", "live", "INTEGER"),
    ("recipients", "label_id", "INTEGER"),
    ("recipients", "unsubscribe_token", "TEXT"),
    ("recipients", "added_by", "TEXT"),
)

# Recipient labels, website only. They're for telling people apart and don't grant access.
STARTING_LABELS = ("Admin", "EH&S", "SCMG", "Other")
# The public sign-up's label, from before EH&S asked for alerts to stay off the public
OLD_PUBLIC = "Public"

READING_COLUMNS = (
    "conductivity",
    "depth",
    "temperature",
    "dissolved_oxygen",
    "floating_conductivity",
    "battery",
)

# The draft's owner locally, where nobody signs in by name
LOCAL = "local"


class Conflict(RuntimeError):
    """Someone applied a newer version after this draft started from an older one."""

    def __init__(self, applied_at, applied_by=None):
        super().__init__("someone else applied changes first")
        self.applied_at = applied_at
        self.applied_by = applied_by


def data_dir():
    """The per-user folder for the SQLite file, made private to its owner on macOS and Linux."""
    import platformdirs

    folder = Path(platformdirs.user_data_dir("strawberrywatch", appauthor=False))
    folder.mkdir(parents=True, exist_ok=True)
    private(folder)
    return folder


def private(path):
    if sys.platform != "win32":
        os.chmod(path, 0o700 if Path(path).is_dir() else 0o600)


def _in(values):
    return ", ".join("?" for _ in values) or "NULL"


class Store:
    """
    One connection per call, under one lock. Flask serves requests on threads
    and the watcher has its own, and sqlite is happier not sharing a connection.
    """

    IntegrityError = sqlite3.IntegrityError

    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else data_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        private(self.folder)
        self.path = self.folder / DB_NAME
        self._lock = threading.RLock()
        with self.db() as db:
            db.executescript(SCHEMA)
            for table, column, kind in ADDED_COLUMNS:
                have = {r["name"] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
                if column not in have:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            # Old files kept one note on the item itself. Move it to the notes table.
            db.execute(
                "INSERT INTO notes (noticed_id, written_at, text) "
                "SELECT id, COALESCE(noted_at, created_at), note FROM noticed "
                "WHERE note IS NOT NULL AND note != ''"
            )
            db.execute("UPDATE noticed SET note = NULL, noted_at = NULL WHERE note IS NOT NULL")
            # Old files had one shared draft. It becomes the local one, based on
            # what's applied now.
            db.execute(
                "INSERT OR IGNORE INTO drafts (owner, settings, updated_at, base_id) "
                "SELECT ?, settings, updated_at, (SELECT MAX(id) FROM settings_version) "
                "FROM draft WHERE id = 1",
                (LOCAL,),
            )
            db.execute("DELETE FROM draft")
        private(self.path)

    def db(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return _Session(conn, self._lock)

    # Settings

    def applied(self):
        """(version id, settings) of what's applied now, or (None, None) before the first apply."""
        with self.db() as db:
            row = db.execute("SELECT * FROM settings_version ORDER BY id DESC LIMIT 1").fetchone()
        return (row["id"], json.loads(row["settings"])) if row else (None, None)

    def applied_row(self):
        """The newest version with when and by whom, or None."""
        with self.db() as db:
            row = db.execute("SELECT * FROM settings_version ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def apply(self, settings, when=None, by=None, base_id=None, owner=LOCAL):
        """
        Save a new version and clear this owner's draft. With base_id, refuse
        with Conflict when a newer version went in after the draft started.
        """
        with self.db() as db:
            if base_id is not None:
                newest = db.execute(
                    "SELECT * FROM settings_version ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if newest is not None and newest["id"] != base_id:
                    raise Conflict(newest["applied_at"], newest["applied_by"])
            cur = db.execute(
                "INSERT INTO settings_version (applied_at, settings, applied_by) VALUES (?, ?, ?)",
                (when or time.time(), json.dumps(settings, sort_keys=True), by),
            )
            db.execute("DELETE FROM drafts WHERE owner = ?", (owner,))
            return cur.lastrowid

    def version(self, version_id):
        """One version's settings, or None."""
        with self.db() as db:
            row = db.execute(
                "SELECT settings FROM settings_version WHERE id = ?", (version_id,)
            ).fetchone()
        return json.loads(row["settings"]) if row else None

    def versions(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM settings_version ORDER BY id").fetchall()
        return [(r["id"], r["applied_at"], json.loads(r["settings"])) for r in rows]

    def version_authors(self):
        """{version id: who applied it}, for History on the website."""
        with self.db() as db:
            rows = db.execute("SELECT id, applied_by FROM settings_version").fetchall()
        return {r["id"]: r["applied_by"] for r in rows}

    def settings_at(self, when):
        """The settings in force at a moment, or the first ones if it's before them."""
        with self.db() as db:
            row = db.execute(
                "SELECT settings FROM settings_version WHERE applied_at <= ? ORDER BY id DESC LIMIT 1",
                (when,),
            ).fetchone()
            row = (
                row
                or db.execute(
                    "SELECT settings FROM settings_version ORDER BY id LIMIT 1"
                ).fetchone()
            )
        return json.loads(row["settings"]) if row else None

    def draft(self, owner=LOCAL):
        """The owner's saved draft, or None when they have no unapplied changes."""
        with self.db() as db:
            row = db.execute("SELECT settings FROM drafts WHERE owner = ?", (owner,)).fetchone()
        return json.loads(row["settings"]) if row else None

    def draft_base(self, owner=LOCAL):
        """The version the owner's draft started from, or None."""
        with self.db() as db:
            row = db.execute("SELECT base_id FROM drafts WHERE owner = ?", (owner,)).fetchone()
        return row["base_id"] if row else None

    def save_draft(self, settings, owner=LOCAL):
        """Saves the draft. A new draft remembers which version it started from."""
        with self.db() as db:
            row = db.execute("SELECT base_id FROM drafts WHERE owner = ?", (owner,)).fetchone()
            if row is None:
                newest = db.execute("SELECT MAX(id) AS id FROM settings_version").fetchone()
                base = newest["id"] if newest else None
            else:
                base = row["base_id"]
            db.execute(
                "REPLACE INTO drafts (owner, settings, updated_at, base_id) VALUES (?, ?, ?, ?)",
                (owner, json.dumps(settings, sort_keys=True), time.time(), base),
            )

    def discard_draft(self, owner=LOCAL):
        with self.db() as db:
            db.execute("DELETE FROM drafts WHERE owner = ?", (owner,))

    # Where readings come from. The token and passwords aren't kept here.

    def source(self):
        with self.db() as db:
            row = db.execute("SELECT * FROM source ORDER BY id DESC LIMIT 1").fetchone()
        return _source_row(row) if row else None

    def set_source(self, kind, folder=None, event=None, detail=None):
        with self.db() as db:
            db.execute(
                "INSERT INTO source (changed_at, kind, folder, event, detail) VALUES (?, ?, ?, ?, ?)",
                (time.time(), kind, folder, event, json.dumps(detail) if detail else None),
            )

    def source_changes(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM source ORDER BY id").fetchall()
        return [_source_row(r) for r in rows]

    # Checks

    def start_check(
        self, source, settings_id, window_start, window_end, started_at=None, detail=None
    ):
        with self.db() as db:
            cur = db.execute(
                "INSERT INTO checks (started_at, source, settings_id, window_start, window_end, "
                "source_detail) VALUES (?, ?, ?, ?, ?, ?)",
                (started_at or time.time(), source, settings_id, window_start, window_end, detail),
            )
            return cur.lastrowid

    def finish_check(self, check_id, ok, message, durations, weather_ok=None):
        with self.db() as db:
            db.execute(
                "UPDATE checks SET finished_at = ?, ok = ?, message = ?, durations = ?, "
                "weather_ok = ? WHERE id = ?",
                (
                    time.time(),
                    int(ok),
                    message,
                    json.dumps(durations),
                    None if weather_ok is None else int(weather_ok),
                    check_id,
                ),
            )

    def checks(self, since=None, limit=None, kinds=None):
        where, args = [], []
        if since is not None:
            where.append("window_end >= ?")
            args.append(since)
        if kinds is not None:
            where.append(f"source IN ({_in(kinds)})")
            args.extend(kinds)
        sql = "SELECT * FROM checks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id"
        if limit:
            sql += " DESC LIMIT ?"
            args.append(int(limit))
        with self.db() as db:
            rows = [dict(r) for r in db.execute(sql, args).fetchall()]
        for row in rows:
            row["durations"] = json.loads(row["durations"] or "{}")
        return rows

    def last_check(self, kinds=None):
        rows = self.checks(limit=1, kinds=kinds)
        return rows[0] if rows else None

    def last_finished_check(self, kinds=None):
        """The newest check that got as far as finishing, well or not."""
        sql, args = "SELECT * FROM checks WHERE finished_at IS NOT NULL", []
        if kinds is not None:
            sql += f" AND source IN ({_in(kinds)})"
            args.extend(kinds)
        with self.db() as db:
            row = db.execute(sql + " ORDER BY finished_at DESC LIMIT 1", args).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["durations"] = json.loads(out["durations"] or "{}")
        return out

    # Scores, readings and weather the checks saw

    def save_scores(self, check_id, model, rows):
        with self.db() as db:
            db.executemany(
                "REPLACE INTO scores (model, site, variable, rule, time, score, threshold, base, "
                "check_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        model,
                        r.site,
                        r.variable,
                        r.rule,
                        r.time.timestamp(),
                        r.score,
                        r.threshold,
                        r.base,
                        check_id,
                    )
                    for r in rows
                ],
            )

    def scores(self, model=None, since=None, until=None, site=None, variable=None, rule=None):
        where, args = [], []
        for col, value in (
            ("model", model),
            ("site", site),
            ("variable", variable),
            ("rule", rule),
        ):
            if value is not None:
                where.append(f"{col} = ?")
                args.append(value)
        if since is not None:
            where.append("time >= ?")
            args.append(since)
        if until is not None:
            where.append("time <= ?")
            args.append(until)
        sql = "SELECT * FROM scores"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY model, site, variable, rule, time"
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def save_readings(self, rows, live=True):
        """
        rows: (site, time, conductivity, depth, temperature, dissolved_oxygen,
        floating_conductivity, battery). live says whether a live source read
        them. A replay's readings don't show as the creek today.
        """
        with self.db() as db:
            db.executemany(
                "REPLACE INTO readings (site, time, conductivity, depth, temperature, "
                "dissolved_oxygen, floating_conductivity, battery, live) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(*row, int(live)) for row in rows],
            )

    def readings(self, site, since=None, until=None, live=None):
        sql, args = "SELECT * FROM readings WHERE site = ?", [site]
        if since is not None:
            sql += " AND time >= ?"
            args.append(since)
        if until is not None:
            sql += " AND time <= ?"
            args.append(until)
        if live is not None:
            sql += " AND live = ?"
            args.append(int(live))
        with self.db() as db:
            return [dict(r) for r in db.execute(sql + " ORDER BY time", args).fetchall()]

    def last_readings(self, live=None):
        # live=None matches every row through the IS NULL
        live = None if live is None else int(live)
        with self.db() as db:
            rows = db.execute(
                "SELECT r.* FROM readings r JOIN (SELECT site, MAX(time) AS t FROM readings"
                " WHERE ? IS NULL OR live = ? GROUP BY site) m ON r.site = m.site AND r.time = m.t",
                [live, live],
            ).fetchall()
        return {r["site"]: dict(r) for r in rows}

    def save_weather(self, rows):
        with self.db() as db:
            db.executemany("REPLACE INTO weather (time, rain_mm) VALUES (?, ?)", rows)

    def weather(self, since, until):
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM weather WHERE time >= ? AND time <= ? ORDER BY time", (since, until)
            ).fetchall()
        return [(r["time"], r["rain_mm"]) for r in rows]

    # Alerts

    NOTICED_SELECT = (
        "SELECT n.*, c.started_at AS checked_at, c.source AS check_source, "
        "c.source_detail AS check_detail, c.weather_ok AS check_weather_ok "
        "FROM noticed n JOIN checks c ON c.id = n.check_id"
    )

    def add_noticed(self, check_id, item, flagged, explanations, scale):
        with self.db() as db:
            cur = db.execute(
                "INSERT INTO noticed (check_id, model, site, variable, rule, first_over, last_over, "
                "peak, base, scale, flagged, explanations, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    check_id,
                    item["model"],
                    item["site"],
                    item["variable"],
                    item["rule"],
                    item["first_over"],
                    item["last_over"],
                    item["peak"],
                    item["base"],
                    scale,
                    int(flagged),
                    json.dumps(explanations),
                    time.time(),
                ),
            )
            return cur.lastrowid

    def extend_noticed(self, noticed_id, last_over, peak):
        with self.db() as db:
            db.execute(
                "UPDATE noticed SET "
                "last_over = CASE WHEN last_over > ? THEN last_over ELSE ? END, "
                "peak = CASE WHEN peak > ? THEN peak ELSE ? END WHERE id = ?",
                (last_over, last_over, peak, peak, noticed_id),
            )

    def noticed(self, since=None, flagged=None, model=None, kinds=None, detail=None):
        """
        Alerts, newest first. kinds and detail filter by the check's source.
        That keeps live and replay apart.
        """
        where, args = [], []
        if since is not None:
            where.append("n.first_over >= ?")
            args.append(since)
        if flagged is not None:
            where.append("n.flagged = ?")
            args.append(int(flagged))
        if model is not None:
            where.append("n.model = ?")
            args.append(model)
        if kinds is not None:
            where.append(f"c.source IN ({_in(kinds)})")
            args.extend(kinds)
        if detail is not None:
            where.append("c.source_detail = ?")
            args.append(detail)
        sql = self.NOTICED_SELECT
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY n.first_over DESC, n.id DESC"
        with self.db() as db:
            rows = [dict(r) for r in db.execute(sql, args).fetchall()]
        for row in rows:
            row["explanations"] = json.loads(row["explanations"] or "{}")
        return rows

    def noticed_item(self, noticed_id):
        with self.db() as db:
            row = db.execute(self.NOTICED_SELECT + " WHERE n.id = ?", (noticed_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["explanations"] = json.loads(out["explanations"] or "{}")
        return out

    def last_noticed(self, model, site, variable, rule):
        """The newest item for one (model, site, variable, rule), with when its check ran."""
        with self.db() as db:
            row = db.execute(
                self.NOTICED_SELECT + " WHERE n.model = ? AND n.site = ? AND n.variable = ? "
                "AND n.rule = ? ORDER BY n.id DESC LIMIT 1",
                (model, site, variable, rule),
            ).fetchone()
        return dict(row) if row else None

    # Notes

    def add_note(self, noticed_id, text, author=None, when=None):
        with self.db() as db:
            cur = db.execute(
                "INSERT INTO notes (noticed_id, written_at, author, text) VALUES (?, ?, ?, ?)",
                (noticed_id, when or time.time(), author, text),
            )
            return cur.lastrowid

    def delete_note(self, note_id):
        with self.db() as db:
            db.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def note(self, note_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return dict(row) if row else None

    def notes(self, noticed_id=None, ids=None, since=None, until=None, site=None):
        """Notes newest first, with the site of the item each is about."""
        where, args = [], []
        if noticed_id is not None:
            where.append("t.noticed_id = ?")
            args.append(noticed_id)
        if ids is not None:
            where.append(f"t.id IN ({_in(ids)})")
            args.extend(ids)
        if since is not None:
            where.append("t.written_at >= ?")
            args.append(since)
        if until is not None:
            where.append("t.written_at < ?")
            args.append(until)
        if site is not None:
            where.append("n.site = ?")
            args.append(site)
        sql = "SELECT t.*, n.site FROM notes t JOIN noticed n ON n.id = t.noticed_id"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY t.written_at DESC, t.id DESC"
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    # Email: the website's list (the local one is in .env), the on/off switch,
    # and what was sent

    def recipients(self):
        with self.db() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM recipients ORDER BY added_at, id")]
            # Rows from before unsubscribe links get their token the first time they're read
            for row in rows:
                if not row.get("unsubscribe_token"):
                    row["unsubscribe_token"] = secrets.token_urlsafe(24)
                    db.execute(
                        "UPDATE recipients SET unsubscribe_token = ? WHERE id = ?",
                        (row["unsubscribe_token"], row["id"]),
                    )
        return rows

    def add_recipient(self, email, name=None, label_id=None, added_by=None):
        """Returns False if the address is already on the list, ignoring case."""
        try:
            with self.db() as db:
                db.execute(
                    "INSERT INTO recipients "
                    "(email, name, added_at, label_id, unsubscribe_token, added_by) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        email,
                        name or None,
                        time.time(),
                        label_id,
                        secrets.token_urlsafe(24),
                        added_by,
                    ),
                )
        except self.IntegrityError:
            return False
        return True

    def recipient_by_token(self, token):
        if not token:
            return None
        with self.db() as db:
            row = db.execute(
                "SELECT * FROM recipients WHERE unsubscribe_token = ?", (token,)
            ).fetchone()
        return dict(row) if row else None

    def set_label(self, recipient_id, label_id):
        with self.db() as db:
            db.execute("UPDATE recipients SET label_id = ? WHERE id = ?", (label_id, recipient_id))

    def labels(self):
        """Every label, oldest first. The starting four are made the first time anyone asks."""
        with self.db() as db:
            rows = db.execute("SELECT id, name FROM labels ORDER BY id").fetchall()
        if rows:
            names = {r["name"].lower(): r["id"] for r in rows}
            if OLD_PUBLIC.lower() in names:
                self._retire_public(names)
                with self.db() as db:
                    rows = db.execute("SELECT id, name FROM labels ORDER BY id").fetchall()
            return [dict(r) for r in rows]
        try:
            with self.db() as db:
                for name in STARTING_LABELS:
                    db.execute(
                        "INSERT INTO labels (name, added_at) VALUES (?, ?)", (name, time.time())
                    )
        except self.IntegrityError:
            pass  # another process made them first
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id, name FROM labels ORDER BY id")]

    def label_named(self, name):
        return next((lb for lb in self.labels() if lb["name"].lower() == name.lower()), None)

    def add_label(self, name):
        """Returns False if there's already a label by that name, ignoring case."""
        self.labels()
        try:
            with self.db() as db:
                db.execute("INSERT INTO labels (name, added_at) VALUES (?, ?)", (name, time.time()))
        except self.IntegrityError:
            return False
        return True

    def _retire_public(self, names):
        """Public becomes Other, or its people join Other when there already is one."""
        public = names[OLD_PUBLIC.lower()]
        with self.db() as db:
            if "other" in names:
                db.execute(
                    "UPDATE recipients SET label_id = ? WHERE label_id = ?",
                    (names["other"], public),
                )
                db.execute("DELETE FROM labels WHERE id = ?", (public,))
            else:
                db.execute("UPDATE labels SET name = ? WHERE id = ?", ("Other", public))

    def rename_label(self, label_id, name):
        """Returns False if there's no such label, or another one already has that name."""
        names = {lb["id"]: lb["name"] for lb in self.labels()}
        if label_id not in names:
            return False
        try:
            with self.db() as db:
                db.execute("UPDATE labels SET name = ? WHERE id = ?", (name, label_id))
        except self.IntegrityError:
            return False
        return True

    def recipient(self, recipient_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM recipients WHERE id = ?", (recipient_id,)).fetchone()
        return dict(row) if row else None

    def remove_recipient(self, recipient_id):
        with self.db() as db:
            db.execute("DELETE FROM recipients WHERE id = ?", (recipient_id,))

    def clear_recipients(self):
        with self.db() as db:
            db.execute("DELETE FROM recipients")

    def pref(self, key, default=None):
        with self.db() as db:
            row = db.execute("SELECT value FROM prefs WHERE `key` = ?", (key,)).fetchone()
        return default if row is None else row["value"]

    def set_pref(self, key, value):
        with self.db() as db:
            db.execute("REPLACE INTO prefs (`key`, value) VALUES (?, ?)", (key, value))

    def log_email(self, kind, people, ok, message, check_id=None):
        with self.db() as db:
            db.execute(
                "INSERT INTO emails (sent_at, kind, check_id, people, ok, message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), kind, check_id, people, int(ok), message),
            )

    def emails(self, limit=None):
        sql = "SELECT * FROM emails ORDER BY sent_at DESC, id DESC"
        args = []
        if limit:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self.db() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def record_email_result(self, email, kind, ok, message):
        """How the last email to one address went, shown beside it on Email alerts."""
        with self.db() as db:
            db.execute(
                "REPLACE INTO email_results (email, sent_at, kind, ok, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (email.lower(), time.time(), kind, int(ok), message),
            )

    def email_results(self):
        """{address in lower case: its last result}."""
        with self.db() as db:
            rows = db.execute("SELECT * FROM email_results").fetchall()
        return {r["email"]: dict(r) for r in rows}

    def prune(self, now=None):
        now = now or time.time()
        with self.db() as db:
            db.execute("DELETE FROM scores WHERE time < ?", (now - SCORE_DAYS * 86400,))
            db.execute("DELETE FROM readings WHERE time < ?", (now - READING_DAYS * 86400,))
            db.execute("DELETE FROM weather WHERE time < ?", (now - SCORE_DAYS * 86400,))

    def prune_daily(self, now=None):
        """Prune at most once a day, whatever the check interval."""
        now = now or time.time()
        last = self.pref("pruned_at")
        if last is not None and now - float(last) < PRUNE_EVERY:
            return False
        self.prune(now)
        self.set_pref("pruned_at", repr(now))
        return True


def _source_row(row):
    out = dict(row)
    out["detail"] = json.loads(out["detail"]) if out.get("detail") else {}
    return out


class _Session:
    """A connection that commits on the way out, under the store's lock."""

    def __init__(self, conn, lock):
        self.conn, self.lock = conn, lock

    def __enter__(self):
        self.lock.acquire()
        return self.conn

    def __exit__(self, kind, *_):
        try:
            if kind is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
            self.lock.release()

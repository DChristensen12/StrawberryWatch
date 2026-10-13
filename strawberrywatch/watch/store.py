"""
The one SQLite file: settings versions and the draft, where readings come from,
every check, the scores and readings it saw, and what it noticed.

Every query is parameterized. The API token is never in here; it lives in the
system keychain (see sources.py).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

DB_NAME = "strawberrywatch.db"
SCORE_DAYS = 30
READING_DAYS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings_version (
    id INTEGER PRIMARY KEY, applied_at REAL NOT NULL, settings TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS draft (
    id INTEGER PRIMARY KEY CHECK (id = 1), settings TEXT NOT NULL, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS source (
    id INTEGER PRIMARY KEY, changed_at REAL NOT NULL, kind TEXT NOT NULL,
    folder TEXT, event TEXT);
CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY, started_at REAL NOT NULL, finished_at REAL,
    window_start REAL, window_end REAL, source TEXT, settings_id INTEGER,
    ok INTEGER, message TEXT, durations TEXT);
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
    PRIMARY KEY (site, time));
CREATE TABLE IF NOT EXISTS weather (time REAL PRIMARY KEY, rain_mm REAL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, noticed_id INTEGER NOT NULL, written_at REAL NOT NULL,
    author TEXT, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS notes_noticed ON notes (noticed_id);
CREATE TABLE IF NOT EXISTS recipients (
    id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE, name TEXT,
    added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY, sent_at REAL NOT NULL, kind TEXT NOT NULL, check_id INTEGER,
    people INTEGER NOT NULL, ok INTEGER NOT NULL, message TEXT);
CREATE TABLE IF NOT EXISTS prefs (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS noticed_key ON noticed (model, site, variable, rule, first_over);
CREATE INDEX IF NOT EXISTS checks_time ON checks (started_at);
"""

READING_COLUMNS = (
    "conductivity",
    "depth",
    "temperature",
    "dissolved_oxygen",
    "floating_conductivity",
    "battery",
)


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


class Store:
    """
    One connection per call, under one lock. Flask serves requests on threads
    and the watcher has its own, and sqlite is happier not sharing a connection.
    """

    def __init__(self, folder=None):
        self.folder = Path(folder) if folder else data_dir()
        self.folder.mkdir(parents=True, exist_ok=True)
        private(self.folder)
        self.path = self.folder / DB_NAME
        self._lock = threading.RLock()
        with self.db() as db:
            db.executescript(SCHEMA)
            # Notes used to be one per item, kept on the item itself. Each
            # becomes the first note in the notes table.
            db.execute(
                "INSERT INTO notes (noticed_id, written_at, text) "
                "SELECT id, COALESCE(noted_at, created_at), note FROM noticed "
                "WHERE note IS NOT NULL AND note != ''"
            )
            db.execute("UPDATE noticed SET note = NULL, noted_at = NULL WHERE note IS NOT NULL")
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

    def apply(self, settings, when=None):
        """Save a new version and clear the draft."""
        with self.db() as db:
            cur = db.execute(
                "INSERT INTO settings_version (applied_at, settings) VALUES (?, ?)",
                (when or time.time(), json.dumps(settings, sort_keys=True)),
            )
            db.execute("DELETE FROM draft")
            return cur.lastrowid

    def versions(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM settings_version ORDER BY id").fetchall()
        return [(r["id"], r["applied_at"], json.loads(r["settings"])) for r in rows]

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

    def draft(self):
        """The saved draft, or None when there's nothing unapplied."""
        with self.db() as db:
            row = db.execute("SELECT settings FROM draft WHERE id = 1").fetchone()
        return json.loads(row["settings"]) if row else None

    def save_draft(self, settings):
        with self.db() as db:
            db.execute(
                "INSERT OR REPLACE INTO draft (id, settings, updated_at) VALUES (1, ?, ?)",
                (json.dumps(settings, sort_keys=True), time.time()),
            )

    def discard_draft(self):
        with self.db() as db:
            db.execute("DELETE FROM draft")

    # Where readings come from. The token is never here.

    def source(self):
        with self.db() as db:
            row = db.execute("SELECT * FROM source ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def set_source(self, kind, folder=None, event=None):
        with self.db() as db:
            db.execute(
                "INSERT INTO source (changed_at, kind, folder, event) VALUES (?, ?, ?, ?)",
                (time.time(), kind, folder, event),
            )

    def source_changes(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM source ORDER BY id").fetchall()]

    # Checks

    def start_check(self, source, settings_id, window_start, window_end, started_at=None):
        with self.db() as db:
            cur = db.execute(
                "INSERT INTO checks (started_at, source, settings_id, window_start, window_end) "
                "VALUES (?, ?, ?, ?, ?)",
                (started_at or time.time(), source, settings_id, window_start, window_end),
            )
            return cur.lastrowid

    def finish_check(self, check_id, ok, message, durations):
        with self.db() as db:
            db.execute(
                "UPDATE checks SET finished_at = ?, ok = ?, message = ?, durations = ? WHERE id = ?",
                (time.time(), int(ok), message, json.dumps(durations), check_id),
            )

    def checks(self, since=None, limit=None):
        sql = "SELECT * FROM checks"
        args = []
        if since is not None:
            sql += " WHERE window_end >= ?"
            args.append(since)
        sql += " ORDER BY id"
        if limit:
            sql += " DESC LIMIT ?"
            args.append(int(limit))
        with self.db() as db:
            rows = [dict(r) for r in db.execute(sql, args).fetchall()]
        for row in rows:
            row["durations"] = json.loads(row["durations"] or "{}")
        return rows

    def last_check(self):
        rows = self.checks(limit=1)
        return rows[0] if rows else None

    # Scores, readings and weather the checks saw

    def save_scores(self, check_id, model, rows):
        with self.db() as db:
            db.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def save_readings(self, rows):
        """rows: (site, time, conductivity, depth, temperature, dissolved_oxygen, floating_conductivity, battery)."""
        with self.db() as db:
            db.executemany("INSERT OR REPLACE INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def readings(self, site, since=None, until=None):
        sql, args = "SELECT * FROM readings WHERE site = ?", [site]
        if since is not None:
            sql += " AND time >= ?"
            args.append(since)
        if until is not None:
            sql += " AND time <= ?"
            args.append(until)
        with self.db() as db:
            return [dict(r) for r in db.execute(sql + " ORDER BY time", args).fetchall()]

    def last_readings(self):
        with self.db() as db:
            rows = db.execute(
                "SELECT r.* FROM readings r JOIN (SELECT site, MAX(time) AS t FROM readings GROUP BY site) m "
                "ON r.site = m.site AND r.time = m.t"
            ).fetchall()
        return {r["site"]: dict(r) for r in rows}

    def save_weather(self, rows):
        with self.db() as db:
            db.executemany("INSERT OR REPLACE INTO weather VALUES (?, ?)", rows)

    def weather(self, since, until):
        with self.db() as db:
            rows = db.execute(
                "SELECT * FROM weather WHERE time >= ? AND time <= ? ORDER BY time", (since, until)
            ).fetchall()
        return [(r["time"], r["rain_mm"]) for r in rows]

    # What was noticed

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
                "UPDATE noticed SET last_over = MAX(last_over, ?), peak = MAX(peak, ?) WHERE id = ?",
                (last_over, peak, noticed_id),
            )

    def noticed(self, since=None, flagged=None, model=None):
        where, args = [], []
        if since is not None:
            where.append("first_over >= ?")
            args.append(since)
        if flagged is not None:
            where.append("flagged = ?")
            args.append(int(flagged))
        if model is not None:
            where.append("model = ?")
            args.append(model)
        sql = "SELECT n.*, c.started_at AS checked_at FROM noticed n JOIN checks c ON c.id = n.check_id"
        if where:
            sql += " WHERE " + " AND ".join(f"n.{w}" for w in where)
        sql += " ORDER BY n.first_over DESC, n.id DESC"
        with self.db() as db:
            rows = [dict(r) for r in db.execute(sql, args).fetchall()]
        for row in rows:
            row["explanations"] = json.loads(row["explanations"] or "{}")
        return rows

    def noticed_item(self, noticed_id):
        with self.db() as db:
            row = db.execute(
                "SELECT n.*, c.started_at AS checked_at FROM noticed n JOIN checks c ON c.id = n.check_id "
                "WHERE n.id = ?",
                (noticed_id,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["explanations"] = json.loads(out["explanations"] or "{}")
        return out

    def last_noticed(self, model, site, variable, rule):
        """The newest item for one (model, site, variable, rule), with when its check ran."""
        with self.db() as db:
            row = db.execute(
                "SELECT n.*, c.started_at AS checked_at FROM noticed n JOIN checks c ON c.id = n.check_id "
                "WHERE n.model = ? AND n.site = ? AND n.variable = ? AND n.rule = ? "
                "ORDER BY n.id DESC LIMIT 1",
                (model, site, variable, rule),
            ).fetchone()
        return dict(row) if row else None

    # Notes: each one is kept with when it was written, and on the website by whom

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
            where.append(f"t.id IN ({', '.join('?' for _ in ids) or 'NULL'})")
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

    # Who gets emailed, whether sending is on, and what was sent

    def recipients(self):
        with self.db() as db:
            rows = db.execute("SELECT * FROM recipients ORDER BY added_at, id").fetchall()
        return [dict(r) for r in rows]

    def add_recipient(self, email, name=None):
        """False if the address is already on the list, in any capitalization."""
        try:
            with self.db() as db:
                db.execute(
                    "INSERT INTO recipients (email, name, added_at) VALUES (?, ?, ?)",
                    (email, name or None, time.time()),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def recipient(self, recipient_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM recipients WHERE id = ?", (recipient_id,)).fetchone()
        return dict(row) if row else None

    def remove_recipient(self, recipient_id):
        with self.db() as db:
            db.execute("DELETE FROM recipients WHERE id = ?", (recipient_id,))

    def pref(self, key, default=None):
        with self.db() as db:
            row = db.execute("SELECT value FROM prefs WHERE key = ?", (key,)).fetchone()
        return default if row is None else row["value"]

    def set_pref(self, key, value):
        with self.db() as db:
            db.execute(
                "INSERT INTO prefs (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

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

    def prune(self, now=None):
        now = now or time.time()
        with self.db() as db:
            db.execute("DELETE FROM scores WHERE time < ?", (now - SCORE_DAYS * 86400,))
            db.execute("DELETE FROM readings WHERE time < ?", (now - READING_DAYS * 86400,))
            db.execute("DELETE FROM weather WHERE time < ?", (now - SCORE_DAYS * 86400,))


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

"""
Store on the website's MySQL database. It runs Store's own queries, adding the
sw_ prefix to every table and swapping ? placeholders for %s. It only touches
our own tables. Connects with Night Heron's MYSQL settings.
"""

from __future__ import annotations

import re
import secrets
import threading

from strawberrywatch.watch.store import Store

PREFIX = "sw_"
TABLES = (
    "settings_version",
    "drafts",
    "source",
    "checks",
    "scores",
    "noticed",
    "readings",
    "weather",
    "notes",
    "recipients",
    "labels",
    "emails",
    "email_results",
    "prefs",
)
# A name right after one of these is a table, not a column with the same name.
# ON is only here for indexes, since joins always put an alias after ON.
_TABLE_AFTER = re.compile(
    rf"\b(FROM|JOIN|INTO|UPDATE|TABLE|EXISTS|ON)\s+({'|'.join(TABLES)})\b", re.IGNORECASE
)

# The MySQL spelling of store.SCHEMA. Text keys get a length, since MySQL can't
# index an unbounded TEXT column.
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS settings_version (
        id INT AUTO_INCREMENT PRIMARY KEY, applied_at DOUBLE NOT NULL,
        settings MEDIUMTEXT NOT NULL, applied_by VARCHAR(150))""",
    """CREATE TABLE IF NOT EXISTS drafts (
        owner VARCHAR(150) PRIMARY KEY, settings MEDIUMTEXT NOT NULL,
        updated_at DOUBLE NOT NULL, base_id INT)""",
    """CREATE TABLE IF NOT EXISTS source (
        id INT AUTO_INCREMENT PRIMARY KEY, changed_at DOUBLE NOT NULL, kind VARCHAR(16) NOT NULL,
        folder TEXT, event VARCHAR(128), detail TEXT)""",
    """CREATE TABLE IF NOT EXISTS checks (
        id INT AUTO_INCREMENT PRIMARY KEY, started_at DOUBLE NOT NULL, finished_at DOUBLE,
        window_start DOUBLE, window_end DOUBLE, source VARCHAR(16), settings_id INT,
        ok TINYINT, message TEXT, durations TEXT, source_detail VARCHAR(255), weather_ok TINYINT,
        INDEX checks_time (started_at))""",
    """CREATE TABLE IF NOT EXISTS scores (
        model VARCHAR(64) NOT NULL, site VARCHAR(64) NOT NULL, variable VARCHAR(64) NOT NULL,
        rule VARCHAR(64) NOT NULL, time DOUBLE NOT NULL, score DOUBLE, threshold DOUBLE,
        base DOUBLE, check_id INT, PRIMARY KEY (model, site, variable, rule, time))""",
    """CREATE TABLE IF NOT EXISTS noticed (
        id INT AUTO_INCREMENT PRIMARY KEY, check_id INT NOT NULL, model VARCHAR(64) NOT NULL,
        site VARCHAR(64) NOT NULL, variable VARCHAR(64) NOT NULL, rule VARCHAR(64) NOT NULL,
        first_over DOUBLE NOT NULL, last_over DOUBLE NOT NULL, peak DOUBLE, base DOUBLE,
        scale DOUBLE, flagged TINYINT NOT NULL, explanations MEDIUMTEXT, note TEXT,
        noted_at DOUBLE, created_at DOUBLE NOT NULL,
        INDEX noticed_key (model, site, variable, rule, first_over))""",
    """CREATE TABLE IF NOT EXISTS readings (
        site VARCHAR(64) NOT NULL, time DOUBLE NOT NULL, conductivity DOUBLE, depth DOUBLE,
        temperature DOUBLE, dissolved_oxygen DOUBLE, floating_conductivity DOUBLE,
        battery DOUBLE, live TINYINT, PRIMARY KEY (site, time))""",
    """CREATE TABLE IF NOT EXISTS weather (time DOUBLE PRIMARY KEY, rain_mm DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS notes (
        id INT AUTO_INCREMENT PRIMARY KEY, noticed_id INT NOT NULL, written_at DOUBLE NOT NULL,
        author VARCHAR(150), text TEXT NOT NULL, INDEX notes_noticed (noticed_id))""",
    """CREATE TABLE IF NOT EXISTS recipients (
        id INT AUTO_INCREMENT PRIMARY KEY, email VARCHAR(254) NOT NULL UNIQUE,
        name VARCHAR(100), added_at DOUBLE NOT NULL, label_id INT,
        unsubscribe_token VARCHAR(64), added_by VARCHAR(150))""",
    """CREATE TABLE IF NOT EXISTS labels (
        id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(50) NOT NULL UNIQUE,
        added_at DOUBLE NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS emails (
        id INT AUTO_INCREMENT PRIMARY KEY, sent_at DOUBLE NOT NULL, kind VARCHAR(16) NOT NULL,
        check_id INT, people INT NOT NULL, ok TINYINT NOT NULL, message TEXT)""",
    """CREATE TABLE IF NOT EXISTS email_results (
        email VARCHAR(254) PRIMARY KEY, sent_at DOUBLE NOT NULL, kind VARCHAR(16) NOT NULL,
        ok TINYINT NOT NULL, message TEXT)""",
    """CREATE TABLE IF NOT EXISTS prefs (`key` VARCHAR(100) PRIMARY KEY, value TEXT)""",
)


# Columns added after the first install. setup_tables adds any that are missing.
ADDED_COLUMNS = (
    ("recipients", "label_id", "INT"),
    ("recipients", "unsubscribe_token", "VARCHAR(64)"),
    ("recipients", "added_by", "VARCHAR(150)"),
)


def prefixed(sql, placeholder="%s"):
    """Adds the sw_ prefix to every table in a query and swaps the placeholders."""
    sql = _TABLE_AFTER.sub(lambda m: f"{m.group(1)} {PREFIX}{m.group(2)}", sql)
    return sql.replace("?", placeholder) if placeholder != "?" else sql


class MysqlStore(Store):
    """Store on MySQL. Tables aren't created on open, setup_tables does that."""

    def __init__(self, credentials=None):
        import mysql.connector

        self.IntegrityError = mysql.connector.IntegrityError
        self._credentials = credentials
        self._lock = threading.RLock()
        # What runner.read_settings keys its fallback on. Not a password
        self.path = "mysql:sw"
        self.folder = None

    def db(self):
        from strawberrywatch.ingest import sql_client

        return _MysqlSession(sql_client._connect(self._credentials))

    def create_tables(self):
        with self.db() as db:
            for statement in SCHEMA:
                db.execute(statement + " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4")
            for table, column, kind in ADDED_COLUMNS:
                have = {r["Field"] for r in db.execute(f"SHOW COLUMNS FROM {table}").fetchall()}
                if column not in have:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    def session_key(self):
        """The dashboard's cookie signing key, made once and kept in sw_prefs."""
        key = self.pref("session_signing_key")
        if not key:
            key = secrets.token_hex(32)
            self.set_pref("session_signing_key", key)
        return key


class _MysqlSession:
    """A connection that rewrites each query, commits on the way out, and always closes."""

    def __init__(self, connect):
        self._connect = connect
        self.conn = None

    def __enter__(self):
        self.conn = self._connect.__enter__()
        return self

    def execute(self, sql, args=()):
        cursor = self.conn.cursor(dictionary=True, buffered=True)
        cursor.execute(prefixed(sql), tuple(args))
        return cursor

    def executemany(self, sql, rows):
        cursor = self.conn.cursor()
        cursor.executemany(prefixed(sql), [tuple(r) for r in rows])
        return cursor

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self._connect.__exit__(kind, value, traceback)

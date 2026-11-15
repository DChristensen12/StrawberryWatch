"""The data folder, its SQLite file, the keychain and the saved calibrations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

KEYCHAIN = "StrawberryWatch Workbench"
SECRETS = ("api_token", "mysql_password")
SHIPPED = "shipped"

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS readings (source TEXT, site TEXT, reading TEXT, time_utc TEXT,
        value REAL, PRIMARY KEY (source, site, reading, time_utc))""",
    """CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, shipped_id TEXT, name TEXT,
        kind TEXT, sites TEXT, start_utc TEXT, end_utc TEXT, should_alert INTEGER, note TEXT,
        deleted INTEGER DEFAULT 0, updated_utc TEXT)""",
    """CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, source TEXT, sites TEXT,
        readings TEXT, start_date TEXT, end_date TEXT, leave_out_events INTEGER,
        leave_out_rain INTEGER, loaded INTEGER, explored INTEGER, model TEXT,
        false_alarms TEXT, notes TEXT, check_json TEXT, step INTEGER, finished INTEGER,
        updated_utc TEXT)""",
    """CREATE TABLE IF NOT EXISTS drift (model TEXT, calibration TEXT, computed_utc TEXT,
        result_json TEXT)""",
    "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)",
)


def default_folder():
    import platformdirs

    return Path(platformdirs.user_data_dir("StrawberryWatch")) / "workbench"


def ready(folder):
    folder = Path(folder)
    (folder / "calibrations").mkdir(parents=True, exist_ok=True)
    with connect(folder) as db:
        for statement in SCHEMA:
            db.execute(statement)
    return folder


def connect(folder):
    db = sqlite3.connect(Path(folder) / "workbench.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    return db


def now_utc():
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


# Settings and secrets


def setting(folder, key, default=None):
    with connect(folder) as db:
        row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(folder, key, value):
    with connect(folder) as db:
        db.execute("REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))


def secret(key):
    try:
        return keyring.get_password(KEYCHAIN, key)
    except KeyringError:
        return None


def set_secret(key, value):
    keyring.set_password(KEYCHAIN, key, value)


def forget_secret(key):
    try:
        keyring.delete_password(KEYCHAIN, key)
    except (PasswordDeleteError, KeyringError):
        pass


# Sessions

SESSION_FIELDS = (
    "source",
    "sites",
    "readings",
    "start_date",
    "end_date",
    "leave_out_events",
    "leave_out_rain",
    "loaded",
    "explored",
    "model",
    "false_alarms",
    "notes",
    "check_json",
    "step",
    "finished",
)
# Stored as JSON lists
LIST_FIELDS = ("sites", "readings")


def _session(row):
    if row is None:
        return None
    out = dict(row)
    for key in LIST_FIELDS:
        out[key] = json.loads(out[key] or "[]")
    out["check"] = json.loads(out["check_json"]) if out.get("check_json") else None
    return out


def new_session(folder, **fields):
    fields = {"loaded": 0, "explored": 0, "step": 1, "finished": 0, **fields}
    for key in LIST_FIELDS:
        if key in fields:
            fields[key] = json.dumps(list(fields[key]))
    names = [k for k in fields if k in SESSION_FIELDS]
    with connect(folder) as db:
        cursor = db.execute(
            f"INSERT INTO sessions ({', '.join(names)}, updated_utc) "
            f"VALUES ({', '.join('?' for _ in names)}, ?)",
            [fields[k] for k in names] + [now_utc()],
        )
        return cursor.lastrowid


def session(folder, session_id):
    with connect(folder) as db:
        row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return _session(row)


def update_session(folder, session_id, **fields):
    for key in LIST_FIELDS:
        if key in fields:
            fields[key] = json.dumps(list(fields[key]))
    if "check" in fields:
        check = fields.pop("check")
        fields["check_json"] = None if check is None else json.dumps(check)
    names = [k for k in fields if k in SESSION_FIELDS]
    if not names:
        return
    with connect(folder) as db:
        db.execute(
            f"UPDATE sessions SET {', '.join(f'{k} = ?' for k in names)}, updated_utc = ? "
            "WHERE id = ?",
            [fields[k] for k in names] + [now_utc(), session_id],
        )


def unfinished_session(folder):
    with connect(folder) as db:
        row = db.execute(
            "SELECT * FROM sessions WHERE finished = 0 ORDER BY updated_utc DESC, id DESC LIMIT 1"
        ).fetchone()
    return _session(row)


# Readings


def clear_readings(folder):
    with connect(folder) as db:
        db.execute("DELETE FROM readings")


# Drift results


def drift_result(folder, model, calibration):
    with connect(folder) as db:
        row = db.execute(
            "SELECT * FROM drift WHERE model = ? AND calibration = ? "
            "ORDER BY computed_utc DESC LIMIT 1",
            (model, calibration),
        ).fetchone()
    if row is None:
        return None
    return {"computed_utc": row["computed_utc"], **json.loads(row["result_json"])}


def save_drift(folder, model, calibration, result):
    with connect(folder) as db:
        db.execute(
            "INSERT INTO drift (model, calibration, computed_utc, result_json) VALUES (?, ?, ?, ?)",
            (model, calibration, now_utc(), json.dumps(result)),
        )


# Calibrations


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_dir(folder, model, calibration_id):
    return Path(folder) / "calibrations" / model / calibration_id


def saved_calibrations(folder, model):
    """Saved ones for a model, newest first, as (id, record)."""
    root = Path(folder) / "calibrations" / model
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir(), reverse=True):
        record = path / "record.json"
        if path.name != SHIPPED and record.is_file():
            out.append((path.name, json.loads(record.read_text())))
    return out


def new_calibration_id():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def save_calibration(folder, model, filename, blob, record, reference=None):
    """Writes the calibration folder and returns its id. The record holds the file's SHA-256."""
    calibration_id = new_calibration_id()
    target = calibration_dir(folder, model, calibration_id)
    while target.exists():
        time.sleep(1)
        calibration_id = new_calibration_id()
        target = calibration_dir(folder, model, calibration_id)
    temp = target.with_name(f".{calibration_id}.tmp")
    temp.mkdir(parents=True)
    path = temp / filename
    path.write_text(json.dumps(blob, indent=1, sort_keys=True))
    record = {**record, "file": filename, "sha256": sha256(path), "saved_utc": now_utc()}
    (temp / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    if reference is not None:
        reference.to_csv(temp / "reference.csv.gz", index=False)
    temp.rename(target)
    return calibration_id


def calibration_record(folder, model, calibration_id):
    path = calibration_dir(folder, model, calibration_id) / "record.json"
    return json.loads(path.read_text()) if path.is_file() else None


def verified(folder, model, calibration_id):
    """The saved calibration's file if its SHA-256 still matches its record, else None."""
    record = calibration_record(folder, model, calibration_id)
    if record is None:
        return None
    path = calibration_dir(folder, model, calibration_id) / record["file"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        return None
    return path


def in_use(folder):
    """{model: "shipped" or a saved id}. An id whose file fails its SHA-256 counts as shipped."""
    path = Path(folder) / "in_use.json"
    chosen = json.loads(path.read_text()) if path.is_file() else {}
    out = {}
    for model, calibration_id in chosen.items():
        if calibration_id != SHIPPED and verified(folder, model, calibration_id) is None:
            calibration_id = SHIPPED
        out[model] = calibration_id
    return out


def use(folder, model, calibration_id):
    path = Path(folder) / "in_use.json"
    chosen = json.loads(path.read_text()) if path.is_file() else {}
    chosen[model] = calibration_id
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(chosen, indent=1, sort_keys=True))
    temp.replace(path)

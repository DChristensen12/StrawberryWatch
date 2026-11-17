"""Shipped events from watch/events plus the ones added or changed here. Shipped files are only read."""

from __future__ import annotations

import functools

import pandas as pd

from strawberrywatch.watch import sources
from strawberrywatch.workbench import store

# Kind and whether a model should alert, per shipped folder. Sites come from its
# labels, or from every CSV in it when it has none, unless given here
SHIPPED = {
    "anomaly_2025_06_12_spill_sf": ("spill", True, None),
    "anomaly_2025_08_sprinklers_nf0": ("spill", True, None),
    "anomaly_2025_09_10_overnight_sf": ("spill", True, None),
    "anomaly_2025_11_05_foam_nf1": ("spill", True, None),
    "anomaly_2025_11_13_rain_nf1": ("rain", False, None),
    "anomaly_2026_01_botanical_actuator": ("other", False, ("botanical_garden",)),
    "anomaly_2026_03_20_hydrant_nf0": ("spill", True, None),
    "anomaly_2026_04_01_rainfall": ("rain", False, None),
}
# File stems that aren't site ids
FILE_SITES = {"footbridge": "scnf010"}


def _ts(value):
    return pd.Timestamp(value, tz="UTC") if value else None


@functools.cache
def shipped():
    out = []
    for entry in sources.replay_events():
        folder = entry["folder"]
        kind, should_alert, sites = SHIPPED.get(folder, ("other", False, None))
        files = sorted((sources.EVENTS_DIR / folder).glob("*.csv"))
        stamps = []
        for path in files:
            times = pd.read_csv(path, usecols=lambda c: c in sources.TIME_COLUMNS)
            stamps.append(pd.to_datetime(times.iloc[:, 0], utc=True))
        everywhere = [FILE_SITES.get(p.stem, p.stem) for p in files]
        out.append(
            {
                "id": folder,
                "shipped_id": folder,
                "name": entry["title"],
                "kind": kind,
                "sites": list(sites or entry.get("labels") or everywhere),
                "start": min(s.min() for s in stamps),
                "end": max(s.max() for s in stamps),
                "should_alert": should_alert,
                "note": "",
                "shipped": True,
            }
        )
    return tuple(out)


def _local(row):
    return {
        "id": row["shipped_id"] or str(row["id"]),
        "row": row["id"],
        "shipped_id": row["shipped_id"],
        "name": row["name"],
        "kind": row["kind"],
        "sites": [s for s in (row["sites"] or "").split(",") if s],
        "start": _ts(row["start_utc"]),
        "end": _ts(row["end_utc"]),
        "should_alert": bool(row["should_alert"]),
        "note": row["note"] or "",
        "shipped": bool(row["shipped_id"]),
        "deleted": bool(row["deleted"]),
    }


def every(folder):
    """Shipped events with this computer's changes on top, then the added ones, oldest first."""
    with store.connect(folder) as db:
        rows = [_local(r) for r in db.execute("SELECT * FROM events ORDER BY id").fetchall()]
    overrides = {r["shipped_id"]: r for r in rows if r["shipped_id"]}
    out = []
    for event in shipped():
        mine = overrides.get(event["shipped_id"])
        out.append({**event, **{k: v for k, v in (mine or {}).items() if k != "id"}})
    out += [r for r in rows if not r["shipped_id"] and not r["deleted"]]
    return sorted(out, key=lambda e: e["start"])


def one(folder, event_id):
    return next((e for e in every(folder) if e["id"] == event_id), None)


def save(folder, event_id, fields):
    """Saves an edit, or a new event when event_id is None. Returns the event's id."""
    values = (
        fields["name"],
        fields["kind"],
        ",".join(fields["sites"]),
        fields["start"].strftime("%Y-%m-%d %H:%M:%S"),
        fields["end"].strftime("%Y-%m-%d %H:%M:%S"),
        int(bool(fields["should_alert"])),
        fields.get("note", ""),
        store.now_utc(),
    )
    with store.connect(folder) as db:
        if event_id is None:
            cursor = db.execute(
                "INSERT INTO events (name, kind, sites, start_utc, end_utc, should_alert, note, "
                "updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            return str(cursor.lastrowid)
        current = one(folder, event_id)
        if current is None:
            return None
        if current["shipped"]:
            row = db.execute("SELECT id FROM events WHERE shipped_id = ?", (event_id,)).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO events (shipped_id, name, kind, sites, start_utc, end_utc, "
                    "should_alert, note, updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (event_id, *values),
                )
                return event_id
            row_id = row["id"]
        else:
            row_id = int(event_id)
        db.execute(
            "UPDATE events SET name = ?, kind = ?, sites = ?, start_utc = ?, end_utc = ?, "
            "should_alert = ?, note = ?, updated_utc = ? WHERE id = ?",
            (*values, row_id),
        )
    return event_id


def delete(folder, event_id):
    """Only events added here can go. Returns whether one did."""
    event = one(folder, event_id)
    if event is None or event["shipped"]:
        return False
    with store.connect(folder) as db:
        db.execute(
            "UPDATE events SET deleted = 1, updated_utc = ? WHERE id = ?",
            (store.now_utc(), int(event_id)),
        )
    return True


def shipped_tables(event, names):
    """The shipped event's own readings, {site: tidy frame}, for its chart and the past-events check."""
    replay = sources.ReplaySource(event["shipped_id"])
    start, end = replay.window(None)
    tables = replay.tables(names, start, end)
    return {name: sources.tidy(frame) for name, frame in tables.items()}, replay.weather(start, end)


def shipped_raw(event, names):
    replay = sources.ReplaySource(event["shipped_id"])
    start, end = replay.window(None)
    return replay.tables(names, start, end), replay.weather(start, end)

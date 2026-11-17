"""Readings from the three sources, cached in workbench.sqlite, and what gets left out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strawberrywatch.watch import sources
from strawberrywatch.workbench import store, text

KINDS = ("api", "csv", "mysql")
STEP = pd.Timedelta(minutes=15)
# 96 readings a day at 15 minutes
PER_DAY = 96
WEATHER_SITE = "_weather"
WEATHER = ("rain_mm", "air_temp_c", "shortwave_radiation")
# Everything the models read, not only what step 1 shows
STORED = (*text.READINGS, "floating_conductivity")


def is_set_up(folder, kind):
    if kind == "api":
        return bool(store.secret("api_token"))
    if kind == "csv":
        return bool(store.setting(folder, "csv_folder"))
    if kind == "mysql":
        return all(store.setting(folder, k) for k in ("mysql_host", "mysql_database", "mysql_user"))
    return False


def credentials(folder):
    host = store.setting(folder, "mysql_host") or ""
    # host:port is allowed
    name, _, port = host.partition(":")
    return {
        "host": name,
        "port": int(port) if port.isdigit() else 3306,
        "user": store.setting(folder, "mysql_user"),
        "password": store.secret("mysql_password") or "",
        "database": store.setting(folder, "mysql_database"),
    }


def source(folder, kind):
    if not is_set_up(folder, kind):
        # Callers check is_set_up first and show the deck's sentence
        raise LookupError(kind)
    if kind == "api":
        return sources.ApiSource(store.secret("api_token"))
    if kind == "csv":
        return sources.CsvSource(store.setting(folder, "csv_folder"))
    sql = sources.SqlSource()
    # SqlSource reads .env unless it's handed a login
    sql._credentials = credentials(folder)
    return sql


def utc(day, end=False):
    """A local date as the UTC moment its day starts, or ends."""
    moment = pd.Timestamp(day).tz_localize(text.LOCAL)
    if end:
        moment = moment + pd.Timedelta(days=1) - STEP
    return moment.tz_convert("UTC")


def _stamp(moment):
    return pd.Timestamp(moment).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")


def _covered(db, kind, site, start, end):
    row = db.execute(
        "SELECT MIN(time_utc) AS lo, MAX(time_utc) AS hi FROM readings "
        "WHERE source = ? AND site = ? AND time_utc >= ? AND time_utc <= ?",
        (kind, site, _stamp(start), _stamp(end)),
    ).fetchone()
    if row["lo"] is None:
        return False
    # A day of slack at each end, since a quiet logger leaves its own gap
    slack = pd.Timedelta(days=1)
    return (
        pd.Timestamp(row["lo"], tz="UTC") <= start + slack
        and pd.Timestamp(row["hi"], tz="UTC") >= end - slack
    )


def _rows(kind, site, frame, columns):
    out = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        for moment, value in values.items():
            out.append((kind, site, column, _stamp(moment), float(value)))
    return out


def fetch(folder, kind, sites, start, end, again=False):
    """Fills the cache for these sites and dates, weather included. Cached readings aren't fetched again."""
    with store.connect(folder) as db:
        todo = [s for s in sites if again or not _covered(db, kind, s, start, end)]
        weather_due = again or not _covered(db, kind, WEATHER_SITE, start, end)
    if not todo and not weather_due:
        return
    src = source(folder, kind)
    rows = []
    if todo:
        for name, frame in src.tables(todo, start, end).items():
            rows += _rows(kind, name, sources.tidy(frame), STORED)
    if weather_due:
        weather = src.weather(start, end)
        if weather is not None:
            rows += _rows(kind, WEATHER_SITE, weather.resample("15min").mean(), WEATHER)
    with store.connect(folder) as db:
        db.executemany("REPLACE INTO readings VALUES (?, ?, ?, ?, ?)", rows)


def readings(folder, kind, sites, start, end, columns=STORED):
    """{site: frame} on the 15 minute grid, UTC index, one column per reading."""
    with store.connect(folder) as db:
        rows = db.execute(
            f"SELECT site, reading, time_utc, value FROM readings WHERE source = ? "
            f"AND site IN ({', '.join('?' for _ in sites)}) AND time_utc >= ? AND time_utc <= ?",
            [kind, *sites, _stamp(start), _stamp(end)],
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows], columns=["site", "reading", "time_utc", "value"])
    df = df[df["reading"].isin(columns)]
    out = {}
    for site, part in df.groupby("site"):
        frame = part.pivot_table(index="time_utc", columns="reading", values="value")
        frame.index = pd.to_datetime(frame.index, utc=True)
        out[site] = frame.sort_index()
    return out


def weather(folder, kind, start, end):
    frame = readings(folder, kind, [WEATHER_SITE], start, end, WEATHER).get(WEATHER_SITE)
    return None if frame is None or frame.empty else frame


def reviewed(tables):
    """The cached readings at the reviewed level, which a re-fit reads."""
    from strawberrywatch import field_log
    from strawberrywatch import inventory as inv
    from strawberrywatch.ingest import levels

    inventory = inv.load()
    return levels.reviewed(tables, inventory, field_log.load(inventory=inventory))[0]


def raw_tables(tables):
    """The cached readings under the logger's own column names, which the models read."""
    back = sources._raw_columns()
    return {
        site: frame.rename(columns={c: back[c] for c in frame.columns if c in back})
        for site, frame in tables.items()
    }


def rainy(index, rain):
    """True where the models' rain gate would be raising the alert line."""
    from strawberrywatch.anomalies.rain_gate import RainGate

    if rain is None or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    wet = rain.reindex(index).fillna(0.0).to_numpy(dtype=float)
    gate = RainGate(base_threshold=1.0)
    return gate.multipliers(wet, index.tz_convert("UTC").tz_localize(None).to_numpy()) > 1.0 + 1e-9


def left_out(index, site, events, rain, leave_events, leave_rain):
    """True at the steps step 1 leaves out for one site: known events there, and rainy hours."""
    mask = np.zeros(len(index), dtype=bool)
    if leave_events:
        for event in events:
            if site in event["sites"]:
                mask |= (index >= event["start"]) & (index <= event["end"])
    if leave_rain:
        mask |= rainy(index, rain)
    return mask


def grid(start, end):
    return pd.date_range(start, end, freq="15min")


def expected_pairs(sites, chosen):
    """(site, reading) pairs that exist in the inventory, among the sites and readings picked."""
    from strawberrywatch import inventory as inv

    inventory = inv.load()
    out = []
    for site in sites:
        try:
            have = set(inventory.site(site).scored_variables)
        except Exception:
            continue
        out += [(site, r) for r in chosen if r in have]
    return out


# Sensor status on Home and Sensors


def refresh_status(folder, kind, sites, now=None):
    """The last day of readings for every site, fetched again."""
    now = now or pd.Timestamp.now(tz="UTC")
    fetch(folder, kind, sites, now - pd.Timedelta(days=1), now, again=True)


def last_seen(folder, kind, sites):
    """{site: {reading: last UTC time}} over everything cached."""
    with store.connect(folder) as db:
        rows = db.execute(
            "SELECT site, reading, MAX(time_utc) AS last FROM readings WHERE source = ? "
            "GROUP BY site, reading",
            (kind,),
        ).fetchall()
    out = {site: {} for site in sites}
    for row in rows:
        if row["site"] in out:
            out[row["site"]][row["reading"]] = pd.Timestamp(row["last"], tz="UTC")
    return out


def hour_counts(folder, kind, site, now=None):
    """Readings that arrived in each of the last 24 hours, oldest first, counted on the busiest reading."""
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    start = now - pd.Timedelta(hours=23)
    tables = readings(folder, kind, [site], start, now + pd.Timedelta(hours=1) - STEP)
    frame = tables.get(site)
    hours = pd.date_range(start, now, freq="h")
    if frame is None or frame.empty:
        return [0] * 24
    counts = frame.notna().groupby(frame.index.floor("h")).sum().max(axis=1)
    return [int(min(counts.get(h, 0), 4)) for h in hours]

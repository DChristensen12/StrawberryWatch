"""The ingest job: readings to raw and screened, and weather, every 15 minutes, apart from checks."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from strawberrywatch.ingest import levels
from strawberrywatch.support_modules import qc_tests as qc
from strawberrywatch.watch import health, sources
from strawberrywatch.watch import settings as settings_mod

logger = logging.getLogger(__name__)

INTERVAL = timedelta(minutes=settings_mod.INTERVAL_MINUTES)
LOOKBACK = timedelta(days=settings_mod.LOOKBACK_DAYS)
# The furthest back a pass reaches after an outage
CATCH_UP = timedelta(days=30)
# Saved weather older than this is used, and the check's alerts say so
WEATHER_STALE = 3600
# The raw level's name for each live source
RAW_SOURCE = {sources.API: "api", sources.SITE: "sql"}
# How the data health email goes. A check's own pass sends none
LOCAL_MAIL, WEBSITE_MAIL = "local", "website"
# The old readings table's columns back to the logger's
BATTERY = "EnviroDIY_Mayfly_Batt"

# One pass at a time in this process
_running = threading.Lock()


# Extract, transform and load, the split HydroServer's ETL uses
# https://hydroserver2.github.io/hydroserver/applications/hydroserverpy/etl.html


def extract(source, sites, start, end):
    """The readings as fetched, {site: logger columns at their own stamps}."""
    return source.raw_tables(sites, start, end)


def transform(tables):
    """Raw tables as rows of (site, column, epoch seconds, value). Only readings with a value."""
    rows = []
    for site, frame in tables.items():
        for col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            values = values[np.isfinite(values)]
            rows += [(site, col, stamp.timestamp(), float(v)) for stamp, v in values.items()]
    return rows


def load(store, rows, kind, fetched_at):
    """Appends the rows to the raw level and returns the ones that were new."""
    return store.save_raw(rows, kind, fetched_at)


def _quiet_steps(store, site, table, last):
    """Each column's cadence from its newest two days of readings, for a sensor gone quiet."""
    busy = len(table) and all(table[c].notna().sum() >= 3 for c in table.columns)
    if busy or last is None:
        return {}
    older = levels.table(store.raw(site, last - 2 * 86400, last))
    return {c: levels.cadence(older[c].dropna().index) for c in older.columns}


def screen(store, inventory, sites, start, end, now=None):
    """Flags each site's raw readings over [start, end] and saves them to the screened level."""
    # far enough back for the flat line test to count a full run
    context = levels.STEP * (inventory.flat_line_fail_count + 1)
    last = store.raw_last()
    for site in sites:
        if site not in inventory.sites:
            continue
        rows = store.raw(site, (start - context).timestamp(), end.timestamp())
        table = levels.table(rows) if rows else pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
        fetched = {(r["col"], r["time"]): r["fetched_at"] for r in rows}
        steps = _quiet_steps(store, site, table, last.get(site))
        flags = levels.screen_table(table, site, inventory, start - context, end, now, steps)

        out = []
        for col, frame in flags.items():
            frame = frame[frame.index >= start]
            times = frame.index.as_unit("ns").asi8 / 1e9
            tests = frame.drop(columns="flag").to_numpy()
            names = list(frame.columns[1:])
            for when, flag, row in zip(times, frame["flag"].to_numpy(), tests, strict=True):
                # only the tests that didn't pass, and none for a step with no reading
                record = None
                if flag != qc.MISSING and (row != qc.GOOD).any():
                    record = json.dumps(
                        {n: int(v) for n, v in zip(names, row, strict=True) if v != qc.GOOD}
                    )
                out.append((col, float(when), int(flag), record, fetched.get((col, float(when)))))
        store.save_screened(site, start.timestamp(), end.timestamp(), out, now)


def weather(store, source, start, end, now):
    """Fetches the window's weather and saves it. False when none came back, and nothing is saved."""
    frame = source.weather(start, end)
    if frame is None or frame.empty or "rain_mm" not in frame.columns:
        return False
    frame = frame.apply(pd.to_numeric, errors="coerce")
    rows = [
        (stamp.timestamp(), *[_num(values.get(c)) for c in sources.WEATHER_COLUMNS])
        for stamp, values in frame.iterrows()
    ]
    store.save_weather(rows, fetched_at=now)
    return True


def _num(value):
    return None if value is None or not np.isfinite(value) else float(value)


def _reach_back(store, start, end):
    """An earlier start after an outage: back to the last pass that got readings, 30 days at most."""
    last = store.last_ingest(ok=True)
    if last is None:
        return start
    resumed = pd.Timestamp(last["started_at"], unit="s", tz="UTC") - LOOKBACK
    return max(min(start, resumed), end - CATCH_UP)


def _pass(store, token, clock, source=None, lookback=LOOKBACK, window=None, mail=None):
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import runner

    config = store.source()
    if config is None or not sources.is_live(config["kind"]) or not store.has_levels():
        return None

    began = clock()
    ingest_id = store.start_ingest(began)
    problems, added, got_readings = [], [], False
    tables = {}
    try:
        inventory = inv.load()
        sites = list(inventory.sites)
        source = source or sources.from_config(config, token=token)
        start, end = window or source.window(lookback)
        start = _reach_back(store, start, end)
        try:
            tables = extract(source, sites, start, end)
            added = load(store, transform(tables), RAW_SOURCE.get(source.kind, source.kind), began)
            # the dashboard's site pages and status read these
            gridded = {site: sources.on_grid(frame) for site, frame in tables.items()}
            runner._save_readings(store, gridded, live=True)
            got_readings = True
        except Exception as exc:
            logger.warning("ingest couldn't fetch readings: %s", runner._plain(exc))
            problems.append(runner._plain(exc))
        # Screened after a failed fetch too, and the steps that went missing get their rows
        first = min([t for _, _, t, _ in added], default=start.timestamp())
        screen(
            store,
            inventory,
            sites,
            min(start, pd.Timestamp(first, unit="s", tz="UTC")),
            end,
            clock(),
        )
        if not weather(store, source, start, end, clock()):
            problems.append("Weather fetch failed")
        # Raw is saved by now, and the health checks can't stop it
        health.after_pass(store, inventory, tables, start, end, clock(), mail)
    except Exception as exc:
        logger.exception("ingest pass failed")
        problems.append(runner._plain(exc))
    store.finish_ingest(ingest_id, got_readings, "; ".join(problems), len(added), clock())
    return store.last_ingest()


def run(store, token=None, clock=time.time, source=None, lookback=LOOKBACK, mail=None):
    """One ingest pass, or None without a live source or the level tables. Doesn't raise."""
    with _running:
        return _pass(store, token, clock, source, lookback, mail=mail)


def ensure(store, token=None, clock=time.time, source=None, window=None, max_age=None, mail=None):
    """A pass, unless one finished in the last few minutes. The thread and a check share it."""
    max_age = INTERVAL.total_seconds() / 2 if max_age is None else max_age
    with _running:
        last = store.last_ingest()
        if last is not None and clock() - last["finished_at"] < max_age:
            return last
        return _pass(store, token, clock, source, LOOKBACK, window, mail)


def start(store, token=lambda: None, interval=None, stop=None):
    """Runs a pass on its own thread every 15 minutes until stop is set. Returns (thread, stop)."""
    interval = INTERVAL.total_seconds() if interval is None else interval
    stop = stop or threading.Event()

    def loop():
        while not stop.is_set():
            try:
                ensure(store, token=token(), mail=LOCAL_MAIL)
            except Exception:
                logger.exception("ingest pass failed")
            stop.wait(interval)

    thread = threading.Thread(target=loop, name="strawberrywatch-ingest", daemon=True)
    thread.start()
    return thread, stop


def backfill(store, raw_dir=None, clock=time.time):
    """Copies data/raw_data and the store's live readings into the raw level, once."""
    import glob
    import os

    from strawberrywatch import inventory as inv
    from strawberrywatch import paths
    from strawberrywatch.ingest import raw_data_loader as rdl

    inventory = inv.load()
    counts = {"archive": 0, "store": 0}

    raw_dir = raw_dir if raw_dir is not None else paths.raw_data_dir()
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        frame = rdl._read_flexible_csv(path)
        column = next((c for c in rdl._TIME_CANDIDATES if c in frame.columns), None)
        if column is None:
            continue
        frame.index = pd.to_datetime(frame.pop(column), utc=True, errors="coerce")
        frame = frame[frame.index.notna()].drop(
            columns=[c for c in ("uuid", "site_code", "station_id") if c in frame.columns]
        )
        site = rdl._table_name_for_file(path)
        # the file's own date is the nearest record of when it was fetched
        added = load(store, transform({site: frame}), "archive", os.path.getmtime(path))
        counts["archive"] += len(added)

    raw = sources._raw_columns() | {"battery": BATTERY}
    for site in inventory.sites:
        rows = []
        for reading in store.readings(site, live=True):
            rows += [
                (site, raw[name], reading["time"], float(reading[name]))
                for name in raw
                if reading.get(name) is not None
            ]
        counts["store"] += len(load(store, rows, "store", clock()))

    now = clock()
    for site, newest in store.raw_last().items():
        oldest = store.raw(site, until=newest)[0]["time"]
        screen(
            store,
            inventory,
            [site],
            pd.Timestamp(oldest, unit="s", tz="UTC"),
            pd.Timestamp(newest, unit="s", tz="UTC"),
            now,
        )
    return counts

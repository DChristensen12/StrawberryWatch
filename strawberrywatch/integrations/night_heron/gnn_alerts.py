"""
pending_alerts(), the one function Night Heron's alert daemon calls each cycle.
It reads their creek tables once, runs every model in GNN_MODELS, and hands
back alerts shaped for their fire_alerts_task. Only the standard library is
imported at module scope, and importing this costs their daemon almost nothing.

With GNN_DASHBOARD=1 the models and settings come from the dashboard instead.
Each check runs in its own process with a time limit and a memory floor, and
can fail without taking Night Heron's alerts down. StrawberryWatch emails its
own list from that process. Night Heron only gets alerts to text.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


def _env_number(name, default, cast):
    """
    Reads a numeric setting, falling back to the default if it's unusable. These
    run at import, inside their daemon's try/except ImportError, where a
    ValueError from a .env typo would take the daemon down. Bad values get logged and
    ignored, and blank counts as unset.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(raw)
    except ValueError:
        logger.warning("gnn: %s=%r is not a number, using %s instead", name, raw, default)
        return cast(default)


def _model_names():
    """
    Which models to run, from GNN_MODELS, comma separated. The old single-model
    GNN_MODEL_NAME still counts if it's unset. Names are checked when the worker
    loads them, since a bad name at import would crash their startup.
    """
    raw = os.getenv("GNN_MODELS") or os.getenv("GNN_MODEL_NAME") or "dusk_crayfish"
    names = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name and name not in names:
            names.append(name)
    return names or ["dusk_crayfish"]


# Their daemon loops every twenty seconds and the sensors report every fifteen
# minutes. Scoring every pass would give the same answer forty times over.
SCORE_INTERVAL = timedelta(minutes=_env_number("GNN_SCORE_INTERVAL_MINUTES", 15, int))

# How long a (model, site, sensor, rule) stays quiet after it alerts. Long enough
# that a real event that lasts all afternoon sends one email, not eighty.
COOLDOWN = timedelta(hours=_env_number("GNN_ALERT_COOLDOWN_HOURS", 6, float))

# How much history to pull. Dusk Crayfish wants 30 real readings before it will
# judge a site, and Cobble Shoal wants enough behind its newest step that a node
# gone quiet looks properly stale. Two days covers both with room to spare.
LOOKBACK = timedelta(days=_env_number("GNN_LOOKBACK_DAYS", 2, int))

# The off switch. Otherwise the only way to quiet the models is uninstalling,
# which takes the checkpoints and 800MB of torch with it.
ENABLED = os.getenv("GNN_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

CHECKPOINT_DIR = os.getenv("GNN_CHECKPOINT_DIR")
MODELS = _model_names()

# Off until someone sets it. Installing the dashboard leaves the live alerts
# alone until the website's half is in place too
DASHBOARD = os.getenv("GNN_DASHBOARD", "0").strip().lower() in ("1", "true", "yes", "on")

# Failed or skipped checks in a row before the checks pause, until an editor
# presses Restart checks. Stops a crash loop.
PAUSE_AFTER = 3

# Inventory names to Night Heron's sensor names. Theirs where they have one,
# since that key picks the unit in their email, and the raw column code where
# they don't (as their own COL map does).
SENSOR_NAMES = {
    "conductivity": "conductivity",
    "depth": "depth",
    "temperature": "temperature",
    "dissolved_oxygen": "AtlasSci_DO",
    "floating_conductivity": "AtlasSci_FloatCond",
}

# Dusk Crayfish shipped as gnn_anomaly_<rule> and their AlertEvent table already
# holds rows under those names. It keeps them. Every model after it goes out
# under its own name. Their event_type is a CharField(max_length=50), and the
# longest this makes today is cobble_shoal_combined_fisher, at 28.
ALERT_PREFIXES = {"dusk_crayfish": "gnn_anomaly"}

# On all of their tables, and not a reading
_BOOKKEEPING = ("uuid", "site_code", "station_id")

_state = {"last_scored": None, "sent": {}, "restored": False}
_ready = []
_worker = None
_lock = threading.Lock()


def reset_state():
    """Clear the throttles and the queue. For tests, and for a clean first pass."""
    with _lock:
        _state["last_scored"] = None
        _state["sent"] = {}
        _state["restored"] = False
        _ready.clear()


def _recipients(name, default=""):
    raw = os.getenv(name, default)
    return [part.strip() for part in raw.split(",") if part.strip()]


def _checkpoint_dir():
    if CHECKPOINT_DIR:
        return CHECKPOINT_DIR
    # Falling back to the checkout we are installed from. Works for a developer
    # running out of a clone, raises with a useful message anywhere else.
    from strawberrywatch import paths

    try:
        return str(paths.checkpoints_dir())
    except RuntimeError as exc:
        raise RuntimeError(
            "no checkpoint directory. Set GNN_CHECKPOINT_DIR to the folder holding the "
            "model checkpoints, since there is no StrawberryWatch checkout to find one "
            "relative to."
        ) from exc


def _alert_type(model, rule):
    return f"{ALERT_PREFIXES.get(model, model)}_{rule}"


def _detectors():
    """
    Every model in MODELS, loaded, in the order named.

    Loading is cached. After the first pass this is a dict lookup per model. A
    model that won't load is logged and left out of the pass, and the rest
    still run.
    """
    from strawberrywatch import paths
    from strawberrywatch.serving import detector_class

    loaded = []
    for name in MODELS:
        try:
            folder = paths.model_dir(name, _checkpoint_dir())
            loaded.append(detector_class(name).cached(folder, device="cpu"))
        except Exception:
            logger.exception("gnn: %s would not load, running the others without it", name)
    return loaded


def _read_tables(names, start, end):
    """
    Recent rows for each table from Night Heron's MySQL, read once a pass, as
    {table: frame} indexed by reading time. Models share a read, and an empty
    table is left out so one dead logger can't stop the rest being scored.
    """
    from strawberrywatch.ingest.sql_client import fetch_creek_data_sql

    tables = {}
    for name in names:
        raw = fetch_creek_data_sql(name, start, end)
        if raw.empty:
            logger.info("gnn: no rows for %s", name)
            continue
        raw = raw.drop(columns=[c for c in _BOOKKEEPING if c in raw.columns])
        tables[name] = raw.set_index("timestamp")
    return tables


def _weather(start, end):
    """
    Open-Meteo for the window, fetched once a pass for every model that wants it.
    Night Heron doesn't keep weather, and their database has none to join
    against. A failed fetch comes back None and the models go on without it.
    """
    try:
        from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

        weather = fetch_open_meteo_weather(start, end)
    except Exception as exc:
        logger.warning("gnn: weather fetch failed (%s), scoring without it", exc)
        return None
    return None if weather.empty else weather


def _due(now):
    last = _state["last_scored"]
    return last is None or now - last >= SCORE_INTERVAL


def pending_alerts(now=None):
    """
    Alerts worth emailing now, as dicts with values, site, sensor, alert_type,
    emails and phones for fire_alerts_task. An empty list is the normal answer.

    Returns immediately. Their loop pings a systemd watchdog every pass, and a
    cold model load plus a weather fetch can take long enough to look like a
    hang. Scoring runs on a background thread and this returns what the last
    finished pass found. Alerts arrive one cycle (twenty seconds) late.
    """
    global _worker
    if not ENABLED:
        return []

    now = now or datetime.now(UTC)

    with _lock:
        ready, _ready[:] = list(_ready), []
        busy = _worker is not None and _worker.is_alive()
        if _due(now) and not busy:
            _state["last_scored"] = now
            _worker = threading.Thread(target=_run_pass, args=(now,), name="gnn-score", daemon=True)
            _worker.start()

    return ready


def _run_pass(now):
    """
    One scoring pass, on the worker thread. Doesn't raise. A daemon that's been
    up for months shouldn't fall over because a model had a bad day.
    """
    try:
        found = _check_in_child(now) if DASHBOARD else _score(now)
    except Exception:
        logger.exception("gnn: anomaly pass failed, no alerts this cycle")
        return
    if found:
        with _lock:
            _ready.extend(found)


def _score(now):
    import torch

    # torch otherwise sizes its thread pool to the whole box. We're a guest in
    # their daemon. Take one thread and let the pass run a little longer.
    torch.set_num_threads(1)

    detectors = _detectors()
    if not detectors:
        return []

    names = []
    for detector in detectors:
        names += [name for name in detector.table_names if name not in names]

    start, end = now - LOOKBACK, now
    tables = _read_tables(names, start, end)
    if not tables:
        logger.info("gnn: no creek data in the last %s, nothing to score", LOOKBACK)
        return []

    weather = _weather(start, end) if any(d.wants_weather for d in detectors) else None
    emails, phones = _recipients("GNN_ALERT_EMAILS"), _recipients("GNN_ALERT_PHONES")

    alerts = []
    for detector in detectors:
        mine = {name: tables[name] for name in detector.table_names if name in tables}
        try:
            found = detector.findings(mine, weather)
        except Exception:
            logger.exception("gnn: %s failed this pass, the others still ran", detector.name)
            continue
        for finding in found:
            alert = _alert(detector.name, finding, now, emails, phones)
            if alert is not None:
                alerts.append(alert)
    return alerts


def _alert(model, finding, now, emails, phones):
    """One Finding as the dict fire_alerts_task takes, or None while it is cooling down."""
    sensor = SENSOR_NAMES.get(finding.variable, finding.variable)
    key = (model, finding.site, sensor, finding.rule)
    with _lock:
        last = _state["sent"].get(key)
        if last is not None and now - last < COOLDOWN:
            return None
        _state["sent"][key] = now

    logger.warning(
        "gnn: %s flagged %s %s by %s, peak %.2f against %.2f, first over at %s",
        model,
        finding.site,
        sensor,
        finding.rule,
        finding.peak,
        finding.threshold,
        finding.when,
    )
    return {
        "values": [float(v) for v in finding.readings.tail(200)],
        "site": finding.site,
        "sensor": sensor,
        "alert_type": _alert_type(model, finding.rule),
        "emails": emails,
        "phones": phones,
    }


# The dashboard's way: each check in a child process, under the website's limits


def _check_in_child(now, store=None, run=subprocess.run):
    """
    One dashboard check as python -m strawberrywatch.watch.check_once, read back
    as JSON. Skipped when the server is low on memory, stopped when it runs
    past the time limit, and paused after PAUSE_AFTER failures in a row.
    """
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch.store_mysql import MysqlStore

    store = store or MysqlStore()
    limits = ws.website_limits()
    _restore_cooldowns(store, now)
    if store.pref("checks_paused") == "yes":
        logger.info("gnn: model checks are paused until someone presses Restart checks")
        return []
    free = ws.free_memory_mb()
    if free is not None and free < limits["min_free_memory_mb"]:
        _failed(store, "Skipped: the server was low on memory")
        logger.warning(
            "gnn: %.0f MB free, under the %.0f MB floor, check skipped",
            free,
            limits["min_free_memory_mb"],
        )
        return []
    minutes = limits["check_timeout_minutes"]
    try:
        done = run(
            [sys.executable, "-m", "strawberrywatch.watch.check_once"],
            capture_output=True,
            text=True,
            timeout=minutes * 60,
        )
    except subprocess.TimeoutExpired:
        _failed(store, f"Stopped: took longer than {minutes:g} minutes")
        return []
    lines = [line for line in (done.stdout or "").splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1])
    except (IndexError, ValueError):
        reason = (
            "Stopped: the check's process was ended before it finished"
            if done.returncode < 0
            else "The check stopped with an error before it could say what it found"
        )
        _failed(store, reason)
        logger.error(
            "gnn: check process exited %s: %s", done.returncode, (done.stderr or "")[-500:]
        )
        return []
    if result.get("ok"):
        store.set_pref("check_failures", "0")
    else:
        _count_failure(store)
    phones = _recipients("GNN_ALERT_PHONES") if result.get("email_on") else []
    if not phones:
        return []
    alerts = []
    for item in result.get("alerts") or []:
        alert = _queue(
            item["model"],
            item["site"],
            item["variable"],
            item["rule"],
            item["values"],
            now,
            [],
            phones,
        )
        if alert is not None:
            alerts.append(alert)
    return alerts


def _failed(store, message):
    """Record a check that didn't run or didn't finish, for the dashboard to show."""
    from strawberrywatch.watch import sources

    last = store.last_check()
    if last is not None and last.get("finished_at") is None:
        store.finish_check(last["id"], False, message, {})
    else:
        check = store.start_check(sources.SITE, None, None, None)
        store.finish_check(check, False, message, {})
    _count_failure(store)


def _count_failure(store):
    try:
        count = int(store.pref("check_failures") or 0) + 1
    except ValueError:
        count = 1
    store.set_pref("check_failures", str(count))
    if count >= PAUSE_AFTER:
        store.set_pref("checks_paused", "yes")
        logger.warning("gnn: %s checks failed in a row, model checks paused", count)


def _restore_cooldowns(store, now):
    """
    What was alerted in the last COOLDOWN, from the dashboard's own record. A
    restart of their daemon won't send the same alerts again.
    """
    if _state["restored"]:
        return
    since = (now - COOLDOWN).timestamp()
    for item in store.noticed(flagged=True):
        if item["checked_at"] < since:
            continue
        sensor = SENSOR_NAMES.get(item["variable"], item["variable"])
        key = (item["model"], item["site"], sensor, item["rule"])
        when = datetime.fromtimestamp(item["checked_at"], UTC)
        with _lock:
            if key not in _state["sent"] or _state["sent"][key] < when:
                _state["sent"][key] = when
    _state["restored"] = True


def _queue(model, site, variable, rule, values, now, emails, phones):
    """One dashboard finding as the dict fire_alerts_task takes, or None while it's cooling down."""
    sensor = SENSOR_NAMES.get(variable, variable)
    key = (model, site, sensor, rule)
    with _lock:
        last = _state["sent"].get(key)
        if last is not None and now - last < COOLDOWN:
            return None
        _state["sent"][key] = now
    logger.warning("gnn: %s flagged %s %s by %s", model, site, sensor, rule)
    return {
        "values": [float(v) for v in values],
        "site": site,
        "sensor": sensor,
        "alert_type": _alert_type(model, rule),
        "emails": emails,
        "phones": phones,
    }

"""
One function Night Heron's alert daemon calls once a cycle: pending_alerts().

It reads their creek tables once, runs every model named in GNN_MODELS over
them, and hands back alerts already shaped for the fire_alerts_task they already
have. They never see a tensor, a checkpoint, or torch. Nothing is imported at
module scope except the standard library, so importing this costs their daemon
almost nothing at startup.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)


def _env_number(name, default, cast):
    """
    Read a numeric setting, falling back to the default if it is unusable.

    These run at import, and their daemon imports us inside a try/except
    ImportError. A ValueError out of int() sails straight past that and takes the
    whole daemon down, static and moving threshold alerts included, over a typo in
    a .env file. Nothing here is worth that, so a bad value gets logged and
    ignored. Blank counts as unset, since that is what an empty .env line means.
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
    Which models to run, from GNN_MODELS, comma separated.

    GNN_MODEL_NAME was the setting back when there could only be one, so it still
    counts if GNN_MODELS is unset. Names are checked when the worker loads them
    rather than here, because a bad name at import would be a crash at their
    startup.
    """
    raw = os.getenv("GNN_MODELS") or os.getenv("GNN_MODEL_NAME") or "dusk_crayfish"
    names = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name and name not in names:
            names.append(name)
    return names or ["dusk_crayfish"]


# Their daemon loops every twenty seconds and the sensors report every fifteen
# minutes, so scoring every pass would just be the same answer forty times over.
SCORE_INTERVAL = timedelta(minutes=_env_number("GNN_SCORE_INTERVAL_MINUTES", 15, int))

# How long a (model, site, sensor, rule) stays quiet after it alerts. Long enough
# that a real event that lasts all afternoon sends one email, not eighty.
COOLDOWN = timedelta(hours=_env_number("GNN_ALERT_COOLDOWN_HOURS", 6, float))

# How much history to pull. Dusk Crayfish wants 30 real readings before it will
# judge a site, and Cobble Shoal wants enough behind its newest step that a node
# gone quiet looks properly stale. Two days covers both with room to spare.
LOOKBACK = timedelta(days=_env_number("GNN_LOOKBACK_DAYS", 2, int))

# The off switch. Without one the only way to quiet the models is to uninstall
# the package, which also takes away the checkpoints and 800MB of torch, and
# their daemon has no other way to skip us.
ENABLED = os.getenv("GNN_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

CHECKPOINT_DIR = os.getenv("GNN_CHECKPOINT_DIR")
MODELS = _model_names()

# Inventory names to Night Heron's sensor names. Theirs where they have one,
# since that key picks the unit in their email, and the raw column code where
# they do not, which is what their own COL map does.
SENSOR_NAMES = {
    "conductivity": "conductivity",
    "depth": "depth",
    "temperature": "temperature",
    "dissolved_oxygen": "AtlasSci_DO",
    "floating_conductivity": "AtlasSci_FloatCond",
}

# Dusk Crayfish shipped as gnn_anomaly_<rule> and their AlertEvent table already
# holds rows under those names, so it keeps them. Every model after it goes out
# under its own name. Their event_type is a CharField(max_length=50), and the
# longest this makes today is cobble_shoal_combined_fisher, at 28.
ALERT_PREFIXES = {"dusk_crayfish": "gnn_anomaly"}

# On every one of their tables and never a reading
_BOOKKEEPING = ("uuid", "site_code", "station_id")

_state = {"last_scored": None, "sent": {}}
_ready = []
_worker = None
_lock = threading.Lock()


def reset_state():
    """Forget the throttles and anything queued. For tests, and for a clean first pass."""
    with _lock:
        _state["last_scored"] = None
        _state["sent"] = {}
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

    Loading is cached, so after the first pass this is a dict lookup per model.
    One that will not load is logged and left out of the pass rather than taking
    the rest down with it.
    """
    from strawberrywatch.serving import detector_class

    loaded = []
    for name in MODELS:
        try:
            loaded.append(detector_class(name).cached(_checkpoint_dir(), device="cpu"))
        except Exception:
            logger.exception("gnn: %s would not load, running the others without it", name)
    return loaded


def _read_tables(names, start, end):
    """
    Recent rows for each table out of Night Heron's MySQL, read once a pass.

    Comes back as {table: frame} indexed by reading time with the raw logger
    columns, which is what every detector takes. Two models wanting the same table
    share the one read. A table with no rows is left out rather than raising,
    because one dead logger should not stop the others being scored.
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

    Night Heron does not store weather. It fetches it per rule from WeatherAPI and
    throws it away, so there is nothing in their database to join against. A
    failed fetch comes back None and each model goes on without it, which mostly
    costs us during storms.
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
    Anomalies worth emailing right now, ready to hand to fire_alerts_task.

    Returns a list of dicts with values, site, sensor, alert_type, emails and
    phones. An empty list is the normal answer and means one of: nothing fired,
    what fired is still in cooldown, it is not time to score again, the pass is
    still running, or GNN_ENABLED is off.

    Returns immediately, whatever the models are doing. Their loop pings a systemd
    watchdog every pass, and a cold model load plus a weather fetch can take long
    enough that systemd decides the daemon has hung. So scoring runs on a
    background thread and this hands back what the last finished pass found,
    which makes alerts one cycle late: twenty seconds against fifteen minutes.
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
    One scoring pass, on the worker thread.

    Never raises. A daemon that has been running for months should not fall over
    because a model had a bad day, and an exception escaping a thread would be
    invisible to the caller anyway.
    """
    try:
        found = _score(now)
    except Exception:
        logger.exception("gnn: anomaly pass failed, no alerts this cycle")
        return
    if found:
        with _lock:
            _ready.extend(found)


def _score(now):
    import torch

    # torch otherwise sizes its thread pool to the whole box. We are a guest in
    # their daemon, so take one and let the pass take a little longer.
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

"""
One function Night Heron's alert daemon calls once a cycle: pending_alerts().

It reads their creek tables, runs Dusk Crayfish, and hands back a list of alerts
already shaped for the fire_alerts_task they already have. They never see a
tensor, a checkpoint, or torch. Everything that could break lives here, where the
traceback points at our code.

Their daemon loops roughly every twenty seconds. Scoring on every pass would be
pointless, since sensors report every fifteen minutes, so there are two throttles
below. SCORE_INTERVAL stops us running the model more often than the data
changes, and the per site cooldown stops a standing anomaly mailing somebody
every cycle for as long as it lasts.

pending_alerts never blocks. Their loop pings a systemd watchdog on every pass,
and a cold model load plus a weather fetch can take tens of seconds, which is
long enough that systemd could decide the daemon has hung and restart it. So the
scoring runs on a background thread and pending_alerts hands back whatever the
last finished pass produced. Alerts arrive one cycle later than they otherwise
would, which is twenty seconds against a fifteen minute scoring cadence.

Nothing here is imported at module scope except the standard library. torch and
pandas come in when the worker actually runs, so importing this costs their
daemon almost nothing at startup.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

# How often to actually run the model, whatever the caller's loop does.
SCORE_INTERVAL = timedelta(minutes=int(os.getenv("GNN_SCORE_INTERVAL_MINUTES", "15")))

# How long a site stays quiet after it alerts. Long enough that a real event that
# lasts all afternoon sends one email, not eighty.
COOLDOWN = timedelta(hours=float(os.getenv("GNN_ALERT_COOLDOWN_HOURS", "6")))

# How much history to pull. The model needs 24 rows for one window, and the
# detection rules need 30 real readings before they will judge a site at all, so
# two days at a fifteen minute cadence leaves plenty of headroom.
LOOKBACK = timedelta(days=int(os.getenv("GNN_LOOKBACK_DAYS", "2")))

CHECKPOINT_DIR = os.getenv("GNN_CHECKPOINT_DIR")
MODEL_NAME = os.getenv("GNN_MODEL_NAME", "dusk_crayfish")

# Raw logger column names to what the model calls them. The same mapping
# ingest/data_loader.py uses, repeated here so this module does not import the
# ingest package and drag the API client along with it.
COLUMNS = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
    "timestamp": "datetime",
    "station_id": "location",
}

# Which sensor the alert is about, for their email subject and their unit lookup.
# Both rules score conductivity, so this is always conductivity today.
ALERT_SENSOR = "conductivity"
ALERT_TYPE = "gnn_anomaly"

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
            "no checkpoint directory. Set GNN_CHECKPOINT_DIR to the folder holding "
            "dusk_crayfish_weights.pt, since there is no StrawberryWatch checkout to "
            "find one relative to."
        ) from exc


def _read_creek(sites, start, end):
    """
    Pull recent rows for each site out of Night Heron's MySQL and stack them.

    Reads the same per site tables their get_creek_data daemon writes. We only
    ever read. A site with no table or no rows is skipped rather than raising,
    because one dead logger should not stop the other three being scored.
    """
    import pandas as pd

    from strawberrywatch.ingest.sql_client import fetch_creek_data_sql

    frames = []
    for site in sites:
        raw = fetch_creek_data_sql(site, start, end)
        if raw.empty:
            logger.info("gnn: no rows for %s", site)
            continue
        raw = raw.rename(columns=COLUMNS)
        raw["location"] = site
        frames.append(raw)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame.set_index("datetime").sort_index()


def _add_weather(frame, features, start, end):
    """
    Merge the weather columns the model trained on, if we can get them.

    Night Heron does not store weather. It fetches it per rule from WeatherAPI and
    throws it away, so there is nothing in their database to join against. We pull
    the same Open Meteo series the model was trained against instead.

    If that fetch fails, the missing columns are left absent and the caller fills
    them with the training mean. The model still runs. It just loses the weather
    context, which mostly costs us during storms. Returns whether we got it.
    """
    wanted = [c for c in ("rain_mm", "air_temp_c", "shortwave_radiation") if c in features]
    if not wanted:
        return frame, False

    try:
        from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

        weather = fetch_open_meteo_weather(start, end)
    except Exception as exc:
        logger.warning("gnn: weather fetch failed (%s), scoring without it", exc)
        return frame, False

    if weather.empty:
        return frame, False

    have = [c for c in wanted if c in weather.columns]
    if not have:
        return frame, False

    # Weather is on its own 15 minute grid. Snap each reading to the quarter hour
    # it falls in rather than merging on an exact timestamp match that will miss.
    keyed = frame.copy()
    keyed["_quarter"] = keyed.index.floor("15min")
    merged = keyed.merge(
        weather[have].resample("15min").mean(),
        how="left",
        left_on="_quarter",
        right_index=True,
    ).drop(columns=["_quarter"])
    merged.index = frame.index
    return merged, True


def _fill_absent(frame, features, normalization):
    """
    Give every trained feature a column, filling any we could not get.

    A missing feature gets its training mean, which lands on zero once the window
    builder normalizes. That is the same thing main.py does when a weather fetch
    fails mid run, and it means "no signal" rather than "a reading of zero".
    """
    out = frame.copy()
    for name in features:
        if name not in out.columns:
            out[name] = normalization[name][0]
    return out


def _due(now):
    last = _state["last_scored"]
    return last is None or now - last >= SCORE_INTERVAL


def _cooling_down(site, rule, now):
    last = _state["sent"].get((site, rule))
    return last is not None and now - last < COOLDOWN


def pending_alerts(now=None):
    """
    Anomalies worth emailing right now, ready to hand to fire_alerts_task.

    Returns a list of dicts with values, site, sensor, alert_type, emails and
    phones. An empty list is the normal answer and means one of: nothing fired,
    what fired is still in cooldown, it is not time to score again, or the pass
    is still running.

    Returns immediately. The model runs on a background thread, so a slow load or
    a slow weather fetch cannot stall the caller's loop. Call it as often as you
    like.
    """
    global _worker
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
    because our model had a bad day, and an exception escaping a thread would be
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
    from strawberrywatch.serving import DuskCrayfishDetector

    detector = DuskCrayfishDetector.cached(_checkpoint_dir(), MODEL_NAME, device="cpu")

    start, end = now - LOOKBACK, now
    frame = _read_creek(detector.sites, start, end)
    if frame.empty:
        logger.info("gnn: no creek data in the last %s, nothing to score", LOOKBACK)
        return []

    frame, got_weather = _add_weather(frame, detector.features, start, end)
    if not got_weather:
        logger.info("gnn: scoring without weather context")

    normalization = detector.checkpoint.normalization()
    sensor_cols = [c for c in detector.features if c in frame.columns] + ["location"]
    frame = _fill_absent(frame[sensor_cols], detector.features, normalization)

    rain = frame["rain_mm"] if got_weather and "rain_mm" in frame.columns else None
    result = detector.score(frame, rain=rain)
    if not result["windows"]:
        logger.info("gnn: not enough consecutive readings to fill a window")
        return []

    emails, phones = _recipients("GNN_ALERT_EMAILS"), _recipients("GNN_ALERT_PHONES")
    alerts = []
    for site, verdict in result["verdicts"].items():
        if not verdict.get("judged") or not verdict.get("flagged"):
            continue
        for rule in verdict["rules_fired"]:
            if _cooling_down(site, rule, now):
                continue
            with _lock:
                _state["sent"][(site, rule)] = now

            readings = frame.loc[frame["location"] == site, "conductivity"].dropna()
            peak = verdict["rule1" if rule == "forecast_residual" else "rule2"]["peak_deviation"]
            logger.warning(
                "gnn: %s flagged by %s, peak %.2f, window ending %s",
                site,
                rule,
                peak,
                result["window_end"],
            )
            alerts.append(
                {
                    "values": [float(v) for v in readings.tail(200)],
                    "site": site,
                    "sensor": ALERT_SENSOR,
                    "alert_type": ALERT_TYPE,
                    "emails": emails,
                    "phones": phones,
                }
            )
    return alerts

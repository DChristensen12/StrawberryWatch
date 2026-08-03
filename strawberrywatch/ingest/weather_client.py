from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from strawberrywatch.config import Config

logger = logging.getLogger(__name__)

# Only the NWS property that matches a training feature. Dewpoint, humidity,
# wind, and pressure were dropped from the training set (NWS doesn't have rain or solar).
_NWS_PROPERTIES = [
    # (nws_name,    out_name,     convert)
    ("temperature", "air_temp_c", lambda v: v),
]


def fetch_nws_weather(
    start_time,
    end_time,
    station_id: str | None = None,
) -> pd.DataFrame:
    """
    Fetches air temperature from the NWS station over [start_time, end_time].

    Returns a DataFrame with a UTC DatetimeIndex and an air_temp_c column, or
    an empty DataFrame on failure.

    NWS only keeps a ~7-day rolling window for personal stations like LBNL1.
    Requests for longer windows silently return only what's available.
    """
    station = station_id or Config.NWS_STATION_ID
    base_url = f"https://api.weather.gov/stations/{station}/observations"
    headers = {
        "User-Agent": Config.NWS_USER_AGENT,
        "Accept": "application/geo+json",
    }
    params = {
        "start": _ensure_utc_suffix(start_time),
        "end": _ensure_utc_suffix(end_time),
        "limit": 500,
    }

    all_features = []
    next_url = base_url
    page = 0

    while next_url and page < 50:  # hard cap as safety net
        page += 1
        try:
            resp = requests.get(
                next_url,
                headers=headers,
                params=(params if page == 1 else None),
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"NWS fetch failed on page {page}: {e}")
            break

        if resp.status_code != 200:
            logger.error(
                f"NWS API error {resp.status_code} for station {station}: {resp.text[:200]}"
            )
            break

        data = resp.json()
        all_features.extend(data.get("features", []) or [])

        # pagination.next is a string URL when present; defend against null.
        pagi = data.get("pagination") or {}
        next_url = pagi.get("next") if isinstance(pagi, dict) else None
        if not isinstance(next_url, str) or not next_url.startswith("http"):
            next_url = None

    if not all_features:
        logger.warning(f"No NWS observations returned for station {station}.")
        return pd.DataFrame()

    records = []
    for feat in all_features:
        props = feat.get("properties", {}) or {}
        ts = props.get("timestamp")
        if not ts:
            continue

        row = {"datetime": ts}
        for nws_name, out_name, convert in _NWS_PROPERTIES:
            measurement = props.get(nws_name)
            if isinstance(measurement, dict):
                raw_val = measurement.get("value")
            else:
                raw_val = measurement
            row[out_name] = convert(raw_val) if raw_val is not None else None
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for col in [out for _, out, _ in _NWS_PROPERTIES]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any column that's entirely null
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        logger.info(f"NWS: dropping fully-null columns at {station}: {null_cols}")
        df = df.drop(columns=null_cols)

    logger.info(
        f"NWS: fetched {len(df):,} observations from {station} "
        f"({df.index.min()} to {df.index.max()}) "
        f"with features: {list(df.columns)}"
    )
    return df


def _ensure_utc_suffix(t):
    """Make sure a timestamp string is recognizable as UTC by the NWS API."""
    if isinstance(t, datetime):
        s = t.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        s = str(t)
    if not (s.endswith("Z") or "+00" in s or "-00" in s):
        return s + "Z"
    return s

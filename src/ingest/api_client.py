from __future__ import annotations

import logging
from datetime import datetime
from functools import reduce
from typing import List, Optional

import pandas as pd
import requests

from config.config import Config

logger = logging.getLogger(__name__)


# Sensor columns fetched per site. One HTTP call per column because the API
# only honors the last 'vars' param when multiple are given. cond/depth/temp
# are the minimum useful set; batt is there for sensor health monitoring.
_DEFAULT_VARS = [
    "Meter_Hydros21_Cond",
    "Meter_Hydros21_Depth",
    "Meter_Hydros21_Temp",
    "EnviroDIY_Mayfly_Batt",
]


def _fetch_single_var(
    site: str,
    start_str: str,
    end_str: str,
    var_name: str,
    headers: dict,
) -> pd.DataFrame:
    """
    Pulls one (site, sensor) pair from the API. Returns a DataFrame with
    'timestamp' and one sensor column, or empty on any failure.

    One request per sensor because the API only honors the last 'vars' param
    when multiple are given.
    """
    params = [
        ("site", site),
        ("start", start_str),
        ("end", end_str),
        ("vars", var_name),
    ]
    try:
        response = requests.get(
            Config.API_BASE_URL, headers=headers, params=params, timeout=60
        )
    except requests.exceptions.Timeout:
        logger.error(f"[{site}/{var_name}] API request timed out after 60s")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.error(f"[{site}/{var_name}] API request failed: {e}")
        return pd.DataFrame()

    if response.status_code != 200:
        logger.error(
            f"[{site}/{var_name}] API returned {response.status_code}: "
            f"{response.text[:200]}"
        )
        return pd.DataFrame()

    data = response.json()
    if not data:
        # Site doesn't expose this column, or no observations in window.
        # Either way: not an error, just nothing to merge.
        logger.debug(f"[{site}/{var_name}] no rows")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "DateTimeUTC" in df.columns:
        df = df.rename(columns={"DateTimeUTC": "timestamp"})
    if "timestamp" not in df.columns:
        logger.error(f"[{site}/{var_name}] no timestamp in response")
        return pd.DataFrame()

    return df


def fetch_creek_data(
    site: str,
    start_time,
    end_time,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Queries the Strawberry Creek API for one site over [start_time, end_time].

    Makes one HTTP request per sensor column and merges them on timestamp.
    Returns a DataFrame with 'timestamp', 'station_id', and one column per
    available sensor. Sensors the site doesn't expose are just absent, no error.
    """
    headers = {}
    if Config.API_TOKEN:
        headers["Authorization"] = f"Token {Config.API_TOKEN}"

    start_str = (
        start_time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(start_time, datetime) else str(start_time)
    )
    end_str = (
        end_time.strftime("%Y-%m-%dT%H:%M:%S")
        if isinstance(end_time, datetime) else str(end_time)
    )

    vars_to_request = variables if variables else _DEFAULT_VARS

    frames = []
    for var_name in vars_to_request:
        df = _fetch_single_var(site, start_str, end_str, var_name, headers)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.info(f"[{site}] no data for any requested variable in window")
        return pd.DataFrame()

    merged = reduce(
        lambda left, right: pd.merge(left, right, on="timestamp", how="outer"),
        frames,
    )

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = (
        merged.dropna(subset=["timestamp"])
              .sort_values("timestamp")
              .reset_index(drop=True)
    )
    merged["station_id"] = site

    logger.info(
        f"[{site}] fetched {len(merged):,} rows × {len(merged.columns) - 2} sensors"
    )
    return merged


def fetch_network_snapshot(start_time, end_time) -> pd.DataFrame:
    """
    Pulls data for every site in Config.LOCATIONS and concatenates them.
    Same shape as sql_client.fetch_network_snapshot_sql.
    """
    frames = []
    for site in Config.LOCATIONS:
        print(f"Requesting data: {site}...")
        df_site = fetch_creek_data(site, start_time, end_time)
        if not df_site.empty:
            frames.append(df_site)

    if not frames:
        print("No data retrieved for any site.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

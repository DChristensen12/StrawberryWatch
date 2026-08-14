from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Berkeley coordinates, central to the SCMG sites (all within ~1.5 miles).
# Berkeley weather is uniform enough at this scale that one point is fine.
_BERKELEY_LAT = 37.873
_BERKELEY_LON = -122.260

# Historical Forecast API, not the archive/ERA5 endpoint. ERA5 is 0.25 degree
# (~25km) reanalysis, too coarse for a watershed that fits inside one cell and
# sits in coastal/mountainous terrain Open-Meteo's own docs warn ERA5 misses.
# This one is ~1km resolution with native 15-minutely data for North America.
_OPEN_METEO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Maps Open-Meteo parameter names to our internal column names.
# Units: precipitation in mm/hour, shortwave in W/m², temperature in °C.
_OPEN_METEO_VARIABLES = [
    ("temperature_2m", "air_temp_c"),
    ("precipitation", "rain_mm"),
    ("shortwave_radiation", "shortwave_radiation"),
]


def fetch_open_meteo_weather(
    start_time,
    end_time,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """
    Fetches native 15-minute weather from Open-Meteo's Historical Forecast API
    for the given date range.

    Returns a DataFrame with a UTC DatetimeIndex and the same columns as
    fetch_nws_weather (air_temp_c, rain_mm, shortwave_radiation), so data_loader
    can swap sources without knowing which one ran. rain_mm is the actual mm
    that fell in that 15-min window, not an hourly accumulation, no
    disaggregation needed downstream.

    start_time/end_time can be datetime objects or ISO-8601 strings.
    Latitude/longitude default to central Berkeley if not given.
    """
    lat = latitude if latitude is not None else _BERKELEY_LAT
    lon = longitude if longitude is not None else _BERKELEY_LON

    # Open-Meteo expects start_date and end_date as YYYY-MM-DD strings.
    if isinstance(start_time, datetime):
        start_date = start_time.strftime("%Y-%m-%d")
    else:
        start_date = str(start_time)[:10]
    if isinstance(end_time, datetime):
        end_date = end_time.strftime("%Y-%m-%d")
    else:
        end_date = str(end_time)[:10]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "minutely_15": ",".join(om_name for om_name, _ in _OPEN_METEO_VARIABLES),
        "timezone": "UTC",
    }

    try:
        resp = requests.get(_OPEN_METEO_URL, params=params, timeout=60)
    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo request failed: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        logger.error(f"Open-Meteo returned {resp.status_code}: {resp.text[:300]}")
        return pd.DataFrame()

    data = resp.json()
    quarter_hourly = data.get("minutely_15", {})
    times = quarter_hourly.get("time", [])
    if not times:
        logger.warning(f"Open-Meteo returned no observations for {start_date} to {end_date}")
        return pd.DataFrame()

    # Each variable is a parallel array indexed by 'time'.
    rows = {"datetime": times}
    for om_name, our_name in _OPEN_METEO_VARIABLES:
        rows[our_name] = quarter_hourly.get(om_name, [None] * len(times))

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for _, col in _OPEN_METEO_VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fully-null columns (rare but possible if Open-Meteo lacks coverage)
    null_cols = [c for c in df.columns if df[c].isna().all()]
    if null_cols:
        logger.info(f"Open-Meteo: dropping fully-null columns: {null_cols}")
        df = df.drop(columns=null_cols)

    logger.info(
        f"Open-Meteo: fetched {len(df):,} 15-min observations "
        f"({df.index.min()} to {df.index.max()}) "
        f"for ({lat}, {lon}) "
        f"with features: {list(df.columns)}"
    )
    return df

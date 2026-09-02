"""
Cached Open-Meteo pulls over long spans, chunked by month.

fetch_open_meteo_weather hits the network every call, which is fine for a few
days and not fine for the eleven months a training run wants.
"""

from __future__ import annotations

import logging

import pandas as pd

from strawberrywatch import paths
from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

logger = logging.getLogger(__name__)

COLUMNS = ["air_temp_c", "rain_mm", "shortwave_radiation"]


def _month_starts(start, end):
    """Month boundaries covering [start, end], as (first day, last day) pairs."""
    months = pd.date_range(start.normalize().replace(day=1), end.normalize(), freq="MS", tz="UTC")
    for m in months:
        last = (m + pd.offsets.MonthEnd(1)).normalize()
        yield m, min(last, end.normalize())


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _cache_path(lo, hi, cache_dir):
    return cache_dir / f"rain_{lo:%Y-%m-%d}_{hi:%Y-%m-%d}.csv"


def _read_chunk(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.set_index("datetime").sort_index()


def load_weather(start, end, cache_dir=None, allow_fetch=True):
    """
    15-min air_temp_c/rain_mm/shortwave_radiation over [start, end], UTC.

    Whole months land in data/rain_cache/ under the rain_{start}_{end}.csv name
    the rest of the repo already reads. A partial trailing month is fetched but
    not cached: writing it would leave a file that looks like a full month and
    is silently short the moment the archive grows.
    """
    cache_dir = cache_dir or paths.rain_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    start, end = _utc(start), _utc(end)

    chunks, hits, misses = [], 0, 0
    for lo, hi in _month_starts(start, end):
        path = _cache_path(lo, hi, cache_dir)
        if path.exists():
            chunks.append(_read_chunk(path))
            hits += 1
            continue
        if not allow_fetch:
            raise RuntimeError(f"no cached weather at {path} and fetching is off")
        got = fetch_open_meteo_weather(lo, hi)
        if got.empty:
            raise RuntimeError(f"Open-Meteo returned nothing for {lo:%Y-%m-%d}..{hi:%Y-%m-%d}")
        misses += 1
        # Only whole months get cached. A partial month written under the same
        # name would read back later as complete and be silently short.
        if hi == (lo + pd.offsets.MonthEnd(1)).normalize():
            got.to_csv(path)
        chunks.append(got)

    weather = pd.concat(chunks)
    weather = weather[~weather.index.duplicated(keep="first")].sort_index()
    weather = weather.loc[(weather.index >= start) & (weather.index <= end)]
    logger.info("weather: %d cached months, %d fetched, %d rows", hits, misses, len(weather))
    return weather.reindex(columns=COLUMNS)

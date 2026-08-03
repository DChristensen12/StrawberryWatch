"""
Rebuilds one site's fixture CSV inside an anomaly-event folder from data/raw_data/.

This exists because oxford.csv and university_house.csv in
anomaly_2026_01_botanical_actuator turned out to hold the same data (one
overwrote the other when the folder was built). Use this to regenerate a
corrupted site file from the raw archive instead of hand-patching it.

How the fixtures actually get built (confirmed by matching raw_data against 7 of
oxford's other, uncorrupted fixture files row-for-row): take that site's raw_data,
filter to the event's date window, and write it out in whatever order raw_data
already has it in. No sorting, no snapping to a shared grid. Sites like north_fork_0
just happen to have raw_data that's already chronological, so their fixtures look
sorted, but that's a side effect, not a rule.

Usage:
    python scripts/rebuild_fixture.py --folder data/anomalies/anomaly_2026_01_botanical_actuator --site oxford --output /tmp/oxford_rebuilt.csv
"""

import argparse
import glob
import os

import pandas as pd

from strawberrywatch import paths

RAW_DATA_DIR = paths.raw_data_dir()
RAIN_CACHE_DIR = paths.rain_cache_dir()

FIXTURE_COLUMNS = [
    "DateTimeUTC",
    "Meter_Hydros21_Cond",
    "Meter_Hydros21_Depth",
    "Meter_Hydros21_Temp",
    "air_temp_c",
    "rain_mm",
    "shortwave_radiation",
]

# site name -> raw_data filename, only listed where it doesn't match "{site}.csv"
_RAW_FILENAME_OVERRIDES = {
    "footbridge": "scnf010.csv",
    "university_house": "university_house_1778210630544.csv",
}

_TIME_CANDIDATES = ["DateTimeUTC", "timestamp", "datetime"]

# same cadence the weather merge in data_loader.py assumes for the 15-min grid
_ROWS_PER_HOUR = 4


def _raw_path_for_site(site):
    fname = _RAW_FILENAME_OVERRIDES.get(site, f"{site}.csv")
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no raw_data file for site '{site}' (looked for {path})")
    return path


def _read_flexible_csv(path):
    """
    Raw files aren't consistent about delimiter. Sniff first with sep=None; if
    that collapses everything into one fused column (its name has a tab or
    comma baked into it), the sniffer picked wrong, so retry with tab then comma.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if len(df.columns) == 1 and ("\t" in df.columns[0] or "," in df.columns[0]):
        for sep in ["\t", ","]:
            retry = pd.read_csv(path, sep=sep)
            if len(retry.columns) > 1:
                return retry
    return df


def _load_raw_site(site):
    """
    Loads one site's raw sensor data in its native on-disk row order (not sorted,
    not deduped beyond exact-timestamp collisions), with a DateTimeUTC column.
    """
    path = _raw_path_for_site(site)
    df = _read_flexible_csv(path)

    time_col = next((c for c in _TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"{path}: no recognizable timestamp column among {df.columns.tolist()}")

    df["DateTimeUTC"] = pd.to_datetime(df[time_col], utc=True)
    df = df.drop_duplicates(subset=["DateTimeUTC"], keep="first")

    sensor_cols = ["Meter_Hydros21_Cond", "Meter_Hydros21_Depth", "Meter_Hydros21_Temp"]
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing expected sensor columns {missing}")

    return df[["DateTimeUTC"] + sensor_cols]


def _event_window(folder, exclude_sites):
    """
    The event window is the min/max DateTimeUTC across every other CSV already
    in the folder (the "clean" siblings). exclude_sites can list more than just
    the target site itself, e.g. when a second file in the same folder is also
    under suspicion and shouldn't be trusted to define the window either.
    """
    if isinstance(exclude_sites, str):
        exclude_sites = {exclude_sites}
    else:
        exclude_sites = set(exclude_sites)

    candidates = [
        f
        for f in glob.glob(os.path.join(folder, "*.csv"))
        if os.path.splitext(os.path.basename(f))[0] not in exclude_sites
    ]
    if not candidates:
        raise FileNotFoundError(f"no other CSV in {folder} to derive the event window from")

    starts, ends = [], []
    for f in candidates:
        df = pd.read_csv(f, usecols=["DateTimeUTC"])
        ts = pd.to_datetime(df["DateTimeUTC"], utc=True)
        starts.append(ts.min())
        ends.append(ts.max())

    start, end = min(starts), max(ends)
    print(f"  event window from {len(candidates)} sibling files: {start} to {end}")
    return start, end


def _rain_cache_path(start, end):
    return os.path.join(RAIN_CACHE_DIR, f"rain_{start.date()}_{end.date()}.csv")


def _find_covering_rain_cache(start, end):
    """Returns a cached rain_mm series covering [start, end], or None if nothing covers it."""
    if not os.path.isdir(RAIN_CACHE_DIR):
        return None
    for fname in sorted(os.listdir(RAIN_CACHE_DIR)):
        if not (fname.startswith("rain_") and fname.endswith(".csv")):
            continue
        try:
            _, s, e = fname[:-4].split("_")
            cache_start = pd.Timestamp(s, tz="UTC")
            cache_end = pd.Timestamp(e, tz="UTC") + pd.Timedelta(days=1)
        except ValueError:
            continue
        if cache_start <= start and cache_end >= end:
            path = os.path.join(RAIN_CACHE_DIR, fname)
            cached = pd.read_csv(path)
            cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True)
            print(f"  rain from cache: {fname}")
            return cached.set_index("datetime")["rain_mm"]
    return None


def _get_weather_hourly(start, end):
    """
    Returns hourly air_temp_c, rain_mm, shortwave_radiation for [start, end].
    Rain reuses data/rain_cache/ when a cache file already covers the range.
    air_temp_c/shortwave_radiation have no cache of their own, so they always
    come straight from Open-Meteo. If rain wasn't cached, this fetch supplies
    it too and gets written to rain_cache under the usual rain_{start}_{end}.csv name.
    """
    from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

    cached_rain = _find_covering_rain_cache(start, end)

    print(f"  fetching Open-Meteo weather {start.date()} to {end.date()}...")
    weather = fetch_open_meteo_weather(start, end)
    if weather.empty:
        raise RuntimeError("Open-Meteo returned no weather data for this range")

    if cached_rain is not None:
        weather = weather.drop(columns=["rain_mm"]).join(cached_rain, how="left")
    else:
        os.makedirs(RAIN_CACHE_DIR, exist_ok=True)
        cpath = _rain_cache_path(start, end)
        weather[["rain_mm"]].reset_index().rename(columns={"index": "datetime"}).to_csv(
            cpath, index=False
        )
        print(
            f"  no cache covered this range, fetched rain fresh and cached to {os.path.basename(cpath)}"
        )

    return weather[["air_temp_c", "rain_mm", "shortwave_radiation"]]


def rebuild(site, folder, output_path, exclude_from_window=None):
    print(f"rebuilding '{site}' for {folder}")

    exclude_sites = {site} | set(exclude_from_window or [])
    start, end = _event_window(folder, exclude_sites=exclude_sites)

    raw = _load_raw_site(site)
    in_window = raw[(raw["DateTimeUTC"] >= start) & (raw["DateTimeUTC"] <= end)].copy()
    print(
        f"  {len(in_window)}/{len(raw)} raw rows fall in the event window, kept in raw_data's native order"
    )

    weather_hourly = _get_weather_hourly(start, end).copy()
    # accumulated rain gets split evenly across the sub-hourly rows it covers,
    # same disaggregation data_loader._merge_weather does
    weather_hourly["rain_mm"] = weather_hourly["rain_mm"] / _ROWS_PER_HOUR

    # merge preserves in_window's row order (its native, unsorted order), it
    # only ever adds columns onto the left side
    in_window["_hour_key"] = in_window["DateTimeUTC"].dt.floor("h")
    out = in_window.merge(weather_hourly, how="left", left_on="_hour_key", right_index=True)
    out = out.drop(columns=["_hour_key"])

    out["DateTimeUTC"] = out["DateTimeUTC"].dt.tz_localize(None)
    out = out[FIXTURE_COLUMNS]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  wrote {len(out)} rows to {output_path}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild one site's fixture CSV from data/raw_data/"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="event folder, e.g. data/anomalies/anomaly_2026_01_botanical_actuator",
    )
    parser.add_argument("--site", required=True, help="site name, e.g. oxford")
    parser.add_argument("--output", required=True, help="path to write the rebuilt CSV")
    parser.add_argument(
        "--exclude-from-window",
        nargs="*",
        default=None,
        help="extra site names to exclude from event-window derivation, "
        "beyond --site itself (e.g. another suspect file in the same folder)",
    )
    args = parser.parse_args()
    rebuild(args.site, args.folder, args.output, exclude_from_window=args.exclude_from_window)


if __name__ == "__main__":
    main()

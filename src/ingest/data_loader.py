import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from config.config import Config
from src.ingest.api_client import fetch_network_snapshot


# Numeric columns that are not creek measurements and should never be model features.
# EnviroDIY_Mayfly_Batt is sensor health telemetry; 'valid' is the QC mask (becomes
# node_mask downstream, not a feature); the others are pandas/CSV artifacts.
_NON_FEATURE_COLUMNS = {
    "Unnamed: 0",
    "delta",
    "Precip_Max",
    "EnviroDIY_Mayfly_Batt",
    "valid",
}

# Accumulated (per-interval sum) weather columns, not instantaneous. Precip
# gets summed when collapsing duplicate timestamps onto the 15-min grid.
# Temperature and solar are instantaneous and get averaged instead.
_ACCUMULATED_WEATHER_COLUMNS = {
    "rain_mm",
}


def _is_wide_training_corpus(file_path):
    """True if file_path looks like scripts/build_training_corpus.py's output:
    one row per timestep with a {site}_valid column per node, instead of the
    long one-row-per-(datetime, location) format everything else here uses."""
    if not os.path.exists(file_path):
        return False
    header = pd.read_csv(file_path, nrows=0).columns
    return any(c.endswith("_valid") for c in header)


def _load_wide_corpus_as_long(file_path):
    """
    Converts a wide training corpus into the long (datetime, location) format
    the rest of this pipeline expects, carrying each site's {site}_valid flag
    over as a plain 'valid' column. hour_sin/cos/dayofyear_sin/cos aren't
    carried over -- they get regenerated below from the index either way, so
    keeping the corpus's copy would just create duplicate columns.
    """
    wide = pd.read_csv(file_path)
    wide["datetime"] = pd.to_datetime(wide["datetime"], utc=True)
    wide = wide.set_index("datetime")

    valid_cols = [c for c in wide.columns if c.endswith("_valid")]
    sites = [c[: -len("_valid")] for c in valid_cols]
    shared_cols = [c for c in ("rain_mm", "air_temp_c", "shortwave_radiation") if c in wide.columns]

    frames = []
    for site in sites:
        prefix = f"{site}_"
        rename = {
            c: c[len(prefix):] for c in wide.columns
            if c.startswith(prefix) and not c.endswith("_valid")
        }
        site_df = wide[list(rename.keys()) + [f"{site}_valid"] + shared_cols].rename(columns=rename)
        site_df = site_df.rename(columns={f"{site}_valid": "valid"})
        site_df["location"] = site
        frames.append(site_df)

    long_df = pd.concat(frames).sort_index()
    print(
        f"wide corpus converted: {len(sites)} sites x {len(wide):,} timesteps "
        f"-> {len(long_df):,} long rows"
    )
    return long_df


def load_and_preprocess_data(
    file_path=Config.DATA_FILE,
    force_download=False,
    days=30,
    data_source="api",
):
    """
    Loads creek data from the cache CSV, fetching fresh if the cache is missing
    or force_download=True.

    Cache works as a rolling window: new data is merged in and deduplicated on
    (datetime, location), then trimmed to Config.ROLLING_WINDOW_DAYS. Stays
    bounded in size while remaining useful for debugging and offline reruns.

    data_source: "api" pulls from the REST API (3 sensor features).
                 "sql" pulls from production MySQL (richer).

    If file_path is a wide training-corpus CSV (has {site}_valid columns, see
    scripts/build_training_corpus.py), this skips the fetch/merge/cache path
    entirely and loads it directly -- it's a static, already-QC'd file, not a
    rolling cache to refresh. force_download is ignored in that case.
    """
    if _is_wide_training_corpus(file_path):
        print(f"'{file_path}' is a wide training corpus (has *_valid columns); "
              f"loading directly, skipping fetch/merge.")
        df = _load_wide_corpus_as_long(file_path)
    else:
        if not os.path.exists(file_path) or force_download:
            source_label = "API" if data_source == "api" else "SQL database"
            if force_download:
                print(f"Refresh requested. Fetching last {days} days from {source_label}...")
            else:
                print(f"Local file '{file_path}' not found. Pulling from {source_label}...")

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)

            if data_source == "sql":
                from src.ingest.sql_client import fetch_network_snapshot_sql
                df_raw = fetch_network_snapshot_sql(
                    start_time=start_date.isoformat(),
                    end_time=end_date.isoformat(),
                )
            else:
                df_raw = fetch_network_snapshot(
                    start_time=start_date.isoformat(),
                    end_time=end_date.isoformat(),
                )

            if df_raw.empty:
                print(f"no data from {source_label}.")
                if not os.path.exists(file_path):
                    sys.exit(1)
                print("Falling back to existing local file.")
            else:
                # Rename raw MMW column names to internal ones. API exposes 3 features
                # (cond, depth, temp) plus battery; SQL may have more. Extra columns pass
                # through with their raw names and get picked up as model features.
                column_mapping = {
                    "Meter_Hydros21_Cond":  "conductivity",
                    "Meter_Hydros21_Depth": "depth",
                    "Meter_Hydros21_Temp":  "temperature",
                    "timestamp":            "datetime",
                    "station_id":           "location",
                }
                df_raw = df_raw.rename(columns=column_mapping)

                if Config.USE_NWS_WEATHER:
                    df_raw = _merge_weather(df_raw, start_date, end_date, days)

                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                combined = _merge_with_cache(df_raw, file_path)
                combined = _trim_to_rolling_window(combined, Config.ROLLING_WINDOW_DAYS)
                combined.to_csv(file_path, index=False)
                print(
                    f"Cache updated: {len(combined):,} rows on disk "
                    f"({combined['datetime'].min()} to {combined['datetime'].max()})"
                )

        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()

    print(f"data loaded:")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Range: {df.index.min()} to {df.index.max()}")

    # Missing sites are normal (sensors go down), but worth surfacing.
    sites_present = sorted(df["location"].unique()) if "location" in df.columns else []
    expected = set(Config.LOCATIONS)
    missing = sorted(expected - set(sites_present))
    print(f"Sites present ({len(sites_present)}): {sites_present}")
    if missing:
        print(f"Sites missing (offline this window): {missing}")

    # Feature selection: every numeric column that isn't location or in our
    # exclude set. New columns from SQL or NWS get picked up automatically.
    feature_cols = [
        col for col in df.columns
        if col != "location"
        and col not in _NON_FEATURE_COLUMNS
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        print("no numeric feature columns found in dataset.")
        sys.exit(1)

    carry_cols = [c for c in ("valid",) if c in df.columns]
    df_featured = df[["location"] + feature_cols + carry_cols].copy()

    # Sin/cos pairs for hour and day-of-year. Raw numbers make 11pm and midnight
    # look far apart; circular encoding keeps the wrap smooth. Never NaN since
    # they come from the index, not sensors.
    idx = df_featured.index
    hour_angle = 2 * np.pi * (idx.hour + idx.minute / 60.0) / 24.0
    doy_angle = 2 * np.pi * (idx.dayofyear - 1) / 365.0
    df_featured["hour_sin"] = np.sin(hour_angle)
    df_featured["hour_cos"] = np.cos(hour_angle)
    df_featured["dayofyear_sin"] = np.sin(doy_angle)
    df_featured["dayofyear_cos"] = np.cos(doy_angle)

    time_feature_cols = ["hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos"]
    feature_cols = feature_cols + time_feature_cols

    print(f"Active features ({len(feature_cols)}): {', '.join(feature_cols)}")
    print(f"---------------------------\n")

    return df_featured, df, Config.LOCATIONS


def _merge_with_cache(df_new, file_path):
    """
    Combines freshly-fetched data with whatever's already in the cache file,
    deduplicating on (datetime, location). Newer rows win on conflict, so
    late-arriving backfills can correct earlier values.
    """
    if not os.path.exists(file_path):
        return df_new

    try:
        existing = pd.read_csv(file_path)
        existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
    except Exception as e:
        # Cache file is corrupted or has an incompatible schema. Start fresh.
        print(f"couldn't read cache ({e}), starting fresh.")
        return df_new

    # Union of columns handles schema drift (e.g. a new weather feature).
    # Missing values become NaN, pandas handles it.
    combined = pd.concat([existing, df_new], ignore_index=True)
    combined["datetime"] = pd.to_datetime(combined["datetime"], utc=True)

    # Dedupe: keep the most recent version of each (datetime, location) row.
    # 'keep="last"' relies on df_new having been concatenated after existing.
    combined = combined.drop_duplicates(
        subset=["datetime", "location"], keep="last"
    )
    combined = combined.sort_values("datetime").reset_index(drop=True)
    return combined


def _trim_to_rolling_window(df, window_days):
    """
    Keeps the last `window_days` of data. Anchored on the dataset's latest
    timestamp (not 'now'), so the cache stays useful even when fetches happen
    well after the last real observation.
    """
    if df.empty or window_days is None or window_days <= 0:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    cutoff = df["datetime"].max() - pd.Timedelta(days=window_days)
    trimmed = df[df["datetime"] >= cutoff]

    dropped = len(df) - len(trimmed)
    if dropped > 0:
        print(f"Trimmed {dropped:,} rows older than {window_days} days.")
    return trimmed


def _merge_weather(df_raw, start_date, end_date, days):
    """
    Merges Open-Meteo weather onto df_raw.

    Both training and live inference go through this one source now (used to
    route recent windows to NWS LBNL1 instead) so the model isn't trained on
    one weather distribution and scored against another. Open-Meteo's
    Historical Forecast API blends forecast-model data for the last day or two
    with historical data further back, so it covers both windows fine.

    Native 15-min data, same cadence as the creek grid, so there's no more
    hourly accumulation to split across sub-hourly rows -- resampling to 15min
    here is only to collapse duplicate timestamps if a source ever reports
    more than once per window, not to disaggregate anything.
    """
    from src.ingest.historical_weather_client import fetch_open_meteo_weather
    print(f"Fetching Open-Meteo weather (window: {days}d, source: Historical Forecast API, 15-min native)...")
    df_weather = fetch_open_meteo_weather(start_date, end_date)

    if df_weather.empty:
        print("Weather data unavailable, proceeding without weather features.")
        return df_raw

    accumulated_present = [c for c in df_weather.columns if c in _ACCUMULATED_WEATHER_COLUMNS]
    instantaneous_present = [c for c in df_weather.columns if c not in _ACCUMULATED_WEATHER_COLUMNS]

    parts = []
    if instantaneous_present:
        parts.append(df_weather[instantaneous_present].resample("15min").mean())
    if accumulated_present:
        parts.append(df_weather[accumulated_present].resample("15min").sum())
    df_weather_15min = pd.concat(parts, axis=1) if parts else df_weather.resample("15min").mean()
    # Keep a stable column order matching the source
    df_weather_15min = df_weather_15min[[c for c in df_weather.columns if c in df_weather_15min.columns]]

    creek_dt = pd.to_datetime(df_raw["datetime"], utc=True)
    df_raw["_quarter_key"] = creek_dt.dt.floor("15min")

    df_merged = df_raw.merge(
        df_weather_15min,
        how="left",
        left_on="_quarter_key",
        right_index=True,
    ).drop(columns=["_quarter_key"])

    n_with_weather = (
        df_merged["air_temp_c"].notna().sum()
        if "air_temp_c" in df_merged.columns else 0
    )
    print(
        f"Weather merged: {n_with_weather:,}/{len(df_merged):,} rows "
        f"have weather context "
        f"(features: {list(df_weather_15min.columns)})"
    )
    return df_merged

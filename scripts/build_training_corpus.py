"""
Builds data/processed_data/training_corpus.csv from data/raw_data/, with QC
applied. Unlike full_creek_gnn.csv (long format: one row per site per
timestep), this is WIDE: one row per 15-minute grid step, with per-site
sensor columns plus a per-site {site}_valid flag. That shape is what lets a
bad node get masked out for a timestep while its neighbors keep their data
in the same row -- Part 3 turns these _valid columns directly into the model's
node_mask.

Does not touch full_creek_gnn.csv, which stays the rolling 90-day inference
cache.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from strawberrywatch.config import Config
from strawberrywatch import paths
from strawberrywatch.ingest.raw_data_loader import load_all_raw_sites
from strawberrywatch.ingest.quality_control import (
    inter_site_duplicate_check,
    sentinel_check,
    repeated_streak_check,
)
from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

ANOMALIES_DIR = os.path.join("data", "anomalies")
RAIN_CACHE_DIR = paths.rain_cache_dir()
OUTPUT_PATH = os.path.join("data", "processed_data", "training_corpus.csv")

GRID_FREQ = "15min"
ANOMALY_PAD = pd.Timedelta(hours=12)

# This event is being dropped from the anomaly catalog; its window is ordinary
# wet-season data for the graph nodes and must stay in the corpus.
_KEEP_IN_TRAINING = "anomaly_2026_01_botanical_actuator"

SENSOR_COLS = ["conductivity", "depth", "temperature"]


# Anomaly windows


def _anomaly_windows():
    """
    One (folder_name, start, end) per subfolder in data/anomalies/, padded
    12h each side, derived from the min/max timestamp across that folder's
    CSVs. Skips _KEEP_IN_TRAINING.
    """
    windows = []
    for folder in sorted(glob.glob(os.path.join(ANOMALIES_DIR, "*"))):
        if not os.path.isdir(folder):
            continue
        name = os.path.basename(folder)
        if name == _KEEP_IN_TRAINING:
            continue

        starts, ends = [], []
        for f in glob.glob(os.path.join(folder, "*.csv")):
            df = pd.read_csv(f, usecols=lambda c: c in ("DateTimeUTC", "timestamp", "datetime"))
            tcol = next(
                (c for c in ["DateTimeUTC", "timestamp", "datetime"] if c in df.columns), None
            )
            if tcol is None:
                continue
            ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dropna()
            if ts.empty:
                continue
            starts.append(ts.min())
            ends.append(ts.max())

        if not starts:
            continue
        start = min(starts) - ANOMALY_PAD
        end = max(ends) + ANOMALY_PAD
        windows.append((name, start, end))

    return windows


# Grid + per-site resampling


def _build_grid(raw_sites, node_names):
    """
    Shared 15-min grid spans from when the LAST graph node comes online to
    the latest timestamp any graph node reports. Footbridge has a sparse
    trickle of readings back to 2021 (long before the rest of the network
    existed); starting the grid there would be years of rows where nearly
    every node has nothing to say. Starting at the latest "first reading"
    across the roster keeps the corpus to the period the graph actually ran
    as a system.
    """
    starts = [raw_sites[n].index.min() for n in node_names if not raw_sites[n].empty]
    ends = [raw_sites[n].index.max() for n in node_names if not raw_sites[n].empty]
    grid_start = max(starts).floor(GRID_FREQ)
    grid_end = max(ends).ceil(GRID_FREQ)
    return pd.date_range(grid_start, grid_end, freq=GRID_FREQ, tz="UTC")


def _raw_invalid_mask(site_df, site_name):
    """sentinel OR streak flags on the RAW (pre-resample) data for one site."""
    tagged = site_df.copy()
    tagged["location"] = site_name
    cols = [c for c in SENSOR_COLS if c in tagged.columns]
    return sentinel_check(tagged, sensor_cols=cols) | repeated_streak_check(
        tagged, sensor_cols=cols
    )


def _resample_site(site_df, site_name, grid):
    """
    Returns (values_df, valid_series) both reindexed onto grid. values_df has
    {site}_conductivity/depth/temperature. valid_series is True by default;
    the raw-level sentinel/streak check is resampled with .max() so a single
    bad raw reading inside a 15-min bin is enough to invalidate that bin.
    """
    cols = [c for c in SENSOR_COLS if c in site_df.columns]
    values = site_df[cols].resample(GRID_FREQ).mean().reindex(grid)
    values.columns = [f"{site_name}_{c}" for c in values.columns]

    raw_invalid = _raw_invalid_mask(site_df, site_name)
    grid_invalid = raw_invalid.resample(GRID_FREQ).max().reindex(grid).fillna(False).astype(bool)
    valid = ~grid_invalid
    valid.name = f"{site_name}_valid"
    return values, valid


# Weather


def _load_rain_cache():
    """Concatenates every data/rain_cache/*.csv into one 15-min rain_mm series."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RAIN_CACHE_DIR, "*.csv"))):
        cached = pd.read_csv(path)
        cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True)
        frames.append(cached.set_index("datetime")["rain_mm"])
    if not frames:
        return pd.Series(dtype=float)
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def _merge_weather(grid):
    """
    Native 15-min air_temp_c/rain_mm/shortwave_radiation for the grid's span,
    from Open-Meteo's Historical Forecast API. rain_mm prefers data/rain_cache/
    where a cached 15-min step exists, falls back to the fresh fetch otherwise.
    Source cadence already matches the grid, so there's nothing to disaggregate.
    """
    print(f"fetching Open-Meteo weather {grid.min().date()} to {grid.max().date()}...")
    weather = fetch_open_meteo_weather(grid.min(), grid.max())
    if weather.empty:
        raise RuntimeError("Open-Meteo returned no weather data for this range")

    cached_rain = _load_rain_cache()
    n_from_cache = 0
    if not cached_rain.empty:
        overlap = weather.index.intersection(cached_rain.index)
        n_from_cache = len(overlap)
        weather.loc[overlap, "rain_mm"] = cached_rain.loc[overlap]
    print(
        f"  rain_mm: {n_from_cache:,}/{len(weather):,} 15-min steps from rain_cache, rest from Open-Meteo"
    )

    quarter_key = pd.Series(grid, index=grid).dt.floor("15min")
    merged = pd.DataFrame(index=grid)
    merged["_quarter_key"] = quarter_key
    merged = merged.join(weather, on="_quarter_key").drop(columns=["_quarter_key"])
    return merged


# Main build


def main():
    node_names = list(Config.LOCATION_TO_IDX.keys())
    print(f"graph nodes: {node_names}\n")

    raw_data_dir = paths.raw_data_dir()
    print(f"loading raw data from {raw_data_dir}...")
    all_raw = load_all_raw_sites(raw_data_dir)
    missing = [n for n in node_names if n not in all_raw]
    if missing:
        raise RuntimeError(f"no raw_data file found for graph node(s): {missing}")

    # --- oxford/university_house duplication range, from Part 1's checker ---
    if "university_house" in all_raw:
        dup_df = pd.concat(
            [
                all_raw["oxford"].assign(location="oxford"),
                all_raw["university_house"].assign(location="university_house"),
            ]
        )
        dup_pairs = inter_site_duplicate_check(dup_df, sensor_col="conductivity", return_all=True)
        oxford_dup_ranges = []
        for pair in dup_pairs:
            if {pair["site_a"], pair["site_b"]} == {"oxford", "university_house"}:
                oxford_dup_ranges = pair["ranges"]
                print(
                    f"oxford/university_house duplication: {pair['n_matched']}/{pair['n_coincident']} "
                    f"match ({pair['match_frac']:.1%}), flagged={pair['flagged']}"
                )
                for start, end in oxford_dup_ranges:
                    print(f"  {start} -> {end}")
    else:
        oxford_dup_ranges = []
    print()

    # --- shared grid + per-node resample ---
    grid = _build_grid(all_raw, node_names)
    print(f"shared grid: {grid.min()} -> {grid.max()} ({len(grid):,} steps @ {GRID_FREQ})\n")

    value_frames = []
    valid_series = {}
    for site in node_names:
        values, valid = _resample_site(all_raw[site], site, grid)
        value_frames.append(values)
        valid_series[site] = valid

    corpus = pd.concat(value_frames, axis=1)
    for site, valid in valid_series.items():
        corpus[f"{site}_valid"] = valid

    # oxford invalid for its duplicated stretch (a. runs before sentinel/streak
    # per the spec order, but all three land on the same column so order
    # only matters for the print-out, not the result)
    for start, end in oxford_dup_ranges:
        in_range = (corpus.index >= start) & (corpus.index <= end)
        corpus.loc[in_range, "oxford_valid"] = False

    print("per-node invalid timesteps (sentinel + streak + oxford/university_house duplication):")
    for site in node_names:
        n_invalid = int((~corpus[f"{site}_valid"]).sum())
        pct = 100 * n_invalid / len(corpus)
        print(f"  {site}: {n_invalid:,} / {len(corpus):,} ({pct:.2f}%)")
    print()

    # --- exclusion (a): anomaly windows, full-row removal ---
    windows = _anomaly_windows()
    print(f"anomaly exclusion windows ({len(windows)}, {_KEEP_IN_TRAINING} kept in):")
    total_removed = 0
    keep_mask = pd.Series(True, index=corpus.index)
    for name, start, end in windows:
        in_window = (corpus.index >= start) & (corpus.index <= end)
        n_removed = int((keep_mask & in_window).sum())
        total_removed += n_removed
        keep_mask &= ~in_window
        print(f"  {name}: {start} -> {end}  ({n_removed:,} rows)")
    print(f"  total removed: {total_removed:,}\n")

    corpus = corpus[keep_mask]

    # --- weather ---
    weather = _merge_weather(corpus.index)
    corpus = corpus.join(weather)
    n_with_weather = corpus["air_temp_c"].notna().sum()
    print(f"weather merged: {n_with_weather:,}/{len(corpus):,} rows have weather\n")

    # --- time encoding, exact formulation from data_loader.py ---
    idx = corpus.index
    hour_angle = 2 * np.pi * (idx.hour + idx.minute / 60.0) / 24.0
    doy_angle = 2 * np.pi * (idx.dayofyear - 1) / 365.0
    corpus["hour_sin"] = np.sin(hour_angle)
    corpus["hour_cos"] = np.cos(hour_angle)
    corpus["dayofyear_sin"] = np.sin(doy_angle)
    corpus["dayofyear_cos"] = np.cos(doy_angle)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    corpus.to_csv(OUTPUT_PATH, index_label="datetime")
    print(f"wrote {len(corpus):,} rows to {OUTPUT_PATH}\n")

    _print_summary(corpus, node_names)


def _print_summary(corpus, node_names):
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"date range: {corpus.index.min()} -> {corpus.index.max()}")
    print(f"rows: {len(corpus):,}\n")

    print("rows with real (non-NaN) sensor data, and invalid %, per site:")
    for site in node_names:
        cond_col = f"{site}_conductivity"
        n_real = int(corpus[cond_col].notna().sum()) if cond_col in corpus else 0
        n_invalid = int((~corpus[f"{site}_valid"]).sum())
        pct_invalid = 100 * n_invalid / len(corpus)
        print(
            f"  {site:15s} real={n_real:>6,}/{len(corpus):,}   "
            f"invalid={n_invalid:>6,} ({pct_invalid:5.2f}%)"
        )
    print()

    total_rain = corpus["rain_mm"].sum()
    print(f"total rain_mm: {total_rain:,.1f}")

    rolling_12h = corpus["rain_mm"].rolling(48, min_periods=1).sum()
    print(f"max 12h rolling rain sum: {rolling_12h.max():.1f} mm at {rolling_12h.idxmax()}\n")

    print("distinct rain events (contiguous rain_mm > 0) with >5mm accumulated:")
    is_raining = corpus["rain_mm"] > 0
    run_id = (is_raining != is_raining.shift()).cumsum()
    events = []
    for rid, group in corpus.groupby(run_id):
        if not is_raining.loc[group.index[0]]:
            continue
        accum = group["rain_mm"].sum()
        if accum > 5.0:
            events.append((group.index.min(), group.index.max(), accum))
    for start, end, accum in events:
        print(f"  {start} -> {end}: {accum:.1f} mm")
    if not events:
        print("  (none)")
    print()

    print("rows per month:")
    monthly = corpus.groupby(corpus.index.tz_localize(None).to_period("M")).size()
    for period, count in monthly.items():
        print(f"  {period}: {count:,}")


if __name__ == "__main__":
    main()

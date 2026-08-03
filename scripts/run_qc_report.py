"""
Runs the three quality_control checks across data/raw_data/ and prints a
summary: duplicate site pairs and their date ranges, sentinel counts per
site/column, and streak counts per site/column.
"""

import pandas as pd

from strawberrywatch import paths
from strawberrywatch.ingest.quality_control import (
    inter_site_duplicate_check,
    repeated_streak_check,
    sentinel_check,
)
from strawberrywatch.ingest.raw_data_loader import load_raw_long

SENSOR_COLS = ["conductivity", "depth", "temperature"]


def report_duplicates(df):
    print("=" * 70)
    print("INTER-SITE DUPLICATE CHECK (conductivity)")
    print("=" * 70)
    all_pairs = inter_site_duplicate_check(df, sensor_col="conductivity", return_all=True)
    flagged = [p for p in all_pairs if p["flagged"]]
    near_misses = [p for p in all_pairs if not p["flagged"]]

    if flagged:
        for pair in flagged:
            print(
                f"  FLAGGED: {pair['site_a']} <-> {pair['site_b']}: "
                f"{pair['n_matched']}/{pair['n_coincident']} coincident readings match "
                f"({pair['match_frac']:.1%})"
            )
            for start, end in pair["ranges"]:
                print(f"      {start} -> {end}")
    else:
        print("no pair cleared the match_frac threshold.")

    # Pairs above min_points that didn't clear match_frac still deserve a look:
    # a real duplicated stretch confined to part of the history dilutes toward
    # zero once measured against the full overlap.
    if near_misses:
        print("\n  near-misses (above min_points, below match_frac):")
        for pair in near_misses:
            print(
                f"    {pair['site_a']} <-> {pair['site_b']}: "
                f"{pair['n_matched']}/{pair['n_coincident']} match ({pair['match_frac']:.1%})"
            )
            if pair["match_frac"] > 0.05 and pair["ranges"]:
                longest = max(pair["ranges"], key=lambda r: r[1] - r[0])
                print(f"      longest matched stretch: {longest[0]} -> {longest[1]}")
    print()


def report_sentinels(df):
    print("=" * 70)
    print("SENTINEL CHECK (sentinel values + negative readings)")
    print("=" * 70)
    for site in sorted(df["location"].unique()):
        site_df = df[df["location"] == site]
        counts = {}
        for col in SENSOR_COLS:
            if col not in site_df.columns:
                continue
            n = int(sentinel_check(site_df, sensor_cols=[col]).sum())
            if n > 0:
                counts[col] = n
        if counts:
            parts = ", ".join(f"{col}={n}" for col, n in counts.items())
            print(f"  {site}: {parts}")
    print()


def report_streaks(df, max_streak=96):
    print("=" * 70)
    print(f"REPEATED STREAK CHECK (max_streak={max_streak})")
    print("=" * 70)
    for site in sorted(df["location"].unique()):
        site_df = df[df["location"] == site]
        counts = {}
        for col in SENSOR_COLS:
            if col not in site_df.columns:
                continue
            sub = site_df[["location", col]]
            mask = repeated_streak_check(sub, sensor_cols=[col], max_streak=max_streak)
            n_rows = int(mask.sum())
            if n_rows > 0:
                # number of distinct runs, not just flagged rows
                flagged_times = sub.index[mask]
                gaps = flagged_times.to_series().diff() > pd.Timedelta(hours=1)
                n_runs = int(gaps.sum()) + 1
                counts[col] = (n_rows, n_runs)
        if counts:
            parts = ", ".join(
                f"{col}={n_rows} rows across {n_runs} runs"
                for col, (n_rows, n_runs) in counts.items()
            )
            print(f"  {site}: {parts}")
    print()


def main():
    print(f"loading raw data from {paths.raw_data_dir()}...\n")
    df = load_raw_long()
    sites = sorted(df["location"].unique())
    print(f"sites found ({len(sites)}): {sites}")
    print(f"total rows: {len(df):,}")
    print(f"range: {df.index.min()} to {df.index.max()}\n")

    report_duplicates(df)
    report_sentinels(df)
    report_streaks(df)


if __name__ == "__main__":
    main()

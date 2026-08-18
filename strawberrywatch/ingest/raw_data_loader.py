"""
Loads data/raw_data/ into a standardized long-format frame. Shared by
run_qc_report.py and build_training_corpus.py so both read the raw archive
the same way instead of duplicating the sniffing/renaming logic.
"""

import glob
import os

import pandas as pd

from strawberrywatch import paths

# scnf010 is the Wickson Footbridge site under its old MMW code (see README);
# university_house's filename carries a device id suffix. Every other site
# maps straight off its filename stem.
_FILENAME_TO_SITE_OVERRIDES = {
    "scnf010": "footbridge",
    "university_house_1778210630544": "university_house",
}

_TIME_CANDIDATES = ["DateTimeUTC", "timestamp", "datetime"]

_SENSOR_COLUMN_MAPPING = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
}


def _site_name_for_file(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return _FILENAME_TO_SITE_OVERRIDES.get(stem, stem)


def _read_flexible_csv(path):
    """
    Raw files aren't consistent about delimiter. Sniff first with sep=None; if
    that collapses everything into one fused column, the sniffer picked wrong,
    so retry with tab then comma.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    if len(df.columns) == 1 and ("\t" in df.columns[0] or "," in df.columns[0]):
        for sep in ["\t", ","]:
            retry = pd.read_csv(path, sep=sep)
            if len(retry.columns) > 1:
                return retry
    return df


def load_raw_site(path):
    """
    Load one raw_data CSV into a frame indexed by UTC datetime with
    conductivity/depth/temperature columns (whichever are present). Duplicate
    timestamps within the file are collapsed, keeping the first occurrence.
    """
    df = _read_flexible_csv(path)

    time_col = next((c for c in _TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"{path}: no recognizable timestamp column among {df.columns.tolist()}")

    df["datetime"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])

    df = df.rename(columns=_SENSOR_COLUMN_MAPPING)
    sensor_cols = [c for c in _SENSOR_COLUMN_MAPPING.values() if c in df.columns]
    for col in sensor_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[sensor_cols]


def load_all_raw_sites(raw_dir=None):
    """Return {site_name: DataFrame} for every CSV in raw_dir."""
    raw_dir = raw_dir or paths.raw_data_dir()
    sites = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        site = _site_name_for_file(path)
        sites[site] = load_raw_site(path)
    return sites


def load_raw_long(raw_dir=None):
    """
    Return one concatenated long-format frame: datetime index, 'location'
    column, plus whatever sensor columns each site's file had (missing ones
    are NaN for that site).
    """
    frames = []
    for site, df in load_all_raw_sites(raw_dir).items():
        df = df.copy()
        df["location"] = site
        frames.append(df)
    return pd.concat(frames, axis=0).sort_index()

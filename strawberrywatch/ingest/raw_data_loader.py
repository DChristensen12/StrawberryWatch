"""
Loads data/raw_data/ into a standardized long-format frame. Shared by
run_qc_report.py and build_training_corpus.py so both read the raw archive
the same way instead of duplicating the sniffing/renaming logic.
"""

import glob
import os

import pandas as pd

from strawberrywatch import paths

# scnf010 is the Wickson Footbridge site under its old site code;
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


# Every channel the raw tables carry, including the ones the corpus does not use.
# Keyed the way the inventory names variables, so a caller can join the two.
_ALL_CHANNEL_MAPPING = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
    "AtlasSci_DO": "dissolved_oxygen",
    "AtlasSci_FloatCond": "floating_conductivity",
    "AtlasSci_pH": "ph",
    "BalanceHydro_Stage2_m": "stage",
}


def _table_name_for_file(path):
    """
    Map a raw filename to its SQL table name, which is what the inventory keys on.

    Deliberately not _site_name_for_file. That one renames scnf010 to footbridge
    for the corpus; the inventory speaks table names, and renaming here would
    make every lookup miss.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("university_house"):
        return "university_house"
    return stem


def load_archive_by_table(raw_dir=None, earliest=None, with_report=False):
    """
    Return {sql table name: DataFrame} carrying every channel the file has.

    Sentinels stay in. Naming a sentinel is a quality test's job, and dropping
    them here would hide the wrong SDI-12 address the test exists to report.

    Rows stamped before `earliest` are dropped as logger clock resets rather
    than kept as readings. kingman_hall has one stamped 2000-06-17, which is a
    Mayfly that booted with no time set, and keeping it stretches that site's
    history back 25 years. Nothing is dropped quietly: pass with_report=True to
    get (tables, dropped) and say what went and why.
    """
    raw_dir = raw_dir or paths.raw_data_dir()
    earliest = pd.Timestamp(earliest, tz="UTC") if isinstance(earliest, str) else earliest
    tables = {}
    dropped = []

    for path in sorted(glob.glob(os.path.join(raw_dir, "*.csv"))):
        df = _read_flexible_csv(path)
        time_col = next((c for c in _TIME_CANDIDATES if c in df.columns), None)
        if time_col is None:
            raise ValueError(f"{path}: no recognizable timestamp column")
        df["datetime"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"])
        df = df.rename(columns=_ALL_CHANNEL_MAPPING)
        cols = [c for c in _ALL_CHANNEL_MAPPING.values() if c in df.columns]
        for col in cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("datetime").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        table = _table_name_for_file(path)
        if earliest is not None:
            too_old = df.index < earliest
            if too_old.any():
                dropped.append(
                    {
                        "table": table,
                        "rows": int(too_old.sum()),
                        "worst": f"{df.index[too_old].min():%Y-%m-%dT%H:%MZ}",
                        "earliest": f"{earliest:%Y-%m-%d}",
                        "reason": "logger clock reset, not a measurement",
                    }
                )
                df = df[~too_old]

        tables[table] = df[cols]

    return (tables, dropped) if with_report else tables


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

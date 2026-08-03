from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

import mysql.connector
import pandas as pd
from mysql.connector import Error as MySQLError

from strawberrywatch.config import Config

logger = logging.getLogger(__name__)


@contextmanager
def _connect() -> Iterator[mysql.connector.MySQLConnection]:
    """Open a MySQL connection from Config; always closes."""
    missing = [
        name
        for name, val in [
            ("MYSQL_HOST", Config.MYSQL_HOST),
            ("MYSQL_DATABASE_USER", Config.MYSQL_USER),
            ("MYSQL_DATABASE_PASSWORD", Config.MYSQL_PASSWORD),
            ("MYSQL_DATABASE_NAME", Config.MYSQL_DATABASE),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"sql_client: missing MySQL env vars: {', '.join(missing)}. "
            f"Set them in your .env file before using --data-source sql."
        )

    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DATABASE,
        connection_timeout=30,
    )
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


# Site-to-table-name resolution (mirrors creek_data.py lines 483-489)

# Sites whose production table names preserve capitalization. Extend if
# more MMW sites come online.
_SITE_TABLE_OVERRIDES = {
    "scnf010": "SCNF010",
}


def _table_name_for_site(site_code: str) -> str:
    """Converts a site code to the MySQL table name, handling capitalization edge cases."""
    if not isinstance(site_code, str) or not site_code:
        raise ValueError(f"Invalid site_code: {site_code!r}")
    lowered = site_code.lower()
    if lowered in _SITE_TABLE_OVERRIDES:
        return _SITE_TABLE_OVERRIDES[lowered]
    return "".join(c if c.isalnum() or c == "_" else "_" for c in lowered)


def _is_safe_identifier(name: str) -> bool:
    """Returns True if name is a safe MySQL identifier (no injection risk)."""
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def fetch_creek_data_sql(site: str, start_time, end_time) -> pd.DataFrame:
    """
    Pulls all sensor columns for one site from MySQL over [start_time, end_time].

    Returns a DataFrame with 'timestamp' plus every sensor column the table
    exposes (raw MMW codes like 'Meter_Hydros21_Cond'). Column renaming to
    internal names happens in data_loader, same as the API path.
    """
    table = _table_name_for_site(site)

    # Accept either datetime or ISO string; MySQL handles ISO strings fine.
    start_str = (
        start_time.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(start_time, datetime)
        else str(start_time)
    )
    end_str = (
        end_time.strftime("%Y-%m-%d %H:%M:%S") if isinstance(end_time, datetime) else str(end_time)
    )

    try:
        with _connect() as conn:
            cursor = conn.cursor(buffered=True)
            try:
                try:
                    cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                except MySQLError as e:
                    logger.error(f"[{site}] table `{table}` not accessible: {e}")
                    return pd.DataFrame()

                available = [row[0] for row in cursor.fetchall()]
                if "timestamp" not in available:
                    logger.error(f"[{site}] table `{table}` has no 'timestamp' column")
                    return pd.DataFrame()

                # Safety check on identifier characters before string-interpolating.
                safe_cols = [c for c in available if _is_safe_identifier(c)]
                unsafe = set(available) - set(safe_cols)
                if unsafe:
                    logger.warning(f"[{site}] skipping unsafe column names: {sorted(unsafe)}")

                select_sql = ", ".join(f"`{c}`" for c in safe_cols)
                query = (
                    f"SELECT {select_sql} FROM `{table}` "
                    f"WHERE `timestamp` >= %s AND `timestamp` <= %s "
                    f"ORDER BY `timestamp` ASC"
                )
                df = pd.read_sql(query, conn, params=(start_str, end_str))
            finally:
                cursor.close()
    except Exception as e:
        logger.error(f"[{site}] SQL query failed: {e}", exc_info=True)
        return pd.DataFrame()

    if df.empty:
        print(f"No records found in SQL for {site} in the specified range.")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Tag with location so data_loader's column_mapping can rename it to 'location'
    # via the 'station_id' -> 'location' entry, matching the API path's shape.
    df["station_id"] = site

    logger.info(f"[{site}] fetched {len(df):,} rows from `{table}`")
    return df


def fetch_network_snapshot_sql(start_time, end_time) -> pd.DataFrame:
    """
    Pulls data for every site in Config.LOCATIONS and concatenates them.
    Same shape as api_client.fetch_network_snapshot.
    """
    frames = []
    for site in Config.LOCATIONS:
        print(f"Querying SQL database: {site}...")
        df_site = fetch_creek_data_sql(site, start_time, end_time)
        if not df_site.empty:
            frames.append(df_site)

    if not frames:
        print("No data retrieved from SQL database for any site.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

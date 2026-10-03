"""
Which of Night Heron's tables a site's readings come out of, against a fake MySQL.

Their fetcher writes scnf010 lowercased, while the table their sensor signal made
keeps the site code's case. On a case sensitive server that is two tables.
"""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

import pandas as pd

from strawberrywatch.ingest import sql_client

END = datetime(2026, 9, 3, 22, 0, tzinfo=UTC)


def _rows(n):
    # Naive stamps, the way their DATETIME columns come back
    stamps = pd.date_range(end=END.replace(tzinfo=None), periods=n, freq="15min")
    return pd.DataFrame({"timestamp": stamps, "Meter_Hydros21_Cond": [float(i) for i in range(n)]})


class _Cursor:
    def __init__(self, schema):
        self.schema = schema
        self.rows = []

    def execute(self, sql):
        table = sql.split("`")[1]
        if table not in self.schema:
            raise sql_client.MySQLError(msg=f"Table '{table}' doesn't exist", errno=1146)
        self.rows = [(column,) for column in self.schema[table].columns]

    def fetchall(self):
        return self.rows

    def close(self):
        pass


def _serve(monkeypatch, schema):
    """Point sql_client at a schema of {table: rows}. Returns the tables queried, in order."""
    queried = []

    class Connection:
        def cursor(self, buffered=False):
            return _Cursor(schema)

    @contextmanager
    def connect():
        yield Connection()

    def read_sql(query, conn, params=None):
        table = query.split("FROM `")[1].split("`")[0]
        queried.append(table)
        return schema[table].copy()

    monkeypatch.setattr(sql_client, "_connect", connect)
    monkeypatch.setattr(sql_client.pd, "read_sql", read_sql)
    return queried


def _fetch(site):
    return sql_client.fetch_creek_data_sql(site, END - timedelta(days=2), END)


def test_ordinary_sites_ask_once(monkeypatch):
    queried = _serve(monkeypatch, {"oxford": _rows(3)})
    assert len(_fetch("oxford")) == 3
    assert queried == ["oxford"]


def test_footbridge_falls_back_to_the_lowercase_table(monkeypatch):
    queried = _serve(monkeypatch, {"scnf010": _rows(3)})
    assert len(_fetch("scnf010")) == 3
    assert queried == ["scnf010"]


def test_an_empty_uppercase_table_does_not_hide_the_readings(monkeypatch):
    queried = _serve(monkeypatch, {"SCNF010": _rows(0), "scnf010": _rows(3)})
    assert len(_fetch("scnf010")) == 3
    assert queried == ["SCNF010", "scnf010"]


def test_the_uppercase_table_still_wins_when_it_has_readings(monkeypatch):
    queried = _serve(monkeypatch, {"SCNF010": _rows(2), "scnf010": _rows(3)})
    assert len(_fetch("scnf010")) == 2
    assert queried == ["SCNF010"]


def test_no_table_at_all_is_empty_and_said_once(monkeypatch, caplog):
    _serve(monkeypatch, {})
    assert _fetch("scnf010").empty
    assert caplog.text.count("not accessible") == 1
    assert "`SCNF010` or `scnf010`" in caplog.text

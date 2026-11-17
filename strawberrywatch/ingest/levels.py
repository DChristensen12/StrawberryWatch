"""The screened level: raw readings plus a QARTOD flag per reading and per test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strawberrywatch.support_modules import qc_tests as qc

STEP = pd.Timedelta(minutes=15)
# Kept in each reading's test record but left out of its own flag
RECORD_ONLY = ("time_gap",)
# Flag levels no model reads, the default FLAG_POLICY in models/contracts.py
ALWAYS_OUT = (qc.FAIL, qc.MISSING)
# reviewed()'s mark for a reading the screen took, where a field log line marks its number
SCREENED = -1


def cadence(stamps, default=STEP):
    """Median spacing between a sensor's own readings, to the minute."""
    if len(stamps) < 3:
        return default
    step = pd.Series(stamps).diff().median().round("min")
    return default if pd.isna(step) or step <= pd.Timedelta(0) else step


def expected_steps(stamps, start, end, step):
    """Starts of the steps from start to end that no reading fell in."""
    grid = pd.date_range(start.floor(step), end, freq=step)
    grid = grid[(grid >= start) & (grid + step <= end)]
    return grid.difference(stamps.floor(step))


def _in_creek(sensor, index):
    """True where the inventory has this probe installed and in service."""
    if not sensor.in_service or sensor.install is None:
        return np.zeros(len(index), dtype=bool)
    there = np.asarray(index >= sensor.install)
    if sensor.removed is not None:
        there &= np.asarray(index < sensor.removed)
    return there


def screen_table(table, site, inventory, start=None, end=None, now=None, steps=None):
    """{column: flags} for one raw table, with a row for each missing step from start to end."""
    from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING
    from strawberrywatch.support_modules.settling_pool import flag_one

    entry = inventory.site(site)
    earliest = inventory.earliest_plausible_reading
    earliest = -np.inf if earliest is None else earliest.timestamp()
    out = {}
    for col, variable in _ALL_CHANNEL_MAPPING.items():
        sensor = entry.sensor_for(variable)
        if sensor is None:
            continue
        # a table under the inventory's names reads the same
        if col not in table.columns and variable in table.columns:
            col = variable
        values = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        if col in table.columns:
            values = pd.to_numeric(table[col], errors="coerce").dropna()
            values = values[~values.index.duplicated(keep="first")].sort_index()

        index = values.index
        if start is not None:
            # steps covers a column with too few readings to give its own cadence
            step = cadence(index, (steps or {}).get(col, STEP))
            empty = expected_steps(index, start, end, step)
            index = index.union(empty[_in_creek(sensor, empty)])
        if not len(index):
            continue

        series = values.reindex(index)
        present = series.notna().to_numpy()
        flag, per_test = flag_one(series.to_numpy(), index, site, variable, inventory)

        epoch = index.as_unit("ns").asi8 / 1e9
        dates = qc.impossible_date(epoch, (earliest, np.inf), now)
        per_test["impossible_date"] = np.where(present, dates, qc.MISSING).astype(np.uint8)
        gaps = np.full(len(index), qc.MISSING, dtype=np.uint8)
        if present.any():
            gaps[present] = qc.time_gap(epoch[present], inventory.staleness_hours)
        per_test["time_gap"] = gaps

        flag = qc.worst(flag, per_test["impossible_date"])
        # A reading from a probe the inventory doesn't have in the creek isn't judged
        flag[present & ~_in_creek(sensor, index)] = qc.UNKNOWN

        frame = pd.DataFrame(per_test, index=index)
        frame.insert(0, "flag", flag)
        out[col] = frame
    return _out_of_water(out, entry)


def _out_of_water(out, entry):
    """A step where the depth says the probe is out of the water marks its other readings too."""
    from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING
    from strawberrywatch.support_modules.settling_pool import OUT_OF_WATER_VARIABLE

    depth = next((c for c in out if _ALL_CHANNEL_MAPPING.get(c, c) == OUT_OF_WATER_VARIABLE), None)
    if depth is None or "out_of_water" not in out[depth]:
        return out
    probe = entry.sensor_for(OUT_OF_WATER_VARIABLE)
    dry = out[depth]["out_of_water"] == qc.SUSPECT
    dry = dry[dry].index
    for col, frame in out.items():
        variable = _ALL_CHANNEL_MAPPING.get(col, col)
        if col == depth or variable not in probe.variables:
            continue
        hit = frame.index.isin(dry) & (frame["flag"] != qc.MISSING).to_numpy()
        frame["out_of_water"] = np.where(hit, qc.SUSPECT, qc.GOOD).astype(np.uint8)
        if hit.any():
            frame.loc[hit, "flag"] = qc.worst(
                frame.loc[hit, "flag"].to_numpy(), frame.loc[hit, "out_of_water"].to_numpy()
            )
    return out


def screen(tables, inventory, start=None, end=None, now=None):
    """screen_table over every table the inventory knows, as {site: {column: frame}}."""
    flags = {}
    for site, table in tables.items():
        if site in inventory.sites:
            flags[site] = screen_table(table, site, inventory, start, end, now)
    return flags


def keep(table, flags, levels=ALWAYS_OUT):
    """A copy of a raw table with the readings at these flag levels blanked."""
    out = table.copy()
    for col, frame in (flags or {}).items():
        if col not in out.columns:
            continue
        bad = frame.index[np.isin(frame["flag"].to_numpy(), levels)]
        out.loc[out.index.isin(bad), col] = np.nan
    return out


def removed(table, flags, levels=ALWAYS_OUT):
    """How many readings keep() blanks in this table."""
    count = 0
    for col, frame in (flags or {}).items():
        if col in table.columns:
            bad = frame.index[np.isin(frame["flag"].to_numpy(), levels)]
            count += int((table.index.isin(bad) & table[col].notna().to_numpy()).sum())
    return count


def reviewed(tables, inventory, log, now=None):
    """(tables, why): what a live check would read, minus the field log's periods."""
    from strawberrywatch import field_log
    from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING

    out, why = {}, {}
    for site, frame in tables.items():
        if site not in inventory.sites:
            out[site] = frame
            continue
        flags = screen_table(frame, site, inventory, now=now)
        kept = keep(frame, flags)
        lines = pd.DataFrame(0, index=frame.index, columns=list(flags), dtype=int)
        for col in flags:
            present = pd.to_numeric(frame[col], errors="coerce").notna().to_numpy()
            lines.loc[present & kept[col].isna().to_numpy(), col] = SCREENED
            variable = _ALL_CHANNEL_MAPPING.get(col, col)
            by_log = field_log.excluded(log, site, variable, frame.index)
            hit = present & (by_log > 0)
            lines.loc[hit, col] = by_log[hit]
            kept.loc[hit, col] = np.nan
        out[site], why[site] = kept, lines
    return out, why


def removed_by(why):
    """reviewed()'s record as rows of (site, column, line, readings), one per cause."""
    rows = []
    for site, lines in why.items():
        for col in lines.columns:
            counts = lines[col][lines[col] != 0].value_counts()
            rows += [(site, col, int(line), int(n)) for line, n in counts.items()]
    return pd.DataFrame(rows, columns=["site", "column", "line", "readings"])


def table(rows):
    """store.raw() rows as one raw table, logger columns by UTC reading time."""
    out = pd.DataFrame(rows).pivot(index="time", columns="col", values="value")
    out.index = pd.to_datetime(out.index, unit="s", utc=True)
    return out.rename_axis(index=None, columns=None)


def read(store, names, start, end):
    """(raw tables, flags) for these sites over [start, end], from the store's two levels."""
    tables, flags = {}, {}
    for site in names:
        rows = store.raw(site, start.timestamp(), end.timestamp())
        if not rows:
            continue
        tables[site] = table(rows)
        marks = pd.DataFrame(
            store.screened(site, start.timestamp(), end.timestamp()),
            columns=["col", "time", "flag", "tests"],
        )
        marks.index = pd.to_datetime(marks["time"], unit="s", utc=True)
        flags[site] = {col: part[["flag"]] for col, part in marks.groupby("col")}
    return tables, flags


def weather(store, start, end):
    """The saved weather over [start, end] on its UTC index, or None when there is none."""
    rows = store.weather_rows(start.timestamp(), end.timestamp())
    if not rows:
        return None
    frame = pd.DataFrame(rows).set_index("time").drop(columns=["fetched_at"])
    frame.index = pd.to_datetime(frame.index, unit="s", utc=True)
    frame = frame.rename_axis(None).dropna(axis=1, how="all")
    return None if frame.empty else frame

"""The field log beside the inventory. Check an edit with: python -m strawberrywatch.field_log"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch import inventory as inv

_CSV_NAME = "field_log.csv"

COLUMNS = (
    "date",
    "time",
    "site",
    "sensor",
    "event",
    "end",
    "note",
    "who",
    "confirmed",
    "value_before",
    "value_after",
)

EVENTS = (
    "cleaned",
    "calibrated",
    "swapped",
    "installed",
    "removed",
    "stolen",
    "fouled",
    "out_of_water",
    "buried",
    "units_changed",
    "bad",
    "visit_ok",
    # something other than rain entering the creek, such as a spill or a hydrant flush
    "discharge",
)
# Readings inside one of these aren't measurements of the creek
EXCLUDED = ("bad", "fouled", "out_of_water", "buried")
UNITS_CHANGED = "units_changed"
VISIT_OK = "visit_ok"
ALL = "all"

# RFC 3339, offset included. A date with no time is the whole Pacific day
LOCAL = "America/Los_Angeles"

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


class FieldLogError(ValueError):
    """A line of the field log can't be read. The message names the line and the fix."""


def _problem(path, line, what, fix):
    """The same error shape as the inventory's: where, what, then how to fix it."""
    where = f"{os.path.basename(path)} line {line}" if line else os.path.basename(path)
    return FieldLogError(f"{where}: {what}\n\n{fix}\n")


def day_end(day):
    """The next Pacific midnight after a date or a moment, in UTC. A clock-change day is 23 or 25 hours."""
    local = pd.Timestamp(day)
    local = local.tz_convert(LOCAL).tz_localize(None) if local.tzinfo else local
    following = local.normalize() + pd.Timedelta(days=1)
    return following.tz_localize(LOCAL, ambiguous=True, nonexistent="shift_forward").tz_convert(
        "UTC"
    )


def _moment(path, line, date, clock, where):
    """An RFC 3339 date, and its time with an offset when one is given, as a UTC Timestamp."""
    if not _DATE.match(date):
        raise _problem(
            path,
            line,
            f"{where} is {date!r}, which is not a date",
            "Write the date as year-month-day, for example:\n"
            "    2026-03-05\n"
            "A spreadsheet may rewrite dates when it saves. Set the column to Text first.",
        )
    if clock and not _TIME.match(clock):
        raise _problem(
            path,
            line,
            f"the time is {clock!r}, which is not a time with its offset from UTC",
            "Write the time on a 24 hour clock with seconds and the offset, for example:\n"
            "    14:30:00-08:00    Pacific standard time\n"
            "    14:30:00-07:00    Pacific daylight time\n"
            "    22:30:00Z         UTC\n"
            "or leave it empty for the start of the day.",
        )
    try:
        if clock:
            return pd.Timestamp(f"{date}T{clock}").tz_convert("UTC")
        day = pd.Timestamp(date).tz_localize(LOCAL, ambiguous=True, nonexistent="shift_forward")
        return day.tz_convert("UTC")
    except ValueError:
        raise _problem(
            path, line, f"{where} {date} {clock} is not a real date and time", "Check the day."
        ) from None


def _variables(entry, sensor):
    """The inventory variables a sensor cell names at one site, or None."""
    if sensor == ALL:
        return tuple(entry.variables)
    if sensor in entry.sensors:
        return tuple(entry.sensors[sensor].variables)
    if sensor in entry.variables:
        return (sensor,)
    return None


def _number(path, line, name, cell):
    if not cell:
        return np.nan
    try:
        return float(cell)
    except ValueError:
        raise _problem(
            path,
            line,
            f"{name} is {cell!r}, which is not a number",
            f"Write the reading as a number, for example 412.5, or leave {name} empty.",
        ) from None


def _read_line(path, line, cells, inventory):
    if cells["site"] not in inventory.sites:
        raise _problem(
            path,
            line,
            f"site is {cells['site']!r}, which the inventory doesn't have",
            "Write one of:\n    " + ", ".join(sorted(inventory.sites)),
        )
    entry = inventory.site(cells["site"])
    variables = _variables(entry, cells["sensor"])
    if variables is None:
        probes = [name for name, sensor in entry.sensors.items() if sensor.fitted]
        raise _problem(
            path,
            line,
            f"sensor is {cells['sensor']!r}, which {cells['site']} doesn't have",
            f"Write a probe ({', '.join(probes)}), one of its readings "
            f"({', '.join(entry.variables)}), or all.",
        )
    if cells["event"] not in EVENTS:
        raise _problem(
            path,
            line,
            f"event is {cells['event']!r}, which is not one of the events",
            "Write one of:\n    " + ", ".join(EVENTS),
        )
    if cells["confirmed"] not in ("yes", "no"):
        raise _problem(
            path,
            line,
            f"confirmed is {cells['confirmed']!r}",
            "Write yes if someone saw it at the creek, or no if it was worked out from the data.",
        )

    start = _moment(path, line, cells["date"], cells["time"], "date")
    end = pd.NaT
    if cells["end"]:
        day, clock = cells["end"][:10], cells["end"][11:]
        end = _moment(path, line, day, clock, "end")
        # a day with no time runs to its last minute
        end = end + pd.Timedelta(minutes=1) if clock else day_end(day)
        if end <= start:
            raise _problem(
                path,
                line,
                f"end {cells['end']} is before the line's own date",
                "Write when it ended as year-month-day, with a time and its offset if you know "
                "it, for example:\n"
                "    2026-04-28T16:00:00-07:00\n"
                "or leave end empty if it's still going on.",
            )
    return {
        "line": line,
        "start": start,
        "end": end,
        "site": cells["site"],
        "sensor": cells["sensor"],
        "variables": variables,
        "event": cells["event"],
        "note": cells["note"],
        "who": cells["who"],
        "confirmed": cells["confirmed"] == "yes",
        "value_before": _number(path, line, "value_before", cells["value_before"]),
        "value_after": _number(path, line, "value_after", cells["value_after"]),
        "key": f"{cells['date']} {cells['site']} {cells['sensor']} {cells['event']}",
    }


def _frame(rows):
    columns = [
        "line",
        "start",
        "end",
        "site",
        "sensor",
        "variables",
        "event",
        "note",
        "who",
        "confirmed",
        "value_before",
        "value_after",
        "key",
    ]
    return pd.DataFrame(rows, columns=columns)


def path_for(path=None):
    return Path(path) if path else Path(__file__).resolve().parent / _CSV_NAME


def load(path=None, inventory=None):
    """The log as one row per line, times in UTC, keyed "date site sensor event" for events.yaml."""
    path = path_for(path)
    inventory = inventory or inv.load()
    if not path.exists():
        return _frame([])
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise _problem(
                path,
                1,
                f"the first line is missing {', '.join(missing)}",
                "The first line has to be:\n    " + ",".join(COLUMNS),
            )
        for record in reader:
            cells = {c: (record.get(c) or "").strip() for c in COLUMNS}
            if any(cells.values()):
                rows.append(_read_line(path, reader.line_num, cells, inventory))
    return _frame(rows)


def excluded(log, site, variable, index):
    """The field log line that removes each of a sensor's readings from the reviewed level, or 0."""
    out = np.zeros(len(index), dtype=int)
    for row in log.itertuples():
        if row.site != site or variable not in row.variables:
            continue
        if row.event not in EXCLUDED and row.event != UNITS_CHANGED:
            continue
        inside = np.asarray(index >= row.start)
        if row.event in EXCLUDED and pd.notna(row.end):
            inside &= np.asarray(index < row.end)
        out[inside & (out == 0)] = row.line
    return out


def visits(log):
    """The visit_ok lines. Someone checked the sensor and it was fine."""
    return log[log["event"] == VISIT_OK]


def find(log, key):
    """The lines tests/events.yaml cites by this key."""
    return log[log["key"] == key]


def main(argv=None):
    try:
        log = load(argv[0] if argv else None)
    except FieldLogError as exc:
        print(exc)
        return 1
    print(f"The field log is fine: {len(log)} lines.")
    for event, count in log["event"].value_counts().items():
        print(f"  {event}: {count}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))

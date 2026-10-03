"""
What every serving detector takes and what it hands back.

In: Night Heron's site tables, as they store them. Out: a list of Finding. A
model can be run from outside this repository once it speaks both, which is
what tests/test_serving_conformance.py holds each one to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class DetectorError(RuntimeError):
    """We cannot score what we were given, and the message says why."""


@dataclass(frozen=True)
class Finding:
    """
    One (site, variable, rule) that fired, with what someone needs to act on it.

    site is the table the readings came from, under the name it was handed in as.
    variable uses the inventory's names (conductivity, depth, temperature,
    dissolved_oxygen, floating_conductivity) whatever the model calls it inside,
    so nobody downstream has to learn two vocabularies.

    peak is the worst score the rule saw and threshold is the calibrated bar it
    was held to, before rain moved it. when is the first step over the bar.
    readings are that variable's values at that site in sensor units, the ones
    the model was shown before any scaling.
    """

    site: str
    variable: str
    rule: str
    peak: float
    threshold: float
    when: pd.Timestamp
    readings: pd.Series = field(repr=False, compare=False)


def utc_indexed(name, table):
    """
    One site table with its index as UTC reading times.

    Naive stamps are taken as UTC, which is how Night Heron's DATETIME columns
    hold them. Anything that is not a time index is refused rather than parsed,
    because a RangeIndex read as nanoseconds is 1970 and nothing downstream
    would notice.
    """
    if not isinstance(table.index, pd.DatetimeIndex):
        raise DetectorError(
            f"table {name!r} is indexed by {type(table.index).__name__}, expected the "
            f"reading time as a DatetimeIndex. Set the timestamp column as the index "
            f"before handing it over."
        )
    if table.index.tz is None:
        return table.tz_localize("UTC")
    if str(table.index.tz) != "UTC":
        return table.tz_convert("UTC")
    return table

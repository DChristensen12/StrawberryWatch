"""
What every serving detector takes and what it hands back.

In: Night Heron's site tables, as they store them. Out: a list of Finding. A
model can be run from outside this repository once it speaks both, which is
what tests/test_serving_conformance.py holds each one to.

Optionally a detector also hands out scores(): every recent step it judged,
with the line that step had to cross. notice() turns those back into what the
detector's own findings() would say, at any scale of the line, which is what
lets a dashboard ask "what if the line were higher" without reloading anything.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# What a detector can do about a site somebody doesn't want flagged. Every
# model can at least not flag it. Leaving it out of the input is only safe for
# a model built for missing sites, so a model has to say it can.
DONT_FLAG = "dont_flag"
LEAVE_OUT = "leave_out"
SITE_CHOICES = (DONT_FLAG, LEAVE_OUT)


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


@dataclass(frozen=True)
class Score:
    """
    One judged step: what the rule scored and the line it had to cross.

    threshold is the line at that step, rain included, at a scale of 1. base is
    the calibrated line before rain moved it, the same number a Finding calls
    threshold. A step the rule didn't judge, say a reading that never came, has
    no Score at all rather than a NaN one.
    """

    site: str
    variable: str
    rule: str
    time: pd.Timestamp
    score: float
    threshold: float
    base: float


@dataclass(frozen=True)
class Noticed:
    """What notice() found for one (site, variable, rule): when it first crossed and how far."""

    site: str
    variable: str
    rule: str
    when: pd.Timestamp
    last: pd.Timestamp
    peak: float
    threshold: float
    steps_over: int


def site_choice(detector):
    """What a detector can do about a site: ("dont_flag",) unless it says more."""
    return tuple(getattr(detector, "SITE_CHOICE", (DONT_FLAG,)))


def check_leave_out(detector, leave_out):
    """
    Raise unless the detector can leave these sites out.

    Raising is the point, the same as detector_class with a name it doesn't
    know. A model that quietly ignored leave_out would go on reading a site
    somebody had asked it not to, and nothing on screen would say so.
    """
    leave_out = set(leave_out or ())
    if not leave_out:
        return leave_out
    if LEAVE_OUT not in site_choice(detector):
        name = getattr(detector, "name", type(detector).__name__)
        raise DetectorError(
            f"{name} can't leave sites out (SITE_CHOICE = {site_choice(detector)}), "
            f"but was asked to leave out {', '.join(sorted(leave_out))}"
        )
    return leave_out


def notice(scores, min_steps=1, scale=1.0):
    """
    Which (site, variable, rule) went over its line, given scores().

    A group counts once at least min_steps of its steps are over threshold *
    scale, anywhere in what was scored. That is Dusk Crayfish's own rule with
    its min_timesteps_over, and Cobble Shoal's with 1, so at a scale of 1 this
    says what findings() says. NaN never counts as over, the same as
    rain_gate.fired.
    """
    groups = defaultdict(list)
    for row in scores:
        groups[(row.site, row.variable, row.rule)].append(row)
    out = []
    for (site, variable, rule), rows in groups.items():
        rows.sort(key=lambda r: r.time)
        values = np.array([r.score for r in rows], dtype=float)
        lines = np.array([r.threshold for r in rows], dtype=float) * scale
        over = np.isfinite(values) & (values > lines)
        count = int(over.sum())
        if count < max(1, min_steps):
            continue
        hits = np.flatnonzero(over)
        out.append(
            Noticed(
                site=site,
                variable=variable,
                rule=rule,
                when=rows[hits[0]].time,
                last=rows[hits[-1]].time,
                peak=float(np.nanmax(values)),
                threshold=float(rows[hits[0]].base),
                steps_over=count,
            )
        )
    out.sort(key=lambda n: (n.when, n.site, n.variable, n.rule))
    return out

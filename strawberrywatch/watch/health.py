"""Data health: checks on the readings themselves, apart from the models."""

from __future__ import annotations

import logging

import pandas as pd

from strawberrywatch import inventory as inv
from strawberrywatch.ingest import levels
from strawberrywatch.support_modules import qc_tests as qc

logger = logging.getLogger(__name__)

DAY = 86400
# Mayfly loggers post every 15 minutes, and Balance Hydrologics serves 15 minute rows
CADENCE = {inv.MAYFLY: pd.Timedelta(minutes=15), inv.BALANCE: pd.Timedelta(minutes=15)}
CADENCE_SLACK = pd.Timedelta(minutes=1)
# A logger clock a little ahead isn't a reading from the future
FUTURE_SLACK = pd.Timedelta(minutes=5)

# The level and range checks compare the last CURRENT_DAYS with the reviewed
# HISTORY_DAYS before them
CURRENT_DAYS = 14
HISTORY_DAYS = 180
MIN_READINGS = 1000
# Ordinary archive weeks moved a median by 1.63 times at most, and south_fork_3's
# oxygen units change by 7 to 11
LEVEL_FACTOR = 3.0
# A day counts as muted under both its 5th percentile and a quarter of its median
RANGE_QUANTILE = 0.05
RANGE_SHARE = 0.25
MUTED_DAYS = 2
FULL_DAY = 48
# Depth moving this much is the storm trigger's rise: 10 mm, or 10% of the
# lowest depth when that is more
TRIGGER_MM = 10.0
TRIGGER_SHARE = 0.10

# Kinds of sensor alert
READING = "reading"
OUT_OF_WATER = "out_of_water"
DEAD = "dead_feed"
LEVEL = "level_shift"
MUTED = "muted"
DATA_CHECK = "data_check"
STUCK = "stuck"
# Found every pass, and cleared after a day unseen. The daily kinds clear when
# the next daily run misses them
EACH_PASS = (READING, OUT_OF_WATER, DATA_CHECK, STUCK)

LABELS = {
    "conductivity": "conductivity",
    "temperature": "temperature",
    "depth": "depth",
    "dissolved_oxygen": "dissolved oxygen",
    "floating_conductivity": "floating conductivity",
    "stage": "stage",
}
UNITS = {"conductivity": " uS/cm", "temperature": " C", "depth": " mm"}


def _variable(col):
    from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING

    return _ALL_CHANNEL_MAPPING.get(col, col)


def _column(variable):
    from strawberrywatch.ingest.raw_data_loader import _ALL_CHANNEL_MAPPING

    back = {v: k for k, v in _ALL_CHANNEL_MAPPING.items()}
    return back.get(variable)


def where(site, col, inventory):
    """South Fork 0 conductivity."""
    from strawberrywatch.watch import settings as ws

    name = ws.site_name(site, inventory)
    if not col:
        return name
    return f"{name} {LABELS.get(_variable(col), _variable(col))}"


def problem(site, col, kind, message, value=None):
    return {"site": site, "col": col or "", "kind": kind, "message": message, "value": value}


def _sensors(site, inventory):
    """(logger column, variable) for each reading of the site's in-service probes."""
    out = []
    for variable in inventory.site(site).scored_variables:
        col = _column(variable)
        if col is not None:
            out.append((col, variable))
    return out


# The data check on each fetch


def schema(site, inventory, now):
    """The pandera schema for one site's table as fetched, built from the inventory."""
    import pandera.pandas as pa

    columns = {}
    for col, variable in _sensors(site, inventory):
        checks = []
        span = inventory.span(site, variable)
        if span is not None:
            low, high = span
            checks.append(pa.Check.in_range(low, high, error=f"outside {low:g} to {high:g}"))
        columns[col] = pa.Column(float, checks=checks, nullable=True, coerce=True)

    step = CADENCE[inventory.site(site).source]
    latest = pd.Timestamp(now, unit="s", tz="UTC") + FUTURE_SLACK

    def on_cadence(df):
        if len(df) < 3:
            return True
        median = pd.Series(df.index).diff().median()
        return abs(median - step) <= CADENCE_SLACK

    return pa.DataFrameSchema(
        columns,
        index=pa.Index(unique=True),
        checks=[
            pa.Check(
                lambda df: isinstance(df.index, pd.DatetimeIndex) and str(df.index.tz) == "UTC",
                error="times aren't UTC",
            ),
            pa.Check(lambda df: df.index.is_monotonic_increasing, error="times out of order"),
            pa.Check(
                lambda df: len(df) == 0 or df.index.max() <= latest, error="times in the future"
            ),
            pa.Check(on_cadence, error=f"not every {int(step.total_seconds() // 60)} minutes"),
        ],
        strict=False,
    )


def check_table(site, table, inventory, now):
    """What the schema found in one fetched table, as problems. Empty when it passed."""
    import pandera.errors

    try:
        schema(site, inventory, now).validate(table, lazy=True)
        return []
    except pandera.errors.SchemaErrors as exc:
        cases = exc.failure_cases
    out = []
    for (column, check), part in cases.groupby(["column", "check"], dropna=False, sort=True):
        col = column if isinstance(column, str) and column in table.columns else ""
        what = check if isinstance(check, str) else "failed a check"
        if what.startswith("column_in_dataframe"):
            missing = str(part["failure_case"].iloc[0])
            what, col = f"no {LABELS.get(_variable(missing), missing)} column", ""
        count = len(part)
        detail = f"{count} readings {what}" if col and count > 1 else what
        out.append(
            problem(site, col, DATA_CHECK, f"Data check at {where(site, col, inventory)}: {detail}")
        )
    return out


# Readings the screen failed, and probes out of the water


def reading_problems(site, raw, flags, inventory):
    """Sensor alerts from one site's screened flags: failed readings and a probe out of water."""
    out = []
    for col, frame in (flags or {}).items():
        values = pd.to_numeric(raw[col], errors="coerce") if col in raw else None
        if values is None:
            continue
        failed = frame.index[(frame["flag"] == qc.FAIL).to_numpy()]
        failed = failed[values.reindex(failed).notna().to_numpy()]
        if len(failed):
            value = float(values.reindex(failed).iloc[-1])
            more = f", {len(failed)} readings" if len(failed) > 1 else ""
            out.append(
                problem(
                    site,
                    col,
                    READING,
                    f"Sensor problem at {where(site, col, inventory)}: reading of {value:g}{more}",
                    value,
                )
            )
        if "out_of_water" in frame and _variable(col) == "depth":
            dry = frame.index[(frame["out_of_water"] == qc.SUSPECT).to_numpy()]
            if len(dry):
                value = float(values.reindex(dry).iloc[-1])
                out.append(
                    problem(
                        site,
                        "",
                        OUT_OF_WATER,
                        f"Sensor problem at {where(site, '', inventory)}: depth of {value:g} mm, "
                        f"possibly out of water",
                        value,
                    )
                )
    return out


# Conductivity that holds still while the creek rises


def stuck_steps(cond, depth, hours, min_readings=1):
    """
    (time, value, hours, mm) for each reading whose run of unchanged
    conductivity spans at least hours and min_readings readings, while depth
    moved by more than the storm trigger over the same run.
    """
    cond, depth = cond.dropna(), depth.dropna()
    run = cond.ne(cond.shift()).cumsum()
    began = cond.index.to_series().groupby(run.to_numpy()).transform("min")
    # a sparse feed can hold one value for hours on a handful of readings
    count = cond.groupby(run.to_numpy()).cumcount().to_numpy() + 1
    held = (cond.index - pd.DatetimeIndex(began) >= pd.Timedelta(hours=hours)) & (
        count >= min_readings
    )
    out = []
    for t in cond.index[held]:
        d = depth[(depth.index >= began[t]) & (depth.index <= t)]
        if len(d) < 2:
            continue
        moved = float(d.max() - d.min())
        if moved > max(TRIGGER_MM, TRIGGER_SHARE * abs(float(d.min()))):
            out.append((t, float(cond[t]), (t - began[t]).total_seconds() / 3600, moved))
    return out


def stuck(store, site, inventory, start, end):
    """A sensor alert for conductivity unchanged for stuck_hours while depth moved."""
    cols = {variable: col for col, variable in _sensors(site, inventory)}
    cond, depth = cols.get("conductivity"), cols.get("depth")
    if cond is None or depth is None:
        return []
    # back far enough for the longest run the screen still passes
    since = start - levels.STEP * (inventory.flat_line_fail_count + 1)
    raw, flags = levels.read(store, [site], since, end)
    table = levels.keep(raw.get(site, pd.DataFrame()), flags.get(site, {}))
    if cond not in table or depth not in table:
        return []
    hits = [
        h
        for h in stuck_steps(
            table[cond], table[depth], inventory.stuck_hours, inventory.stuck_min_readings
        )
        if h[0] >= start
    ]
    if not hits:
        return []
    _, value, hours, moved = hits[-1]
    return [
        problem(
            site,
            cond,
            STUCK,
            f"Sensor problem at {where(site, cond, inventory)}: unchanged at {value:g}"
            f"{UNITS['conductivity']} for {hours:.3g} hours while depth moved {moved:.0f} mm, "
            f"possibly stuck",
            value,
        )
    ]


# The daily checks


def dead_feed(site, col, newest, inventory, now):
    """A dead feed past the inventory's staleness_hours, by the data reception test."""
    if newest is None:
        return problem(
            site, col, DEAD, f"Sensor problem at {where(site, col, inventory)}: no readings"
        )
    flag = qc.data_reception([newest], fail_span=inventory.staleness_hours, from_time=now)[0]
    if flag != qc.FAIL:
        return None
    from strawberrywatch.dashboard import view

    return problem(
        site,
        col,
        DEAD,
        f"Sensor problem at {where(site, col, inventory)}: no reading since "
        f"{view.when_words(newest, now)}",
    )


def level_shift(site, col, current, history, inventory):
    """A 14 day median that moved like a unit change or a recalibration."""
    from strawberrywatch.workbench import checks

    current, history = current.dropna(), history.dropna()
    if len(current) < MIN_READINGS or len(history) < MIN_READINGS or _variable(col) == "depth":
        return None
    now_level, usual = float(current.median()), float(history.median())
    if now_level <= 0 or usual <= 0:
        return None
    factor = max(now_level / usual, usual / now_level)
    if factor < LEVEL_FACTOR:
        return None
    drifted = checks.drifted(current.to_numpy(), checks.reference_sample(history.to_numpy()))
    ratio = f", drift ratio {drifted[1]:.1f}" if drifted else ""
    return problem(
        site,
        col,
        LEVEL,
        f"Sensor problem at {where(site, col, inventory)}: level of {now_level:.3g} against a "
        f"usual {usual:.3g}{ratio}, possibly a unit change or recalibration",
        now_level,
    )


def daily_ranges(values):
    """Each full day's range of readings, by UTC day."""
    values = values.dropna()
    days = values.groupby(values.index.floor("D")).agg(["max", "min", "count"])
    days = days[days["count"] >= FULL_DAY]
    return days["max"] - days["min"]


def muted(site, col, current, history, inventory):
    """A daily range far under the sensor's normal on each of the last MUTED_DAYS days."""
    usual = daily_ranges(history)
    recent = daily_ranges(current).iloc[-MUTED_DAYS:]
    if len(usual) < 30 or len(recent) < MUTED_DAYS:
        return None
    floor = min(float(usual.quantile(RANGE_QUANTILE)), float(usual.median()) * RANGE_SHARE)
    if floor <= 0:
        return None
    # a day's range is a window's range over the day, which attenuated_signal tests
    days = [current[current.index.floor("D") == day].dropna().to_numpy() for day in recent.index]
    flags = [qc.attenuated_signal(values, len(values), min_range=floor)[-1] for values in days]
    if any(flag != qc.SUSPECT for flag in flags):
        return None
    return problem(
        site,
        col,
        MUTED,
        f"Sensor problem at {where(site, col, inventory)}: daily range of {recent.iloc[-1]:.3g} "
        f"against a usual {float(usual.median()):.3g}, possibly fouled or stuck",
        float(recent.iloc[-1]),
    )


def daily(raw, flags, inventory, log, now):
    """The slow checks for one site, over CURRENT_DAYS + HISTORY_DAYS of its raw table."""
    site = raw.attrs.get("site")
    cut = pd.Timestamp(now - CURRENT_DAYS * DAY, unit="s", tz="UTC")
    screened = levels.keep(raw, flags)
    reviewed = levels.reviewed({site: raw[raw.index < cut]}, inventory, log)[0][site]
    out = []
    for col, _variable_name in _sensors(site, inventory):
        values = raw[col].dropna() if col in raw else pd.Series(dtype=float)
        newest = values.index.max().timestamp() if len(values) else None
        found = dead_feed(site, col, newest, inventory, now)
        if found is None and col in raw:
            current = screened[col][screened.index >= cut]
            history = reviewed[col] if col in reviewed else pd.Series(dtype=float)
            found = level_shift(site, col, current, history, inventory) or muted(
                site, col, current, history, inventory
            )
        if found is not None:
            out.append(found)
    return out


# Running them, and the summary email


def after_pass(store, inventory, tables, start, end, now, mail=None):
    """Runs after an ingest pass has saved its readings. Doesn't raise."""
    try:
        found = []
        for site, table in tables.items():
            if site in inventory.sites:
                found += check_table(site, table, inventory, now)
        for site in inventory.sites:
            raw, flags = levels.read(store, [site], start, end)
            if site in raw:
                found += reading_problems(site, raw[site], flags[site], inventory)
                found += stuck(store, site, inventory, start, end)
        daily_ran = now - float(store.pref("health_checked_at") or 0) >= DAY
        if daily_ran:
            found += run_daily(store, inventory, now)
            store.set_pref("health_checked_at", repr(now))
        store.note_problems(found, now)
        clear(store, now, daily_ran)
        if mail is not None:
            send_summary(store, now, website=mail == "website")
    except Exception:
        logger.exception("the data health checks failed this pass")


def run_daily(store, inventory, now):
    from strawberrywatch import field_log

    log = field_log.load(inventory=inventory)
    since = pd.Timestamp(now - (CURRENT_DAYS + HISTORY_DAYS) * DAY, unit="s", tz="UTC")
    until = pd.Timestamp(now, unit="s", tz="UTC")
    found = []
    for site in inventory.sites:
        if not inventory.site(site).in_service:
            continue
        raw, flags = levels.read(store, [site], since, until)
        table = raw.get(site, pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")))
        table.attrs["site"] = site
        found += daily(table, flags.get(site), inventory, log, now)
    return found


def clear(store, now, daily_ran):
    """Marks the problems that have ended: per pass ones not seen for a day, daily ones not found."""
    ended = [
        p
        for p in store.sensor_problems()
        if (p["kind"] in EACH_PASS and now - p["last_at"] > DAY)
        or (p["kind"] not in EACH_PASS and daily_ran and p["last_at"] < now)
    ]
    if ended:
        store.clear_problems(ended, now)


def send_summary(store, now, website=False):
    """One email about the problems that started or cleared since the last one."""
    started = [p for p in store.sensor_problems() if p["sent_at"] is None]
    cleared = [
        p
        for p in store.sensor_problems(cleared=True)
        if p["cleared_sent_at"] is None and p["sent_at"] is not None
    ]
    if not started and not cleared:
        return
    from strawberrywatch.dashboard import mail

    if mail.health_summary(store, started, cleared, website):
        store.mark_problems_sent(started, now)
        store.mark_problems_sent(cleared, now, cleared=True)


__all__ = [
    "after_pass",
    "check_table",
    "daily",
    "dead_feed",
    "level_shift",
    "muted",
    "reading_problems",
    "run_daily",
    "schema",
    "send_summary",
    "stuck",
    "stuck_steps",
]

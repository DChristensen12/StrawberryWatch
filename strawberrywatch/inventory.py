"""
Which probe sits where, since when, and whether it is in use.

Reads inventory.yaml and resolves any (site, variable, timestamp) to one of four
states. Non-coders edit that file, so a bad edit has to stop the run with a
message naming the line and what to write instead. Never guess at intent.

Switches live in one grid at the top of the yaml, a row per site and a column
per sensor. A probe set to no drops out of live scoring and the state table and
reads as NOT_INSTALLED downstream, but its old readings still train, which is
what in_service_only=False is for.

python -m strawberrywatch.inventory prints the lot.
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import yaml

_YAML_NAME = "inventory.yaml"

NOT_INSTALLED = "NOT_INSTALLED"
PRESENT = "PRESENT"
STALE = "STALE"
MISSING = "MISSING"
STATES = (NOT_INSTALLED, PRESENT, STALE, MISSING)

MAYFLY = "mayfly"
BALANCE = "balance_hydrologics"
SOURCES = (MAYFLY, BALANCE)

# Oxford is the confluence outlet, Codornices a separate watershed. Nothing else
# is allowed to drain nowhere.
TERMINAL_SITES = ("oxford", "codornices")

# All a grid cell may say. PyYAML would also swallow True, On and 1 as booleans,
# which is the quiet acceptance this file cannot afford.
YES = "yes"
NO = "no"
ABSENT = "-"
SWITCHES = (YES, NO, ABSENT)


class InventoryError(ValueError):
    """Malformed inventory, or a question about something it does not describe."""


def _problem(path, line, what, fix):
    """The one error shape this module raises: where, what, then how to fix it."""
    where = f"{os.path.basename(path)} line {line}" if line else os.path.basename(path)
    return InventoryError(f"{where}: {what}\n\n{fix}\n")


def _walk(node, path=()):
    """Every node keyed by its path. Mappings too, so a bad row gets a line."""
    if isinstance(node, yaml.MappingNode):
        if path:
            yield path, node
        for key, value in node.value:
            yield from _walk(value, (*path, key.value))
    elif isinstance(node, yaml.ScalarNode):
        yield path, node


def _marks(text):
    """Path to node, so a bad value can be reported with its line."""
    root = yaml.compose(text)
    if root is None:
        return {}
    return dict(_walk(root))


def _switch_help(table, sensor):
    """Same three lines for every bad cell, so there is one thing to learn."""
    return (
        f"Write one of these three, in lower case, with no quotes:\n"
        f"    {sensor}: yes   installed and in service\n"
        f"    {sensor}: no    installed, switched off right now\n"
        f"    {sensor}: -     there is no {sensor} probe at {table}"
    )


def _switch_from_value(value):
    """from_dict has no raw text to check spelling against."""
    if value is None:
        return None
    if isinstance(value, bool):
        return YES if value else NO
    return str(value)


def _read_switch(marks, path, keypath, table, sensor, parsed=None):
    """
    One grid cell, refusing everything PyYAML would quietly accept.

    Read off the raw text, not the parsed value: by the time PyYAML is done,
    yes and true are both True and the mistake is invisible.
    """
    node = marks.get(keypath)
    if node is None:
        return _switch_from_value(parsed)
    line = node.start_mark.line + 1

    if not isinstance(node, yaml.ScalarNode):
        raise _problem(
            path,
            line,
            f"the {sensor} switch for {table} is not a single word",
            _switch_help(table, sensor),
        )
    if node.style is not None:
        raise _problem(
            path,
            line,
            f"the {sensor} switch for {table} is in quotes: "
            f"{sensor}: {node.style}{node.value}{node.style}",
            _switch_help(table, sensor),
        )
    if node.value not in SWITCHES:
        raise _problem(
            path,
            line,
            f"the {sensor} switch for {table} is set to {node.value!r}, which is not yes, no or -",
            _switch_help(table, sensor),
        )
    return node.value


def _line_of(marks, keypath):
    node = marks.get(keypath)
    return node.start_mark.line + 1 if node is not None else None


def _parse_date(value, where, path=None, line=None):
    """Read an ISO date, or raise naming where the bad value came from."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return (
            pd.Timestamp(value).tz_localize("UTC") if value.tzinfo is None else pd.Timestamp(value)
        )
    if isinstance(value, dt.date):
        return pd.Timestamp(value, tz="UTC")

    bad = _problem(
        path or _YAML_NAME,
        line,
        f"{where} is set to {value!r}, which is not a date",
        "Write the date as year-month-day in quotes, for example:\n"
        "    install: '2026-03-05'\n"
        "or, if the probe was never fitted:\n"
        "    install: null",
    )
    try:
        stamp = pd.Timestamp(str(value))
    except (ValueError, TypeError):
        raise bad from None
    if pd.isna(stamp):
        raise bad
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp


class Sensor:
    """One probe at one site, with the dates bounding when it was there."""

    def __init__(
        self,
        site,
        name,
        variables,
        install,
        removed,
        inferred,
        evidence,
        unconfirmed,
        switch=YES,
    ):
        self.site = site
        self.name = name
        self.variables = tuple(variables)
        self.install = install
        self.removed = removed
        self.inferred = bool(inferred)
        self.evidence = evidence
        self.unconfirmed = unconfirmed
        self.switch = switch

    @property
    def in_service(self):
        return self.switch == YES

    @property
    def fitted(self):
        """Whether the grid says a probe of this kind exists here at all."""
        return self.switch != ABSENT

    @property
    def ever_installed(self):
        return self.install is not None

    def covers(self, timestamp):
        """Whether this probe was in the creek at one timestamp."""
        if self.install is None:
            return False
        if timestamp < self.install:
            return False
        return self.removed is None or timestamp < self.removed

    def __repr__(self):
        return f"<Sensor {self.site}.{self.name} install={self.install}>"


class Site:
    """One monitoring site, its sensors, its data source and where it drains to."""

    def __init__(self, table, name, source, downstream, sensors, thresholds):
        self.table = table
        self.name = name
        self.source = source
        self.downstream = downstream
        self.sensors = sensors
        self.thresholds = thresholds

    @property
    def in_service(self):
        """In service while any one probe is. There is no separate site switch."""
        return any(sensor.in_service for sensor in self.sensors.values())

    @property
    def variables(self):
        """Every variable any sensor here yields, installed or not."""
        return self._variables(self.sensors.values())

    @property
    def scored_variables(self):
        """What live scoring covers, so a probe set to no leaves the state table."""
        return self._variables(s for s in self.sensors.values() if s.in_service)

    @staticmethod
    def _variables(sensors):
        seen = []
        for sensor in sensors:
            for variable in sensor.variables:
                if variable not in seen:
                    seen.append(variable)
        return tuple(seen)

    def sensor_for(self, variable):
        """Which probe yields a variable here, or None if none does."""
        for sensor in self.sensors.values():
            if variable in sensor.variables:
                return sensor
        return None

    def threshold(self, variable, key):
        """One operator-set threshold, or None where the inventory sets none."""
        return (self.thresholds.get(variable) or {}).get(key)

    def __repr__(self):
        return f"<Site {self.table} in_service={self.in_service}>"


class Inventory:
    """
    Every site in the network, validated on load.

    Validation is the point of loading through a class. A site with no sensors,
    an unknown probe, a switch written as true, a row missing a column, a probe
    switched on with no install date, a dangling downstream and a cycle in the
    flow graph all raise here, before anything reads a reading.
    """

    def __init__(self, raw, text="", path=_YAML_NAME):
        self.path = str(path)
        self.marks = _marks(text) if text else {}
        self.staleness_hours = float(raw.get("staleness_hours", 2))
        self.balance_service_days = float(raw.get("balance_service_days", 7))
        self.earliest_plausible_reading = _parse_date(
            raw.get("earliest_plausible_reading"), "earliest_plausible_reading", path
        )
        self.sentinel_values = tuple(raw.get("sentinel_values", (-9999,)))
        self.duplicate_match_fraction = float(raw.get("duplicate_match_fraction", 0.25))
        self.flat_line_suspect_count = int(raw.get("flat_line_suspect_count", 24))
        self.flat_line_fail_count = int(raw.get("flat_line_fail_count", 96))
        self.window_steps = int(raw.get("window_steps", 24))
        self.sensor_variables = dict(raw.get("sensor_variables") or {})
        self.sensor_columns = dict(raw.get("sensor_columns") or {})
        self.sites = {}

        if not self.sensor_variables:
            raise _problem(
                path,
                None,
                "the inventory declares no sensor_variables, so no probe name is valid",
                "Restore the sensor_variables block near the top of the file.",
            )

        if not self.sensor_columns:
            raise _problem(
                path,
                None,
                "the inventory declares no sensor_columns, so the grid has no shape "
                "to check rows against",
                "Restore the sensor_columns block above sites:.",
            )

        grid = raw.get("in_service")
        if grid is None:
            raise _problem(
                path,
                None,
                "there is no in_service grid",
                "Add one at the top of the file, one row per site:\n"
                "    in_service:\n"
                "      oxford:  {stage: yes, balance_feed: yes}",
            )

        self.grid = {table: self._read_row(table, row, path) for table, row in grid.items()}

        detail = raw.get("sites") or {}
        self._check_switchboard_matches_detail(grid, detail, path)

        for table, spec in detail.items():
            self.sites[table] = self._build_site(table, spec or {}, path)

        if not self.sites:
            raise _problem(path, None, "the inventory describes no sites", "")

        self._validate_topology(path)

    def _read_row(self, table, row, path):
        """One row of the grid, as {sensor: yes | no | -}."""
        if not isinstance(row, dict):
            raise _problem(
                path,
                _line_of(self.marks, ("in_service", table)),
                f"the in_service row for {table} does not list its sensors",
                f"Write the whole row on one line, for example:\n"
                f"    {table}:  {{ctd: yes, do: yes, fc: yes, ph: -}}",
            )
        return {
            sensor: _read_switch(
                self.marks, path, ("in_service", table, sensor), table, sensor, value
            )
            for sensor, value in row.items()
        }

    def _check_switchboard_matches_detail(self, switches, detail, path):
        """Every row needs a block further down, and every block needs a row."""
        for table in switches:
            if table not in detail:
                raise _problem(
                    path,
                    _line_of(self.marks, ("in_service", table)),
                    f"{table} has a row in the grid but no detail block further down",
                    f"Either delete this row, or add a {table}: block under sites:.\n"
                    f"A row with nothing behind it would silently do nothing.",
                )
        for table in detail:
            if table not in switches:
                raise _problem(
                    path,
                    _line_of(self.marks, ("sites", table, "name")),
                    f"{table} has a detail block but no row in the grid at the top",
                    f"Add a row for it in the in_service grid:\n"
                    f"    {table}:  {{ctd: yes, do: -, fc: -, ph: -}}\n"
                    f"A site with no row could never be taken out of service.",
                )

    def _build_site(self, table, spec, path):
        row = self.grid.get(table, {})

        source = spec.get("source")
        if source not in SOURCES:
            raise _problem(
                path,
                _line_of(self.marks, ("sites", table, "source")),
                f"{table} has source {source!r}, which is not a source this code knows",
                f"Write one of:\n    source: {MAYFLY}\n    source: {BALANCE}",
            )

        columns = self._check_row_columns(table, source, row, path)

        sensor_specs = spec.get("sensors") or {}
        if not sensor_specs:
            raise _problem(
                path,
                _line_of(self.marks, ("sites", table, "name")),
                f"{table} declares no sensors",
                "Give the site at least one sensor. A site with no sensors is\n"
                "silent everywhere, and reads downstream as a healthy site with\n"
                "nothing to say, which is worse than an error.",
            )

        sensors = {}
        for name, sensor_spec in sensor_specs.items():
            if name not in self.sensor_variables:
                raise _problem(
                    path,
                    _line_of(self.marks, ("sites", table, "sensors", name, "install")),
                    f"{table} lists a sensor called {name!r}, which is not a known probe",
                    f"Use one of these names:\n    "
                    f"{', '.join(sorted(self.sensor_variables))}\n"
                    f"If this is a new kind of probe, add it to sensor_variables first.",
                )
            if columns and name not in columns:
                raise _problem(
                    path,
                    _line_of(self.marks, ("sites", table, "sensors", name, "install")),
                    f"{table} is a {source} site and has no {name} column, "
                    f"but a {name} block is written here",
                    f"A {source} site carries: {', '.join(columns)}.\n"
                    f"Delete this block, or add {name} to sensor_columns if the\n"
                    f"station really does carry one now.",
                )
            sensor_spec = sensor_spec or {}
            keypath = ("sites", table, "sensors", name)

            if "in_service" in sensor_spec:
                raise _problem(
                    path,
                    _line_of(self.marks, (*keypath, "in_service")),
                    f"{table} {name} carries an in_service line down here",
                    f"Switches live in the grid at the top of the file and nowhere\n"
                    f"else. Delete this line and set the {name} column of the\n"
                    f"{table} row instead.",
                )

            install = _parse_date(
                sensor_spec.get("install"),
                f"the install date for {table} {name}",
                path,
                _line_of(self.marks, (*keypath, "install")),
            )
            removed = _parse_date(
                sensor_spec.get("removed"),
                f"the removed date for {table} {name}",
                path,
                _line_of(self.marks, (*keypath, "removed")),
            )
            if install is not None and removed is not None and removed < install:
                raise _problem(
                    path,
                    _line_of(self.marks, (*keypath, "removed")),
                    f"{table} {name} was removed on {removed.date()}, before it was "
                    f"installed on {install.date()}",
                    "Check which date is wrong and correct it. A probe cannot come\n"
                    "out before it goes in.",
                )
            if install is None and removed is not None:
                raise _problem(
                    path,
                    _line_of(self.marks, (*keypath, "removed")),
                    f"{table} {name} has a removed date but was never installed",
                    "Either give it an install date, or set removed: null.",
                )

            switch = row.get(name, YES)
            self._check_switch_against_install(table, name, switch, install, path)
            sensors[name] = Sensor(
                site=table,
                name=name,
                variables=self.sensor_variables[name],
                install=install,
                removed=removed,
                inferred=sensor_spec.get("inferred", False),
                evidence=sensor_spec.get("evidence"),
                unconfirmed=sensor_spec.get("unconfirmed"),
                switch=switch,
            )

        self._check_switched_on_sensors_exist(table, row, sensors, path)

        return Site(
            table=table,
            name=spec.get("name") or table,
            source=source,
            downstream=spec.get("downstream"),
            sensors=sensors,
            thresholds=spec.get("thresholds") or {},
        )

    def _check_row_columns(self, table, source, row, path):
        """Every row carries its whole column set, so nothing hides."""
        columns = self.sensor_columns.get(source)
        if not columns:
            return ()
        line = _line_of(self.marks, ("in_service", table))
        missing = [c for c in columns if c not in row]
        extra = [c for c in row if c not in columns]
        if missing:
            raise _problem(
                path,
                line,
                f"the {table} row is missing a column: {', '.join(missing)}",
                f"A {source} site carries {', '.join(columns)}. Write the whole row:\n"
                f"    {table}:  {{{', '.join(f'{c}: -' for c in columns)}}}\n"
                f"Use - for a probe that is not there.",
            )
        if extra:
            raise _problem(
                path,
                line,
                f"the {table} row has a column that does not belong to a "
                f"{source} site: {', '.join(extra)}",
                f"A {source} site carries {', '.join(columns)}. Delete the rest.",
            )
        return tuple(columns)

    def _check_switch_against_install(self, table, name, switch, install, path):
        """The switch and the install date cannot disagree about one probe."""
        if switch in (YES, NO) and install is None:
            raise _problem(
                path,
                _line_of(self.marks, ("in_service", table, name)),
                f"the grid says {table} has a {name} probe, but no install date "
                f"is written for it below",
                f"Add the date the probe went in, under sites: {table}: sensors: {name}:\n"
                f"      {name}:\n"
                f"        install: '2026-03-05'\n"
                f"        removed: null\n"
                f"Use the real date. If there is no {name} probe here, set the\n"
                f"{name} column of the {table} row back to -.",
            )
        if switch == ABSENT and install is not None:
            raise _problem(
                path,
                _line_of(self.marks, ("in_service", table, name)),
                f"the grid says {table} has no {name} probe, but an install date "
                f"of {install.date()} is written for it below",
                f"If the probe is there, set the {name} column of the {table} row\n"
                f"to yes, or to no if it is switched off. If it never existed,\n"
                f"set its install date below to null.",
            )

    def _check_switched_on_sensors_exist(self, table, row, sensors, path):
        """A column set to yes or no with no block under sites: at all."""
        for name, switch in row.items():
            if switch in (YES, NO) and name not in sensors:
                self._check_switch_against_install(table, name, switch, None, path)

    def _validate_topology(self, path):
        """Refuse a dangling downstream, a missing outlet, or a loop."""
        for table, site in self.sites.items():
            line = _line_of(self.marks, ("sites", table, "downstream"))
            if site.downstream is None:
                if table not in TERMINAL_SITES:
                    raise _problem(
                        path,
                        line,
                        f"{table} has no downstream neighbour",
                        f"Write the site the water flows to next, for example:\n"
                        f"    downstream: oxford\n"
                        f"Only {' and '.join(TERMINAL_SITES)} are allowed to have none.",
                    )
                continue
            if site.downstream not in self.sites:
                raise _problem(
                    path,
                    line,
                    f"{table} drains to {site.downstream!r}, which is not a site in this file",
                    f"Check the spelling. The sites in this file are:\n    "
                    f"{', '.join(sorted(self.sites))}",
                )
            if site.downstream == table:
                raise _problem(
                    path,
                    line,
                    f"{table} drains to itself",
                    "Write the site downstream of it, or null if it is an outlet.",
                )

        for table in self.sites:
            seen = [table]
            current = self.sites[table].downstream
            while current is not None:
                if current in seen:
                    raise _problem(
                        path,
                        _line_of(self.marks, ("sites", current, "downstream")),
                        f"the flow graph loops back on itself: {' -> '.join([*seen, current])}",
                        "Water cannot flow in a circle. Follow that chain and fix\n"
                        "whichever downstream line points backwards.",
                    )
                seen.append(current)
                current = self.sites[current].downstream

    # Reading the inventory

    def site(self, table):
        try:
            return self.sites[table]
        except KeyError:
            raise InventoryError(
                f"unknown site {table!r}; the inventory describes {', '.join(sorted(self.sites))}"
            ) from None

    @property
    def tables(self):
        return sorted(self.sites)

    def scored_sites(self):
        """The sites live scoring is allowed to look at."""
        return [t for t in self.tables if self.sites[t].in_service]

    def grid_rows(self):
        """The grid in file order, as (site, {sensor: yes | no | -})."""
        return [(table, dict(self.grid[table])) for table in self.grid]

    def switched_off(self):
        """Every probe a person has set to no, as (site, sensor)."""
        return [
            (table, name)
            for table in self.tables
            for name in sorted(self.sites[table].sensors)
            if self.sites[table].sensors[name].switch == NO
        ]

    def out_of_service(self):
        return [t for t in self.tables if not self.sites[t].in_service]

    def edges(self):
        """Flow edges as (upstream, downstream) pairs, sorted."""
        return sorted(
            (table, site.downstream)
            for table, site in self.sites.items()
            if site.downstream is not None
        )

    def isolated(self):
        """Sites with no edge either way, which is what makes a control."""
        connected = set()
        for up, down in self.edges():
            connected.add(up)
            connected.add(down)
        return sorted(set(self.sites) - connected)

    def upstream_of(self, table):
        """Every site that drains directly into one site."""
        return sorted(t for t, s in self.sites.items() if s.downstream == table)

    def upstream_chain(self, table):
        """Every site whose water reaches this one, direct or not."""
        return sorted(t for t in self.sites if table in self.downstream_chain(t))

    def downstream_chain(self, table):
        """The ordered path from one site to its outlet, excluding itself."""
        chain = []
        current = self.site(table).downstream
        while current is not None:
            chain.append(current)
            current = self.sites[current].downstream
        return chain

    def unconfirmed(self):
        """Every entry a person still has to answer, as (site, sensor, question)."""
        out = []
        for table in self.tables:
            for name in sorted(self.sites[table].sensors):
                sensor = self.sites[table].sensors[name]
                if sensor.unconfirmed:
                    out.append((table, name, sensor.unconfirmed))
        return out

    def inferred_dates(self):
        """Every install date guessed from data, as (site, sensor, date, evidence)."""
        out = []
        for table in self.tables:
            for name in sorted(self.sites[table].sensors):
                sensor = self.sites[table].sensors[name]
                if sensor.inferred and sensor.install is not None:
                    out.append((table, name, sensor.install.date().isoformat(), sensor.evidence))
        return out

    def stale_inferred_markers(self):
        """
        Real dates that still carry the guessed marker.

        The evidence names the reading the guess came from, so a date that no
        longer matches it is a human correction with the label left on.
        """
        out = []
        for table, name, date, evidence in self.inferred_dates():
            if evidence and date not in evidence:
                out.append((table, name, date, evidence))
        return out

    # State resolution

    def _switched_off(self, sensor, in_service_only):
        return in_service_only and not sensor.in_service

    def _expected_absence(self, site, timestamp, as_of):
        """
        Whether an absent reading at a balance site is the 7 day window.

        Those sites are scraped and serve a rolling week, so an older hole was
        never retrievable and nobody can call it broken.
        """
        if site.source != BALANCE or as_of is None:
            return False
        return timestamp < as_of - pd.Timedelta(days=self.balance_service_days)

    def resolve_state(
        self, table, variable, timestamp, last_reading=None, as_of=None, in_service_only=True
    ):
        """
        One (site, variable, timestamp) to exactly one state.

        last_reading is the newest value at or before this step, passed in
        rather than looked up so this stays callable from a loop. A probe set
        to no comes back NOT_INSTALLED; in_service_only=False is the training
        path, which cares when the probe was really there.
        """
        site = self.site(table)
        sensor = site.sensor_for(variable)
        if sensor is None or not sensor.covers(timestamp):
            return NOT_INSTALLED
        if self._switched_off(sensor, in_service_only):
            return NOT_INSTALLED

        if last_reading is not None and last_reading == timestamp:
            return PRESENT

        if self._expected_absence(site, timestamp, as_of):
            return NOT_INSTALLED

        if last_reading is None:
            return MISSING

        age = timestamp - last_reading
        if age <= pd.Timedelta(hours=self.staleness_hours):
            return STALE
        return MISSING

    def resolve_series(self, table, variable, values, as_of=None, in_service_only=True):
        """
        A whole time indexed series to one state per timestep.

        values is indexed by UTC timestamp and holds NaN where nothing arrived.
        in_service_only=False ignores the grid, because history does not change
        when a probe is switched off.
        """
        values = pd.Series(values).sort_index()
        index = pd.DatetimeIndex(values.index)
        site = self.site(table)
        sensor = site.sensor_for(variable)

        states = np.full(len(index), NOT_INSTALLED, dtype=object)
        if sensor is None or not sensor.ever_installed or len(index) == 0:
            return pd.Series(states, index=index)
        if self._switched_off(sensor, in_service_only):
            return pd.Series(states, index=index)

        present = values.notna().to_numpy()
        installed = index >= sensor.install
        if sensor.removed is not None:
            installed &= index < sensor.removed

        # Nanoseconds, not timestamps, so a channel that never reported still
        # subtracts cleanly instead of landing in object dtype. as_unit first:
        # pandas takes the resolution off the input, and a microsecond index
        # read every gap as inside the allowed two hours.
        stamps = index.as_unit("ns").asi8.astype("float64")
        last = pd.Series(np.where(present, stamps, np.nan)).ffill().to_numpy()
        age = stamps - last

        as_of_stamp = None if as_of is None else pd.Timestamp(as_of)
        if as_of_stamp is not None and as_of_stamp.tzinfo is None:
            as_of_stamp = as_of_stamp.tz_localize("UTC")
        expected_gap = np.zeros(len(index), dtype=bool)
        if site.source == BALANCE and as_of_stamp is not None:
            cutoff = as_of_stamp - pd.Timedelta(days=self.balance_service_days)
            expected_gap = np.asarray(index < cutoff)

        stale = np.isfinite(age) & (age <= self.staleness_hours * 3600 * 1e9)

        states[installed & present] = PRESENT
        states[installed & ~present & expected_gap] = NOT_INSTALLED
        states[installed & ~present & ~expected_gap & stale] = STALE
        states[installed & ~present & ~expected_gap & ~stale] = MISSING
        return pd.Series(states, index=index)

    def state_counts(self, frames, as_of=None, in_service_only=True):
        """
        Every state per (site, variable) over an archive.

        frames is {table: DataFrame indexed by UTC timestamp}. A probe set to no
        is left out entirely, and a site whose probes are all off leaves with
        them. That is what switching one off means.
        """
        rows = []
        tables = self.scored_sites() if in_service_only else self.tables
        for table in tables:
            site = self.sites[table]
            frame = frames.get(table)
            variables = site.scored_variables if in_service_only else site.variables
            for variable in variables:
                if frame is None or len(frame) == 0:
                    rows.append(
                        {"site": table, "variable": variable, "state": NOT_INSTALLED, "count": 0}
                    )
                    continue
                column = (
                    frame[variable]
                    if variable in frame.columns
                    else pd.Series(np.nan, index=frame.index)
                )
                states = self.resolve_series(
                    table, variable, column, as_of=as_of, in_service_only=in_service_only
                )
                counts = states.value_counts()
                for state in STATES:
                    rows.append(
                        {
                            "site": table,
                            "variable": variable,
                            "state": state,
                            "count": int(counts.get(state, 0)),
                        }
                    )
        return pd.DataFrame(rows)


_CACHE = {}


def _yaml_path(path=None):
    return path or os.path.join(os.path.dirname(__file__), _YAML_NAME)


def load(path=None, reload=False):
    """Load and validate the inventory, caching by resolved path."""
    resolved = _yaml_path(path)
    if reload or resolved not in _CACHE:
        try:
            with open(resolved) as handle:
                text = handle.read()
        except FileNotFoundError:
            raise InventoryError(f"no inventory at {resolved}") from None
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise InventoryError(f"{resolved} did not parse to a mapping")
        _CACHE[resolved] = Inventory(raw, text=text, path=resolved)
    return _CACHE[resolved]


def from_dict(raw, text="", path=_YAML_NAME):
    """From an already parsed mapping, for tests."""
    return Inventory(raw, text=text, path=path)


def from_text(text, path=_YAML_NAME):
    """From YAML text, so line numbers are real."""
    return Inventory(yaml.safe_load(text), text=text, path=path)


# Editing the file


def _grid_lines(rows):
    """
    The grid as aligned text, one line per row. File and report share it.

    Padding goes after the comma, not before: a bare - has to sit against the
    comma or the brace or PyYAML reads it as the start of a list.
    """
    name_width = max((len(table) for table, _ in rows), default=0) + 1
    widths = {}
    for _, cells in rows:
        columns = tuple(cells)
        for sensor, switch in cells.items():
            key = (columns, sensor)
            widths[key] = max(widths.get(key, 0), len(switch))

    lines = {}
    for table, cells in rows:
        columns = tuple(cells)
        parts = []
        for i, (sensor, switch) in enumerate(cells.items()):
            if i == len(cells) - 1:
                parts.append(f"{sensor}: {switch}")
            else:
                pad = len(sensor) + 4 + widths[(columns, sensor)]
                parts.append(f"{sensor}: {switch},".ljust(pad))
        lines[table] = f"  {(table + ':').ljust(name_width + 1)} {{{''.join(parts)}}}"
    return lines


def _rewrite_row(path, table, changes):
    """
    Rewrite one row in place, leaving every other line alone.

    Rebuilds the row rather than dumping the document back out. A dump loses the
    comments, and the comments are why anyone can edit this file at all.
    """
    with open(path) as handle:
        lines = handle.readlines()

    marks = _marks("".join(lines))
    row_node = marks.get(("in_service", table))
    if row_node is None:
        raise InventoryError(f"{table} is not in the in_service grid of {path}")

    cells = {key.value: value for key, value in row_node.value}
    on_one_line = {node.start_mark.line for node in cells.values()} | {row_node.start_mark.line}
    if len(on_one_line) != 1:
        raise InventoryError(
            f"the {table} row in {path} is split across lines; put it back on one "
            f"line before switching a sensor from code"
        )

    switches = {}
    for sensor, node in cells.items():
        switches[sensor] = node.value
    for sensor, value in changes.items():
        if sensor not in switches:
            raise InventoryError(f"{table} has no {sensor} column in the grid of {path}")
        if switches[sensor] == ABSENT:
            raise InventoryError(
                f"{table} {sensor} is set to - , meaning there is no such probe. "
                f"Add its install date under sites: before switching it on."
            )
        switches[sensor] = YES if value else NO

    rows = []
    for other in _all_grid_rows(marks):
        rows.append((other, switches if other == table else _row_switches(marks, other)))
    lines[row_node.start_mark.line] = _grid_lines(rows)[table] + "\n"

    with open(path, "w") as handle:
        handle.writelines(lines)
    _CACHE.pop(path, None)


def _all_grid_rows(marks):
    """Row names in file order, so widths match what is already written."""
    seen = []
    for keypath in marks:
        if len(keypath) == 2 and keypath[0] == "in_service" and keypath[1] not in seen:
            seen.append(keypath[1])
    return seen


def _row_switches(marks, table):
    node = marks[("in_service", table)]
    return {key.value: value.value for key, value in node.value}


def set_in_service(table, sensor, value, path=None):
    """Flip one sensor's cell in the grid."""
    _rewrite_row(_yaml_path(path), table, {sensor: value})


def set_site_in_service(table, value, path=None):
    """
    A whole site in or out of service, meaning every fitted probe on it.

    No site level switch exists in the file on purpose, so this sets each column
    with a probe behind it and leaves the - alone.
    """
    resolved = _yaml_path(path)
    switches = _row_switches(_marks(open(resolved).read()), table)
    _rewrite_row(
        resolved, table, {s: value for s, current in switches.items() if current != ABSENT}
    )


# The command line view


def _yes_no(flag):
    return "yes" if flag else "NO"


def _table(rows, headers):
    """Fixed width columns, no dependencies."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)).rstrip())
    return "\n".join(out)


def _archive_disagreements(inventory):
    """
    Where the file and the data disagree. Reported, never resolved.

    Silence is measured against the newest reading in the archive, not the
    clock, so a stale local copy does not report every site as down.
    """
    try:
        from strawberrywatch.ingest.raw_data_loader import load_archive_by_table

        tables, dropped = load_archive_by_table(
            earliest=inventory.earliest_plausible_reading, with_report=True
        )
    except Exception as exc:
        return [], [
            f"could not read data/raw_data, so the file was not checked "
            f"against the data ({type(exc).__name__}: {exc})"
        ]

    notes = []
    latest = {t: f.index.max() for t, f in tables.items() if len(f)}
    if not latest:
        return dropped, ["data/raw_data has no readings to check the file against"]

    as_of = max(latest.values())
    gap = pd.Timedelta(hours=inventory.staleness_hours)
    notes.append(
        f"newest reading anywhere in data/raw_data is {as_of:%Y-%m-%d %H:%M}Z; "
        f"silence below is measured against that, not against today"
    )

    for table in inventory.tables:
        site = inventory.sites[table]
        seen = latest.get(table)
        if site.in_service and (seen is None or as_of - seen > gap):
            when = "never" if seen is None else f"{seen:%Y-%m-%d %H:%M}Z"
            notes.append(f"{table} is in service but its last reading was {when}")
        if not site.in_service and seen is not None and as_of - seen <= gap:
            notes.append(
                f"{table} is out of service but is still reporting, last at {seen:%Y-%m-%d %H:%M}Z"
            )
    return dropped, notes


def report(inventory=None, check_data=True):
    """The whole file as plain text, for checking an edit worked."""
    inventory = inventory or load(reload=True)
    out = [
        "Strawberry Creek sensor inventory",
        inventory.path,
        "",
        "WHAT IS RUNNING WHERE   (yes in service, no switched off, - no such probe)",
        "",
    ]

    grid = _grid_lines(inventory.grid_rows())
    out += [grid[table] for table, _ in inventory.grid_rows()]

    off = inventory.switched_off()
    out += ["", "SWITCHED OFF"]
    if off:
        out.append(
            f"{len(off)} probe(s) set to no: {', '.join(f'{t} {n}' for t, n in off)}. "
            f"They are dropped from live scoring and from the state table. "
            f"Their older readings are still used for training."
        )
        dark = inventory.out_of_service()
        if dark:
            out.append(f"every probe is off at: {', '.join(dark)}, so the site is out of service")
    else:
        out.append("every probe in the grid is in service")

    out += ["", "SITES"]
    rows = []
    for table in inventory.tables:
        site = inventory.sites[table]
        rows.append(
            [
                table,
                _yes_no(site.in_service),
                site.source,
                site.downstream or "(outlet)",
                ", ".join(sorted(site.sensors)),
            ]
        )
    out.append(_table(rows, ["site", "in service", "source", "drains to", "sensors"]))

    out += ["", "SENSORS", "What the grid cannot show: when each probe went in."]
    rows = []
    for table in inventory.tables:
        for name in sorted(inventory.sites[table].sensors):
            sensor = inventory.sites[table].sensors[name]
            if sensor.install is None:
                installed = "never fitted"
            else:
                installed = sensor.install.date().isoformat()
            rows.append(
                [
                    table,
                    name,
                    sensor.switch,
                    installed,
                    "GUESSED" if sensor.inferred and sensor.install else "",
                    sensor.removed.date().isoformat() if sensor.removed else "",
                ]
            )
    out.append(_table(rows, ["site", "sensor", "grid", "installed", "date is", "removed"]))

    guessed = inventory.inferred_dates()
    out += ["", f"GUESSED INSTALL DATES ({len(guessed)})"]
    out.append(
        "These were taken from the first reading in the archive. Replace any you "
        "know the real date for, then delete the inferred line under it."
    )
    for table, name, date, evidence in guessed:
        out.append(f"  {table} {name}: {date}   from {evidence}")

    stale = inventory.stale_inferred_markers()
    if stale:
        out += ["", "GUESSED MARKERS THAT LOOK OUT OF DATE"]
        for table, name, date, evidence in stale:
            out.append(
                f"  {table} {name}: install is {date} but the evidence still reads "
                f"'{evidence}'. If you corrected this date, delete its inferred line."
            )

    if check_data:
        dropped, notes = _archive_disagreements(inventory)
        if dropped:
            out += ["", "ROWS DROPPED FROM THE ARCHIVE"]
            for entry in dropped:
                out.append(
                    f"  {entry['table']}: {entry['rows']} row(s) stamped before "
                    f"{entry['earliest']}, earliest {entry['worst']}. "
                    f"{entry['reason']}"
                )
        out += ["", "FILE AGAINST DATA"]
        out += [f"  {n}" for n in notes] or ["  nothing to report"]

    unconfirmed = inventory.unconfirmed()
    out += ["", f"STILL UNCONFIRMED ({len(unconfirmed)})"]
    for table, name, question in unconfirmed:
        out.append(f"  {table} {name}: {question}")

    return "\n".join(out)


def main(argv=None):
    """Print the inventory, or whatever a bad edit produced, and exit."""
    argv = sys.argv[1:] if argv is None else argv
    try:
        text = report(check_data="--no-data" not in argv)
    except InventoryError as exc:
        print("The inventory file has a problem and nothing will run until it is fixed.\n")
        print(exc)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSENT",
    "BALANCE",
    "MAYFLY",
    "MISSING",
    "NOT_INSTALLED",
    "PRESENT",
    "SOURCES",
    "NO",
    "STALE",
    "STATES",
    "SWITCHES",
    "YES",
    "TERMINAL_SITES",
    "Inventory",
    "InventoryError",
    "Sensor",
    "Site",
    "from_dict",
    "from_text",
    "load",
    "report",
    "set_in_service",
    "set_site_in_service",
]

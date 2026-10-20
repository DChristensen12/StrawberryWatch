"""
Where readings come from. The website's sensor database, with the Strawberry
Creek API as the local app's backup. Labeled events can be replayed too, for
load_test.py and the test dashboard.

Each one returns Night Heron shaped tables, {table: frame indexed by UTC time
with the raw logger columns}. The serving detectors read these.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

API = "api"
SITE = "site"
CSV = "csv"
REPLAY = "replay"
# Live sources. Replays are the past and don't show up as today's alerts.
LIVE = (API, SITE)

UNREACHABLE = "Couldn't reach the database"

EVENTS_DIR = Path(__file__).resolve().parent / "events"
# We refuse a CSV bigger than this. A year of 15 minute readings at one site
# is a few MB.
MAX_CSV_BYTES = 50 * 1024 * 1024
TIME_COLUMNS = ("DateTimeUTC", "timestamp", "datetime")
WEATHER_COLUMNS = ("rain_mm", "air_temp_c", "shortwave_radiation")
# The corpus calls the Wickson Footbridge footbridge. Night Heron calls it scnf010.
FILE_ALIASES = {"scnf010": ("scnf010", "footbridge")}


class SourceError(RuntimeError):
    """Couldn't get readings. The message says why, in plain words."""


def is_live(kind):
    return kind in LIVE


# Raw logger columns for each inventory variable, the reverse of
# raw_data_loader's mapping, plus the battery for the site page
def _raw_columns():
    from strawberrywatch.ingest import raw_data_loader

    back = {name: raw for raw, name in raw_data_loader._ALL_CHANNEL_MAPPING.items()}
    return back


def tidy(table):
    """One raw table under the inventory's variable names, plus the battery, for display."""
    from strawberrywatch.ingest.raw_data_loader import tidy_channels

    tidied = tidy_channels(table.copy())
    if "EnviroDIY_Mayfly_Batt" in table.columns:
        battery = pd.to_numeric(table["EnviroDIY_Mayfly_Batt"], errors="coerce")
        tidied["battery"] = battery[~battery.index.duplicated(keep="first")].reindex(tidied.index)
    return tidied


def on_grid(frame, step="15min"):
    """
    Readings snapped to their quarter hour, averaged where two land together.
    regrid does the same before the model reads a window, but detect_anomalies
    checks who reported against the unsnapped frame. A Mayfly stamped 07:00:31
    wouldn't count as reporting. Snapping here keeps the two in step.
    """
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.index = numeric.index.floor(step)
    return numeric.groupby(level=0).mean()


def _indexed(frame, where):
    """A frame read from somewhere, indexed by UTC reading time, or SourceError."""
    column = next((c for c in TIME_COLUMNS if c in frame.columns), None)
    if column is None:
        raise SourceError(f"{where} has no time column. It needs one of {', '.join(TIME_COLUMNS)}.")
    stamps = pd.to_datetime(frame.pop(column), utc=True, errors="coerce")
    frame.index = stamps
    frame = frame[frame.index.notna()]
    return frame[~frame.index.duplicated(keep="first")].sort_index()


class Source:
    kind = ""
    # Which folder, event or database. Live and replay results can be told apart later
    detail = None
    # A replay's readings don't change. We check it once, not every 15 minutes
    periodic = True

    def describe(self):
        raise NotImplementedError

    def window(self, lookback):
        """(start, end) of the readings a check reads, as UTC Timestamps."""
        raise NotImplementedError

    def tables(self, names, start, end):
        raise NotImplementedError

    def weather(self, start, end):
        """Open-Meteo weather for the window, same as the models trained on. None if unavailable."""
        try:
            from strawberrywatch.ingest.historical_weather_client import fetch_open_meteo_weather

            frame = fetch_open_meteo_weather(start.to_pydatetime(), end.to_pydatetime())
        except Exception as exc:
            logger.warning("weather fetch failed (%s), checking without it", type(exc).__name__)
            return None
        return None if frame is None or frame.empty else frame


class ApiSource(Source):
    kind = API

    def __init__(self, token):
        if not token:
            raise SourceError("There's no API token in .env (SCMG_API_TOKEN).")
        self._token = token

    def __repr__(self):
        # Not the token
        return "<ApiSource>"

    def describe(self):
        return "the Strawberry Creek API"

    def window(self, lookback):
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        return end - lookback, end

    def tables(self, names, start, end):
        from strawberrywatch import inventory as inv
        from strawberrywatch.ingest.api_client import fetch_creek_data

        inventory = inv.load()
        raw = _raw_columns()
        out = {}
        for name in names:
            try:
                site = inventory.site(name)
                columns = [raw[v] for v in site.scored_variables if v in raw]
            except Exception:
                columns = None
            frame = fetch_creek_data(
                name,
                start.to_pydatetime(),
                end.to_pydatetime(),
                variables=(columns + ["EnviroDIY_Mayfly_Batt"]) if columns else None,
                token=self._token,
            )
            if frame.empty:
                continue
            frame = frame.drop(
                columns=[c for c in ("station_id", "uuid", "site_code") if c in frame.columns]
            )
            out[name] = on_grid(_indexed(frame, f"The API's answer for {name}"))
        return out


class CsvSource(Source):
    """
    A folder with one CSV per site, the way replays read an event. File names
    come from the inventory, and only files in the folder get opened.
    """

    kind = CSV

    def __init__(self, folder):
        folder = Path(folder or "").expanduser()
        if not str(folder) or not folder.is_dir():
            raise SourceError(f"There's no folder at {folder}.")
        self.folder = folder.resolve()
        self.detail = str(self.folder)
        self._cache = None

    def describe(self):
        return f"the CSV files in {self.folder}"

    def _path(self, name):
        for stem in FILE_ALIASES.get(name, (name,)):
            path = (self.folder / f"{stem}.csv").resolve()
            # A link inside the folder that points out of it is still outside it
            if path.parent != self.folder or not path.is_file():
                continue
            if path.stat().st_size > MAX_CSV_BYTES:
                raise SourceError(
                    f"{path.name} is over {MAX_CSV_BYTES // (1024 * 1024)} MB, so it wasn't read."
                )
            return path
        return None

    def _read(self, names):
        if self._cache is None:
            self._cache = {}
        for name in names:
            if name in self._cache:
                continue
            path = self._path(name)
            if path is None:
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                raise SourceError(
                    f"{path.name} couldn't be read as a CSV: {type(exc).__name__}"
                ) from None
            if not set(frame.columns) & set(_raw_columns().values()):
                raise SourceError(
                    f"{path.name} has none of the sensor columns StrawberryWatch reads, "
                    f"such as Meter_Hydros21_Cond."
                )
            self._cache[name] = _indexed(frame, path.name)
        return {name: self._cache[name] for name in names if name in self._cache}

    def _all(self):
        from strawberrywatch import inventory as inv

        return self._read(list(inv.load().sites))

    def window(self, lookback):
        frames = [f for f in self._all().values() if len(f)]
        if not frames:
            raise SourceError(f"None of the files in {self.folder} is named after a site.")
        end = max(f.index.max() for f in frames).floor("15min")
        return end - lookback, end

    def tables(self, names, start, end):
        out = {}
        for name, frame in self._read(names).items():
            part = frame[(frame.index >= start) & (frame.index <= end)]
            part = part.drop(columns=[c for c in WEATHER_COLUMNS if c in part.columns])
            if len(part):
                out[name] = on_grid(part)
        return out

    def weather(self, start, end):
        carried = _carried_weather(self._all(), start, end)
        return carried if carried is not None else super().weather(start, end)


class ReplaySource(Source):
    """A labeled event, read in full and checked at once."""

    kind = REPLAY
    periodic = False

    def __init__(self, folder):
        events = {e["folder"]: e for e in replay_events()}
        if folder not in events:
            raise SourceError(f"There's no event called {folder!r} to replay.")
        self.event = events[folder]
        self.detail = folder
        self._csv = CsvSource(EVENTS_DIR / folder)

    def describe(self):
        return f"a replay of {self.event['title']}"

    def window(self, _lookback):
        # All of the event, however long, to include the labeled moment
        frames = [f for f in self._csv._all().values() if len(f)]
        start = min(f.index.min() for f in frames).floor("15min")
        end = max(f.index.max() for f in frames).floor("15min")
        return start, end

    def tables(self, names, start, end):
        return self._csv.tables(names, start, end)

    def weather(self, start, end):
        return _carried_weather(self._csv._all(), start, end)


class SqlSource(Source):
    """
    The website's sensor tables, read the way gnn_alerts.py reads them, with
    the MYSQL settings in .env that Night Heron already uses.
    """

    kind = SITE

    def __init__(self):
        self._credentials = None

    def __repr__(self):
        return "<SqlSource>"

    def describe(self):
        return "the website's sensor database"

    def window(self, lookback):
        end = pd.Timestamp.now(tz="UTC").floor("15min")
        return end - lookback, end

    def tables(self, names, start, end):
        from strawberrywatch.ingest import sql_client

        try:
            sql_client.ping(self._credentials)
        except Exception:
            raise SourceError(UNREACHABLE) from None
        out = {}
        for name in names:
            frame = sql_client.fetch_creek_data_sql(
                name, start.to_pydatetime(), end.to_pydatetime(), credentials=self._credentials
            )
            if frame.empty:
                continue
            frame = frame.drop(
                columns=[c for c in ("station_id", "uuid", "site_code") if c in frame.columns]
            )
            out[name] = on_grid(_indexed(frame, f"The table for {name}"))
        return out


class WithBackup(Source):
    """
    The sensor database, falling back to the API when it can't be reached. It
    picks one as soon as a check asks for its window. The check is recorded
    under whichever answered.
    """

    def __init__(self, primary, backup):
        self.primary, self.backup = primary, backup
        self.used = primary
        self._tables = None

    @property
    def kind(self):
        return self.used.kind

    @property
    def detail(self):
        return self.used.detail

    def describe(self):
        return self.used.describe()

    def window(self, lookback):
        from strawberrywatch import inventory as inv

        start, end = self.primary.window(lookback)
        try:
            self._tables = self.primary.tables(list(inv.load().sites), start, end)
        except Exception as exc:
            logger.warning("%s, reading the API instead", _plain(exc))
            self.used = self.backup
            return self.backup.window(lookback)
        return start, end

    def tables(self, names, start, end):
        if self.used is self.primary and self._tables is not None:
            return {name: frame for name, frame in self._tables.items() if name in names}
        return self.used.tables(names, start, end)

    def weather(self, start, end):
        return self.used.weather(start, end)


def _plain(exc):
    return str(exc) if isinstance(exc, SourceError) else type(exc).__name__


def _carried_weather(frames, start, end):
    """Weather columns the files carry themselves, one row per quarter hour, or None."""
    parts = []
    for frame in frames.values():
        cols = [c for c in WEATHER_COLUMNS if c in frame.columns]
        if cols:
            parts.append(frame[cols].apply(pd.to_numeric, errors="coerce"))
    if not parts:
        return None
    weather = pd.concat(parts)
    weather = weather[(weather.index >= start) & (weather.index <= end)]
    if weather.empty:
        return None
    weather.index = weather.index.floor("15min")
    return weather.groupby(level=0).mean()


def replay_events():
    return json.loads((EVENTS_DIR / "events.json").read_text())["events"]


def from_config(config, token=None):
    """
    The Source for a store's saved choice. token is the Strawberry Creek API
    token. It comes from .env, not the store.
    """
    if not config:
        raise SourceError("The data source hasn't been set up yet.")
    if config["kind"] == API:
        return ApiSource(token)
    if config["kind"] == SITE:
        return WithBackup(SqlSource(), ApiSource(token)) if token else SqlSource()
    if config["kind"] == REPLAY:
        return ReplaySource(config.get("event"))
    raise SourceError(f"Unknown source {config['kind']!r}")

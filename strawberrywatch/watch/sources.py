"""
Where readings come from: the Strawberry Creek API with the user's own token, a
folder of CSV files, or a replay of a labeled event shipped with the package.

Each hands back Night Heron shaped tables, {table: frame indexed by UTC time
with the raw logger columns}, which is what every serving detector reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

API = "api"
CSV = "csv"
REPLAY = "replay"

KEYRING_SERVICE = "strawberrywatch"
KEYRING_USER = "strawberry-creek-api"
TOKEN_HELP = "https://www.strawberrycreek.org/api/"

EVENTS_DIR = Path(__file__).resolve().parent / "events"
# A CSV bigger than this is refused rather than read. A year of 15 minute
# readings at one site is a few MB.
MAX_CSV_BYTES = 50 * 1024 * 1024
TIME_COLUMNS = ("DateTimeUTC", "timestamp", "datetime")
WEATHER_COLUMNS = ("rain_mm", "air_temp_c", "shortwave_radiation")
# The corpus calls the Wickson Footbridge footbridge. Night Heron calls it scnf010.
FILE_ALIASES = {"scnf010": ("scnf010", "footbridge")}


class SourceError(RuntimeError):
    """Readings can't be had from here, and the message says why in plain words."""


# The token lives in the system keychain and nowhere else


def save_token(token):
    keyring = _keyring()
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USER, token)
    except Exception as exc:
        raise SourceError(
            f"This computer's keychain wouldn't save the token: {type(exc).__name__}"
        ) from None


def get_token():
    keyring = _keyring()
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception as exc:
        raise SourceError(
            f"This computer's keychain couldn't be read: {type(exc).__name__}"
        ) from None


def delete_token():
    keyring = _keyring()
    try:
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:
        pass  # already gone is the same as gone


def _keyring():
    try:
        import keyring
    except ImportError:
        raise SourceError(
            'Saving a token needs the keyring package, from pip install "strawberrywatch[dashboard]"'
        ) from None
    return keyring


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
    Readings snapped to the quarter hour they fall in, averaged where two land
    together. serving.windows.regrid does the same before the model reads a
    window, but detect_anomalies checks which sites really reported against the
    unsnapped frame, so a Mayfly stamped 07:00:31 never counts as reporting and
    its site is never judged. Snapping here, as readings come in, keeps the two
    in agreement.
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
    # A replay's readings never change, so it's checked once rather than every 15 minutes
    periodic = True

    def describe(self):
        raise NotImplementedError

    def window(self, lookback):
        """(start, end) of the readings a check reads, as UTC Timestamps."""
        raise NotImplementedError

    def tables(self, names, start, end):
        raise NotImplementedError

    def weather(self, start, end):
        """Open-Meteo for the window, the same weather the models trained with. None if it can't be had."""
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
            raise SourceError("There's no API token saved yet.")
        self._token = token

    def __repr__(self):
        # Never the token
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
    One CSV per site in a folder the user chose, named by the site, with a time
    column and the raw logger columns. Nothing outside that folder is opened:
    the file names come from the inventory, never from what was typed.
    """

    kind = CSV

    def __init__(self, folder):
        folder = Path(folder or "").expanduser()
        if not str(folder) or not folder.is_dir():
            raise SourceError(f"There's no folder at {folder}.")
        self.folder = folder.resolve()
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
    """A labeled event, read whole and checked at once."""

    kind = REPLAY
    periodic = False

    def __init__(self, folder):
        events = {e["folder"]: e for e in replay_events()}
        if folder not in events:
            raise SourceError(f"There's no event called {folder!r} to replay.")
        self.event = events[folder]
        self._csv = CsvSource(EVENTS_DIR / folder)

    def describe(self):
        return f"a replay of {self.event['title']}"

    def window(self, _lookback):
        # The whole event, however long, so the moment it's known for is in it
        frames = [f for f in self._csv._all().values() if len(f)]
        start = min(f.index.min() for f in frames).floor("15min")
        end = max(f.index.max() for f in frames).floor("15min")
        return start, end

    def tables(self, names, start, end):
        return self._csv.tables(names, start, end)

    def weather(self, start, end):
        return _carried_weather(self._csv._all(), start, end)


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
    """The Source a saved choice describes. The token comes from the keychain, never the store."""
    if not config:
        raise SourceError("Readings haven't been set up yet.")
    if config["kind"] == API:
        return ApiSource(token if token is not None else get_token())
    if config["kind"] == CSV:
        return CsvSource(config.get("folder"))
    if config["kind"] == REPLAY:
        return ReplaySource(config.get("event"))
    raise SourceError(f"Unknown source {config['kind']!r}")

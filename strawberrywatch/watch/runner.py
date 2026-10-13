"""
One check, and the thread that runs one every 15 minutes while the app is open.

Built from gnn_alerts.py's proven parts without importing it: the interval, a
background thread, a busy guard so two checks never overlap, and a check that
logs a failure instead of raising. Its MySQL reads are sources.py here, and
nothing leaves the computer: what a check notices goes in the store and on
the page, nowhere else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch.watch import decide, sources
from strawberrywatch.watch import settings as settings_mod

logger = logging.getLogger(__name__)

INTERVAL = timedelta(minutes=settings_mod.INTERVAL_MINUTES)
LOOKBACK = timedelta(days=settings_mod.LOOKBACK_DAYS)
HASHES_PATH = Path(__file__).resolve().parent / "model_files.sha256.json"


class ModelFileError(RuntimeError):
    """A model's files aren't the ones it was tested with."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(cls, folder, hashes=None):
    """Every file a model loads, checked against the hash it shipped with, before torch sees it."""
    hashes = hashes if hashes is not None else json.loads(HASHES_PATH.read_text())
    for name in cls.FILES:
        path = Path(folder) / name
        if not path.is_file():
            raise ModelFileError(f"{name} isn't in {folder}")
        expected = hashes.get(name)
        if expected is None:
            raise ModelFileError(f"{name} has no recorded hash, so it can't be checked")
        if sha256(path) != expected:
            raise ModelFileError(
                f"{name} doesn't match the file it was tested with. If it was retrained on "
                f"purpose, record its new hash in {HASHES_PATH.name}."
            )


def load_detector(name, folder=None):
    """A serving detector, cached, after its files pass verify_files."""
    from strawberrywatch import paths
    from strawberrywatch.serving import detector_class

    folder = folder or paths.checkpoints_dir()
    cls = detector_class(name)
    verify_files(cls, folder)
    return cls.cached(folder, device="cpu")


def configure_rain(detector, model, rain_on, multiplier):
    """Hand a detector this check's rain settings, as rain_params or a rain_gate dict."""
    if hasattr(detector, "set_rain_gate"):
        detector.set_rain_gate(
            {"mode": model.rain_mode if rain_on else "off", "multiplier": multiplier}
        )
    else:
        detector.rain_params = {"rain_threshold_multiplier": multiplier if rain_on else 1.0}


def _grid(start, end):
    return pd.date_range(start.floor("15min"), end.floor("15min"), freq="15min", tz="UTC")


def _raw_window(tables, sites):
    """The readings under inventory names, one frame with a location column, the way the support modules read them."""
    frames = []
    for site in sites:
        if site in tables:
            frames.append(sources.tidy(tables[site]).assign(location=site))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["location"])


def screen(tables, sites, grid):
    """
    Settling Pool's flags, applied: readings it doesn't trust are made missing
    in a copy before the model reads them. Returns (tables, {site: readings held back}).
    """
    from strawberrywatch.support_modules import qc_tests as qc
    from strawberrywatch.support_modules.settling_pool import SettlingPool

    raw_names = sources._raw_columns()
    window = {"raw": _raw_window(tables, sites), "timestamps": grid}
    flags = SettlingPool().flags(window, [s for s in sites if s in tables])
    out = {name: frame.copy() for name, frame in tables.items()}
    held = {}
    for (site, variable), levels in flags.items():
        column = raw_names.get(variable)
        frame = out.get(site)
        if frame is None or column not in frame.columns:
            continue
        bad = grid[np.isin(levels, (qc.SUSPECT, qc.FAIL))]
        hits = frame.index.floor("15min").isin(bad) & frame[column].notna().to_numpy()
        if hits.any():
            frame.loc[hits, column] = np.nan
            held[site] = held.get(site, 0) + int(hits.sum())
    return out, held


def explain(support, model_name, model_sites, tables, grid, over, held):
    """
    What each explainer turned on for this model says about each site, from
    the modules' own explain(). over is decide.over_steps for the noticed groups.
    """
    from strawberrywatch.support_modules import SupportStack

    stack = SupportStack.load(support, settings_mod.model_class(model_name))
    nodes = list(model_sites)
    fired = np.zeros((len(grid), len(nodes)), dtype=bool)
    position = {stamp.timestamp(): i for i, stamp in enumerate(grid)}
    for (site, _, _), times in over.items():
        if site not in nodes:
            continue
        for t in times:
            i = position.get(float(pd.Timestamp(t, unit="s", tz="UTC").floor("15min").timestamp()))
            if i is not None:
                fired[i, nodes.index(site)] = True

    window = {"raw": _raw_window(tables, nodes).sort_index(), "timestamps": grid}
    accounts = {}
    for module in stack.explainers:
        try:
            results = module.explain(window, nodes, fired)
        except Exception as exc:
            logger.warning("%s couldn't explain this check: %s", module.name, exc)
            continue
        for site, result in zip(nodes, results, strict=True):
            entry = {k: result.get(k) for k in ("verdict", "cause", "confidence", "explanation")}
            if (
                module.name == "newtwork_run"
                and result.get("verdict") == "consistent_with_transport"
            ):
                firsts = {
                    nodes[j]: int(np.flatnonzero(fired[:, j])[0])
                    for j in range(len(nodes))
                    if fired[:, j].any()
                }
                entry["order"] = sorted(firsts, key=firsts.get)
            accounts.setdefault(site, {})[module.name] = entry
    if "settling_pool" in support:
        for site in nodes:
            count = held.get(site, 0)
            accounts.setdefault(site, {})["settling_pool"] = {
                "verdict": "held_back" if count else "clear",
                "explanation": (
                    f"held back {count} readings at {site} it didn't trust, before the model read them"
                    if count
                    else f"let every reading at {site} through"
                ),
            }
    return accounts


def run_check(store, models, token=None, detectors=None, folder=None, clock=time.time):
    """
    One check. Never raises: a failure is recorded on the check and logged, so a
    bad day at the API doesn't stop the next check.

    detectors is {name: detector} to use instead of loading, for tests.
    Returns the check id, or None before anything is applied or set up.
    """
    settings_id, applied = store.applied()
    config = store.source()
    if applied is None or config is None:
        return None

    try:
        source = sources.from_config(config, token=token)
        start, end = source.window(LOOKBACK)
    except Exception as exc:
        check_id = store.start_check(config["kind"], settings_id, None, None)
        store.finish_check(check_id, False, _plain(exc), {})
        logger.warning("check couldn't start: %s", _plain(exc))
        return check_id

    # The check's start is when it noticed anything, for the repeat rule here and in the preview
    started = clock()
    check_id = store.start_check(
        source.kind, settings_id, start.timestamp(), end.timestamp(), started
    )
    durations = {}
    problems = []
    try:
        from strawberrywatch import inventory as inv

        inventory = inv.load()
        tables = source.tables(list(inventory.sites), start, end)
        _save_readings(store, tables)
        on = [m for m in models if applied["models"].get(m.name, {}).get("on")]
        loaded = {}
        for model in on:
            try:
                loaded[model.name] = (detectors or {}).get(model.name) or load_detector(
                    model.name, folder
                )
            except Exception as exc:
                problems.append(f"{model.title} didn't run: {_plain(exc)}")
        wants = any(getattr(d, "wants_weather", False) for d in loaded.values()) or applied["rain"]
        weather = source.weather(start, end) if wants and loaded else None
        _save_weather(store, weather)
        grid = _grid(start, end)

        for model in on:
            detector = loaded.get(model.name)
            if detector is None:
                continue
            spec = applied["models"][model.name]
            began = time.monotonic()
            try:
                configure_rain(detector, model, applied["rain"], spec["rain_multiplier"])
                mine = {name: tables[name] for name in detector.table_names if name in tables}
                held = {}
                if "settling_pool" in spec["support"]:
                    mine, held = screen(mine, model.sites, grid)
                scored = detector.scores(mine, weather, leave_out=set(spec["leave_out"]) or None)
                store.save_scores(check_id, model.name, scored)
                grouped = decide.series(scored)
                items = decide.notice_window(
                    grouped, start.timestamp(), end.timestamp(), spec["scales"], model.min_steps
                )
                _record(
                    store,
                    check_id,
                    model,
                    spec,
                    applied,
                    items,
                    grouped,
                    mine,
                    grid,
                    held,
                    start,
                    end,
                    started,
                )
            except Exception as exc:
                logger.exception("%s failed this check", model.name)
                problems.append(f"{model.title} failed this check: {_plain(exc)}")
            durations[model.name] = round(time.monotonic() - began, 2)
        store.prune(end.timestamp())
        message = "; ".join(problems) if problems else ""
        store.finish_check(check_id, not problems, message, durations)
    except Exception as exc:
        logger.exception("check failed")
        store.finish_check(check_id, False, _plain(exc), durations)
    return check_id


def _record(
    store,
    check_id,
    model,
    spec,
    applied,
    items,
    grouped,
    tables,
    grid,
    held,
    start,
    end,
    checked_at,
):
    """Keep what's new, extend what's the same, skip repeats, and explain the new ones."""
    new = []
    for item in items:
        previous = store.last_noticed(model.name, item["site"], item["variable"], item["rule"])
        outcome = decide.classify(item, previous, checked_at, applied["repeat_hours"])
        if outcome == decide.SAME:
            store.extend_noticed(previous["id"], item["last_over"], item["peak"])
        elif outcome == decide.NEW:
            new.append(item)
    if not new:
        return
    over = decide.over_steps(
        {
            k: v
            for k, v in grouped.items()
            if any(k == (i["site"], i["variable"], i["rule"]) for i in items)
        },
        start.timestamp(),
        end.timestamp(),
        spec["scales"],
    )
    accounts = {}
    if spec["support"]:
        accounts = explain(spec["support"], model.name, model.sites, tables, grid, over, held)
    dont_flag = set(applied["dont_flag"])
    for item in new:
        store.add_noticed(
            check_id,
            {**item, "model": model.name},
            item["site"] not in dont_flag,
            accounts.get(item["site"], {}),
            item["scale"],
        )


def _save_readings(store, tables):
    rows = []
    for site, table in tables.items():
        tidied = sources.tidy(table)
        for stamp, values in tidied.iterrows():
            rows.append(
                (
                    site,
                    stamp.timestamp(),
                    *[
                        _num(values.get(c))
                        for c in (
                            "conductivity",
                            "depth",
                            "temperature",
                            "dissolved_oxygen",
                            "floating_conductivity",
                            "battery",
                        )
                    ],
                )
            )
    if rows:
        store.save_readings(rows)


def _save_weather(store, weather):
    if weather is None or "rain_mm" not in weather.columns:
        return
    rain = pd.to_numeric(weather["rain_mm"], errors="coerce")
    store.save_weather([(stamp.timestamp(), _num(v)) for stamp, v in rain.items()])


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _plain(exc):
    """An error as a sentence, without a traceback and without anything secret in it."""
    text = str(exc).strip() or type(exc).__name__
    return text.splitlines()[0][:300]


class Watcher:
    """
    Runs a check every 15 minutes, and straight away when asked. Two checks
    never overlap: one asked for while another runs waits for the next turn.
    """

    def __init__(
        self,
        store,
        models,
        token_getter=sources.get_token,
        interval=INTERVAL,
        after_check=None,
        **check_kw,
    ):
        self.store = store
        # Called with each check's id once it's recorded, as the dashboard's emails are
        self.after_check = after_check
        self.models = models
        self.token_getter = token_getter
        self.interval = interval.total_seconds()
        self.check_kw = check_kw
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._busy = threading.Lock()
        self._thread = None
        self.next_at = None

    @property
    def busy(self):
        return self._busy.locked()

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, name="strawberrywatch-watcher", daemon=True
            )
            self._thread.start()

    def check_now(self):
        self._wake.set()

    def stop(self, timeout=30):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _periodic(self):
        config = self.store.source()
        return bool(config) and config["kind"] != sources.REPLAY

    def _loop(self):
        while not self._stop.is_set():
            due = self.next_at is not None and time.time() >= self.next_at and self._periodic()
            if self._wake.is_set() or due or (self.next_at is None and self._periodic()):
                self._wake.clear()
                self.check()
            # Wake for the next check, or every few seconds to notice a stop
            wait = 5.0 if self.next_at is None else max(0.05, min(5.0, self.next_at - time.time()))
            self._wake.wait(timeout=wait)

    def check(self):
        """Run one check now unless one is already running. Returns its id or None."""
        if not self._busy.acquire(blocking=False):
            return None
        try:
            began = time.time()
            token = None
            config = self.store.source()
            if config and config["kind"] == sources.API:
                try:
                    token = self.token_getter()
                except sources.SourceError as exc:
                    logger.warning("%s", exc)
            check_id = run_check(self.store, self.models, token=token, **self.check_kw)
            self.next_at = began + self.interval
            if self.after_check is not None:
                self.after_check(check_id)
            return check_id
        finally:
            self._busy.release()

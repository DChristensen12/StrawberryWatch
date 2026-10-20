"""
One check, and the thread that runs one every 15 minutes while the app is open.
Borrows gnn_alerts.py's pattern without importing it: a background thread, a
busy guard to stop checks overlapping, and failures logged instead of raised.
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

# The last settings each store gave that read cleanly. A damaged row can't stop the checks
_last_good = {}

INTERVAL = timedelta(minutes=settings_mod.INTERVAL_MINUTES)
LOOKBACK = timedelta(days=settings_mod.LOOKBACK_DAYS)
HASHES_PATH = Path(__file__).resolve().parent / "model_files.sha256.json"
# Failed or skipped checks in a row before checks pause, as gnn_alerts does on the website
PAUSE_AFTER = 3
# Saved beside a model update's files, the fingerprints they were checked against
HASHES = "hashes.json"


class ModelFileError(RuntimeError):
    """A model's files aren't the ones it was tested with."""


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(cls, folder, hashes=None):
    """Checks every file a model loads against its shipped hash, before torch sees it."""
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
    """
    Loads (and caches) a serving detector once its files pass verify_files. A
    model update's folder brings its own fingerprints.
    """
    from strawberrywatch import paths
    from strawberrywatch.serving import detector_class

    folder = Path(folder or paths.checkpoints_dir())
    cls = detector_class(name)
    listed = folder / HASHES
    verify_files(cls, folder, json.loads(listed.read_text()) if listed.exists() else None)
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
    """Readings the way the support modules want them: inventory names, one frame, a location column."""
    frames = []
    for site in sites:
        if site in tables:
            frames.append(sources.tidy(tables[site]).assign(location=site))
    return pd.concat(frames) if frames else pd.DataFrame(columns=["location"])


def screen(tables, sites, grid):
    """
    Applies Settling Pool. Readings it doesn't trust are blanked in a copy before
    the model sees them. Returns (tables, {site: readings held back}).
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
    Runs each explainer turned on for this model. over is decide.over_steps for
    the flagged groups.
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


def read_settings(store):
    """
    Returns (version id, settings, notice). Broken settings are swapped for the
    last good ones, and the notice says so.
    """
    key = str(getattr(store, "path", id(store)))
    try:
        settings_id, applied = store.applied()
        if applied is not None:
            problems = settings_mod.out_of_range(applied)
            if problems:
                raise ValueError(" ".join(problems))
    except Exception as exc:
        logger.error("the saved settings couldn't be used (%s)", _plain(exc))
        if key not in _last_good:
            raise
        settings_id, applied = _last_good[key]
        return settings_id, applied, SETTINGS_NOTICE
    if applied is not None:
        _last_good[key] = (settings_id, applied)
    return settings_id, applied, ""


SETTINGS_NOTICE = (
    "The saved settings couldn't be read, so this check used the last ones that could."
)
NO_WEATHER = "Weather wasn't available for this check"


def run_check(
    store,
    models,
    token=None,
    detectors=None,
    folder=None,
    clock=time.time,
    memory=None,
    max_models=None,
):
    """
    Runs one check. Doesn't raise. A bad day at the API can't stop the next
    one. Failures are recorded on the check and logged.

    token is the Strawberry Creek API token, the local app's backup source.
    detectors swaps in {name: detector} for tests. memory reads peak memory in
    MB so each model's cost gets recorded, and max_models caps how many run.
    Returns the check id, or None if nothing's applied or set up yet.
    """
    config = store.source()
    try:
        settings_id, applied, notice = read_settings(store)
    except Exception as exc:
        if config is None:
            return None
        check_id = store.start_check(config["kind"], None, None, None)
        store.finish_check(check_id, False, _plain(exc), {})
        return check_id
    if applied is None or config is None:
        return None

    try:
        source = sources.from_config(config, token=token)
        start, end = source.window(LOOKBACK)
    except Exception as exc:
        check_id = store.start_check(
            config["kind"],
            settings_id,
            None,
            None,
            detail=config.get("event") or config.get("folder"),
        )
        store.finish_check(check_id, False, _plain(exc), {})
        logger.warning("check couldn't start: %s", _plain(exc))
        return check_id

    # Counts as the alert time for the repeat rule, here and in the preview
    started = clock()
    check_id = store.start_check(
        source.kind, settings_id, start.timestamp(), end.timestamp(), started, detail=source.detail
    )
    live = sources.is_live(source.kind)
    durations = {}
    problems = [notice] if notice else []
    weather_ok = None
    try:
        from strawberrywatch import inventory as inv
        from strawberrywatch.watch import model_updates

        inventory = inv.load()
        tables = source.tables(list(inventory.sites), start, end)
        _save_readings(store, tables, live)
        on = [m for m in models if applied["models"].get(m.name, {}).get("on")]
        if max_models is not None and len(on) > max_models:
            skipped = ", ".join(m.title for m in on[max_models:])
            problems.append(f"{skipped} didn't run: only {max_models} can run at once.")
            on = on[:max_models]
        loaded = {}
        for model in on:
            try:
                loaded[model.name] = (detectors or {}).get(model.name) or load_detector(
                    model.name, model_updates.model_folder(store, model.name, folder)
                )
            except ModelFileError as exc:
                logger.warning("%s is unavailable: %s", model.name, exc)
                problems.append(f"{model.title} is unavailable: its files are missing or changed")
            except Exception as exc:
                problems.append(f"{model.title} didn't run: {_plain(exc)}")
        wants = any(getattr(d, "wants_weather", False) for d in loaded.values()) or applied["rain"]
        weather = source.weather(start, end) if wants and loaded else None
        if wants and loaded:
            weather_ok = weather is not None
        _save_weather(store, weather)
        grid = _grid(start, end)

        for model in on:
            detector = loaded.get(model.name)
            if detector is None:
                continue
            spec = applied["models"][model.name]
            began = time.monotonic()
            before = memory() if memory else None
            try:
                configure_rain(detector, model, applied["rain"], spec["rain_multiplier"])
                if hasattr(detector, "set_sensitivity"):
                    detector.set_sensitivity(spec.get("rates"), spec.get("thresholds"))
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
            if memory:
                durations.setdefault("memory_mb", {})[model.name] = round(
                    max(0.0, memory() - before), 1
                )
        if memory:
            durations["peak_mb"] = round(memory(), 1)
        if live:
            store.prune_daily(clock())
        message = "; ".join(problems) if problems else ""
        store.finish_check(check_id, not problems, message, durations, weather_ok)
    except Exception as exc:
        logger.exception("check failed")
        store.finish_check(check_id, False, _plain(exc), durations, weather_ok)
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


def _save_readings(store, tables, live=True):
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
        store.save_readings(rows, live)


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
    """An error as a sentence, without a traceback or secrets."""
    text = str(exc).strip() or type(exc).__name__
    return text.splitlines()[0][:300]


class Watcher:
    """
    Runs a check every 15 minutes, and right away when asked. Checks don't
    overlap. A request during one waits for the next turn.

    Follows two of the website's limits: max_models_on from settings.yaml, and
    a pause after PAUSE_AFTER failures in a row until someone presses Restart
    checks.
    """

    def __init__(
        self,
        store,
        models,
        api_token=lambda: None,
        interval=INTERVAL,
        after_check=None,
        **check_kw,
    ):
        self.store = store
        # Called with each check's id once it's saved, which is how emails go out
        self.after_check = after_check
        self.models = models
        # The API token from .env, read at each check so a fixed .env needs no restart
        self.api_token = api_token
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
        return self.store.source() is not None

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
        """Run one check now unless one is already running or checks are paused. Returns its id or None."""
        if not self._busy.acquire(blocking=False):
            return None
        try:
            began = time.time()
            self.next_at = began + self.interval
            if self.store.pref("checks_paused") == "yes":
                return None
            from strawberrywatch.watch.check_once import peak_mb

            check_id = run_check(
                self.store,
                self.models,
                token=self.api_token(),
                memory=peak_mb,
                max_models=int(settings_mod.website_limits()["max_models_on"]),
                **self.check_kw,
            )
            count_outcome(self.store, check_id)
            if self.after_check is not None:
                self.after_check(check_id)
            return check_id
        finally:
            self._busy.release()


def count_outcome(store, check_id):
    """
    A good check clears the count. A failed or skipped one adds to it, and
    PAUSE_AFTER in a row pause the checks, as gnn_alerts does on the website.
    """
    if check_id is None:
        return
    last = store.last_check()
    if last and last.get("ok"):
        store.set_pref("check_failures", "0")
        return
    try:
        count = int(store.pref("check_failures") or 0) + 1
    except ValueError:
        count = 1
    store.set_pref("check_failures", str(count))
    if count >= PAUSE_AFTER:
        store.set_pref("checks_paused", "yes")
        logger.warning("%s checks failed in a row, model checks paused", count)

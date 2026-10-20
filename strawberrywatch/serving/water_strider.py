"""
Serving Water Strider. Same contract as detector.py and cobble.py, but it's one
JSON file and no torch. A check takes milliseconds and a few megabytes.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from strawberrywatch.models import Water_Strider as ws_model
from strawberrywatch.serving.checkpoint import CheckpointError
from strawberrywatch.serving.contract import (
    DONT_FLAG,
    LEAVE_OUT,
    DetectorError,
    Finding,
    Score,
    check_leave_out,
    notice,
    utc_indexed,
)

logger = logging.getLogger(__name__)

CALIBRATION = "water_strider_calibration.json"

_cache = {}
_cache_lock = threading.Lock()


class WaterStriderDetector:
    """
    Every sensor judged on its own readings by three calibrated tests. One
    Finding per site, sensor and rule that crossed its rain-gated threshold.
    """

    name = "water_strider"
    RULES = ws_model.RULES
    # Sensors are judged on their own. Leaving a site out doesn't affect the others
    SITE_CHOICE = (DONT_FLAG, LEAVE_OUT)
    # The spike test's two-readings rule is built into its score. One step over counts
    min_steps = 1
    FILES = (CALIBRATION,)
    # Rain moves its thresholds already. A rain modulator on top would move them twice
    BUILTIN_SUPPORT = ("rain",)

    def __init__(self, calibration, rain_gate=None):
        from strawberrywatch.anomalies.rain_gate import RainGate

        self.calibration = calibration
        self.params = {**ws_model.PARAMS, **calibration.get("params", {})}
        self.gate = RainGate.from_dict({"base_threshold": 1.0, **(rain_gate or {})})
        self.rates = {rule: ws_model.Q_TOTAL for rule in self.RULES}
        self.typed = {}

    @classmethod
    def load(cls, checkpoint_dir, device="cpu", **kw):  # noqa: ARG003 (the contract passes it)
        """Reads the calibration. device is only here for the contract."""
        path = Path(checkpoint_dir) / CALIBRATION
        if not path.exists():
            raise CheckpointError(
                f"no {CALIBRATION} in {checkpoint_dir}. Make one with "
                f"scripts/calibrate_water_strider.py."
            )
        calibration = json.loads(path.read_text())
        if calibration.get("model") != "water_strider" or not calibration.get("sensors"):
            raise DetectorError(f"{path} isn't a Water Strider calibration")
        return cls(calibration, **kw)

    @classmethod
    def cached(cls, checkpoint_dir, device="cpu", **kw):  # noqa: ARG003 (the contract passes it)
        key = (str(checkpoint_dir), cls.name)
        with _cache_lock:
            if key not in _cache:
                _cache[key] = cls.load(checkpoint_dir, device, **kw)
            return _cache[key]

    @property
    def table_names(self):
        """In-service sites with a calibrated sensor, most clean days first."""
        from strawberrywatch import inventory as inventory_module

        inventory = inventory_module.load()
        days = {}
        for entry in self.calibration["sensors"].values():
            site = entry["site"]
            if site not in inventory.sites:
                continue
            if entry["variable"] in inventory.site(site).scored_variables:
                days[site] = max(days.get(site, 0.0), float(entry.get("clean_days", 0.0)))
        return sorted(days, key=lambda site: (-days[site], site))

    @property
    def wants_weather(self):
        return self.gate.mode != "off"

    def set_rain_gate(self, rain_gate=None):
        from strawberrywatch.anomalies.rain_gate import RainGate

        self.gate = RainGate.from_dict({"base_threshold": 1.0, **(rain_gate or {})})

    def set_sensitivity(self, rates=None, thresholds=None):
        """
        Sets the false alarm rate per test, plus any hand-typed thresholds.
        rates is {rule: whole-model rate} and thresholds is
        {"site.variable.rule": z}. We don't refit. New thresholds come from
        the calibrated tails.
        """
        self.rates = {rule: float((rates or {}).get(rule, ws_model.Q_TOTAL)) for rule in self.RULES}
        self.typed = {key: float(z) for key, z in (thresholds or {}).items()}

    def base_threshold(self, key, rule):
        """One sensor's threshold for one test, before rain moves it."""
        typed = self.typed.get(f"{key}.{rule}")
        if typed is not None:
            return typed
        tail = self.calibration["sensors"][key]["tests"][rule]
        rate = self.rates.get(rule, ws_model.Q_TOTAL)
        if abs(rate - self.calibration.get("q_total", ws_model.Q_TOTAL)) < 1e-15:
            return float(tail["threshold"])
        return ws_model.threshold_for(tail, ws_model.per_test(rate))

    def scores(self, tables, weather=None, leave_out=None):
        """Every judged step of every sensor, all three tests, with the rain-gated line."""
        leave_out = check_leave_out(self, leave_out)
        return self._score(tables, weather, leave_out)[0]

    def findings(self, tables, weather=None, leave_out=None):
        """What fired, built from scores() so the two always agree at the tested line."""
        leave_out = check_leave_out(self, leave_out)
        rows, readings = self._score(tables, weather, leave_out)
        found = []
        for hit in notice(rows, self.min_steps, 1.0):
            found.append(
                Finding(
                    site=hit.site,
                    variable=hit.variable,
                    rule=hit.rule,
                    peak=hit.peak,
                    threshold=hit.threshold,
                    when=hit.when,
                    readings=readings.get((hit.site, hit.variable), pd.Series(dtype=float)),
                )
            )
        return found

    def _score(self, tables, weather, leave_out):
        """
        Returns (scores, {(site, variable): readings}). Each sensor is scored on
        its own readings. Only the spike test looks at the site's other sensors,
        for a second reading that agrees.
        """
        from strawberrywatch.ingest.raw_data_loader import tidy_channels

        by_site = {}
        in_service = set(self.table_names)
        for key, entry in self.calibration["sensors"].items():
            site, variable = entry["site"], entry["variable"]
            if site in leave_out or site not in in_service:
                continue
            raw = tables.get(site)
            if raw is None or raw.empty:
                continue
            frame = tidy_channels(utc_indexed(site, raw))
            if variable not in frame.columns:
                continue
            values = ws_model.on_grid(frame[variable])
            if values.dropna().empty:
                continue
            spread, floor = float(entry["spread"]), float(entry["floor"])
            stats = ws_model.statistics(
                values, entry["profile"], spread, floor, self.params, entry.get("rho", 0.0)
            )
            by_site.setdefault(site, []).append((key, variable, values, stats))

        rain_series = _rain(weather)
        rows, readings = [], {}
        for site, sensors in by_site.items():
            ratios = {}
            for key, _variable, _values, stats in sensors:
                ratios[key] = stats["hampel_spike"] / self.base_threshold(key, "hampel_spike")
            for key, variable, values, stats in sensors:
                index = values.index
                multiplier = self._multipliers(index, rain_series)
                siblings = _siblings(key, sensors, ratios, index)
                for rule in self.RULES:
                    base = self.base_threshold(key, rule)
                    statistic = stats[rule]
                    if rule == "hampel_spike":
                        statistic = ws_model.corroborated_spikes(ratios[key], siblings) * base
                    lines = base * multiplier
                    for i in np.flatnonzero(np.isfinite(statistic)):
                        rows.append(
                            Score(
                                site,
                                variable,
                                rule,
                                index[i],
                                float(statistic[i]),
                                float(lines[i]),
                                base,
                            )
                        )
                readings[(site, variable)] = values.dropna()
        return rows, readings

    def _multipliers(self, index, rain_series):
        if rain_series is None or self.gate.mode == "off":
            return np.ones(len(index))
        rain = rain_series.reindex(index).fillna(0.0).to_numpy(dtype=float)
        return self.gate.multipliers(rain, index.tz_localize(None).to_numpy())


def _rain(weather):
    """Rain in mm per quarter hour. A gap counts as dry, as cobble.py's _rain_on does."""
    if weather is None or weather.empty or "rain_mm" not in weather.columns:
        return None
    rain = utc_indexed("weather", weather)["rain_mm"]
    return pd.to_numeric(rain, errors="coerce").resample("15min").sum()


def _siblings(key, sensors, ratios, index):
    """The strongest spike share among the site's other sensors at each step, or None."""
    others = [
        pd.Series(ratios[k], index=v.index).reindex(index).to_numpy(dtype=float)
        for k, _var, v, _stats in sensors
        if k != key
    ]
    if not others:
        return None
    # fmax skips NaN without complaining about a step where every sibling is missing
    return np.fmax.reduce(np.vstack(others), axis=0)


__all__ = ["CALIBRATION", "WaterStriderDetector"]

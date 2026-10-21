"""
What each registered model can do, read off its class and calibration files
without loading any weights. The settings pages need this before a model runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

STEP_SECONDS = 900  # the loggers' 15 minute cadence


@dataclass
class ModelInfo:
    name: str
    rules: tuple
    site_choice: tuple
    # Night Heron table names, in the model's own order
    sites: list
    # (table, inventory variable) for every sensor it scores
    nodes: list
    rain_default: float
    # RainGate keyword arguments other than the multiplier
    rain_params: dict = field(default_factory=dict)
    rain_mode: str = "decay"
    # Below this share of its sensors a moment isn't judged at all
    min_coverage: float | None = None
    # How many steps over the line a rule needs before it counts
    min_steps: int = 1
    note: str = ""
    # "scale" models move each rule's line by a multiple of the tested one. A
    # "rate" model is calibrated, and its slider moves the false alarm rate instead
    sensitivity: str = "scale"
    # A rate model's fitted tails, {"site.variable": {rule: tail}}, from its calibration
    tails: dict = field(default_factory=dict)

    @property
    def title(self):
        return self.name.replace("_", " ").title()

    def gate(self, rain_on=True, multiplier=None):
        """The RainGate that the model's rain handling amounts to, at a line of 1."""
        from strawberrywatch.anomalies.rain_gate import RainGate

        return RainGate(
            base_threshold=1.0,
            mode=self.rain_mode if rain_on else "off",
            multiplier=self.rain_default if multiplier is None else float(multiplier),
            **self.rain_params,
        )


def _dusk_crayfish(cls, _inventory):
    from strawberrywatch import paths
    from strawberrywatch.anomalies.anomaly_detector import MIN_TIMESTEPS_OVER_THRESHOLD
    from strawberrywatch.config import Config

    blob = json.loads((paths.model_dir(cls.name) / "dusk_crayfish_serving.json").read_text())
    sites = sorted(blob["location_to_idx"], key=blob["location_to_idx"].get)
    return ModelInfo(
        name=cls.name,
        rules=tuple(cls.RULES),
        site_choice=tuple(cls.SITE_CHOICE),
        sites=sites,
        # Both rules score conductivity only
        nodes=[(site, "conductivity") for site in sites],
        rain_default=float(Config.RAIN_THRESHOLD_MULTIPLIER),
        min_steps=int(MIN_TIMESTEPS_OVER_THRESHOLD),
        # Its built in taper is RainGate's "deployed" mode, the arithmetic
        # anomaly_detector._rain_multipliers does
        rain_mode="deployed",
        rain_params={
            "decay_hours": float(Config.POST_RAIN_DECAY_HOURS),
            "wet_mm": float(Config.RAIN_AMOUNT_THRESHOLD),
            "lookback_hours": float(Config.RAIN_WINDOW_HOURS),
        },
    )


def _cobble_shoal(cls, inventory):
    from strawberrywatch.anomalies.rain_gate import RainGate
    from strawberrywatch.preprocessing import node_windows as nw
    from strawberrywatch.serving.cobble import MIN_COVERAGE

    names = {canonical: name for name, canonical in nw.VARIABLE_MAP.items()}
    roster = nw.roster_from_inventory(inventory)
    gate = RainGate(base_threshold=1.0)
    return ModelInfo(
        name=cls.name,
        rules=tuple(cls.RULES),
        site_choice=tuple(cls.SITE_CHOICE),
        sites=[nw.SITE_TO_TABLE.get(site, site) for site in nw.SITE_ORDER],
        nodes=[
            (nw.SITE_TO_TABLE.get(site, site), names[var])
            for site, variables in roster.items()
            for var in variables
        ],
        rain_default=gate.multiplier,
        rain_mode=gate.mode,
        min_steps=int(cls.min_steps),
        min_coverage=float(MIN_COVERAGE),
    )


def _water_strider(cls, inventory):
    from strawberrywatch import paths
    from strawberrywatch.anomalies.rain_gate import RainGate

    path = paths.model_dir(cls.name) / "water_strider_calibration.json"
    calibration = json.loads(path.read_text()) if path.exists() else {"sensors": {}}
    sensors = {
        key: entry
        for key, entry in calibration["sensors"].items()
        if entry["site"] in inventory.sites
        and entry["variable"] in inventory.site(entry["site"]).scored_variables
    }
    gate = RainGate(base_threshold=1.0)
    return ModelInfo(
        name=cls.name,
        rules=tuple(cls.RULES),
        site_choice=tuple(cls.SITE_CHOICE),
        sites=sorted({entry["site"] for entry in sensors.values()}),
        nodes=[(entry["site"], entry["variable"]) for entry in sensors.values()],
        rain_default=gate.multiplier,
        rain_mode=gate.mode,
        min_steps=int(cls.min_steps),
        sensitivity="rate",
        tails={key: entry["tests"] for key, entry in sensors.items()},
    )


def _flame_skimmer(cls, inventory):
    """Cobble Shoal's sites and sensors, minus the ones Flame Skimmer didn't see in training."""
    from strawberrywatch import paths
    from strawberrywatch.preprocessing import node_windows as nw
    from strawberrywatch.serving.flame import CALIBRATION

    info = _cobble_shoal(cls, inventory)
    path = paths.model_dir(cls.name) / CALIBRATION
    corpus = json.loads(path.read_text()).get("corpus", {}) if path.exists() else {}
    names = {canonical: name for name, canonical in nw.VARIABLE_MAP.items()}
    offline = set()
    for key in corpus.get("offline_nodes", ()):
        site, var = key.split(".", 1)
        offline.add((nw.SITE_TO_TABLE.get(site, site), names.get(var, var)))
    info.nodes = [node for node in info.nodes if node not in offline]
    return info


# Sites and sensors aren't on the serving contract, and we read them here. A
# model missing from this still lists, just with no sites.
_DETAILS = {
    "dusk_crayfish": _dusk_crayfish,
    "cobble_shoal": _cobble_shoal,
    "water_strider": _water_strider,
    "flame_skimmer": _flame_skimmer,
}


def registered(inventory=None):
    """Every model serving.available() names, in its order."""
    from strawberrywatch import inventory as inv
    from strawberrywatch.serving import available, detector_class, site_choice

    inventory = inventory or inv.load()
    out = []
    for name in available():
        cls = detector_class(name)
        detail = _DETAILS.get(name)
        if detail:
            out.append(detail(cls, inventory))
        else:
            out.append(
                ModelInfo(
                    name=name,
                    rules=tuple(cls.RULES),
                    site_choice=site_choice(cls),
                    sites=[],
                    nodes=[],
                    rain_default=1.0,
                    rain_mode="off",
                    note="The dashboard doesn't know which sites this model reads yet.",
                )
            )
    return out


def rain_curve(model, rain_on, multiplier, storm_hours=6, after_hours=66):
    """
    The alert line as a multiple of normal, through a storm and after it.
    RainGate does the arithmetic, and it's exactly what the model does. Hours
    count from when the rain starts.
    """
    steps = int((storm_hours + after_hours) * 3600 / STEP_SECONDS)
    hours = np.arange(steps) * STEP_SECONDS / 3600
    rain = np.where(hours < storm_hours, 1.0, 0.0)
    multiple = model.gate(rain_on, multiplier).thresholds(rain, hours)
    raised = np.flatnonzero(multiple > 1.0 + 1e-9)
    return {
        "hours": [round(float(h), 2) for h in hours],
        "multiple": [round(float(m), 4) for m in multiple],
        "storm_hours": storm_hours,
        "peak": round(float(multiple.max()), 3),
        "back_to_normal": round(float(hours[raised[-1]] + STEP_SECONDS / 3600 - storm_hours), 1)
        if len(raised)
        else None,
    }

"""
What each model in the serving registry can do, without loading its weights.

The settings pages need a model's rules, its sites, what it can do about a site
and how it handles rain before anything has run, so this reads them off the
class and the calibration files that are in git.
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
    # RainGate keyword arguments besides the multiplier, matching what the model does
    rain_params: dict = field(default_factory=dict)
    rain_mode: str = "decay"
    # Below this share of its sensors a moment isn't judged at all
    min_coverage: float | None = None
    # How many steps over the line a rule needs before it counts
    min_steps: int = 1
    note: str = ""

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

    blob = json.loads((paths.checkpoints_dir() / "dusk_crayfish_serving.json").read_text())
    sites = sorted(blob["location_to_idx"], key=blob["location_to_idx"].get)
    return ModelInfo(
        name=cls.name,
        rules=tuple(cls.RULES),
        site_choice=tuple(cls.SITE_CHOICE),
        sites=sites,
        # Both rules score conductivity and nothing else
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


# Sites and sensors aren't on the serving contract, so each model's are read
# here. One registered later still lists, with no sites, until it gets a line.
_DETAILS = {"dusk_crayfish": _dusk_crayfish, "cobble_shoal": _cobble_shoal}


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
    The line as a multiple of normal through a storm and the hours after it.

    RainGate does the arithmetic, so the picture is what the model does. Hours
    count from the start of the rain.
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

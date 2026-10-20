"""Which models can run outside this repository, and the only way to get one by name."""

from __future__ import annotations

from strawberrywatch.serving.cobble import CobbleShoalDetector
from strawberrywatch.serving.detector import DuskCrayfishDetector
from strawberrywatch.serving.flame import FlameSkimmerDetector
from strawberrywatch.serving.water_strider import WaterStriderDetector

# The names GNN_MODELS and everything else outside use. A new model is a
# contract.py detector plus a line here, ready once
# tests/test_serving_conformance.py passes for it.
SERVING_REGISTRY = {
    DuskCrayfishDetector.name: DuskCrayfishDetector,
    CobbleShoalDetector.name: CobbleShoalDetector,
    WaterStriderDetector.name: WaterStriderDetector,
    FlameSkimmerDetector.name: FlameSkimmerDetector,
}


class UnknownDetector(LookupError):
    """A name that is not in SERVING_REGISTRY."""


def available():
    return sorted(SERVING_REGISTRY)


def detector_class(name):
    """
    The detector registered under one name, or raise listing the valid ones.

    It has to raise. A misspelt model in someone's .env that resolved to no
    detector would look exactly like a healthy creek.
    """
    try:
        return SERVING_REGISTRY[name]
    except KeyError:
        raise UnknownDetector(
            f"unknown model {name!r}; valid options: {', '.join(available())}"
        ) from None

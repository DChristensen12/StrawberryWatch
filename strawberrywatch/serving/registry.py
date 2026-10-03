"""Which models can run outside this repository, and the only way to get one by name."""

from __future__ import annotations

from strawberrywatch.serving.cobble import CobbleShoalDetector
from strawberrywatch.serving.detector import DuskCrayfishDetector

# What GNN_MODELS, or anything else outside, names a model by. A new model is a
# detector speaking contract.py plus a line here, and it is ready once
# tests/test_serving_conformance.py passes for it. The Night Heron runner does
# not change.
SERVING_REGISTRY = {
    DuskCrayfishDetector.name: DuskCrayfishDetector,
    CobbleShoalDetector.name: CobbleShoalDetector,
}


class UnknownDetector(LookupError):
    """A name that is not in SERVING_REGISTRY."""


def available():
    return sorted(SERVING_REGISTRY)


def detector_class(name):
    """
    The detector registered under one name, or raise listing the valid ones.

    Raising is the point. A misspelt model in someone's .env that quietly
    resolved to nothing would look exactly like a creek with nothing wrong in it.
    """
    try:
        return SERVING_REGISTRY[name]
    except KeyError:
        raise UnknownDetector(
            f"unknown model {name!r}; valid options: {', '.join(available())}"
        ) from None

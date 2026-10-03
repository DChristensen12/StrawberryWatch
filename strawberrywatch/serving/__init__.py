"""Running a trained model from outside this repository."""

from strawberrywatch.serving.checkpoint import (
    Checkpoint,
    CheckpointError,
    export_sidecar,
    load_checkpoint,
)
from strawberrywatch.serving.cobble import CobbleShoalDetector
from strawberrywatch.serving.contract import DetectorError, Finding
from strawberrywatch.serving.detector import DuskCrayfishDetector
from strawberrywatch.serving.registry import (
    SERVING_REGISTRY,
    UnknownDetector,
    available,
    detector_class,
)
from strawberrywatch.serving.windows import WindowError

__all__ = [
    "SERVING_REGISTRY",
    "Checkpoint",
    "CheckpointError",
    "CobbleShoalDetector",
    "DetectorError",
    "DuskCrayfishDetector",
    "Finding",
    "UnknownDetector",
    "WindowError",
    "available",
    "detector_class",
    "export_sidecar",
    "load_checkpoint",
]

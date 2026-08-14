"""
Calibrated artifacts for the Riffle Darner detector, loaded at inference.

Nothing here fits anything. A detector that refits its own null on the window
it is judging has no null, it has a description of the data it just saw.

    cal = load_calibration()
    fired = model.score(batch, cal.nulls) > cal.z_q
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from strawberrywatch.anomalies import channel_scoring
from strawberrywatch.paths import checkpoints_dir

CALIBRATION_FILENAME = "riffle_darner_calibration.json"
WEIGHTS_FILENAME = "riffle_darner_weights.pt"


class CalibrationError(RuntimeError):
    """The artifacts are missing or do not describe a usable threshold."""


@dataclass(frozen=True)
class RiffleCalibration:
    """Everything the alerting path needs that is not the weights."""

    nulls: channel_scoring.ChannelNulls
    z_q: float
    operating_q: float
    seed: int
    source_checkpoint: str
    pot: dict

    def threshold_at(self, q):
        """
        z_q at one of the calibrated nominal rates.

        Only the rates fitted at calibration time are available. Interpolating
        between them would be inventing a threshold, and choosing q is a
        science decision that belongs to whoever owns the alerting budget.
        """
        key = f"combined_fisher@{q:g}"
        if key not in self.pot:
            raise CalibrationError(
                f"no calibrated threshold at q={q:g}; "
                f"available: {sorted(k for k in self.pot if k.startswith('combined_fisher@'))}"
            )
        return float(self.pot[key]["threshold"])


def calibration_path(checkpoint_dir=None):
    return Path(checkpoint_dir or checkpoints_dir()) / CALIBRATION_FILENAME


def weights_path(checkpoint_dir=None):
    return Path(checkpoint_dir or checkpoints_dir()) / WEIGHTS_FILENAME


def load_calibration(checkpoint_dir=None):
    """Read the calibration artifact. Raises rather than guessing a default."""
    path = calibration_path(checkpoint_dir)
    if not path.exists():
        raise CalibrationError(
            f"no calibration artifact at {path}. The Riffle Darner alerting path "
            f"needs the channel nulls and z_q that were fitted with the weights; "
            f"it will not fit its own."
        )
    blob = json.loads(path.read_text())
    try:
        nulls = channel_scoring.ChannelNulls.from_dict(blob["channel_nulls"])
        z_q = float(blob["z_q"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{path} is not a usable calibration: {exc}") from exc
    if not (z_q == z_q) or z_q in (float("inf"), float("-inf")):
        raise CalibrationError(f"{path} has a non-finite z_q ({z_q})")
    return RiffleCalibration(
        nulls=nulls,
        z_q=z_q,
        operating_q=float(blob.get("operating_q", 1e-4)),
        seed=int(blob.get("seed", -1)),
        source_checkpoint=str(blob.get("source_checkpoint", "")),
        pot=blob.get("pot", {}),
    )

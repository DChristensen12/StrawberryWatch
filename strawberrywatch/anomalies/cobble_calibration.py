"""
Calibrated artifacts for the Cobble Shoal detector, loaded at inference.

Nothing here fits anything. A detector that refits its own null on the window
it is judging has no null, it has a description of the data it just saw.

    cal = load_calibration(REAL)
    win = node_windows.build_window(tables, start, end, scaler=cal.node_scaler)
    fired = model.score(batch, cal.nulls) > cal.z_q

Two calibrations ship, and they are not interchangeable. SYNTHETIC was fitted
on generated windows because that was the only corpus the model could read
before the node_windows adapter landed. REAL was fitted on data/raw_data
through that adapter. Their thresholds are close (39.13 against 39.69) and
their score scales are not: on one identical window the same weights score
7.3 to 28.4 under the synthetic nulls and 0.25 to 2.0 under the real ones.
Pick deliberately. There is no default that is right for both.

A null is fitted in a normalization space, so it only means anything applied to
data in that same space. REAL records the NodeScaler it was fitted with and
node_scaler hands it back, which is what the serving path must pass to
build_window rather than letting it fit a fresh one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from strawberrywatch.anomalies import channel_scoring
from strawberrywatch.paths import checkpoints_dir

# Fitted on tests/synthetic windows, shipped with the weights.
SYNTHETIC = "cobble_shoal_calibration.json"
# Fitted on data/raw_data through preprocessing/node_windows.
REAL = "cobble_shoal_calibration_real.json"

# Kept as the historical name for SYNTHETIC. rain_gate.py's recorded
# measurements are against this artifact's z_q.
CALIBRATION_FILENAME = SYNTHETIC
WEIGHTS_FILENAME = "cobble_shoal_weights.pt"


class CalibrationError(RuntimeError):
    """The artifacts are missing or do not describe a usable threshold."""


@dataclass(frozen=True)
class CobbleCalibration:
    """Everything the alerting path needs that is not the weights."""

    nulls: channel_scoring.ChannelNulls
    z_q: float
    operating_q: float
    seed: int
    source_checkpoint: str
    pot: dict
    # Which file this came from, so a report can say which calibration it used
    # rather than leaving the reader to guess from the number.
    filename: str = SYNTHETIC
    # The normalization the nulls were fitted in. None for SYNTHETIC, which
    # predates the adapter and recorded no scaler.
    node_scaler: object | None = None
    corpus: dict = field(default_factory=dict)

    @property
    def is_real(self):
        """Whether this calibration describes the real archive."""
        return bool(self.corpus)

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
                f"no calibrated threshold at q={q:g} in {self.filename}; "
                f"available: {sorted(k for k in self.pot if k.startswith('combined_fisher@'))}"
            )
        return float(self.pot[key]["threshold"])

    def window_scaler(self):
        """
        The NodeScaler to hand build_window, or raise saying why there is none.

        Raising is the point. build_window(scaler=None) fits a fresh scaler off
        whatever window it was given, and scores computed in that space against
        nulls fitted in another are finite, plausible, and meaningless. That
        failure has no symptom, so it has to be refused rather than reported.
        """
        if self.node_scaler is None:
            raise CalibrationError(
                f"{self.filename} records no node_scaler, so there is no way to put a "
                f"real window in the space its nulls were fitted in. Score real data "
                f"against {REAL} instead, or refit with "
                f"scripts/refit_cobble_calibration.py."
            )
        return self.node_scaler


def calibration_path(checkpoint_dir=None, filename=SYNTHETIC):
    return Path(checkpoint_dir or checkpoints_dir()) / filename


def weights_path(checkpoint_dir=None):
    return Path(checkpoint_dir or checkpoints_dir()) / WEIGHTS_FILENAME


def load_calibration(filename=SYNTHETIC, checkpoint_dir=None):
    """
    Read one calibration artifact. Raises rather than guessing a default.

    filename is SYNTHETIC or REAL. It stays an argument with no clever
    fallback: silently substituting one for the other is exactly the swap the
    module docstring says must never happen quietly.
    """
    path = calibration_path(checkpoint_dir, filename)
    if not path.exists():
        raise CalibrationError(
            f"no calibration artifact at {path}. The Cobble Shoal alerting path "
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

    scaler = None
    if blob.get("node_scaler") is not None:
        # Imported here rather than at module scope: this module is on the
        # alerting path and node_windows pulls in pandas and the inventory.
        from strawberrywatch.preprocessing.node_windows import NodeScaler

        try:
            scaler = NodeScaler.from_dict(blob["node_scaler"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"{path} has an unreadable node_scaler: {exc}") from exc

    return CobbleCalibration(
        nulls=nulls,
        z_q=z_q,
        operating_q=float(blob.get("operating_q", 1e-4)),
        seed=int(blob.get("seed", -1)),
        source_checkpoint=str(blob.get("source_checkpoint", "")),
        pot=blob.get("pot", {}),
        filename=filename,
        node_scaler=scaler,
        corpus=blob.get("corpus", {}) or {},
    )

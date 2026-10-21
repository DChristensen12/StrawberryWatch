"""
Calibrated artifacts for the Cobble Shoal detector, loaded at inference.

This module fits nothing. A detector that refits its own null on the window
it's judging has no null, just a description of the data it saw.

    cal = load_calibration(REAL)
    win = node_windows.build_window(tables, start, end, scaler=cal.node_scaler)
    fired = model.score(batch, cal.nulls) > cal.z_q

Two calibrations ship and they aren't interchangeable. SYNTHETIC was fitted on
generated windows, REAL on data/raw_data through node_windows. Their thresholds
are close (39.13 against 39.69) but their score scales aren't: on one window the
same weights score 7.3 to 28.4 under the synthetic nulls and 0.25 to 2.0 under
the real ones. Pick with care.

A null only means something in the normalization space it was fitted in. REAL
records its NodeScaler, and the serving path must pass that to build_window
instead of letting it fit a fresh one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from strawberrywatch.anomalies import channel_scoring
from strawberrywatch.paths import model_dir

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
    # Which file this came from. A report can say which calibration it used,
    # and the reader doesn't have to guess from the number.
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
        The NodeScaler to hand build_window, or an error saying why there isn't
        one. Without it build_window fits a fresh scaler, and the scores come out
        plausible and meaningless, with no symptom to catch.
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
    return Path(checkpoint_dir or model_dir("cobble_shoal")) / filename


def weights_path(checkpoint_dir=None):
    return Path(checkpoint_dir or model_dir("cobble_shoal")) / WEIGHTS_FILENAME


def load_calibration(filename=SYNTHETIC, checkpoint_dir=None):
    """
    Reads one calibration file, SYNTHETIC or REAL. No default and no fallback,
    since swapping one for the other unnoticed is what we have to prevent.
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
        # Imported here, not at module scope: this module is on the alerting
        # path and node_windows pulls in pandas and the inventory.
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

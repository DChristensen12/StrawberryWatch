"""
Serving Flame Skimmer. It reads the creek through Cobble Shoal's encoder and
scores with the same combined Fisher rule. Apart from loading it's all
CobbleShoalDetector's. Two extra steps match how its calibration was fitted:
impossible readings are dropped, and sensors training didn't see are held
offline and not scored.

Experimental, with one trained checkpoint. See
documents/Math_and_Design_of_FlameSkimmer.pdf.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from strawberrywatch.serving.checkpoint import CheckpointError
from strawberrywatch.serving.cobble import WINDOW, CobbleShoalDetector
from strawberrywatch.serving.contract import DetectorError

WEIGHTS = "flame_skimmer_temp_clock.pt"
CALIBRATION = "flame_skimmer_temp_clock_calibration.json"


class FlameSkimmerDetector(CobbleShoalDetector):
    """Cobble Shoal's detector with Flame Skimmer's weights, calibration and cleaning."""

    name = "flame_skimmer"
    FILES = (WEIGHTS, CALIBRATION)

    def __init__(self, model, calibration, inventory, *args, **kw):
        super().__init__(model, calibration, inventory, *args, **kw)
        corpus = calibration.corpus
        self.offline = frozenset(corpus.get("offline_nodes", ()))
        self.plausible = {
            column: (float(lo), float(hi))
            for column, (lo, hi) in (corpus.get("plausible_range") or {}).items()
        }

    @classmethod
    def load(cls, checkpoint_dir, device="cpu", window=WINDOW, **kw):
        """
        Builds the model from its weights file and loads the calibration. torch
        is imported here for the same reason as in cobble.py. A checkpoint
        trained on another context gets refused, since its scores would look
        fine and be meaningless.
        """
        import torch

        from strawberrywatch import inventory as inventory_module
        from strawberrywatch.anomalies import cobble_calibration
        from strawberrywatch.models.Cobble_Shoal import (
            CANONICAL_VARS,
            CONTEXT_FEATURES,
            SITE_ORDER,
            build_node_registry,
        )
        from strawberrywatch.models.Flame_Skimmer import build_flame_skimmer

        calibration = cobble_calibration.load_calibration(CALIBRATION, checkpoint_dir)
        weights = Path(checkpoint_dir) / WEIGHTS
        if not weights.exists():
            raise CheckpointError(
                f"no weights at {weights}. The checkpoint directory is passed in rather "
                f"than discovered, so this is usually a wrong path, not a missing model."
            )
        blob = torch.load(weights, map_location=device, weights_only=True)
        meta = blob["meta"]
        context = list(meta.get("context_features", CONTEXT_FEATURES))
        if context != list(CONTEXT_FEATURES) or int(meta["window"]) != window:
            raise DetectorError(
                f"{weights.name} was trained at window {meta['window']} with context "
                f"{context}, but serving builds window {window} with {list(CONTEXT_FEATURES)}"
            )
        if calibration.source_checkpoint not in ("", WEIGHTS):
            raise DetectorError(
                f"{CALIBRATION} was fitted for {calibration.source_checkpoint}, not {WEIGHTS}"
            )

        model = build_flame_skimmer(
            len(SITE_ORDER),
            len(CANONICAL_VARS),
            int(meta["num_context"]),
            len(build_node_registry()[0]),
            int(meta["seed"]),
            int(meta["window"]),
            head_width=int(meta["head_width"]),
            prior_sigma=float(meta["prior_sigma"]),
            rho_init=float(meta["rho_init"]),
            n_train=int(meta["n_train_targets"]),
        )
        try:
            model.load_state_dict(blob["state_dict"])
        except RuntimeError as exc:
            raise DetectorError(
                f"weights at {weights} do not fit Flame Skimmer with the node roster in "
                f"Cobble_Shoal.SITE_INVENTORY. Underlying error: {exc}"
            ) from exc
        model.to(device).eval()

        return cls(model, calibration, inventory_module.load(), device, window, **kw)

    def clean(self, frame):
        """NaN out readings no creek produces, with the limits the calibration's corpus used."""
        out = frame.copy()
        for column, (lo, hi) in self.plausible.items():
            if column in out.columns:
                bad = out[column].notna() & ((out[column] < lo) | (out[column] > hi))
                out.loc[bad, column] = np.nan
        return out

    def prepare(self, win):
        """
        Makes the sensors training didn't see look dead (zero value, maximum
        staleness, no target), the way they were in every calibration window.
        """
        ceiling = float(len(win["grid"]))
        for k, node in enumerate(win["nodes"]):
            if node.key in self.offline:
                win["values"][:, k] = 0.0
                win["staleness"][:, k] = ceiling
                win["target_val"][:, k] = 0.0
                win["target_mask"][:, k] = False
        return win

    def score(self, tables, weather=None, leave_out=None):
        """Cobble Shoal's, with the offline sensors' scores blanked. They can't fire or show."""
        result = super().score(tables, weather, leave_out=leave_out)
        for j, node in enumerate(result["nodes"]):
            if node.key in self.offline:
                result["scores"][:, j] = np.nan
                result["fired"][:, j] = False
        return result


__all__ = ["CALIBRATION", "FlameSkimmerDetector", "WEIGHTS"]

"""
Cobble Shoal, loaded and ready to score Night Heron's tables.

Same contract as detector.py, different insides. A live window goes through the
same steps the REAL calibration was fitted through, so a score here means what
its z_q says it means.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from strawberrywatch.serving.checkpoint import CheckpointError
from strawberrywatch.serving.contract import DetectorError, Finding, utc_indexed

logger = logging.getLogger(__name__)

# What every Cobble Shoal artifact so far was trained and calibrated at. The
# reconstruction head is sized by it, so the wrong one fails at load, loudly.
WINDOW = 24

RULE = "combined_fisher"

# Anchors this close to the newest reading get scored each pass. Covers the
# scoring interval plus however late a logger uploads. Anything older is history,
# and the cooldown upstream stops one crossing alerting again every time it is
# rescored on its way out of here.
RECENT = pd.Timedelta(hours=2)

# An anchor is judged once half the nodes have a reading at it, the bar every
# anchor the nulls were fitted on cleared. Below that the score gets compared
# against a distribution fitted on a busier network than the one reporting.
MIN_COVERAGE = 0.5

_cache = {}
_cache_lock = threading.Lock()


class CobbleShoalDetector:
    """
    Cobble Shoal, loaded and ready to score.

    One score per (site, variable) node per anchor, the combined Fisher
    statistic, held to the calibrated z_q once rain_gate has raised it for any
    rain. That is the whole alerting rule, and rain_gate.fired is the one place
    it is written down.
    """

    name = "cobble_shoal"
    RULES = (RULE,)
    # REAL only. The synthetic calibration scores on another scale entirely and
    # records no scaler, see cobble_calibration.py.
    FILES = ("cobble_shoal_weights.pt", "cobble_shoal_calibration_real.json")

    def __init__(
        self,
        model,
        calibration,
        inventory,
        device="cpu",
        window=WINDOW,
        recent=RECENT,
        min_coverage=MIN_COVERAGE,
        rain_gate=None,
    ):
        from strawberrywatch.anomalies.rain_gate import RainGate

        self.model = model
        self.calibration = calibration
        self.inventory = inventory
        self.device = device
        self.window = window
        self.recent = pd.Timedelta(recent)
        self.min_coverage = min_coverage
        # Raised here rather than on the first pass. A window scaled by a fresh fit
        # and scored against nulls fitted in another space looks fine and means
        # nothing.
        self.scaler = calibration.window_scaler()
        # The gate's own defaults unless told otherwise. Which operating point to
        # run is a call for whoever owns the alerting budget, not for this file.
        self.gate = RainGate.from_dict({"base_threshold": calibration.z_q, **(rain_gate or {})})

    @classmethod
    def load(cls, checkpoint_dir, device="cpu", window=WINDOW, **kw):
        """
        Build the model and load its weights and the REAL calibration.

        torch is imported here and not at module scope for the same reason as in
        detector.py: importing the package should not cost the daemon a torch
        import before anything is scored.
        """
        import torch

        from strawberrywatch import inventory as inventory_module
        from strawberrywatch.anomalies import cobble_calibration
        from strawberrywatch.models.Cobble_Shoal import CobbleShoal

        calibration = cobble_calibration.load_calibration(cobble_calibration.REAL, checkpoint_dir)
        weights = cobble_calibration.weights_path(checkpoint_dir)
        if not weights.exists():
            raise CheckpointError(
                f"no weights at {weights}. The checkpoint directory is passed in rather "
                f"than discovered, so this is usually a wrong path, not a missing model."
            )

        model = CobbleShoal.from_metadata({"seed": calibration.seed, "window": window})
        blob = torch.load(weights, map_location=device, weights_only=True)
        try:
            model.load_state_dict(blob["state_dict"] if "state_dict" in blob else blob)
        except RuntimeError as exc:
            raise DetectorError(
                f"weights at {weights} do not fit Cobble Shoal at window {window} with "
                f"the node roster in Cobble_Shoal.SITE_INVENTORY. Underlying error: {exc}"
            ) from exc
        model.to(device).eval()

        return cls(model, calibration, inventory_module.load(), device, window, **kw)

    @classmethod
    def cached(cls, checkpoint_dir, device="cpu", **kw):
        """Same as load, but a repeat call for the same checkpoint reuses the model."""
        key = (str(checkpoint_dir), cls.name, str(device))
        with _cache_lock:
            if key not in _cache:
                _cache[key] = cls.load(checkpoint_dir, device, **kw)
            return _cache[key]

    @property
    def table_names(self):
        """Night Heron tables this reads. footbridge is scnf010 to them."""
        from strawberrywatch.preprocessing import node_windows as nw

        return [nw.SITE_TO_TABLE.get(site, site) for site in nw.SITE_ORDER]

    @property
    def wants_weather(self):
        # Rain is all it takes, and only to move the bar
        return self.gate.mode != "off"

    def findings(self, tables, weather=None):
        """
        Score Night Heron's site tables and say what fired.

        One Finding per node that went over its bar at any recent anchor, with the
        worst score it reached and the anchor it first crossed at.
        """
        from strawberrywatch.preprocessing import node_windows as nw

        result = self.score(tables, weather)
        if not result["anchors"]:
            logger.info("%s: no recent step with enough of the creek reporting to judge", self.name)
            return []
        if not result["rain_applied"] and self.wants_weather:
            logger.info("%s: no rain to go on, every step held to the dry bar", self.name)

        inventory_name = {canonical: name for name, canonical in nw.VARIABLE_MAP.items()}
        scores, over, grid = result["scores"], result["fired"], result["grid"]

        found = []
        for j in np.flatnonzero(over.any(axis=0)):
            node = result["nodes"][j]
            rows = np.flatnonzero(over[:, j])
            worst = rows[np.argmax(scores[rows, j])]
            frame = result["frames"].get(node.site, pd.DataFrame())
            found.append(
                Finding(
                    site=nw.SITE_TO_TABLE.get(node.site, node.site),
                    variable=inventory_name[node.var],
                    rule=RULE,
                    peak=float(scores[worst, j]),
                    threshold=float(self.gate.base_threshold),
                    when=grid[result["anchors"][rows[0]]],
                    readings=frame.get(node.var, pd.Series(dtype=float)).dropna(),
                )
            )
        return found

    def score(self, tables, weather=None):
        """
        Every recent anchor scored and held to its rain-gated bar.

        The grid runs from the earliest reading handed in to the newest, so the
        caller bounds the history. Give it a couple of days: a node that has gone
        quiet then looks properly stale rather than a few hours old, which is
        closer to what training showed it.

        Returns grid, nodes, anchors (grid positions scored), scores and thresholds
        (anchor by node), fired (the same shape, bool), frames (each site's
        readings in sensor units as the model saw them) and rain_applied.
        """
        from strawberrywatch.anomalies.rain_gate import fired

        # The one thing taken from ingest. It is how the archive the calibration
        # was fitted on got shaped, so a live table goes through the same lines.
        from strawberrywatch.ingest.raw_data_loader import tidy_channels
        from strawberrywatch.preprocessing import node_windows as nw

        channels = {}
        for name in self.table_names:
            raw = tables.get(name)
            if raw is not None and not raw.empty:
                channels[name] = tidy_channels(utc_indexed(name, raw))
        stamps = [frame.index for frame in channels.values() if len(frame)]
        if not stamps:
            return {
                "grid": pd.DatetimeIndex([], tz="UTC"),
                "nodes": [],
                "anchors": [],
                "scores": np.empty((0, 0)),
                "thresholds": np.empty((0, 0)),
                "fired": np.zeros((0, 0), dtype=bool),
                "frames": {},
                "rain_applied": False,
            }

        start = min(s.min() for s in stamps).floor(nw.GRID_FREQ)
        end = max(s.max() for s in stamps).floor(nw.GRID_FREQ)
        win = nw.build_window(
            channels, start, end, inventory=self.inventory, scaler=self.scaler, as_of=end
        )
        grid, nodes = win["grid"], win["nodes"]

        covered = win["target_mask"].sum(axis=1) >= self.min_coverage * len(nodes)
        fresh = grid > grid[-1] - self.recent
        anchors = [a for a in range(self.window, len(grid)) if fresh[a] and covered[a]]

        scores = np.full((len(anchors), len(nodes)), np.nan)
        for i, anchor in enumerate(anchors):
            batch = self._on_device(nw.to_batch(win, anchor, self.window))
            scores[i] = np.asarray(self.model.score(batch, self.calibration.nulls)).ravel()

        rain, rain_applied = _rain_on(grid, weather)
        bars = self.gate.thresholds(
            rain, grid.tz_localize(None).to_numpy(), nodes=[node.key for node in nodes]
        )[anchors]

        return {
            "grid": grid,
            "nodes": nodes,
            "anchors": anchors,
            "scores": scores,
            "thresholds": bars,
            "fired": fired(scores, bars),
            "frames": nw.site_frames_from_archive(
                channels, self.inventory, as_of=end, roster=win["roster"]
            ),
            "rain_applied": rain_applied,
        }

    def _on_device(self, batch):
        if str(self.device) == "cpu":
            return batch
        import torch

        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _rain_on(grid, weather):
    """
    Rain in mm per grid step, and whether there was any to go on.

    A gap counts as dry. rain_gate refuses to guess, so the guess is made here,
    and dry is the trigger happy direction, the same one Dusk Crayfish's rain sum
    takes when it skips a NaN.
    """
    if weather is None or weather.empty or "rain_mm" not in weather.columns:
        return np.zeros(len(grid)), False
    rain = utc_indexed("weather", weather)["rain_mm"].resample(grid.freq).sum()
    return rain.reindex(grid).fillna(0.0).to_numpy(dtype=float), True


__all__ = ["CobbleShoalDetector", "MIN_COVERAGE", "RECENT", "RULE", "WINDOW"]

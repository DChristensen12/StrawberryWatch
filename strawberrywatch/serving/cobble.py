"""
Cobble Shoal, loaded and ready to score Night Heron's tables.

Same contract as detector.py, different insides. A live window goes through the
same steps the REAL calibration was fitted through. A score here means what its
z_q says it means.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from strawberrywatch.serving.checkpoint import CheckpointError
from strawberrywatch.serving.contract import (
    DONT_FLAG,
    LEAVE_OUT,
    DetectorError,
    Finding,
    Score,
    check_leave_out,
    utc_indexed,
)

logger = logging.getLogger(__name__)

# What all Cobble Shoal artifacts so far were trained and calibrated at. The
# reconstruction head is sized by it, and the wrong one fails loudly at load.
WINDOW = 24

RULE = "combined_fisher"

# The scoring interval plus however late a logger uploads. Older anchors are
# history, and the cooldown upstream stops a crossing re-alerting as it ages out.
RECENT = pd.Timedelta(hours=2)

# An anchor is judged once half the nodes have a reading at it, the bar all
# anchors the nulls were fitted on cleared. Below that the score gets compared
# against a distribution fitted on a busier network than the one reporting.
MIN_COVERAGE = 0.5

_cache = {}
_cache_lock = threading.Lock()


class CobbleShoalDetector:
    """
    One score per (site, variable) node per anchor, the combined Fisher
    statistic, held to the calibrated z_q once rain_gate has raised it for rain.
    That rule is written down in rain_gate.fired and only there.
    """

    name = "cobble_shoal"
    RULES = (RULE,)
    # Built for missing sensors: a left-out site's readings are just missing,
    # masked like any other gap, and MIN_COVERAGE still decides what's judged
    SITE_CHOICE = (DONT_FLAG, LEAVE_OUT)
    # One anchor over the gated line is a firing
    min_steps = 1
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
        # Raised here and not on the first pass. A window scaled by a fresh fit
        # and scored against nulls fitted in another space looks fine and is
        # meaningless.
        self.scaler = calibration.window_scaler()
        # The gate's own defaults unless told otherwise. Which operating point to
        # run is a call for whoever owns the alerting budget, not for this file.
        self.gate = RainGate.from_dict({"base_threshold": calibration.z_q, **(rain_gate or {})})

    @classmethod
    def load(cls, checkpoint_dir, device="cpu", window=WINDOW, **kw):
        """
        Builds the model and loads its weights and the REAL calibration. torch is
        imported here so importing the package doesn't cost the daemon a torch import.
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

    def set_rain_gate(self, rain_gate=None):
        """Swap the rain gate's settings without reloading the model. z_q stays the calibrated one."""
        from strawberrywatch.anomalies.rain_gate import RainGate

        self.gate = RainGate.from_dict(
            {"base_threshold": self.calibration.z_q, **(rain_gate or {})}
        )

    def scores(self, tables, weather=None, leave_out=None):
        """
        Every recent anchor and node scored, with the gated line it was held to.

        A node at a left-out site doesn't appear. Neither does an anchor the
        node couldn't be scored at (NaN from the model).
        """
        from strawberrywatch.preprocessing import node_windows as nw

        leave_out = check_leave_out(self, leave_out)
        result = self.score(tables, weather, leave_out=leave_out)
        inventory_name = {canonical: name for name, canonical in nw.VARIABLE_MAP.items()}
        grid, base = result["grid"], float(self.gate.base_threshold)
        out = []
        for j, node in enumerate(result["nodes"]):
            table = nw.SITE_TO_TABLE.get(node.site, node.site)
            if table in leave_out:
                continue
            for i, anchor in enumerate(result["anchors"]):
                value = result["scores"][i, j]
                if np.isfinite(value):
                    out.append(
                        Score(
                            table,
                            inventory_name[node.var],
                            RULE,
                            grid[anchor],
                            float(value),
                            float(result["thresholds"][i, j]),
                            base,
                        )
                    )
        return out

    def findings(self, tables, weather=None, leave_out=None):
        """
        Score Night Heron's site tables and say what fired.

        One Finding per node that went over its bar at any recent anchor, with the
        worst score it reached and the anchor it first crossed at.
        """
        from strawberrywatch.preprocessing import node_windows as nw

        leave_out = check_leave_out(self, leave_out)
        result = self.score(tables, weather, leave_out=leave_out)
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
            if nw.SITE_TO_TABLE.get(node.site, node.site) in leave_out:
                continue
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

    def score(self, tables, weather=None, leave_out=None):
        """
        Every recent anchor scored and held to its rain-gated bar.

        The grid runs from the earliest reading handed in, and the caller bounds
        the history. Give it a couple of days. A quiet node then looks properly
        stale, like it did in training.

        Returns grid, nodes, anchors (grid positions scored), scores and
        thresholds (anchor by node), fired (same shape, bool), frames (each
        site's readings as the model saw them) and rain_applied. We don't read
        left-out tables, and they count against MIN_COVERAGE like any other gap.
        """
        from strawberrywatch.anomalies.rain_gate import fired

        # Our only import from ingest. It shaped the archive the calibration was
        # fitted on, and a live table goes through the same lines.
        from strawberrywatch.ingest.raw_data_loader import tidy_channels
        from strawberrywatch.preprocessing import node_windows as nw

        leave_out = set(leave_out or ())
        channels = {}
        for name in self.table_names:
            if name in leave_out:
                continue
            raw = tables.get(name)
            if raw is not None and not raw.empty:
                channels[name] = self.clean(tidy_channels(utc_indexed(name, raw)))
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
        win = self.prepare(
            nw.build_window(
                channels, start, end, inventory=self.inventory, scaler=self.scaler, as_of=end
            )
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

    def clean(self, frame):
        """
        Hook for cleaning one site's readings before the window is built. A
        no-op here, since the REAL calibration was fitted on raw readings.
        """
        return frame

    def prepare(self, win):
        """Hook for the built window before scoring. Also a no-op here."""
        return win

    def _on_device(self, batch):
        if str(self.device) == "cpu":
            return batch
        import torch

        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _rain_on(grid, weather):
    """
    Rain in mm per grid step, and whether there was any to go on.

    A gap counts as dry. rain_gate refuses to guess, and we guess here. Dry errs
    toward more alerts, as Dusk Crayfish's rain sum does when it skips a NaN.
    """
    if weather is None or weather.empty or "rain_mm" not in weather.columns:
        return np.zeros(len(grid)), False
    rain = utc_indexed("weather", weather)["rain_mm"].resample(grid.freq).sum()
    return rain.reindex(grid).fillna(0.0).to_numpy(dtype=float), True


__all__ = ["CobbleShoalDetector", "MIN_COVERAGE", "RECENT", "RULE", "WINDOW"]

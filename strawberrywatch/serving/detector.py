"""
The one object an outside caller needs to run Dusk Crayfish.

Hand it a directory with a checkpoint in it and a frame of readings, get back a
verdict per site. Nothing in here reads settings.yaml, resolves a repository
root, opens a .env, or touches the ingest package. Everything that used to come
from a module level constant is a constructor argument with our value as the
default.

Loading is cached, keyed by directory and device, because the caller running
this is a daemon that loops every twenty seconds and reading weights off disk
every pass would be silly.
"""

from __future__ import annotations

import threading

import pandas as pd

from strawberrywatch.serving import windows
from strawberrywatch.serving.checkpoint import CheckpointError, load_checkpoint

DEFAULT_WINDOW = 24

# One entry per (directory, model, device). Small and long lived, so a plain
# dict under a lock beats anything with an eviction policy.
_cache = {}
_cache_lock = threading.Lock()


class DetectorError(RuntimeError):
    """We cannot score what we were given, and the message says why."""


class DuskCrayfishDetector:
    """
    Dusk Crayfish, loaded and ready to score.

    Built for this one model on purpose. Cobble Shoal reads per (site, variable)
    node series rather than a site by feature tensor, so it needs a different
    windower, and writing a shared base class before that one exists would be
    guessing at it. What is already shared sits in checkpoint.py, which knows
    nothing about either model.
    """

    def __init__(
        self,
        model,
        checkpoint,
        device,
        window=DEFAULT_WINDOW,
        operating_point=None,
        rain_params=None,
        imputation_limit_hours=3.0,
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.window = window
        self.operating_point = operating_point
        self.rain_params = rain_params
        self.imputation_limit_hours = imputation_limit_hours
        self._normalization = checkpoint.normalization()

    @classmethod
    def load(
        cls,
        checkpoint_dir,
        model_name="dusk_crayfish",
        device="cpu",
        window=DEFAULT_WINDOW,
        operating_point=None,
        rain_params=None,
        imputation_limit_hours=3.0,
    ):
        """
        Build the model and load its weights.

        torch and torch_geometric are imported here rather than at module scope so
        importing this package stays cheap. That matters for the Django daemon,
        which imports us at startup and only pays for torch on the first cycle
        that actually scores something.
        """
        import torch

        from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

        checkpoint = load_checkpoint(checkpoint_dir, model_name)
        model = DuskCrayfish(
            num_node_features=len(checkpoint.feature_cols),
            num_nodes=len(checkpoint.location_to_idx),
            **checkpoint.architecture,
        )
        state = torch.load(checkpoint.weights_path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise DetectorError(
                f"weights at {checkpoint.weights_path} do not fit the model described by "
                f"{checkpoint.source}. Usually the checkpoint predates recording its own "
                f"architecture and the fallback guess was wrong. Underlying error: {exc}"
            ) from exc
        model.to(device).eval()

        return cls(
            model, checkpoint, device, window, operating_point, rain_params, imputation_limit_hours
        )

    @classmethod
    def cached(cls, checkpoint_dir, model_name="dusk_crayfish", device="cpu", **kw):
        """Same as load, but a repeat call for the same checkpoint reuses the model."""
        key = (str(checkpoint_dir), model_name, str(device))
        with _cache_lock:
            if key not in _cache:
                _cache[key] = cls.load(checkpoint_dir, model_name, device, **kw)
            return _cache[key]

    @property
    def sites(self):
        return self.checkpoint.sites

    @property
    def features(self):
        return list(self.checkpoint.feature_cols)

    def expected_cadence(self, frame):
        """
        What one window covers, given how often this frame samples.

        The window is 24 rows, not 24 hours. On our 15 minute grid that is six
        hours of creek. Hand the same model 5 minute data and the same 24 rows is
        two hours, and the dynamics it learned no longer line up. Nothing in the
        model checks, so a caller that cares should.
        """
        step = windows.cadence(frame)
        return None if step is None else step * self.window

    def score(self, frame, rain=None):
        """
        Run the model over a frame of readings and judge every site.

        frame is one row per (timestamp, location), DatetimeIndex in UTC, one
        column per feature in self.features. Clock features get added for you if
        they are absent.

        rain is an optional Series of rain_mm indexed by time. Without it the rain
        adjustment never engages, which leaves Rule 1 comparing against its dry
        weather bar during a storm. That direction is the trigger happy one, so we
        say so in the result rather than letting it pass quietly.

        Returns {"verdicts": {...}, "windows": int, "window_end": Timestamp or None,
        "rain_applied": bool}. verdicts is what detect_anomalies returns, keyed by
        site, and is empty when there was not enough data to build a window.
        """
        from strawberrywatch.anomalies.anomaly_detector import detect_anomalies
        from strawberrywatch.utils.graph_utils import create_graph_topology

        frame = windows.add_time_features(frame)
        sequences, targets, stamps, node_mask = windows.build_windows(
            frame,
            self.checkpoint.feature_cols,
            self.checkpoint.location_to_idx,
            self._normalization,
            self.window,
            self.imputation_limit_hours,
        )

        if len(sequences) == 0:
            return {"verdicts": {}, "windows": 0, "window_end": None, "rain_applied": False}

        edge_index, _, _ = create_graph_topology(
            location_to_idx=self.checkpoint.location_to_idx,
            device=self.device,
            announce=False,
        )

        # detect_anomalies reads two things off the raw frame: which steps a site
        # really reported at, and rain. It wants them on one frame, so attach rain
        # here rather than changing its signature.
        raw = frame
        if rain is not None:
            raw = frame.copy()
            raw["rain_mm"] = pd.Series(rain).reindex(frame.index)

        verdicts, _ = detect_anomalies(
            self.model,
            sequences,
            targets,
            stamps,
            node_mask,
            edge_index,
            self.checkpoint.detection_metadata(),
            df_original=raw,
            locations=self.sites,
            device=self.device,
            operating_point=self.operating_point,
            rain_params=self.rain_params,
        )

        return {
            "verdicts": verdicts,
            "windows": len(sequences),
            "window_end": stamps[-1],
            "rain_applied": rain is not None,
        }


__all__ = ["DuskCrayfishDetector", "DetectorError", "CheckpointError"]

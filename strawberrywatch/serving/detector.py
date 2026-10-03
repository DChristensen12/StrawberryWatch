"""
The one object an outside caller needs to run Dusk Crayfish.

Hand it a directory with a checkpoint in it and Night Heron's site tables, get
back a Finding for whatever fired. Nothing in here reads settings.yaml, resolves
a repository root, opens a .env, or touches the ingest package. Everything that
used to come from a module level constant is a constructor argument with our
value as the default.

Loading is cached, keyed by directory and device, because the caller running
this is a daemon that loops every twenty seconds and reading weights off disk
every pass would be silly.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pandas as pd

from strawberrywatch.serving import windows
from strawberrywatch.serving.checkpoint import CheckpointError, load_checkpoint
from strawberrywatch.serving.contract import DetectorError, Finding, utc_indexed

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 24

# Raw logger column names to what the model calls them. The same mapping
# ingest/data_loader.py uses, repeated so this module never imports ingest.
COLUMNS = {
    "Meter_Hydros21_Cond": "conductivity",
    "Meter_Hydros21_Depth": "depth",
    "Meter_Hydros21_Temp": "temperature",
}

WEATHER = ("rain_mm", "air_temp_c", "shortwave_radiation")

# One entry per (directory, model, device). Small and long lived, so a plain
# dict under a lock beats anything with an eviction policy.
_cache = {}
_cache_lock = threading.Lock()


class DuskCrayfishDetector:
    """
    Dusk Crayfish, loaded and ready to score.

    No base class shared with Cobble Shoal's detector in cobble.py. Their insides
    have nothing worth sharing, so all the two agree on is contract.py: raw site
    tables in, Findings out.
    """

    name = "dusk_crayfish"
    # As detect_anomalies names them in rules_fired
    RULES = ("forecast_residual", "level_shift")
    # All a checkpoint folder needs for load() to work with no checkout around it
    FILES = ("dusk_crayfish_weights.pt", "dusk_crayfish_serving.json")

    def __init__(
        self,
        model,
        checkpoint,
        device,
        window=DEFAULT_WINDOW,
        operating_point=None,
        rain_params=None,
        imputation_limit_hours=3.0,
        grid="15min",
    ):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.window = window
        self.operating_point = operating_point
        self.rain_params = rain_params
        self.imputation_limit_hours = imputation_limit_hours
        self.grid = grid
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
        grid="15min",
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
            model,
            checkpoint,
            device,
            window,
            operating_point,
            rain_params,
            imputation_limit_hours,
            grid,
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

    @property
    def table_names(self):
        """Night Heron tables this reads. Its site names already are their table names."""
        return self.sites

    @property
    def wants_weather(self):
        return any(c in self.features for c in WEATHER)

    def findings(self, tables, weather=None):
        """
        Score Night Heron's site tables and say what fired.

        tables is {table: DataFrame} indexed by reading time, raw logger columns.
        weather is Open-Meteo on its 15 minute UTC index, or None. Comes back as a
        list of Finding, empty when nothing fired or nothing could be judged.
        """
        from strawberrywatch.anomalies.anomaly_detector import LEVEL_SHIFT_K

        frame = self._frame(tables)
        if frame.empty:
            logger.info("%s: none of %s had readings", self.name, ", ".join(self.sites))
            return []
        if "conductivity" not in frame.columns:
            # Both rules score it, and _fill_absent would otherwise hand
            # detect_anomalies the training mean dressed up as a reading.
            logger.info("%s: no conductivity column on any table, nothing to judge", self.name)
            return []

        frame, got_weather = _merge_weather(frame, self.features, weather)
        if not got_weather:
            logger.info("%s: scoring without weather context", self.name)

        sensor_cols = [c for c in self.features if c in frame.columns] + ["location"]
        frame = _fill_absent(frame[sensor_cols], self.features, self._normalization)

        rain = frame["rain_mm"] if got_weather and "rain_mm" in frame.columns else None
        result = self.score(frame, rain=rain)
        if not result["windows"]:
            logger.info("%s: not enough consecutive readings to fill a window", self.name)
            return []

        # Rule 1 is held to the node's calibrated threshold and Rule 2 to the flat
        # k, the same split main.py reports. Rain moves Rule 1's bar per step, so
        # this is the calibrated one rather than wherever it sat at the peak.
        node_bars = self.checkpoint.calibration.get("node_thresholds", {})
        level_k = (self.operating_point or {}).get("level_shift_k", LEVEL_SHIFT_K)
        stamps = result["timestamps"]

        found = []
        for site, verdict in result["verdicts"].items():
            if not verdict.get("judged") or not verdict.get("flagged"):
                continue
            readings = frame.loc[frame["location"] == site, "conductivity"].dropna()
            for rule in verdict["rules_fired"]:
                detail = verdict["rule1" if rule == "forecast_residual" else "rule2"]
                crossed = detail.get("over_timesteps")
                when = (
                    stamps[int(np.argmax(crossed))]
                    if crossed is not None and crossed.any()
                    else result["window_end"]
                )
                found.append(
                    Finding(
                        site=site,
                        variable="conductivity",
                        rule=rule,
                        peak=float(detail["peak_deviation"]),
                        threshold=float(
                            node_bars.get(site, np.nan) if rule == "forecast_residual" else level_k
                        ),
                        when=when,
                        readings=readings,
                    )
                )
        return found

    def _frame(self, tables):
        """
        The site tables stacked into the one long frame score() reads.

        Only the logger columns this model knows come along, so weather or anything
        else riding on a table cannot collide with what _merge_weather adds. A site
        with no table or no rows is skipped, because one dead logger should not
        stop the other three being scored.
        """
        frames = []
        for site in self.sites:
            raw = tables.get(site)
            if raw is None or raw.empty:
                continue
            raw = utc_indexed(site, raw)
            raw = raw[[c for c in COLUMNS if c in raw.columns]].rename(columns=COLUMNS)
            frames.append(raw.assign(location=site))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames).rename_axis("datetime").sort_index()

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
        "timestamps": [...], "rain_applied": bool}. verdicts is what
        detect_anomalies returns, keyed by site, and is empty when there was not
        enough data to build a window. timestamps are the window ends the rules'
        over_timesteps line up with.
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
            self.grid,
        )

        if len(sequences) == 0:
            return {
                "verdicts": {},
                "windows": 0,
                "window_end": None,
                "timestamps": [],
                "rain_applied": False,
            }

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
            "timestamps": stamps,
            "rain_applied": rain is not None,
        }


def _merge_weather(frame, features, weather):
    """
    Merge the weather columns the model trained on, if the caller had any.

    Night Heron does not store weather, so whoever calls us fetches it. Anything
    missing stays absent and _fill_absent gives it the training mean, which costs
    the model its weather context, mostly during storms, but not its ability to
    run. Returns whether any weather made it in.
    """
    wanted = [c for c in WEATHER if c in features]
    if not wanted or weather is None or weather.empty:
        return frame, False

    have = [c for c in wanted if c in weather.columns]
    if not have:
        return frame, False

    # Weather is on its own 15 minute grid. Snap each reading to the quarter hour
    # it falls in rather than merging on an exact timestamp match that will miss.
    keyed = frame.copy()
    keyed["_quarter"] = keyed.index.floor("15min")
    merged = keyed.merge(
        weather[have].resample("15min").mean(),
        how="left",
        left_on="_quarter",
        right_index=True,
    ).drop(columns=["_quarter"])
    merged.index = frame.index
    return merged, True


def _fill_absent(frame, features, normalization):
    """
    Give every trained feature a column, filling any we could not get.

    A missing feature gets its training mean, which lands on zero once the window
    builder normalizes. That is the same thing main.py does when a weather fetch
    fails mid run, and it means "no signal" rather than "a reading of zero".
    """
    out = frame.copy()
    for name in features:
        if name not in out.columns:
            out[name] = normalization[name][0]
    return out


__all__ = ["DuskCrayfishDetector", "DetectorError", "CheckpointError"]

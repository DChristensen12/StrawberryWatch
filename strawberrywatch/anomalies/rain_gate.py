"""
Rain gating for the Cobble Shoal alerting path.

The model has no weather input and gains none here. This raises the alerting
bar during and after rain, because runoff genuinely moves conductivity and an
operator does not want that arriving as an alert.

The threshold moves, the score never does. POT fitted z_q against the
fault-free distribution of the combined score, so a score scaled by a rain
factor is no longer a draw from the distribution that fit was made on and every
p-value under it quietly stops meaning what it says. tests/test_rain_gate.py
asserts the scores are bit-identical across modes.

Rain comes from the caller as a plain array of per-timestep amounts. This
module never fetches weather.

The measured cost, and why it is written down here

These tables were measured in August 2026 against the comparison harness
corpus, which has since been deleted. THEY CANNOT BE RECOMPUTED FROM THIS
REPOSITORY. They are the evidence behind the multiplier and decay_hours
defaults, and anyone moving those parameters needs to know what the last
measurement said. Cobble Shoal, seed 20260806, combined Fisher at q=1e-4,
z_q = 39.130, multiplier 2.0, decay_hours 36, lookback_hours 12, wet_mm 0.1.

A synthetic hourly rain series was laid over the 275 stored fault-sweep cases
so 30% fell in a raised-threshold window. The sweep cases have no real time
axis, so that assignment is arbitrary by construction. These show the shape of
the trade, not a measurement of this creek.

  shape       n     % wet   off      step     decay    decay - off
  decouple    17    0%      100.0%   100.0%   100.0%   +0.0%
  drift       51    73%     100.0%   100.0%   100.0%   +0.0%
  partial     119   35%     98.3%    88.2%    88.2%    -10.1%
  slow_all    3     0%      100.0%   100.0%   100.0%   +0.0%
  spike       51    0%      100.0%   100.0%   100.0%   +0.0%
  stale       17    24%     100.0%   100.0%   100.0%   +0.0%
  stuck       17    0%      100.0%   100.0%   100.0%   +0.0%
  all         275   30%     99.3%    94.9%    94.9%    -4.4%

The whole cost lands on partial, and it is 10 points of detection on the one
fault shape whose signal is weakest. Every other shape clears 2*z_q as easily
as it cleared z_q. A gate tuned for rain is in practice a gate that trades away
partial faults specifically.

False alarms on 3536 fault-free windows by 17 nodes:

  mode    detection (all shapes)   false alarms   detections lost   alarms removed
  off     99.3%                    2.394%         0.0%              0.000%
  step    94.9%                    2.249%         4.4%              0.145%
  decay   94.9%                    2.128%         4.4%              0.266%

Restricted to wet windows, where the gate does anything, the alarm rate falls
from 4.86% (off) to 2.46% (step) to 0.44% (decay).

Two caveats. Those fault-free windows are the regime-swept ones, spanning noise
and diurnal regimes the nulls were not fitted on, which is why the ungated rate
is 2.4% and not the nominal 1e-4. Only the comparison between rows is
meaningful. And step and decay tie on detection because of how the arbitrary
case-to-time mapping fell, not because the rules are equivalent.

The delayed first flush, which is the failure the old step design documents.
Rain falls for six hours and a conductivity pulse arrives 21 hours after it
stops, outside the 12 hour lookback and inside the 36 hour decay:

  pulse         off     step    decay        deployed
  1.10 * z_q    fires   fires   suppressed   suppressed
  1.25 * z_q    fires   fires   suppressed   suppressed
  1.50 * z_q    fires   fires   fires        suppressed
  1.75 * z_q    fires   fires   fires        suppressed
  1.90 * z_q    fires   fires   fires        fires

At the pulse step is back at 1.00*z_q, so it flags a delayed first flush at any
magnitude. decay holds 1.42*z_q and deployed holds 1.75*z_q. Neither is
asserted correct and the tests do not assert one. decay_hours is the dial and
this table is what moving it trades.

No operating point is recommended. The choice belongs to whoever owns the
alerting budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

MODES = ("off", "step", "decay", "deployed")

# The project's sampling grid. Only used when the caller passes rain without
# timestamps, in which case the array is assumed to be evenly spaced at this
# interval; pass timestamps if it is not.
DEFAULT_SAMPLE_HOURS = 0.25


class RainGateConfigError(ValueError):
    """A parameter an operator can set was set to something meaningless."""


def _check(name, value, lo, hi, integerish=False):
    """One range check with an error message an operator can act on."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise RainGateConfigError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(v):
        raise RainGateConfigError(f"{name} must be finite, got {value!r}")
    if not (lo <= v <= hi):
        raise RainGateConfigError(f"{name} must be between {lo} and {hi}, got {v!r}")
    return int(v) if integerish else v


@dataclass
class RainGate:
    """
    Per-timestep alerting threshold as a function of a rain series.

    Parameters, all operator-settable and all documented here:

      base_threshold  the POT threshold z_q for the combined score. The gate
                      never computes this; it is whatever calibration produced.
      multiplier      how far the bar is raised when it is as wet as it gets.
                      1.0 disables the raise without disabling the gate.
      decay_hours     how long after the last wet reading the threshold takes
                      to come back to base. "decay" and "deployed" only.
      wet_mm          the rain amount that counts as wet at all. Below it a
                      reading is treated as dry, which keeps sensor dribble and
                      trace condensation from holding the threshold up.
      lookback_hours  the window whose rain total the "step" and "deployed"
                      rules test against wet_mm.
      mode            "off", "step", "decay" or "deployed"; see below.
      sample_hours    spacing of the rain array when no timestamps are given.
      per_node        optional {node: multiplier} overriding `multiplier` at
                      named nodes. None means one catchment-wide threshold,
                      which is the default because rain is catchment-wide.

    Modes:

      "off"       tau = base everywhere. The identity case, so the gate can be
                  left in the pipeline and turned off.

      "step"      tau = multiplier*base while the lookback-window rain total
                  exceeds wet_mm, base otherwise. A hard edge in both
                  directions.

      "decay"     tau = base * (1 + (m-1) * clip(1 - h/decay_hours, 0, 1)),
                  h = hours since the last wet reading. Full raise while it is
                  raining, then a linear return to base over decay_hours. This
                  is the mode that covers a first-flush pulse arriving after
                  the lookback window has already closed.

      "deployed"  what strawberrywatch/anomalies/anomaly_detector.py
                  `_rain_multipliers` actually does today: hold the full
                  multiplier while the lookback total is wet, then taper over
                  decay_hours measured from the moment the lookback window
                  closes. Carried so the two new modes can be read against
                  production rather than against a description of it.

    "step" and "deployed" ask whether the lookback TOTAL clears wet_mm.
    "decay" measures from the last single reading above wet_mm. The two
    readings of wet_mm are deliberate and are what each rule's own definition
    calls for, but it means wet_mm is a per-window total in one mode and a
    per-reading floor in the other.
    """

    base_threshold: float
    multiplier: float = 2.0
    decay_hours: float = 36.0
    wet_mm: float = 0.1
    lookback_hours: float = 12.0
    mode: str = "decay"
    sample_hours: float = DEFAULT_SAMPLE_HOURS
    per_node: dict | None = None
    node_multipliers: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise RainGateConfigError(f"mode must be one of {', '.join(MODES)}, got {self.mode!r}")
        self.base_threshold = _check("base_threshold", self.base_threshold, -1e12, 1e12)
        # A multiplier below 1 would lower the bar during rain, which is the
        # opposite of what this layer is for and is far more likely a typo.
        self.multiplier = _check("multiplier", self.multiplier, 1.0, 100.0)
        self.decay_hours = _check("decay_hours", self.decay_hours, 1e-6, 24 * 365.0)
        self.wet_mm = _check("wet_mm", self.wet_mm, 0.0, 1000.0)
        self.lookback_hours = _check("lookback_hours", self.lookback_hours, 1e-6, 24 * 365.0)
        self.sample_hours = _check("sample_hours", self.sample_hours, 1e-6, 24.0)

        self.node_multipliers = {}
        if self.per_node is not None:
            if not isinstance(self.per_node, dict):
                raise RainGateConfigError(
                    f"per_node must be a mapping of node to multiplier, got "
                    f"{type(self.per_node).__name__}"
                )
            for node, m in self.per_node.items():
                self.node_multipliers[node] = _check(f"per_node[{node!r}]", m, 1.0, 100.0)

    # Configuration

    @classmethod
    def from_dict(cls, cfg):
        """
        Build from a plain dict, which is also what a YAML or JSON block parses
        to. Unknown keys are refused rather than ignored, because a silently
        dropped `mutliplier` is a gate that quietly does nothing.
        """
        if not isinstance(cfg, dict):
            raise RainGateConfigError(f"config must be a mapping, got {type(cfg).__name__}")
        known = {
            "base_threshold",
            "multiplier",
            "decay_hours",
            "wet_mm",
            "lookback_hours",
            "mode",
            "sample_hours",
            "per_node",
        }
        unknown = set(cfg) - known
        if unknown:
            raise RainGateConfigError(
                f"unknown config key(s): {', '.join(sorted(unknown))}. "
                f"Known keys: {', '.join(sorted(known))}"
            )
        if "base_threshold" not in cfg:
            raise RainGateConfigError(
                "base_threshold is required; it is the calibrated POT threshold "
                "z_q and the gate does not invent one"
            )
        return cls(**cfg)

    def to_dict(self):
        """Round-trips through from_dict."""
        return {
            "base_threshold": self.base_threshold,
            "multiplier": self.multiplier,
            "decay_hours": self.decay_hours,
            "wet_mm": self.wet_mm,
            "lookback_hours": self.lookback_hours,
            "mode": self.mode,
            "sample_hours": self.sample_hours,
            "per_node": dict(self.node_multipliers) or None,
        }

    @classmethod
    def from_yaml(cls, text):
        """Parse a YAML block. JSON is valid YAML, so this takes both."""
        try:
            import yaml
        except ImportError:
            return cls.from_dict(json.loads(text))
        return cls.from_dict(yaml.safe_load(text))

    def to_yaml(self):
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    # Rain arithmetic

    def _hours(self, rain_mm, timestamps):
        """Elapsed hours at each sample, from timestamps or the fixed grid."""
        rain = np.asarray(rain_mm, dtype=float).ravel()
        if np.any(~np.isfinite(rain)):
            raise RainGateConfigError(
                "rain_mm contains non-finite values; the caller must decide "
                "whether a gap means dry or unknown before passing it here"
            )
        if np.any(rain < 0):
            raise RainGateConfigError("rain_mm contains negative amounts")
        if timestamps is None:
            return rain, np.arange(rain.size, dtype=float) * self.sample_hours
        ts = np.asarray(timestamps)
        if ts.size != rain.size:
            raise RainGateConfigError(
                f"timestamps has {ts.size} entries but rain_mm has {rain.size}"
            )
        if np.issubdtype(ts.dtype, np.datetime64) or ts.dtype == object:
            ts = np.asarray(ts, dtype="datetime64[s]").astype("int64") / 3600.0
            ts = ts - ts[0]
        else:
            ts = ts.astype(float)
        if np.any(np.diff(ts) < 0):
            raise RainGateConfigError("timestamps must be non-decreasing")
        return rain, ts

    def hours_since_wet(self, rain_mm, timestamps=None):
        """
        Hours since the last reading above wet_mm, per timestep. Zero while it
        is raining, +inf before any rain has been seen.

        Exposed on its own because it is the answer to "why was the bar up at
        13:45", so an operator can read it straight off without re-deriving the
        rule from the threshold.
        """
        rain, hours = self._hours(rain_mm, timestamps)
        wet = rain > self.wet_mm
        last = np.where(wet, hours, -np.inf)
        last = np.maximum.accumulate(last)
        with np.errstate(invalid="ignore"):
            return np.where(np.isfinite(last), hours - last, np.inf)

    def lookback_total(self, rain_mm, timestamps=None):
        """
        Rain total over the trailing lookback_hours at each timestep,
        inclusive of the current reading. This is what "step" and "deployed"
        test against wet_mm.
        """
        rain, hours = self._hours(rain_mm, timestamps)
        # searchsorted rather than a fixed sample count, so an irregular series
        # gets a real time window instead of a count of rows.
        start = np.searchsorted(hours, hours - self.lookback_hours, side="left")
        cum = np.concatenate([[0.0], np.cumsum(rain)])
        return cum[np.arange(rain.size) + 1] - cum[start]

    def multipliers(self, rain_mm, timestamps=None):
        """
        The catchment-wide multiplier per timestep, before any per-node
        override. Always >= 1 and <= self.multiplier.
        """
        rain, hours = self._hours(rain_mm, timestamps)
        n = rain.size
        m = self.multiplier

        if self.mode == "off":
            return np.ones(n)

        if self.mode == "step":
            return np.where(self.lookback_total(rain, timestamps) > self.wet_mm, m, 1.0)

        if self.mode == "decay":
            h = self.hours_since_wet(rain, timestamps)
            frac = np.clip(1.0 - h / self.decay_hours, 0.0, 1.0)
            return 1.0 + (m - 1.0) * frac

        # "deployed": plateau while the lookback total is wet, then taper from
        # the moment the lookback window closes. Same arithmetic as
        # _rain_multipliers in the production detector.
        wet_window = self.lookback_total(rain, timestamps) > self.wet_mm
        h = self.hours_since_wet(rain, timestamps) - self.lookback_hours
        with np.errstate(invalid="ignore"):
            frac = np.clip(h / self.decay_hours, 0.0, 1.0)
        taper = 1.0 + (m - 1.0) * (1.0 - frac)
        taper = np.where(np.isfinite(h), taper, 1.0)
        return np.where(wet_window, m, taper)

    def thresholds(self, rain_mm, timestamps=None, nodes=None):
        """
        Per-timestep alerting threshold.

        Returns (T,) when nodes is None, which is the catchment-wide default.
        Given a sequence of node identifiers it returns (T, len(nodes)), with
        any node named in per_node using its own multiplier.
        """
        mult = self.multipliers(rain_mm, timestamps)
        if nodes is None:
            return self.base_threshold * mult
        # A per-node multiplier rescales only the raised part: at zero rain
        # every node sits on the same base threshold, which is the calibrated
        # one, and only the size of the rain-time raise differs.
        out = np.empty((mult.size, len(nodes)), dtype=float)
        for j, node in enumerate(nodes):
            m_node = self.node_multipliers.get(node, self.multiplier)
            if self.multiplier > 1.0:
                scale = (mult - 1.0) / (self.multiplier - 1.0)
            else:
                scale = np.zeros_like(mult)
            out[:, j] = self.base_threshold * (1.0 + (m_node - 1.0) * scale)
        return out

    # Audit trail

    def decisions(self, scores, pvalues, rain_mm, timestamps=None, nodes=None):
        """
        One record per timestep per node, carrying everything needed to answer
        "why did this alert fire" or "why did it not" without re-running
        anything: the raw score and its p-value exactly as the detector
        produced them, the calibrated base threshold, the multiplier this gate
        applied, the threshold that came out, the hours since wet that drove
        it, and whether it fired.

        scores and pvalues are (T,) or (T, N) and are copied through
        untouched. Nothing here may modify them; that is the whole point.
        """
        s = np.asarray(scores, dtype=float)
        p = np.asarray(pvalues, dtype=float)
        if s.shape != p.shape:
            raise RainGateConfigError(
                f"scores {s.shape} and pvalues {p.shape} must have the same shape"
            )
        if s.ndim == 1:
            s, p = s[:, None], p[:, None]
        n_t, n_n = s.shape
        if nodes is None:
            nodes = list(range(n_n))
        elif len(nodes) != n_n:
            raise RainGateConfigError(f"{len(nodes)} node names for {n_n} score columns")

        tau = self.thresholds(rain_mm, timestamps, nodes=nodes)
        hsw = self.hours_since_wet(rain_mm, timestamps)
        if tau.shape[0] != n_t:
            raise RainGateConfigError(f"rain series has {tau.shape[0]} steps but scores have {n_t}")

        ts = None if timestamps is None else np.asarray(timestamps)
        base = self.base_threshold
        # One definition of the firing rule, shared with the array path. Writing
        # the comparison out again here is how the two quietly disagree later.
        fired_grid = fired(s, tau)
        records = []
        for i in range(n_t):
            for j, node in enumerate(nodes):
                score = float(s[i, j])
                thr = float(tau[i, j])
                records.append(
                    {
                        "timestep": i,
                        "timestamp": None if ts is None else ts[i],
                        "node": node,
                        "score": score,
                        "pvalue": float(p[i, j]),
                        "base_threshold": base,
                        "multiplier": thr / base if base else float("nan"),
                        "threshold": thr,
                        "hours_since_wet": float(hsw[i]),
                        "mode": self.mode,
                        "fired": bool(fired_grid[i, j]),
                        "scorable": bool(np.isfinite(score)),
                    }
                )
        return records


def fired(scores, thresholds):
    """
    The alerting decision, and the only place it is written down.

    NaN never fires. A score that could not honestly be computed is "not
    scorable", which is not the same as "did not exceed", and must not reach an
    operator as a quiet all-clear.
    """
    s = np.asarray(scores, dtype=float)
    return np.isfinite(s) & (s > np.asarray(thresholds, dtype=float))


# An operator edits this block, not the code. Every value is named, and the
# comment beside it says which direction makes alerts more sensitive.
EXAMPLE_CONFIG_YAML = """
# Rain gate for the Cobble Shoal alerting path. Optional: mode "off" disables it.
base_threshold: 39.13047989749433   # POT z_q at q=1e-4. Comes from calibration.
mode: decay                         # off | step | decay | deployed
multiplier: 2.0                     # higher = fewer alerts during rain
decay_hours: 36.0                   # longer = suppression persists further past rain
wet_mm: 0.1                         # higher = more rain needed before suppressing
lookback_hours: 12.0                # step/deployed only: the window tested against wet_mm
sample_hours: 0.25                  # spacing of the rain array when no timestamps given
per_node: null                      # or {node_7: 3.0} for a site that runs off harder
"""

# The rain gate

**The model has no weather input, and this layer does not give it one.** Riffle Darner
sees conductivity, the graph, the clock and nothing else. The rain gate is an
optional alerting layer that sits on top of the calibrated detector and raises
the alerting threshold during and after rain. It can be turned off (`mode:
off`) without removing it from the pipeline, and with it off the detection
results are bit-identical to the ungated path, asserted, not assumed, in
`tests/test_rain_gate.py`.

## The one rule

**The threshold moves. The score never does.**

POT fitted `z_q` against the fault-free distribution of the combined Fisher
score. A score multiplied by a rain factor is no longer a draw from the
distribution that fit was made on, so the nominal false alarm rate and every
p-value under it would quietly stop meaning what they say, and nothing would
report that it had happened. Multiplying the *threshold* leaves the score, the
p-value and the calibration exactly where they were, and turns the adjustment
into a stated alerting decision an operator can read off the audit record.

`test_item11_scores_and_pvalues_identical_across_modes` asserts the scores and
p-values are bit-identical across all four modes. If that test ever fails, the
gate has become part of the detector and the calibration is void.

## Parameters

All of these are settable from a dict, a YAML block or a JSON block. Nobody
needs to edit Python to change them. `RainGate.from_dict` refuses unknown keys
rather than ignoring them, so a misspelled `mutliplier` is an error and not a
gate that silently does nothing.

| parameter | plain language | default |
|---|---|---|
| `base_threshold` | The calibrated POT threshold `z_q`. The gate never computes this; it is whatever calibration produced. Required. |, |
| `multiplier` | How far the bar is raised when it is as wet as it gets. `2.0` means an alert needs twice the usual score during rain. `1.0` disables the raise. | `2.0` |
| `decay_hours` | How long after the last wet reading the threshold takes to come back to normal. `decay` and `deployed` modes only. | `36.0` |
| `wet_mm` | How much rain counts as rain at all. Below this, a reading is treated as dry. Keeps trace condensation and gauge dribble from holding the bar up for days. | `0.1` |
| `lookback_hours` | The trailing window whose rain *total* is tested against `wet_mm`. `step` and `deployed` modes only. | `12.0` |
| `mode` | `off`, `step`, `decay`, `deployed`. See below. | `decay` |
| `sample_hours` | Spacing of the rain array when the caller passes no timestamps. `0.25` is the project's 15-minute grid. Pass timestamps instead if the series is irregular. | `0.25` |
| `per_node` | Optional `{node: multiplier}` for a site known to respond differently to runoff. `null` means one catchment-wide threshold, which is the default because rain is catchment-wide. Every node still sits on the same calibrated base when it is dry; only the size of the rain-time raise differs. | `null` |

### The modes

- **`off`**, `tau = base` everywhere. The identity case.
- **`step`**, `tau = multiplier * base` while the lookback-window rain total
  exceeds `wet_mm`, `base` otherwise. A hard edge in both directions.
- **`decay`**, `tau = base * (1 + (m-1) * clip(1 - h/decay_hours, 0, 1))`,
  where `h` is hours since the last wet reading. Full raise while it is
  raining, then a smooth linear return to normal.
- **`deployed`**, what
  ``anomaly_detector._rain_multipliers`` actually
  does in production today: hold the full multiplier while the lookback total is
  wet, then taper over `decay_hours` measured from the moment the lookback
  window closes.

**`step` is not what production runs.** The brief described the deployed rule as
a pure step; it is not. Production already holds the multiplier through the
lookback window and then tapers. `step` is kept because it is the clean baseline
the brief asks to compare against, and `deployed` was added so the comparison is
against the code rather than against a description of it. There is also a *third*
rain formula in this codebase, `_rain_adjusted_thresholds` in
`tests/test_anomaly_detection.py` scales the multiplier by how much rain fell,
between `RAIN_AMOUNT_THRESHOLD` and `RAIN_SATURATION_MM`. The production source
already carries a note that the two disagree. That reconciliation is not in
scope here and is not done.

One subtlety worth knowing: `step` and `deployed` ask whether the lookback
**total** clears `wet_mm`; `decay` measures from the last single **reading**
above `wet_mm`. Each rule's own definition calls for that reading, but it means
`wet_mm` is a per-window total in one mode and a per-reading floor in the other.

## Making alerts more or less sensitive during rain

| you want | change |
|---|---|
| fewer alerts during rain | raise `multiplier` |
| fewer alerts in the hours *after* rain | raise `decay_hours` |
| more alerts during rain, keep the gate | lower `multiplier` toward `1.0` |
| the gate to ignore drizzle | raise `wet_mm` |
| the gate to react to light rain | lower `wet_mm` |
| one site suppressed harder than the rest | `per_node: {that_node: 3.0}` |
| the gate off entirely | `mode: off` |

`multiplier` and `decay_hours` are the two real dials. `wet_mm` decides *when*
the gate engages at all; the other two decide *how hard* and *for how long*.

```yaml
# Rain gate for the Riffle Darner alerting path. Optional: mode "off" disables it.
base_threshold: 39.13047989749433   # POT z_q at q=1e-4. Comes from calibration.
mode: decay                         # off | step | decay | deployed
multiplier: 2.0                     # higher = fewer alerts during rain
decay_hours: 36.0                   # longer = suppression persists further past rain
wet_mm: 0.1                         # higher = more rain needed before suppressing
lookback_hours: 12.0                # step/deployed only: window tested against wet_mm
sample_hours: 0.25                  # spacing of the rain array when no timestamps given
per_node: null                      # or {node_7: 3.0} for a site that runs off harder
```

## Rain comes from the caller

`thresholds(rain_mm, timestamps=None, nodes=None)` takes a plain array of
per-timestep rain amounts. This module never fetches weather, never reads a
config file of its own and has no opinion about where the numbers came from.
Whatever rain signal the web application already has is what drives it.

## The audit trail

`decisions(scores, pvalues, rain_mm, ...)` returns one record per timestep per
node carrying the raw score, the p-value, the base threshold `z_q`, the applied
multiplier, the resulting threshold, the hours since wet, and whether it fired.
`hours_since_wet(rain_mm)` is exposed on its own so an operator can read *why*
the bar was up at a given moment without re-deriving the rule.

```
score=58.70  p=8.40e-10  z_q=39.13  x2.00 -> tau=78.26  h_since_wet=0.0  fired=False
```

A NaN score records as `scorable: False, fired: False`. "Not scorable" is not
the same as "did not exceed", and must not read as a quiet all-clear.

## The measured cost

Riffle Darner, seed 20260806, combined Fisher at `q=1e-4`, `z_q = 39.130`. A synthetic
hourly rain series is laid over the 275 stored fault-sweep cases so that 30% of
them fall in a raised-threshold window. **The sweep cases have no real time
axis, so that assignment is arbitrary by construction.** These numbers show the
shape of the trade; they are not a measurement of this creek.

Detection rate per fault shape, `multiplier=2.0`, `decay_hours=36`,
`lookback_hours=12`, `wet_mm=0.1`:

| shape | n | % wet | off | step | decay | decay − off |
|---|---|---|---|---|---|---|
| decouple | 17 | 0% | 100.0% | 100.0% | 100.0% | +0.0% |
| drift | 51 | 73% | 100.0% | 100.0% | 100.0% | +0.0% |
| partial | 119 | 35% | 98.3% | 88.2% | 88.2% | −10.1% |
| slow_all | 3 | 0% | 100.0% | 100.0% | 100.0% | +0.0% |
| spike | 51 | 0% | 100.0% | 100.0% | 100.0% | +0.0% |
| stale | 17 | 24% | 100.0% | 100.0% | 100.0% | +0.0% |
| stuck | 17 | 0% | 100.0% | 100.0% | 100.0% | +0.0% |
| **all** | **275** | **30%** | **99.3%** | **94.9%** | **94.9%** | **−4.4%** |

The whole cost lands on `partial`, and it is not small: 10 points of detection
on the one fault shape whose signal is weakest. Every other shape clears `2 ×
z_q` as easily as it cleared `z_q`, so doubling the bar costs nothing there. A
gate tuned for rain is therefore, in practice, a gate that trades away partial
faults specifically. That is worth knowing before choosing a multiplier.

False alarms on 3536 fault-free windows × 17 nodes:

| mode | detection (all shapes) | false alarms | detections lost | alarms removed |
|---|---|---|---|---|
| off | 99.3% | 2.394% | 0.0% | 0.000% |
| step | 94.9% | 2.249% | 4.4% | 0.145% |
| decay | 94.9% | 2.128% | 4.4% | 0.266% |

Restricted to the wet windows, where the gate actually does something, the
alarm rate falls from 4.86% (off) to 2.46% (step) to 0.44% (decay).

Two caveats on that table. The fault-free windows are the regime-swept ones
from the calibration corpus, which span noise and diurnal regimes the nulls were not
fitted on, that is why the ungated rate is 2.4% and not the nominal 1e-4. The
regime shift causes that, not the gate; only the comparison between the three
rows is meaningful. And `step` and `decay` come out identical on detection here
because of how the arbitrary case-to-time mapping fell, not because the two
rules are equivalent, item 8 below is where they visibly differ.

**No operating point is recommended.** This is the trade; the choice is the
team's.

## The delayed first flush

This is the failure the old step design documents. Rain falls for six hours; a
conductivity pulse arrives 21 hours after it stops, outside the 12-hour
lookback, inside the 36-hour decay.

| pulse | off | step | decay | deployed |
|---|---|---|---|---|
| 1.10 × z_q | fires | fires | suppressed | suppressed |
| 1.25 × z_q | fires | fires | suppressed | suppressed |
| 1.50 × z_q | fires | fires | **fires** | suppressed |
| 1.75 × z_q | fires | fires | fires | suppressed |
| 1.90 × z_q | fires | fires | fires | **fires** |

At the pulse, `step` is back at `1.00 × z_q`, its threshold dropped the moment
the lookback window closed, so it flags a delayed first flush at any magnitude.
`decay` holds `1.42 × z_q` and suppresses anything below that. `deployed` holds
`1.75 × z_q` and suppresses more.

Neither is asserted correct, and the tests do not assert one. A rain-driven
first flush is a real creek response and suppressing it is right; a genuine
fault landing in the same window is missed. `decay_hours` is the dial, and this
table is what moving it trades.

## Running the tests

```
.venv/bin/python -m pytest tests/test_rain_gate.py -s -q
```

`-s` matters: items 8, 9 and 10 are measurements printed to stdout, not
assertions. Nothing loads a model, runs a forward pass or trains anything ,
the scores were computed once by the calibration run and are replayed at
different thresholds, which is the only reason a threshold study is cheap.


## Where the measured numbers came from

The detection-cost and false-alarm tables above were measured against the
comparison harness corpus, which has since been deleted along with the rest of
the prototype scratch. They cannot be recomputed from this repository. They are
kept here because they are the evidence behind the design, and because anyone
changing `multiplier` or `decay_hours` needs to know what the last measurement
said before they move it.

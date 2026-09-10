# Future work

Things worth doing that are not in flight yet. Each one says what we know, so
nobody has to reconstruct it later.

## The 2026-09-03 storm, and rain coverage

Thursday 2026-09-03. At 21:57 north_fork_0 fired an alert and south_fork_1
fired as well. It was raining a good amount at the time.

This is the best rain case we have. It has a wall-clock time, a known set of
nodes that fired, and rain we can corroborate, which is more than either of the
fixtures it replaces had. Both of those, `anomaly_2025_11_13_rain_nf1` and
`anomaly_2026_04_01_rainfall`, came out of `tests/events.yaml` on 2026-09-06.
The fixtures are still on disk.

**This leaves us with no true negatives at all.** The two April rainfall rows
were the only ones, so `test_true_negative_not_flagged` currently parametrizes
over an empty list and tests nothing. `test_no_true_negative_coverage_yet` in
`tests/test_event_catalog.py` holds that hole open until it is filled. Filling
it is the point of this section.

The blocker is data. `data/raw_data/` stops at 2026-05-08 and has zero rows for
2026-09-03 at any site, so the readings have to be pulled from the live tables
before a fixture can be cut:

    python scripts/rebuild_fixture.py --folder anomaly_2026_09_03_storm \
        --site north_fork_0 --output data/anomalies/anomaly_2026_09_03_storm

then the same for south_fork_1, then one entry per site in `tests/events.yaml`.
No test code changes beyond deleting `test_no_true_negative_coverage_yet` and
putting the rows back in the graded set.

**The label is an open question, and it is the interesting part.** Rain plus a
firing detector is either a false positive we should be suppressing, which makes
it a `true_negative`, or a real excursion the rain happened to coincide with,
which makes it an `anomaly`. Two nodes firing rather than all of them is
evidence for the second reading: a rain-driven false alarm should light up
everything that got wet. Decide this from the traces before labelling, and write
the reasoning into the entry's `note` rather than just picking one.

## Why only two nodes fired

Rain fell across the catchment but only north_fork_0 and south_fork_1 responded.
Worth understanding rather than filing away.

Candidates, none checked yet: the other nodes were offline or stale for that
window; the rain gate suppressed them and not these two; the two that fired sit
below different impervious cover or a different outfall, so first flush reaches
them harder; or the per-node thresholds are simply calibrated tighter at those
two. `node_thresholds` in the calibration artifact would settle the last one
quickly.

If it turns out to be catchment rather than calibration, that is an argument for
rain being a per-node effect rather than a single global gate, which is not how
`rain_gate` is built today.

## Rain data sources

Most weather sources did not pick this storm up. The LBNL Berkeley tower did.

That matters because the rain gate is only as good as the series feeding it, and
we currently pull Open-Meteo in `serving`, ERA5 into `data/rain_cache_era5/`,
and the Night Heron daemon fetches WeatherAPI per rule. A storm that a gridded
product misses entirely is one the gate cannot react to, and on 2026-09-03 that
is what happened to most of them.

### Getting at the tower

Checked on 2026-09-06, both routes:

- **NWS**, `api.weather.gov/stations/LBNL1`, is what `ingest/weather_client.py`
  already uses. It resolves, but `precipitationLast3Hours` came back null on all
  200 observations and there is no shortwave. It serves temperature, dewpoint,
  wind, pressure, humidity, and that is all. The rolling window is also short
  enough that 2026-09-03 had already aged out by the 6th.
- **Synoptic**, formerly MesoWest, is where LBNL1 actually archives: 15-minute
  data back to 2011, which is the shape we want. It answers `401 Missing token`.
  Access is free for academic research through their Open Access program and a
  berkeley.edu address qualifies. Somebody needs to register and put the token
  in `.env`.

Once that token exists the swap is small. `weather_frame` in
`scripts/train_cobble_shoal_real.py` already takes a `cache_dir`, so an LBNL
adapter writing the same columns into a parallel cache drops in beside the
Open-Meteo one. It would be a hybrid rather than a replacement: LBNL has no
shortwave, so that channel still has to come from Open-Meteo.

### How much this actually matters, measured

`scripts/rain_sensitivity.py` answers the question underneath the swap without
needing the token: hold the weights and windows fixed, change only the three
rain channels, and see what moves. Run on both weather checkpoints on
2026-09-06:

| event | rain mm | Open-Meteo | rain zeroed |
|---|---|---|---|
| apr26_rainfall | 10.4 | 36.5, 2/456 anchors | 69.1, **27**/456 |
| nov25_rain | 18.5 | 34.3, 0/360 | 70.7, **39**/360 |
| mar26_hydrant | 0.0 | 52.1, 17/104 | 52.1, 17/104 |
| sep25_overnight | 0.0 | 59.9, 8/360 | 59.9, 9/360 |

Three things, and they hold on `weather_blocked` too, harder (apr26 goes 6 to 75
anchors there, nov25 goes 49 to 186):

1. A missed storm turns the detector inside out. Zeroing the rain roughly
   doubles the peak and turns near-silence into sustained firing. That is the
   2026-09-03 case exactly, and it is the argument for chasing the token.
2. Dry events do not move at all. mar26_hydrant is identical across variants, so
   a better rain source costs nothing on non-rain detection.
3. The sensitivity is asymmetric. Doubling the rain barely registers, zeroing it
   is a cliff, which is the log1p on the rain channels. A source that
   over-reports is close to harmless; one that under-reports is dangerous. For a
   real gauge that sees local cells, that asymmetry is in our favour.

This says the rain source matters. It does not say LBNL beats Open-Meteo, which
still needs the token and a 2026-09-03 creek fixture.

### The training corpus has the same problem

A storm only one source sees is also a warning about what the model learned
from. Cobble Shoal's weather channels came from Open-Meteo's historical forecast
API. If that product misses local storms, the weather context in training missed
them too, which is one candidate explanation for why the clock variant beat the
weather variant on both splits.

## Cobble Shoal scores nodes on unequal footing

Measured 2026-09-06 over 400 fault-free windows, shipped weights against
`cobble_shoal_calibration_real.json`.

`combine(rule="fisher")` sums `-2*log(p)` with `nansum`, so a node's degrees of
freedom is however many of its four channels are live that step. That is not
constant. Mean live channels per node:

| node | mean live channels | own p99 fisher |
|---|---|---|
| north_fork_0.* | 3.92 | ~21 |
| south_fork_2.*, oxford.* | ~3.9 | ~18-28 |
| south_fork_1.* | 3.33 | ~18-25 |
| footbridge.conductivity/depth_dev/temperature | 0.58 | ~6-9 |
| footbridge.do_pct, float_cond | 0.12 | ~7-10 |

The statistic scales with dof, so the global threshold of about 33 is set by the
well observed nodes and footbridge cannot reach it. Its own 99.9th percentile is
8.0 at depth_dev. **footbridge is structurally unflaggable**, not merely quiet.

This is worth holding next to the `events.yaml` notes on `nov25_foam` and
`nov25_rain`, both of which say footbridge is the labelled site but the target
falls back to north_fork_0 because its sensor is broken for the window. Some of
that is a broken sensor. Some of it is this.

The pooled nulls have the same shape of problem one level down. At the pooled
99.9th percentile the per-node fault-free rate runs 0 to 1.3e-2 against a nominal
1e-3, worst at `oxford.depth_dev` on dispersion. The `loo` channel's median is
about 9.5 at temperature, 6 at conductivity and 3 at depth_dev, so most of that
spread is per-variable rather than per-site.

### What was tried and did not work

All four measured against the shipped rule on the same fault-free windows, each
at a matched 1e-3 fault-free rate, counting anchors over threshold. None of them
detected an event the shipped rule missed, and all but one fired substantially
more during rain:

| variant | anomalies detected | false-alarm steps on rain events |
|---|---|---|
| shipped: pooled nulls, raw Fisher | 1/4 | 10 |
| Cauchy combination (ACAT, Liu and Xie 2020) | 1/4 | 26 |
| per-variable nulls | 1/4 | 19 |
| per-node channel nulls | 1/4 | 12, but half the detection margin |
| dof-standardised Fisher, `(stat - 2k) / 2*sqrt(k)` | 1/4 | 31 |
| per-node thresholds on the combined statistic | 1/4 | 29 |

The pattern is consistent and worth understanding before anyone tries again:
every fix that lets a sparsely observed node reach the threshold also promotes
the sparsely observed nodes during rain, because those are the same nodes. Rain
false alarms and footbridge's unreachability are the same knob viewed from two
ends, and none of these variants separates them.

Cauchy is the clearest case. It is dominated by the smallest p, so a single
spurious channel carries it, where Fisher's demand for corroboration across
channels is what suppresses rain. That is an argument that the shipped choice is
right for this problem rather than merely conventional.

Per-node thresholds do equalise the fault-free rate exactly, 2.5e-3 at every
node against a global range of 0 to 5e-3. If footbridge coverage becomes the
priority, that is the variant to revisit, paired with something that separates
rain from sparse observation rather than treating them as one.

### Where to look instead

Detection is 1/4 at a 1e-3 fault-free rate, and `jun25_spill`, `mar26_hydrant`
and `aug25_sprinklers` all score zero anchors over threshold. The scoring rule is
not what is costing those. Three of the catalog notes say the same thing in
different words: the peak clears the bar on one or two steps where flagging needs
three. That points at temporal aggregation of evidence, accumulating a weak but
sustained signal rather than counting threshold crossings, which is untried here.

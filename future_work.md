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

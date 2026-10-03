# Running StrawberryWatch's models inside Night Heron

Night Heron is the Django site and alert daemon at strawberrycreek.org. This
package is everything it needs from StrawberryWatch. Their repository holds one
call in `email_alerts.py` and nothing else, and that call does not change when a
model is added.

## What their daemon does

Once per cycle it calls `gnn_alerts.pending_alerts()` and hands whatever comes
back to the `fire_alerts_task` it already had. That function sends the email and
the SMS and writes the AlertEvent row, same as it does for a static or moving
threshold, so anomalies we fire land in the same audit trail as everything else.

Anomalies we suppress do not. Their rules write a row when rain pauses one; our
cooldown drops it with nothing but a log line, so the AlertEvent table
undercounts what the models actually saw. Worth closing, but it changes the shape
of what `pending_alerts` returns and therefore their call site too, so it is not
part of this integration.

Every model named in `GNN_MODELS` runs each pass. They share one read of each
table and one weather fetch, and one model failing to load or to score is logged
and skipped without costing the others their pass.

The `alert_type` on each dict says which model and which rule fired:

    gnn_anomaly_forecast_residual    Dusk Crayfish, Rule 1
    gnn_anomaly_level_shift          Dusk Crayfish, Rule 2
    cobble_shoal_combined_fisher     Cobble Shoal

Dusk Crayfish keeps the `gnn_anomaly_` prefix it shipped with, because their
AlertEvent table already holds rows under it. Every model after it goes out
under its own name. A site can trip more than one of these in a pass, and their
email carries no alert type, so two alerts about one site read the same in an
inbox. The AlertEvent rows are what tell them apart.

Everything else happens here: loading the models, pulling readings, fetching
weather, building windows, running the rules, deciding when to score, and
deciding when something has already been reported recently enough to stay quiet.

## Setup

Install this package into the environment their daemon runs in.

    pip install -e /path/to/SCMG_AnDeSys

Copy a checkpoint folder somewhere that environment can read, holding the files
each model in `GNN_MODELS` lists in its `FILES`, and nothing else:

    dusk_crayfish   dusk_crayfish_weights.pt, dusk_crayfish_serving.json
    cobble_shoal    cobble_shoal_weights.pt, cobble_shoal_calibration_real.json

Dusk Crayfish's JSON is produced by `strawberrywatch.serving.export_sidecar` and
holds the feature list, the node order, the normalization statistics, and the
per site thresholds. Ship it instead of the `.pkl`, because the pickle contains a
live scikit-learn object and unpickling it means matching our scikit-learn
version and running our code inside their process. Cobble Shoal's is the REAL
calibration, never the synthetic one, which scores on a different scale and
carries no scaler for live data.

## Environment

    GNN_ENABLED               default 1. Set to 0 to leave every model switched
                              off without uninstalling anything.
    GNN_MODELS                default dusk_crayfish. Comma separated, any of the
                              names in strawberrywatch/serving/registry.py.
                              GNN_MODEL_NAME, the old single model setting, still
                              counts when this is unset.
    GNN_CHECKPOINT_DIR        where the weights live. Required off a checkout.
    GNN_ALERT_EMAILS          comma separated, who hears about anomalies
    GNN_ALERT_PHONES          comma separated, optional
    GNN_SCORE_INTERVAL_MINUTES   default 15, how often to actually run the models
    GNN_ALERT_COOLDOWN_HOURS     default 6, quiet period per model, site, sensor
                                 and rule
    GNN_LOOKBACK_DAYS            default 2, how much history to pull

A number we cannot parse is logged and replaced with the default rather than
raised, and an unknown model name is logged when the worker tries to load it
rather than at import. Their daemon imports us inside a try/except ImportError,
and anything else raised at import would go straight past that and stop the
whole thing starting, threshold alerts and all, over a typo in a .env file.

Reading the creek tables uses the same `MYSQL_*` variables their daemon already
has. We only ever read.

## Adding a model

A detector in `strawberrywatch/serving/` that speaks `contract.py` (Night Heron's
raw site tables in, a list of `Finding` out), one line in `registry.py`, and a
pass from `tests/test_serving_conformance.py`. Then add its name to `GNN_MODELS`
and its files to the checkpoint folder. Nothing in this package changes, and
nothing in theirs does either.

## Things worth knowing

`pending_alerts` never raises. Their daemon has been running for months and
should not start crashing because a model had a bad day, so anything that goes
wrong is logged here and comes back as an empty list.

Weather is fetched from Open Meteo rather than read from their database, because
they fetch weather per alert rule from WeatherAPI and never store it. If that
fetch fails we score without it, which leaves both models' rain adjustment
switched off. That is the trigger happy direction, so it gets logged.

Dusk Crayfish's checkpoint sidecar records the architecture it was trained
with, so serving builds the model from the JSON rather than falling back to
`settings.yaml`. It does not cover everything: the rain tuning
(`RAIN_WINDOW_HOURS` and friends) still comes off `settings.yaml` through
`Config`, because the runner passes no `rain_params`. Retuning rain in our repo
therefore changes how their alerts behave after the next reinstall. That is a
calibration decision rather than a bug, so it is left as it is, but it is worth
knowing before anyone edits that block.

Dusk Crayfish's window is 24 rows, not 24 hours. At their fifteen minute cadence
that is six hours of creek. If a site ever starts reporting every five minutes,
the same 24 rows becomes two hours and the model is being asked about dynamics
it never saw. `DuskCrayfishDetector.expected_cadence` tells you what a window
currently covers.

Dusk Crayfish judges its whole lookback, so a flag from yesterday afternoon can
alert again once its cooldown runs out. Cobble Shoal only scores the steps within
two hours of the newest reading, so each crossing alerts once.

Cobble Shoal judges a step only once at least half its nodes have a reading
there, the bar every step its nulls were fitted on cleared. With two of the four
main sites down it goes quiet and says so in the log, rather than being held
against a null fitted on a fuller creek. One step over the bar is an alert:
`rain_gate.fired` is the whole rule, and the gate runs on its own defaults
(decay, a 2x raise, 36 hours back to normal). Which operating point to run is a
call for whoever owns the alerting budget, see `anomalies/rain_gate.py`.

Cobble Shoal is scored on readings exactly as they arrive. Its calibration was
fitted without dropping readings no creek produces, so this does not drop them
either, and a probe that logs something like the 332 C north_fork_0 reported in
April 2026 will alert.

Do not expect a footbridge alert from Cobble Shoal. `future_work.md` has why it
cannot reach the bar. Its readings still feed the model, and footbridge is
`scnf010` to Night Heron. `sql_client` looks under `SCNF010` and then `scnf010`,
because their fetcher writes tables lowercased while the table their sensor
signal makes keeps the site code's case.

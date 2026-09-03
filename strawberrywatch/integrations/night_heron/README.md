# Running Dusk Crayfish inside Night Heron

Night Heron is the Django site and alert daemon at strawberrycreek.org. This
package is everything it needs from StrawberryWatch. Their repository holds one
call in `email_alerts.py` and nothing else.

## What their daemon does

Once per cycle it calls `gnn_alerts.pending_alerts()` and hands whatever comes
back to the `fire_alerts_task` it already had. That function sends the email and
the SMS and writes the AlertEvent row, same as it does for a static or moving
threshold, so anomalies land in the same audit trail as everything else.

Everything else happens here: loading the model, pulling readings, fetching
weather, building windows, running the rules, deciding when to score, and
deciding when a site has already been reported recently enough to stay quiet.

## Setup

Install this package into the environment their daemon runs in.

    pip install -e /path/to/SCMG_AnDeSys

Copy a checkpoint somewhere that environment can read. It needs three files:
`dusk_crayfish_weights.pt`, `dusk_crayfish_serving.json`, and nothing else. The
JSON is produced by `strawberrywatch.serving.export_sidecar` and holds the
feature list, the node order, the normalization statistics, and the per site
thresholds. Ship it instead of the `.pkl`, because the pickle contains a live
scikit-learn object and unpickling it means matching our scikit-learn version and
running our code inside their process.

## Environment

    GNN_CHECKPOINT_DIR        where the weights live. Required off a checkout.
    GNN_ALERT_EMAILS          comma separated, who hears about anomalies
    GNN_ALERT_PHONES          comma separated, optional
    GNN_SCORE_INTERVAL_MINUTES   default 15, how often to actually run the model
    GNN_ALERT_COOLDOWN_HOURS     default 6, quiet period per site and rule
    GNN_LOOKBACK_DAYS            default 2, how much history to pull

Reading the creek tables uses the same `MYSQL_*` variables their daemon already
has. We only ever read.

## Things worth knowing

`pending_alerts` never raises. Their daemon has been running for months and
should not start crashing because our model had a bad day, so anything that goes
wrong is logged here and comes back as an empty list.

The window is 24 rows, not 24 hours. At their fifteen minute cadence that is six
hours of creek. If a site ever starts reporting every five minutes, the same 24
rows becomes two hours and the model is being asked about dynamics it never saw.
`DuskCrayfishDetector.expected_cadence` tells you what a window currently covers.

Weather is fetched from Open Meteo rather than read from their database, because
they fetch weather per alert rule from WeatherAPI and never store it. If that
fetch fails we score without it, which leaves the rain adjustment switched off.
That is the trigger happy direction, so it gets logged.

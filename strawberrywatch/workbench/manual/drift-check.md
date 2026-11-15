# Drift check

This page explains the drift check on Home.

A calibration describes the creek as it was in the data the calibration was made from. Over months, sensors age and the creek changes with the seasons. The drift check asks whether the last 30 days at each site still look like that data.

For each site and reading a model uses, we compare the last 30 days of readings with a sample of up to 5,000 readings from the calibration's data. Known events and rainy hours are left out on both sides. A pair counts as drifted when the Wasserstein distance between the two samples is more than 0.1 of the calibration sample's standard deviation. A pair needs at least 1,000 readings on each side, and pairs with fewer are skipped.

The check runs at most once a day for each model, in the background, after Home opens. Home then shows Drift found with the readings and sites that moved, No drift found, or a note that the data the calibration was made from is not on this computer.

See the drift check opens step 2 with the last 30 days, the drifted sites and readings, and the data source in use. Recalibrate opens step 1 with the model's sites and readings ticked.

> Drift is a reason to look, not a reason to recalibrate straight away. We recommend checking the drifted readings in Explore first, since a sensor problem can look like drift.

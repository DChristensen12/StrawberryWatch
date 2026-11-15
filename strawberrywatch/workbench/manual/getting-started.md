# Getting started

This page explains what the Workbench is for and the order in which we recommend using its tabs.

The Workbench runs on this computer. It reads Strawberry Creek readings from one of three data sources: (1) the Strawberry Creek website, (2) a folder of CSV files, or (3) a MySQL database. Each source is set up once in Settings. An API token or a database password is kept in this computer's keychain, never in a file.

Home shows the creek right now, any unfinished tuning, and each model with its drift check. Sensors lists every sensor and when it last sent a reading. Events lists known spills, rain, sensor problems and field visits. Model tuning has 3 steps: Pick data, Explore and Recalibrate. Past calibrations keeps every calibration saved on this computer, and the one each model shipped with.

A model changes only after a new calibration is saved in step 3. Saving does not change the shipped calibration, and the shipped calibration stays available in Past calibrations.

For a first recalibration, we recommend (1) setting up a data source in Settings, (2) checking Sensors for quiet sites, (3) reading What recalibrating does, and then (4) starting Model tuning.

> We recommend recalibrating on calm weeks with no known events, so that the new alert line (threshold) describes ordinary creek conditions.

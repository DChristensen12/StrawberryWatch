# When something looks wrong

This page explains what to check when the Workbench shows something unexpected.

If no readings load, we recommend checking (1) that the data source is set up in Settings, (2) that Test connection says Connected, and (3) that the picked dates have readings. A folder of CSV files needs one file per site, named after the site.

If a site shows Quiet on the Sensors tab, the logger likely lost its cellular signal or its battery is low. We recommend telling the field team, and leaving that site out when picking data.

If Explore shows many missing readings, stretches stuck at one value, or readings far from the rest, we recommend picking different dates before recalibrating.

If the past-events check shows Missed for a known spill after recalibrating, the new alert line (threshold) is likely too high. We recommend choosing more false alarms, or picking data with no events in it. If it shows Alerted for rain, the picked data may have left out too much rain.

If the check shows Not enough data, the readings around that event are missing from the data source.

If a model alerts more often after a new calibration, Use this one in Past calibrations switches back to the earlier calibration straight away.

If the downloaded readings look out of date, Clear downloaded readings in Settings removes them, and they download again when needed. Saved calibrations and events stay.

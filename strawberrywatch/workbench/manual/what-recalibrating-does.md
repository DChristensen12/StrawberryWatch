# What recalibrating does

Each model turns the latest readings into a score. When the score crosses the alert line (threshold), the model raises an alert. Recalibrating moves the alert line using data you pick. The model itself does not change.

The new alert line is set so that, on the picked data, alerts occur about as often as the false alarm choice allows. The picked data therefore matters. If a spill is present in it, the alert line rises, and the model becomes less likely to catch similar spills. For this reason, step 1 leaves out known events by default.

## Choosing how many false alarms are acceptable

Fewer false alarms raise the alert line, and a higher alert line may also let small, real changes go unnoticed. When unsure, we recommend (1) choosing about 1 false alarm a week, and (2) checking the past events before saving.

> Before recalibrating, we recommend looking over the picked data in Explore. An unexpected rise or drop in the readings is a reason to pick different dates.

# Events

This page explains the Events tab and how Model tuning uses events.

The table lists every known event in the creek data, including spills, rain, sensor problems and field visits. Each event has a Kind, its Sites, its Dates, whether a model should alert on it, and its Source. Shipped events come with StrawberryWatch. Events added here are kept on this computer only. The Kind list shows one kind of event at a time.

Selecting an event opens its editor below the table. Its chart shows the first ticked site, with the event's time range marked. The reading buttons change which reading the chart shows. Dragging across the chart sets where the event starts and ends, and the Starts and Ends fields change with it in 15 minute steps.

Should a model alert on this? decides how the past-events check judges a calibration. A spill or leak, or a sensor problem, defaults to Yes. Rain, a field visit and other events default to No. Save changes keeps the edit. Changes to a shipped event are saved on this computer only, and the shipped files never change. Only events added here can be deleted.

Mark a new event opens an empty editor with a Name field. An event needs at least 1 site and a start and end time.

Step 1 leaves these events out of the picked data, and step 3 checks each new calibration against them. Export event list (CSV) saves the table as a file.

> We recommend logging sensor cleanings and swaps as field visits. Readings can shift after a cleaning, and that shift should stay out of calibration data.

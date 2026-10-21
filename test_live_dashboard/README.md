# Test live dashboard

The dashboard exactly as strawberrycreek.org will show it, running on this
computer. Use it to try out buttons and looks before anything goes on the
website.

    python test_live_dashboard/run.py

Then open http://127.0.0.1:8791/strawberrywatch/

## Where to make changes

Not in this folder. It only holds the launcher. The pages come straight from
the package, the same files the website mounts:

- `strawberrywatch/dashboard/templates/` for the pages
- `strawberrywatch/dashboard/static/app.css` for the look
- `strawberrywatch/dashboard/static/app.js` for what buttons do in the browser
- `strawberrywatch/dashboard/routes.py` and `view.py` for what each page shows

Template, CSS and JS edits show on the next refresh. Python edits need a
restart (Ctrl+C, then run it again). Whatever you see here is what the website
gets with the next StrawberryWatch install.

## Options

| Option | What it does |
| --- | --- |
| `--calm` | Nothing unusual in the last 24 hours, so Overview shows the quiet day. Starts over. |
| `--paused` | Shows the "Model checks paused after 3 failed checks" banner and its Restart checks button |
| `--view-only` | Signed in as someone outside `STRAWBERRYWATCH_EDITOR_GROUP`: no forms, and posts are refused |
| `--event NAME` | Readings from a different past event (any folder in `strawberrywatch/watch/events/`). Starts over. |
| `--models A,B` | Which models are on and run, for example `--models dusk_crayfish,flame_skimmer`. The website runs at most 2 at once (`max_models_on` in `settings.yaml`). Starts over. |
| `--github STATE` | What the stand-in GitHub holds for Settings > Model updates: `newer` (the default, a recalibrated Water Strider), `same` (nothing new), `broken` (also Cobble Shoal weights that won't load) or `offline` (can't be reached) |
| `--user NAME` | The username settings changes and notes are signed with. `test_user` by default. |
| `--fresh` | Start over, dropping the notes, settings changes and email list edits made here |
| `--port N` | 8791 by default, one up from the local dashboard's 8790 |

Without `--fresh`, `--calm`, `--event` or `--models`, what you changed last time is kept.

To see Flame Skimmer flag something on Unusual readings, try
`--event anomaly_2026_03_20_hydrant_nf0 --models dusk_crayfish,flame_skimmer`.

## What's real and what's a stand-in

Real, the same as on the website:

- The website profile: one data source with no Change link, no setup page,
  no Workbench, no replays, and the email list kept in the database for Night
  Heron to send to
- Mounted at `/strawberrywatch/`, with the dashboard's cookie kept to that path
- The pages, styles, scripts and security headers
- Readings and what the models noticed, from real checks of a past event,
  run the way the website runs them (each reads the 2 days before its moment,
  with the website's `settings.yaml` limits)

Stand-ins:

- **Sign-in.** You're always signed in. On the website the site's own login
  page comes first.
- **The data.** A SQLite file in `test_live_dashboard/data/` instead of the
  `sw_` tables on the website's MySQL. Git ignores it.
- **The readings.** The September 2025 overnight spike on the south fork,
  checked every 2 hours over its last 3 days (the website checks every 15
  minutes; every 2 hours still covers every moment Cobble Shoal and Flame
  Skimmer judge), then moved so the last reading is the latest quarter hour.
  While it runs, the checks slide forward every 15 minutes, so the pages never
  go stale. Two example.org people are on the email list and sending is on,
  so the email page has something in it.
- **No new checks.** After Apply, the website runs a new check within 15
  minutes. Here none comes, so to see a model's findings, start with `--models`.
- **GitHub.** Model updates reads `test_live_dashboard/data/github`, a folder
  laid out like StrawberryWatch's main branch, never the real GitHub. Updates
  are written beside a copy of the model files in
  `test_live_dashboard/data/checkpoints`, one folder per model like the
  website's, never into the repository's `checkpoints` folder. On the website
  each update runs in a process of its own; here it runs on a thread.
- **The rest of the site.** `/` goes to the dashboard and Sign out goes to a
  one-line placeholder page.
- **http, not https.** So the cookie isn't marked Secure here.

Nothing here can reach the website's database or send an email. Adding someone
to the email list does look up their address's domain, as the website's form
does, so that needs the internet.

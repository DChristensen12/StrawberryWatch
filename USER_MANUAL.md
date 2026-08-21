# StrawberryWatch User Manual

DOCUMENT IS A WORK IN PROGRESS. COMMANDS AND ACTUAL CONTENTS NEED TO BE UPDATED AS STRAWBERRYWATCH GETS DEVELOPED AND INTEGRATED WITH THE LIVE WEBSITE REPOSITORY.

## What this is and who should read it

This explains how to run StrawberryWatch, what its parts are, and how to change
it. You don't need to have written Python before.

The system watches Strawberry Creek for contamination. Sensors in the creek
report every 15 minutes, a model learns what normal looks like, and anything
that doesn't match gets flagged. This manual covers installing it, running it,
reading what it prints, and making the small edits that come up most often.

Read it start to finish the first time. After that, the section headings are the
index.

One thing worth knowing before you begin. Nothing in this repository
sends anything to the creek or to the public website. It reads data and prints
results. You can't break the creek from here.

If a command in this manual doesn't do what the manual says it does, the
manual is wrong, and would require fixing.

---

## 1. Getting set up

### What a terminal is

A terminal is a window where you type commands instead of clicking. On macOS it
is called Terminal, on Linux it's usually Terminal or Console, on Windows use
PowerShell or Windows Terminal. When this manual shows a grey box, you type what
is inside it and press Enter.

### Python 3.12

The project needs Python 3.12 or newer. Check what you have:

```
python3 --version
```

You should see `Python 3.12.x` or higher. If you see 3.11 or lower, or "command
not found", install Python 3.12 from python.org before continuing.

### Getting the code

This downloads the repository into a folder called `StrawberryWatch`:

```
git clone https://github.com/DChristensen12/StrawberryWatch.git
```

Then move into it. Every later command assumes you are inside this folder:

```
cd StrawberryWatch
```

### The virtual environment

A virtual environment is a private copy of Python that belongs to this project
alone. Without one, installing StrawberryWatch's dependencies would change the
Python your whole computer uses, and two projects wanting different versions of
the same library would fight. The environment keeps that fight from happening.

Create one. This makes a hidden folder called `.venv`:

```
python3 -m venv .venv
```

Now activate it. This tells your terminal to use that private Python instead of
the system one:

```
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`. That's how you know it worked. On
Windows PowerShell the activate command is `.venv\Scripts\Activate.ps1` instead.

You need to run the activate command again every time you open a new terminal.
Forgetting it's the single most common cause of "it worked yesterday".

### Installing

```
pip install -e ".[dev]"
```

This reads `pyproject.toml`, installs PyTorch and the other dependencies, and
installs StrawberryWatch itself. The `[dev]` part adds ruff, pre-commit and
pytest, which you need for section 7 and section 8. It takes several minutes the
first time, mostly downloading PyTorch.

The `-e` means editable, and it's not optional. Without it, pip copies the code
into the environment, and your edits to the files in this folder then have no
effect on what actually runs. You would change a line, run the code, and see the
old behaviour, with nothing to tell you why. With `-e`, Python reads the files
where they sit, so an edit takes effect the next time you run.

Check it worked:

```
python -c "import strawberrywatch; print('ok')"
```

That should print `ok` and nothing else.

### The .env file

`.env` holds settings that should not be committed to git, mostly credentials.
Copy the example and edit your copy:

```
cp .env.example .env
```

`.env` is listed in `.gitignore`, so your copy won't be committed. Never put a
real password in `.env.example`.

**Required for a normal run:**

| Variable | What it is | What breaks without it |
|---|---|---|
| `SCMG_API_BASE_URL` | Where creek data is fetched from | Defaults to the public URL in `config.py`, so a missing value is usually fine |
| `NWS_USER_AGENT` | An identifying string sent with weather requests | Defaults to `SCMG-AnDeSys/1.0`. Put a real email in it, since it is how the weather service contacts you if something goes wrong |

**Optional:**

| Variable | What it is | What breaks without it |
|---|---|---|
| `SCMG_API_TOKEN` | API token | Nothing. The public API currently needs no token |
| `ALERT_EMAIL_SENDER`, `ALERT_EMAIL_PASSWORD`, `ALERT_EMAIL_RECEIVER`, `SMTP_SERVER`, `SMTP_PORT` | Email alerting | The run completes normally and prints `Failed to send email alert` at the end. Detection is unaffected |
| `MYSQL_HOST`, `MYSQL_DATABASE_USER`, `MYSQL_DATABASE_PASSWORD`, `MYSQL_DATABASE_NAME`, `MYSQL_PORT` | Direct database access, admin only | Only `--data-source sql` fails. The default `--data-source api` does not touch these |
| `USE_NWS_WEATHER` | Whether to merge weather into the dataset | Defaults to `true` |
| `ROLLING_WINDOW_DAYS` | How many days the local cache keeps | Defaults to 90 |
| `LOG_LEVEL` | How much detail gets logged | Defaults to `INFO` |

The email variables are the ones people most often leave blank. That's fine.
The run still works and still prints what it found.

---

## 2. Running the system

Everything runs through `main.py`. To see the options at any time:

```
python main.py --help
```

```
usage: main.py [-h] [--mode {train,update,inference}]
               [--data-source {api,sql}]
               [--model {dusk_crayfish,cobble_shoal}]
               [--support NAME [NAME ...]] [--q-total Q_TOTAL]
               [--data-file DATA_FILE]

SCMG GNN Pipeline

options:
  -h, --help            show this help message and exit
  --mode {train,update,inference}
  --data-source {api,sql}
                        Where to pull data from: REST API (default) or SQL
                        database
  --model {dusk_crayfish,cobble_shoal}
                        Which model architecture to use
  --support NAME [NAME ...]
                        Support modules to attach, space separated. Order does
                        not matter. Omit for a run with no support. Valid
                        names: newtwork_run, settling_pool, trial_bed
  --q-total Q_TOTAL     Total false alarm rate for the whole run, divided
                        across the primary and every attached detector rather
                        than handed to each. Default 0.0001.
  --data-file DATA_FILE
                        Path to the data CSV, overriding the default cache.
```

### The three modes

**`--mode inference`** pulls the last 2 days of data, loads the saved weights,
and scores. It changes no weights. This is the one you'll run most, and it's
what the monitoring loop runs.

```
python main.py --mode inference
```

**`--mode update`** pulls 30 days, continues training from the existing weights,
and saves them. It overwrites `checkpoints/*_weights.pt` and
`checkpoints/*_metadata.pkl`.

```
python main.py --mode update
```

**`--mode train`** pulls 30 days and trains from scratch, discarding the
existing weights.

```
python main.py --mode train
```

Both `update` and `train` overwrite the saved weights, and the current weights
are the baseline every historical event test is judged against. Don't run
either one because you are curious. Ask first. See section 11.

If no weights exist at all, `update` and `inference` say so and fall back to
training, so a fresh clone with an empty `checkpoints/` folder will train on its
first run.

### What a normal run looks like

Here is the beginning of a real `--mode inference` run:

```
SCMG Anomaly Detection System
Execution Mode: INFERENCE
Model:          dusk_crayfish
Device:         cpu
Data file:      /home/you/StrawberryWatch/data/processed_data/full_creek_gnn.csv
Support:        none
False alarm budget: q_total=0.0001, primary=0.0001
Refresh requested. Fetching last 2 days from API...
Requesting data: north_fork_0...
Requesting data: south_fork_2...
Requesting data: south_fork_1...
Requesting data: oxford...
Fetching Open-Meteo weather (window: 2d, source: Historical Forecast API, 15-min native)...
Weather merged: 759/759 rows have weather context (features: ['air_temp_c', 'rain_mm', 'shortwave_radiation'])
Trimmed 1,092 rows older than 90 days.
Cache updated: 32,195 rows on disk (2026-05-22 22:45:00+00:00 to 2026-08-20 22:45:00+00:00)
data loaded:
Rows: 32,195
Range: 2026-05-22 22:45:00+00:00 to 2026-08-20 22:45:00+00:00
Sites present (4): ['north_fork_0', 'oxford', 'south_fork_1', 'south_fork_2']
Active features (10): conductivity, depth, temperature, air_temp_c, rain_mm, shortwave_radiation, hour_sin, hour_cos, dayofyear_sin, dayofyear_cos
```

Reading that: it fetched four sites, merged weather onto every row, trimmed the
cache back to 90 days, and ended up with about 32,000 rows. `Device: cpu` is
normal. A graphics card would say `cuda` and would only be faster.

Then it builds the model's input and scores. There's a long progress bar during
`Pivoting data`, which is normal and takes about twenty seconds.

The last lines are the answer. A quiet creek looks like this:

```
Detection cycle finished. Nodes judged: 4/4. Nodes flagged: 0.
```

### What a run that found something looks like

```
Detection cycle finished. Nodes judged: 4/4. Nodes flagged: 3.
  north_fork_0: forecast_residual, peak_deviation=68.48, duration=4 timesteps (31 total over threshold)
  north_fork_0: level_shift, peak_deviation=4.35, duration=5 timesteps (8 total over threshold)
  south_fork_2: forecast_residual, peak_deviation=188.68, duration=3 timesteps (42 total over threshold)
  south_fork_1: forecast_residual, peak_deviation=354.83, duration=7 timesteps (55 total over threshold)
```

Reading that:

- **Nodes judged: 4/4** means all four modelled sites had enough data to score.
  A site with a dead sensor would show as judged 3/4.
- **Nodes flagged: 3** is how many sites crossed their threshold.
- **forecast_residual** means the reading differed from what the model expected
  next. **level_shift** means the reading moved to a new level and stayed there.
  A site can trip both, as north_fork_0 did here.
- **peak_deviation** is how far past the threshold it got at its worst. Bigger is
  a stronger signal. It's not a concentration and it's not in any physical
  unit.
- **duration** is the longest unbroken run of timesteps over the threshold. The
  number in brackets is the total count over the threshold anywhere in the
  window. A long unbroken run matters more than a scattered total.

A flag is not a confirmed spill. It says these readings don't match what the
creek normally does. Look at the readings, check whether it rained, and escalate
by whatever your group's procedure is.

If email is configured, an alert is sent. If it's not, you'll see this and
can ignore it:

```
Failed to send email alert: (535, b'5.7.8 Username and Password not accepted.
```

### Continuous monitoring

To run inference every 15 minutes, indefinitely:

```
python scripts/run_live.py
```

It prints a timestamped banner per cycle, runs `main.py --mode inference` as a
separate process so each cycle starts with clean memory, then sleeps 15 minutes.

**To stop it, press Ctrl and C together in that terminal.** It prints
`Monitoring stopped by user.` and exits. Closing the terminal window also stops
it. There's no background service and no PID file, so if you want it to survive
a logout you need systemd or cron, which is not set up here. See the gaps in
section 12.

---

## 3. The models

Models are named after creek biota. Each one is a different approach to the same
question. The README table is the short version, this is the longer one.

For which model is currently deployed, see the line at the top of the README. It
is date stamped. This manual doesn't name it on purpose.

### Dusk Crayfish

A graph neural network paired with an LSTM. It treats the creek as a connected
graph and predicts what the network as a whole should read next, then flags
timesteps where reality differs.

- **Good at:** events that show up as a whole site behaving oddly, and events
  that propagate downstream. It knows which site is upstream of which.
- **Can't see:** which individual sensor at a site is wrong. It scores a site,
  not a probe, so a site with a bad conductivity probe and three healthy ones
  scores as one number. It also can't run on sites outside its four-node core.
- **Runs on:** north_fork_0, south_fork_1, south_fork_2, oxford. The other seven
  sites are not modelled yet.
- **Currently runnable:** yes.

```
python main.py --model dusk_crayfish --mode inference
```

### Cobble Shoal

Also a graph neural network with an LSTM, but it scores every sensor separately.
A node is a (site, variable) pair, so it sees 17 nodes where Dusk Crayfish sees 4
sites. It learns how fast each reading goes stale and combines several
independent checks, so it can name which sensor looks wrong.

- **Good at:** naming the specific probe. It also has a check for a sensor that
  has frozen, which a forecast-based approach structurally can't see.
- **Can't see:** it has no weather input at all, by design, so it can't tell a
  rain event from a contamination event on its own.
- **Currently runnable through `main.py`: no.** It's registered as a `--model`
  choice, but running it stops here:

```
python main.py --model cobble_shoal --mode inference
```

```
ERROR: weights exist at .../checkpoints/cobble_shoal_weights.pt but metadata is
missing at .../checkpoints/cobble_shoal_metadata.pkl. Cannot determine the
trained feature set. Retrain with --mode train.
```

  Its weights exist but its metadata file doesn't, and `main.py` won't guess
  at a feature set. It can be scored outside `main.py`, where it reads the data
  through a different path:

```
python scripts/run_audit_comparison.py
```

  Wiring it into `main.py` properly is unfinished work. See section 12.

### Flame Skimmer

A Bayesian neural network. It would learn a distribution over its weights rather
than fixed values, so every forecast would carry a calibrated uncertainty
estimate.

**Not built.** `strawberrywatch/models/Flame_Skimmer.py` contains one line, a
TODO. There's nothing to run.

### Water Strider, Berry Delight, Crimson Parker

Named in the README table. Berry Delight is described there as a logistic
regression model. The other two are marked TBD.

**Not built.** No file exists for any of them.

---

## 4. The support modules

A support module attaches to any model. It has no weights of its own and is not
a model. You attach one or more with `--support`:

```
python main.py --mode inference --support trial_bed newtwork_run settling_pool
```

The run then prints what attached:

```
Support:        newtwork_run, settling_pool, trial_bed
  newtwork_run: reads firing order along the flow graph to separate transport from a catchment wide cause (explainer, no budget)
  settling_pool: flags readings against the inventory's per site thresholds and sensor state (screen, no budget)
  trial_bed: matches the direction each channel moved against the spill signature table (explainer, no budget)
False alarm budget: q_total=0.0001, primary=0.0001
```

Order doesn't matter. Omit `--support` entirely for a run with nothing attached.

### Explaining versus testing, and why it matters

Modules come in four kinds, and the difference decides whether switching one on
changes how sensitive the system is.

- An **explainer** reads a detection that already happened and says what it looks
  like. It adds no test. Switching it on can't change what fires.
- A **screen** filters readings before the model sees them. It adds no test
  either, but it does change the model's input, so the thresholds were fitted on
  unscreened data and no longer describe it exactly.
- A **detector** adds its own score, which is a second test. Two tests running at
  the same false alarm rate produce twice the false alarms, so the budget
  `--q-total` gets divided between them rather than handed to each. Attaching a
  detector makes the primary model slightly less sensitive, because it's now
  spending part of the budget.
- A **modulator** moves the threshold up or down without adding a test.

**All three modules currently in the repository are explainers or a screen. None
is a detector, so none of them costs you any false alarm budget today.** The line
`(explainer, no budget)` in the output above is how you check. If a module ever
prints a `q=` line under it, it's a detector and it's spending budget.

### The three modules

**Trial Bed** (explainer). Identifies what kind of contamination a detection
looks like. It reads which direction each measurement moved and matches that
pattern against a table of known signatures, then reports the likely cause with
the readings behind it. Costs nothing.

**Newtwork Run** (explainer). Confirms a detection travelled the way water does.
A spill reaches downstream sites after a delay set by the creek's flow and never
crosses between forks. Rain arrives everywhere at once. Comparing the timing
across sites separates the two. Costs nothing.

**Settling Pool** (screen). Screens incoming data before the model sees it.
Catches placeholder values passing as real readings, one feed silently
duplicating another, sensors re-zeroed between deployments, and gaps that were
filled rather than measured. Costs no budget, but it does change what the model
is fed.

---

## 5. The inventory

This is the file you are most likely to need to edit. It records which sensor is
at which site, when it went in, and whether it's in use right now.

It lives at `strawberrywatch/inventory.yaml`.

### The grid at the top

Open the file. The first thing you see is this:

```yaml
# ==================================================================
# WHAT IS RUNNING WHERE
#
#   yes  installed and in service
#   no   installed, switched off right now
#   -    no such probe at this site
#
# Change yes to no to take a sensor out of service, no to yes to put it back.
# A site with every sensor set to no is out of service. Lower case only:
# true, 1, On and "yes" in quotes are all refused.
#
# New probe fitted? Change - to yes here, then add its install date in that
# site's block below. Forget the date and the error tells you what to write.
#
# Check your edit:  python -m strawberrywatch.inventory
# ==================================================================
in_service:
  botanical_garden:  {ctd: yes, do: -,   fc: -,   ph: -}
  south_fork_0:      {ctd: yes, do: -,   fc: -,   ph: -}
  south_fork_1:      {ctd: yes, do: -,   fc: -,   ph: -}
  south_fork_2:      {ctd: yes, do: -,   fc: -,   ph: -}
  south_fork_3:      {ctd: yes, do: yes, fc: yes, ph: -}
  kingman_hall:      {ctd: yes, do: -,   fc: -,   ph: -}
  north_fork_0:      {ctd: yes, do: -,   fc: -,   ph: -}
  scnf010:           {ctd: yes, do: yes, fc: yes, ph: -}
  university_house:  {stage: yes, balance_feed: yes}
  oxford:            {stage: yes, balance_feed: yes}
  codornices:        {stage: yes, balance_feed: yes}
```

One row per site, one column per sensor. The column names are `ctd` (the
conductivity, temperature and depth probe, which is one instrument reporting
three channels), `do` (dissolved oxygen), `fc` (floating conductivity) and `ph`.
The three legacy Balance Hydrologics sites at the bottom have different columns,
`stage` and `balance_feed`, because they are a different kind of station.

Three values are allowed and nothing else:

- `yes` means the probe is installed and in use.
- `no` means the probe is installed but switched off right now.
- `-` means there's no such probe at that site.

Note the difference between `no` and `-`. `no` is a probe that exists and is
temporarily out. `-` is a probe that was never fitted. Getting these mixed up is
the mistake the file is designed to catch.

### Taking a sensor out of service

Say the dissolved oxygen probe at scnf010 has been pulled for cleaning. Find its
row, change `do: yes` to `do: no`, and save:

```yaml
  scnf010:           {ctd: yes, do: no,  fc: yes, ph: -}
```

That's the whole edit. Nothing else in the file changes and no code changes.
Keep the spacing so the columns stay lined up, but the file will load either way.

To put it back, change `no` to `yes`.

To take a whole site out, set every one of its sensors to `no`. There's no
separate site level switch, on purpose, because a site switch and a sensor
switch could contradict each other.

Switching a probe off removes it from live scoring. Its older readings are still
used for training, because history doesn't change when you unplug something.

### Checking your edit worked

```
python -m strawberrywatch.inventory
```

This prints the grid back to you, plus everything the grid can't show. Compare
the grid it prints against the file. The first part looks like this:

```
Strawberry Creek sensor inventory
/home/you/StrawberryWatch/strawberrywatch/inventory.yaml

WHAT IS RUNNING WHERE   (yes in service, no switched off, - no such probe)

  botanical_garden:  {ctd: yes, do: -,   fc: -,   ph: -}
  south_fork_0:      {ctd: yes, do: -,   fc: -,   ph: -}
  ...

SWITCHED OFF
every probe in the grid is in service
```

If you switched something off, the SWITCHED OFF section names it instead.

Further down it prints install dates, which dates were guessed rather than
known, any rows dropped from the archive, and a FILE AGAINST DATA section that
names disagreements between what the file claims and what the data shows. A site
marked in service that has not reported in two hours gets named there, and so
does a site marked out of service that's still sending readings. It never
changes the file for you.

### The error you'll get if you write `true`

`yes` and `no` are the only spellings accepted. `true`, `True`, `1`, `on` and
`"yes"` in quotes are all rejected. That's deliberate. YAML treats several of
those as the same thing without telling you, and a typo that silently switches a
sensor off is worse than an error.

If you write `ctd: true` and run the check, you get this:

```
The inventory file has a problem and nothing will run until it is fixed.

inventory.yaml line 20: the ctd switch for south_fork_1 is set to 'true', which is not yes, no or -

Write one of these three, in lower case, with no quotes:
    ctd: yes   installed and in service
    ctd: no    installed, switched off right now
    ctd: -     there is no ctd probe at south_fork_1
```

**What to do:** go to the line number it names, in the file it names, and write
one of the three spellings it lists. That's all. Every validation error in this
file follows the same shape: where the problem is, what is wrong, and what to
write instead.

Nothing runs until you fix it, on purpose. A run that dropped a site because of
a typo and said nothing is worse than a run that refuses to start.

### When a new probe is installed

A new probe needs two edits, because the grid can't hold a date.

First, change that sensor's `-` to `yes` in the grid. Then add its install date
in the site's detail block further down the file. If you forget the second step,
the error tells you exactly what to write:

```
the grid says upstream has a do probe, but no install date is written for it below

Add the date the probe went in, under sites: upstream: sensors: do:
      do:
        install: '2026-03-05'
        removed: null
Use the real date. If there is no do probe here, set the
do column of the upstream row back to -.
```

Use the real install date, not today's date, and not a guess. The date decides
which past readings are treated as real, so a wrong one either invents years of
missing data or hides real gaps.

### Correcting a guessed install date

Many install dates in the file were guessed from the first reading in the
archive because nobody recorded the real one. They are marked:

```yaml
      ctd:
        # GUESSED off the first reading. Replace when you know the real date.
        install: '2025-12-04'
        removed: null
        inferred: true
```

If you know the real date, write it in and delete the `inferred: true` line and
the comment above it. The check command lists every guessed date under GUESSED
INSTALL DATES, so you can see what still needs an answer.

The file marks one entry as the most likely wrong, at the top of scnf010's block.
Fix that one first if you can find out the answer.

---

## 6. Making a change to the code

### Setting up the checks

If you installed with `".[dev]"` in section 1, you already have the tools. Now
install the git hooks:

```
pre-commit install
```

This makes git run the checks automatically every time you commit. You only do
this once per clone.

### What the two tools do

**`ruff format`** rewrites your code into the project's layout. Indentation,
where lines break, spaces around commas. It never changes what the code does,
only how it looks. This is so that a change in git history is a real change and
not somebody's editor reformatting a file.

```
ruff format .
```

**`ruff check`** looks for actual problems. Unused imports, a variable assigned
and never read, an import block in the wrong order. Some of these are harmless
and some are bugs.

```
ruff check .
```

When both are happy:

```
All checks passed!
84 files already formatted
```

`ruff check` can fix many of its own findings:

```
ruff check --fix .
```

### Why a commit gets rejected

If either tool fails, the commit is stopped. The reason is that everyone reads
this code, it's handed to new people every year, and code that's formatted
five different ways is harder to read than code that's formatted one way. The
check is not a judgement of your work.

### When pre-commit changes your files and aborts the commit

This surprises everybody the first time. You commit, and you see something like:

```
ruff-format..............................................................Failed
- hook id: ruff-format
- files were modified by this hook
```

and the commit didn't happen.

**This is not an error and nothing is lost.** The formatter found something to
tidy, tidied it, and stopped so you can see what it did. Your changes are still
there, plus the formatting fixes.

The fix is to add the files again and commit again:

```
git add -A
```

```
git commit -m "your message"
```

The second time it passes, because the files are already formatted. If it fails
again with a different hook, read what that hook said. `trailing-whitespace` and
`end-of-file-fixer` behave the same way: they fix it, then stop.

`ruff check` is different. It doesn't usually fix things for you, so a failure
there's something you need to read and address.

### Skipping the hooks

You can commit without running the hooks:

```
git commit --no-verify -m "your message"
```

Reasonable uses: you are committing work in progress on a branch nobody else
reads, or a hook is broken and blocking you and you have said so to someone.

Not reasonable: the check is failing and you don't want to deal with it. It will
fail in CI instead, where it's more annoying and more public. The badge at the
top of the README goes red.

---

## 7. Running the tests

The full suite:

```
python -m pytest tests/ -q
```

It takes about seven minutes, most of it in the historical event tests, which
load models and score real data.

The historical event suite on its own:

```
python -m pytest tests/test_anomaly_detection.py -q
```

About six minutes.

A single fast file, useful while editing:

```
python -m pytest tests/test_inventory.py -q
```

```
.......................                                                  [100%]
23 passed in 0.46s
```

### The three known failures

The full suite currently ends like this:

```
FAILED tests/test_anomaly_detection.py::test_anomaly_detected[dusk_crayfish-sep25_overnight/anomaly_2025_09_10_overnight_sf1]
FAILED tests/test_anomaly_detection.py::test_anomaly_detected[dusk_crayfish-mar26_hydrant/anomaly_2026_03_20_hydrant_nf0]
FAILED tests/test_anomaly_detection.py::test_true_negative_not_flagged[dusk_crayfish-apr26_rainfall/anomaly_2026_04_01_rainfall0]
3 failed, 344 passed, 7 warnings in 400.96s (0:06:40)
```

**These three are expected. You didn't break them.** They are:

| Test | Event | What it means |
|---|---|---|
| `sep25_overnight` at south_fork_2 | September 2025 overnight conductivity spike | The model should flag it and does not |
| `mar26_hydrant` at north_fork_0 | March 2026 fire hydrant spill | The model should flag it and does not |
| `apr26_rainfall` at north_fork_0 | April 2026 heavy rain | The model should stay quiet and does not |

They are open problems with the detection, recorded as failing tests rather than
hidden. If you see exactly these three, the suite is in its expected state. If
you see a different one, that's worth investigating.

Before asking anyone about a test failure, check it against this list.

### Two further failures, at the time of writing

As this manual was written, two more tests were also failing:

```
FAILED tests/test_inventory_editing.py::test_the_instructions_sit_above_the_grid
FAILED tests/test_inventory_editing.py::test_the_comments_stay_short_enough_not_to_intimidate
```

These are not detection problems. Both check the wording and length of the
comment block at the top of `inventory.yaml`, and that block was rewritten after
the tests were written, so the tests describe a version of the file that's no
longer there. Either the comments or the tests need to change, and which one is
a decision for the project lead rather than something to guess at. See section
12.

---

## 8. Where things live

```
StrawberryWatch/
  main.py                    the entry point, all modes run through here
  USER_MANUAL.md             this file
  README.md                  overview, and the deployed model line
  pyproject.toml             dependencies and tool settings
  .env                       your credentials, never committed
  .pre-commit-config.yaml    which checks run on commit

  strawberrywatch/           all the library code
    config.py                settings read from .env and settings.yaml
    paths.py                 every file location, resolved in one place
    inventory.yaml           which sensor is where, section 5
    inventory.py             reads and validates the inventory
    ingest/                  fetching data: API, SQL, weather, raw files
    preprocessing/           turning readings into model input
    models/                  model architectures, one file each
    anomalies/              scoring, thresholds, rain handling
    support_modules/         the attachable modules from section 4
    training/                the training loop
    utils/                   graph helpers, plotting, email

  tests/                     the test suite
  scripts/                   one-off and operational tools
  data/
    raw_data/                the sensor archive, committed
    anomalies/               labelled historical events, committed
    processed_data/          generated caches, NOT committed
  checkpoints/               trained weights, NOT committed
  documents/                 written reports and audits
  assets/                    images used by the README
  notebooks/                 exploratory notebooks
```

### Where to put a new file

- A new model goes in `strawberrywatch/models/`, one file per model, and gets
  added to `_MODEL_REGISTRY` at the top of `main.py`. Nothing else in `main.py`
  needs to change.
- A new support module goes in `strawberrywatch/support_modules/` and gets added
  to `SUPPORT_REGISTRY` in `registry.py`.
- A test goes in `tests/`, named `test_*.py` or pytest won't find it.
- A one-off analysis goes in `scripts/`.

### Where not to put things

- **Don't put file paths in your code.** Every location lives in
  `strawberrywatch/paths.py`. Import from there so the code works no matter which
  directory it's run from.
- **Don't put generated files in `data/raw_data/`.** That folder is the archive
  and is committed. Generated things go in `data/processed_data/`, which is
  ignored by git.
- **Don't commit weights, `.csv` caches, or `.env`.** `.gitignore` already
  excludes them. If git offers to commit one, something is wrong.
- **Don't put test helpers in `strawberrywatch/`** unless the library actually
  uses them.

---

## 9. Common problems

### `ModuleNotFoundError: No module named 'strawberrywatch'`

Any command dies immediately with that message. Either the virtual environment
isn't active, or the editable install never happened. Activate first:

```
source .venv/bin/activate
```

If your prompt already shows `(.venv)`, reinstall:

```
pip install -e ".[dev]"
```

### Your edit to a file has no effect

You change a line, run the code, and the old behaviour happens anyway. The
package went in without `-e`, so Python is reading a copy of it somewhere else.
Reinstall:

```
pip install -e ".[dev]"
```

### `Failed to send email alert`

The run finishes, prints its results, then prints an SMTP error about a
username and password. That's the `ALERT_EMAIL_*` variables in `.env` being
blank or wrong.

You can ignore it if you don't need email. Detection already finished and
everything above that line is valid. If you do want email, fill in the five
`ALERT_EMAIL_*` and `SMTP_*` variables. Gmail wants an app password, not your
normal one.

### The weather service is unreachable

The run stalls or errors around `Fetching Open-Meteo weather (window: 2d, ...)`.
Either you have no internet or Open-Meteo is down. Check:

```
python -c "import requests; print(requests.get('https://api.open-meteo.com/v1/forecast', timeout=10).status_code)"
```

`200` means it's reachable. If it's not, wait, or set `USE_NWS_WEATHER=false`
in `.env` to run on the creek's own sensors alone. Results will differ, because
the model normally uses rain as an input.

### The results look stale, or the cache is wrong

The row count and date range don't match what you expect, or a change to the
data isn't showing up. The local cache at `data/processed_data/full_creek_gnn.csv`
is carrying old rows. Delete it and let the next run rebuild:

```
python scripts/clear_cache.py
```

The next run downloads fresh data. It's safe: the cache is generated, never the
source of truth, and is not committed.

### The inventory refuses to load

You get `The inventory file has a problem and nothing will run until it's
fixed.` Read the message. It names the file, the line number, and what to write.
See section 5.

### A test fails

Check them against the three known failures in section 7 first. If they're
those three, nothing is wrong.

---

## 10. Who to ask

*This section needs a person to fill in. The repository doesn't record who is
responsible for what. `CONTRIBUTORS.md` exists but is empty.*

| Topic | Who | Contact |
|---|---|---|
| Running the system day to day | | |
| The models and retraining | | |
| Sensors, field work, install dates | | |
| EH&S escalation when something is flagged | | |
| Repository access and permissions | | |

Before asking about a failing test, check section 7. Before asking about a
flagged detection, check what the readings actually did and whether it rained.

---

## 11. What this manual doesn't yet cover

This is a first draft. These gaps are known, and each one needs information that
is not in the repository.

1. **Who to ask.** Section 10 is an empty table. **Needed from:** the project
   lead, the current names and contacts.

2. **What to do when a real detection fires.** This manual explains how to read
   the output but not what the group's escalation procedure is, who gets called,
   or in what timeframe. **Needed from:** EH&S, the actual procedure.

3. **Running the monitor as a service.** `scripts/run_live.py` stops when the
   terminal closes. Whether it's meant to run under systemd, cron, or on a
   specific machine is not recorded. **Needed from:** whoever operates it now.

4. **What "currently deployed" means in practice.** The README names a deployed
   model, but where it runs, on what schedule, and whether this repository is
   that deployment or a research copy of it's not stated anywhere.
   **Needed from:** the project lead.

5. **Cobble Shoal through `main.py`.** It's a `--model` choice that can't run,
   for the reason in section 3. Whether the intended fix is to generate the
   missing metadata or to wire in its separate data path is an open decision.
   **Needed from:** whoever owns that model.

6. **The seven unmodelled sites.** Four of eleven sites are modelled. The README
   says the rest are "slated to be brought in as their data ingestion is
   completed", with no order or timeline. **Needed from:** the project lead.

7. **Retraining.** Section 2 says don't run `--mode train` and ask first. What
   the actual approval process is, how often retraining should happen, and how a
   new set of weights gets validated before replacing the old ones are not
   recorded. **Needed from:** whoever owns the models.

8. **The `settings.yaml` file.** `strawberrywatch/settings.yaml` holds model
   architecture settings read by `config.py`. Which of them are safe for an
   intern to change and which are not is not documented. **Needed from:**
   whoever owns the models.

9. **Windows.** Every command here was verified on Linux. The activation command
   for Windows is given from the standard Python documentation but was not
   tested. **Needed:** somebody to run through section 1 on Windows.

10. **The two failing inventory comment tests.** Described in section 7. The
    header comment in `inventory.yaml` and the tests that check it disagree.
    **Needed from:** the project lead, a decision on which one is correct.

11. **The scripts folder.** Nine scripts exist and only three are described in
    this manual. The others are analysis and audit tools whose current status is
    unclear. **Needed from:** whoever wrote them.

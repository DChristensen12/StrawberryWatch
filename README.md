<div align="center">

<h1>StrawberryWatch</h1>

<p>A modular anomaly detection system for the Strawberry Creek urban watershed</p>

<p align="center">
  <img src="assets/logos/SCMGlogo.jpg" width="575">
</p>

<p><a href="https://github.com/DChristensen12/StrawberryWatch/actions/workflows/lint.yml"><img src="https://github.com/DChristensen12/StrawberryWatch/actions/workflows/lint.yml/badge.svg" alt="CI"></a> <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a> <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit"></a> <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+"> <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-EE87A0" alt="License: Apache 2.0"></a></p>

</div>

This repository is the research and development counterpart to the production monitoring platform. It is where anomaly detection models are built, tested against historical events, and validated!

New here? Start with the [User Manual](USER_MANUAL.md). Currently deployed: Dusk Crayfish (as of 2026-08-20).


## The Creek

The creek is monitored at eleven locations: UC Botanical Gardens, Women's Faculty Club (south fork 0), Stephens Hall (south fork 1), Downstream of Sather Gate (south fork 2), Weill Hall (south fork 3), Kingman Hall Garden, University House, Giannini Hall (north fork 0), Wickson Footbridge (north fork 1, also sometimes labeled as scnf010), Oxford Street, and Codornices Creek. The eleventh site, Codornices, is a separate watershed monitored as a standalone point.

### Sensors + Measured Metrics

Every site runs an [EnviroDIY Mayfly Data Logger](https://www.envirodiy.org/mayfly/),
sampling every 15 minutes. Which sensors are present at which site varies, and
that is recorded in the inventory config instead of being inferred from the data, so
it is able to tell when a sensor is down and when a sensor was never installed.

| Metric | Probe | Protocol |
|---|---|---|
| Conductivity, temperature, depth | METER HYDROS 21 CTD | SDI-12 |
| Dissolved oxygen | AtlasScientific DO | analog or I2C, depending on probe generation |
| Floating conductivity | AtlasScientific Mini Conductivity K 1.0 | I2C |
| pH | AtlasScientific pH | I2C |

Conductivity, temperature and depth come from one probe, so a site has all three
or none. The rest are installed separately.

Floating conductivity sits in a sealed float inside a perforated housing and
reads the top of the water column, which is what separates a surface
contaminant like oil from one that mixes through.

Three legacy Balance Hydrologics sites are read from a separate system and serve
only the past seven days.


## Models

<table align="center">
  <tr>
    <td align="center"><b>Dusk Crayfish</b></td>
    <td align="center"><b>Water Strider</b></td>
    <td align="center"><b>Flame Skimmer</b></td>
  </tr>
  <tr>
    <td align="center" width="33%">A graph neural network paired with an LSTM. It predicts what the creek as a whole should be reading next, and flags timesteps where the network differs from that.</td>
    <td align="center" width="33%">TBD</td>
    <td align="center" width="33%">A Bayesian neural network. It learns a distribution over its weights rather than fixed values, so every forecast carries a calibrated uncertainty estimate.</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/biota/Crayfish.jpeg" width="351"></td>
    <td align="center"><img src="assets/biota/WaterStrider.jpeg" width="351"></td>
    <td align="center"><img src="assets/biota/Flame_Skimmer.jpeg" width="351"></td>
  </tr>
   <tr>
    <td align="center"><b>Berry Delight</b></td>
    <td align="center"><b>Cobble Shoal</td>
    <td align="center"><b>Crimson Parker</b></td>
  </tr>
  <tr>
    <td align="center" width="33%">Logistic Regression Model</td>
    <td align="center" width="33%">A graph neural network paired with an LSTM that scores each sensor separately rather than the network as a whole. It treats every sensor at every site as its own node, learns how quickly each reading goes stale, and combines several independent checks so it can name which sensor looks wrong and how confident it is.</td>
    <td align="center" width="33%">TBD</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/biota/Berries.jpeg" width="351"></td>
    <td align="center"><img src="assets/biota/Fish.jpeg" width="351"></td>
    <td align="center"><img src="assets/biota/Spider.jpeg" width="351"></td>
  </tr>
</table>

## Support Modules

Support modules attach to any model. They are not models themselves and have no
weights of their own. Some explain a detection after it happens while others cover something a model structurally cannot see (adds tests,
so switching one on makes the model slightly less sensitive elsewhere).

<table align="center">
  <tr>
    <td align="center"><b>Trial Bed</b></td>
    <td align="center"><b>Newtwork Run</b></td>
    <td align="center"><b>Settling Pool</b></td>
  </tr>
  <tr>
    <td align="center" width="33%">Identifies what kind of contamination a detection looks like. It reads the direction each measurement moved, matches that pattern against known signatures, and reports the likely cause alongside the readings behind it.</td>
    <td align="center" width="33%">Confirms a detection travelled the way water does. A spill reaches downstream sites after a delay set by the creek's flow and never crosses forks, while rain arrives everywhere at once; comparing timing across sites separates the two and flags anything water could not have carried.</td>
    <td align="center" width="33%">Screens incoming data before it gets passed into a model. Catches placeholder values passing as real readings, feeds silently mirroring another site, sensors re-zeroed between deployments, and gaps that were filled rather than measured, each one something a model would otherwise learn from.</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/biota/Rose_Bed_BG.jpeg" width="800"></td>
    <td align="center"><img src="assets/biota/Newt.jpeg" width="800"></td>
    <td align="center"><img src="assets/biota/JPond_BG.jpeg" width="800"></td>
  </tr>
</table>


## What is deployed

**Dusk Crayfish** is the model currently running, as of 2026-08-20. Update the
date whenever the deployment changes. Nothing else in the code records which
model is live.

Dusk Crayfish learns the creek's normal behavior from sensor data and flags
deviations that look like spills or contamination events, without being trained
on labeled anomalies. It treats the creek as a connected graph of sensor sites
and combines a graph neural network with a Long Short Term Memory architecture
to reason about both where a sensor sits in the flow and how its readings change
over time.

The map below shows the sensors on the actual creek, with each fork tracing the
real water flow down to the Oxford Street confluence. To see the creek in more
detail, view the map on the Strawberry Creek website.

<p align="center">
  <img src="assets/diagrams/Strawberry_Creek_Physical_Graph_Topology.png" width="80%">
</p>

<p align="center"><i>The physical sensor network along Strawberry Creek, overlaid on the creek's real flow. Both forks converge at the Oxford Street confluence. This is the full field deployment, not the modeled graph: four of these sites are modeled, north_fork_0, south_fork_1, south_fork_2, and Oxford.</i></p>

The model is trained and validated on a four-node core of that network, the main
flow path where conductivity data is reliable. The remaining sites are brought in
as their data ingestion is completed. The north path runs straight from
north_fork_0 to Oxford because the footbridge node between them was retired; the
water still flows that way, there is just no sensor reporting in between.

```mermaid
graph LR
    %% Focused Horizontal Network Layout with No Legend
    subgraph ST_Backbone [DuskCrayfish&nbsp;Network&nbsp;Topology]
        %% Active South Fork Path
        SF1([South Fork 1]) --> SF2([South Fork 2])

        %% Active North Fork Path, contracted through the retired footbridge node
        NF0([North Fork 0])

        %% Convergence Sink
        SF2 --> OX{Oxford Street}
        NF0 --> OX

        %% Isolated Node kept inline horizontally to the right of the sink
        CC[[Codornices Creek]]
        OX ~~~ CC
    end

    %% Color Palette Configurations
    classDef nodeStyle fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef targetStyle fill:#065f46,stroke:#10b981,stroke-width:2px,color:#fff;
    classDef controlStyle fill:#334155,stroke:#94a3b8,stroke-width:2px,color:#cbd5e1,stroke-dasharray: 5 5;

    class SF1,SF2,NF0 nodeStyle;
    class OX targetStyle;
    class CC controlStyle;
```

<p align="center"><i>Dusk Crayfish Active Graph Topology: the 4-node subnetwork the model actually trains and serves on.</i></p>

The system pulls sensor readings, merges in weather, learns what normal looks
like across the whole network, and then scores new data by how badly the model
fails to predict it. A large, sustained prediction error on conductivity that
shows up across connected sites is the signature of a real event.

The model is unsupervised. It is trained only to predict the next reading from
recent history. Anything it cannot predict well is, by definition, something it
has not seen before, which is what an anomaly is.

## Learning to use StrawberryWatch

Everything about installing, running and changing this system lives in the
**[User Manual](USER_MANUAL.md)**. It is the place to start, and it is beginner friendly
(including to those who have never written in Python and/or have never used a terminal before).

It covers setting up Python and a virtual environment from scratch, running each
mode with the output you should expect on your screen, editing the sensor
inventory, running the tests, and what to do when something goes wrong. Work
through it in order the first time.

| I want to | Go to |
|---|---|
| Install it for the first time | [Getting set up](USER_MANUAL.md#1-getting-set-up) |
| Run it and understand what it prints | [Running the system](USER_MANUAL.md#2-running-the-system) |
| Know what each model does | [The models](USER_MANUAL.md#3-the-models) |
| Take a sensor out of service | [The inventory](USER_MANUAL.md#5-the-inventory) |
| Change some code and commit it | [Making a change to the code](USER_MANUAL.md#6-making-a-change-to-the-code) |
| Run the tests | [Running the tests](USER_MANUAL.md#7-running-the-tests) |
| Fix an error I am seeing | [Common problems](USER_MANUAL.md#9-common-problems) |

---

## Thank you to our contributors!

<a href="https://github.com/DChristensen12/StrawberryWatch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=DChristensen12/StrawberryWatch" width="65" alt="Contributors" />
</a>

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for everyone who has helped, including
those whose contributions do not appear in the commit history.

---

<p align="center">
  <img src="assets/logos/SCMGBacklogo.png" width="400">
</p>

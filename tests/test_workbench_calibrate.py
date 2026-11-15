"""The re-fit adapters, saving and switching calibrations, and the shipped files."""
# The shipped Dusk Crayfish lines can't be rebuilt from the repository's data: training code
# changed after the 27 July weights, up to 48% apart at South Fork 1. The re-fit is tested
# against trainer.py's own formulas on the same arrays to 1e-9, and Cobble Shoal's against
# the shipped refit script's steps

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strawberrywatch.anomalies import channel_scoring
from strawberrywatch.workbench import calibrate, store

REPO = Path(__file__).resolve().parent.parent


def repo_hashes():
    folders = [REPO / "checkpoints", REPO / "strawberrywatch" / "watch" / "events"]
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for f in folders
        for p in f.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def folder(tmp_path):
    return store.ready(tmp_path / "wb")


def fake_dusk(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-06-01", periods=n, freq="15min", tz="UTC")
    return {
        site: {
            "times": times,
            "error": np.abs(rng.normal(0.01, 0.01, n)),
            "rain": np.ones(n),
            "level": rng.normal(0.5, 0.3, n),
            "k": 4.0,
        }
        for site in calibrate.model_sites("dusk_crayfish")
    }


def test_dusk_refit_matches_the_trainer_formulas():
    scored = fake_dusk()
    keep = {site: np.ones(len(d["times"]), dtype=bool) for site, d in scored.items()}
    q = calibrate.per_step_rate("dusk_crayfish", "week")
    base = calibrate.blob(None, "dusk_crayfish", store.SHIPPED)
    new = calibrate.dusk_refit(scored, keep, q, base)
    for site, d in scored.items():
        # trainer.py: median and IQR of the errors, a flat IQR counted as 1
        median = np.median(d["error"])
        q75, q25 = np.percentile(d["error"], [75, 25])
        assert new["error_median"][site] == pytest.approx(median, rel=1e-9)
        assert new["error_iqr"][site] == pytest.approx(q75 - q25, rel=1e-9)
        line = channel_scoring.pot_diagnostics((d["error"] - median) / (q75 - q25), q=q)[
            "threshold"
        ]
        assert new["node_thresholds"][site] == pytest.approx(line, rel=1e-9)
        lq75, lq25 = np.percentile(d["level"], [75, 25])
        assert new["cond_median"][site] == pytest.approx(np.median(d["level"]), rel=1e-9)
        assert new["cond_iqr"][site] == pytest.approx(lq75 - lq25, rel=1e-9)
    # The base it started from is untouched
    assert base == calibrate.blob(None, "dusk_crayfish", store.SHIPPED)


def test_rarer_false_alarms_raise_the_line():
    scored = fake_dusk()
    keep = {site: np.ones(len(d["times"]), dtype=bool) for site, d in scored.items()}
    base = calibrate.blob(None, "dusk_crayfish", store.SHIPPED)
    lines = {
        choice: calibrate.dusk_refit(
            scored, keep, calibrate.per_step_rate("dusk_crayfish", choice), base
        )
        for choice in ("day", "week", "month")
    }
    for site in scored:
        assert (
            lines["day"]["node_thresholds"][site]
            < lines["week"]["node_thresholds"][site]
            < lines["month"]["node_thresholds"][site]
        )


def test_cobble_refit_follows_the_shipped_refit_steps():
    rng = np.random.default_rng(3)
    n_anchors, n_nodes = 600, 17
    channels = [
        {
            name: np.abs(rng.normal(0, 1, n_nodes))
            for name in ("excursion", "loo", "reconstruction", "dispersion")
        }
        for _ in range(n_anchors)
    ]
    grid = pd.date_range("2026-06-01", periods=n_anchors + 24, freq="15min", tz="UTC")
    scored = {
        "grid": grid,
        "anchors": list(range(24, n_anchors + 24)),
        "channels": channels,
        "rain": np.ones((n_anchors, n_nodes)),
        "keys": [("north_fork_0", "conductivity")] * n_nodes,
        "window": 24,
    }
    q = calibrate.per_step_rate("cobble_shoal", "week")
    base = calibrate.blob(None, "cobble_shoal", store.SHIPPED)
    new = calibrate.cobble_refit(scored, [True] * n_anchors, q, base)
    picks = sorted({int(i) for i in np.linspace(0, n_anchors - 1, calibrate.NULL_ANCHORS)})
    nulls = channel_scoring.ChannelNulls().fit(
        {k: np.concatenate([channels[i][k] for i in picks]) for k in channels[0]}
    )
    combined = np.concatenate(
        [channel_scoring.combine(pvalues=nulls.pvalues(channels[i]), rule="fisher") for i in picks]
    )
    expected = channel_scoring.pot_diagnostics(combined, q=q)["threshold"]
    assert new["z_q"] == pytest.approx(expected, rel=1e-9)
    assert new["operating_q"] == q and new["n_null_windows"] == len(picks)
    assert new["node_scaler"] == base["node_scaler"]


def test_saving_puts_it_in_use_and_switching_back_restores_shipped(folder):
    before = repo_hashes()
    shipped = calibrate.blob(folder, "dusk_crayfish", store.SHIPPED)
    new = json.loads(json.dumps(shipped))
    new["node_thresholds"] = {site: 20.0 for site in new["node_thresholds"]}
    session = {
        "source": "csv",
        "sites": calibrate.model_sites("dusk_crayfish"),
        "readings": ["conductivity"],
        "start_date": "2026-06-01",
        "end_date": "2026-06-30",
        "leave_out_events": 1,
        "leave_out_rain": 1,
        "notes": "calm",
    }
    check = {
        "model": "dusk_crayfish",
        "false_alarms": "week",
        "calibration": store.SHIPPED,
        "events": [],
        "before": 3,
        "after": 1,
        "blob": new,
        "reference": [["north_fork_0", "conductivity", 500.0]],
    }
    saved = calibrate.save(folder, session, check)
    assert calibrate.in_use(folder, "dusk_crayfish") == saved
    assert calibrate.blob(folder, "dusk_crayfish", saved) == new
    store.use(folder, "dusk_crayfish", store.SHIPPED)
    assert (
        calibrate.blob(folder, "dusk_crayfish", calibrate.in_use(folder, "dusk_crayfish"))
        == shipped
    )
    assert repo_hashes() == before


def test_a_changed_file_falls_back_to_shipped(folder):
    shipped = calibrate.blob(folder, "cobble_shoal", store.SHIPPED)
    saved = store.save_calibration(
        folder, "cobble_shoal", calibrate.FILENAMES["cobble_shoal"], {**shipped, "z_q": 50.0}, {}
    )
    store.use(folder, "cobble_shoal", saved)
    assert calibrate.in_use(folder, "cobble_shoal") == saved
    path = (
        store.calibration_dir(folder, "cobble_shoal", saved) / calibrate.FILENAMES["cobble_shoal"]
    )
    path.write_text(path.read_text().replace("50.0", "51.0"))
    assert calibrate.in_use(folder, "cobble_shoal") == store.SHIPPED
    assert calibrate.blob(folder, "cobble_shoal", saved) == shipped


def test_alerts_count_the_way_checks_do():
    # Three steps over the line in a row at one site make one Dusk Crayfish alert, not three
    times = np.arange(0, 96 * 900, 900, dtype=float) + 1.78e9
    scores = np.zeros(len(times))
    scores[40:43] = 10.0
    lines = np.full(len(times), 5.0)
    grouped = {("north_fork_0", "conductivity", "forecast_residual"): (times, scores, lines, lines)}
    found = calibrate.alerts(
        "dusk_crayfish",
        grouped,
        pd.Timestamp(times[0], unit="s", tz="UTC"),
        pd.Timestamp(times[-1], unit="s", tz="UTC"),
    )
    assert len(found) == 1 and found[0]["first_over"] == times[40]
    scores[42] = 0.0
    assert not calibrate.alerts(
        "dusk_crayfish",
        grouped,
        pd.Timestamp(times[0], unit="s", tz="UTC"),
        pd.Timestamp(times[-1], unit="s", tz="UTC"),
    )

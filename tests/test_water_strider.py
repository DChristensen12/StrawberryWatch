"""
Water Strider, from the arithmetic up to the dashboard's slider. The first
three tests are the slider's sensitivity checks.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import paths
from strawberrywatch.models import Water_Strider as wsm
from strawberrywatch.serving.contract import notice

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = paths.checkpoints_dir() / "water_strider_calibration.json"


@pytest.fixture(scope="module")
def calibration():
    return json.loads(CALIBRATION.read_text())


@pytest.fixture(scope="module")
def detector():
    from strawberrywatch.serving.water_strider import WaterStriderDetector

    return WaterStriderDetector.load(paths.checkpoints_dir())


def _tail(gamma):
    return {"t": 4.0, "sigma": 1.5, "gamma": gamma, "n": 50_000, "n_t": 120}


# The slider's sensitivity checks


def test_at_the_calibrated_stop_the_recomputed_threshold_equals_the_stored_one(calibration):
    for entry in calibration["sensors"].values():
        for tail in entry["tests"].values():
            again = wsm.threshold_for(tail, wsm.per_test(calibration["q_total"]))
            assert again == pytest.approx(tail["threshold"], rel=1e-12)


def test_a_typed_threshold_to_a_rate_and_back_is_the_same_threshold():
    # A negative shape has a finite end, t - sigma / gamma, here 9, and nothing past it maps back
    for gamma, highest in ((-0.3, 8.5), (0.0, 14.0), (0.25, 14.0), (0.6, 14.0)):
        tail = _tail(gamma)
        for z in (5.0, 7.0, highest):
            assert wsm.threshold_for(tail, wsm.rate_for(tail, z)) == pytest.approx(z, rel=1e-9)


def test_a_shape_of_zero_and_just_either_side_of_it_give_matching_thresholds():
    q = wsm.per_test(wsm.Q_TOTAL)
    exactly = wsm.threshold_for(_tail(0.0), q)
    for gamma in (1e-9, -1e-9, 1e-7, -1e-7):
        assert wsm.threshold_for(_tail(gamma), q) == pytest.approx(exactly, rel=1e-5)
        assert wsm.rate_for(_tail(gamma), exactly) == pytest.approx(q, rel=1e-5)


# The statistics


def _series(values, start="2026-01-05"):
    index = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_the_hampel_test_counts_in_scaled_mads():
    resid = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 0.0, 1.0, -1.0, 12.0])
    window = resid[-13:]
    mad_scaled = np.median(np.abs(window - np.median(window))) * wsm.MAD_SCALE
    expected = abs(12.0 - np.median(window)) / mad_scaled
    assert wsm.hampel(resid, floor=0.01)[-1] == pytest.approx(expected)
    # A flat sensor is judged against its floor, not a near-zero MAD
    flat = np.concatenate([np.zeros(12), [0.3]])
    assert wsm.hampel(flat, floor=1.0)[-1] == pytest.approx(0.3)


def test_a_window_with_too_few_real_readings_is_not_judged():
    resid = np.full(13, np.nan)
    resid[[-1, -2, -3]] = [5.0, 0.0, 0.0]
    assert np.isnan(wsm.hampel(resid, floor=0.1)[-1])


def test_cusum_keeps_adding_for_a_shift_that_holds_and_rests_at_zero_otherwise():
    quiet = np.zeros(40)
    assert np.nanmax(wsm.cusum(quiet)) == 0.0
    shifted = np.concatenate([np.zeros(20), np.full(20, 2.0)])
    stat = wsm.cusum(shifted)
    assert stat[19] == 0.0 and stat[-1] == pytest.approx(20 * (2.0 - 0.5))
    assert np.all(np.diff(stat[20:]) > 0)


def test_ewma_reads_in_its_own_spread_and_follows_a_drift():
    lam = wsm.PARAMS["ewma_lambda"]
    steady = wsm.ewma(np.ones(400))
    assert steady[-1] == pytest.approx(1.0 / np.sqrt(lam / (2 - lam)), rel=1e-6)
    drift = wsm.ewma(np.linspace(0, 3, 200))
    assert drift[-1] > drift[100] > drift[10]


def test_the_daily_profile_takes_out_a_normal_day():
    index = pd.date_range("2026-01-01", periods=96 * 20, freq="15min", tz="UTC")
    slot = wsm.slot_of(index)
    day = np.sin(2 * np.pi * slot / 96) * 5.0
    values = pd.Series(300.0 + day, index=index)
    raw = values.to_numpy()
    above = raw - wsm.level(raw)
    per_step, latest = wsm.rolling_profiles(index, above)
    resid = above - per_step
    # Once two weeks of profile are in, the rhythm is gone
    assert np.nanmax(np.abs(resid[-96:])) < 1e-9
    assert np.allclose(wsm.residuals(values, latest)[-96:], 0.0, atol=1e-9)


def test_tails_are_fitted_to_cluster_peaks():
    rng = np.random.default_rng(0)
    statistic = rng.exponential(1.0, 20_000)
    # A long excursion is one cluster, however many steps it spends over the line
    statistic[5000:5300] = 30.0
    mask = np.ones(statistic.size, dtype=bool)
    tail = wsm.fit_tail(statistic, mask, gap=8)
    peaks = wsm.clusters(statistic, mask, tail["t"], 8)
    assert tail["n_t"] == peaks.size and 30.0 in peaks
    assert tail["n_t"] < int((statistic > tail["t"]).sum())


# The detector


def _site_tables(entry_site, values_by_column, start="2026-01-05"):
    index = pd.date_range(
        start, periods=len(next(iter(values_by_column.values()))), freq="15min", tz="UTC"
    )
    return {entry_site: pd.DataFrame(values_by_column, index=index)}


def test_a_lone_spike_is_left_to_settling_pool_and_two_in_a_row_count(detector, calibration):
    rng = np.random.default_rng(1)
    entry = calibration["sensors"]["oxford.conductivity"]
    base = 400 + rng.normal(0, entry["spread"] / 2, 96 * 2)
    # Far past the spike test's line, against readings that wobble by half a spread
    jump = 10 * entry["tests"]["hampel_spike"]["threshold"] * entry["spread"]
    lone = base.copy()
    lone[-10] += jump
    pair = base.copy()
    pair[-3:-1] += jump
    for values, expect in ((lone, False), (pair, True)):
        tables = _site_tables("oxford", {"Meter_Hydros21_Cond": values})
        found = detector.findings(tables, None)
        spikes = [f for f in found if f.variable == "conductivity" and f.rule == "hampel_spike"]
        assert bool(spikes) is expect
        assert {(f.site, f.variable, f.rule, f.when) for f in found} == {
            (n.site, n.variable, n.rule, n.when) for n in notice(detector.scores(tables, None))
        }


def test_moving_the_rate_moves_every_threshold_from_its_own_tail(calibration):
    from strawberrywatch.serving.water_strider import WaterStriderDetector

    d = WaterStriderDetector(calibration)
    key = "oxford.conductivity"
    stored = calibration["sensors"][key]["tests"]["cusum_shift"]["threshold"]
    assert d.base_threshold(key, "cusum_shift") == pytest.approx(stored)
    d.set_sensitivity({"cusum_shift": 1e-3})
    looser = d.base_threshold(key, "cusum_shift")
    assert looser < stored
    assert looser == pytest.approx(
        wsm.threshold_for(calibration["sensors"][key]["tests"]["cusum_shift"], wsm.per_test(1e-3))
    )
    d.set_sensitivity(thresholds={f"{key}.cusum_shift": 42.0})
    assert d.base_threshold(key, "cusum_shift") == 42.0


def test_rain_raises_every_line(detector):
    rng = np.random.default_rng(2)
    values = 400 + rng.normal(0, 3, 96 * 2)
    tables = _site_tables("oxford", {"Meter_Hydros21_Cond": values})
    index = tables["oxford"].index
    weather = pd.DataFrame(
        {"rain_mm": np.where(np.arange(len(index)) > 150, 2.0, 0.0)}, index=index
    )
    dry = {(s.rule, s.time): s.threshold for s in detector.scores(tables, None)}
    wet = {(s.rule, s.time): s.threshold for s in detector.scores(tables, weather)}
    late = [k for k in dry if k[1] > index[160]]
    assert late and all(wet[k] > dry[k] for k in late)


def test_it_never_imports_torch():
    code = (
        "import sys, numpy as np, pandas as pd\n"
        "from strawberrywatch import paths\n"
        "from strawberrywatch.serving.water_strider import WaterStriderDetector\n"
        "d = WaterStriderDetector.load(paths.checkpoints_dir())\n"
        "i = pd.date_range('2026-01-05', periods=192, freq='15min', tz='UTC')\n"
        "d.findings({'oxford': pd.DataFrame({'Meter_Hydros21_Cond': 400 + np.zeros(192)}, index=i)})\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_the_shipped_calibration_passed_and_is_hash_checked(calibration):
    from strawberrywatch.watch import runner

    assert calibration["passed"] and calibration["model"] == "water_strider"
    for rule, entry in calibration["held_out"].items():
        assert entry["passed"], rule
    hashes = json.loads(runner.HASHES_PATH.read_text())
    assert hashes["water_strider_calibration.json"] == runner.sha256(CALIBRATION)
    assert all(len(e["profile"]) == wsm.STEPS_PER_DAY for e in calibration["sensors"].values())


# Settings


@pytest.fixture(scope="module")
def models():
    from strawberrywatch.watch import models as wm

    return wm.registered()


def test_fewer_balanced_and_more_are_the_middle_three_stops(models):
    from strawberrywatch.watch import settings as ws

    settings = ws.defaults(models)
    assert settings["models"]["water_strider"]["rates"] == dict.fromkeys(wsm.RULES, wsm.Q_TOTAL)
    assert ws.alerts_choice(settings) == "balanced"
    for choice, rate in (("fewer", 3e-5), ("more", 3e-4), ("balanced", 1e-4)):
        moved = ws.apply_choice(settings, choice)
        assert set(moved["models"]["water_strider"]["rates"].values()) == {rate}
        assert ws.alerts_choice(moved) == choice
    custom = ws.apply_choice(settings, "more")
    custom["models"]["water_strider"]["rates"]["ewma_drift"] = 1e-3
    assert ws.alerts_choice(custom) == "custom"


def test_rates_must_be_stops_and_typed_thresholds_are_checked(models):
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import settings as ws

    sites = list(inv.load().sites)
    settings = ws.defaults(models)
    spec = settings["models"]["water_strider"]
    spec["rates"]["hampel_spike"] = 5e-5
    spec["thresholds"] = {"oxford.conductivity.cusum_shift": 0.001, "nowhere.depth.ewma_drift": 3}
    clean, problems, warnings = ws.check(settings, models, sites)
    assert any("spike test must be one of the five slider settings" in p for p in problems)
    assert any("no sensor and test called 'nowhere.depth.ewma_drift'" in p for p in problems)
    assert any("the lowest value the calibration covers" in w for w in warnings)
    assert clean["models"]["water_strider"]["thresholds"] == {
        "oxford.conductivity.cusum_shift": 0.001
    }


def test_history_says_what_moved(models):
    from strawberrywatch.watch import settings as ws

    before = ws.defaults(models)
    after = json.loads(json.dumps(before))
    after["models"]["water_strider"]["rates"]["ewma_drift"] = 1e-3
    after["models"]["water_strider"]["thresholds"] = {"oxford.conductivity.hampel_spike": 9.0}
    sentences = [row["sentence"] for row in ws.describe(before, after, models)]
    assert "Water Strider drift test set to far more false alarms." in sentences
    assert "Threshold for the spike test for conductivity at Oxford Street set to 9." in sentences


def test_saved_settings_get_water_strider_switched_off(models):
    from strawberrywatch.watch import settings as ws

    old = ws.defaults([m for m in models if m.name != "water_strider"])
    filled = ws.with_new_models(old, models)
    assert filled["models"]["water_strider"]["on"] is False
    assert "water_strider" not in old["models"]


# The dashboard


def _csrf(page):
    return re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()


@pytest.fixture
def client(models, tmp_path):
    from strawberrywatch import inventory as inv
    from strawberrywatch.dashboard import create_app, security
    from strawberrywatch.watch import settings as ws
    from strawberrywatch.watch import sources
    from strawberrywatch.watch import store as st

    class Watcher:
        busy, next_at = False, None

        def check_now(self):
            pass

    store = st.Store(tmp_path / "s")
    store.set_source(sources.REPLAY, event="anomaly_2025_06_12_spill_sf")
    settings = ws.defaults(models)
    settings["models"]["water_strider"]["on"] = True
    clean, _, _ = ws.check(settings, models, list(inv.load().sites))
    store.apply(clean)
    # One real check, which gives the 7-day preview scores to count
    from strawberrywatch.watch import runner

    runner.run_check(store, models)
    app = create_app(store=store, models=models, watcher=Watcher(), login=security.Login("t"))
    c = app.test_client()
    c.get("/login?token=t")
    return c, store


def _advanced_form(store, models, **changes):
    draft = store.draft() or store.applied()[1]
    data = {"repeat_hours": str(draft["repeat_hours"])}
    for m in models:
        spec = draft["models"][m.name]
        data[f"model_{m.name}"] = "on" if spec["on"] else "off"
        data[f"rain_{m.name}"] = str(spec["rain_multiplier"])
        data[f"support_{m.name}"] = spec["support"]
        if "leave_out" in m.site_choice:
            data[f"leave_{m.name}"] = spec["leave_out"]
        for rule in m.rules:
            if m.sensitivity == "rate":
                data[f"rate_{m.name}_{rule}"] = "2"
            else:
                data[f"scale_{m.name}_{rule}"] = str(spec["scales"][rule])
    data.update(changes)
    return data


def test_advanced_settings_has_a_slider_per_test_and_back_to_calibrated(client, models):
    c, store = client
    page = c.get("/settings/advanced").data.decode()
    for name in (
        "rate_water_strider_hampel_spike",
        "rate_water_strider_cusum_shift",
        "rate_water_strider_ewma_drift",
    ):
        assert f'name="{name}"' in page and 'type="range"' in page
    for label in ("Far fewer", "Fewer", "Calibrated", "More", "Far more"):
        assert label in page
    assert "Reset to calibrated" in page and "Set a threshold by hand" in page
    assert 'name="scale_water_strider_hampel_spike"' not in page
    token = _csrf(c.get("/settings/advanced").data)
    form = _advanced_form(store, models, rate_water_strider_ewma_drift="4", csrf_token=token)
    form.update(
        typed_key_water_strider="oxford.conductivity.hampel_spike", typed_value_water_strider="9"
    )
    answer = c.post(
        "/settings/advanced", data=form, headers={"X-Requested-With": "fetch"}
    ).get_json()
    # The drift stop, the typed threshold, and How many alerts moving to custom
    assert answer["changes"] == 3
    assert "Last 7 days at this setting:" in json.dumps(answer["rate_previews"])
    draft = store.draft()["models"]["water_strider"]
    assert draft["rates"]["ewma_drift"] == 1e-3
    assert draft["thresholds"] == {"oxford.conductivity.hampel_spike": 9.0}
    page = c.get("/settings/advanced").data.decode()
    assert "False alarm rate" in page and "per reading" in page
    review = c.get("/settings/review?from=advanced").data.decode()
    assert "Far more" in review
    c.post("/settings/apply", data={"csrf_token": _csrf(c.get("/settings/review").data)})
    assert "Water Strider uses custom sensitivity." in c.get("/").data.decode()
    assert (
        "Water Strider drift test set to far more false alarms."
        in c.get("/history?show=changes").data.decode()
    )
    # Back to calibrated, without JavaScript
    form = _advanced_form(store, models, csrf_token=_csrf(c.get("/settings/advanced").data))
    c.post("/settings/advanced", data={**form, "calibrated": "water_strider"})
    draft = store.draft()["models"]["water_strider"]
    assert set(draft["rates"].values()) == {wsm.Q_TOTAL} and draft["thresholds"] == {}


def test_the_models_card_and_the_readme_say_what_it_is(client):
    c, _ = client
    page = c.get("/models").data.decode()
    assert "Checks each sensor for spikes, lasting shifts and slow drift" in page
    readme = (ROOT / "README.md").read_text()
    assert "checks each sensor for spikes, lasting shifts and slow drift." in readme
    assert (
        '<td align="center" width="33%">TBD</td>\n    <td align="center" width="33%">A Bayesian'
        not in readme
    )

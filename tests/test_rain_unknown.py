"""Rain that's missing is unknown, never dry, and the check falls back to the dry line."""

import numpy as np
import pandas as pd

from strawberrywatch.anomalies import anomaly_detector as ad
from strawberrywatch.watch import runner

STEPS = pd.date_range("2026-11-13 00:00", periods=4 * 48, freq="15min", tz="UTC")


def _rain(gap=None):
    rain = pd.Series(0.0, index=STEPS)
    rain.iloc[96:104] = 2.0
    if gap is not None:
        rain.iloc[gap] = np.nan
    return rain


def test_rain_is_known_only_where_the_whole_lookback_has_a_value():
    known = ad.rain_known(STEPS, _rain(gap=slice(120, 124)), 12)
    # the first 12 hours look back before the data starts
    assert not known[:48].any() and known[48:120].all()
    # the gap, and every step whose 12 hour lookback holds it
    assert not known[120 : 124 + 48].any() and known[124 + 48 :].all()


def test_unknown_rain_keeps_the_dry_line_in_dusk_crayfish():
    args = (12, 2.0, 0.1, 36)
    known_lines, _ = ad._rain_multipliers(STEPS, _rain(), *args)
    lines, flags = ad._rain_multipliers(STEPS, _rain(gap=slice(120, 124)), *args)
    gap = slice(120, 124 + 48)
    assert (known_lines[gap] > 1.0).all()
    assert (lines[gap] == 1.0).all() and not flags[gap].any()
    assert np.array_equal(lines[124 + 48 :], known_lines[124 + 48 :])


def test_rain_comes_from_any_site_that_has_it():
    stamps = STEPS[:8]
    frame = pd.DataFrame(
        {
            "location": ["north_fork_0"] * 4 + ["south_fork_1"] * 4,
            "rain_mm": [1.0, np.nan, np.nan, 1.0, np.nan, 0.5, 0.5, np.nan],
        },
        index=list(stamps[:4]) + list(stamps[:4]),
    )
    rain = ad._extract_rain_series(frame, ["north_fork_0", "south_fork_1"])
    assert rain.tolist() == [1.0, 0.5, 0.5, 1.0]


def test_unknown_rain_keeps_the_dry_line_in_water_strider():
    from strawberrywatch.serving.water_strider import WaterStriderDetector
    from strawberrywatch.serving.water_strider import _rain as per_quarter

    detector = WaterStriderDetector({"sensors": {}}, rain_gate={"mode": "deployed"})
    weather = pd.DataFrame({"rain_mm": _rain(gap=slice(120, 124))})
    lines = detector._multipliers(STEPS, per_quarter(weather))
    full = detector._multipliers(STEPS, per_quarter(pd.DataFrame({"rain_mm": _rain()})))
    gap = slice(120, 124 + 48)
    assert (full[gap] > 1.0).all() and (lines[gap] == 1.0).all()
    # an empty quarter hour is NaN, not 0 mm
    assert np.isnan(per_quarter(weather).iloc[121])


def test_a_check_says_rain_is_unavailable_when_a_judged_step_lacks_it():
    start, end = STEPS[0], STEPS[-1]
    assert not runner.rain_covered(None, start, end)
    full = pd.DataFrame({"rain_mm": _rain()})
    assert runner.rain_covered(full, start, end)
    assert not runner.rain_covered(
        pd.DataFrame({"rain_mm": _rain(gap=slice(150, 151))}), start, end
    )
    # the first judged step is 12 hours in, and it needs those 12 hours of rain
    assert not runner.rain_covered(pd.DataFrame({"rain_mm": _rain()[24:]}), start, end)


# A node isn't judged until its input window is whole again after an outage


def _spill_tables():
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import sources

    replay = sources.ReplaySource("anomaly_2025_06_12_spill_sf")
    start, end = replay.window(None)
    return replay.tables(list(inv.load().sites), start, end), replay.weather(start, end)


def _judged(tables, weather, site="south_fork_1"):
    from strawberrywatch import paths
    from strawberrywatch.serving import detector_class

    detector = detector_class("dusk_crayfish").cached(paths.model_dir("dusk_crayfish"))
    times = {s.time for s in detector.scores(tables, weather) if s.site == site}
    return pd.DatetimeIndex(sorted(times))


def test_after_a_long_outage_a_node_waits_a_full_window_before_it_is_judged():
    tables, weather = _spill_tables()
    sf1 = tables["south_fork_1"]
    gap_start = sf1.index[200]
    gap_end = gap_start + pd.Timedelta(hours=6)
    tables["south_fork_1"] = sf1[(sf1.index < gap_start) | (sf1.index >= gap_end)]
    judged = _judged(tables, weather)
    # nothing in the gap or the 24 steps after it, judged again from then on
    waiting = (judged >= gap_start) & (judged < gap_end + pd.Timedelta(minutes=15 * 24))
    assert not waiting.any()
    assert (judged >= gap_end + pd.Timedelta(minutes=15 * 24)).any()
    assert (judged < gap_start).any()


def test_a_gap_the_fill_covers_does_not_stop_judging():
    tables, weather = _spill_tables()
    sf1 = tables["south_fork_1"]
    gap_start = sf1.index[200]
    gap_end = gap_start + pd.Timedelta(hours=2)
    full = _judged(tables, weather)
    tables["south_fork_1"] = sf1[(sf1.index < gap_start) | (sf1.index >= gap_end)]
    judged = _judged(tables, weather)
    after = gap_end + pd.Timedelta(minutes=15)
    assert after in judged and after in full

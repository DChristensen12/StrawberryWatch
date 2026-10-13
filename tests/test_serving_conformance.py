"""
Every model in the serving registry, held to what the Night Heron runner relies
on: it loads from a bare folder, reads raw site tables, hands back Findings, and
shrugs off a site or a column that is not there. A model passing this can be
named in GNN_MODELS. Whether it is any good is graded elsewhere.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import paths
from strawberrywatch.integrations.night_heron import gnn_alerts
from strawberrywatch.preprocessing import node_windows as nw
from strawberrywatch.serving import (
    LEAVE_OUT,
    SERVING_REGISTRY,
    DetectorError,
    Finding,
    Score,
    notice,
    site_choice,
)

# Every site either model reads reports for all four days of this one
EVENT = "anomaly_2025_09_10_overnight_sf"
WEATHER = ["rain_mm", "air_temp_c", "shortwave_radiation"]


def _event(folder=EVENT):
    """One event folder as Night Heron would hand it over, plus the weather beside it."""
    tables, weather = {}, []
    for path in sorted((paths.anomalies_dir() / folder).glob("*.csv")):
        df = pd.read_csv(path)
        df.index = pd.to_datetime(df.pop("DateTimeUTC"), utc=True)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        weather.append(df[[c for c in WEATHER if c in df.columns]])
        # Their tables carry no weather, and footbridge is scnf010 to them
        tables[nw.SITE_TO_TABLE.get(path.stem, path.stem)] = df.drop(columns=WEATHER)
    weather = pd.concat(weather)
    weather.index = weather.index.floor("15min")
    return tables, weather.groupby(level=0).mean()


def _spiked(tables, table="south_fork_1", rows=6, by=1500.0):
    """A conductivity step on one site's newest readings, every table trimmed to end with it."""
    end = tables[table].index.max()
    out = {name: df[df.index <= end].copy() for name, df in tables.items()}
    out[table].iloc[-rows:, out[table].columns.get_loc("Meter_Hydros21_Cond")] += by
    return out


def _well_formed(finding, detector, tables):
    assert isinstance(finding, Finding)
    assert finding.site in tables and finding.site in detector.table_names
    assert finding.variable in nw.VARIABLE_MAP
    assert finding.rule in detector.RULES
    assert np.isfinite(finding.peak) and finding.peak > finding.threshold
    assert isinstance(finding.when, pd.Timestamp) and str(finding.when.tz) == "UTC"
    assert len(finding.readings) and pd.api.types.is_numeric_dtype(finding.readings)
    assert finding.readings.notna().all()


@pytest.fixture(scope="module", params=sorted(SERVING_REGISTRY))
def detector(request, tmp_path_factory):
    """Each registered model, loaded from a folder holding only the files it says it needs."""
    cls = SERVING_REGISTRY[request.param]
    bare = tmp_path_factory.mktemp(request.param)
    for name in cls.FILES:
        source = paths.checkpoints_dir() / name
        if not source.exists():
            pytest.skip(f"no {name} in checkpoints/")
        shutil.copy(source, bare / name)
    # The fallback points at an empty folder, so a detector that quietly reached
    # for the repo's checkpoints instead of the one it was handed fails here
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STRAWBERRYWATCH_CHECKPOINTS_DIR", str(bare / "not_here"))
        return cls.load(bare)


def test_declares_what_the_runner_reads(detector):
    assert SERVING_REGISTRY[detector.name] is type(detector)
    assert detector.RULES and detector.FILES and detector.table_names
    assert isinstance(detector.wants_weather, bool)


def test_every_alert_type_fits_their_column(detector):
    """AlertEvent.event_type is a CharField(max_length=50). Longer and their insert fails."""
    for rule in detector.RULES:
        assert len(gnn_alerts._alert_type(detector.name, rule)) <= 50


def test_an_obvious_step_comes_back_as_a_finding(detector):
    """
    Plumbing, not grading. A 1500 uS/cm step on the newest readings is something
    every model here should say, so an empty answer means the tables never made
    it as far as the model.
    """
    tables, weather = _event()
    found = detector.findings(_spiked(tables), weather)
    assert ("south_fork_1", "conductivity") in {(f.site, f.variable) for f in found}
    for finding in found:
        _well_formed(finding, detector, tables)


def test_a_real_event_runs_clean(detector):
    """Whatever it makes of the event as recorded, it answers in Findings without raising."""
    tables, weather = _event()
    for finding in detector.findings(tables, weather):
        _well_formed(finding, detector, tables)


def test_missing_sites_and_columns_are_not_errors(detector):
    tables, weather = _event()
    assert detector.findings({}, weather) == []
    assert detector.findings({"codornices": tables["codornices"]}, None) == []
    assert detector.findings({name: df.iloc[:0] for name, df in tables.items()}, weather) == []
    assert isinstance(detector.findings({"south_fork_1": tables["south_fork_1"]}, None), list)
    no_cond = {name: df.drop(columns="Meter_Hydros21_Cond") for name, df in tables.items()}
    assert isinstance(detector.findings(no_cond, weather), list)


def test_the_callers_tables_come_back_untouched(detector):
    """The runner hands every model the same frames, so an edit here would feed the next model."""
    tables, weather = _event()
    spiked = _spiked(tables)
    before = {name: df.copy() for name, df in spiked.items()}
    detector.findings(spiked, weather)
    for name, df in before.items():
        pd.testing.assert_frame_equal(spiked[name], df)


def test_naive_stamps_are_read_as_utc(detector):
    """How Night Heron's DATETIME columns hold them."""
    tables, weather = _event()
    spiked = _spiked(tables)
    naive = {name: df.tz_localize(None) for name, df in spiked.items()}

    def key(found):
        return [(f.site, f.variable, f.rule, f.peak, f.when) for f in found]

    assert key(detector.findings(naive, weather)) == key(detector.findings(spiked, weather))


def test_a_table_not_indexed_by_time_is_refused(detector):
    """A RangeIndex read as nanoseconds is 1970, so it has to be an error rather than a parse."""
    tables, _ = _event()
    name = detector.table_names[0]
    with pytest.raises(DetectorError, match="DatetimeIndex"):
        detector.findings({name: tables[name].reset_index(drop=True)}, None)


def _key(found):
    return sorted((f.site, f.variable, f.rule, f.when) for f in found)


def test_scores_agree_with_findings_at_a_scale_of_one(detector):
    """
    notice() over scores() is what a dashboard decides from, so at the tested
    line it has to fire exactly where findings() does, and first cross at the
    same step.
    """
    tables, weather = _event()
    for case in (tables, _spiked(tables)):
        scored = detector.scores(case, weather)
        assert scored, "nothing scored, so the comparison would pass on nothing"
        assert _key(notice(scored, detector.min_steps, 1.0)) == _key(
            detector.findings(case, weather)
        )


def test_scores_are_well_formed(detector):
    tables, weather = _event()
    for row in detector.scores(_spiked(tables), weather):
        assert isinstance(row, Score)
        assert row.site in detector.table_names and row.variable in nw.VARIABLE_MAP
        assert row.rule in detector.RULES
        assert np.isfinite(row.score) and np.isfinite(row.threshold)
        assert isinstance(row.time, pd.Timestamp) and str(row.time.tz) == "UTC"


def test_a_higher_line_never_notices_more(detector):
    tables, weather = _event()
    scored = detector.scores(_spiked(tables), weather)
    counts = [len(notice(scored, detector.min_steps, scale)) for scale in (0.8, 1.0, 1.25)]
    assert counts == sorted(counts, reverse=True)


def test_left_out_sites_never_appear(detector):
    if LEAVE_OUT not in site_choice(detector):
        pytest.skip(f"{detector.name} can't leave sites out")
    tables, weather = _event()
    spiked = _spiked(tables)
    left = {"south_fork_1"}
    assert all(row.site not in left for row in detector.scores(spiked, weather, leave_out=left))
    assert all(f.site not in left for f in detector.findings(spiked, weather, leave_out=left))


def test_a_model_without_leave_out_raises_when_handed_one(detector):
    """Ignoring it quietly would go on reading a site somebody asked it not to."""
    if LEAVE_OUT in site_choice(detector):
        pytest.skip(f"{detector.name} can leave sites out")
    tables, weather = _event()
    with pytest.raises(DetectorError, match="can't leave sites out"):
        detector.findings(tables, weather, leave_out={"oxford"})
    with pytest.raises(DetectorError, match="can't leave sites out"):
        detector.scores(tables, weather, leave_out={"oxford"})
    # An empty one is no request at all
    assert isinstance(detector.findings(tables, weather, leave_out=set()), list)

"""A checkpoint's rain from one Open-Meteo model, with every other model keeping the default's."""

import numpy as np
import pandas as pd
import pytest

from strawberrywatch import paths
from strawberrywatch.ingest import data_loader, weather_archive
from strawberrywatch.ingest import historical_weather_client as client
from strawberrywatch.serving import checkpoint as ck
from strawberrywatch.serving import detector as det

NBM = "ncep_nbm_conus"
SAVED = paths.project_root() / "checkpoints" / "saved" / "dusk_crayfish"


class Answer:
    def __init__(self, payload):
        self.status_code, self._payload, self.text = 200, payload, ""

    def json(self):
        return self._payload


class Session:
    """Open-Meteo faked: each model's rain is a different constant."""

    def __init__(self, rain=None):
        self.asked = []
        self.rain = rain or {None: 1.0, NBM: 0.25}

    def get(self, url, params, timeout):
        self.asked.append(dict(params))
        times = pd.date_range(params["start_date"], periods=8, freq="15min").strftime(
            "%Y-%m-%dT%H:%M"
        )
        rain = self.rain.get(params.get("models"))
        if rain is None:
            return Answer({})
        return Answer(
            {
                "minutely_15": {
                    "time": list(times),
                    "precipitation": [rain] * 8,
                    "temperature_2m": [10.0] * 8,
                    "shortwave_radiation": [0.0] * 8,
                }
            }
        )


# Fetching


def test_the_fetcher_names_a_model_only_when_asked():
    session = Session()
    default = client.fetch_open_meteo_weather("2026-11-13", "2026-11-13", session=session)
    nbm = client.fetch_open_meteo_weather("2026-11-13", "2026-11-13", session=session, models=NBM)
    assert "models" not in session.asked[0] and session.asked[1]["models"] == NBM
    assert (default["rain_mm"] == 1.0).all() and (nbm["rain_mm"] == 0.25).all()


def test_model_rain_arrives_in_its_own_column_and_leaves_rain_mm_alone():
    session = Session()
    weather = client.fetch_open_meteo_weather("2026-11-13", "2026-11-13", session=session)
    out = client.add_model_rain(weather, "2026-11-13", "2026-11-13", session=session)
    assert (out["rain_mm"] == 1.0).all() and (out["rain_nbm_mm"] == 0.25).all()
    assert (out["air_temp_c"] == 10.0).all()

    # a model that sends nothing back leaves its column out, never zeros
    quiet = client.add_model_rain(weather, "2026-11-13", "2026-11-13", session=Session({None: 1.0}))
    assert "rain_nbm_mm" not in quiet.columns and quiet.equals(weather)


def test_the_archive_swaps_in_one_models_rain_a_month_at_a_time(tmp_path, monkeypatch):
    asked = []

    def fetch(lo, hi, models=None, **kw):
        asked.append(models)
        # named as the real fetch names it
        index = pd.date_range(
            lo, pd.Timestamp(hi) + pd.Timedelta(hours=23, minutes=45), freq="15min", name="datetime"
        )
        rain = 0.25 if models == NBM else 1.0
        return pd.DataFrame(
            {"air_temp_c": 10.0, "rain_mm": rain, "shortwave_radiation": 0.0}, index=index
        )

    monkeypatch.setattr(weather_archive, "fetch_open_meteo_weather", fetch)
    start, end = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-02-28 23:45", tz="UTC")
    weather = weather_archive.load_weather(start, end, cache_dir=tmp_path, rain_model=NBM)
    assert (weather["rain_mm"] == 0.25).all() and (weather["air_temp_c"] == 10.0).all()
    assert sorted(p.name for p in tmp_path.glob(f"rain_{NBM}_*.csv")) == [
        f"rain_{NBM}_2026-01-01_2026-01-31.csv",
        f"rain_{NBM}_2026-02-01_2026-02-28.csv",
    ]
    # the default model's monthly files are untouched by it
    assert len(list(tmp_path.glob("rain_2026-*.csv"))) == 2

    # read back from the cache, and refused rather than fetched when fetching is off
    asked.clear()
    again = weather_archive.model_rain(NBM, start, end, cache_dir=tmp_path, allow_fetch=False)
    assert asked == [] and (again == 0.25).all() and len(again) == len(weather)
    with pytest.raises(RuntimeError):
        weather_archive.model_rain(
            NBM, start, end + pd.Timedelta(days=31), cache_dir=tmp_path, allow_fetch=False
        )


def test_training_on_a_models_rain_swaps_the_corpus_rain(tmp_path, monkeypatch):
    stamps = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    corpus = pd.DataFrame(
        {
            "datetime": stamps,
            "south_fork_1_conductivity": 600.0,
            "south_fork_1_depth": 60.0,
            "south_fork_1_valid": True,
            "rain_mm": 1.0,
            "air_temp_c": 10.0,
        }
    )
    path = tmp_path / "corpus.csv"
    corpus.to_csv(path, index=False)
    nbm = pd.Series(np.arange(6) / 10, index=stamps)
    monkeypatch.setattr(weather_archive, "model_rain", lambda model, lo, hi: nbm)

    _, df, _ = data_loader.load_and_preprocess_data(file_path=str(path), rain_model=NBM)
    assert df["rain_mm"].tolist() == nbm.tolist() and (df["air_temp_c"] == 10.0).all()
    _, df, _ = data_loader.load_and_preprocess_data(file_path=str(path))
    assert (df["rain_mm"] == 1.0).all()


# Checkpoints


def _checkpoint(rain_model=None):
    return ck.Checkpoint(
        "w.pt",
        ["conductivity", "rain_mm"],
        {"south_fork_1": 0},
        [600.0, 0.01],
        [100.0, 0.1],
        {"node_thresholds": {"south_fork_1": 10.0}},
        rain_model=rain_model,
    )


def test_a_checkpoint_names_its_rain_only_when_it_has_one():
    assert "rain_model" not in _checkpoint().as_dict()["architecture"]
    blob = _checkpoint(NBM).as_dict()
    assert blob["architecture"]["rain_model"] == NBM
    loaded = ck._from_blob(blob, "w.pt", "test")
    assert loaded.rain_model == NBM and "rain_model" not in loaded.architecture
    # the metadata pickle keeps it beside architecture
    assert ck._from_blob({**blob, "architecture": {}, "rain_model": NBM}, "w.pt", "x").rain_model


def test_a_sidecar_newer_than_this_code_is_refused():
    blob = _checkpoint(NBM).as_dict()
    assert blob["format_version"] == ck.FORMAT_VERSION
    # sidecars from before format_version are 1
    del blob["format_version"]
    assert ck._from_blob(blob, "w.pt", "old").rain_model == NBM
    with pytest.raises(ck.CheckpointError, match="needs a newer StrawberryWatch"):
        ck._from_blob({**blob, "format_version": ck.FORMAT_VERSION + 1}, "w.pt", "new")


def test_code_from_before_rain_model_cant_build_an_nbm_checkpoint():
    """Older code passes architecture to the model class whole, so it refuses the files."""
    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

    architecture = _checkpoint(NBM).as_dict()["architecture"]
    with pytest.raises(TypeError, match="rain_model"):
        DuskCrayfish(num_node_features=2, num_nodes=1, **architecture)


def test_dusk_crayfish_reads_its_checkpoints_rain_column():
    index = pd.date_range("2026-11-13", periods=4, freq="15min", tz="UTC")
    weather = pd.DataFrame({"rain_mm": 1.0, "rain_nbm_mm": 0.25, "air_temp_c": 10.0}, index=index)
    assert det._own_rain(weather, None) is weather
    own = det._own_rain(weather, NBM)
    assert (own["rain_mm"] == 0.25).all() and "rain_nbm_mm" not in own.columns
    # without NBM's column there is no rain at all, never the default model's
    alone = det._own_rain(weather.drop(columns="rain_nbm_mm"), NBM)
    assert "rain_mm" not in alone.columns and (alone["air_temp_c"] == 10.0).all()


def _nbm_detector():
    folders = sorted(SAVED.glob("*_nbm-rain-seed-0"))
    if not folders:
        pytest.skip("no saved NBM checkpoint")
    from strawberrywatch.serving import detector_class

    return detector_class("dusk_crayfish").load(folders[-1])


def _fixture():
    from strawberrywatch.watch import sources

    replay = sources.ReplaySource("anomaly_2025_11_13_rain_nf1")
    start, end = replay.window(None)
    return replay.tables(list(replay._csv._all()), start, end), replay.weather(start, end)


def test_an_nbm_checkpoint_scores_on_nbm_rain_and_ignores_the_default():
    detector = _nbm_detector()
    assert detector.checkpoint.rain_model == NBM
    tables, weather = _fixture()
    assert weather["rain_nbm_mm"].notna().all()

    def scored(w):
        return [(s.site, s.time, s.score, s.threshold) for s in detector.scores(tables, w)]

    base = scored(weather)
    # the default model's rain makes no difference to it
    assert scored(weather.assign(rain_mm=weather["rain_mm"] * 5 + 3)) == base
    # its own does
    assert scored(weather.assign(rain_nbm_mm=0.0)) != base


def test_missing_rain_is_no_rain_not_the_training_mean():
    detector = _nbm_detector()
    tables, weather = _fixture()
    _, with_rain = detector._judge(tables, weather)
    _, without = detector._judge(tables, weather.drop(columns="rain_nbm_mm"))
    assert with_rain["rain_applied"] and not without["rain_applied"]

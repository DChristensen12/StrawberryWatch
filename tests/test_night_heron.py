"""
The Night Heron runner, with their database and Open-Meteo swapped for fakes.

Checks the runner's half of the deal: the dict their fire_alerts_task takes,
the cooldown, one model failing without taking the rest down, and each table
read once however many models want it. The last test runs both real models
through the whole pass on an event fixture.
"""

import threading
import uuid
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import torch

from strawberrywatch import paths
from strawberrywatch.ingest import historical_weather_client, sql_client
from strawberrywatch.integrations.night_heron import gnn_alerts
from strawberrywatch.serving import Finding

NOW = datetime(2026, 9, 3, 22, 0, tzinfo=UTC)


class _Fake:
    """A detector that fires whatever it was built with."""

    def __init__(self, name, tables, found=(), wants_weather=False, boom=False):
        self.name = name
        self.table_names = list(tables)
        self.wants_weather = wants_weather
        self.found = list(found)
        self.boom = boom
        self.seen = None

    def findings(self, tables, weather=None):
        if self.boom:
            raise RuntimeError("bad day")
        self.seen = (tables, weather)
        return list(self.found)


def _finding(site="north_fork_0", variable="conductivity", rule="forecast_residual"):
    readings = pd.Series(range(300), dtype=float)
    return Finding(site, variable, rule, 9.0, 4.0, pd.Timestamp(NOW), readings)


def _use(monkeypatch, *detectors):
    monkeypatch.setattr(gnn_alerts, "_detectors", lambda: list(detectors))


@pytest.fixture
def reads(monkeypatch):
    """The runner with its MySQL read and weather stubbed, throttles cleared. Yields the reads."""
    reads = []

    def read(names, start, end):
        reads.append(list(names))
        stamps = pd.date_range(end=end, periods=4, freq="15min")
        return {
            name: pd.DataFrame({"Meter_Hydros21_Cond": [1.0, 2, 3, 4]}, stamps) for name in names
        }

    monkeypatch.setattr(gnn_alerts, "_read_tables", read)
    monkeypatch.setattr(gnn_alerts, "_weather", lambda start, end: pd.DataFrame({"rain_mm": [0.0]}))
    monkeypatch.setattr(gnn_alerts, "ENABLED", True)
    monkeypatch.setenv("GNN_ALERT_EMAILS", "a@example.com, b@example.com")
    monkeypatch.delenv("GNN_ALERT_PHONES", raising=False)
    # _score takes torch down to one thread for the daemon's sake. Put it back,
    # or every test after this one runs single threaded.
    threads = torch.get_num_threads()
    gnn_alerts.reset_state()
    yield reads
    gnn_alerts.reset_state()
    torch.set_num_threads(threads)


def test_alerts_are_shaped_for_fire_alerts_task(reads, monkeypatch):
    _use(monkeypatch, _Fake("dusk_crayfish", ["north_fork_0"], [_finding()]))
    [alert] = gnn_alerts._score(NOW)
    assert set(alert) == {"values", "site", "sensor", "alert_type", "emails", "phones"}
    assert alert["values"] == [float(v) for v in range(100, 300)]
    assert (alert["site"], alert["sensor"]) == ("north_fork_0", "conductivity")
    assert alert["alert_type"] == "gnn_anomaly_forecast_residual"
    assert alert["emails"] == ["a@example.com", "b@example.com"]
    assert alert["phones"] == []


def test_later_models_alert_under_their_own_name_and_night_herons_sensors(reads, monkeypatch):
    found = [_finding("scnf010", "dissolved_oxygen", "combined_fisher")]
    _use(monkeypatch, _Fake("cobble_shoal", ["scnf010"], found))
    [alert] = gnn_alerts._score(NOW)
    assert alert["alert_type"] == "cobble_shoal_combined_fisher"
    assert (alert["site"], alert["sensor"]) == ("scnf010", "AtlasSci_DO")


def test_cooldown_is_per_model_site_sensor_and_rule(reads, monkeypatch):
    dusk = _Fake("dusk_crayfish", ["north_fork_0"], [_finding()])
    cobble = _Fake("cobble_shoal", ["north_fork_0"], [_finding()])
    _use(monkeypatch, dusk, cobble)

    assert len(gnn_alerts._score(NOW)) == 2, "same site and rule, different models"
    assert gnn_alerts._score(NOW + timedelta(minutes=15)) == []

    dusk.found.append(_finding(variable="temperature"))
    later = gnn_alerts._score(NOW + timedelta(minutes=30))
    assert [a["sensor"] for a in later] == ["temperature"]

    after = gnn_alerts._score(NOW + gnn_alerts.COOLDOWN + timedelta(minutes=1))
    assert sorted(a["alert_type"] for a in after) == [
        "cobble_shoal_forecast_residual",
        "gnn_anomaly_forecast_residual",
    ]


def test_one_model_failing_leaves_the_others(reads, monkeypatch, caplog):
    _use(
        monkeypatch,
        _Fake("cobble_shoal", ["oxford"], boom=True),
        _Fake("dusk_crayfish", ["oxford"], [_finding("oxford")]),
    )
    assert [a["site"] for a in gnn_alerts._score(NOW)] == ["oxford"]
    assert "cobble_shoal failed this pass" in caplog.text


def test_each_table_is_read_once_a_pass(reads, monkeypatch):
    dusk = _Fake("dusk_crayfish", ["north_fork_0", "oxford"])
    cobble = _Fake("cobble_shoal", ["north_fork_0", "scnf010", "oxford"])
    _use(monkeypatch, dusk, cobble)
    gnn_alerts._score(NOW)
    assert reads == [["north_fork_0", "oxford", "scnf010"]]
    assert set(dusk.seen[0]) == {"north_fork_0", "oxford"}
    assert set(cobble.seen[0]) == {"north_fork_0", "scnf010", "oxford"}


def test_weather_is_fetched_only_when_a_model_wants_it(reads, monkeypatch):
    fetched = []
    monkeypatch.setattr(gnn_alerts, "_weather", lambda start, end: fetched.append(1))
    _use(monkeypatch, _Fake("dusk_crayfish", ["oxford"]))
    gnn_alerts._score(NOW)
    assert fetched == []
    _use(
        monkeypatch, _Fake("dusk_crayfish", ["oxford"]), _Fake("x", ["oxford"], wants_weather=True)
    )
    gnn_alerts._score(NOW)
    assert fetched == [1]


def test_pending_alerts_never_waits_on_the_models(reads, monkeypatch):
    """Their loop pings a systemd watchdog every pass, so a slow model cannot hold it up."""
    release = threading.Event()

    class Slow(_Fake):
        def findings(self, tables, weather=None):
            release.wait(10)
            return super().findings(tables, weather)

    _use(monkeypatch, Slow("dusk_crayfish", ["oxford"], [_finding("oxford")]))
    assert gnn_alerts.pending_alerts(NOW) == []
    assert gnn_alerts.pending_alerts(NOW + timedelta(seconds=20)) == []

    release.set()
    gnn_alerts._worker.join(10)
    [alert] = gnn_alerts.pending_alerts(NOW + timedelta(seconds=40))
    assert alert["site"] == "oxford"
    assert gnn_alerts.pending_alerts(NOW + timedelta(seconds=60)) == [], "handed over once"


def test_the_model_list_reads_gnn_models_then_the_old_name(monkeypatch):
    monkeypatch.setenv("GNN_MODELS", " Dusk_Crayfish, cobble_shoal,,dusk_crayfish ")
    monkeypatch.setenv("GNN_MODEL_NAME", "ignored")
    assert gnn_alerts._model_names() == ["dusk_crayfish", "cobble_shoal"]

    monkeypatch.delenv("GNN_MODELS")
    monkeypatch.setenv("GNN_MODEL_NAME", "cobble_shoal")
    assert gnn_alerts._model_names() == ["cobble_shoal"]

    monkeypatch.delenv("GNN_MODEL_NAME")
    assert gnn_alerts._model_names() == ["dusk_crayfish"]


def test_a_model_that_will_not_load_is_left_out(monkeypatch, caplog):
    monkeypatch.setattr(gnn_alerts, "MODELS", ["no_such_model", "dusk_crayfish"])
    monkeypatch.setattr(gnn_alerts, "CHECKPOINT_DIR", str(paths.checkpoints_dir()))
    loaded = gnn_alerts._detectors()
    assert [d.name for d in loaded] == ["dusk_crayfish"]
    assert "no_such_model would not load" in caplog.text


def _night_heron_tables(folder, spike_site, add, rows):
    """An event folder as sql_client returns it out of their MySQL, weather apart."""
    weather_cols = ["rain_mm", "air_temp_c", "shortwave_radiation"]
    tables, weather = {}, []
    for path in sorted((paths.anomalies_dir() / folder).glob("*.csv")):
        df = pd.read_csv(path)
        stamps = pd.to_datetime(df.pop("DateTimeUTC"), utc=True)
        weather.append(df[weather_cols].set_axis(stamps))
        name = "scnf010" if path.stem == "footbridge" else path.stem
        df = df.drop(columns=weather_cols)
        df.insert(0, "timestamp", stamps)
        df.insert(0, "site_code", name)
        df.insert(0, "uuid", [str(uuid.uuid4()) for _ in range(len(df))])
        df["station_id"] = name
        tables[name] = df.sort_values("timestamp").reset_index(drop=True)
    spiked = tables[spike_site]
    spiked.loc[spiked.index[-rows:], "Meter_Hydros21_Cond"] += add
    weather = pd.concat(weather)
    weather.index = weather.index.floor("15min")
    return tables, weather.groupby(level=0).mean()


def test_both_real_models_through_a_whole_pass(monkeypatch):
    """
    Dusk Crayfish and Cobble Shoal from their checkpoints, an event fixture
    standing in for MySQL, and a step on south_fork_1. Both should say so, each
    under its own alert type.
    """
    tables, weather = _night_heron_tables(
        "anomaly_2025_09_10_overnight_sf", "south_fork_1", 1500.0, 6
    )

    def fetch(site, start, end):
        df = tables.get(site, pd.DataFrame())
        if df.empty:
            return df
        keep = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
        return df[keep].reset_index(drop=True)

    monkeypatch.setattr(sql_client, "fetch_creek_data_sql", fetch)
    monkeypatch.setattr(
        historical_weather_client, "fetch_open_meteo_weather", lambda start, end: weather
    )
    monkeypatch.setattr(gnn_alerts, "MODELS", ["dusk_crayfish", "cobble_shoal"])
    monkeypatch.setattr(gnn_alerts, "CHECKPOINT_DIR", str(paths.checkpoints_dir()))
    monkeypatch.setenv("GNN_ALERT_EMAILS", "a@example.com")

    threads = torch.get_num_threads()
    gnn_alerts.reset_state()
    try:
        now = tables["south_fork_1"]["timestamp"].max().to_pydatetime() + timedelta(minutes=10)
        alerts = gnn_alerts._score(now)
    finally:
        gnn_alerts.reset_state()
        torch.set_num_threads(threads)

    flagged = {(a["site"], a["sensor"], a["alert_type"]) for a in alerts}
    assert ("south_fork_1", "conductivity", "gnn_anomaly_level_shift") in flagged
    assert ("south_fork_1", "conductivity", "cobble_shoal_combined_fisher") in flagged
    for alert in alerts:
        assert alert["values"] and all(isinstance(v, float) for v in alert["values"])
        assert len(alert["alert_type"]) <= 50

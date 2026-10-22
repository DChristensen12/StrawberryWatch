"""
Each model's minutes between alerts on the website: after a model alerts, its
new alerts are held until the time is up, the way Night Heron switches a rule
off until its reactivate_at.
"""

import pytest

from strawberrywatch.watch import check_once
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from tests.test_overview_alerts_settings import _csrf, _site_store, _website

M = 60
T0 = 1_800_000_000.0


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _alert(store, model, at, site="south_fork_1"):
    """One new flagged item from its own check, the way a check records one."""
    check = store.start_check("site", 1, at - 3600, at, started_at=at)
    store.finish_check(check, True, "", {})
    store.add_noticed(
        check,
        {
            "model": model,
            "site": site,
            "variable": "conductivity",
            "rule": "forecast_residual",
            "first_over": at - 900,
            "last_over": at,
            "peak": 9.0,
            "base": 3.0,
        },
        True,
        {},
        1.0,
    )
    return check


def _sent(store, check, at):
    return [n["model"] for n in check_once.sendable(store, check, clock=lambda: at)]


def test_a_model_sends_then_holds_its_alerts_until_its_minutes_are_up(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    assert _sent(store, _alert(store, "dusk_crayfish", T0), T0) == ["dusk_crayfish"]
    # The next three checks, 15 minutes apart, are inside the hour
    for k in (1, 2, 3):
        at = T0 + 15 * k * M
        assert _sent(store, _alert(store, "dusk_crayfish", at), at) == []
    # Held alerts are still on the dashboard
    assert len(store.noticed(flagged=True)) == 4
    at = T0 + 60 * M
    assert _sent(store, _alert(store, "dusk_crayfish", at), at) == ["dusk_crayfish"]


def test_each_model_keeps_its_own_time(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    _sent(store, _alert(store, "dusk_crayfish", T0), T0)
    at = T0 + 15 * M
    assert _sent(store, _alert(store, "water_strider", at), at) == ["water_strider"]


def test_the_minutes_come_from_the_applied_settings(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    settings = store.applied()[1]
    settings["models"]["dusk_crayfish"]["alert_gap_minutes"] = 180.0
    store.apply(settings)
    _sent(store, _alert(store, "dusk_crayfish", T0), T0)
    at = T0 + 120 * M
    assert _sent(store, _alert(store, "dusk_crayfish", at), at) == []
    at = T0 + 180 * M
    assert _sent(store, _alert(store, "dusk_crayfish", at), at) == ["dusk_crayfish"]


def test_the_setting_has_a_range_a_default_and_a_history_line(models):
    defaults = ws.defaults(models)
    assert defaults["models"]["dusk_crayfish"]["alert_gap_minutes"] == ws.ALERT_GAP_DEFAULT == 60
    sites = ["south_fork_1"]
    too_short = ws.defaults(models)
    too_short["models"]["dusk_crayfish"]["alert_gap_minutes"] = 5
    _, problems, _ = ws.check(too_short, models, sites)
    assert "Dusk Crayfish minutes between alerts must be between 15 and 1440" in problems
    # Settings saved before there was a gap get the default
    old = ws.defaults(models)
    del old["models"]["dusk_crayfish"]["alert_gap_minutes"]
    clean, problems, _ = ws.check(old, models, sites)
    assert not problems and clean["models"]["dusk_crayfish"]["alert_gap_minutes"] == 60
    after = ws.defaults(models)
    after["models"]["dusk_crayfish"]["alert_gap_minutes"] = 120.0
    sentences = [r["sentence"] for r in ws.describe(ws.defaults(models), after, models)]
    assert sentences == ["Dusk Crayfish minutes between alerts set to 120."]


def test_admins_set_it_in_advanced_settings_on_the_website(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    client = _website(store, models)
    page = client.get("/strawberrywatch/settings/advanced").data.decode()
    assert "Minutes between alerts" in page and 'name="gap_dusk_crayfish"' in page
    data = {"csrf_token": _csrf(page), "repeat_hours": "6"}
    applied = store.applied()[1]
    for m in models:
        spec = applied["models"][m.name]
        data[f"model_{m.name}"] = "on" if spec["on"] else "off"
        data[f"rain_{m.name}"] = str(spec["rain_multiplier"])
        data[f"gap_{m.name}"] = "60"
        data[f"support_{m.name}"] = spec["support"]
        if "leave_out" in m.site_choice:
            data[f"leave_{m.name}"] = spec["leave_out"]
        for rule in m.rules if m.sensitivity == "scale" else ():
            data[f"scale_{m.name}_{rule}"] = str(spec["scales"][rule])
    data["gap_dusk_crayfish"] = "120"
    client.post("/strawberrywatch/settings/advanced", data=data)
    assert store.draft("ana")["models"]["dusk_crayfish"]["alert_gap_minutes"] == 120.0
    review = client.get("/strawberrywatch/settings/review").data.decode()
    assert "Dusk Crayfish, minutes between alerts" in review


def test_the_local_dashboard_has_no_minutes_between_alerts(models, tmp_path):
    from tests.test_dashboard_changes import _client, _store

    store = _store(tmp_path / "s", models)
    page = _client(store, models).get("/settings/advanced").data.decode()
    assert "Minutes between alerts" not in page

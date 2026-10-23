"""
Each person's alerts on the website: which models they hear about, and
pausing them for a while or until someone turns them back on.
"""

import sqlite3
import time
from datetime import date

import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.dashboard import mail, view
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import store as st
from tests.test_alert_gaps import _alert
from tests.test_dashboard_changes import NOW, _client, _store
from tests.test_overview_alerts_settings import _csrf, _site_store, _website
from tests.test_website_load import _fake_mail

DAY = 86400


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _person(store, address="ana@example.org", name=None):
    store.add_recipient(address, name, store.label_named("SCMG")["id"])
    return next(p for p in store.recipients() if p["email"] == address)


def _page(client, person):
    return client.get(f"/strawberrywatch/settings/email/people/{person['id']}")


def _save(client, person, models, pause, pause_date=""):
    url = f"/strawberrywatch/settings/email/people/{person['id']}"
    page = client.get(url).data.decode()
    data = {
        "csrf_token": _csrf(page),
        "name": person.get("name") or "",
        "models": models,
        "pause": pause,
        "pause_date": pause_date,
    }
    return client.post(url, data=data)


def test_pauses_lapse_on_their_own_and_open_ended_ones_dont(models):
    items = [{"model": "dusk_crayfish"}, {"model": "water_strider"}]
    everyone = {"models": None, "paused_at": None, "paused_until": None}
    assert mail.picked(everyone, items, NOW) == [0, 1]
    assert mail.picked({**everyone, "models": '["water_strider"]'}, items, NOW) == [1]
    week = {**everyone, "paused_at": NOW, "paused_until": NOW + 7 * DAY}
    assert mail.picked(week, items, NOW + DAY) == []
    assert mail.picked(week, items, NOW + 7 * DAY) == [0, 1]
    always = {**everyone, "paused_at": NOW}
    assert mail.picked(always, items, NOW + 365 * DAY) == []


def test_old_files_get_the_new_columns(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    db = sqlite3.connect(folder / st.DB_NAME)
    db.execute(
        "CREATE TABLE recipients (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE, "
        "name TEXT, added_at REAL NOT NULL)"
    )
    db.execute("INSERT INTO recipients (email, added_at) VALUES ('ana@example.org', 1.0)")
    db.commit()
    db.close()
    store = st.Store(folder)
    person = store.recipients()[0]
    assert mail.models_for(person) is None and not mail.paused(person, NOW)
    store.set_models(person["id"], ["water_strider"])
    store.pause(person["id"], NOW)
    saved = store.recipient(person["id"])
    assert mail.models_for(saved) == ["water_strider"] and mail.paused(saved, NOW + 365 * DAY)


def test_each_person_is_emailed_only_their_models_and_nothing_while_paused(
    models, tmp_path, monkeypatch
):
    sent = _fake_mail(monkeypatch)
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    _person(store, "ana@example.org")
    ben = _person(store, "ben@example.org")
    cy = _person(store, "cy@example.org")
    store.set_models(ben["id"], ["water_strider"])
    store.pause(cy["id"], time.time(), time.time() + DAY)
    check = _alert(store, "dusk_crayfish", NOW)
    _alert(store, "water_strider", NOW)
    items = store.noticed(flagged=True)
    mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch", items)
    by_person = {to[0]: message for message, to in sent}
    assert set(by_person) == {"ana@example.org", "ben@example.org"}
    assert by_person["ana@example.org"]["Subject"] == "StrawberryWatch: 2 alerts"
    ben_text = by_person["ben@example.org"].get_body(("plain",)).get_content()
    assert "Flagged by Water Strider" in ben_text and "Dusk Crayfish" not in ben_text
    assert store.emails()[0]["people"] == 2


def test_nobody_wanting_the_alerts_sends_nothing(models, tmp_path, monkeypatch):
    sent = _fake_mail(monkeypatch)
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    ana = _person(store)
    store.set_models(ana["id"], ["water_strider"])
    check = _alert(store, "dusk_crayfish", NOW)
    mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch")
    assert sent == [] and store.emails() == []


def test_a_name_opens_the_persons_alert_settings(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store, name="Ana")
    page = _website(store, models).get("/strawberrywatch/settings/email").data.decode()
    assert f'<a href="/strawberrywatch/settings/email/people/{ana["id"]}">Ana</a>' in page
    assert "<td>All models</td>" in page
    page = _page(_website(store, models), ana).data.decode()
    assert "<h1>Ana</h1>" in page and "ana@example.org. Labelled SCMG." in page
    for m in models:
        assert f'name="models" value="{m.name}" checked' in page
    assert 'value="off" checked' in page and "Don&#39;t pause" in page


def test_admins_pick_models_and_pause_for_a_week(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    client = _website(store, models)
    response = _save(client, ana, ["water_strider"], "7")
    assert response.status_code == 302
    saved = store.recipient(ana["id"])
    assert mail.models_for(saved) == ["water_strider"]
    assert saved["paused_at"] == NOW and saved["paused_until"] == NOW + 7 * DAY
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Saved alert settings for ana@example.org." in page
    assert "<td>Paused until 20 Oct, 2:30 pm</td>" in page
    # Saving again keeps the pause where it was
    page = _page(client, ana).data.decode()
    assert 'value="keep" checked' in page and "Keep paused until 20 Oct, 2:30 pm" in page
    _save(client, ana, ["water_strider"], "keep")
    assert store.recipient(ana["id"])["paused_until"] == NOW + 7 * DAY
    _save(client, ana, ["water_strider"], "off")
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert f"<td>1 of {len(models)} models</td>" in page


def test_every_model_ticked_means_new_models_too(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    store.set_models(ana["id"], ["water_strider"])
    _save(_website(store, models), ana, [m.name for m in models], "off")
    assert store.recipient(ana["id"])["models"] is None


def test_paused_until_turned_back_on(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    client = _website(store, models)
    _save(client, ana, [m.name for m in models], "always")
    saved = store.recipient(ana["id"])
    assert saved["paused_at"] == NOW and saved["paused_until"] is None
    assert "<td>Paused</td>" in client.get("/strawberrywatch/settings/email").data.decode()
    assert "Keep paused until turned back on" in _page(client, ana).data.decode()
    _save(client, ana, [m.name for m in models], "off")
    assert store.recipient(ana["id"])["paused_at"] is None


def test_paused_until_a_date_starts_again_at_midnight_that_day(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    client = _website(store, models)
    page = _page(client, ana).data.decode()
    assert 'data-sw-reveal-for="pause-5"' in page and 'type="date" id="pause_date"' in page
    _save(client, ana, [m.name for m in models], "date", "2026-11-03")
    until = view.midnight(date(2026, 11, 3))
    assert store.recipient(ana["id"])["paused_until"] == until
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert f"<td>Paused until {view.moment(until, NOW)}</td>" in page


@pytest.mark.parametrize(
    "typed, error",
    [
        ("", "Enter the date alerts start again"),
        ("3 November", "Enter a date like 2026-11-03"),
        ("2026-10-13", "Date must be after today"),
        ("2027-12-01", "Date must be within a year. For longer, choose Until turned back on"),
    ],
)
def test_a_pause_date_must_be_a_day_in_the_coming_year(models, tmp_path, typed, error):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    page = _save(_website(store, models), ana, [m.name for m in models], "date", typed)
    assert error in page.data.decode() and store.recipient(ana["id"])["paused_at"] is None


def test_a_date_is_ignored_unless_until_a_date_is_chosen(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    assert (
        _save(_website(store, models), ana, ["water_strider"], "off", "nonsense").status_code == 302
    )


def test_people_can_pause_themselves_for_a_week_from_the_email_link(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    public = _website(store, models, user=None)
    url = f"/strawberrywatch/unsubscribe/{ana['unsubscribe_token']}"
    page = public.get(url).data.decode()
    assert "Pause alerts for 1 week" in page and store.recipient(ana["id"])["paused_at"] is None
    page = public.post(url, data={"csrf_token": _csrf(page), "action": "pause"}).data.decode()
    saved = store.recipient(ana["id"])
    assert saved["paused_until"] == NOW + 7 * DAY and saved["models"] is None
    assert "Alerts paused" in page and "until 20 Oct, 2:30 pm" in page
    page = public.get(url).data.decode()
    assert "paused until 20 Oct, 2:30 pm" in page and "Pause alerts for 1 week" not in page
    public.post(url, data={"csrf_token": _csrf(page), "action": "unsubscribe"})
    assert store.recipients() == []


def test_a_longer_pause_isnt_cut_to_a_week(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    store.pause(ana["id"], NOW, NOW + 30 * DAY)
    public = _website(store, models, user=None)
    url = f"/strawberrywatch/unsubscribe/{ana['unsubscribe_token']}"
    page = public.get(url).data.decode()
    public.post(url, data={"csrf_token": _csrf(page), "action": "pause"})
    assert store.recipient(ana["id"])["paused_until"] == NOW + 30 * DAY


def test_the_email_footer_offers_the_pause():
    text, html = mail.alert_email("Subject", "New alert.", [], "https://x/unsubscribe/t")
    assert "To pause them for a week or unsubscribe: https://x/unsubscribe/t" in text
    assert ">Pause for a week or unsubscribe</a>" in html


def test_no_models_ticked_is_an_error(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    page = _save(_website(store, models), ana, [], "off").data.decode()
    assert "Tick at least one model, or pause alerts instead" in page
    assert store.recipient(ana["id"])["models"] is None


def test_an_unknown_model_or_pause_isnt_saved(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    client = _website(store, models)
    _save(client, ana, ["not_a_model"], "off")
    _save(client, ana, ["water_strider"], "365")
    saved = store.recipient(ana["id"])
    assert saved["models"] is None and saved["paused_at"] is None


def test_a_person_copied_without_a_name_can_be_given_one(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    client = _website(store, models)
    response = _save(client, {**ana, "name": "Ana Flores"}, [m.name for m in models], "off")
    assert response.status_code == 302
    assert store.recipient(ana["id"])["name"] == "Ana Flores"
    assert "Saved alert settings for Ana Flores." in client.get(response.location).data.decode()
    _save(client, {**ana, "name": ""}, [m.name for m in models], "off")
    assert store.recipient(ana["id"])["name"] is None


def test_view_only_sees_the_choices_and_cant_change_them(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ana = _person(store)
    store.set_models(ana["id"], ["water_strider"])
    viewer = _website(store, models, user="ben", can_edit=False)
    page = _page(viewer, ana).data.decode()
    assert "<dd>Water Strider</dd>" in page and "<dd>On</dd>" in page
    assert 'name="models"' not in page
    response = viewer.post(
        f"/strawberrywatch/settings/email/people/{ana['id']}", data={"pause": "always"}
    )
    assert response.status_code in (400, 403) and store.recipient(ana["id"])["paused_at"] is None


def test_an_unknown_person_is_a_404_and_the_local_app_has_no_such_page(models, tmp_path):
    website = _website(_site_store(tmp_path / "s", models), models)
    assert website.get("/strawberrywatch/settings/email/people/99").status_code == 404
    local = _client(_store(tmp_path / "l", models), models)
    assert local.get("/settings/email/people/1").status_code == 404

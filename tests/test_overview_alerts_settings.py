"""
The models line, hourly reporting marks, Alert examples, the settings rows, and
recipient labels with public sign-up on the website.
"""

import re
from datetime import datetime

import pytest
from werkzeug.test import Client

from strawberrywatch.dashboard import create_app, examples, mail, profiles, view
from strawberrywatch.dashboard.security import SiteLogin
from strawberrywatch.integrations.scmg_website import NoWatcher
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st
from tests.test_dashboard_changes import NOW, _client, _store
from tests.test_scmg_website import PrefixedStore
from tests.test_website_load import _fake_mail

H = 3600


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _csrf(page):
    return re.search(r'name="csrf_token"[^>]*value="([^"]+)"', page).group(1)


def _website(store, models, user="ana", can_edit=True, copy_list=None, updater=None):
    """The website dashboard, signed in as user, or signed out with user=None."""
    app = create_app(
        store=store,
        models=models,
        watcher=NoWatcher(),
        profile=profiles.WEBSITE,
        signing_key="test",
        clock=lambda: NOW,
        copy_list=copy_list,
        updater=updater,
    )
    app.config["PUBLIC_ADDRESS"] = "https://localhost/strawberrywatch"

    def site(environ, start_response):
        if user:
            environ[SiteLogin.USER] = user
            environ[SiteLogin.CAN_EDIT] = can_edit
        environ["SCRIPT_NAME"] = "/strawberrywatch"
        environ["PATH_INFO"] = environ["PATH_INFO"].removeprefix("/strawberrywatch")
        return app(environ, start_response)

    client = Client(site)
    real_get, real_post = client.get, client.post
    client.get = lambda url, **k: real_get(url, base_url="https://localhost", **k)
    client.post = lambda url, **k: real_post(
        url, base_url="https://localhost", headers={"Referer": "https://localhost" + url}, **k
    )
    return client


def _site_store(folder, models):
    store = PrefixedStore(folder)
    store.set_source("site")
    clean, _, _ = ws.check(ws.defaults(models), models, ["south_fork_1", "oxford"])
    store.apply(clean)
    return store


# Overview: the models line


def test_only_running_models_are_named(models, tmp_path):
    store = _store(tmp_path / "s", models)
    applied = store.applied()[1]
    on = [m.title for m in models if applied["models"][m.name]["on"]]
    line = view.model_titles(models, applied)
    assert line == f"Models on: {', '.join(on)}." and "Off:" not in line
    none_on = {**applied, "models": {n: {**s, "on": False} for n, s in applied["models"].items()}}
    assert view.model_titles(models, none_on) == "No models are on."
    page = _client(store, models).get("/").data.decode()
    assert re.search(r'Models on: [^<]*<a href="/models">See all models</a>', page)


# Overview: reporting marks


class Readings:
    def __init__(self, times):
        self.times = times

    def readings(self, site, since, until, live):
        return [{"time": t} for t in self.times if since <= t <= until]


def test_one_mark_per_hour_full_partial_or_empty():
    ref = datetime(2026, 10, 13, 14, 50, tzinfo=view.PACIFIC).timestamp()
    two = datetime(2026, 10, 13, 13, 0, tzinfo=view.PACIFIC).timestamp()
    noon = two - H
    times = [two + q * 900 for q in range(4)] + [noon, noon + 1800]
    marks = view.reporting_marks(Readings(times), "oxford", ref)["marks"]
    assert len(marks) == 24
    by_label = {m["label"].split(":")[0]: m for m in marks}
    assert by_label["1 to 2 pm"] == {
        "state": "full",
        "label": "1 to 2 pm: 4 of 4 readings",
        "count": 4,
    }
    assert by_label["12 to 1 pm"]["state"] == "partial"
    assert by_label["11 am to 12 pm"]["state"] == "empty"
    assert by_label["11 am to 12 pm"]["label"] == "11 am to 12 pm: 0 of 4 readings"
    # Two of the current hour's readings are due by 2:50 and neither came
    assert marks[-1] == {"state": "empty", "label": "2 to 3 pm: 0 of 4 readings so far", "count": 0}
    just_started = view.reporting_marks(Readings([]), "oxford", ref - 45 * 60)["marks"][-1]
    assert just_started["state"] == "pending"
    assert view.hour_span(ref - ref % H + 9 * H) == "11 pm to 12 am"


def test_the_overview_marks_carry_their_own_labels(models, tmp_path):
    store = _store(tmp_path / "s", models)
    page = _client(store, models).get("/").data.decode()
    rows = page.count('class="sw-ticks"')
    assert rows == len(store.applied()[1]["dont_flag"]) or rows > 0
    assert page.count('class="sw-tick sw-tick--') == 24 * rows
    assert re.search(r'role="img" aria-label="\d+ (am|pm)? ?to \d+ (am|pm): 0 of 4 readings"', page)
    # Only the newest mark is in the tab order, the arrow keys reach the rest
    assert page.count('tabindex="0"></span>') == rows
    assert 'data-sw-chart="strip"' not in page


# Alerts: the examples page


def test_alert_examples_shows_every_sentence_everywhere(models, tmp_path):
    store = _store(tmp_path / "s", models)
    client = _client(store, models)
    page = client.get("/alerts/examples").data.decode()
    samples = examples.items(NOW)
    for s in samples:
        sentence = s["view"]["happened"].replace("'", "&#39;")
        # Alerts tab, alert page and email
        assert page.count(sentence) >= 3, s["name"]
    for _, _, text, _ in examples.sentences(NOW, samples):
        assert (
            text.replace("'", "&#39;").replace("&", "&amp;").replace("&amp;#39;", "&#39;") in page
        )
    assert "Proposed" in page and "stopped matching the model&#39;s forecast" in page
    assert "No alerts in the last 24 hours." in page and "Waiting for the first check" in page
    assert "StrawberryWatch: 3 alerts" in page
    assert "Flagged by Cobble Shoal." in page and "Flagged by Water Strider." in page
    assert '<a href="/alerts/examples">Alert examples</a>' in client.get("/unusual").data.decode()


def test_the_example_emails_open_on_their_own(models, tmp_path):
    store = _store(tmp_path / "s", models)
    client = _client(store, models)
    response = client.get("/alerts/examples/email/1")
    html = response.data.decode()
    assert response.status_code == 200 and "/static/email/emblem.png" in html
    assert "/static/email/sponsors.png" in html and "Unsubscribe" in html
    assert "style-src 'unsafe-inline'" in response.headers["Content-Security-Policy"]
    assert client.get("/alerts/examples/email/several").status_code == 200
    assert client.get("/alerts/examples/email/99").status_code == 404
    # Every other page keeps the strict policy
    assert "unsafe-inline" not in client.get("/").headers["Content-Security-Policy"]


def test_alert_examples_needs_sign_in_on_the_website(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    assert (
        _website(store, models, user=None).get("/strawberrywatch/alerts/examples").status_code
        == 403
    )
    page = _website(store, models).get("/strawberrywatch/alerts/examples").data.decode()
    assert "Sign-up confirmation" in page and "Confirm your StrawberryWatch alerts" in page


# Settings: the rows


def test_settings_links_are_full_rows_without_badges(models, tmp_path):
    store = _store(tmp_path / "s", models)
    page = _client(store, models).get("/settings").data.decode()
    assert "More settings:" not in page
    rows = re.findall(r'<li><a href="([^"]+)"><span class="sw-rows__title">([^<]+)</span>', page)
    assert rows == [
        ("/settings/advanced", "Advanced settings"),
        ("/settings/email", "Email alerts"),
    ]
    assert "Who gets emailed, and test emails" in page
    assert "badge" not in page and "pill" not in page
    website = _website(_site_store(tmp_path / "w", models), models, updater=object())
    page = website.get("/strawberrywatch/settings").data.decode()
    assert "Get the newest model files from GitHub" in page


# Settings: recipient labels, website only


def test_labels_start_with_four_and_public_keeps_its_name(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    assert [lb["name"] for lb in store.labels()] == ["SCMG", "EH&S", "Admin", "Public"]
    public, scmg = store.label_named("Public"), store.label_named("SCMG")
    assert not store.rename_label(public["id"], "Everyone")
    assert not store.rename_label(scmg["id"], "public")
    assert store.rename_label(scmg["id"], "SCMG staff")
    assert not store.add_label("eh&s")


def test_recipients_filter_and_sort_by_label(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    ids = {lb["name"]: lb["id"] for lb in store.labels()}
    store.add_recipient("zed@example.org", None, ids["Admin"])
    store.add_recipient("amy@example.org", None, ids["Public"])
    store.add_recipient("bo@example.org", None, ids["EH&S"])
    client = _website(store, models)
    page = client.get("/strawberrywatch/settings/email?label=Public").data.decode()
    assert "amy@example.org" in page and "zed@" not in page and "bo@" not in page
    page = client.get("/strawberrywatch/settings/email?sort=label").data.decode()
    order = [m for m in re.findall(r"<td>(\w+@example\.org)</td>", page)]
    assert order == ["zed@example.org", "bo@example.org", "amy@example.org"]
    # Plain text in its own column, no badge
    assert "<td>EH&amp;S</td>" in page
    response = client.post(
        "/strawberrywatch/settings/email/relabel",
        data={
            "csrf_token": _csrf(page),
            "recipient": str(store.recipients()[0]["id"]),
            "label_id": str(ids["SCMG"]),
        },
    )
    assert response.status_code == 302
    assert store.recipients()[0]["label_id"] == ids["SCMG"]


def test_editors_add_and_rename_labels(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    client = _website(store, models)
    page = client.get("/strawberrywatch/settings/email").data.decode()
    token = _csrf(page)
    client.post(
        "/strawberrywatch/settings/email/labels", data={"csrf_token": token, "name": "Volunteers"}
    )
    assert store.label_named("Volunteers")
    public = store.label_named("Public")
    client.post(
        "/strawberrywatch/settings/email/labels",
        data={"csrf_token": token, "name": "Everyone", "label_id": str(public["id"])},
    )
    assert store.label_named("Public") and not store.label_named("Everyone")
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Public can&#39;t be renamed" in page
    viewer = _website(store, models, user="ben", can_edit=False).get(
        "/strawberrywatch/settings/email"
    )
    assert "Volunteers" in viewer.data.decode() and "Add label" not in viewer.data.decode()


def test_a_signup_confirms_by_email_and_lands_as_public(models, tmp_path, monkeypatch):
    import email_validator

    class Checked:
        def __init__(self, address):
            self.normalized, self.mx = address, [(10, "mx")]

    monkeypatch.setattr(email_validator, "validate_email", lambda a, **k: Checked(a))
    sent = _fake_mail(monkeypatch)
    store = _site_store(tmp_path / "s", models)
    store.add_recipient("ana@example.org", "Ana", store.label_named("SCMG")["id"])
    public = _website(store, models, user=None)
    page = public.get("/strawberrywatch/signup").data.decode()
    # Signed out, nobody else's address or label is anywhere on the page
    assert "ana@example.org" not in page and "SCMG" not in page and "<select" not in page
    response = public.post(
        "/strawberrywatch/signup", data={"csrf_token": _csrf(page), "email": "cy@example.org"}
    )
    assert "Check your email" in response.data.decode()
    assert [p["email"] for p in store.recipients()] == ["ana@example.org"]
    (message, (to,)) = sent[0]
    assert to == "cy@example.org" and message["Subject"] == "Confirm your StrawberryWatch alerts"
    link = re.search(
        r"https://localhost/strawberrywatch(/signup/confirm/[\w-]+)",
        message.get_body(("plain",)).get_content(),
    )
    confirm = public.get("/strawberrywatch" + link.group(1)).data.decode()
    assert "Send StrawberryWatch's alerts" in confirm and "cy@example.org" in confirm
    # Opening the link alone doesn't sign anyone up
    assert len(store.recipients()) == 1
    done = public.post("/strawberrywatch" + link.group(1), data={"csrf_token": _csrf(confirm)})
    assert "You're on the list" in done.data.decode()
    cy = store.recipients()[-1]
    assert cy["email"] == "cy@example.org" and cy["label_id"] == store.label_named("Public")["id"]
    assert public.get("/strawberrywatch" + link.group(1)).status_code == 404
    # Asking for someone already on the list says the same thing and sends no email
    page = public.get("/strawberrywatch/signup").data.decode()
    again = public.post(
        "/strawberrywatch/signup", data={"csrf_token": _csrf(page), "email": "ana@example.org"}
    )
    assert "Check your email" in again.data.decode() and len(sent) == 1


def test_signed_out_sees_no_list_and_unsubscribe_needs_the_token(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    store.add_recipient("ana@example.org", None, store.label_named("Admin")["id"])
    public = _website(store, models, user=None)
    assert public.get("/strawberrywatch/settings/email").status_code == 403
    assert public.get("/strawberrywatch/unsubscribe/not-a-token").status_code == 404
    token = store.recipients()[0]["unsubscribe_token"]
    page = public.get(f"/strawberrywatch/unsubscribe/{token}").data.decode()
    assert "ana@example.org" in page and len(store.recipients()) == 1
    public.post(f"/strawberrywatch/unsubscribe/{token}", data={"csrf_token": _csrf(page)})
    assert store.recipients() == []


def test_the_local_dashboard_has_no_labels_or_signup(models, tmp_path):
    store = _store(tmp_path / "s", models)
    client = _client(store, models)
    assert client.get("/signup").status_code == 404
    assert "Labels" not in client.get("/settings/email").data.decode()


def test_night_herons_list_is_copied_once(models, tmp_path):
    store = _site_store(tmp_path / "s", models)
    store.add_recipient("ana@example.org")
    copy_list = lambda: ["Bo@example.org", "not an address", "ana@example.org"]  # noqa: E731
    client = _website(store, models, copy_list=copy_list)
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Copy Night Heron's recipients" in page
    eh = store.label_named("EH&S")["id"]
    client.post(
        "/strawberrywatch/settings/email/copy",
        data={"csrf_token": _csrf(page), "label_id": str(eh)},
    )
    people = {p["email"]: p for p in store.recipients()}
    assert set(people) == {"ana@example.org", "Bo@example.org"}
    assert people["Bo@example.org"]["label_id"] == eh
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Copy Night Heron" not in page and "Copied 1 address from Night Heron" in page
    page_token = _csrf(page)
    again = client.post(
        "/strawberrywatch/settings/email/copy", data={"csrf_token": page_token, "label_id": str(eh)}
    )
    assert again.status_code == 404


def test_the_html_email_has_both_logos_and_an_unsubscribe_link():
    text, html = mail.alert_email(
        "Subject", "New alert in Strawberry Creek.", [], "https://x/unsubscribe/t"
    )
    assert "unsubscribe: https://x/unsubscribe/t" in text
    message = mail.html_message(
        "watch@example.org", "ana@example.org", "Subject", text, html, "https://x/unsubscribe/t"
    )
    cids = {
        part.get("Content-ID") for part in message.walk() if part.get_content_maintype() == "image"
    }
    assert cids == {"<emblem>", "<sponsors>"}


@pytest.mark.parametrize("store_kind", ["sqlite", "prefixed"])
def test_old_recipients_get_an_unsubscribe_token(models, tmp_path, store_kind):
    store = st.Store(tmp_path / "s") if store_kind == "sqlite" else PrefixedStore(tmp_path / "p")
    with store.db() as db:
        db.execute(
            "INSERT INTO recipients (email, added_at) VALUES (?, ?)", ("old@example.org", 1.0)
        )
    token = store.recipients()[0]["unsubscribe_token"]
    assert token and store.recipients()[0]["unsubscribe_token"] == token

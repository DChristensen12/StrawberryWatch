"""The website's emails go through Cloudflare's email API with Night Heron's account."""
# The token and account id stay out of every page, log and record

import logging

import pytest
import requests

from strawberrywatch import inventory as inv
from strawberrywatch.dashboard import mail
from strawberrywatch.watch import models as wm
from tests.test_alert_gaps import _alert
from tests.test_dashboard_changes import NOW
from tests.test_overview_alerts_settings import _csrf, _site_store, _three_recipients, _website
from tests.test_website_load import _fake_mail

SECRETS = ("token-for-tests", "account-for-tests")


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _shown(store, *pages):
    """Everything someone could read afterwards: pages, the email log and each address's line."""
    rows = [str(r) for r in store.emails()] + [str(r) for r in store.email_results().values()]
    return "\n".join([*pages, *rows])


def _test_email(client):
    page = client.get("/strawberrywatch/settings/email").data.decode()
    client.post("/strawberrywatch/settings/email/test", data={"csrf_token": _csrf(page)})
    return client.get("/strawberrywatch/settings/email").data.decode()


def test_the_token_goes_only_in_the_request_header(models, tmp_path, monkeypatch):
    sent = _fake_mail(monkeypatch)
    store = _site_store(tmp_path / "s", models)
    _three_recipients(store)
    page = _test_email(_website(store, models))
    assert "Test email sent to 3 people." in page and "sent from watch@example.org." in page
    assert sent.seen_by[0] == (
        "https://api.cloudflare.com/client/v4/accounts/account-for-tests/email/sending/send",
        "Bearer token-for-tests",
    )
    for email, _ in sent:
        assert not any(secret in str(email) for secret in SECRETS)
    assert not any(secret in _shown(store, page) for secret in SECRETS)


def test_when_cloudflare_cant_be_reached_nothing_shows_the_account(
    models, tmp_path, monkeypatch, caplog
):
    def unreachable(email):
        # requests puts the whole URL in its message, account id and all
        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='api.cloudflare.com', port=443): Max retries exceeded "
            "with url: /client/v4/accounts/account-for-tests/email/sending/send"
        )

    sent = _fake_mail(monkeypatch, unreachable)
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    _three_recipients(store)
    page = _test_email(_website(store, models))
    assert "Test email not sent. The email didn&#39;t go. Couldn&#39;t reach Cloudflare." in page
    # The rest would wait the same timeout, and the page has gunicorn's 30 seconds
    assert len(sent) == 1
    check = _alert(store, "dusk_crayfish", NOW)
    with caplog.at_level(logging.DEBUG):
        mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch")
    assert store.emails()[0]["message"] == "The email didn't go. Couldn't reach Cloudflare."
    assert not any(secret in _shown(store, page, caplog.text) for secret in SECRETS)


def test_a_turned_down_token_stops_the_emails(models, tmp_path, monkeypatch):
    def refused(email):
        error = {"code": 10101, "message": "email.sending.error.authentication.unauthorized"}
        return 401, {"success": False, "errors": [error], "messages": [], "result": None}

    sent = _fake_mail(monkeypatch, refused)
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    _three_recipients(store)
    check = _alert(store, "dusk_crayfish", NOW)
    mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch")
    assert len(sent) == 1
    last = store.emails()[0]
    assert last["ok"] == 0
    assert last["message"] == "The email didn't go. Cloudflare turned down the API token in .env."
    page = _website(store, models).get("/strawberrywatch/settings/email").data.decode()
    assert "Alert, " in page and "Cloudflare turned down the API token in .env." in page


def test_a_bounce_is_noted_beside_that_address_only(models, tmp_path, monkeypatch):
    def answer(email):
        to = email["to"]
        bounced = to == ["ben@example.org"]
        result = {
            "delivered": [] if bounced else to,
            "queued": [],
            "permanent_bounces": to if bounced else [],
        }
        return 200, {"success": True, "errors": [], "messages": [], "result": result}

    sent = _fake_mail(monkeypatch, answer)
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    _three_recipients(store)
    check = _alert(store, "dusk_crayfish", NOW)
    mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch")
    assert len(sent) == 3
    results = store.email_results()
    assert results["ben@example.org"]["message"] == "The address bounced."
    assert results["ana@example.org"]["ok"] == 1 and results["cy@example.org"]["ok"] == 1
    assert store.emails()[0]["message"] == "1 of 3 didn't go."


def test_cloudflares_other_errors_keep_only_the_code(models, tmp_path, monkeypatch):
    answers = iter(
        [
            (429, {"success": False, "errors": [{"code": 10004, "message": "throttled"}]}),
            (500, {"success": False, "errors": [{"code": 10002, "message": "token-for-tests"}]}),
            (502, "not json"),
        ]
    )
    _fake_mail(monkeypatch, lambda email: next(answers))
    store = _site_store(tmp_path / "s", models)
    store.set_pref("email_on", "yes")
    _three_recipients(store)
    check = _alert(store, "dusk_crayfish", NOW)
    mail.website_alert(store, inv.load(), check, "https://localhost/strawberrywatch")
    results = store.email_results()
    assert [
        results[a]["message"] for a in ("ana@example.org", "ben@example.org", "cy@example.org")
    ] == [
        "Cloudflare's sending limit was reached.",
        "Cloudflare didn't send it (error 10002).",
        "Cloudflare didn't send it (HTTP 502).",
    ]
    assert not any(secret in _shown(store) for secret in SECRETS)


def test_without_the_account_the_page_names_whats_missing(models, tmp_path, monkeypatch):
    sent = _fake_mail(monkeypatch)
    monkeypatch.delenv("CLOUDFLARE_EMAIL_API_TOKEN")
    store = _site_store(tmp_path / "s", models)
    _three_recipients(store)
    client = _website(store, models)
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Missing: CLOUDFLARE_EMAIL_API_TOKEN." in page
    client.post(
        "/strawberrywatch/settings/email/sending", data={"csrf_token": _csrf(page), "on": "yes"}
    )
    page = client.get("/strawberrywatch/settings/email").data.decode()
    assert "Email isn&#39;t set up on the server, so sending can&#39;t be turned on." in page
    assert store.pref("email_on") != "yes" and sent == []

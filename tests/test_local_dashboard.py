"""
The local dashboard's pages through Flask's test client, against a store with
one real replay check in it. Security first, then the build spec's steps 3 and 4.
"""

import csv
import io
import re
import shutil
import smtplib
import time
from datetime import datetime

import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.dashboard import create_app, mail, security, view
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import sources
from strawberrywatch.watch import store as st
from tests.live_sample import EVENT, live_store


class FakeWatcher:
    """Counts the checks asked for instead of running them."""

    busy = False
    next_at = None

    def __init__(self):
        self.asked = 0

    def check_now(self):
        self.asked += 1


@pytest.fixture(scope="module")
def models():
    return wm.registered()


@pytest.fixture(scope="module")
def sample(tmp_path_factory, models):
    folder = tmp_path_factory.mktemp("sample")
    live_store(folder, models)
    return folder


def _app(folder, models, tmp_path):
    shutil.copytree(folder, tmp_path / "data")
    store = st.Store(tmp_path / "data")
    watcher = FakeWatcher()
    app = create_app(
        store=store, models=models, watcher=watcher, login=security.Login("test-token")
    )
    return app, store, watcher


@pytest.fixture
def signed_in(sample, models, tmp_path):
    app, store, watcher = _app(sample, models, tmp_path)
    client = app.test_client()
    assert client.get("/login?token=test-token").status_code == 302
    return client, store, watcher


def _csrf(page):
    return re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()


def _post(client, path, data, fetch=False):
    page = client.get(path).data
    headers = {"X-Requested-With": "fetch"} if fetch else {}
    return client.post(path, data={"csrf_token": _csrf(page), **data}, headers=headers)


def _settings_form(store, models, **changes):
    """What the Settings form posts for the draft as it stands, with some answers changed."""
    draft = store.draft() or store.applied()[1]
    sites = [s for s in inv.load().sites if s not in draft["dont_flag"]]
    data = {"watch": sites, "rain": "y" if draft["rain"] else ""}
    for m in models:
        data[f"model_{m.name}"] = "on" if draft["models"][m.name]["on"] else "off"
    data.update(changes)
    return data


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
            data[f"scale_{m.name}_{rule}"] = str(spec["scales"][rule])
    data.update(changes)
    return data


# Security


def test_a_host_that_isnt_this_computer_is_refused(sample, models, tmp_path):
    """Build order step 1."""
    app, _, _ = _app(sample, models, tmp_path)
    client = app.test_client()
    for host in ("evil.example", "127.0.0.1.evil.example", "localhost.evil.example"):
        response = client.get("/login?token=test-token", headers={"Host": host})
        assert response.status_code == 403 and b"only answers on this computer" in response.data
    assert (
        security.host_ok("localhost:8790")
        and security.host_ok("127.0.0.1")
        and security.host_ok("[::1]:8790")
    )
    assert not security.host_ok("192.168.1.5:8790")


def test_the_login_link_works_once_and_sets_a_strict_cookie(sample, models, tmp_path):
    app, _, _ = _app(sample, models, tmp_path)
    client = app.test_client()
    assert client.get("/").status_code == 403
    assert client.get("/login?token=wrong").status_code == 403
    response = client.get("/login?token=test-token")
    assert response.status_code == 302 and response.headers["Location"] == "/"
    cookie = response.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert client.get("/").status_code == 200
    assert app.test_client().get("/login?token=test-token").status_code == 403


def test_every_page_sends_a_strict_csp(signed_in):
    client, _, _ = signed_in
    for path in ("/", "/history", "/models", "/settings", "/settings/advanced", "/sites/oxford"):
        response = client.get(path)
        csp = response.headers["Content-Security-Policy"]
        assert response.status_code == 200
        assert (
            "script-src 'self';" in csp and "unsafe" not in csp and "frame-ancestors 'none'" in csp
        )
        # No inline script except the JSON chart blocks, which the browser doesn't run
        for tag in re.findall(rb"<script(?![^>]*\bsrc=)[^>]*>", response.data):
            assert b'type="application/json"' in tag


def test_state_only_changes_through_a_post_with_a_csrf_token(signed_in, models):
    client, store, _ = signed_in
    assert (
        client.post("/settings", data=_settings_form(store, models, alerts="fewer")).status_code
        == 400
    )
    assert store.draft() is None
    assert client.get("/settings/apply").status_code == 405


def test_error_pages_show_no_traceback(signed_in, monkeypatch):
    client, _, _ = signed_in

    def broken(*_args, **_kw):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(view, "recent_noticed", broken)
    response = client.get("/")
    assert response.status_code == 500
    assert b"Traceback" not in response.data and b"secret detail" not in response.data


# Overview, Alerts, History and Models


def test_a_quiet_day_says_everything_looks_normal_and_nothing_else(models, tmp_path):
    store = st.Store(tmp_path / "quiet")
    store.set_source(sources.API)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    app = create_app(store=store, models=models, watcher=FakeWatcher(), login=security.Login("t"))
    client = app.test_client()
    client.get("/login?token=t")
    assert "Waiting for the first check" in client.get("/").data.decode()
    now = time.time()
    check = store.start_check(sources.API, 1, now - 3600, now, started_at=now)
    store.finish_check(check, True, "", {})
    store.save_readings(
        [(site, now - 60, 1.0, 1.0, 1.0, None, None, None) for site in inv.load().sites]
    )
    page = client.get("/").data
    assert b"No alerts in the last 24 hours" in page and b"Alert at " not in page
    assert b"No alerts in the last 24 hours." in client.get("/unusual").data


def test_something_noticed_shows_on_overview_and_in_full_on_unusual_readings(signed_in):
    client, _, _ = signed_in
    overview = client.get("/").data.decode()
    assert "Alert at South Fork 1" in overview
    assert 'href="/unusual"' in overview and 'class="nav-count"' in overview
    page = client.get("/unusual").data.decode()
    assert "Conductivity alert at South Fork 1" in page
    assert (
        "Conductivity didn&#39;t match the model&#39;s forecast from" in page
        and "Flagged by Dusk Crayfish." in page
    )
    assert 'data-sw-chart="reading"' in page and "View details" in page


def test_history_mixes_noticed_items_and_changes_newest_first(signed_in, models):
    client, store, _ = signed_in
    _post(client, "/settings", _settings_form(store, models, model_cobble_shoal="on"))
    client.post("/settings/apply", data={"csrf_token": _csrf(client.get("/settings/review").data)})
    page = client.get("/history").data.decode()
    change = page.index("Cobble Shoal turned on.")
    setup = page.index("StrawberryWatch set up with default settings.")
    item = page.index("Conductivity alert at South Fork 1 (Dusk Crayfish)")
    assert change < setup < item
    assert "No alerts." in page
    for label in ("All", "Alerts", "Settings changes", "By hour"):
        assert label in page
    only = client.get("/history?show=changes").data.decode()
    assert "(Dusk Crayfish)" not in only and "Cobble Shoal turned on." in only
    unusual = client.get("/history?show=unusual").data.decode()
    assert "(Dusk Crayfish)" in unusual and "Cobble Shoal turned on." not in unusual


ROW = r'class="sw-hour(?: sw-hour--quiet)?"'


def test_by_hour_folds_quiet_hours_and_a_day_shows_every_hour(signed_in):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    page = client.get("/history?by=hour").data.decode()
    assert "(Dusk Crayfish)" in page and "All day" in page
    # The noticed day: the quiet hours either side fold into one row each
    hour = datetime.fromtimestamp(item["first_over"], view.PACIFIC).hour
    day = datetime.fromtimestamp(item["first_over"], view.PACIFIC).date().isoformat()
    section = page[page.index(f'/history/day/{day}"') :]
    section = section[: section.index("</section>")]
    assert len(re.findall(ROW, section)) == 1 + (hour > 0) + (hour < 23)
    assert "– midnight" in section and "No alerts." in section
    hourly = client.get(f"/history/day/{day}").data.decode()
    assert len(re.findall(ROW, hourly)) == 24 and "(Dusk Crayfish)" in hourly
    today = datetime.now(view.PACIFIC).date().isoformat()
    assert "1 check." in client.get(f"/history/day/{today}").data.decode()
    assert client.get("/history/day/not-a-day").status_code == 404


def test_notes_add_up_on_a_reading_and_show_under_it_in_history(signed_in):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    path = f"/history/{item['id']}"
    page = client.get(path).data.decode()
    assert (
        "Support module output" in page
        and "Newtwork Run" in page
        and 'data-sw-chart="score"' in page
        and "No notes yet." in page
    )
    assert _post(client, path, {"note": "Went to look, cloudy water"}).status_code == 302
    assert _post(client, path, {"note": "  Clear again the next day  "}).status_code == 302
    blank = _post(client, path, {"note": "   "})
    assert blank.status_code == 200 and "Enter a note" in blank.data.decode()
    notes = store.notes(noticed_id=item["id"])
    assert [n["text"] for n in notes] == ["Clear again the next day", "Went to look, cloudy water"]
    page = client.get(path).data.decode()
    assert page.index("Went to look") < page.index("Clear again")
    history = client.get("/history").data.decode()
    assert "Went to look, cloudy water" in history and "Notes (2)" in history
    deleted = client.post(
        f"/history/notes/{notes[0]['id']}/delete", data={"csrf_token": _csrf(client.get(path).data)}
    )
    assert deleted.status_code == 302 and [n["text"] for n in store.notes()] == [
        "Went to look, cloudy water"
    ]


def test_notes_page_filters_and_downloads_a_csv_or_a_page_to_save_as_pdf(signed_in):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    store.add_note(item["id"], '=HYPERLINK("http://example.invalid")')
    second = store.add_note(item["id"], "Cloudy water, µS/cm way up")
    page = client.get("/history/notes").data.decode()
    assert "Cloudy water" in page and 'name="note"' in page and "Download as CSV" in page
    assert "No notes match these filters." in client.get("/history/notes?site=oxford").data.decode()
    assert (
        "No notes match these filters."
        in client.get("/history/notes?from=2999-01-01").data.decode()
    )

    response = client.get("/history/notes/export?format=csv")
    assert response.mimetype == "text/csv"
    assert response.headers["Content-Disposition"].startswith("attachment;")
    text = response.data.decode("utf-8")
    assert text.startswith("\ufeff")
    rows = list(csv.reader(io.StringIO(text[1:])))
    assert rows[0][:4] == ["Written", "Note", "Site", "Site name"]
    # Oldest first, and a note that looks like a formula can't run as one
    assert [r[1] for r in rows[1:]] == [
        '\'=HYPERLINK("http://example.invalid")',
        "Cloudy water, µS/cm way up",
    ]
    assert rows[1][2] == "south_fork_1"
    ticked = client.get(f"/history/notes/export?format=csv&note={second}").data.decode("utf-8")
    assert "Cloudy water" in ticked and "HYPERLINK" not in ticked

    printable = client.get("/history/notes/export?format=pdf").data.decode()
    assert 'data-sw-print="yes"' in printable and "Cloudy water" in printable
    assert 'data-sw-chart="score"' in printable and "2 notes." in printable
    assert client.get("/history/notes/export?format=doc").status_code == 400
    assert client.get("/history/notes/export?format=csv&note=x").status_code == 400


def test_a_note_from_before_notes_added_up_becomes_the_first_note(tmp_path):
    store = st.Store(tmp_path / "old")
    with store.db() as db:
        db.execute("INSERT INTO checks (started_at) VALUES (1)")
        db.execute(
            "INSERT INTO noticed (check_id, model, site, variable, rule, first_over, last_over, "
            "flagged, note, noted_at, created_at) "
            "VALUES (1, 'dusk_crayfish', 'oxford', 'depth', 'level_shift', 1, 1, 1, 'Old note', 5, 2)"
        )
    st.Store(tmp_path / "old")
    again = st.Store(tmp_path / "old")
    assert [(n["text"], n["written_at"]) for n in again.notes()] == [("Old note", 5.0)]


def test_models_shows_all_six_cards_with_on_and_off_only_on_the_four_that_run(signed_in):
    client, _, _ = signed_in
    page = client.get("/models").data.decode()
    for name in (
        "Dusk Crayfish",
        "Cobble Shoal",
        "Flame Skimmer",
        "Berry Delight",
        "Water Strider",
        "Crimson Parker",
    ):
        assert name in page
    assert page.count("></span>Not available yet</span>") == 2
    assert "></span>On</span>" in page and "></span>Off</span>" in page


# Settings, Advanced settings, review and apply


def test_edits_survive_switching_pages_and_reloading(signed_in, models):
    client, store, _ = signed_in
    assert (
        _post(
            client, "/settings", _settings_form(store, models, alerts="fewer"), fetch=True
        ).status_code
        == 200
    )
    assert store.draft()["models"]["dusk_crayfish"]["scales"]["forecast_residual"] == 1.25
    advanced = client.get("/settings/advanced").data.decode()
    assert 'value="1.25"' in advanced
    again = client.get("/settings").data.decode()
    assert "1 unsaved change." in again
    assert (
        'id="sw-discard"' in again and 'id="sw-discard" form="sw-discard-form" hidden' not in again
    )
    assert again.count("Save changes") == 1


def test_save_changes_goes_on_to_the_check_page(signed_in, models):
    client, store, _ = signed_in
    response = _post(
        client, "/settings", {**_settings_form(store, models, alerts="fewer"), "save": "y"}
    )
    assert response.status_code == 302 and response.location.endswith(
        "/settings/review?from=settings"
    )
    discarded = client.post(
        "/settings/discard",
        data={"csrf_token": _csrf(client.get("/settings").data), "back": "settings"},
    )
    assert discarded.status_code == 302 and store.draft() is None
    again = _post(client, "/settings", {**_settings_form(store, models), "save": "y"})
    assert again.location.endswith("/settings")
    page = client.get("/settings").data.decode()
    assert "No changes to save" in page
    assert 'id="sw-discard" form="sw-discard-form" hidden' in page


def test_autosave_answers_with_the_banner(signed_in, models):
    client, store, _ = signed_in
    response = _post(client, "/settings", _settings_form(store, models, rain=""), fetch=True)
    assert response.json["changes"] == 1 and "unsaved change" in response.json["banner"]


def test_review_lists_every_change_before_and_after(signed_in, models):
    client, store, _ = signed_in
    _post(
        client, "/settings", _settings_form(store, models, alerts="more", model_cobble_shoal="on")
    )
    page = client.get("/settings/review").data.decode()
    assert "Alert sensitivity" in page and "Before: Balanced" in page and "After: More" in page
    assert "Cobble Shoal" in page and "Before: Off" in page and "After: On" in page
    assert "Check your changes" in page and "Save changes" in page
    assert "Go back" in page and "hasn&#39;t been retrained" in page


def test_apply_saves_a_version_and_starts_a_check_right_away(signed_in, models):
    client, store, watcher = signed_in
    before = len(store.versions())
    _post(client, "/settings", _settings_form(store, models, alerts="fewer"))
    response = client.post(
        "/settings/apply", data={"csrf_token": _csrf(client.get("/settings/review").data)}
    )
    assert response.status_code == 302 and watcher.asked == 1
    assert len(store.versions()) == before + 1 and store.draft() is None
    assert ws.alerts_choice(store.applied()[1]) == "fewer"
    assert "Settings saved. A check is running now." in client.get("/").data.decode()


def test_turning_a_model_off_folds_its_settings_and_keeps_them(signed_in, models):
    client, store, _ = signed_in
    form = _advanced_form(
        store, models, model_dusk_crayfish="off", scale_dusk_crayfish_level_shift="1.3"
    )
    assert _post(client, "/settings/advanced", form).status_code == 302
    spec = store.draft()["models"]["dusk_crayfish"]
    assert spec["on"] is False and spec["scales"]["level_shift"] == 1.3
    page = client.get("/settings/advanced").data.decode()
    assert "Turn on to see settings. Saved values are kept." in page


def test_settings_says_custom_when_advanced_values_match_no_choice(signed_in, models):
    client, store, _ = signed_in
    _post(
        client,
        "/settings/advanced",
        _advanced_form(store, models, scale_dusk_crayfish_level_shift="1.1"),
    )
    assert "Custom values are set in Advanced settings." in client.get("/settings").data.decode()


def test_settings_pages_carry_what_the_leave_prompt_reads(signed_in, models):
    """The prompt itself is app.js; the page has to say how many changes aren't saved."""
    client, store, _ = signed_in
    _post(client, "/settings", _settings_form(store, models, alerts="more"))
    page = client.get("/settings").data.decode()
    assert 'data-unsaved="1"' in page and 'data-settings-page="yes"' in page
    assert 'data-settings-page="no"' in client.get("/").data.decode()


def test_a_bad_number_comes_back_as_an_error_summary(signed_in, models):
    client, store, _ = signed_in
    response = _post(
        client,
        "/settings/advanced",
        _advanced_form(store, models, scale_dusk_crayfish_level_shift="lots"),
    )
    page = response.data.decode()
    assert (
        response.status_code == 200 and "There is a problem" in page and "must be a number" in page
    )


# Model limits, signing out and unknown pages


def test_dusk_crayfish_shows_leave_a_site_out_as_not_available(signed_in):
    client, _, _ = signed_in
    assert (
        "Not available. Dusk Crayfish needs all 4 of its sites."
        in client.get("/settings/advanced").data.decode()
    )


def test_cobble_shoal_under_half_is_refused_at_review(signed_in, models):
    client, store, _ = signed_in
    form = _advanced_form(
        store,
        models,
        model_cobble_shoal="on",
        leave_cobble_shoal=["scnf010", "north_fork_0", "south_fork_1"],
    )
    _post(client, "/settings/advanced", form)
    page = client.get("/settings/review").data.decode()
    assert "There is a problem" in page and "under half its sensors" in page
    assert "Apply changes" not in page


def test_sign_out_ends_the_session_and_says_how_to_sign_in_again(signed_in):
    client, _, _ = signed_in
    assert 'href="/signout"' in client.get("/").data.decode()
    page = client.get("/signout").data.decode()
    assert "Signed out" in page and "restart StrawberryWatch" in page
    assert client.get("/").status_code == 403


def test_an_unknown_site_page_is_a_404(signed_in):
    client, _, _ = signed_in
    assert client.get("/sites/..%2Fsecrets").status_code == 404
    assert client.get("/sites/not_a_site").status_code == 404
    assert 'data-sw-chart="spark"' in client.get("/sites/south_fork_1").data.decode()


# Email alerts


class FakeSMTP:
    """A fake mail server. Keeps what would have been sent, and refuses a wrong password or a bad address."""

    sent = []
    refuse = set()
    logins = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host = host

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        FakeSMTP.logins.append(user)
        if password == "wrong":
            raise smtplib.SMTPAuthenticationError(535, b"no")

    def send_message(self, message, from_addr=None, to_addrs=None):
        if set(to_addrs) & FakeSMTP.refuse:
            raise smtplib.SMTPRecipientsRefused({a: (550, b"no such user") for a in to_addrs})
        FakeSMTP.sent.append((message, to_addrs))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Checked:
    """What email-validator hands back, without a DNS lookup."""

    def __init__(self, address, mx=True):
        self.normalized = address
        self.mx = [(10, "mx.example.org")] if mx else None


def fake_validate(address, check_deliverability=True, timeout=None):
    from email_validator import EmailUndeliverableError

    domain = address.rsplit("@", 1)[1].lower()
    if domain == "berkely.edu":
        raise EmailUndeliverableError("The domain name berkely.edu does not exist.")
    if domain == "offline.example":
        raise EmailUndeliverableError(
            "There was an error while checking if the domain name in the email address is deliverable: no network"
        )
    return Checked(address, mx=domain != "slow.example")


ENV_START = "# kept as is\nALERT_EMAIL_SENDER=watch@example.org\nSOMETHING_ELSE='a, b'\n"


@pytest.fixture
def mail_account(monkeypatch, tmp_path):
    """A mail account from the environment, and a .env of its own, not this computer's real one."""
    import email_validator

    monkeypatch.setattr(mail, "load_dotenv", lambda *a, **k: None)
    env = tmp_path / "account" / ".env"
    env.parent.mkdir()
    env.write_text(ENV_START)
    monkeypatch.setattr(mail, "env_path", lambda: env)
    monkeypatch.setattr(email_validator, "validate_email", fake_validate)
    for name, value in {
        "ALERT_EMAIL_SENDER": "watch@example.org",
        "ALERT_EMAIL_PASSWORD": "secret",
        "SMTP_SERVER": "smtp.example.org",
        "SMTP_PORT": "587",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ALERT_EMAIL_FROM", raising=False)
    monkeypatch.setattr(mail.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent = []
    FakeSMTP.refuse = set()
    FakeSMTP.logins = []
    monkeypatch.env = env
    return monkeypatch


def _remove(client, address):
    return client.post(
        "/settings/email/remove",
        data={"csrf_token": _csrf(client.get("/settings/email").data), "email": address},
    )


def test_people_are_added_and_removed_from_the_email_list_straight_away(signed_in, mail_account):
    client, store, _ = signed_in
    env = mail_account.env
    assert "Email alerts" in client.get("/settings").data.decode()
    assert (
        _post(client, "/settings/email", {"email": " ana@example.org ", "name": "Ana"}).status_code
        == 302
    )
    again = _post(client, "/settings/email", {"email": "ANA@example.org"}).data.decode()
    assert "ANA@example.org is already a recipient" in again
    assert (
        "Enter an email address with an @"
        in _post(client, "/settings/email", {"email": "ana at example"}).data.decode()
    )
    assert mail.mailing_list(store).people() == [{"email": "ana@example.org", "name": "Ana"}]
    page = client.get("/settings/email").data.decode()
    assert "ana@example.org" in page and "Email alerts are off. No one is emailed." in page
    assert _remove(client, "ana@example.org").status_code == 302
    assert mail.mailing_list(store).people() == []
    assert _remove(client, "nobody@example.org").status_code == 404
    assert env.read_text().startswith(ENV_START)


def test_sending_needs_the_account_in_env_and_changes_the_footer(signed_in, mail_account):
    client, store, _ = signed_in
    mail_account.delenv("SMTP_SERVER")
    page = client.get("/settings/email").data.decode()
    assert "Email isn't set up on this computer." in page and "SMTP_SERVER" in page
    token = _csrf(client.get("/settings/email").data)
    client.post("/settings/email/sending", data={"csrf_token": token, "on": "yes"})
    assert store.pref("email_on") != "yes"
    mail_account.setenv("SMTP_SERVER", "smtp.example.org")
    client.post("/settings/email/sending", data={"csrf_token": token, "on": "yes"})
    assert store.pref("email_on") == "yes"
    assert "emails new alerts to the recipients list" in client.get("/").data.decode()


def test_a_check_emails_each_person_separately_and_only_from_live_readings(
    signed_in, mail_account, tmp_path
):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    book = mail.mailing_list(store)
    book.add("ana@example.org")
    book.add("ben@example.org")
    inventory = inv.load()
    # Off: no email
    mail.after_check(store, inventory, item["check_id"])
    assert FakeSMTP.sent == []
    store.set_pref("email_on", "yes")
    # On, but a replay of a past event, as load_test.py runs: still no email
    store.set_source(sources.REPLAY, event=EVENT)
    mail.after_check(store, inventory, item["check_id"])
    assert FakeSMTP.sent == []
    # On, and live: one email each, and nobody sees anyone else
    store.set_source(sources.SITE)
    mail.after_check(store, inventory, item["check_id"])
    assert [to for _, to in FakeSMTP.sent] == [["ana@example.org"], ["ben@example.org"]]
    message = FakeSMTP.sent[0][0]
    assert message["To"] == "ana@example.org" and "ben@" not in message.as_string()
    # Site names the way the website writes them, not the table name
    assert message["Subject"] == "StrawberryWatch: conductivity alert at South Fork 1"
    body = message.get_content()
    assert "Conductivity alert at South Fork 1, Stephens Hall" in body
    assert "Flagged by Dusk Crayfish." in body and "take you off it" in body
    assert store.emails()[0]["ok"] == 1 and store.emails()[0]["people"] == 2
    assert "Alert sent" in client.get("/settings/email").data.decode()


def test_from_address_can_differ_from_the_login(mail_account):
    mail.send("Test", "hello", [{"email": "ana@example.org"}])
    assert FakeSMTP.sent[-1][0]["From"] == "watch@example.org"
    # Elastic Email logs in with one name and sends from another
    mail_account.setenv("ALERT_EMAIL_FROM", "alerts@example.org")
    mail.send("Test", "hello", [{"email": "ana@example.org"}])
    assert FakeSMTP.logins == ["watch@example.org", "watch@example.org"]
    assert FakeSMTP.sent[-1][0]["From"] == "alerts@example.org"
    assert mail.sender() == "alerts@example.org"


def test_one_rejected_address_doesnt_stop_the_others(signed_in, mail_account):
    client, store, _ = signed_in
    book = mail.mailing_list(store)
    for address in ("ana@example.org", "gone@example.org", "ben@example.org"):
        book.add(address)
    FakeSMTP.refuse = {"gone@example.org"}
    response = client.post(
        "/settings/email/test", data={"csrf_token": _csrf(client.get("/settings/email").data)}
    )
    assert response.status_code == 302
    assert [to for _, to in FakeSMTP.sent] == [["ana@example.org"], ["ben@example.org"]]
    results = store.email_results()
    assert results["gone@example.org"]["ok"] == 0 and results["ben@example.org"]["ok"] == 1
    page = client.get("/settings/email").data.decode()
    assert "Test email sent to 2 of 3 people." in page
    assert "The mail server turned this address down." in page


def test_a_test_email_goes_only_to_the_people_ticked(signed_in, mail_account):
    client, store, _ = signed_in
    book = mail.mailing_list(store)
    for address in ("ana@example.org", "ben@example.org", "cy@example.org"):
        book.add(address)
    raw = client.get("/settings/email").data
    page = raw.decode()
    assert "No one ticked, so it goes to everyone on the list." in page
    assert 'name="to" value="ben@example.org" form="sw-test-email"' in page
    client.post(
        "/settings/email/test",
        data={"csrf_token": _csrf(raw), "to": ["ana@example.org", "CY@example.org"]},
    )
    assert [to for _, to in FakeSMTP.sent] == [["ana@example.org"], ["cy@example.org"]]
    assert "Test email sent to 2 people." in client.get("/settings/email").data.decode()
    # Removed in another tab after the page loaded
    client.post("/settings/email/test", data={"csrf_token": _csrf(raw), "to": "gone@example.org"})
    assert "The people ticked aren&#39;t on the list any more." in (
        client.get("/settings/email").data.decode()
    )
    assert len(FakeSMTP.sent) == 2


def test_a_failed_email_is_recorded_and_never_stops_the_checks(signed_in, mail_account):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    mail.mailing_list(store).add("ana@example.org")
    store.set_pref("email_on", "yes")
    store.set_source(sources.API)
    mail_account.setenv("ALERT_EMAIL_PASSWORD", "wrong")
    mail.after_check(store, inv.load(), item["check_id"])
    last = store.emails()[0]
    assert last["ok"] == 0 and "turned down the password" in last["message"]
    assert "turned down the password" in client.get("/settings/email").data.decode()
    mail_account.setenv("ALERT_EMAIL_PASSWORD", "secret")
    response = client.post(
        "/settings/email/test", data={"csrf_token": _csrf(client.get("/settings/email").data)}
    )
    assert response.status_code == 302
    assert "Test email sent to 1 person." in client.get("/settings/email").data.decode()
    assert FakeSMTP.sent[-1][0]["Subject"] == "StrawberryWatch test email"

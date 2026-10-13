"""
The local dashboard's pages, through Flask's test client: the security list,
then the build spec's checks for steps 3 and 4, against a store with one real
replay check in it.
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
from strawberrywatch.watch import runner, sources
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

SPILL = "anomaly_2025_06_12_spill_sf"


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
def replayed(tmp_path_factory, models):
    folder = tmp_path_factory.mktemp("replayed")
    store = st.Store(folder)
    store.set_source(sources.REPLAY, event=SPILL)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    runner.run_check(store, models)
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
def signed_in(replayed, models, tmp_path):
    app, store, watcher = _app(replayed, models, tmp_path)
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


def test_a_host_that_isnt_this_computer_is_refused(replayed, models, tmp_path):
    """Build order step 1."""
    app, _, _ = _app(replayed, models, tmp_path)
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


def test_the_login_link_works_once_and_sets_a_strict_cookie(replayed, models, tmp_path):
    app, _, _ = _app(replayed, models, tmp_path)
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
        # No inline script but the JSON chart blocks the browser never runs
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

    def broken(_store):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(view, "reference_now", broken)
    response = client.get("/")
    assert response.status_code == 500
    assert b"Traceback" not in response.data and b"secret detail" not in response.data


# Step 3: Overview, Unusual readings, History and Models


def test_a_quiet_day_says_everything_looks_normal_and_nothing_else(models, tmp_path):
    store = st.Store(tmp_path / "quiet")
    store.set_source(sources.REPLAY, event=SPILL)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    app = create_app(store=store, models=models, watcher=FakeWatcher(), login=security.Login("t"))
    client = app.test_client()
    client.get("/login?token=t")
    assert "The first check hasn't finished yet." in client.get("/").data.decode()
    now = time.time()
    check = store.start_check(sources.REPLAY, 1, now - 3600, now, started_at=now)
    store.finish_check(check, True, "", {})
    store.save_readings(
        [(site, now - 60, 1.0, 1.0, 1.0, None, None, None) for site in inv.load().sites]
    )
    page = client.get("/").data
    assert b"Everything looks normal." in page and b"Something looks unusual" not in page
    assert b"Nothing looks unusual right now." in client.get("/unusual").data


def test_something_noticed_shows_on_overview_and_in_full_on_unusual_readings(signed_in):
    client, _, _ = signed_in
    overview = client.get("/").data.decode()
    assert "Something looks unusual at south_fork_1." in overview
    assert 'href="/unusual"' in overview and 'class="nav-count"' in overview
    page = client.get("/unusual").data.decode()
    assert "Unusual conductivity at south_fork_1" in page
    assert (
        "Conductivity was far from what the model expected at" in page
        and "Dusk Crayfish noticed." in page
    )
    assert 'data-sw-chart="score"' in page and "See what happened at south_fork_1" in page


def test_history_mixes_noticed_items_and_changes_newest_first(signed_in, models):
    client, store, _ = signed_in
    _post(client, "/settings", _settings_form(store, models, model_cobble_shoal="on"))
    client.post("/settings/apply", data={"csrf_token": _csrf(client.get("/settings/review").data)})
    page = client.get("/history").data.decode()
    change = page.index("You turned Cobble Shoal on.")
    setup = page.index("You set up StrawberryWatch.")
    item = page.index("Unusual conductivity at south_fork_1. Dusk Crayfish noticed it.")
    assert change < setup < item
    assert "Nothing unusual." in page
    for label in ("Everything", "Unusual readings only", "Settings changes only", "By hour"):
        assert label in page
    only = client.get("/history?show=changes").data.decode()
    assert "Dusk Crayfish noticed it" not in only and "You turned Cobble Shoal on." in only
    unusual = client.get("/history?show=unusual").data.decode()
    assert "Dusk Crayfish noticed it" in unusual and "You turned Cobble Shoal on." not in unusual


ROW = r'class="sw-hour(?: sw-hour--quiet)?"'


def test_by_hour_folds_quiet_hours_and_a_day_shows_every_hour(signed_in):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    page = client.get("/history?by=hour").data.decode()
    assert "Dusk Crayfish noticed it" in page and "All day" in page
    # The noticed day: the quiet hours either side fold into one row each
    hour = datetime.fromtimestamp(item["first_over"]).astimezone().hour
    section = page[page.index("2025") :]
    section = section[: section.index("</section>")]
    assert len(re.findall(ROW, section)) == 1 + (hour > 0) + (hour < 23)
    assert "– midnight" in section and "Nothing unusual." in section
    day = datetime.fromtimestamp(item["first_over"]).astimezone().date().isoformat()
    hourly = client.get(f"/history/day/{day}").data.decode()
    assert len(re.findall(ROW, hourly)) == 24 and "Dusk Crayfish noticed it" in hourly
    today = datetime.now().astimezone().date().isoformat()
    assert "1 check ran." in client.get(f"/history/day/{today}").data.decode()
    assert client.get("/history/day/not-a-day").status_code == 404


def test_notes_add_up_on_a_reading_and_show_under_it_in_history(signed_in):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    path = f"/history/{item['id']}"
    page = client.get(path).data.decode()
    assert (
        "What did each extra say?" in page
        and "Newtwork Run" in page
        and 'data-sw-chart="score"' in page
        and "No notes yet." in page
    )
    assert _post(client, path, {"note": "Went to look, cloudy water"}).status_code == 302
    assert _post(client, path, {"note": "  Clear again the next day  "}).status_code == 302
    blank = _post(client, path, {"note": "   "})
    assert blank.status_code == 200 and "Write the note before adding it" in blank.data.decode()
    notes = store.notes(noticed_id=item["id"])
    assert [n["text"] for n in notes] == ["Clear again the next day", "Went to look, cloudy water"]
    page = client.get(path).data.decode()
    assert page.index("Went to look") < page.index("Clear again")
    history = client.get("/history").data.decode()
    assert "Went to look, cloudy water" in history and "Your notes (2)" in history
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
    assert "No notes match." in client.get("/history/notes?site=oxford").data.decode()
    assert "No notes match." in client.get("/history/notes?from=2999-01-01").data.decode()

    response = client.get("/history/notes/export?format=csv")
    assert response.mimetype == "text/csv"
    assert response.headers["Content-Disposition"].startswith("attachment;")
    text = response.data.decode("utf-8")
    assert text.startswith("\ufeff")
    rows = list(csv.reader(io.StringIO(text[1:])))
    assert rows[0][:4] == ["Note written", "Note", "Site", "Site name"]
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


def test_models_shows_all_six_cards_with_on_and_off_only_on_the_two_that_run(signed_in):
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
    assert page.count("></span>Not ready yet</span>") == 4
    assert "></span>On</span>" in page and "></span>Off</span>" in page


# Step 4: Settings, Advanced settings, review and apply


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
    assert "You have 1 change that isn&#39;t saved yet." in again
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
    assert "nothing to save" in page
    assert 'id="sw-discard" form="sw-discard-form" hidden' in page


def test_autosave_answers_with_the_banner(signed_in, models):
    client, store, _ = signed_in
    response = _post(client, "/settings", _settings_form(store, models, rain=""), fetch=True)
    assert response.json["changes"] == 1 and "isn't saved yet" in response.json["banner"]


def test_review_lists_every_change_before_and_after(signed_in, models):
    client, store, _ = signed_in
    _post(
        client, "/settings", _settings_form(store, models, alerts="more", model_cobble_shoal="on")
    )
    page = client.get("/settings/review").data.decode()
    assert "How many alerts" in page and "Before: Balanced" in page and "After: More" in page
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
    assert "Changes saved. Checking the creek now." in client.get("/").data.decode()


def test_turning_a_model_off_folds_its_settings_and_keeps_them(signed_in, models):
    client, store, _ = signed_in
    form = _advanced_form(
        store, models, model_dusk_crayfish="off", scale_dusk_crayfish_level_shift="1.3"
    )
    assert _post(client, "/settings/advanced", form).status_code == 302
    spec = store.draft()["models"]["dusk_crayfish"]
    assert spec["on"] is False and spec["scales"]["level_shift"] == 1.3
    page = client.get("/settings/advanced").data.decode()
    assert "Off. Turn it on to see its settings. Its saved values are kept." in page


def test_settings_says_custom_when_advanced_values_match_no_choice(signed_in, models):
    client, store, _ = signed_in
    _post(
        client,
        "/settings/advanced",
        _advanced_form(store, models, scale_dusk_crayfish_level_shift="1.1"),
    )
    assert "You've set custom values in Advanced settings." in client.get("/settings").data.decode()


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


# Step 5 on the page


def test_dusk_crayfish_shows_leave_a_site_out_as_not_available(signed_in):
    client, _, _ = signed_in
    assert (
        "Not available. Dusk Crayfish needs all 4 sites."
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
    assert "StrawberryWatch found a problem" in page and "under half its sensors" in page
    assert "Apply changes" not in page


# Setup and the token


def test_setup_with_a_replay_applies_defaults_and_checks(models, tmp_path):
    store = st.Store(tmp_path / "new")
    watcher = FakeWatcher()
    app = create_app(store=store, models=models, watcher=watcher, login=security.Login("t"))
    client = app.test_client()
    client.get("/login?token=t")
    assert client.get("/").headers["Location"] == "/setup"
    response = _post(client, "/setup", {"source": "replay", "event": SPILL})
    assert response.status_code == 302 and watcher.asked == 1
    assert store.source()["event"] == SPILL and store.applied()[1]["models"]["dusk_crayfish"]["on"]


def test_the_token_goes_to_the_keychain_and_never_comes_back(models, tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(sources, "save_token", lambda token: saved.update(token=token))
    monkeypatch.setattr(sources, "get_token", lambda: saved.get("token"))
    store = st.Store(tmp_path / "new")
    app = create_app(store=store, models=models, watcher=FakeWatcher(), login=security.Login("t"))
    client = app.test_client()
    client.get("/login?token=t")
    assert b"Enter your API token" in _post(client, "/setup", {"source": "api", "token": ""}).data
    _post(client, "/setup", {"source": "api", "token": "my-secret-token"})
    assert saved["token"] == "my-secret-token" and store.source()["kind"] == "api"
    for path in ("/setup", "/settings/advanced", "/token", "/"):
        assert b"my-secret-token" not in client.get(path).data
    for path in (tmp_path / "new").iterdir():
        assert b"my-secret-token" not in path.read_bytes()


def test_an_unknown_site_page_is_a_404(signed_in):
    client, _, _ = signed_in
    assert client.get("/sites/..%2Fsecrets").status_code == 404
    assert client.get("/sites/not_a_site").status_code == 404
    assert 'data-sw-chart="spark"' in client.get("/sites/south_fork_1").data.decode()


# Email alerts


class FakeSMTP:
    """Stands in for the mail server: keeps what would have gone, and turns down a wrong password."""

    sent = []

    def __init__(self, host, port, timeout=None, context=None):
        self.host = host

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        if password == "wrong":
            raise smtplib.SMTPAuthenticationError(535, b"no")

    def send_message(self, message, from_addr=None, to_addrs=None):
        FakeSMTP.sent.append((message, to_addrs))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture
def mail_account(monkeypatch):
    """A mail account from the environment only, never this computer's real .env."""
    monkeypatch.setattr(mail, "load_dotenv", lambda *a, **k: None)
    for name, value in {
        "ALERT_EMAIL_SENDER": "watch@example.org",
        "ALERT_EMAIL_PASSWORD": "secret",
        "SMTP_SERVER": "smtp.example.org",
        "SMTP_PORT": "587",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(mail.smtplib, "SMTP", FakeSMTP)
    FakeSMTP.sent = []
    return monkeypatch


def test_people_are_added_and_removed_from_the_email_list_straight_away(signed_in, mail_account):
    client, store, _ = signed_in
    assert "Email alerts" in client.get("/settings").data.decode()
    assert (
        _post(client, "/settings/email", {"email": " ana@example.org ", "name": "Ana"}).status_code
        == 302
    )
    again = _post(client, "/settings/email", {"email": "ANA@example.org"}).data.decode()
    assert "ANA@example.org is already on the list" in again
    bad = _post(client, "/settings/email", {"email": "ana at example"}).data.decode()
    assert "Enter a whole email address" in bad
    assert [(p["email"], p["name"]) for p in store.recipients()] == [("ana@example.org", "Ana")]
    page = client.get("/settings/email").data.decode()
    assert "ana@example.org" in page and "Sending is off. Nobody is emailed." in page
    person = store.recipients()[0]
    removed = client.post(
        f"/settings/email/{person['id']}/remove",
        data={"csrf_token": _csrf(client.get("/settings/email").data)},
    )
    assert removed.status_code == 302 and store.recipients() == []


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
    assert "it emails the people on its email list" in client.get("/").data.decode()


def test_a_check_emails_what_it_flagged_with_every_address_hidden(
    signed_in, mail_account, tmp_path
):
    _, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    store.add_recipient("ana@example.org")
    store.add_recipient("ben@example.org")
    inventory = inv.load()
    # Off: nothing goes
    mail.after_check(store, inventory, item["check_id"])
    assert FakeSMTP.sent == []
    # On, but a replay: still nothing
    store.set_pref("email_on", "yes")
    mail.after_check(store, inventory, item["check_id"])
    assert FakeSMTP.sent == []
    # On, and a live source: one email for the check, to both
    store.set_source(sources.CSV, folder=str(tmp_path))
    mail.after_check(store, inventory, item["check_id"])
    ((message, to),) = FakeSMTP.sent
    assert to == ["ana@example.org", "ben@example.org"]
    assert message["To"] == "undisclosed-recipients:;" and "ana@" not in message.as_string()
    assert message["Subject"] == "StrawberryWatch: unusual conductivity at south_fork_1"
    body = message.get_content()
    assert "Dusk Crayfish noticed." in body and "take you off it" in body
    assert store.emails()[0]["ok"] == 1 and store.emails()[0]["people"] == 2


def test_a_failed_email_is_recorded_and_never_stops_the_checks(signed_in, mail_account, tmp_path):
    client, store, _ = signed_in
    item = store.noticed(flagged=True)[0]
    store.add_recipient("ana@example.org")
    store.set_pref("email_on", "yes")
    store.set_source(sources.CSV, folder=str(tmp_path))
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

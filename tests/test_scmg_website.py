"""
The website version: the dashboard at /strawberrywatch
behind a real Django login, with the store on sw_ tables.

There's no MySQL server here. The store runs on SQLite through store_mysql's
prefixing, and any table it misses fails with "no such table". Point
STRAWBERRYWATCH_TEST_MYSQL at a JSON login for a throwaway database to run the
store checks on real MySQL.
"""

import json
import os
import re
import sqlite3
import threading
from importlib import import_module
from pathlib import Path

import pytest
from django.urls import path

from strawberrywatch.watch import sources
from strawberrywatch.watch import store as st
from strawberrywatch.watch.store_mysql import prefixed


def _login_page(request):
    from django.http import HttpResponse

    return HttpResponse("the site's login page")


urlpatterns = [
    path("login/", _login_page, name="login"),
    path("logout/", _login_page, name="logout"),
]


@pytest.fixture(scope="module")
def django_site():
    import django
    from django.conf import settings

    if not settings.configured:
        settings.configure(
            SECRET_KEY="only-for-these-tests",
            INSTALLED_APPS=[
                "django.contrib.contenttypes",
                "django.contrib.auth",
                "django.contrib.sessions",
            ],
            DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
            SESSION_ENGINE="django.contrib.sessions.backends.signed_cookies",
            ROOT_URLCONF=__name__,
            LOGIN_URL="login",
            USE_TZ=True,
            PASSWORD_HASHERS=["django.contrib.auth.hashers.MD5PasswordHasher"],
        )
        django.setup()
    from django.core.management import call_command

    call_command("migrate", verbosity=0, run_syncdb=True)
    return settings


class PrefixedStore(st.Store):
    """Store on SQLite, every query through store_mysql's sw_ prefixing."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / "website.db"
        self._lock = threading.RLock()
        with self.db() as db:
            for statement in st.SCHEMA.split(";"):
                if statement.strip() and "TABLE IF NOT EXISTS draft (" not in statement:
                    db.execute(statement)

    def db(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return _Prefixed(conn, self._lock)

    def session_key(self):
        from strawberrywatch.watch.store_mysql import MysqlStore

        return MysqlStore.session_key(self)

    def tables(self):
        conn = sqlite3.connect(self.path)
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        conn.close()
        return names


class _Prefixed(st._Session):
    def __enter__(self):
        super().__enter__()
        return self

    def execute(self, sql, args=()):
        return self.conn.execute(prefixed(sql, "?"), args)

    def executemany(self, sql, rows):
        return self.conn.executemany(prefixed(sql, "?"), rows)


def exercise(store):
    """Most of Store, enough that every table is written and read at least once."""
    from strawberrywatch import inventory as inv
    from strawberrywatch.watch import models as wm
    from strawberrywatch.watch import settings as ws

    models = wm.registered()
    store.set_source(sources.SITE)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    first = store.apply(clean, by="ana")
    assert store.applied()[0] == first and store.version_authors()[first] == "ana"
    store.save_draft({**clean, "repeat_hours": 9.0}, owner="ben")
    assert store.draft("ben")["repeat_hours"] == 9.0 and store.draft_base("ben") == first
    check = store.start_check(sources.SITE, first, 0.0, 900.0, detail=None)
    store.finish_check(check, True, "", {"dusk_crayfish": 1.0}, weather_ok=True)
    assert store.last_finished_check(kinds=sources.LIVE)["id"] == check
    store.save_readings([("oxford", 900.0, 1.0, 2.0, 3.0, None, None, 4.1)])
    assert store.last_readings(live=True)["oxford"]["battery"] == 4.1
    store.save_weather([(900.0, 0.2)])
    item = store.add_noticed(
        check,
        {
            "model": "dusk_crayfish",
            "site": "oxford",
            "variable": "conductivity",
            "rule": "forecast_residual",
            "first_over": 900.0,
            "last_over": 900.0,
            "peak": 9.0,
            "base": 3.0,
        },
        True,
        {},
        1.0,
    )
    store.extend_noticed(item, 1800.0, 12.0)
    assert store.noticed(kinds=sources.LIVE)[0]["last_over"] == 1800.0
    store.add_note(item, "went to look", author="ana")
    assert store.notes(site="oxford")[0]["author"] == "ana"
    assert store.add_recipient("ana@example.org") and not store.add_recipient("ana@example.org")
    other = store.label_named("Other")
    assert {lb["name"] for lb in store.labels()} == set(st.STARTING_LABELS)
    assert store.add_label("Volunteers") and not store.add_label("volunteers")
    store.set_label(store.recipients()[0]["id"], other["id"])
    assert (
        store.recipient_by_token(store.recipients()[0]["unsubscribe_token"])["label_id"]
        == other["id"]
    )
    store.record_email_result("ana@example.org", "alert", True, "Sent")
    store.log_email("alert", 1, True, "subject", check)
    store.set_pref("email_on", "yes")
    assert store.pref("email_on") == "yes"
    store.prune_daily()
    with pytest.raises(st.Conflict):
        store.apply(clean, base_id=first - 1)


def test_every_store_query_reaches_only_sw_tables(tmp_path):
    store = PrefixedStore(tmp_path / "site")
    exercise(store)
    assert store.tables() and all(t.startswith("sw_") for t in store.tables())


@pytest.mark.skipif(
    not os.getenv("STRAWBERRYWATCH_TEST_MYSQL"), reason="no throwaway MySQL database given"
)
def test_the_store_on_mysql_itself():
    from strawberrywatch.watch.store_mysql import MysqlStore

    store = MysqlStore(json.loads(os.environ["STRAWBERRYWATCH_TEST_MYSQL"]))
    store.create_tables()
    exercise(store)


# Through the site, with a real Django session


class _SignedIn:
    """A Django session for one user, put in the test client's cookie jar when used."""

    def __init__(self, user):
        from django.conf import settings
        from django.contrib.auth import BACKEND_SESSION_KEY, HASH_SESSION_KEY, SESSION_KEY

        session = import_module(settings.SESSION_ENGINE).SessionStore()
        session[SESSION_KEY] = str(user.pk)
        session[BACKEND_SESSION_KEY] = "django.contrib.auth.backends.ModelBackend"
        session[HASH_SESSION_KEY] = user.get_session_auth_hash()
        session.save()
        self.name, self.value = settings.SESSION_COOKIE_NAME, session.session_key


def _session_cookie(user):
    return _SignedIn(user)


def _site(tmp_path):
    """The site's own app with the dashboard mounted, as INSTALL.md says to."""
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from werkzeug.test import Client

    from strawberrywatch import inventory as inv
    from strawberrywatch.integrations.scmg_website import MOUNT, make_dashboard
    from strawberrywatch.watch import models as wm
    from strawberrywatch.watch import settings as ws

    def django_app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"the rest of the site"]

    store = PrefixedStore(tmp_path / "site")
    models = wm.registered()
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean, by="setup_tables")
    app = DispatcherMiddleware(django_app, {MOUNT: make_dashboard(store=store)})
    return Client(app), store


def _as(client, cookie):
    """Sign the client in as whoever the cookie is for. Werkzeug sends cookies from its jar only."""
    if cookie is None:
        client.delete_cookie("sessionid", domain="localhost")
    else:
        client.set_cookie(cookie.name, cookie.value, domain="localhost")


def _get(client, url, cookie=None):
    _as(client, cookie)
    return client.get(url, base_url="https://localhost")


def _post(client, url, cookie, data, **headers):
    """A form post the way a browser sends it, Referer included, since the site is HTTPS."""
    _as(client, cookie)
    headers.setdefault("Referer", "https://localhost" + url)
    return client.post(url, base_url="https://localhost", headers=headers, data=data)


def _csrf(page):
    return re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()


def test_signed_out_goes_to_the_sites_login_and_comes_back(django_site, tmp_path):
    client, _ = _site(tmp_path)
    response = _get(client, "/strawberrywatch/history?by=hour")
    assert response.status_code == 302
    assert response.headers["Location"] == "/login/?next=/strawberrywatch/history%3Fby%3Dhour"
    assert _get(client, "/").data == b"the rest of the site"


def test_signed_in_sees_the_website_dashboard(django_site, tmp_path):
    from django.contrib.auth.models import User

    user = User.objects.create_user("ana")
    client, store = _site(tmp_path)
    cookie = _session_cookie(user)
    response = _get(client, "/strawberrywatch/", cookie)
    page = response.data.decode()
    assert response.status_code == 200, page[:500]
    assert "Data source: strawberrycreek.org sensor database" in page
    assert 'href="/strawberrywatch/setup"' not in page and ">Change<" not in page
    assert ">Workbench<" not in page and 'href="/logout/"' in page
    assert 'data-root="/strawberrywatch"' in page
    assert store.source()["kind"] == sources.SITE
    # The first page with a form starts the dashboard's own session, kept to its path
    set_cookie = _get(client, "/strawberrywatch/settings", cookie).headers.get("Set-Cookie", "")
    assert "strawberrywatch=" in set_cookie and "Path=/strawberrywatch" in set_cookie
    assert "Secure" in set_cookie and "HttpOnly" in set_cookie
    assert response.headers["Referrer-Policy"] == "same-origin"


def test_the_picker_workbench_and_replay_are_page_not_found(django_site, tmp_path):
    from django.contrib.auth.models import User

    user = User.objects.create_user("ben")
    client, _ = _site(tmp_path)
    cookie = _session_cookie(user)
    for url in (
        "/setup",
        "/setup?choose=replay",
        "/token",
        "/workbench",
        "/workbench/jobs/abcdef012345",
    ):
        assert _get(client, "/strawberrywatch" + url, cookie).status_code == 404, url
    page = _get(client, "/strawberrywatch/settings", cookie).data
    response = _post(client, "/strawberrywatch/source/live", cookie, {"csrf_token": _csrf(page)})
    assert response.status_code == 404


def test_anyone_signed_in_edits_until_an_editor_group_is_set(django_site, tmp_path):
    from django.contrib.auth.models import Group, User
    from django.test.utils import override_settings

    viewer = User.objects.create_user("cara")
    editor = User.objects.create_user("dev")
    editor.groups.add(Group.objects.create(name="StrawberryWatch editors"))
    boss = User.objects.create_superuser("erin")
    client, store = _site(tmp_path)

    def apply_change(user, hours):
        cookie = _session_cookie(user)
        page = _get(client, "/strawberrywatch/settings/advanced", cookie).data
        data = {"csrf_token": _csrf(page), "repeat_hours": str(hours)}
        _post(
            client,
            "/strawberrywatch/settings/advanced",
            cookie,
            {**data, **_advanced(store)},
            **{"X-Requested-With": "fetch"},
        )
        page = _get(client, "/strawberrywatch/settings/review", cookie).data
        return _post(client, "/strawberrywatch/settings/apply", cookie, {"csrf_token": _csrf(page)})

    assert apply_change(viewer, 7).status_code == 302
    assert store.applied()[1]["repeat_hours"] == 7.0
    assert store.version_authors()[store.applied()[0]] == "cara"
    with override_settings(STRAWBERRYWATCH_EDITOR_GROUP="StrawberryWatch editors"):
        cookie = _session_cookie(viewer)
        assert _get(client, "/strawberrywatch/settings", cookie).status_code == 200
        # View only: the email page shows no forms. The token comes from Settings
        assert (
            b"Turn on email alerts"
            not in _get(client, "/strawberrywatch/settings/email", cookie).data
        )
        page = _get(client, "/strawberrywatch/settings", cookie).data
        refused = _post(
            client,
            "/strawberrywatch/settings/email/sending",
            cookie,
            {"csrf_token": _csrf(page), "on": "yes"},
        )
        assert refused.status_code == 403 and b"view-only access" in refused.data
        assert apply_change(editor, 8).status_code == 302
        assert apply_change(boss, 9).status_code == 302
    assert store.applied()[1]["repeat_hours"] == 9.0


def _advanced(store):
    from strawberrywatch.watch import models as wm

    draft = store.applied()[1]
    data = {}
    for m in wm.registered():
        spec = draft["models"][m.name]
        data[f"model_{m.name}"] = "on" if spec["on"] else "off"
        data[f"rain_{m.name}"] = str(spec["rain_multiplier"])
        data[f"support_{m.name}"] = spec["support"]
        if "leave_out" in m.site_choice:
            data[f"leave_{m.name}"] = spec["leave_out"]
        for rule in m.rules:
            data[f"scale_{m.name}_{rule}"] = str(spec["scales"][rule])
    return data


def test_the_website_email_list_lives_in_its_database(django_site, tmp_path, monkeypatch):
    import email_validator
    from django.contrib.auth.models import User

    class Checked:
        def __init__(self, address):
            self.normalized, self.mx = address, [(10, "mx")]

    monkeypatch.setattr(email_validator, "validate_email", lambda a, **k: Checked(a))
    user = User.objects.create_user("fay")
    client, store = _site(tmp_path)
    cookie = _session_cookie(user)
    page = _get(client, "/strawberrywatch/settings/email", cookie)
    assert b"StrawberryWatch emails everyone on this list" in page.data
    assert b"Send a test email" in page.data
    _post(
        client,
        "/strawberrywatch/settings/email",
        cookie,
        {
            "csrf_token": _csrf(page.data),
            "email": "ana@example.org",
            "label": str(store.label_named("SCMG")["id"]),
        },
    )
    assert [p["email"] for p in store.recipients()] == ["ana@example.org"]

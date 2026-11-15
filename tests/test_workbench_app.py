"""Every route answers, POSTs need CSRF, other hosts are refused, no secret leaks, and the colors contrast."""

import logging
import re
import time
from pathlib import Path

import keyring
import pytest
from keyring.backend import KeyringBackend

from strawberrywatch.dashboard import security
from strawberrywatch.watch import sources
from strawberrywatch.workbench import store
from strawberrywatch.workbench.app import create_app

TOKEN = "test-token-6b1f0c"
PASSWORD = "test-password-93d2a"
HOST = {"Host": "127.0.0.1"}
EVENT = "anomaly_2025_06_12_spill_sf"
CSS = (
    Path(__file__).resolve().parent.parent
    / "strawberrywatch"
    / "workbench"
    / "static"
    / "workbench.css"
)


class Memory(KeyringBackend):
    priority = 1

    def __init__(self):
        super().__init__()
        self.saved = {}

    def get_password(self, service, username):
        return self.saved.get((service, username))

    def set_password(self, service, username, password):
        self.saved[(service, username)] = password

    def delete_password(self, service, username):
        self.saved.pop((service, username), None)


def offline(*_args, **_kwargs):
    # The real API would get the test token. This one refuses it the way a bad token would
    raise ConnectionError(f"refused {TOKEN} at https://example.invalid/?token={TOKEN}")


@pytest.fixture
def bench(tmp_path, monkeypatch):
    from strawberrywatch.workbench import app as app_module

    monkeypatch.setattr(app_module, "test_api", offline)
    # A CSV folder without weather columns would otherwise ask Open-Meteo
    monkeypatch.setattr(sources.Source, "weather", lambda self, start, end: None)
    previous = keyring.get_keyring()
    keyring.set_keyring(Memory())
    folder = tmp_path / "wb"
    login = security.Login()
    app = create_app(folder=folder, browser=True, login=login)
    client = app.test_client()
    assert client.get(f"/login?token={login.token}", headers=HOST).status_code == 302
    page = client.get("/", headers=HOST).data.decode()
    csrf = re.search(r'name="csrf-token" content="([^"]+)"', page).group(1)
    # The stub source: a folder of CSV files, one of the shipped events
    store.set_setting(folder, "csv_folder", str(sources.EVENTS_DIR / EVENT))
    store.set_setting(folder, "source", "csv")
    yield client, csrf, folder
    keyring.set_keyring(previous)


def wait(client, location):
    job = re.search(r"job=([0-9a-f]+)", location).group(1)
    for _ in range(300):
        state = client.get(f"/jobs/{job}", headers=HOST).get_json()
        if state["state"] != "running":
            return state
        time.sleep(0.5)
    raise AssertionError("job never finished")


def test_every_route_answers_and_no_secret_leaks(bench, caplog):
    client, csrf, folder = bench
    caplog.set_level(logging.DEBUG)
    responses = []

    def get(url):
        response = client.get(url, headers=HOST)
        assert response.status_code == 200, url
        responses.append(response.data)
        return response

    for kind, field in (("api", "api_token"), ("mysql", "mysql_password")):
        value = TOKEN if kind == "api" else PASSWORD
        data = {"csrf_token": csrf, "what": "token" if kind == "api" else "mysql", field: value}
        if kind == "mysql":
            data.update(mysql_host="127.0.0.1:1", mysql_database="creek", mysql_user="reader")
        assert client.post("/settings", headers=HOST, data=data).status_code == 302
        client.post(f"/settings/test/{kind}", headers=HOST, data={"csrf_token": csrf})
        page = get("/settings").data.decode()
        assert "Could not connect:" in page

    for url in (
        "/",
        "/sensors",
        "/sensors?show=quiet&site=oxford",
        "/events",
        "/events?kind=rain",
        "/events/new",
        f"/events/{EVENT}",
        f"/api/event-series?event={EVENT}&site=south_fork_1&reading=conductivity",
        "/settings",
        "/manual/getting-started",
        "/manual?q=false%20alarms",
        "/manual?q=nothing-matches",
        "/calibrations",
        "/tuning/pick",
        "/tuning/pick?model=dusk_crayfish",
        "/export/sensors.csv",
        "/export/events.csv",
    ):
        get(url)

    response = client.post(
        "/tuning/pick",
        headers=HOST,
        data={
            "csrf_token": csrf,
            "source": "csv",
            "sites": ["south_fork_1", "south_fork_2", "north_fork_0", "oxford"],
            "readings": ["conductivity", "depth", "temperature"],
            "start_date": "2025-06-10",
            "end_date": "2025-06-16",
            "leave_out_events": "yes",
            "leave_out_rain": "yes",
        },
    )
    assert wait(client, response.headers["Location"])["state"] == "done"
    for url in (
        "/tuning/explore",
        "/tuning/explore?site=oxford&reading=depth",
        "/api/series?site=oxford&reading=depth",
        "/api/typical-day?site=oxford&reading=depth",
        "/api/histogram?site=oxford&reading=depth",
        "/export/readings.csv",
        "/tuning/recalibrate",
    ):
        get(url)
    response = client.post(
        "/tuning/check",
        headers=HOST,
        data={"csrf_token": csrf, "model": "dusk_crayfish", "false_alarms": "week"},
    )
    assert wait(client, response.headers["Location"])["state"] == "done"
    page = get("/tuning/recalibrate?model=dusk_crayfish&false_alarms=week").data.decode()
    assert "Alerts on the picked data:" in page and "Not run yet" not in page
    response = client.post(
        "/tuning/save",
        headers=HOST,
        data={"csrf_token": csrf, "model": "dusk_crayfish", "false_alarms": "week"},
    )
    assert response.status_code == 302
    get("/tuning/recalibrate?saved=1")
    saved = store.in_use(folder)["dusk_crayfish"]
    for url in (
        "/calibrations",
        f"/calibrations/dusk_crayfish/{saved}/results",
        f"/calibrations/dusk_crayfish/{saved}/summary.pdf",
        f"/calibrations/dusk_crayfish/{saved}/compare",
    ):
        get(url)
    assert (
        client.post(
            "/calibrations/dusk_crayfish/shipped/use", headers=HOST, data={"csrf_token": csrf}
        ).status_code
        == 302
    )
    assert store.in_use(folder)["dusk_crayfish"] == store.SHIPPED

    files = b"".join(p.read_bytes() for p in Path(folder).rglob("*") if p.is_file())
    for secret in (TOKEN, PASSWORD):
        assert all(secret.encode() not in body for body in responses)
        assert secret.encode() not in files
        assert secret not in caplog.text


def test_posts_without_csrf_fail(bench):
    client, _, _ = bench
    for url in ("/settings", "/tuning/pick", "/tuning/notes", "/settings/clear-readings"):
        assert (
            client.post(url, headers=HOST, data={"what": "theme", "theme": "dark"}).status_code
            == 400
        )


def test_another_host_is_refused(bench):
    client, _, _ = bench
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403


def test_signed_out_gets_nothing(tmp_path):
    app = create_app(folder=tmp_path / "wb", browser=True)
    client = app.test_client()
    assert client.get("/", headers=HOST).status_code == 403


# Contrast, WCAG 2 relative luminance


def tokens(block):
    return dict(re.findall(r"--(wb-[a-z-]+):\s*(#[0-9A-Fa-f]{6})", block))


def luminance(color):
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


PAIRS = (
    ("wb-ink", "wb-ground"),
    ("wb-ink", "wb-surface"),
    ("wb-ink", "wb-rail"),
    ("wb-link", "wb-ground"),
    ("wb-link", "wb-rail"),
    ("wb-on-accent", "wb-accent"),
    ("wb-axis", "wb-surface"),
)


def test_light_and_dark_tokens_contrast():
    css = CSS.read_text()
    light = tokens(css.split(":root {", 1)[1].split("}", 1)[0])
    dark = tokens(css.split('html[data-theme="dark"] {', 1)[1].split("}", 1)[0])
    system = tokens(css.split('html[data-theme="system"] {', 1)[1].split("}", 1)[0])
    assert dark == system
    for name, palette in (("light", light), ("dark", dark)):
        for fg, bg in PAIRS:
            ratio = contrast(palette[fg], palette[bg])
            assert ratio >= 4.5, f"{name} {fg} on {bg} is {ratio:.2f}:1"

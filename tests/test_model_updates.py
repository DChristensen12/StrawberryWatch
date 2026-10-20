"""
Model updates, with a folder standing in for GitHub. The test runs are real.
"""

import hashlib
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path

import pytest

from strawberrywatch import inventory as inv
from strawberrywatch.dashboard import create_app, profiles, security
from strawberrywatch.dashboard.security import SiteLogin
from strawberrywatch.integrations.scmg_website import NoWatcher
from strawberrywatch.watch import model_updates as mu
from strawberrywatch.watch import models as wm
from strawberrywatch.watch import runner, sources
from strawberrywatch.watch import settings as ws
from strawberrywatch.watch import store as st

ROOT = Path(__file__).resolve().parent.parent
CALIBRATION = "water_strider_calibration.json"


@pytest.fixture(scope="module")
def models():
    return wm.registered()


def _github(folder, change=None):
    """A main branch in a folder: the installed files and fingerprints, with change(files) made."""
    files = folder / mu.FILES_IN_REPO
    files.mkdir(parents=True)
    hashes = mu.installed_hashes()
    for name in hashes:
        shutil.copy(ROOT / "checkpoints" / name, files / name)
    if change:
        change(files)
    for name in hashes:
        hashes[name] = hashlib.sha256((files / name).read_bytes()).hexdigest()
    listed = folder / mu.HASHES_IN_REPO
    listed.parent.mkdir(parents=True)
    listed.write_text(json.dumps(hashes))
    return mu.FolderGitHub(folder)


def _recalibrated(files):
    calibration = json.loads((files / CALIBRATION).read_text())
    calibration["made"] = "2026-10-20"
    (files / CALIBRATION).write_text(json.dumps(calibration))


def _broken_cobble(files):
    weights = files / "cobble_shoal_weights.pt"
    weights.write_bytes(weights.read_bytes()[:4096])


def _base(tmp_path):
    """The site's checkpoints folder, as installed."""
    base = tmp_path / "checkpoints"
    base.mkdir()
    for name in mu.installed_hashes():
        shutil.copy(ROOT / "checkpoints" / name, base / name)
    return base


def _store(tmp_path, models):
    store = st.Store(tmp_path / "data")
    store.set_source(sources.SITE)
    clean, _, _ = ws.check(ws.defaults(models), models, list(inv.load().sites))
    store.apply(clean)
    return store


def test_nothing_new_on_github_says_everything_is_already_up_to_date(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    result = mu.update_all(store, models, base, _github(tmp_path / "gh"), by="ana")
    assert result["headline"] == "All models are already up to date." and result["lines"] == []
    assert not (base / mu.UPDATES).exists()
    assert all(mu.current(store, m.name) is None for m in models)


def test_a_recalibration_on_main_is_checked_test_run_and_used(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    github = _github(tmp_path / "gh", _recalibrated)
    result = mu.update_all(store, models, base, github, by="ana")
    assert result["headline"] == "Updated Water Strider. The other models were already up to date."
    folder = mu.model_folder(store, "water_strider", base)
    assert folder == base / mu.UPDATES / github.latest()
    assert json.loads((folder / CALIBRATION).read_text())["made"] == "2026-10-20"
    assert mu.model_folder(store, "dusk_crayfish", base) == base
    # A check loads it from there, held to GitHub's fingerprints, not the installed ones
    assert runner.load_detector("water_strider", folder) is not None
    assert mu.history(store)[-1][1] == (
        f"Water Strider updated from GitHub ({github.latest()[:7]}) by ana."
    )
    # Pressed again, there's no newer version
    again = mu.update_all(store, models, base, github)
    assert again["headline"] == "All models are already up to date."


def test_a_file_that_doesnt_match_its_fingerprint_is_refused(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    github = _github(tmp_path / "gh", _recalibrated)
    # Changed after its fingerprint was listed
    (tmp_path / "gh" / mu.FILES_IN_REPO / CALIBRATION).write_text("{}")
    result = mu.update_all(store, models, base, github)
    assert result["headline"] == "No models were updated."
    assert result["lines"] == [
        "Water Strider wasn't updated: water_strider_calibration.json on GitHub doesn't match "
        "its listed fingerprint. It keeps the files it had."
    ]
    assert mu.current(store, "water_strider") is None


def test_a_model_that_fails_its_test_run_keeps_the_files_it_had(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    result = mu.update_all(store, models, base, _github(tmp_path / "gh", _broken_cobble))
    assert result["headline"] == "No models were updated."
    assert len(result["lines"]) == 1
    assert result["lines"][0].startswith("Cobble Shoal wasn't updated: it failed its test run")
    assert mu.current(store, "cobble_shoal") is None
    assert mu.model_folder(store, "cobble_shoal", base) == base


def test_undo_goes_back_to_the_files_before_the_last_update(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    mu.update_all(store, models, base, _github(tmp_path / "gh", _recalibrated), by="ana")
    assert mu.undo(store, "water_strider", by="ben")
    assert mu.model_folder(store, "water_strider", base) == base
    assert not mu.undo(store, "water_strider")
    assert mu.history(store)[-1][1] == "Water Strider went back to its previous files by ben."


def test_installing_newer_files_wins_over_an_older_update(tmp_path, models, monkeypatch):
    store, base = _store(tmp_path, models), _base(tmp_path)
    mu.update_all(store, models, base, _github(tmp_path / "gh", _recalibrated))
    reinstalled = {**mu.installed_hashes(), CALIBRATION: "0" * 64}
    monkeypatch.setattr(mu, "installed_hashes", lambda: reinstalled)
    assert mu.model_folder(store, "water_strider", base) == base


def test_github_out_of_reach_changes_nothing(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    result = mu.update_all(store, models, base, mu.FolderGitHub(tmp_path / "gh", offline=True))
    assert result["headline"] == "Nothing changed. Couldn't reach GitHub."
    assert not (base / mu.UPDATES).exists()


def test_one_update_runs_at_a_time(tmp_path, models):
    store = _store(tmp_path, models)
    assert mu.begin(store, "ana") and not mu.begin(store, "ben")
    # One that died without saying so stops counting after a while
    assert mu.begin(store, "ben", clock=lambda: time.time() + mu.STALE_SECONDS + 1)


def test_github_reads_every_file_at_one_commit(monkeypatch):
    urls = []
    commit = "a" * 40

    class Answer:
        def __init__(self, data):
            self.data = data

        def read(self, _limit):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake(request, timeout):
        urls.append(request.full_url)
        return Answer(commit.encode() if "api.github.com" in request.full_url else b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    github = mu.GitHub("owner/repo", "main")
    assert github.latest() == commit
    github.read(commit, "checkpoints/x.pt")
    assert urls == [
        "https://api.github.com/repos/owner/repo/commits/main",
        f"https://raw.githubusercontent.com/owner/repo/{commit}/checkpoints/x.pt",
    ]
    # A commit, not a branch name. It can't change halfway through an update
    with pytest.raises(mu.UpdateError):
        github.read("main", "checkpoints/x.pt")


# The page, on the website


class RightAway:
    """Runs the update before answering. A test sees the result straight away."""

    def __init__(self, models, base, github):
        self.models, self.base, self.github = models, base, github

    def start(self, store, by):
        if not mu.begin(store, by):
            return False
        mu.update_all(store, self.models, self.base, self.github, by=by)
        return True


def _client(store, models, updater, can_edit=True):
    app = create_app(
        store=store,
        models=models,
        watcher=NoWatcher(),
        profile=profiles.WEBSITE,
        signing_key="test",
        updater=updater,
    )
    client = app.test_client()
    client.environ_base.update({SiteLogin.USER: "ana", SiteLogin.CAN_EDIT: can_edit})
    return client


def _get(client, path):
    return client.get(path, base_url="https://localhost")


def _post(client, path, data, page):
    token = re.search(rb'name="csrf_token"[^>]*value="([^"]+)"', page).group(1).decode()
    return client.post(
        path,
        base_url="https://localhost",
        data={"csrf_token": token, **data},
        headers={"Referer": "https://localhost" + path},
        follow_redirects=True,
    )


def test_the_button_updates_every_model_and_says_how_it_went(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    updater = RightAway(models, base, _github(tmp_path / "gh", _recalibrated))
    client = _client(store, models, updater)
    assert 'href="/settings/models"' in _get(client, "/settings").data.decode()
    page = _get(client, "/settings/models").data
    text = page.decode()
    assert "No update has been run yet." in text and "Update all models from GitHub" in text
    assert text.count("Installed with StrawberryWatch") == len(models)
    done = _post(client, "/settings/models/update", {}, page).data.decode()
    assert "Updated Water Strider. The other models were already up to date." in done
    assert "From GitHub" in done and "Undo last update" in done
    again = _post(client, "/settings/models/update", {}, done.encode()).data.decode()
    assert "All models are already up to date." in again
    back = _post(client, "/settings/models/undo", {"model": "water_strider"}, again.encode())
    assert "Water Strider is back on its previous files, from the next check." in back.data.decode()
    history = _get(client, "/history?show=changes").data.decode()
    assert "Water Strider updated from GitHub" in history
    assert "Water Strider went back to its previous files by ana." in history


def test_view_only_accounts_see_the_status_but_no_buttons(tmp_path, models):
    store, base = _store(tmp_path, models), _base(tmp_path)
    client = _client(store, models, RightAway(models, base, None), can_edit=False)
    page = _get(client, "/settings/models").data
    assert b"Update all models from GitHub" not in page and b"Model files in use" in page
    refused = _post(client, "/settings/models/update", {}, _get(client, "/settings").data)
    assert refused.status_code == 403


def test_the_local_app_has_no_model_updates(tmp_path, models):
    store = _store(tmp_path, models)
    app = create_app(store=store, models=models, watcher=NoWatcher(), login=security.Login("t"))
    client = app.test_client()
    client.get("/login?token=t")
    assert client.get("/settings/models").status_code == 404
    assert "Model updates" not in client.get("/settings").data.decode()

"""
Model updates for the website, straight from the newest files on GitHub main.

main keeps every served model file under checkpoints/, with fingerprints in
strawberrywatch/watch/model_files.sha256.json. An update downloads the files
whose fingerprints changed, checks each one, and test-runs the model on a past
event before switching to it. Each update gets its own folder, and the last one
can be undone. Only model files are downloaded, not code.

    python -m strawberrywatch.watch.model_updates --by ana
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO = os.getenv("STRAWBERRYWATCH_MODELS_REPO", "DChristensen12/StrawberryWatch")
BRANCH = os.getenv("STRAWBERRYWATCH_MODELS_BRANCH", "main")
HASHES_IN_REPO = "strawberrywatch/watch/model_files.sha256.json"
FILES_IN_REPO = "checkpoints"
# Under the checkpoints folder, one folder per GitHub commit an update came from
UPDATES = "updates"
# Far bigger than any model file, but stops a wrong file from filling the disk
MAX_BYTES = 200 * 1024 * 1024
TIMEOUT_SECONDS = 30
# Every registered model can score this event as shipped
TEST_EVENT = "anomaly_2025_09_10_overnight_sf"

# In the store's prefs, where all of the website's processes see it
VERSIONS = "model_versions:{}"
STATE = "model_update_state"
LOG = "model_update_log"
# An update still marked running after this long died without saying so
STALE_SECONDS = 15 * 60


class UpdateError(RuntimeError):
    """Something an update couldn't do, in plain words for the page."""


class NotOnGitHub(UpdateError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class GitHub:
    """Files at one commit of a public repository, read without signing in."""

    def __init__(self, repo=REPO, branch=BRANCH):
        self.repo, self.branch = repo, branch

    def latest(self):
        """The branch's newest commit. Every file an update reads is from the same moment."""
        try:
            commit = self._get(
                f"https://api.github.com/repos/{self.repo}/commits/{self.branch}",
                accept="application/vnd.github.sha",
            )
            commit = commit.decode().strip()
        except UpdateError:
            # The API only allows 60 anonymous requests an hour per address, and
            # the server's address is shared. Fall back to git
            commit = self._ls_remote()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise UpdateError("GitHub didn't say which commit is newest")
        return commit

    def read(self, commit, path):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise UpdateError("that isn't a commit")
        return self._get(f"https://raw.githubusercontent.com/{self.repo}/{commit}/{path}")

    def _get(self, url, accept=None):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": "StrawberryWatch"})
        if accept:
            request.add_header("Accept", accept)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                data = response.read(MAX_BYTES + 1)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotOnGitHub(f"{url.rsplit('/', 1)[-1]} isn't on GitHub yet") from None
            raise UpdateError(f"GitHub answered {exc.code}") from None
        except OSError as exc:
            raise UpdateError(f"couldn't reach GitHub ({type(exc).__name__})") from None
        if len(data) > MAX_BYTES:
            raise UpdateError(f"{url.rsplit('/', 1)[-1]} is too big to be a model file")
        return data

    def _ls_remote(self):
        try:
            out = subprocess.run(
                [
                    "git",
                    "ls-remote",
                    f"https://github.com/{self.repo}",
                    f"refs/heads/{self.branch}",
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            raise UpdateError("couldn't reach GitHub") from None
        return out.split()[0] if out.split() else ""


class FolderGitHub:
    """
    A fake GitHub from a folder laid out like the repository, for tests and the
    test dashboard. The commit is named after the fingerprint list. Changing a
    fingerprint makes a new commit.
    """

    def __init__(self, folder, offline=False):
        self.folder, self.offline = Path(folder), offline

    def latest(self):
        if self.offline:
            raise UpdateError("couldn't reach GitHub")
        return hashlib.sha1((self.folder / HASHES_IN_REPO).read_bytes()).hexdigest()

    def read(self, commit, path):
        if self.offline:
            raise UpdateError("couldn't reach GitHub")
        if commit != self.latest():
            raise UpdateError("that commit isn't here")
        file = self.folder / path
        if not file.is_file():
            raise NotOnGitHub(f"{Path(path).name} isn't on GitHub yet")
        return file.read_bytes()


# Which files each model uses


def installed_hashes():
    from strawberrywatch.watch.runner import HASHES_PATH

    return json.loads(HASHES_PATH.read_text())


def _files(name):
    from strawberrywatch.serving import detector_class

    return tuple(detector_class(name).FILES)


def updates_of(store, name):
    """Every update a model went through, oldest first."""
    try:
        return json.loads(store.pref(VERSIONS.format(name)) or "[]")
    except ValueError:
        return []


def current(store, name):
    """
    The update a model runs on, or None for the installed files. Reinstalling
    after an update wins, since it's the newer choice.
    """
    updates = updates_of(store, name)
    if not updates:
        return None
    latest = updates[-1]
    installed = installed_hashes()
    if {f: installed.get(f) for f in _files(name)} != latest.get("installed"):
        return None
    return latest


def model_folder(store, name, base=None):
    """Where a model loads its files from: its update's folder, or the checkpoints folder itself."""
    from strawberrywatch import paths

    base = Path(base or paths.checkpoints_dir())
    update = current(store, name)
    return base / UPDATES / update["commit"] if update else base


def running_hashes(store, name):
    update = current(store, name)
    if update:
        return dict(update["hashes"])
    installed = installed_hashes()
    return {f: installed.get(f) for f in _files(name)}


# An update


def test_run(name, folder):
    """
    Loads a model from a new folder and scores the past event like a check
    would. Raises UpdateError if it can't.
    """
    from strawberrywatch import inventory as inv
    from strawberrywatch.serving import detector_class
    from strawberrywatch.watch import runner, sources

    cls = detector_class(name)
    try:
        hashes = json.loads((Path(folder) / runner.HASHES).read_text())
        runner.verify_files(cls, folder, hashes)
        detector = cls.load(folder)
        replay = sources.ReplaySource(TEST_EVENT)
        start, end = replay.window(None)
        tables = replay.tables(list(inv.load().sites), start, end)
        scores = detector.scores(tables, replay.weather(start, end))
    except Exception as exc:
        logger.warning("%s failed its test run: %s", name, exc, exc_info=True)
        raise UpdateError(
            f"it failed its test run ({type(exc).__name__}). Either its files on GitHub "
            "are broken, or they need a newer StrawberryWatch installed first"
        ) from None
    if not scores:
        raise UpdateError("its test run gave no scores")


def _download(source, commit, files, wanted, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        data = source.read(commit, f"{FILES_IN_REPO}/{name}")
        if sha256(data) != wanted[name]:
            raise UpdateError(f"{name} on GitHub doesn't match its listed fingerprint")
        part = folder / f".{name}.part"
        part.write_bytes(data)
        part.replace(folder / name)


def update_all(store, models, base, source, by=None, test=test_run, clock=time.time):
    """
    Updates every registered model to the newest files on GitHub. Returns what
    the page shows, and keeps a copy in the store.
    """
    started = clock()
    try:
        commit = source.latest()
        remote = json.loads(source.read(commit, HASHES_IN_REPO))
        if not isinstance(remote, dict):
            raise ValueError
    except (UpdateError, ValueError) as exc:
        return _finish(store, by, started, None, f"Nothing changed. {_said(exc)}", [])

    from strawberrywatch.watch import runner

    folder = Path(base) / UPDATES / commit
    updated, problems = [], []
    for model in models:
        files = _files(model.name)
        wanted = {f: remote.get(f) for f in files}
        if wanted == running_hashes(store, model.name):
            continue
        try:
            if None in wanted.values():
                missing = ", ".join(f for f, h in wanted.items() if h is None)
                raise UpdateError(f"{missing} isn't listed in the fingerprints on GitHub")
            if not (folder / runner.HASHES).exists():
                folder.mkdir(parents=True, exist_ok=True)
                (folder / runner.HASHES).write_text(json.dumps(remote, indent=2, sort_keys=True))
            _download(source, commit, files, wanted, folder)
            test(model.name, folder)
        except UpdateError as exc:
            problems.append(f"{model.title} wasn't updated: {exc}. It keeps the files it had.")
            continue
        installed = installed_hashes()
        updates = updates_of(store, model.name) + [
            {
                "commit": commit,
                "at": clock(),
                "by": by,
                "hashes": wanted,
                "installed": {f: installed.get(f) for f in files},
            }
        ]
        store.set_pref(VERSIONS.format(model.name), json.dumps(updates))
        _log(store, clock(), f"{model.title} updated from GitHub ({commit[:7]})", by)
        updated.append(model.title)

    if updated:
        headline = f"Updated {_join(updated)}."
        if not problems:
            headline += " The other models were already up to date."
    elif problems:
        headline = "No models were updated."
    else:
        headline = "All models are already up to date."
    return _finish(store, by, started, commit, headline, problems)


def undo(store, name, by=None, clock=time.time):
    """Put a model back on the files it had before its last update. False when there's no update to undo."""
    if current(store, name) is None:
        return False
    updates = updates_of(store, name)[:-1]
    store.set_pref(VERSIONS.format(name), json.dumps(updates))
    title = name.replace("_", " ").title()
    _log(store, clock(), f"{title} went back to its previous files", by)
    return True


def _join(titles):
    return titles[0] if len(titles) == 1 else ", ".join(titles[:-1]) + " and " + titles[-1]


def _said(exc):
    text = (
        str(exc) if isinstance(exc, UpdateError) else "GitHub's fingerprint list couldn't be read"
    )
    return text[0].upper() + text[1:] + "."


def _log(store, at, text, by):
    try:
        entries = json.loads(store.pref(LOG) or "[]")
    except ValueError:
        entries = []
    entries.append({"at": at, "text": f"{text}{f' by {by}' if by else ''}."})
    store.set_pref(LOG, json.dumps(entries[-200:]))


def history(store):
    """(when, sentence) for each update and undo, for History."""
    try:
        return [(e["at"], e["text"]) for e in json.loads(store.pref(LOG) or "[]")]
    except (ValueError, KeyError, TypeError):
        return []


# Whether one is running


def state(store):
    try:
        return json.loads(store.pref(STATE) or "{}")
    except ValueError:
        return {}


def is_running(store, clock=time.time):
    now = state(store)
    return bool(now.get("running")) and clock() - now.get("started_at", 0) < STALE_SECONDS


def begin(store, by, clock=time.time):
    """Mark an update as running, unless one already is. False when one is."""
    if is_running(store, clock):
        return False
    store.set_pref(STATE, json.dumps({"running": True, "by": by, "started_at": clock()}))
    return True


def _finish(store, by, started, commit, headline, lines):
    result = {
        "running": False,
        "by": by,
        "started_at": started,
        "finished_at": time.time(),
        "commit": commit,
        "headline": headline,
        "lines": lines,
    }
    store.set_pref(STATE, json.dumps(result))
    return result


class ChildUpdater:
    """Runs each update in its own process. No page waits on it, and a crash only ends the update."""

    def start(self, store, by):
        if not begin(store, by):
            return False
        # Output goes where the site's own log goes
        subprocess.Popen(
            [sys.executable, "-m", "strawberrywatch.watch.model_updates", "--by", by or ""],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True


class ThreadUpdater:
    """On a thread of this process, for the test dashboard and its stand-in GitHub."""

    def __init__(self, models, base, source, test=test_run):
        self.models, self.base, self.source, self.test = models, Path(base), source, test

    def start(self, store, by):
        if not begin(store, by):
            return False
        threading.Thread(
            target=update_all,
            args=(store, self.models, self.base, self.source),
            kwargs={"by": by, "test": self.test},
            daemon=True,
        ).start()
        return True


def main(argv=None):
    from strawberrywatch import paths
    from strawberrywatch.watch import models as watch_models
    from strawberrywatch.watch.store_mysql import MysqlStore

    parser = argparse.ArgumentParser(description="Update the website's models from GitHub.")
    parser.add_argument("--by", default="", help="who pressed the button")
    args = parser.parse_args(argv)
    store = MysqlStore()
    began = time.time()
    try:
        result = update_all(
            store, watch_models.registered(), paths.checkpoints_dir(), GitHub(), by=args.by or None
        )
    except Exception as exc:
        logger.exception("model update failed")
        _finish(
            store,
            args.by or None,
            began,
            None,
            "The update stopped with an error. Models it already updated keep their new files.",
            [type(exc).__name__],
        )
        return 1
    logger.info("model update: %s", result["headline"])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    raise SystemExit(main())

"""Fingerprints for the event fixtures, recorded in data/anomalies/fixtures.sha256.json."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST = "fixtures.sha256.json"


def _root(root=None):
    from strawberrywatch import paths

    return Path(root) if root else Path(paths.anomalies_dir())


def fingerprints(root=None):
    """{folder/file.csv: sha256} for every fixture on disk."""
    from strawberrywatch.watch.runner import sha256

    root = _root(root)
    return {
        path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.glob("*/*.csv"))
    }


def recorded(root=None):
    path = _root(root) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def changed(root=None):
    """Fixtures whose fingerprint isn't the recorded one, including new and missing files."""
    now, then = fingerprints(root), recorded(root)
    return sorted(name for name in set(now) | set(then) if now.get(name) != then.get(name))


def record(root=None):
    """Writes the manifest from what's on disk and returns the names that changed."""
    names = changed(root)
    path = _root(root) / MANIFEST
    path.write_text(json.dumps(fingerprints(root), indent=2, sort_keys=True) + "\n")
    return names


if __name__ == "__main__":
    for name in record():
        print(f"recorded {name}")

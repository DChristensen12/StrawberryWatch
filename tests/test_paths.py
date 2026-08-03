"""
Guards the two ways path resolution can go wrong for a consumer who installed
this package but has no checkout: silently writing into their cwd, and doing
filesystem work at import time.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from strawberrywatch import paths


def _run_isolated(code, tmp_path, env_extra=None):
    """
    Run code in a fresh interpreter from tmp_path with no project root anywhere
    above it. Has to be a subprocess: strawberrywatch is already imported in the
    test process, and project_root() caches nothing but the modules do.
    """
    env = dict(os.environ)
    env.pop("STRAWBERRYWATCH_ROOT", None)
    env.pop("STRAWBERRYWATCH_CHECKPOINTS_DIR", None)
    # The repo is on sys.path via the editable install, which is how a consumer
    # would have it too, so this stays representative.
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )


def test_project_root_found_in_checkout():
    """From inside the repo, the walk-up finds the root by its pyproject.toml."""
    root = paths.project_root()
    assert root is not None
    assert (root / "pyproject.toml").exists()


def test_env_var_wins_over_walkup(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path))
    assert paths.project_root() == tmp_path.resolve()


def test_project_root_is_none_without_a_root(tmp_path):
    """
    No env var, no marker anywhere up the tree, so there is no honest answer.
    Returning Path.cwd() here is how files end up in a stranger's directory.
    """
    deep = tmp_path / "somewhere" / "deeper"
    deep.mkdir(parents=True)
    result = _run_isolated(
        """
        import sys
        # Pretend the package lives outside any checkout, like a site-packages
        # install, by pointing the walk-up at a directory with no markers.
        from pathlib import Path
        import strawberrywatch.paths as p
        p.__file__ = str(Path.cwd() / "fake" / "paths.py")
        print("ROOT:", p.project_root())
        """,
        deep,
    )
    assert result.returncode == 0, result.stderr
    assert "ROOT: None" in result.stdout


def test_missing_root_raises_actionable_error(tmp_path):
    """
    The error has to name both escape hatches, or whoever hits it has to read
    our source to find out how to fix it.
    """
    deep = tmp_path / "no" / "root" / "here"
    deep.mkdir(parents=True)
    result = _run_isolated(
        """
        from pathlib import Path
        import strawberrywatch.paths as p
        p.__file__ = str(Path.cwd() / "fake" / "paths.py")
        try:
            p.raw_data_dir()
        except RuntimeError as e:
            print("ERR:", e)
        else:
            print("NO ERROR RAISED")
        """,
        deep,
    )
    assert result.returncode == 0, result.stderr
    assert "NO ERROR RAISED" not in result.stdout
    assert paths.ROOT_ENV_VAR in result.stdout
    assert "explicit" in result.stdout.lower()


@pytest.mark.parametrize(
    "module",
    [
        "strawberrywatch",
        "strawberrywatch.config",
        "strawberrywatch.anomalies.metrics",
        "strawberrywatch.utils.notifier",
    ],
)
def test_import_creates_nothing(module, tmp_path):
    """
    Importing must not resolve a path or make a directory. A consumer with no
    data/ or checkpoints/ should still be able to import and use the pieces
    that do not need them.
    """
    work = tmp_path / "clean"
    work.mkdir()
    result = _run_isolated(f"import {module}", work)
    assert result.returncode == 0, result.stderr
    assert list(work.iterdir()) == [], f"{module} created {list(work.iterdir())}"

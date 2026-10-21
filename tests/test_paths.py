"""
Guards the two ways path resolution can go wrong for a consumer who installed
this package but has no checkout: writing into their cwd without a word, and
doing filesystem work at import time.
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
    test process, and the modules cache paths even though project_root()
    doesn't.
    """
    env = dict(os.environ)
    env.pop("STRAWBERRYWATCH_ROOT", None)
    env.pop("STRAWBERRYWATCH_CHECKPOINTS_DIR", None)
    # The repo is on sys.path via the editable install, as a consumer would
    # have it too.
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
    No env var and no marker up the tree, and no right answer. Returning
    Path.cwd() here is how files end up in a stranger's directory.
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
        "strawberrywatch.support_modules.spill_signatures",
        "strawberrywatch.support_modules.trial_bed",
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


def test_each_model_has_its_own_folder(tmp_path):
    (tmp_path / "water_strider").mkdir()
    assert paths.model_dir("water_strider", tmp_path) == tmp_path / "water_strider"
    # Nothing there yet, which is where a training script writes its first files
    assert paths.model_dir("cobble_shoal", tmp_path) == tmp_path / "cobble_shoal"


def test_files_loose_in_the_folder_from_an_older_install_still_load(tmp_path):
    """The website's checkpoints folder held every file loose before the per-model folders."""
    (tmp_path / "dusk_crayfish_weights.pt").write_bytes(b"")
    assert paths.model_dir("dusk_crayfish", tmp_path) == tmp_path
    # Once the model has a folder, that wins
    (tmp_path / "dusk_crayfish").mkdir()
    assert paths.model_dir("dusk_crayfish", tmp_path) == tmp_path / "dusk_crayfish"


def test_the_checkpoints_folder_has_one_folder_per_served_model():
    from strawberrywatch.serving import available, detector_class

    base = paths.checkpoints_dir()
    for model in available():
        for name in detector_class(model).FILES:
            assert (base / model / name).is_file(), f"{name} isn't in checkpoints/{model}/"
    loose = [p.name for p in base.iterdir() if p.is_file()]
    assert loose == [], f"loose in checkpoints/: {loose}"

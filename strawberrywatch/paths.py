"""
Where the data and checkpoints live.

Nothing here runs at import time. A consumer that only wants the classifier or
the alert sender should never trip over a missing data/ directory, so paths are
resolved when something actually reads or writes, not when the module loads.
"""
import os
from pathlib import Path

ROOT_ENV_VAR = "STRAWBERRYWATCH_ROOT"

# Files that mean "this is the top of a checkout." pyproject first since it is
# the thing pip looks for too.
_ROOT_MARKERS = ("pyproject.toml", ".git")


def project_root():
    """
    Find the repo root, or None if there isn't one.

    Returns None rather than guessing at the cwd. Falling back to Path.cwd() is
    how you end up writing a creek dataset into whatever directory someone
    happened to launch from.
    """
    override = os.getenv(ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    # Walk up from this file. Covers the dev checkout and pip install -e, both
    # of which leave the package sitting inside the repo.
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return None


def _require_root(what, argument):
    root = project_root()
    if root is None:
        raise RuntimeError(
            f"Cannot locate {what}: no project root found. Either set "
            f"{ROOT_ENV_VAR} to the checkout containing it, or pass {argument} "
            f"explicitly. Refusing to guess relative to the current directory."
        )
    return root


def data_dir(root=None):
    """Top of the data tree. Everything else here hangs off it."""
    return (root or _require_root("the data directory", "an explicit path")) / "data"


def raw_data_dir(root=None):
    return data_dir(root) / "raw_data"


def processed_dir(root=None):
    return data_dir(root) / "processed_data"


def rain_cache_dir(root=None):
    return data_dir(root) / "rain_cache"


def anomalies_dir(root=None):
    return data_dir(root) / "anomalies"


def checkpoints_dir(root=None):
    """
    Trained weights and metadata. Called checkpoints rather than models so it
    does not collide with strawberrywatch.models, which is code.
    """
    env = os.getenv("STRAWBERRYWATCH_CHECKPOINTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (root or _require_root("the checkpoints directory", "model_dir")) / "checkpoints"


def reports_dir(root=None):
    return (root or _require_root("the reports directory", "an output path")) / "reports"


def data_file(root=None):
    """The rolling inference cache that data_loader reads and writes."""
    return processed_dir(root) / "full_creek_gnn.csv"


def training_corpus(root=None):
    return processed_dir(root) / "training_corpus.csv"

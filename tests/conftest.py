import hashlib
import pickle
from pathlib import Path

import pytest
import torch

from strawberrywatch import paths

ROOT = Path(__file__).parent.parent
MODEL_DIR = paths.checkpoints_dir()

# The real .env holds the email account. No test may change it, or this run fails
REAL_ENV = (ROOT / ".env", ROOT / ".env.bak")


def _env_fingerprints():
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for path in REAL_ENV
    }


@pytest.fixture(scope="session", autouse=True)
def real_env_untouched():
    before = _env_fingerprints()
    yield before
    changed = [p.name for p, digest in _env_fingerprints().items() if digest != before[p]]
    assert not changed, f"the test run changed the real {', '.join(changed)}"


class _NoSMTP:
    """Stands in for smtplib.SMTP. A test that sends mail brings its own fake."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("a test tried to open a real SMTP connection")


@pytest.fixture(autouse=True)
def no_real_mail(monkeypatch, tmp_path, real_env_untouched):
    """A .env of each test's own, no SMTP, no Cloudflare, and the real .env checked after."""
    import smtplib

    import requests

    from strawberrywatch.dashboard import mail

    monkeypatch.setattr(mail, "env_path", lambda: tmp_path / ".env")
    monkeypatch.setattr(smtplib, "SMTP", _NoSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _NoSMTP)
    post = requests.Session.post

    def guarded(self, url, *args, **kwargs):
        if "cloudflare.com" in str(url):
            raise AssertionError("a test tried to reach Cloudflare's email API")
        return post(self, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "post", guarded)
    yield
    changed = [
        p.name for p, digest in _env_fingerprints().items() if digest != real_env_untouched[p]
    ]
    if changed:
        pytest.fail(f"this test changed the real {', '.join(changed)}")


def _load_metadata(model_name):
    path = paths.model_dir(model_name, MODEL_DIR) / f"{model_name}_metadata.pkl"
    if not path.exists():
        pytest.skip(
            f"No {model_name}_metadata.pkl found. Run "
            f"'python main.py --mode train --model {model_name}' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_model(model_name, metadata):
    from strawberrywatch.config import Config

    weights_path = paths.model_dir(model_name, MODEL_DIR) / f"{model_name}_weights.pt"
    if not weights_path.exists():
        pytest.skip(f"No {model_name}_weights.pt found.")

    import main
    from strawberrywatch.models import model_calls

    model_cls = main._MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        pytest.skip(f"'{model_name}' is not in main._MODEL_REGISTRY.")
    model = model_calls.build_from_metadata(model_cls, metadata, device=Config.DEVICE)

    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model


# Each test runs once per model, against that model's own trained threshold and node_mask
# Only sequence-contract models. Cobble Shoal scores every node at once against its
# calibration's POT threshold, and scripts/run_audit_comparison.py grades it instead
_MODELS = [
    ("dusk_crayfish", True),
]


@pytest.fixture(scope="session", params=_MODELS, ids=[m[0] for m in _MODELS])
def model_bundle(request):
    name, use_mask = request.param
    metadata = _load_metadata(name)
    model = _build_model(name, metadata)
    return {"name": name, "model": model, "metadata": metadata, "use_mask": use_mask}


@pytest.fixture(scope="session")
def edge_index():
    from strawberrywatch.utils.graph_utils import create_graph_topology

    ei, _, _ = create_graph_topology()
    return ei

import pickle
from pathlib import Path

import pytest
import torch

from strawberrywatch import paths

ROOT = Path(__file__).parent.parent
MODEL_DIR = paths.checkpoints_dir()


def _load_metadata(model_name):
    path = MODEL_DIR / f"{model_name}_metadata.pkl"
    if not path.exists():
        pytest.skip(
            f"No {model_name}_metadata.pkl found. Run "
            f"'python main.py --mode train --model {model_name}' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_model(model_name, metadata):
    from strawberrywatch.config import Config

    weights_path = MODEL_DIR / f"{model_name}_weights.pt"
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


# Each test runs once per model, judged against that model's own trained
# threshold from its own metadata. The model gets its node_mask passed so masked
# pooling and feature propagation switch on.
#
# Only sequence-contract models belong here. The event tests drive the model
# with (batch, seq_len, sites, features) windows, and Cobble Shoal reads
# per-node series off the node registry instead, so adding it would not fail on
# a keyword argument, it would have nothing to read.
#
# The adapter it was waiting on landed as preprocessing/node_windows.py, so the
# blocker is no longer ingestion. What is left is that these tests score one
# node's conductivity against a per-node threshold out of a metadata pickle,
# and Cobble Shoal has neither: it scores every node at once against the POT
# threshold in its calibration artifact. Grading it belongs in a test that
# speaks that contract, which is what scripts/run_audit_comparison.py does by
# hand today.
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

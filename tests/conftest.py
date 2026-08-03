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

    num_features = len(metadata["feature_cols"])
    if model_name == "dusk_crayfish":
        from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

        num_nodes = len(metadata["location_to_idx"])
        model = DuskCrayfish(num_node_features=num_features, num_nodes=num_nodes).to(Config.DEVICE)
    else:
        pytest.skip(f"Don't know how to build model '{model_name}'.")

    model.load_state_dict(torch.load(weights_path, map_location=Config.DEVICE, weights_only=True))
    model.eval()
    return model


# Each test runs once per model, judged against that model's own trained
# threshold from its own metadata. The model gets its node_mask passed so masked
# pooling and feature propagation switch on.
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

import pickle
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parent.parent
ANOMALY_DIR = ROOT / "data" / "anomalies"
MODEL_DIR = ROOT / "models"

# Default model the suite validates against.
MODEL_NAME = "dusk_crayfish"


@pytest.fixture(scope="session")
def model_metadata():
    path = MODEL_DIR / f"{MODEL_NAME}_metadata.pkl"
    if not path.exists():
        pytest.skip(
            f"No {MODEL_NAME}_metadata.pkl found. "
            "Run 'python main.py --mode train' first to generate models/."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def trained_model(model_metadata):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.models.Dusk_Crayfish import DuskCrayfish
    from config.config import Config

    weights_path = MODEL_DIR / f"{MODEL_NAME}_weights.pt"
    if not weights_path.exists():
        pytest.skip(f"No {MODEL_NAME}_weights.pt found.")

    num_features = len(model_metadata["feature_cols"])
    model = DuskCrayfish(num_node_features=num_features).to(Config.DEVICE)
    model.load_state_dict(
        torch.load(weights_path, map_location=Config.DEVICE, weights_only=True)
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def edge_index(model_metadata):
    import sys
    sys.path.insert(0, str(ROOT))
    from src.utils.graph_utils import create_graph_topology
    ei, _, _ = create_graph_topology()
    return ei

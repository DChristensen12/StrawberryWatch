import os

import torch
import yaml
from dotenv import load_dotenv

from strawberrywatch import paths

load_dotenv()

# Names a yaml laid over settings.yaml for one run, same sections, holding only
# what changes. The dashboard writes one per run. Trying a number doesn't mean
# editing the file everyone shares.
SETTINGS_ENV_VAR = "STRAWBERRYWATCH_SETTINGS"


def _read_overrides(path):
    """
    The override yaml at path, or {} with no path.

    A path that isn't there raises. A run that asked for its own numbers and
    got settings.yaml's instead would look fine and be wrong.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"{SETTINGS_ENV_VAR} points at {path}, which does not exist")
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path} should hold sections like settings.yaml, got {overrides!r}")
    return overrides


def _layered(base, overrides):
    """base with overrides laid over it a section at a time. A key left out keeps its value."""
    merged = {section: dict(values or {}) for section, values in (base or {}).items()}
    for section, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f"override section {section!r} should be a mapping, got {values!r}")
        merged.setdefault(section, {}).update(values)
    return merged


class Config:
    API_TOKEN = os.getenv("SCMG_API_TOKEN")
    API_BASE_URL = os.getenv(
        "SCMG_API_BASE_URL",
        "https://www.strawberrycreek.org/api/creek-data/",
    )

    # No API key, just a User-Agent with a contact. USE_NWS_RAIN=false in .env
    # turns the rain merge off.
    NWS_STATION_ID = os.getenv("NWS_STATION_ID", "LBNL1")
    NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "SCMG-AnDeSys/1.0")
    USE_NWS_RAIN = os.getenv("USE_NWS_RAIN", "true").lower() == "true"
    USE_NWS_WEATHER = os.getenv("USE_NWS_WEATHER", "true").lower() == "true"

    # Only needed with --data-source sql. Schema is one table per site (lowercased
    # site_code); sql_client uses SHOW COLUMNS, and needs no column overrides.
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_USER = os.getenv("MYSQL_DATABASE_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_DATABASE_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE_NAME")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))

    _current_dir = os.path.dirname(__file__)
    _yaml_path = os.path.join(_current_dir, "settings.yaml")

    try:
        with open(_yaml_path) as f:
            _s = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: settings.yaml not found at {_yaml_path}. Using hardcoded defaults.")
        _s = {}

    # Set when a run brought its own values. main.py prints it, and a log says
    # which numbers it ran on.
    SETTINGS_OVERRIDES = os.getenv(SETTINGS_ENV_VAR) or None
    _s = _layered(_s, _read_overrides(SETTINGS_OVERRIDES))

    # Model architecture
    HIDDEN_DIM = _s.get("model_architecture", {}).get("hidden_dim", 16)
    GNN_LAYERS = _s.get("model_architecture", {}).get("gnn_layers", 1)
    TEMPORAL_LAYERS = _s.get("model_architecture", {}).get("temporal_layers", 1)
    DROPOUT = _s.get("model_architecture", {}).get("dropout", 0.1)
    GNN_TYPE = _s.get("model_architecture", {}).get("gnn_type", "GCN")

    # Training
    EPOCHS = _s.get("training", {}).get("epochs", 10)
    BATCH_SIZE = _s.get("training", {}).get("batch_size", 128)
    LEARNING_RATE = _s.get("training", {}).get("learning_rate", 0.001)
    PATIENCE = _s.get("training", {}).get("patience", 5)
    TRAIN_SPLIT = _s.get("training", {}).get("train_split", 0.8)

    # Detection
    THRESHOLD_PERCENTILE = _s.get("detection", {}).get("threshold_percentile", 99)
    RAIN_WINDOW_HOURS = _s.get("detection", {}).get("rain_window_hours", 12)
    RAIN_THRESHOLD_MULTIPLIER = _s.get("detection", {}).get("rain_multiplier", 2.0)
    RAIN_AMOUNT_THRESHOLD = _s.get("detection", {}).get("rain_amount_threshold", 0.1)
    POST_RAIN_DECAY_HOURS = _s.get("detection", {}).get("post_rain_decay_hours", 36)

    # Preprocessing
    IMPUTATION_LIMIT_HOURS = _s.get("preprocessing", {}).get("imputation_limit_hours", 3)

    # All 10 features go in and 10 come out, but the loss scores only these.
    # Weather and the cyclical time encodings are context. Rain 15 minutes out
    # is not learnable from conductivity/depth/temp, and scoring it adds error
    # that peaks during storms, when conductivity gradients matter most.
    SCORED_TARGET_FEATURES = ["conductivity", "depth", "temperature"]

    SEQUENCE_LENGTH = 24
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Footbridge is off the roster. The API didn't serve it, while the corpus
    # imputed it at 100% of steps. It fed oxford a synthetic upstream signal
    # and oxford's skill sat at -7 to -8. Four nodes means the model trains on
    # the same set it serves.
    LOCATIONS = ["north_fork_0", "south_fork_2", "south_fork_1", "oxford"]
    LOCATION_TO_IDX = {loc: idx for idx, loc in enumerate(LOCATIONS)}
    # How many days of recent data to retain in the local CSV cache.
    # Fetches are merged into this rolling window; older rows are trimmed.
    ROLLING_WINDOW_DAYS = int(os.getenv("ROLLING_WINDOW_DAYS", "90"))

    @classmethod
    def data_file(cls):
        """
        The rolling inference cache. A method, since it needs a project root and
        resolving that on import would break Config for anyone without a checkout.
        """
        return paths.data_file()

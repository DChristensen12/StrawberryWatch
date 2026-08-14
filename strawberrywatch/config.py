import os

import torch
import yaml
from dotenv import load_dotenv

from strawberrywatch import paths

load_dotenv()


class Config:
    API_TOKEN = os.getenv("SCMG_API_TOKEN")
    API_BASE_URL = os.getenv(
        "SCMG_API_BASE_URL",
        "https://www.strawberrycreek.org/api/creek-data/",
    )

    # No API key needed, just a descriptive User-Agent.
    # Set USE_NWS_RAIN=false in .env to disable rain-merge entirely.
    NWS_STATION_ID = os.getenv("NWS_STATION_ID", "LBNL1")
    NWS_USER_AGENT = os.getenv("NWS_USER_AGENT", "SCMG-AnDeSys/1.0")
    USE_NWS_RAIN = os.getenv("USE_NWS_RAIN", "true").lower() == "true"
    USE_NWS_WEATHER = os.getenv("USE_NWS_WEATHER", "true").lower() == "true"

    # Only needed with --data-source sql. Variable names match Night Heron's .env
    # so the same credentials work for both. Schema is one table per site (lowercased
    # site_code); sql_client uses SHOW COLUMNS, so no column overrides needed.
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
    # There was a RAIN_SATURATION_MM here, the upper end of an amount-scaled
    # ramp. Nothing read it: production applies the full multiplier as soon as
    # the lookback total clears RAIN_AMOUNT_THRESHOLD, and the only consumer was
    # a test helper that graded against a rule production does not implement.
    # Removed rather than wired, because wiring it would change the shape of the
    # production rain multiplier, which is a science decision and not a defect.

    # Preprocessing
    IMPUTATION_LIMIT_HOURS = _s.get("preprocessing", {}).get("imputation_limit_hours", 3)

    # Model still takes all 10 features as input and predicts all 10 (output
    # layer shape unchanged), but the loss only scores these. Weather (rain_mm,
    # air_temp_c, shortwave_radiation) and the 4 cyclical time encodings are
    # context, not something a creek sensor model should be forecasting. Rain
    # 15 minutes out isn't learnable from conductivity/depth/temp, and scoring
    # it just adds irreducible error that's worst exactly during storms, which
    # is when conductivity gradients (the thing we actually alert on) matter most.
    SCORED_TARGET_FEATURES = ["conductivity", "depth", "temperature"]

    SEQUENCE_LENGTH = 24
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Footbridge is off the roster. The API never served it (0 rows in the live
    # cache) while the corpus had it present at 100% of steps via imputation, so
    # the model trained on a footbridge that production never provides. It fed
    # oxford a synthetic upstream signal, which is why oxford's skill was -7 to
    # -8 while every other node sat near persistence parity. Four nodes means the
    # model trains on the same set it serves.
    LOCATIONS = ["north_fork_0", "south_fork_2", "south_fork_1", "oxford"]
    LOCATION_TO_IDX = {loc: idx for idx, loc in enumerate(LOCATIONS)}
    # How many days of recent data to retain in the local CSV cache.
    # Fetches are merged into this rolling window; older rows are trimmed.
    ROLLING_WINDOW_DAYS = int(os.getenv("ROLLING_WINDOW_DAYS", "90"))

    # NUM_FEATURES is not used anywhere; the model reads sequences.shape[3]
    # at runtime, which is the right thing. Leaving the line out intentionally;
    # if anything tries to import it, that's a stale reference to clean up.

    @classmethod
    def data_file(cls):
        """
        The rolling inference cache. A method rather than a constant because it
        needs a project root, and resolving that at class-body time would mean
        importing Config fails for anyone without a checkout.
        """
        return paths.data_file()

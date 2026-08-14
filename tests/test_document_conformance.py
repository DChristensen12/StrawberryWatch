"""
The design document against the code.

documents/Math_and_Design_of_DuskCrayfish.tex states the deployed
configuration, the graph, the score and the thresholds as facts. They drifted
apart once already: the document described a pure step rain rule, a five node
graph and one shared prediction for every node, none of which had been true for
some time. These assertions are what stops that happening again.

Read the document, not a copy of its numbers, so editing the .tex without
editing the code fails here.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

from strawberrywatch.anomalies import anomaly_detector as ad
from strawberrywatch.config import Config
from strawberrywatch.utils.graph_utils import create_graph_topology

DOC = Path(__file__).resolve().parents[1] / "documents" / "Math_and_Design_of_DuskCrayfish.tex"
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


@pytest.fixture(scope="module")
def doc():
    if not DOC.exists():
        pytest.skip(f"{DOC} not found")
    return DOC.read_text()


@pytest.fixture(scope="module")
def flat(doc):
    """
    The document with runs of whitespace collapsed. Prose assertions match
    against this so that reflowing a paragraph, which changes nothing anyone
    cares about, does not fail the suite.
    """
    return re.sub(r"\s+", " ", doc)


def table_value(doc_text, label):
    """Pull one row out of the deployed configuration table."""
    for line in doc_text.splitlines():
        if line.strip().startswith(label):
            cell = line.split("&", 1)[1]
            return cell.replace(r"\\", "").strip()
    raise AssertionError(f"no table row starting {label!r}")


@pytest.mark.parametrize(
    "label,value",
    [
        ("Sequence length", lambda: Config.SEQUENCE_LENGTH),
        ("Hidden dimension", lambda: Config.HIDDEN_DIM),
        ("Graph convolution layers", lambda: Config.GNN_LAYERS),
        ("LSTM layers", lambda: Config.TEMPORAL_LAYERS),
        ("Dropout", lambda: Config.DROPOUT),
        ("Batch size", lambda: Config.BATCH_SIZE),
        ("Learning rate", lambda: Config.LEARNING_RATE),
        ("Threshold percentile", lambda: Config.THRESHOLD_PERCENTILE),
        ("Rain lookback window", lambda: Config.RAIN_WINDOW_HOURS),
        ("Rain threshold multiplier", lambda: Config.RAIN_THRESHOLD_MULTIPLIER),
        ("Rain amount that counts as wet", lambda: Config.RAIN_AMOUNT_THRESHOLD),
        ("Post rain decay window", lambda: Config.POST_RAIN_DECAY_HOURS),
        ("Imputation gap limit", lambda: Config.IMPUTATION_LIMIT_HOURS),
    ],
)
def test_config_table_matches_config(doc, label, value):
    cell = table_value(doc, label)
    numbers = re.findall(r"-?\d+\.?\d*", cell)
    assert numbers, f"no number in table cell for {label!r}: {cell!r}"
    assert float(numbers[0]) == float(value()), f"{label}: document {cell!r}, code {value()}"


def test_document_states_the_deployed_gnn_type(doc):
    assert Config.GNN_TYPE in table_value(doc, "Graph convolution type")


def test_document_states_epochs_and_patience(doc):
    cell = table_value(doc, "Training epochs")
    assert str(Config.EPOCHS) in cell
    assert str(Config.PATIENCE) in cell


def test_document_states_the_train_split(doc):
    cell = table_value(doc, "Train / validation split")
    assert f"{Config.TRAIN_SPLIT:g}" in cell


def test_feature_count_matches_the_trained_metadata(doc, flat):
    """F is stated in the table and again in the prose. Both must agree."""
    import pickle

    meta_path = CHECKPOINTS / "dusk_crayfish_metadata.pkl"
    if not meta_path.exists():
        pytest.skip("no trained metadata")
    with open(meta_path, "rb") as fh:
        n_features = len(pickle.load(fh)["feature_cols"])

    cell = table_value(doc, "Features per node")
    assert str(n_features) in cell, f"table says {cell!r}, metadata has {n_features}"
    assert f"F = {n_features}" in flat, "the prose still states a different F"


def test_node_count_and_adjacency_match(doc, flat):
    edge_index, locations, _ = create_graph_topology()
    n_edges = edge_index.shape[1]

    assert len(locations) == 4, "code no longer has four nodes"
    assert n_edges == 3, "code no longer has three edges"
    assert "four-node core" in flat, "document no longer describes a four node core"
    assert "five-node core" not in flat, "document is back to describing five nodes"

    # the contracted edge, and the two south branch edges
    for a, b in (
        ("north fork 0", "oxford"),
        ("south fork 1", "south fork 2"),
        ("south fork 2", "oxford"),
    ):
        assert rf"\text{{{a}}} \rightarrow \text{{{b}}}" in flat, f"document lost {a} to {b}"
    assert "footbridge" not in [loc.lower() for loc in locations]


def test_footbridge_coverage_number_is_the_measured_one(flat):
    """The justification for dropping footbridge carries its number."""
    assert "1,191" in flat and "29,123" in flat and "4.1 percent" in flat


def test_score_is_absolute_per_node_and_robust(doc, flat):
    """The document's score expression against what the detector computes."""
    src = Path(ad.__file__).read_text()
    assert "np.abs(predictions[" in src, "detector no longer uses absolute error"
    assert "(errors_node - error_median) / error_iqr" in src, "robust normalisation is gone"

    assert r"\left\lvert \hat{\mathbf{Y}}_{ij} - \mathbf{Y}_{ij} \right\rvert" in flat, (
        "document no longer states absolute error"
    )
    assert r"\operatorname{IQR}_i" in doc, "document no longer states robust normalisation"
    # the pooling step legitimately averages over nodes; the score must not
    score_section = flat[flat.index("model all, alert one") : flat.index("The rain adjustment")]
    assert r"s = \frac{1}{N}" not in score_section, (
        "document is back to averaging the score over nodes"
    )


def test_threshold_derivation_matches(flat):
    assert r"\operatorname{percentile}_{99}" in flat
    assert str(Config.THRESHOLD_PERCENTILE) in flat
    assert r"\tau_i" in flat, "document no longer states a per-node threshold"


def test_sustained_crossing_count_matches(flat):
    assert "at least three scored timesteps" in flat
    assert ad.MIN_TIMESTEPS_OVER_THRESHOLD == 3


def test_level_shift_k_matches(flat):
    assert f"$k = {ad.LEVEL_SHIFT_K:g}$" in flat or f"$k = {ad.LEVEL_SHIFT_K}$" in flat
    assert ad.LEVEL_SHIFT_K == 4.0


def test_loss_channel_restriction_is_documented(doc, flat):
    for channel in Config.SCORED_TARGET_FEATURES:
        assert channel in flat, f"document does not name scored channel {channel}"
    assert r"\frac{1}{N F}" not in doc, "document is back to averaging the loss over all F"


def test_rain_rule_is_step_plus_taper_not_a_pure_step(doc, flat):
    """
    The taper is deliberate and covers the delayed first flush. Pin both the
    document and the code so neither silently reverts to the step.
    """
    assert r"D = 36" in flat or "Post rain decay window" in flat
    assert "decay" in flat.lower(), "document no longer describes the taper"

    import pandas as pd

    grid = pd.date_range("2026-04-01", periods=400, freq="15min")
    rain = pd.Series(np.zeros(len(grid)), index=grid)
    rain.iloc[20:44] = 0.5  # six hours of rain
    mult, _flags = ad._rain_multipliers(
        grid,
        rain,
        Config.RAIN_WINDOW_HOURS,
        Config.RAIN_THRESHOLD_MULTIPLIER,
        Config.RAIN_AMOUNT_THRESHOLD,
        Config.POST_RAIN_DECAY_HOURS,
    )
    m = Config.RAIN_THRESHOLD_MULTIPLIER
    assert mult[0] == 1.0, "dry before the storm should be the base threshold"
    assert mult[30] == pytest.approx(m), "full multiplier while it is raining"
    assert mult[60] == pytest.approx(m), "full multiplier inside the lookback window"

    tapering = mult[(mult > 1.0 + 1e-9) & (mult < m - 1e-9)]
    assert tapering.size > 0, "a pure step has no taper, the rain rule was reverted"
    assert mult[-1] == pytest.approx(1.0), "threshold should be back to base long after rain"

    after_lookback = mult[100:]
    assert np.all(np.diff(after_lookback) <= 1e-12), "the taper is not monotone"


def test_node_prediction_spread_is_not_zero(flat):
    """
    The document states per-node predictions and gives the measured spread.
    A model that pooled node identity away would give exactly zero here.
    """
    import pickle

    from strawberrywatch.models import contracts
    from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish

    weights = CHECKPOINTS / "dusk_crayfish_weights.pt"
    meta_path = CHECKPOINTS / "dusk_crayfish_metadata.pkl"
    if not (weights.exists() and meta_path.exists()):
        pytest.skip("no trained Dusk Crayfish checkpoint")
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)

    model = contracts.build_from_metadata(DuskCrayfish, meta)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    edge_index, locations, _ = create_graph_topology()

    torch.manual_seed(0)
    x = torch.randn(4, Config.SEQUENCE_LENGTH, len(locations), len(meta["feature_cols"]))
    with torch.no_grad():
        pred = contracts.run_sequence_model(model, x, edge_index)

    spread = float((pred.max(dim=1).values - pred.min(dim=1).values).max())
    assert spread > 0.0, "every node received the same prediction, node identity is gone"
    assert "4.592" in flat, "document no longer carries the measured spread"
    assert "0.188" in flat and "4.591" in flat, "document lost the spread decomposition"

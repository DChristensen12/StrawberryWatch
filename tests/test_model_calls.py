"""
The calling-contract layer, and the Dusk Crayfish gate that goes with it.

Dusk Crayfish's weights are the audit baseline. The two-failed-four-passed
suite, the hydrant diagnosis and the -23.9 oxford skill number are all measured
against them, so if routing its forward pass through model_calls.py moved its
output by a single bit those numbers would stop being comparable to anything
recorded before.
"""

import pytest
import torch

from strawberrywatch.models import model_calls
from strawberrywatch.models.Cobble_Shoal import CobbleShoal
from strawberrywatch.models.Dusk_Crayfish import DuskCrayfish


def _dusk(num_nodes=5, num_features=6, seed=0):
    torch.manual_seed(seed)
    model = DuskCrayfish(num_node_features=num_features, num_nodes=num_nodes)
    model.eval()
    return model


def _inputs(num_nodes=5, num_features=6, batch=3, seq_len=24, seed=1):
    torch.manual_seed(seed)
    seq = torch.randn(batch, seq_len, num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    mask = torch.ones(batch, seq_len, num_nodes, dtype=torch.bool)
    mask[:, :, 2] = False
    return seq, edge_index, mask


def test_contracts_are_declared_not_guessed():
    assert model_calls.input_contract(DuskCrayfish) == model_calls.SEQUENCE_TENSOR
    assert model_calls.input_contract(CobbleShoal) == model_calls.NESTED_NODE_BATCH


def test_dusk_crayfish_is_bit_identical_through_the_dispatch():
    """
    The gate. Old call shape against the new one, same weights, same inputs.
    """
    model = _dusk()
    seq, edge_index, mask = _inputs()

    with torch.no_grad():
        direct = model(seq, edge_index, batch_size=len(seq), num_nodes=seq.shape[2])
        routed = model_calls.run_sequence_model(model, seq, edge_index)
    assert direct.shape == routed.shape
    assert torch.equal(direct, routed), "dispatch changed the unmasked forward pass"

    with torch.no_grad():
        direct_m = model(
            seq, edge_index, batch_size=len(seq), num_nodes=seq.shape[2], node_mask=mask
        )
        routed_m = model_calls.run_sequence_model(model, seq, edge_index, mask)
    assert torch.equal(direct_m, routed_m), "dispatch changed the masked forward pass"

    # and the mask still does something, or the check above proves nothing
    assert not torch.equal(direct, direct_m), "node_mask stopped changing the output"
    moved = float((direct - direct_m).abs().max())
    print(
        f"\n[gate] Dusk Crayfish bit-identical through dispatch; mask moves output by {moved:.4f}"
    )


def test_node_mask_is_only_passed_when_accepted():
    model = _dusk()
    seq, _edge_index, mask = _inputs()
    assert model_calls.accepts_node_mask(model)
    assert "node_mask" in model_calls.sequence_forward_kwargs(model, seq, mask)
    assert "node_mask" not in model_calls.sequence_forward_kwargs(model, seq, None)

    class NoMask(torch.nn.Module):
        INPUT_CONTRACT = model_calls.SEQUENCE_TENSOR

        def forward(self, x, edge_index, batch_size, num_nodes):
            return x

    assert not model_calls.accepts_node_mask(NoMask())
    assert "node_mask" not in model_calls.sequence_forward_kwargs(NoMask(), seq, mask)


def test_nested_model_refuses_sequence_input_loudly():
    """
    Cobble Shoal reads per-node series off the node registry. Handed a
    sequence tensor it must raise, not score whatever it was given: a plausible
    looking number from the wrong input is the failure that is hard to catch.
    """
    seq, edge_index, _mask = _inputs()
    model = CobbleShoal.from_metadata({"seed": 1, "window": 24})
    with pytest.raises(model_calls.ContractMismatch, match="nested_node_batch"):
        model_calls.run_sequence_model(model, seq, edge_index)


def test_unknown_contract_is_rejected():
    class Weird:
        INPUT_CONTRACT = "interpretive_dance"

    with pytest.raises(model_calls.ContractMismatch, match="interpretive_dance"):
        model_calls.input_contract(Weird())


def test_build_from_metadata_uses_the_class_not_a_name_table():
    metadata = {"feature_cols": ["a", "b"], "location_to_idx": {"x": 0, "y": 1, "z": 2}}
    model = model_calls.build_from_metadata(DuskCrayfish, metadata)
    assert isinstance(model, DuskCrayfish)
    assert model.node_embedding.num_embeddings == 3

    class NoBuilder:
        pass

    with pytest.raises(model_calls.ContractMismatch, match="from_metadata"):
        model_calls.build_from_metadata(NoBuilder, metadata)


def test_forward_kwargs_track_the_batch_they_are_given():
    model = _dusk()
    small, _e, _m = _inputs(batch=2)
    big, _e2, _m2 = _inputs(batch=7)
    a = model_calls.sequence_forward_kwargs(model, small)
    b = model_calls.sequence_forward_kwargs(model, big)
    assert a["batch_size"] == 2 and b["batch_size"] == 7
    assert a != b, "kwargs do not depend on the input they describe"
    assert a["num_nodes"] == small.shape[2]


def test_cobble_shoal_score_is_one_number_per_node():
    """
    The detector-facing contract: the alerting path sees a score, never a
    channel. Guarded here so a future refactor cannot widen it back.
    """
    import inspect

    sig = inspect.signature(CobbleShoal.score)
    assert list(sig.parameters) == ["self", "batch", "nulls", "return_channels"]
    assert sig.parameters["return_channels"].default is False

    src = inspect.getsource(CobbleShoal.score)
    assert "combine" in src, "score() stopped combining the channels"

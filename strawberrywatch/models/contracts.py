"""
How a model wants to be called.

Dusk Crayfish takes a (batch, seq_len, sites, features) tensor and an
edge_index. Riffle Darner takes per-node series where a node is a
(site, variable) pair, so it sees 17 nodes on the same creek Dusk Crayfish sees
5 sites on. Different data model, not a different keyword argument, so
signature inspection cannot bridge it and call sites ask the model instead.
"""

from __future__ import annotations

import inspect

# A (batch, seq_len, sites, features) tensor plus edge_index, which is what
# preprocessing.data_processor.prepare_sequences_normalized produces.
SEQUENCE_TENSOR = "sequence_tensor"

# A dict of per-node series keyed by the encoder's argument names, built from
# the node registry. See models/Riffle_Darner.py.
NESTED_NODE_BATCH = "nested_node_batch"

CONTRACTS = (SEQUENCE_TENSOR, NESTED_NODE_BATCH)


class ContractMismatch(TypeError):
    """A model was handed inputs its contract does not accept."""


def input_contract(model):
    """
    Which contract a model speaks. Models that predate the declaration are
    treated as sequence models, which is what they were.
    """
    contract = getattr(model, "INPUT_CONTRACT", SEQUENCE_TENSOR)
    if contract not in CONTRACTS:
        raise ContractMismatch(
            f"{type(model).__name__} declares unknown INPUT_CONTRACT {contract!r}; "
            f"expected one of {', '.join(CONTRACTS)}"
        )
    return contract


def accepts_node_mask(model):
    """
    Whether forward() takes node_mask. Dusk Crayfish does; older checkpoints of
    it did not, and passing the argument anyway is a TypeError rather than a
    no-op, so this stays a real check and not an assumption.
    """
    try:
        return "node_mask" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def sequence_forward_kwargs(model, seq_tensor, node_mask=None):
    """
    The keyword arguments for one forward pass of a sequence-contract model.

    Raises for a nested-batch model rather than quietly dropping it, because a
    model that cannot see this input would otherwise score whatever it was
    handed and the number would look plausible.
    """
    contract = input_contract(model)
    if contract != SEQUENCE_TENSOR:
        raise ContractMismatch(
            f"{type(model).__name__} speaks {contract!r} and cannot be driven from "
            f"sequence tensors. It needs per-node series built from the node "
            f"registry, which prepare_sequences_normalized does not produce. "
            f"See PORT_PROGRESS.md for the missing ingestion adapter."
        )
    kwargs = {"batch_size": len(seq_tensor), "num_nodes": seq_tensor.shape[2]}
    if node_mask is not None and accepts_node_mask(model):
        kwargs["node_mask"] = node_mask
    return kwargs


def run_sequence_model(model, seq_tensor, edge_index, node_mask=None):
    """One forward pass, with the contract checked first."""
    return model(seq_tensor, edge_index, **sequence_forward_kwargs(model, seq_tensor, node_mask))


def build_from_metadata(model_cls, metadata, device=None):
    """
    Construct a model from a trained metadata blob.

    Each class knows its own constructor arguments, so it provides
    from_metadata and this just calls it. The alternative, a table of
    per-name construction rules living away from the models, is the thing that
    kept going stale.
    """
    builder = getattr(model_cls, "from_metadata", None)
    if builder is None:
        raise ContractMismatch(
            f"{model_cls.__name__} has no from_metadata classmethod, so nothing "
            f"knows how to construct it from a checkpoint"
        )
    model = builder(metadata)
    return model if device is None else model.to(device)

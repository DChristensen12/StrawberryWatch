import torch

from strawberrywatch.config import Config


def _create_edge_index(edges, location_to_idx):
    """Convert an edge list to PyTorch Geometric format (shape [2, num_edges])."""
    edge_list = [[location_to_idx[src], location_to_idx[dst]] for src, dst in edges]
    # .t() transposes to [2, num_edges] format required by PyTorch Geometric
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


# north_fork_0 used to reach oxford through footbridge. With footbridge off the
# roster (see Config.LOCATIONS) that edge is contracted rather than cut, since
# the water still flows that way, we just have no sensor in between.
DEFAULT_FLOW_EDGES = [
    ("north_fork_0", "oxford"),
    ("south_fork_1", "south_fork_2"),
    ("south_fork_2", "oxford"),
]


def create_graph_topology(location_to_idx=None, edges=None, device=None, announce=True):
    """
    Build the creek flow graph.

    Called with nothing it reads Config and behaves exactly as it always has,
    which is what main.py and the scripts want. Serving passes the node ordering
    in from the checkpoint instead, because node identity is positional: the
    model has one learned embedding row per node index, so feeding it a different
    ordering than it trained on silently attaches every site to the wrong
    embedding. Nothing raises. The numbers just quietly stop meaning anything.

    announce is there because a library has no business printing.
    """
    if location_to_idx is None:
        location_to_idx = Config.LOCATION_TO_IDX
    if edges is None:
        edges = DEFAULT_FLOW_EDGES
    if device is None:
        device = Config.DEVICE

    locations = [site for site, _ in sorted(location_to_idx.items(), key=lambda kv: kv[1])]

    missing = {site for edge in edges for site in edge} - set(location_to_idx)
    if missing:
        raise KeyError(
            f"edges mention sites that are not in the node ordering: {sorted(missing)}. "
            f"Known sites are {sorted(location_to_idx)}."
        )

    edge_index = _create_edge_index(edges, location_to_idx).to(device)

    if announce:
        print(
            f"graph topology: {len(locations)} nodes, {len(edges)} edges, device={edge_index.device}\n"
        )

    return edge_index, locations, location_to_idx

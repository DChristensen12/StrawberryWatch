import torch
from strawberrywatch.config import Config

def _create_edge_index(edges, location_to_idx):
    """Converts an edge list to PyTorch Geometric format (shape [2, num_edges])."""
    edge_list = [[location_to_idx[src], location_to_idx[dst]] for src, dst in edges]
    # .t() transposes to [2, num_edges] format required by PyTorch Geometric
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def create_graph_topology():
    """
    Builds the creek flow graph from the hardcoded edge list and returns the edge index.
    Matches the Colab topology.
    """
    locations = Config.LOCATIONS
    location_to_idx = Config.LOCATION_TO_IDX

    # north_fork_0 used to reach oxford through footbridge. With footbridge off
    # the roster (see Config.LOCATIONS) that edge is contracted rather than cut,
    # since the water still flows that way, we just have no sensor in between.
    edges = [
        ('north_fork_0', 'oxford'),
        ('south_fork_1', 'south_fork_2'),
        ('south_fork_2', 'oxford'),
    ]

    edge_index = _create_edge_index(edges, location_to_idx).to(Config.DEVICE)

    print(f"graph topology: {len(locations)} nodes, {len(edges)} edges, device={edge_index.device}\n")

    return edge_index, locations, location_to_idx
import torch
from config.config import Config

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

    edges = [
        ('north_fork_0', 'footbridge'),
        ('footbridge', 'oxford'),
        ('south_fork_1', 'south_fork_2'),
        ('south_fork_2', 'oxford'),
    ]

    edge_index = _create_edge_index(edges, location_to_idx).to(Config.DEVICE)

    print(f"--- Graph Topology Report ---")
    print(f"Nodes: {len(locations)} sensors")
    print(f"Edges: {len(edges)} flow connections")
    print(f"Device: {edge_index.device}")
    print(f"-----------------------------\n")

    return edge_index, locations, location_to_idx
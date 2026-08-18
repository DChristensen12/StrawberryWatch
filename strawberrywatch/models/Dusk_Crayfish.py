"""Spatial GCN plus temporal LSTM backbone, aware of missing nodes."""

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv

from strawberrywatch.config import Config
from strawberrywatch.models import model_calls


def build_symmetric_norm_adjacency(edge_index, num_nodes, device):
    """
    Build the symmetric normalized adjacency D^-1/2 (A + I) D^-1/2 for the graph.

    Feature propagation needs an undirected, self-looped, normalized adjacency to
    diffuse over. The creek graph is stored as directed flow edges, so we
    symmetrize here (water carries a signal downstream, but for filling in a
    missing reading we want information to flow both ways between neighbors).
    Self loops keep a node's own value in the mix during diffusion.

    Returns a dense (num_nodes, num_nodes) tensor. Fine at this graph's size,
    do not use this path for a large graph.
    """
    adj = torch.zeros(num_nodes, num_nodes, device=device)
    src, dst = edge_index[0], edge_index[1]
    adj[src, dst] = 1.0
    # Symmetrize: if water flows src to dst, let the signal diffuse dst to src too
    adj[dst, src] = 1.0
    # Self loops
    adj = adj + torch.eye(num_nodes, device=device)
    # D^-1/2 (A+I) D^-1/2
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    return norm_adj


def propagate_missing_features(x, node_mask, norm_adj, num_iters=40):
    """
    Fill missing node features by diffusing the known ones across the graph.

    The version this replaced treated an offline sensor as if it were reading
    the average value. An offline site between two reporting sites now ends up
    with an estimate that sits between them, which on a creek is physically
    sensible.

    x:         (batch, num_nodes, num_features) for a single timestep
    node_mask: (batch, num_nodes) bool, True where the node has real data
    norm_adj:  (num_nodes, num_nodes) symmetric normalized adjacency
    num_iters: how many diffusion steps. Convergence is fast and the result does
               not depend on the unknown init, so a few dozen is plenty here.

    Each step multiplies by norm_adj (the diffusion), then resets the known nodes
    back to their real values so they anchor the solution. Missing nodes settle
    to a smooth blend of their neighbors. Runs per feature channel automatically
    since the matmul hits all channels at once.
    """
    known = node_mask.unsqueeze(-1).float()  # (batch, num_nodes, 1)
    x_known = x * known  # zero out the missing rows to start

    out = x_known.clone()
    for _ in range(num_iters):
        # Diffuse: (num_nodes, num_nodes) times (batch, num_nodes, feat) per batch
        # einsum keeps the batch dimension clean
        out = torch.einsum("ij,bjf->bif", norm_adj, out)
        # Reset the known nodes to their real values so they stay fixed
        out = out * (1.0 - known) + x_known * known
    return out


def masked_mean_pool(h, node_mask):
    """
    Mean across nodes counting only the ones that have real data.

    h:         (batch, num_nodes, hidden)
    node_mask: (batch, num_nodes) bool, True where the node is real

    Sum the kept node rows, divide by how many were kept. If somehow every node
    at a timestep is missing, fall back to a plain mean so we never divide by
    zero (that timestep should have been filtered upstream anyway).
    """
    mask = node_mask.unsqueeze(-1).float()  # (batch, num_nodes, 1)
    summed = (h * mask).sum(dim=1)  # (batch, hidden)
    count = mask.sum(dim=1)  # (batch, 1)
    safe = count.clamp(min=1.0)
    pooled = summed / safe
    # If a row had zero real nodes, fall back to the unmasked mean for that row
    empty = count.squeeze(-1) == 0
    if empty.any():
        pooled[empty] = h[empty].mean(dim=1)
    return pooled


class DuskCrayfish(nn.Module):
    """
    Missing-node-aware temporal GNN.

    forward returns (batch, num_nodes, num_features) predictions for the next
    timestep. The extra node_mask argument is optional. Pass it to turn on
    feature propagation and masked pooling. Leave it out and every node is
    treated as present, which is what lets training run unchanged.
    """

    INPUT_CONTRACT = model_calls.SEQUENCE_TENSOR

    # What this model already does internally. The loader refuses to attach a
    # module named here, see registry.check_collision. Rain: anomaly_detector
    # applies a multiplier to Rule 1's threshold, so a rain modulator on top
    # would raise the bar twice.
    BUILTIN_SUPPORT = ("rain",)

    @classmethod
    def from_metadata(cls, metadata):
        """Rebuild from a trained metadata blob, for loading a checkpoint."""
        return cls(
            num_node_features=len(metadata["feature_cols"]),
            num_nodes=len(metadata["location_to_idx"]),
        )

    def __init__(self, num_node_features, num_nodes, node_embed_dim=8):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.gnn_type = Config.GNN_TYPE
        gnn_layers = Config.GNN_LAYERS
        temporal_layers = Config.TEMPORAL_LAYERS
        dropout = Config.DROPOUT

        # How many diffusion steps feature propagation runs. Small graph, cheap.
        self.fp_iters = 40

        # one learned vector per site so the LSTM can tell south_fork_2 apart
        # from north_fork_0 instead of every node looking the same to it
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)

        # Spatial layers, same as the original
        self.gnn_layers = nn.ModuleList()
        if self.gnn_type == "GCN":
            self.gnn_layers.append(GCNConv(num_node_features, self.hidden_dim))
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        elif self.gnn_type == "GAT":
            self.gnn_layers.append(
                GATConv(num_node_features, self.hidden_dim, heads=4, concat=True)
            )
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(GATConv(self.hidden_dim * 4, self.hidden_dim))

        # the node's own GCN output and the pooled graph context come out of the
        # same GNN stack so they're the same width, this covers both plus the
        # node embedding tacked on top
        gnn_out_dim = self.hidden_dim * 4 if self.gnn_type == "GAT" else self.hidden_dim
        lstm_input_dim = gnn_out_dim * 2 + node_embed_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0,
        )

        self.output_layer = nn.Linear(self.hidden_dim, num_node_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x_sequence, edge_index, batch_size, num_nodes, node_mask=None):
        """
        x_sequence: (batch, seq_len, num_nodes, num_features)
        node_mask:  optional (batch, seq_len, num_nodes) bool, True where a node
                    has real data at that timestep. When given, missing nodes get
                    feature-propagated before the GCN and excluded from the pool.
        """
        batch_size, seq_len, num_nodes, num_features = x_sequence.shape

        # Batched edge index for the per-timestep GCN, same trick as the original
        data_list = [
            Data(x=torch.zeros(num_nodes, num_features), edge_index=edge_index)
            for _ in range(batch_size)
        ]
        batch_loader = Batch.from_data_list(data_list).to(x_sequence.device)
        batch_edge_index = batch_loader.edge_index

        # Dense normalized adjacency for feature propagation, built once per call
        norm_adj = None
        if node_mask is not None:
            norm_adj = build_symmetric_norm_adjacency(edge_index, num_nodes, x_sequence.device)

        node_states = []
        graph_states = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :, :]  # (batch, num_nodes, num_features)

            if node_mask is not None:
                mask_t = node_mask[:, t, :]  # (batch, num_nodes)
                # Fill missing nodes by diffusing the real ones before the GCN
                x_t = propagate_missing_features(x_t, mask_t, norm_adj, num_iters=self.fp_iters)

            x_flat = x_t.reshape(-1, num_features)
            h = x_flat
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, batch_edge_index)
                h = self.activation(h)
                h = self.dropout(h)

            h = h.reshape(batch_size, num_nodes, -1)

            # graph level context. still masked so dead nodes do not drag the mean
            # down, but this gets concatenated onto each node now instead of
            # replacing it, which is what wiped out per node identity before.
            if node_mask is not None:
                h_graph = masked_mean_pool(h, node_mask[:, t, :])
            else:
                h_graph = h.mean(dim=1)

            node_states.append(h)
            graph_states.append(h_graph)

        # last timestep the GCN actually saw, propagated if node_mask filled
        # it in, raw otherwise. This is what the output delta gets added to.
        last_x_t = x_t

        node_seq = torch.stack(node_states, dim=1)
        graph_seq = torch.stack(graph_states, dim=1)
        graph_seq = graph_seq.unsqueeze(2).expand(-1, -1, num_nodes, -1)

        # one learned vector per site, same across batch and time. without this the
        # LSTM has no signal that south_fork_2 normally runs hotter than the others.
        node_ids = self.node_embedding.weight
        node_ids = node_ids.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        lstm_input = torch.cat([node_seq, graph_seq, node_ids], dim=-1)

        # fold nodes into the batch dim so the LSTM runs over time per node rather
        # than over one pooled sequence for the whole graph
        lstm_input = lstm_input.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_input)
        last_hidden = lstm_out[:, -1, :]

        # Predict the change from the last observed reading instead of the
        # absolute value from scratch. Without this the model has no notion
        # of "where things currently are" and falls back on whatever shape
        # (climatology, mostly) minimizes average error across the training
        # set. All in normalized units, same space as last_x_t, no denorm.
        delta = self.output_layer(last_hidden)
        delta = delta.reshape(batch_size, num_nodes, -1)
        predictions = last_x_t + delta
        return predictions

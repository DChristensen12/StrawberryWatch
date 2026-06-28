import math
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from config.config import Config


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Shape: (1, max_len, d_model), buffer so it moves with .to(device) but
        # isn't a learned parameter.
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class WaterStrider(nn.Module):
    """
    Same spatial backbone as DuskCrayfish, but Transformer encoder replaces LSTM.
    Named after the Water Striders that live in Strawberry Creek!
    """

    def __init__(self, num_node_features):
        super().__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.gnn_type = Config.GNN_TYPE
        gnn_layers = Config.GNN_LAYERS
        temporal_layers = Config.TEMPORAL_LAYERS
        dropout = Config.DROPOUT

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
                self.gnn_layers.append(
                    GATConv(self.hidden_dim * 4, self.hidden_dim)
                )

        transformer_input_dim = (
            self.hidden_dim * 4 if self.gnn_type == "GAT" else self.hidden_dim
        )

        # nhead must divide transformer_input_dim evenly. Pick the largest
        # power of 2 that does (works for hidden_dim in {16, 32, 64, 128}).
        nhead = max(h for h in [1, 2, 4, 8] if transformer_input_dim % h == 0)

        self.positional_encoding = PositionalEncoding(transformer_input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim,
            nhead=nhead,
            dim_feedforward=transformer_input_dim * 4,  # standard 4x rule
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_layers
        )

        self.output_layer = nn.Linear(transformer_input_dim, num_node_features)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x_sequence, edge_index, batch_size, num_nodes):
        """Same interface as DuskCrayfish.forward but runs Transformer instead of LSTM."""
        batch_size, seq_len, num_nodes, num_features = x_sequence.shape

        # Build batched edge index once (same trick as DuskCrayfish)
        data_list = [
            Data(x=torch.zeros(num_nodes, num_features), edge_index=edge_index)
            for _ in range(batch_size)
        ]
        batch_loader = Batch.from_data_list(data_list).to(x_sequence.device)
        batch_edge_index = batch_loader.edge_index

        gnn_outputs = []
        for t in range(seq_len):
            x_t = x_sequence[:, t, :, :]
            x_flat = x_t.reshape(-1, num_features)
            h = x_flat
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, batch_edge_index)
                h = self.activation(h)
                h = self.dropout(h)
            h = h.reshape(batch_size, num_nodes, -1)
            h_graph = h.mean(dim=1)  # (batch, hidden_dim)
            gnn_outputs.append(h_graph)

        temporal_input = torch.stack(gnn_outputs, dim=1)  # (batch, seq_len, hidden_dim)
        temporal_input = self.positional_encoding(temporal_input)

        # Transformer encoder, outputs same shape as input.
        temporal_output = self.transformer(temporal_input)

        # Use the final timestep's representation (like LSTM's last hidden state)
        last_hidden = temporal_output[:, -1, :]

        # Expand back to per-node and project to feature space
        expanded = last_hidden.unsqueeze(1).expand(-1, num_nodes, -1)
        predictions = self.output_layer(expanded)

        return predictions

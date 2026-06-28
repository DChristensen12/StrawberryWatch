import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from config.config import Config

class DuskCrayfish(nn.Module):
    """
    GNN-LSTM model for predicting creek sensor readings one timestep ahead.
    Each forward pass alternates spatial (GCN or GAT per timestep) and temporal (LSTM).
    Named after the crayfish that live in Strawberry Creek.
    """
    def __init__(self, num_node_features):
        super(DuskCrayfish, self).__init__()

        self.hidden_dim = Config.HIDDEN_DIM
        self.gnn_type = Config.GNN_TYPE
        gnn_layers = Config.GNN_LAYERS
        temporal_layers = Config.TEMPORAL_LAYERS
        dropout = Config.DROPOUT

        self.gnn_layers = nn.ModuleList()

        if self.gnn_type == 'GCN':
            self.gnn_layers.append(GCNConv(num_node_features, self.hidden_dim))
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
        elif self.gnn_type == 'GAT':
            self.gnn_layers.append(GATConv(num_node_features, self.hidden_dim, heads=4, concat=True))
            for _ in range(gnn_layers - 1):
                self.gnn_layers.append(GATConv(self.hidden_dim * 4, self.hidden_dim))

        lstm_input_dim = self.hidden_dim * 4 if self.gnn_type == 'GAT' else self.hidden_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0
        )

        self.output_layer = nn.Linear(self.hidden_dim, num_node_features)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x_sequence, edge_index, batch_size, num_nodes):
        """
        Standard forward pass. The edge_index is batched once up front rather than
        rebuilt per timestep, which is why batch_size and num_nodes are explicit args.
        """
        batch_size, seq_len, num_nodes, num_features = x_sequence.shape

        # Batch the edge index once up front rather than rebuilding it each timestep.
        data_list = [Data(x=torch.zeros(num_nodes, num_features), edge_index=edge_index) for _ in range(batch_size)]
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

            # Reshape back to batch and pool across nodes to get graph-level context
            h = h.reshape(batch_size, num_nodes, -1)
            h_graph = h.mean(dim=1)  # (batch_size, hidden_dim)
            gnn_outputs.append(h_graph)

        temporal_input = torch.stack(gnn_outputs, dim=1).contiguous()  # (batch_size, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(temporal_input)

        last_hidden = lstm_out[:, -1, :]

        expanded = last_hidden.unsqueeze(1).expand(-1, num_nodes, -1)
        predictions = self.output_layer(expanded)

        return predictions

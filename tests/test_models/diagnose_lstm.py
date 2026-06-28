import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

print(f"PyTorch: {torch.__version__}")

batch_size = 8
seq_len = 24
num_nodes = 5
num_features = 10
hidden_dim = 16

# Same graph topology shape as the model uses
edge_index = torch.tensor([
    [1, 0, 3, 2],
    [0, 4, 2, 4],
], dtype=torch.long)

gcn = GCNConv(num_features, hidden_dim)
lstm = nn.LSTM(
    input_size=hidden_dim,
    hidden_size=hidden_dim,
    num_layers=1,
    batch_first=True,
)

# Build batched edge index (mirrors model.py)
data_list = [
    Data(x=torch.zeros(num_nodes, num_features), edge_index=edge_index)
    for _ in range(batch_size)
]
batch_edge_index = Batch.from_data_list(data_list).edge_index
print(f"batch_edge_index shape: {batch_edge_index.shape}")

# Build input with random sequence data
x_sequence = torch.randn(batch_size, seq_len, num_nodes, num_features)

gnn_outputs = []
for t in range(seq_len):
    x_t = x_sequence[:, t, :, :]
    x_flat = x_t.reshape(-1, num_features)
    h = gcn(x_flat, batch_edge_index)
    h = torch.relu(h)
    h = h.reshape(batch_size, num_nodes, -1)
    h_graph = h.mean(dim=1)
    gnn_outputs.append(h_graph)

temporal_input = torch.stack(gnn_outputs, dim=1)
print(f"temporal_input shape: {temporal_input.shape}")
print(f"  dtype:      {temporal_input.dtype}")
print(f"  contiguous: {temporal_input.is_contiguous()}")
print(f"  device:     {temporal_input.device}")
print(f"  requires_grad: {temporal_input.requires_grad}")
print(f"  grad_fn:    {temporal_input.grad_fn}")
print(f"  has nan:    {torch.isnan(temporal_input).any().item()}")
print(f"  has inf:    {torch.isinf(temporal_input).any().item()}")

print("\nTest: LSTM in train mode")
lstm.train()
try:
    out, _ = lstm(temporal_input)
    print(f"  ✓ shape={out.shape}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")

print("\nTest: LSTM with autocast (matches what the trainer wraps)")
try:
    with torch.amp.autocast(device_type='cpu'):
        out, _ = lstm(temporal_input)
    print(f"  ✓ shape={out.shape}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")
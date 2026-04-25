"""
models/gat.py
-------------
Graph Attention Network for graph-level regression on QM9.
Built from scratch using PyTorch Geometric GATConv layers.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(nn.Module):
    """
    GAT encoder + MLP regression head.

    Architecture:
        GATConv(node_dim -> hidden_dim, heads) -> BN -> ReLU -> Dropout
        GATConv(hidden_dim -> hidden_dim, heads) -> BN -> ReLU -> Dropout  (repeated num_layers-1 times)
        global_mean_pool
        Linear(hidden_dim -> hidden_dim) -> BN -> ReLU -> Linear(hidden_dim -> 1)

    Each GATConv uses concat=True with out_channels = hidden_dim // heads,
    so the concatenated output is always hidden_dim regardless of head count.
    """

    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()

        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        )
        per_head_dim = hidden_dim // heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(node_dim, per_head_dim, heads=heads, concat=True, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, per_head_dim, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

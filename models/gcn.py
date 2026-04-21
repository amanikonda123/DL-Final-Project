"""
models/gcn.py
-------------
Graph Convolutional Network for graph-level regression on QM9.
Built from scratch using PyTorch Geometric GCNConv layers.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(nn.Module):
    """
    GCN encoder + MLP regression head.

    Architecture:
        GCNConv(node_dim -> hidden) -> ReLU -> Dropout
        GCNConv(hidden   -> hidden) -> ReLU -> Dropout   (repeated num_layers-1 times)
        global_mean_pool
        Linear(hidden -> hidden) -> ReLU -> Linear(hidden -> 1)
    """

    def __init__(self, node_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(node_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
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
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

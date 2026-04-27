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

    def _encode(self, x, edge_index, batch):
        """Shared encoder logic. Returns graph embedding and per-layer node embeddings."""
        layer_embeds = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
            layer_embeds.append(x.detach())

        x_graph = global_mean_pool(x, batch)
        return x_graph, layer_embeds

    def forward(self, x, edge_index, batch):
        x_graph, _ = self._encode(x, edge_index, batch)
        return self.head(x_graph).squeeze(-1)

    @torch.no_grad()
    def get_embedding(self, x, edge_index, batch):
        """Return (graph_embedding, list_of_node_embeddings_per_layer)."""
        self.eval()
        return self._encode(x, edge_index, batch)

    @torch.no_grad()
    def get_attention_weights(self, x, edge_index, batch):
        """Return list of (edge_index, attention_weights) per layer."""
        self.eval()
        attn_list = []
        for conv, bn in zip(self.convs, self.bns):
            x, (ei, alpha) = conv(x, edge_index, return_attention_weights=True)
            x = bn(x)
            x = torch.relu(x)
            attn_list.append((ei, alpha.detach()))
        return attn_list

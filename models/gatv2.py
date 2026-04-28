"""
models/gatv2.py
---------------
Graph Attention Network v2 for graph-level regression on QM9.
Built using PyTorch Geometric GATv2Conv layers.

Key differences from GAT (GATConv):
    1. GATv2Conv instead of GATConv — dynamic attention, full-rank interaction
    2. edge_dim=4 passed to every conv — bond type one-hots enter the score
    3. share_weights=False — W_l and W_r are separate matrices
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2(nn.Module):
    """
    GATv2 encoder + MLP regression head.

    Architecture:
        GATv2Conv(node_dim -> hidden_dim, heads, edge_dim=4) -> BN -> ReLU -> Dropout
        GATv2Conv(hidden_dim -> hidden_dim, heads, edge_dim=4) -> BN -> ReLU -> Dropout
        global_mean_pool
        Linear(hidden_dim -> hidden_dim) -> BN -> ReLU -> Linear(hidden_dim -> 1)

    Each GATv2Conv uses concat=True with out_channels = hidden_dim // heads,
    so the concatenated output is always hidden_dim regardless of head count.

    Unlike GATConv, GATv2Conv:
        - Concatenates raw node features BEFORE the linear transform,
          making attention dynamic (full-rank key-query interaction)
        - Accepts edge_dim, allowing bond types to enter the attention score
          inside the LeakyReLU — keeping the interaction coupled
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.0,
        share_weights: bool = False,
    ):
        super().__init__()

        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        )
        per_head_dim = hidden_dim // heads

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Layer 0: node_dim → hidden_dim
        self.convs.append(GATv2Conv(
            in_channels=node_dim,
            out_channels=per_head_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            share_weights=share_weights,
        ))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Layers 1..num_layers-1: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(
                in_channels=hidden_dim,
                out_channels=per_head_dim,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=dropout,
                share_weights=share_weights,
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def _encode(self, x, edge_index, edge_attr, batch):
        """Shared encoder logic. Returns graph embedding and per-layer node embeddings."""
        layer_embeds = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
            layer_embeds.append(x.detach())

        x_graph = global_mean_pool(x, batch)
        return x_graph, layer_embeds

    def forward(self, x, edge_index, edge_attr, batch):
        x_graph, _ = self._encode(x, edge_index, edge_attr, batch)
        return self.head(x_graph).squeeze(-1)

    @torch.no_grad()
    def get_embedding(self, x, edge_index, edge_attr, batch):
        """Return (graph_embedding, list_of_node_embeddings_per_layer)."""
        self.eval()
        return self._encode(x, edge_index, edge_attr, batch)
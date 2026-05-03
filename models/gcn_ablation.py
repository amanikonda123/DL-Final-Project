"""
models/gcn_ablation.py
----------------------
GCN variant for Ablation Study 3: Identity Adjacency.

Adds a `use_graph_structure` flag to the standard GCN.

When use_graph_structure=True  → standard GCNConv (message passing on real bonds)
When use_graph_structure=False → nn.Linear replaces GCNConv (no neighbor aggregation)

WHY THIS IS NOT JUST AN MLP
-----------------------------
A raw MLP cannot handle graphs directly because molecules have variable
numbers of atoms. We still need:
    1. A per-node transformation (shared weights across all atoms)
    2. A permutation-invariant global pooling to produce a fixed-size graph vector

So the "no-graph" variant is:
    node_encoder → [Linear → BN → ReLU → Dropout] × num_layers → global_mean_pool + global_max_pool → head

This is NOT a standard MLP. It is:
    - Permutation invariant (mean+max pool)
    - Variable-size input safe (pooling handles N atoms → fixed vector)
    - Identical to GCN except: no neighborhood aggregation at all
    - Every atom is updated only by its own features — bonds are invisible

This isolates the question: "Does knowing which atoms are bonded to which
actually help, or is U0 mostly predictable from atomic composition alone?"

ARCHITECTURE DIAGRAM
---------------------
use_graph_structure=True (standard GCN):
    x [N,d] ──► node_encoder ──► [GCNConv → BN → ReLU → Dropout → residual] × L
                                                    ▼
                                            global_mean_pool + global_max_pool
                                                    ▼
                                            Linear(2H→H) → BN → ReLU → Linear(H→1)

use_graph_structure=False (no-graph / node-only):
    x [N,d] ──► node_encoder ──► [Linear → BN → ReLU → Dropout → residual] × L
                                                    ▼
                                            global_mean_pool + global_max_pool
                                                    ▼
                                            Linear(2H→H) → BN → ReLU → Linear(H→1)

The only difference: GCNConv(h_i, edge_index) vs nn.Linear(h_i).
Everything else — residuals, BN, pooling, head — is identical.

Reference:
    - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
      ICLR 2017. https://arxiv.org/abs/1609.02907
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNAblation(nn.Module):
    """
    GCN encoder + MLP regression head, with optional graph structure removal.

    Args:
        node_dim (int):           Input node feature dimension (5 for topology, 11 for full)
        hidden_dim (int):         Hidden layer width
        num_layers (int):         Number of message-passing (or linear) layers
        dropout (float):          Dropout probability
        use_graph_structure (bool):
            True  → standard GCNConv — aggregates neighbor features over real bonds
            False → nn.Linear — each node updated independently, no message passing
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        use_graph_structure: bool = True,
    ):
        super().__init__()

        self.use_graph_structure = use_graph_structure

        # Projects raw node features to hidden_dim — same in both modes
        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for _ in range(num_layers):
            if use_graph_structure:
                # Standard GCN: h_i = W · mean({h_j : j ∈ N(i) ∪ {i}})
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            else:
                # No-graph: h_i = W · h_i  (atom sees only itself)
                self.convs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Head takes concatenated mean+max pool → scalar prediction
        # hidden_dim * 2 because we concat mean and max pooled vectors
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def _encode(self, x, edge_index, batch):
        """
        Core encoder. Returns (graph_embedding, per_layer_node_embeddings).

        In both modes:
            - x starts at [N, node_dim], gets projected to [N, hidden_dim]
            - Each layer applies a residual update
            - Final x is [N, hidden_dim]
            - Pooling collapses to [B, 2*hidden_dim]

        In use_graph_structure=True mode:
            conv(x, edge_index) → each atom aggregates bonded neighbors

        In use_graph_structure=False mode:
            conv(x) → each atom updated from its own features only
            edge_index is passed in but completely ignored
        """
        x = self.node_encoder(x)
        layer_embeds = [x.detach()]

        for conv, bn in zip(self.convs, self.bns):
            residual = x

            if self.use_graph_structure:
                x = conv(x, edge_index)   # GCNConv — uses bond graph
            else:
                x = conv(x)               # nn.Linear — ignores bonds entirely

            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + residual              # residual connection — same in both modes
            layer_embeds.append(x.detach())

        # Permutation-invariant pooling — handles variable N across molecules
        x_mean = global_mean_pool(x, batch)   # [B, H]
        x_max  = global_max_pool(x, batch)    # [B, H]
        x_graph = torch.cat([x_mean, x_max], dim=-1)  # [B, 2H]

        return x_graph, layer_embeds

    def forward(self, x, edge_index, batch):
        x_graph, _ = self._encode(x, edge_index, batch)
        return self.head(x_graph).squeeze(-1)

    @torch.no_grad()
    def get_embedding(self, x, edge_index, batch):
        """Return (graph_embedding, list_of_node_embeddings_per_layer)."""
        self.eval()
        return self._encode(x, edge_index, batch)

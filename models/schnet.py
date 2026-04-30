"""
models/schnet.py
----------------
Wrapper around PyTorch Geometric's SchNet for QM9 graph-level regression.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet as PyGSchNet


class QM9SchNet(nn.Module):
    """
    Thin wrapper that exposes a clean QM9-oriented interface.

    PyG SchNet expects:
      - z: atomic numbers with shape [num_atoms]
      - pos: atomic positions with shape [num_atoms, 3]
      - batch: graph assignment for each atom
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
    ):
        super().__init__()
        self.model = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            readout=readout,
            dipole=dipole,
        )

    def forward(self, z, pos, batch):
        pred = self.model(z=z, pos=pos, batch=batch)
        return pred.view(-1)

    @torch.no_grad()
    def get_embedding(self, z, pos, batch):
        """
        Return (graph_embedding, []) using a forward hook on the readout layer.
        SchNet's internals are opaque so we only get the graph-level vector.
        """
        self.eval()
        captured = {}

        def _hook(module, inp, out):
            captured["emb"] = inp[0].detach()

        handle = self.model.lin2.register_forward_hook(_hook)
        self.model(z=z, pos=pos, batch=batch)
        handle.remove()

        return captured["emb"], []

    @torch.no_grad()
    def get_interaction_graph(self, z, pos, batch):
        """
        Return a visualization-friendly summary of SchNet's geometric interactions.

        The returned edge strengths are the mean filter norms across interaction
        blocks, which provides a useful proxy for which atom pairs the model is
        emphasizing during message passing.
        """
        self.eval()
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.model.embedding(z)
        edge_index, edge_weight = self.model.interaction_graph(pos, batch)
        edge_attr = self.model.distance_expansion(edge_weight)

        edge_strengths = []
        for interaction in self.model.interactions:
            cutoff_term = 0.5 * (
                torch.cos(edge_weight * torch.pi / interaction.conv.cutoff) + 1.0
            )
            filters = interaction.conv.nn(edge_attr) * cutoff_term.view(-1, 1)
            edge_strengths.append(filters.norm(dim=-1))
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if edge_strengths:
            edge_strength = torch.stack(edge_strengths, dim=0).mean(dim=0)
        else:
            edge_strength = torch.zeros_like(edge_weight)

        atom_strength = h.norm(dim=-1)

        return {
            "pos": pos.detach(),
            "edge_index": edge_index.detach(),
            "edge_weight": edge_weight.detach(),
            "edge_strength": edge_strength.detach(),
            "atom_strength": atom_strength.detach(),
            "batch": batch.detach(),
        }

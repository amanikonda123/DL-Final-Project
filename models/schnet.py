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

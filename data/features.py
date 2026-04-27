"""
data/features.py
----------------
Controls which features are passed to each model.

QM9 node features (data.x) in PyG — 11 columns:
  [0]  Atomic number (one-hot, 5 types: H, C, N, O, F)
  [1-4] One-hot continued
  [5]  Aromatic flag
  [6]  Hybridization (sp, sp2, sp3)  ← inferred via RDKit from SMILES
  [7-8] Hybridization continued
  [9]  Number of hydrogens
  [10] Formal charge

QM9 edge features (data.edge_attr) — 4 columns:
  [0-3] Bond type one-hot: single, double, triple, aromatic

Geometric features (data.pos) — [N, 3] 3D coordinates (Angstrom)

Reference:
  - Ramakrishnan et al., Sci. Data 1:140022 (2014)
  - PyG QM9 docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html
"""

import torch
from torch_geometric.utils import remove_self_loops
from torch_geometric.transforms import RadiusGraph
from torch import Tensor


def select_features(data, mode: str = "topology"):
    """
    Preprocesses a QM9 Data object to expose the right features per model.

    Modes:
        topology  → node type (atomic one-hot) + bond type edges
                    Used by: GCN, GAT
        geometry  → adds pairwise distances as edge features
                    Used by: SchNet (distances computed internally from pos,
                    but this mode ensures pos is available and edges are built)
        full      → topology + distances + all 11 node features
                    Used by: ablation studies

    Args:
        data: PyG Data object from QM9
        mode: one of "topology", "geometry", "full"

    Returns:
        Modified Data object (does not mutate original)
    """
    data = data.clone()

    if mode == "topology":
        # Keep only atom-type one-hot (first 5 dims) as node features
        # Keep bond-type one-hot as edge features
        data.x = data.x[:, :5].float()          # [N, 5]  atom type only
        # edge_attr already has bond type [E, 4] — keep as-is
        data.edge_attr = data.edge_attr[:, :4].float()

    elif mode == "geometry":
        # Full node features + 3D positions (SchNet uses pos directly)
        data.x = data.x[:, :5].float()
        if hasattr(data, "z") and data.z is not None:
            data.z = data.z.long()
        data.pos = data.pos.float()              # [N, 3]  required for SchNet
        data.edge_attr = None                    # SchNet recomputes from distances

    elif mode == "full":
        # All 11 node features + bond type edges
        data.x = data.x.float()                 # [N, 11]
        data.edge_attr = data.edge_attr[:, :4].float()

    else:
        raise ValueError(f"Unknown feature mode: '{mode}'. Choose topology | geometry | full")

    return data


def get_feature_dims(mode: str) -> dict:
    """
    Returns input dimensionalities for a given feature mode.
    Use this to auto-configure model input layers.
    """
    dims = {
        "topology": {"node_dim": 5,  "edge_dim": 4},
        "geometry": {"node_dim": 5,  "edge_dim": 0},   # SchNet ignores edge_attr
        "full":     {"node_dim": 11, "edge_dim": 4},
    }
    if mode not in dims:
        raise ValueError(f"Unknown feature mode: '{mode}'")
    return dims[mode]

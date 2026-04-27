"""
models/__init__.py
------------------
Model factory — instantiate any registered architecture by name string.
"""

import torch.nn as nn
from .gcn import GCN
from .gat import GAT
from .schnet import QM9SchNet
from .gatv2 import GATv2

def build_model(name: str, config: dict, feature_dims: dict) -> nn.Module:
    """
    Build and return a model instance.

    Args:
        name:         model key ("gcn", "schnet")
        config:       full config dict (needs config["model"])
        feature_dims: dict with "node_dim" and "edge_dim" from get_feature_dims()

    Returns:
        Constructed nn.Module ready for .to(device)
    """
    mcfg = config["model"]

    if name == "gcn":
        return GCN(
            node_dim=feature_dims["node_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg.get("dropout", 0.0),
        )

    if name == "gat":
        return GAT(
            node_dim=feature_dims["node_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            heads=mcfg.get("heads", 4),
            dropout=mcfg.get("dropout", 0.0),
        )

    if name == "schnet":
        target_idx = config["dataset"]["target"]
        return QM9SchNet(
            hidden_channels=mcfg.get("hidden_channels", mcfg.get("hidden_dim", 128)),
            num_filters=mcfg.get("num_filters", mcfg.get("hidden_dim", 128)),
            num_interactions=mcfg.get("num_interactions", mcfg.get("num_layers", 6)),
            num_gaussians=mcfg.get("num_gaussians", 50),
            cutoff=mcfg.get("cutoff", 10.0),
            max_num_neighbors=mcfg.get("max_num_neighbors", 32),
            readout=mcfg.get("readout", "add"),
            dipole=mcfg.get("dipole", target_idx == 0),
        )
    
    if name == "gatv2":
        return GATv2(
            node_dim   = feature_dims["node_dim"],
            hidden_dim = config["model"]["hidden_dim"],
            num_layers = config["model"]["num_layers"],
            heads      = config["model"]["heads"],
            edge_dim   = config["model"].get("edge_dim", 4),
            dropout    = config["model"].get("dropout", 0.0),
            share_weights = config["model"].get("share_weights", False),
        )

    raise ValueError(f"Unknown model: '{name}'. Available: gcn, gat, schnet, gatv2")

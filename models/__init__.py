"""
models/__init__.py
------------------
Model factory — instantiate any registered architecture by name string.
"""

import torch.nn as nn
from .gcn import GCN


def build_model(name: str, config: dict, feature_dims: dict) -> nn.Module:
    """
    Build and return a model instance.

    Args:
        name:         model key ("gcn")
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

    raise ValueError(f"Unknown model: '{name}'. Available: gcn")

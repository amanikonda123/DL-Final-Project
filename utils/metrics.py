"""
utils/metrics.py
----------------
Shared metric functions for training and evaluation.
All metrics operate on denormalized values (real physical units).
"""

import torch
import math


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error. Returns a Python float."""
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error. Returns a Python float."""
    return math.sqrt(((pred - target) ** 2).mean().item())

"""
data/loader.py
--------------
Single source of truth for QM9 data loading.
All models use this to ensure identical splits and normalization.
"""

import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
import numpy as np
from pathlib import Path


# QM9 target index → property name (Ramakrishnan et al., 2014, Table 3)
TARGET_NAMES = {
    0:  "mu (Dipole moment, D)",
    1:  "alpha (Polarizability, a0^3)",
    2:  "eps_HOMO (Ha)",
    3:  "eps_LUMO (Ha)",
    4:  "delta_eps (gap, Ha)",
    5:  "R2 (Electronic spatial extent, a0^2)",
    6:  "ZPVE (Ha)",
    7:  "U0 (Internal energy at 0K, Ha)",
    8:  "U (Internal energy at 298K, Ha)",
    9:  "H (Enthalpy at 298K, Ha)",
    10: "G (Free energy at 298K, Ha)",
    11: "Cv (Heat capacity, cal/mol/K)",
}


class Normalizer:
    """Stores mean/std of training targets for denormalization at eval."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / (self.std + 1e-8)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        return y * (self.std + 1e-8) + self.mean


def get_dataloaders(config: dict, root: str = "./data/qm9_raw"):
    """
    Downloads QM9, applies consistent train/val/test splits,
    computes normalization stats from training set only.

    Args:
        config: dict with keys dataset.target, dataset.split, dataset.seed,
                training.batch_size
        root:   directory to store raw/processed QM9 files

    Returns:
        train_loader, val_loader, test_loader, normalizer
    """
    target_idx   = config["dataset"]["target"]
    split_ratios = config["dataset"]["split"]   # [train, val, test]
    seed         = config["dataset"]["seed"]
    batch_size   = config["training"]["batch_size"]

    print(f"[loader] Target: {TARGET_NAMES.get(target_idx, target_idx)}")
    print(f"[loader] Loading QM9 from '{root}'...")

    dataset = QM9(root=root)

    # QM9 in PyG: data.y has shape [1, 19] — we slice to [1] for our target
    # We wrap the dataset to pre-select the target column
    dataset = dataset.copy()  # avoids mutating cached data
    dataset = _SliceTarget(dataset, target_idx)

    # Reproducible split
    N = len(dataset)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_train = int(split_ratios[0] * N)
    n_val   = int(split_ratios[1] * N)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_data = [dataset[int(i)] for i in train_idx]
    val_data   = [dataset[int(i)] for i in val_idx]
    test_data  = [dataset[int(i)] for i in test_idx]

    # Compute normalizer from training set only (no data leakage)
    train_targets = torch.cat([d.y for d in train_data], dim=0)
    normalizer = Normalizer(
        mean=train_targets.mean().item(),
        std=train_targets.std().item()
    )
    print(f"[loader] Train target mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    # Apply normalization in-place to all splits
    for d in train_data: d.y = normalizer.normalize(d.y)
    for d in val_data:   d.y = normalizer.normalize(d.y)
    for d in test_data:  d.y = normalizer.normalize(d.y)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[loader] Split sizes — train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")

    return train_loader, val_loader, test_loader, normalizer


class _SliceTarget:
    """
    Lightweight wrapper that replaces data.y with a single target column.
    QM9 data.y is shape [1, 19] — we extract column `target_idx` as shape [1].
    """
    def __init__(self, dataset, target_idx: int):
        self._dataset    = dataset
        self._target_idx = target_idx

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx].clone()
        data.y = data.y[:, self._target_idx].unsqueeze(-1)  # shape [1, 1] → squeeze → [1]
        data.y = data.y.view(1)
        return data

    def copy(self):
        return self

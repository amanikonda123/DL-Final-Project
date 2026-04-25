"""
training/trainer.py
-------------------
Separate training loop for SchNet on QM9.
"""

import copy
import csv
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.features import select_features
from data.loader import get_dataloaders
from models import build_model
from utils.metrics import mae


def _run_schnet_epoch(model, loader, optimizer, device, normalizer, train=True):
    """Run one SchNet train/validation epoch on geometry-aware QM9 batches."""
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = select_features(batch, mode="geometry")
            batch = batch.to(device)

            pred = model(batch.z, batch.pos, batch.batch)
            target = batch.y.view(-1)
            loss = F.mse_loss(pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            all_preds.append(normalizer.denormalize(pred.detach().cpu()))
            all_targets.append(normalizer.denormalize(target.detach().cpu()))

    n = sum(t.size(0) for t in all_targets)
    avg_loss = total_loss / n
    epoch_mae = mae(torch.cat(all_preds), torch.cat(all_targets))
    return avg_loss, epoch_mae


def train_schnet_model(
    config,
    device="cpu",
    output_dir="outputs",
    data_root="./data/qm9_raw",
):
    """
    Train SchNet on QM9 using 3D coordinates and atomic numbers.

    Args:
        config: dict with dataset/training/model keys
        device: "cpu", "cuda", or "mps"
        output_dir: root for checkpoints and logs
        data_root: where to store/load QM9 raw data

    Returns:
        dict with best_val_mae, history, and checkpoint path
    """
    device = torch.device(device)

    requested_mode = config["dataset"].get("feature_mode", "geometry")
    if requested_mode != "geometry":
        print(f"[schnet] Overriding feature_mode '{requested_mode}' -> 'geometry'")

    train_loader, val_loader, _, normalizer = get_dataloaders(config, root=data_root)
    model = build_model("schnet", config, feature_dims={"node_dim": 0, "edge_dim": 0}).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[schnet] Model: schnet  |  params: {param_count:,}")

    tcfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    scheduler = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"]
        )

    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    ckpt_path = f"{output_dir}/checkpoints/schnet_best.pt"
    log_path = f"{output_dir}/logs/schnet_history.csv"

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    patience = tcfg.get("patience", 20)
    history = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "lr"],
        )
        writer.writeheader()

    pbar = tqdm(range(1, tcfg["epochs"] + 1), desc="Training schnet")
    for epoch in pbar:
        train_loss, _ = _run_schnet_epoch(
            model, train_loader, optimizer, device, normalizer, train=True
        )
        val_loss, val_mae_epoch = _run_schnet_epoch(
            model, val_loader, None, device, normalizer, train=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_mae": f"{val_mae_epoch:.6f}",
            "lr": f"{current_lr:.2e}",
        }
        history.append(row)

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "lr"],
            )
            writer.writerow(row)

        pbar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_mae=f"{val_mae_epoch:.4f}",
            lr=f"{current_lr:.1e}",
        )

        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[schnet] Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[schnet] Best val MAE: {best_val_mae:.6f}  — checkpoint saved to {ckpt_path}")

    return {
        "best_val_mae": best_val_mae,
        "history": history,
        "checkpoint": ckpt_path,
    }

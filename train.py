"""
train.py
--------
Training loop for GNN models on QM9.
Can be run standalone (reads config/default.yaml) or called from a notebook.
"""

import os
import csv
import copy
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.loader import get_dataloaders
from data.features import select_features, get_feature_dims
from models import build_model
from utils.metrics import mae


def _run_epoch(model, loader, optimizer, device, normalizer, feature_mode, train=True):
    """Run one epoch of training or validation."""
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            batch = select_features(batch, feature_mode)
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(pred, batch.y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.num_graphs

            all_preds.append(normalizer.denormalize(pred.detach().cpu()))
            all_targets.append(normalizer.denormalize(batch.y.detach().cpu()))

    n = sum(t.size(0) for t in all_targets)
    avg_loss = total_loss / n
    val_mae = mae(torch.cat(all_preds), torch.cat(all_targets))
    return avg_loss, val_mae


def train_model(config, device="cpu", output_dir="outputs", data_root="./data/qm9_raw",
                model_name="gcn"):
    """
    Full training run.

    Args:
        config:     dict with dataset/training/model keys
        device:     "cpu" or "cuda"
        output_dir: root for checkpoints and logs
        data_root:  where to store/load QM9 raw data
        model_name: architecture key passed to build_model

    Returns:
        dict with best_val_mae, history (list of dicts), and model state_dict path
    """
    device = torch.device(device)

    feature_mode = config["dataset"].get("feature_mode", "topology")
    feature_dims = get_feature_dims(feature_mode)

    train_loader, val_loader, _, normalizer = get_dataloaders(config, root=data_root)

    model = build_model(model_name, config, feature_dims).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model: {model_name}  |  params: {param_count:,}")

    tcfg = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])

    scheduler = None
    if tcfg.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tcfg["epochs"]
        )

    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)

    ckpt_path = f"{output_dir}/checkpoints/{model_name}_best.pt"
    log_path = f"{output_dir}/logs/{model_name}_history.csv"

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0
    patience = tcfg.get("patience", 20)
    history = []

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "lr"])
        writer.writeheader()

    pbar = tqdm(range(1, tcfg["epochs"] + 1), desc=f"Training {model_name}")
    for epoch in pbar:
        train_loss, train_mae = _run_epoch(
            model, train_loader, optimizer, device, normalizer, feature_mode, train=True
        )
        val_loss, val_mae_epoch = _run_epoch(
            model, val_loader, None, device, normalizer, feature_mode, train=False
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
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "lr"])
            writer.writerow(row)

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_mae=f"{val_mae_epoch:.4f}",
                         lr=f"{current_lr:.1e}")

        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"[train] Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"[train] Best val MAE: {best_val_mae:.6f}  — checkpoint saved to {ckpt_path}")

    return {
        "best_val_mae": best_val_mae,
        "history": history,
        "checkpoint": ckpt_path,
    }


if __name__ == "__main__":
    with open("config/default.yaml") as f:
        cfg = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    result = train_model(cfg, device=device)
    print(f"\nDone. Best val MAE = {result['best_val_mae']:.6f}")

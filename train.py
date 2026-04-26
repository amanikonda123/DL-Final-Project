"""
train.py
--------
Unified training entry point for all GNN models on QM9.
Supports GCN, GAT, and SchNet from a single script.
"""

import argparse
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


def _run_epoch(model, loader, optimizer, device, normalizer, feature_mode, model_name, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = select_features(batch, feature_mode)
            batch = batch.to(device)

            if model_name == "schnet":
                pred = model(batch.z, batch.pos, batch.batch)
            else:
                pred = model(batch.x, batch.edge_index, batch.batch)

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


def train_model(config, device="cpu", output_dir="outputs", data_root="./data/qm9_raw",
                model_name="gcn", loaders=None):
    """
    Train any registered GNN model on QM9.

    Args:
        config:     dict with dataset/training/model keys
        device:     "cpu", "cuda", or "mps"
        output_dir: root for checkpoints and logs
        data_root:  where to store/load QM9 raw data
        model_name: one of "gcn", "gat", "schnet"

    Returns:
        dict with best_val_mae, history, and checkpoint path
    """
    device = torch.device(device)

    feature_mode = config["dataset"].get("feature_mode", "topology")
    feature_dims = get_feature_dims(feature_mode)

    if loaders is not None:
        train_loader, val_loader, normalizer = loaders
    else:
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
    log_path  = f"{output_dir}/logs/{model_name}_history.csv"

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
        train_loss, _ = _run_epoch(
            model, train_loader, optimizer, device, normalizer, feature_mode, model_name, train=True
        )
        val_loss, val_mae_epoch = _run_epoch(
            model, val_loader, None, device, normalizer, feature_mode, model_name, train=False
        )

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch":      epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss":   f"{val_loss:.6f}",
            "val_mae":    f"{val_mae_epoch:.6f}",
            "lr":         f"{current_lr:.2e}",
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
        "history":      history,
        "checkpoint":   ckpt_path,
    }


def _get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model on QM9.")
    parser.add_argument("--config", default="config/gcn.yaml", help="Path to YAML config file.")
    parser.add_argument("--model",  default="gcn", choices=["gcn", "gat", "schnet"],
                        help="Model architecture to train.")
    parser.add_argument("--device",     default=None,      help="Override device: cpu, cuda, or mps.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for logs/checkpoints.")
    parser.add_argument("--data-root",  default="./data/qm9_raw", help="Directory for QM9 cache.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or _get_default_device()
    result = train_model(cfg, device=device, output_dir=args.output_dir,
                         data_root=args.data_root, model_name=args.model)
    print(f"\nDone. Best val MAE = {result['best_val_mae']:.6f}")

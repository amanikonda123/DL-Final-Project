"""
evaluate.py
-----------
Load a saved checkpoint, evaluate on the QM9 test set,
and export per-sample predictions.
"""

import os
import csv
import yaml
import torch

from data.loader import get_dataloaders
from data.features import select_features, get_feature_dims
from models import build_model
from utils.metrics import mae, rmse


def evaluate_model(config, checkpoint_path, device="cpu", output_dir="outputs",
                   data_root="./data/qm9_raw", model_name="gcn"):
    """
    Evaluate a trained model on the test set.

    Args:
        config:          dict with dataset/training/model keys
        checkpoint_path: path to the .pt state_dict file
        device:          "cpu" or "cuda"
        output_dir:      root for saving predictions
        data_root:       QM9 data directory
        model_name:      architecture key

    Returns:
        dict with test_mae, test_rmse, and path to predictions CSV
    """
    device = torch.device(device)

    feature_mode = config["dataset"].get("feature_mode", "topology")
    feature_dims = get_feature_dims(feature_mode)

    _, _, test_loader, normalizer = get_dataloaders(config, root=data_root)

    model = build_model(model_name, config, feature_dims).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = select_features(batch, feature_mode)
            batch = batch.to(device)

            if model_name == "schnet":
                pred = model(batch.z, batch.pos, batch.batch)
            elif model_name == "gatv2":
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                pred = model(batch.x, batch.edge_index, batch.batch)

            all_preds.append(normalizer.denormalize(pred.cpu()))
            all_targets.append(normalizer.denormalize(batch.y.cpu()))

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    test_mae = mae(preds, targets)
    test_rmse = rmse(preds, targets)

    print(f"[eval] Test MAE  = {test_mae:.6f}")
    print(f"[eval] Test RMSE = {test_rmse:.6f}")
    print(f"[eval] Samples   = {preds.size(0)}")

    os.makedirs(f"{output_dir}/results", exist_ok=True)
    pred_path = f"{output_dir}/results/{model_name}_predictions.csv"

    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "predicted", "target", "abs_error"])
        for i in range(preds.size(0)):
            p, t = preds[i].item(), targets[i].item()
            writer.writerow([i, f"{p:.6f}", f"{t:.6f}", f"{abs(p - t):.6f}"])

    print(f"[eval] Predictions saved to {pred_path}")

    return {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "predictions_path": pred_path,
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
    ckpt = "outputs/checkpoints/gcn_best.pt"

    result = evaluate_model(cfg, ckpt, device=device)
    print(f"\nTest MAE = {result['test_mae']:.6f}, RMSE = {result['test_rmse']:.6f}")

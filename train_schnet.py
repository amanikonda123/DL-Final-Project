"""
train_schnet.py
---------------
Standalone entry point for training SchNet on QM9.
"""

import argparse
import torch
import yaml

from training.trainer import train_schnet_model


def _get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SchNet on QM9.")
    parser.add_argument("--config", default="config/schnet.yaml", help="Path to YAML config file.")
    parser.add_argument("--device", default=None, help="Override device: cpu, cuda, or mps.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for logs/checkpoints.")
    parser.add_argument("--data-root", default="./data/qm9_raw", help="Directory for QM9 cache.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or _get_default_device()

    result = train_schnet_model(
        cfg,
        device=device,
        output_dir=args.output_dir,
        data_root=args.data_root,
    )
    print(f"\nDone. Best val MAE = {result['best_val_mae']:.6f}")

"""Minimal BC training script on synthetic imitation data.

Usage::

    python demos/mujoco-manip-il/scripts/train_bc_state.py \\
        --seed 42 --obs-dim 8 --action-dim 4 --num-samples 512 \\
        --batch-size 64 --epochs 5 --lr 0.001 --output-dir outputs/day02_bc_smoke

All reusable components live in ``robot_learning_lab.training_skeleton``.
"""

import argparse
import json
import os
import sys

# -- path setup: allow the script to find src/ from the repo root -----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch
from torch.utils.data import DataLoader, random_split

from robot_learning_lab.training_skeleton import (
    MLPPolicy,
    SyntheticImitationDataset,
    evaluate,
    save_checkpoint,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a state-based BC policy on synthetic data."
    )
    parser.add_argument("--obs-dim", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="outputs/day02_bc_smoke")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dataset -----------------------------------------------------------
    dataset = SyntheticImitationDataset(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---- model & optimizer -------------------------------------------------
    model = MLPPolicy(args.obs_dim, args.action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- training loop -----------------------------------------------------
    metrics = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )
        metrics.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        save_checkpoint(model, optimizer, args.output_dir, epoch)

    # ---- save metrics ------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    print("Training complete. Checkpoints saved to", args.output_dir)


if __name__ == "__main__":
    main()

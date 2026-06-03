import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SyntheticImitationDataset(Dataset):
    def __init__(self, obs_dim, action_dim, num_samples, seed=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_samples = num_samples
        rng = np.random.RandomState(seed)
        self.W = rng.randn(obs_dim, action_dim).astype(np.float32)
        self.b = rng.randn(action_dim).astype(np.float32)
        self.observations = rng.randn(num_samples, obs_dim).astype(np.float32)
        noise = 0.01 * rng.randn(num_samples, action_dim).astype(np.float32)
        self.actions = np.tanh(self.observations @ self.W + self.b) + noise

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        obs = torch.from_numpy(self.observations[index])
        act = torch.from_numpy(self.actions[index])
        return obs, act


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.network(obs)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for obs, act in dataloader:
        obs = obs.to(device)
        act = act.to(device)
        pred = model(obs)
        loss = criterion(pred, act)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * obs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for obs, act in dataloader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
            loss = criterion(pred, act)
            total_loss += loss.item() * obs.size(0)
    return total_loss / len(dataloader.dataset)


def save_checkpoint(model, optimizer, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
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
    model = MLPPolicy(args.obs_dim, args.action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        save_checkpoint(model, optimizer, args.output_dir, epoch)
    print("Training complete. Checkpoints saved to", args.output_dir)


if __name__ == "__main__":
    main()

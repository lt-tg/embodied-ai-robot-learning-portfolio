"""Reusable PyTorch training skeleton for imitation learning / behavior cloning.

Provides:
- Reproducible seed control
- Synthetic imitation dataset (no MuJoCo dependency)
- Simple MLP policy
- Train / eval loop helpers
- Checkpoint save/load

All components are designed for PyTorch >= 1.8 (no torch.compile, no new AMP).
"""

import os
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Also forces cuDNN deterministic mode and disables benchmark autotune
    so that every run with the same seed produces identical results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SyntheticImitationDataset(Dataset):
    """Generates synthetic imitation-learning data::

        action = tanh(observation @ W + b) + small_noise

    where *W* and *b* are fixed per dataset instance (controlled by *seed*).

    Parameters
    ----------
    obs_dim : int
        Dimensionality of observation vectors.
    action_dim : int
        Dimensionality of action vectors.
    num_samples : int
        Number of (obs, action) pairs to generate.
    seed : int
        Seed for the internal NumPy generator (default 0).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_samples: int,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_samples = num_samples
        rng = np.random.RandomState(seed)
        self.W = rng.randn(obs_dim, action_dim).astype(np.float32)
        self.b = rng.randn(action_dim).astype(np.float32)
        self.observations = rng.randn(num_samples, obs_dim).astype(np.float32)
        noise = 0.01 * rng.randn(num_samples, action_dim).astype(np.float32)
        self.actions = np.tanh(self.observations @ self.W + self.b) + noise

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        obs = torch.from_numpy(self.observations[index])
        act = torch.from_numpy(self.actions[index])
        return obs, act


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class MLPPolicy(nn.Module):
    """Simple feed-forward policy for state-based behavior cloning.

    Architecture: Linear → ReLU → Linear → Tanh

    Parameters
    ----------
    obs_dim : int
        Input observation dimension.
    action_dim : int
        Output action dimension.
    hidden_dim : int
        Hidden layer size (default 128).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return predicted actions for a batch of observations.

        Parameters
        ----------
        obs : torch.Tensor
            Shape ``[batch_size, obs_dim]``.

        Returns
        -------
        torch.Tensor
            Shape ``[batch_size, action_dim]``.
        """
        return self.network(obs)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch, return average MSE loss."""
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


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model on a dataset, return average MSE loss (no grad)."""
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


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    epoch: int,
    extra: Optional[dict] = None,
) -> str:
    """Save a training checkpoint and return the file path.

    The checkpoint dict contains:
    - ``model_state_dict``
    - ``optimizer_state_dict``
    - ``epoch``
    - ``extra`` (optional user dict, e.g. config / metrics)
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra is not None:
        checkpoint["extra"] = extra
    path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, path)
    return path

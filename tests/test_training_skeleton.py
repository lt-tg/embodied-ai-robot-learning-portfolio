"""Unit tests for the training skeleton module.

Verifies: seed reproducibility, dataset shapes, policy forward pass.
"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.training_skeleton import (
    MLPPolicy,
    SyntheticImitationDataset,
    set_seed,
)


class SetSeedTests(unittest.TestCase):
    """Reproducibility: same seed → same random numbers."""

    def test_torch_randn_reproducible(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        self.assertTrue(torch.equal(a, b),
                        "Same seed must produce identical randn tensors.")

    def test_torch_randn_different_seeds_differ(self):
        set_seed(1)
        a = torch.randn(10)
        set_seed(2)
        b = torch.randn(10)
        self.assertFalse(torch.equal(a, b),
                         "Different seeds should yield different tensors "
                         "(extremely unlikely to fail by chance).")


class SyntheticImitationDatasetTests(unittest.TestCase):
    """Dataset contract: correct shapes and lengths."""

    def setUp(self):
        self.obs_dim = 8
        self.action_dim = 4
        self.num_samples = 100
        self.dataset = SyntheticImitationDataset(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_samples=self.num_samples,
            seed=0,
        )

    def test_len_returns_num_samples(self):
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem_returns_correct_shapes(self):
        obs, act = self.dataset[0]
        self.assertEqual(obs.shape, torch.Size([self.obs_dim]))
        self.assertEqual(act.shape, torch.Size([self.action_dim]))

    def test_getitem_returns_correct_shapes_for_any_index(self):
        obs, act = self.dataset[99]  # last sample
        self.assertEqual(obs.shape, torch.Size([self.obs_dim]))
        self.assertEqual(act.shape, torch.Size([self.action_dim]))

    def test_data_is_reproducible_across_instances(self):
        d1 = SyntheticImitationDataset(8, 4, 50, seed=42)
        d2 = SyntheticImitationDataset(8, 4, 50, seed=42)
        for i in range(50):
            o1, a1 = d1[i]
            o2, a2 = d2[i]
            self.assertTrue(torch.equal(o1, o2),
                            f"Observation {i} differs between same-seed datasets.")
            self.assertTrue(torch.equal(a1, a2),
                            f"Action {i} differs between same-seed datasets.")


class MLPPolicyTests(unittest.TestCase):
    """Policy network: input/output shape contract."""

    def setUp(self):
        self.obs_dim = 8
        self.action_dim = 4
        self.model = MLPPolicy(self.obs_dim, self.action_dim)

    def test_forward_output_shape(self):
        batch_size = 32
        obs = torch.randn(batch_size, self.obs_dim)
        pred = self.model(obs)
        self.assertEqual(pred.shape, torch.Size([batch_size, self.action_dim]))

    def test_forward_with_batch_size_one(self):
        obs = torch.randn(1, self.obs_dim)
        pred = self.model(obs)
        self.assertEqual(pred.shape, torch.Size([1, self.action_dim]))

    def test_output_range_is_tanh_bounded(self):
        """Tanh output should lie in [-1, 1]."""
        batch_size = 128
        obs = torch.randn(batch_size, self.obs_dim)
        with torch.no_grad():
            pred = self.model(obs)
        self.assertTrue((pred >= -1.0).all() and (pred <= 1.0).all(),
                        "MLPPolicy output must lie in [-1, 1] because of Tanh.")


if __name__ == "__main__":
    unittest.main()

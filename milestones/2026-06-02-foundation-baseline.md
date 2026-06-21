# Day 02 - PyTorch Training Skeleton

Date: 2026-06-02

## Goal

Build a reusable behavior-cloning training skeleton with reproducible seed control, synthetic imitation data, train/eval split, MSE metric, checkpoint saving, and unit tests.

## Commands

**Full test suite (20 tests):**
```powershell
python -m unittest discover -s tests
```

**BC smoke training:**
```powershell
python demos\mujoco-manip-il\scripts\train_bc_state.py `
  --seed 42 `
  --obs-dim 8 `
  --action-dim 4 `
  --num-samples 512 `
  --batch-size 64 `
  --epochs 5 `
  --lr 0.001 `
  --output-dir outputs\day02_bc_smoke
```

## Result

- **Tests**: 20 tests passed OK (11 pre-existing + 9 new training skeleton tests).
- **Training output** (seed=42):

| Epoch | train_loss | val_loss |
|-------|-----------|----------|
| 1 | 0.735916 | 0.658141 |
| 2 | 0.581844 | 0.533611 |
| 3 | 0.476437 | 0.439745 |
| 4 | 0.396481 | 0.365145 |
| 5 | 0.330212 | 0.305323 |

- **Checkpoints**: `outputs/day02_bc_smoke/checkpoint_epoch_1.pt` through `checkpoint_epoch_5.pt`
- **Metrics**: `outputs/day02_bc_smoke/metrics.json`
- **Environment**: Python 3.8.8, PyTorch 1.8.0+cu111, CUDA True, NVIDIA GeForce GTX 1660 Ti

## Artifact

### Files created / modified

| File | Status | Purpose |
|------|--------|---------|
| `src/robot_learning_lab/training_skeleton.py` | Created | Reusable training library: `set_seed`, `SyntheticImitationDataset`, `MLPPolicy`, `train_one_epoch`, `evaluate`, `save_checkpoint` |
| `src/robot_learning_lab/__init__.py` | Modified | Added `training_skeleton` to `__all__` |
| `demos/mujoco-manip-il/scripts/train_bc_state.py` | Refactored | CLI-only script, imports from library |
| `tests/test_training_skeleton.py` | Created | 9 unit tests: seed reproducibility (2), dataset shape/length/reproducibility (4), policy forward shape/range (3) |
| `notes/training_discipline.md` | Created | Training discipline reference: reproducibility, Dataset/DataLoader boundary, training loop, checkpoint, BC framing, shape discipline, Python 3.8 notes |

### Architecture

```
                     src/robot_learning_lab/training_skeleton.py  (library)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
           demos/.../train_bc_state.py    tests/test_training_skeleton.py
           (CLI entry point)             (verification)
```

## Lesson

- Python 3.8 does not support `dict | None` type union syntax — use `Optional[dict]` from `typing`.
- `np.random.RandomState(seed)` creates an isolated generator; two Dataset instances with the same seed produce identical data regardless of global `np.random` state.
- `torch.utils.data.random_split` consumes `torch.random` — `set_seed()` before the split guarantees deterministic train/val partition.
- `optimizer.state_dict()` stores Adam momentum buffers (`exp_avg`, `exp_avg_sq`, step); saving only `model.state_dict()` breaks seamless resume.
- `loss.item()` detaches the computation graph; `* obs.size(0)` ensures sample-weighted averaging across uneven batches.
- Dataset returns 1D single samples; DataLoader's `collate_fn` stacks them into 2D batches. Training loop never touches Dataset directly.
- `model.train()` / `model.eval()` / `with torch.no_grad()` are habits even when the current MLP has no dropout/batchnorm.

## Next Step

Apply the training skeleton to state-based behavior cloning with real robot or simulated data (Day 10-11). Next immediate step: robot coordinate frames and SE(3) math (Day 03).

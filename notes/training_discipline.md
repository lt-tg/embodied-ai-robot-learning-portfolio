# Training Discipline Notes

> Day 02 core takeaway: every robot learning experiment must answer six questions — where does the data come from, what is the seed, how is train/val split, how is loss computed, what does the checkpoint contain, and is the command reproducible.

## Reproducibility

### Seed control

`set_seed(seed)` must touch five sources:

| Source | Call | Why |
|--------|------|-----|
| Python stdlib | `random.seed(seed)` | `random_split` occasionally uses Python random |
| NumPy | `np.random.seed(seed)` | Data generation, data augmentation |
| PyTorch CPU | `torch.manual_seed(seed)` | Weight initialization, dropout |
| PyTorch CUDA | `torch.cuda.manual_seed_all(seed)` | GPU-side ops |
| cuDNN | `deterministic=True, benchmark=False` | Forces deterministic conv algorithm selection |

**cuDNN note**: `benchmark=False` disables auto-tuning (costs 10-20% speed on conv-heavy models). `deterministic=True` excludes algorithms that use atomic-add or non-deterministic parallel reductions. Both are irrelevant for MLP-only networks but become critical with ConvNet / Transformer policies.

### RandomState isolation

`Dataset.__init__` should use `np.random.RandomState(seed)` — a **local** generator — rather than touching the global `np.random`. This guarantees two Dataset instances created with the same seed produce byte-identical data regardless of global state.

### Deterministic split

`torch.utils.data.random_split` consumes `torch.random`. Calling `set_seed()` immediately before `random_split` ensures the same train/val partition every run.

## Dataset / DataLoader Boundary

### The contract

```
Dataset:     "what is one sample?"   → __len__, __getitem__
DataLoader:  "how to feed samples?"  → batch_size, shuffle, num_workers
```

`__getitem__` returns a single (obs, act) pair — 1D tensors, not batched. `DataLoader` internally:
1. Collects `batch_size` indices (shuffled if `shuffle=True`)
2. Calls `dataset[i]` for each index
3. `collate_fn` (default: `torch.stack`) assembles them into `[batch_size, dim]`

Training loop only interacts with `DataLoader`; it should never know whether data is synthetic, from disk, or streamed.

### Shuffle rationale

- **Train**: `shuffle=True` — prevents ordering bias; each batch approximates an unbiased gradient estimate.
- **Val**: `shuffle=False` — makes per-batch loss comparable across epochs; the same data subset always maps to the same batch.

## Training Loop Discipline

### `train_one_epoch` checklist

```python
model.train()          # enable dropout, batchnorm (no-op for vanilla MLP, but a habit)
for obs, act in loader:
    obs = obs.to(device)    # CPU → GPU copy (required; model is already on GPU)
    pred = model(obs)       # forward
    loss = MSE(pred, act)   # scalar with computation graph
    optimizer.zero_grad()   # CRITICAL: PyTorch accumulates grads by default
    loss.backward()         # compute gradients via autograd
    optimizer.step()        # apply gradients
    total_loss += loss.item() * batch_size   # .item() detaches from graph; weight by batch
return total_loss / len(dataset)
```

### `evaluate` checklist (differences from training)

```python
model.eval()                    # disable dropout, freeze batchnorm stats
with torch.no_grad():           # skip computation graph → saves VRAM, faster
    ...
    # NO zero_grad / backward / step — pure observer
```

### Why `.item()` matters

`loss` is a 0-d tensor with a computation graph attached. Keeping it alive prevents the graph from being freed. `loss.item()` returns a Python float and releases the graph.

### Why weight by `obs.size(0)`

The last batch is often smaller than `batch_size` (e.g., 13 vs 64). Summing `loss * obs.size(0)` then dividing by total samples gives a **sample-weighted** mean — every sample contributes equally regardless of batch size.

### Why `zero_grad()` is mandatory

PyTorch gradients **accumulate** in `.grad` buffers. Skipping `zero_grad()` means gradient from batch N is added to batch N-1's gradient, which is almost never intended.

### Why mini-batch (not full-batch, not single-sample)

- vs. SGD (batch=1): GPU parallelism + gradient stability from averaging
- vs. full-batch (batch=461): multiple updates per epoch → faster convergence; gradient noise biases toward flatter minima → better generalization
- 64 is a practical default; the optimal value depends on model size and GPU memory

## Checkpoint Discipline

### What to save

```python
checkpoint = {
    "model_state_dict": model.state_dict(),        # layer weights
    "optimizer_state_dict": optimizer.state_dict(),# Adam m_t, v_t, step counter
    "epoch": epoch,                                # resume point
    # extra: config dict (seed, lr, obs_dim, ...)
}
```

### Why optimizer state matters

Adam maintains per-parameter momentum buffers:

```
m_t = β₁·m_{t-1} + (1-β₁)·grad        (exp_avg)
v_t = β₂·v_{t-1} + (1-β₂)·grad²       (exp_avg_sq)
```

These are **cumulative statistics across all training steps**. If you only save model weights and resume training, `m_t` and `v_t` reset to zero — the first ~50 steps after resume will have incorrect update directions and magnitudes.

### `model.state_dict()` vs `model.parameters()`

`state_dict()` returns an `OrderedDict` mapping layer names to tensors. `parameters()` returns an iterator over tensor objects. Always use `state_dict()` for serialization — it preserves the name mapping needed for `load_state_dict()`.

### Save strategy (Day 02 minimal)

Save every epoch (`checkpoint_epoch_N.pt`). For production: track best `val_loss` and only overwrite `best.pt`.

## Behavior Cloning Framing

Behavior cloning (BC) is **supervised learning with a specific label**: expert actions.

```
Input:  observation (state)       → what the robot sees
Target: expert action             → what the expert did in that state
Model:  MLPPolicy(obs → action)   → learn the mapping
Loss:   MSE(predicted, target)    → minimize action prediction error
```

BC treats the expert trajectory as a labeled dataset and ignores environment dynamics. It does **not** interact with the environment during training. This makes it simple but vulnerable to compounding errors at deployment (the policy sees states it never saw during training → errors accumulate).

## Shape Discipline

Always know the shape at every layer boundary:

```
Dataset[i]       → obs: [obs_dim],         act: [action_dim]
DataLoader batch → obs: [batch, obs_dim],  act: [batch, action_dim]
Linear(8→128)    → [batch, 8]   → [batch, 128]
ReLU             → [batch, 128] → [batch, 128]    (shape unchanged)
Linear(128→4)    → [batch, 128] → [batch, 4]
Tanh             → [batch, 4]   → [batch, 4]      (shape unchanged, range [-1,1])
MSE(pred, act)   → scalar
```

## Python Version Notes

Project targets `>=3.8`. Avoid Python 3.10+ syntax:

| Avoid (3.10+) | Use instead (3.8+) |
|---------------|-------------------|
| `dict \| None` | `Optional[dict]` |
| `list[int]` | `List[int]` |
| `X \| Y` | `Union[X, Y]` |

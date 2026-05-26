# Demo 1: MuJoCo Manipulation IL Benchmark

Goal: build a manipulation benchmark that proves you can collect demonstrations, train policies, evaluate success rates, and explain failures.

## MVP

- One simulated task: start with robosuite `Lift`, then move to `PickPlace`.
- 100-300 scripted demonstrations.
- State-based Behavior Cloning baseline.
- Evaluation over 50-100 episodes.
- README, result table, training curve, and demo GIF/video.

## Recommended Stack

- Python, PyTorch
- MuJoCo + robosuite
- robomimic or LeRobot dataset conventions
- TensorBoard or W&B
- Optional ROS2 wrapper after the policy works

## Directory Contract

```text
demos/mujoco-manip-il/
  configs/
  datasets/
  envs/
  policies/
  scripts/
  reports/
  videos/
```

## Milestones

1. Run a robosuite manipulation environment.
2. Save observations/actions from a scripted controller.
3. Train BC on state observations.
4. Add image observations.
5. Add ACT and Diffusion Policy baselines.
6. Evaluate with domain randomization.
7. Export a ROS2-style policy interface.

## Metrics

| Policy | Dataset Size | Success Rate | Action MSE | Latency | Notes |
|---|---:|---:|---:|---:|---|
| BC-state | TBD | TBD | TBD | TBD | first baseline |
| BC-vision | TBD | TBD | TBD | TBD | RGB + proprioception |
| ACT | TBD | TBD | TBD | TBD | action chunks |
| Diffusion Policy | TBD | TBD | TBD | TBD | receding horizon |

## Resume Bullet

Built a MuJoCo-based robot manipulation benchmark for pick-and-place, collected scripted demonstrations, and compared BC/ACT/Diffusion Policy with reproducible training/evaluation pipelines and ROS2 policy wrapper.

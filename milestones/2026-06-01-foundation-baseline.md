# Foundation Baseline Verification

Date: 2026-06-01

## Purpose

Establish a reproducible baseline for the public portfolio: unit tests, robotics math examples, command parsing, and CUDA visibility.

## Commands

```powershell
python -m unittest discover -s tests
python examples/day03_robot_math.py
python examples/day04_planar_arm.py
python examples/day25_parse_command.py
python -c "import torch; print(torch.__version__); print('cuda_available:', torch.cuda.is_available()); print('cuda_device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Result

- Unit tests: 11 tests ran and passed.
- Robot math example:
  - `world_point: [0.4707, 0.2707, 0.1]`
  - `recovered_object_point: [0.1, 0.0, 0.0]`
  - `z_axis_quaternion_xyzw: [0.0, 0.0, 0.3827, 0.9239]`
- Planar arm example:
  - `target: [1.0, 1.0]`
  - `joint_angles_rad: [1.5708, -1.5708]`
  - `joint_angles_deg: [90.0, -90.0]`
  - `reached: [1.0, 1.0]`
  - `jacobian: [[-1.0, -0.0], [1.0, 1.0]]`
- Command parser example:
  - `pick the red cube and place it on the blue plate -> action=pick_place, object=red cube, target=blue plate`
  - `pick green block -> action=pick, object=green block, target=None`
- PyTorch smoke test:
  - `torch: 1.8.0+cu111`
  - `cuda_available: True`
  - `cuda_device: NVIDIA GeForce GTX 1660 Ti`

## Evidence

- Tested utility package under `src/`.
- Reproducible examples under `examples/`.
- Demo scaffolds under `demos/`.
- Environment baseline with local CUDA visibility.

## Follow-Up

Build the first behavior-cloning training skeleton for the manipulation demo.

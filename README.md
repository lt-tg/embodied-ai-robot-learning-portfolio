# Embodied AI Robot Learning Portfolio

This is the public portfolio repository and the single runnable source of truth for embodied-AI robot-learning code, tests, examples, and demo projects.

It records public evidence of implementation work in embodied AI, robot learning, manipulation, simulation, ROS2-style interfaces, and VLA evaluation: runnable examples, tested utilities, demo scaffolds, technical notes, result logs, and weekly reports.

## Focus Areas

- Robot math: SE(3), quaternions, forward/inverse kinematics, Jacobians.
- Simulation: MuJoCo/robosuite manipulation tasks.
- Robot learning: behavior cloning, ACT, Diffusion Policy, evaluation metrics.
- VLA evaluation: image-language-action model inputs, action decoding, prompt robustness.
- System integration: ROS2-style command parsing and policy bridge design.

## Repository Structure

```text
daily/      Daily public progress notes.
demos/      Portfolio demo projects.
examples/   Runnable learning exercises.
notes/      Technical notes and paper reading notes.
reports/    Weekly summaries and result reports.
src/        Tested Python utilities.
tests/      Unit tests.
```

Non-public planning and career-preparation material lives outside this repository.

## Quick Start

```powershell
python -m unittest discover -s tests
python examples/day03_robot_math.py
python examples/day04_planar_arm.py
python examples/day25_parse_command.py
```

## Current Public Demos

| Demo | Status | Goal | Public Evidence |
|---|---|---|---|
| `demos/mujoco-manip-il` | scaffold | MuJoCo manipulation + imitation learning benchmark | configs, result template, planned metrics |
| `demos/vla-eval-pipeline` | scaffold | VLA-style evaluation pipeline | prompt set, evaluation report template |
| `demos/ros2-language-robot` | scaffold | Language command to robot task interface | parser examples and ROS2 interface sketch |

## Progress Log

Daily public notes live in `daily/`. Each entry records:

- goal
- commands
- result
- artifact
- blocker or lesson
- next step

## Verification

Current lightweight verification:

```powershell
python -m unittest discover -s tests
```

The heavy robotics dependencies are intentionally added only when the corresponding demo needs them.

# Embodied AI Robot Learning Portfolio

Public demo portfolio for embodied-AI and robot-learning systems: manipulation benchmarks, VLA evaluation utilities, ROS2-style language interfaces, and tested robotics foundations.

This repository is the single runnable source of truth for public code, examples, demos, result reports, and technical writeups.

## Demo Showcase

| Demo | Status | What It Shows | Evidence |
|---|---|---|---|
| `demos/mujoco-manip-il` | scaffold | Manipulation benchmark with imitation-learning configs for BC, ACT, and Diffusion Policy | Configs, result template, planned metrics |
| `demos/vla-eval-pipeline` | scaffold | VLA-style prompt and action-output evaluation pipeline | Prompt set, evaluation report template |
| `demos/ros2-language-robot` | scaffold | Language command parsing and ROS2-style task interface | Parser examples, launch sketch, policy bridge design |

## Quick Verification

```powershell
python -m unittest discover -s tests
python examples/day03_robot_math.py
python examples/day04_planar_arm.py
python examples/day25_parse_command.py
```

## Technical Components

- Robot math utilities: transforms, rotations, quaternions, and coordinate-frame checks.
- Kinematics utilities: 2D planar-arm forward kinematics, inverse kinematics, and Jacobian calculation.
- Evaluation utilities: success rate, action MSE, and simple parser metrics.
- Language interface utilities: natural-language command parsing for robot task sketches.

## Results And Reports

- `reports/`: result-oriented technical reports and demo summaries.
- `milestones/`: public milestone records for completed portfolio evidence.
- `notes/`: technical design notes that support demos and reproducible explanations.

## Repository Structure

```text
demos/       Public demo projects.
src/         Reusable tested robotics utilities.
tests/       Unit tests for reusable components.
examples/    Minimal reproducible examples.
reports/     Result reports and technical writeups.
notes/       Technical design notes.
milestones/  Portfolio evidence checkpoints.
```

Non-public planning and career-preparation material lives outside this repository.

## Dependency Policy

The root package stays lightweight. Heavy robotics dependencies such as MuJoCo, robosuite, LeRobot, ROS2, or Isaac Lab should be added only inside the corresponding demo when they are needed for a runnable result.

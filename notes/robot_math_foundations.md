# Robot Math Foundations

This note tracks the robotics math needed for manipulation and VLA policy evaluation.

## Topics

- Coordinate frames.
- Homogeneous transforms.
- Rotation matrices.
- Quaternions.
- Forward kinematics.
- Inverse kinematics.
- Jacobians.

## Runnable Examples

```powershell
python examples/day03_robot_math.py
python examples/day04_planar_arm.py
```

## Current Implementation

- `src/robot_learning_lab/robot_math.py`
- `src/robot_learning_lab/planar_arm.py`
- `tests/test_robot_math.py`
- `tests/test_planar_arm.py`

## Interview-Level Explanation Checklist

- Explain what a transform maps from and to.
- Explain why quaternion convention matters.
- Derive 2-link planar arm forward kinematics.
- Explain why the Jacobian maps joint velocity to end-effector velocity.

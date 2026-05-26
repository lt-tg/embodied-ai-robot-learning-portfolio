import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.planar_arm import PlanarArm2D


if __name__ == "__main__":
    arm = PlanarArm2D([1.0, 1.0])
    target = [1.0, 1.0]
    solution = arm.inverse_kinematics(target, elbow_up=False)
    reached = arm.forward_kinematics(solution)
    jacobian = arm.jacobian(solution)

    print("target:", target)
    print("joint_angles_rad:", [round(value, 4) for value in solution])
    print("joint_angles_deg:", [round(math.degrees(value), 2) for value in solution])
    print("reached:", [round(value, 4) for value in reached])
    print("jacobian:", [[round(value, 4) for value in row] for row in jacobian])

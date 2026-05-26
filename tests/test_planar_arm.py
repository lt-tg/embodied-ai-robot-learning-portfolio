import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.planar_arm import PlanarArm2D


class PlanarArmTests(unittest.TestCase):
    def test_forward_kinematics_for_straight_arm(self):
        arm = PlanarArm2D([1.0, 0.5])

        x, y = arm.forward_kinematics([0.0, 0.0])

        self.assertAlmostEqual(x, 1.5, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)

    def test_inverse_kinematics_reaches_known_target(self):
        arm = PlanarArm2D([1.0, 1.0])
        target = [1.0, 1.0]

        solution = arm.inverse_kinematics(target, elbow_up=False)
        x, y = arm.forward_kinematics(solution)

        self.assertAlmostEqual(x, target[0], places=5)
        self.assertAlmostEqual(y, target[1], places=5)

    def test_inverse_kinematics_rejects_unreachable_target(self):
        arm = PlanarArm2D([1.0, 1.0])

        with self.assertRaises(ValueError):
            arm.inverse_kinematics([3.0, 0.0])

    def test_jacobian_shape_and_straight_arm_values(self):
        arm = PlanarArm2D([1.0, 1.0])

        jacobian = arm.jacobian([0.0, 0.0])

        self.assertEqual(len(jacobian), 2)
        self.assertEqual(len(jacobian[0]), 2)
        self.assertAlmostEqual(jacobian[0][0], 0.0, places=6)
        self.assertAlmostEqual(jacobian[1][0], 2.0, places=6)
        self.assertAlmostEqual(jacobian[1][1], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

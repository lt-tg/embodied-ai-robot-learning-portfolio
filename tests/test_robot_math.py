import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.robot_math import (
    Transform,
    quaternion_from_axis_angle,
    rotation_matrix_z,
)


class RobotMathTests(unittest.TestCase):
    def test_transform_inverse_round_trips_point(self):
        transform = Transform.from_translation_rotation(
            [1.0, -2.0, 0.5],
            rotation_matrix_z(math.pi / 2),
        )

        point = [0.25, 0.5, -1.0]
        world = transform.apply(point)
        recovered = transform.inverse().apply(world)

        for actual, expected in zip(recovered, point):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_transform_composition_matches_sequential_application(self):
        first = Transform.from_translation_rotation([1.0, 0.0, 0.0], rotation_matrix_z(math.pi / 2))
        second = Transform.from_translation_rotation([0.0, 2.0, 0.0], rotation_matrix_z(-math.pi / 2))
        point = [0.5, 0.25, 0.0]

        composed = first @ second
        sequential = first.apply(second.apply(point))

        for actual, expected in zip(composed.apply(point), sequential):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_axis_angle_quaternion_is_unit_length(self):
        quat = quaternion_from_axis_angle([0.0, 0.0, 2.0], math.pi / 3)
        norm = math.sqrt(sum(value * value for value in quat))

        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertGreater(quat[3], 0.0)


if __name__ == "__main__":
    unittest.main()

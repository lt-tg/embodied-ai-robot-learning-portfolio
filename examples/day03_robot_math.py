import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robot_learning_lab.robot_math import Transform, quaternion_from_axis_angle, rotation_matrix_z


if __name__ == "__main__":
    base_to_object = Transform.from_translation_rotation(
        [0.4, 0.2, 0.1],
        rotation_matrix_z(math.pi / 4),
    )
    object_point = [0.1, 0.0, 0.0]
    world_point = base_to_object.apply(object_point)
    recovered = base_to_object.inverse().apply(world_point)
    quat = quaternion_from_axis_angle([0.0, 0.0, 1.0], math.pi / 4)

    print("world_point:", [round(value, 4) for value in world_point])
    print("recovered_object_point:", [round(value, 4) for value in recovered])
    print("z_axis_quaternion_xyzw:", [round(value, 4) for value in quat])

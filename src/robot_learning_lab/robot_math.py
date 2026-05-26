import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


Vector3 = Sequence[float]
Matrix3 = Sequence[Sequence[float]]


def _as_vector3(values: Iterable[float]) -> List[float]:
    vector = [float(value) for value in values]
    if len(vector) != 3:
        raise ValueError("Expected a 3D vector.")
    return vector


def _as_matrix3(values: Matrix3) -> List[List[float]]:
    matrix = [[float(value) for value in row] for row in values]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Expected a 3x3 matrix.")
    return matrix


def _matmul(left: Matrix3, right: Matrix3) -> List[List[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _matvec(matrix: Matrix3, vector: Vector3) -> List[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _transpose(matrix: Matrix3) -> List[List[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


@dataclass(frozen=True)
class Transform:
    """Rigid transform with rotation matrix R and translation t."""

    translation: List[float]
    rotation: List[List[float]]

    @classmethod
    def identity(cls) -> "Transform":
        return cls(
            translation=[0.0, 0.0, 0.0],
            rotation=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

    @classmethod
    def from_translation_rotation(cls, translation: Iterable[float], rotation: Matrix3) -> "Transform":
        return cls(translation=_as_vector3(translation), rotation=_as_matrix3(rotation))

    def apply(self, point: Iterable[float]) -> List[float]:
        vector = _as_vector3(point)
        rotated = _matvec(self.rotation, vector)
        return [rotated[i] + self.translation[i] for i in range(3)]

    def inverse(self) -> "Transform":
        inverse_rotation = _transpose(self.rotation)
        inverse_translation = _matvec(inverse_rotation, [-value for value in self.translation])
        return Transform(translation=inverse_translation, rotation=inverse_rotation)

    def __matmul__(self, other: "Transform") -> "Transform":
        rotation = _matmul(self.rotation, other.rotation)
        translated = self.apply(other.translation)
        return Transform(translation=translated, rotation=rotation)


def rotation_matrix_z(angle_radians: float) -> List[List[float]]:
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return [
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ]


def quaternion_from_axis_angle(axis: Iterable[float], angle_radians: float) -> List[float]:
    x, y, z = _as_vector3(axis)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Axis must have non-zero length.")

    half_angle = angle_radians / 2.0
    scale = math.sin(half_angle) / norm
    return [x * scale, y * scale, z * scale, math.cos(half_angle)]

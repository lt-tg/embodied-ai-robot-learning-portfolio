import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence


def _as_pair(values: Iterable[float], name: str) -> List[float]:
    pair = [float(value) for value in values]
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    return pair


@dataclass(frozen=True)
class PlanarArm2D:
    """Two-link planar arm for quick FK/IK/Jacobian practice."""

    link_lengths: Sequence[float]

    def __post_init__(self) -> None:
        lengths = _as_pair(self.link_lengths, "link_lengths")
        if any(length <= 0.0 for length in lengths):
            raise ValueError("Link lengths must be positive.")
        object.__setattr__(self, "link_lengths", lengths)

    def forward_kinematics(self, joint_angles: Iterable[float]) -> List[float]:
        theta1, theta2 = _as_pair(joint_angles, "joint_angles")
        link1, link2 = self.link_lengths
        theta12 = theta1 + theta2
        return [
            link1 * math.cos(theta1) + link2 * math.cos(theta12),
            link1 * math.sin(theta1) + link2 * math.sin(theta12),
        ]

    def inverse_kinematics(self, target: Iterable[float], elbow_up: bool = True) -> List[float]:
        x, y = _as_pair(target, "target")
        link1, link2 = self.link_lengths
        radius_squared = x * x + y * y
        cos_theta2 = (radius_squared - link1 * link1 - link2 * link2) / (2.0 * link1 * link2)

        tolerance = 1e-9
        if cos_theta2 < -1.0 - tolerance or cos_theta2 > 1.0 + tolerance:
            raise ValueError("Target is outside the reachable workspace.")

        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        sin_theta2 = math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
        if not elbow_up:
            sin_theta2 = -sin_theta2

        theta2 = math.atan2(sin_theta2, cos_theta2)
        theta1 = math.atan2(y, x) - math.atan2(link2 * sin_theta2, link1 + link2 * cos_theta2)
        return [theta1, theta2]

    def jacobian(self, joint_angles: Iterable[float]) -> List[List[float]]:
        theta1, theta2 = _as_pair(joint_angles, "joint_angles")
        link1, link2 = self.link_lengths
        theta12 = theta1 + theta2
        return [
            [
                -link1 * math.sin(theta1) - link2 * math.sin(theta12),
                -link2 * math.sin(theta12),
            ],
            [
                link1 * math.cos(theta1) + link2 * math.cos(theta12),
                link2 * math.cos(theta12),
            ],
        ]

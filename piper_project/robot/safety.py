"""Basic command validation for Piper arm motion."""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.robot_config import RobotConfig


class SafetyError(ValueError):
    """Raised when a command exceeds configured robot limits."""


@dataclass(frozen=True)
class SafetyChecker:
    config: RobotConfig

    def validate_speed(self, speed_percent: int) -> None:
        if not 0 <= speed_percent <= 100:
            raise SafetyError(f"speed_percent must be in [0, 100], got {speed_percent}")

    def validate_pose_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
    ) -> None:
        checks = (
            ("x_mm", x_mm, self.config.x_limit_mm),
            ("y_mm", y_mm, self.config.y_limit_mm),
            ("z_mm", z_mm, self.config.z_limit_mm),
            ("rx_deg", rx_deg, self.config.rx_limit_deg),
            ("ry_deg", ry_deg, self.config.ry_limit_deg),
            ("rz_deg", rz_deg, self.config.rz_limit_deg),
        )
        for name, value, limit in checks:
            if not limit.contains(value):
                raise SafetyError(
                    f"{name}={value} outside [{limit.min_value}, {limit.max_value}]"
                )
        horizontal_radius = math.hypot(x_mm, y_mm)
        if horizontal_radius > self.config.work_radius_mm:
            raise SafetyError(
                f"horizontal radius={horizontal_radius} mm outside "
                f"[0, {self.config.work_radius_mm}]"
            )

    def validate_joints_rad(self, joints_rad: tuple[float, ...] | list[float]) -> None:
        if len(joints_rad) != 6:
            raise SafetyError(f"expected 6 joint angles, got {len(joints_rad)}")
        for index, (value, limit) in enumerate(
            zip(joints_rad, self.config.joint_limits_rad), start=1
        ):
            if not limit.contains(value):
                raise SafetyError(
                    f"joint_{index}={value} rad outside "
                    f"[{limit.min_value}, {limit.max_value}]"
                )

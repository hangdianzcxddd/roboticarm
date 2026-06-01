"""High-level motion helpers for robot-frame commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from robot_control.shared.geometry import EndPose
from robot_control.linux.arm.piper_arm import PiperController


@dataclass
class MotionPlanner:
    controller: PiperController

    def go_home(
        self,
        joints_rad: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        speed_percent: int | None = None,
    ) -> None:
        self.controller.move_joints(list(joints_rad), speed_percent=speed_percent)

    def move_above(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        approach_height_mm: float = 80.0,
        speed_percent: int | None = None,
    ) -> None:
        pose = EndPose(
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm + approach_height_mm,
            rx_deg=rx_deg,
            ry_deg=ry_deg,
            rz_deg=rz_deg,
        )
        self.controller.move_to_pose(pose, speed_percent=speed_percent, move_mode="P")

    def linear_to(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        rx_deg: float,
        ry_deg: float,
        rz_deg: float,
        speed_percent: int | None = None,
    ) -> None:
        pose = EndPose(x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg)
        self.controller.move_to_pose(pose, speed_percent=speed_percent, move_mode="L")

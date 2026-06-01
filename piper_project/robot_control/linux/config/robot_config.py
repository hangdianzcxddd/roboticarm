"""Piper robot configuration used by the control layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


MoveMode = Literal["P", "J", "L"]


@dataclass(frozen=True)
class PoseLimit:
    min_value: float
    max_value: float

    def contains(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value


@dataclass(frozen=True)
class ModifiedDhParam:
    d_mm: float
    a_mm: float
    alpha_rad: float
    theta_offset_rad: float


@dataclass(frozen=True)
class RobotConfig:
    degrees_of_freedom: int = 6
    payload_kg: float = 1.5
    body_weight_kg: float = 4.2
    work_radius_mm: float = 626.75
    repeatability_mm: float = 0.1
    can_bitrate: int = 1_000_000

    can_name: str = "can0"
    can_auto_init: bool = True
    judge_flag: bool = True
    connect_timeout_s: float = 3.0
    command_interval_s: float = 0.01
    default_speed_percent: int = 20
    default_move_mode: MoveMode = "P"
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True
    auto_enable: bool = True
    park_before_disable: bool = True
    disable_park_speed_percent: int = 10
    disable_park_timeout_s: float = 10.0
    disable_park_tolerance_rad: float = math.radians(2.0)
    disable_park_settle_s: float = 0.5
    disable_park_joints_rad: tuple[float, ...] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    x_limit_mm: PoseLimit = field(default_factory=lambda: PoseLimit(-626.75, 626.75))
    y_limit_mm: PoseLimit = field(default_factory=lambda: PoseLimit(-626.75, 626.75))
    z_limit_mm: PoseLimit = field(default_factory=lambda: PoseLimit(0.0, 800.0))
    rx_limit_deg: PoseLimit = field(default_factory=lambda: PoseLimit(-180.0, 180.0))
    ry_limit_deg: PoseLimit = field(default_factory=lambda: PoseLimit(-180.0, 180.0))
    rz_limit_deg: PoseLimit = field(default_factory=lambda: PoseLimit(-180.0, 180.0))

    joint_limits_rad: tuple[PoseLimit, ...] = field(
        default_factory=lambda: (
            PoseLimit(math.radians(-150.0), math.radians(150.0)),
            PoseLimit(0.0, math.radians(180.0)),
            PoseLimit(math.radians(-170.0), 0.0),
            PoseLimit(math.radians(-100.0), math.radians(100.0)),
            PoseLimit(math.radians(-70.0), math.radians(70.0)),
            PoseLimit(math.radians(-120.0), math.radians(120.0)),
        )
    )
    joint_max_speeds_rad_s: tuple[float, ...] = field(
        default_factory=lambda: (
            math.radians(180.0),
            math.radians(195.0),
            math.radians(180.0),
            math.radians(225.0),
            math.radians(225.0),
            math.radians(225.0),
        )
    )
    modified_dh_params: tuple[ModifiedDhParam, ...] = field(
        default_factory=lambda: (
            ModifiedDhParam(123.0, 0.0, 0.0, 0.0),
            ModifiedDhParam(0.0, 0.0, -math.pi / 2, math.radians(-172.22)),
            ModifiedDhParam(0.0, 285.03, 0.0, math.radians(-102.78)),
            ModifiedDhParam(250.75, -21.98, math.pi / 2, 0.0),
            ModifiedDhParam(0.0, 0.0, -math.pi / 2, 0.0),
            ModifiedDhParam(9.0, 0.0, math.pi / 2, 0.0),
        )
    )


ROBOT_CONFIG = RobotConfig()

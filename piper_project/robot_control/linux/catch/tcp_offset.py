"""TCP offset helpers for the official Piper gripper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from robot_control.shared.geometry import EndPose


OFFICIAL_GRIPPER_TCP_OFFSET_MM = (0.0, 0.0, 145.0)


@dataclass(frozen=True)
class CatchPlan:
    target_tcp_pose: EndPose
    approach_tcp_pose: EndPose
    grasp_tcp_pose: EndPose
    lift_tcp_pose: EndPose
    approach_j6_pose: EndPose
    grasp_j6_pose: EndPose
    lift_j6_pose: EndPose


def rotation_matrix_xyz_deg(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """Return Rz * Ry * Rx for Piper SDK XYZ Euler angles."""

    rx, ry, rz = (math.radians(value) for value in (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ],
        dtype=float,
    )


def j6_pose_for_tcp_pose(
    tcp_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float] = OFFICIAL_GRIPPER_TCP_OFFSET_MM,
) -> EndPose:
    """Convert a desired TCP pose into the J6 pose sent to Piper."""

    offset_base = rotation_matrix_xyz_deg(
        tcp_pose.rx_deg,
        tcp_pose.ry_deg,
        tcp_pose.rz_deg,
    ) @ _vector3(offset_j6_tcp_mm, "offset_j6_tcp_mm")
    return EndPose(
        x_mm=float(tcp_pose.x_mm - offset_base[0]),
        y_mm=float(tcp_pose.y_mm - offset_base[1]),
        z_mm=float(tcp_pose.z_mm - offset_base[2]),
        rx_deg=tcp_pose.rx_deg,
        ry_deg=tcp_pose.ry_deg,
        rz_deg=tcp_pose.rz_deg,
    )


def tcp_pose_for_j6_pose(
    j6_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float] = OFFICIAL_GRIPPER_TCP_OFFSET_MM,
) -> EndPose:
    """Compute the TCP pose in base frame from a reported J6 pose."""

    offset_base = rotation_matrix_xyz_deg(
        j6_pose.rx_deg,
        j6_pose.ry_deg,
        j6_pose.rz_deg,
    ) @ _vector3(offset_j6_tcp_mm, "offset_j6_tcp_mm")
    return EndPose(
        x_mm=float(j6_pose.x_mm + offset_base[0]),
        y_mm=float(j6_pose.y_mm + offset_base[1]),
        z_mm=float(j6_pose.z_mm + offset_base[2]),
        rx_deg=j6_pose.rx_deg,
        ry_deg=j6_pose.ry_deg,
        rz_deg=j6_pose.rz_deg,
    )


def build_catch_plan(
    target_tcp_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float] = OFFICIAL_GRIPPER_TCP_OFFSET_MM,
    approach_height_mm: float = 50.0,
    lift_height_mm: float = 50.0,
    target_z_offset_mm: float = 0.0,
) -> CatchPlan:
    if approach_height_mm < 0:
        raise ValueError("approach_height_mm must be >= 0")
    if lift_height_mm < 0:
        raise ValueError("lift_height_mm must be >= 0")

    grasp_tcp_pose = EndPose(
        target_tcp_pose.x_mm,
        target_tcp_pose.y_mm,
        target_tcp_pose.z_mm + target_z_offset_mm,
        target_tcp_pose.rx_deg,
        target_tcp_pose.ry_deg,
        target_tcp_pose.rz_deg,
    )
    approach_tcp_pose = EndPose(
        grasp_tcp_pose.x_mm,
        grasp_tcp_pose.y_mm,
        grasp_tcp_pose.z_mm + approach_height_mm,
        grasp_tcp_pose.rx_deg,
        grasp_tcp_pose.ry_deg,
        grasp_tcp_pose.rz_deg,
    )
    lift_tcp_pose = EndPose(
        grasp_tcp_pose.x_mm,
        grasp_tcp_pose.y_mm,
        grasp_tcp_pose.z_mm + lift_height_mm,
        grasp_tcp_pose.rx_deg,
        grasp_tcp_pose.ry_deg,
        grasp_tcp_pose.rz_deg,
    )
    return CatchPlan(
        target_tcp_pose=target_tcp_pose,
        approach_tcp_pose=approach_tcp_pose,
        grasp_tcp_pose=grasp_tcp_pose,
        lift_tcp_pose=lift_tcp_pose,
        approach_j6_pose=j6_pose_for_tcp_pose(approach_tcp_pose, offset_j6_tcp_mm),
        grasp_j6_pose=j6_pose_for_tcp_pose(grasp_tcp_pose, offset_j6_tcp_mm),
        lift_j6_pose=j6_pose_for_tcp_pose(lift_tcp_pose, offset_j6_tcp_mm),
    )


def _vector3(vector: Sequence[float], label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{label} must contain exactly 3 values")
    return value

"""Piper forward kinematics helpers."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from robot_control.linux.config.robot_config import ModifiedDhParam, ROBOT_CONFIG
from robot_control.shared.geometry import EndPose


def forward_kinematics_transform(
    joints_rad: Sequence[float],
    dh_params: Sequence[ModifiedDhParam] = ROBOT_CONFIG.modified_dh_params,
) -> np.ndarray:
    """Return the base-to-J6 transform for Piper modified DH parameters."""

    if len(joints_rad) != len(dh_params):
        raise ValueError(f"expected {len(dh_params)} joint values, got {len(joints_rad)}")

    transform = np.eye(4, dtype=float)
    for joint_rad, param in zip(joints_rad, dh_params):
        transform = (
            transform
            @ _rotation_x(param.alpha_rad)
            @ _translation_x(param.a_mm)
            @ _rotation_z(float(joint_rad) + param.theta_offset_rad)
            @ _translation_z(param.d_mm)
        )
    return transform


def forward_kinematics_pose(
    joints_rad: Sequence[float],
    dh_params: Sequence[ModifiedDhParam] = ROBOT_CONFIG.modified_dh_params,
) -> EndPose:
    transform = forward_kinematics_transform(joints_rad, dh_params)
    rx_deg, ry_deg, rz_deg = euler_xyz_deg_from_rotation_matrix(transform[:3, :3])
    return EndPose(
        x_mm=float(transform[0, 3]),
        y_mm=float(transform[1, 3]),
        z_mm=float(transform[2, 3]),
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
    )


def euler_xyz_deg_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return Piper extrinsic XYZ Euler angles from Rz * Ry * Rx rotation."""

    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")

    sin_ry = max(-1.0, min(1.0, -float(rotation[2, 0])))
    ry_rad = math.asin(sin_ry)
    cos_ry = math.cos(ry_rad)
    if abs(cos_ry) > 1e-9:
        rx_rad = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        rz_rad = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        rx_rad = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        rz_rad = 0.0

    return (
        normalize_degrees(math.degrees(rx_rad)),
        normalize_degrees(math.degrees(ry_rad)),
        normalize_degrees(math.degrees(rz_rad)),
    )


def normalize_degrees(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-9):
        return 180.0
    return normalized


def _rotation_x(angle_rad: float) -> np.ndarray:
    cos_v = math.cos(angle_rad)
    sin_v = math.sin(angle_rad)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_v, -sin_v],
            [0.0, sin_v, cos_v],
        ],
        dtype=float,
    )
    return transform


def _rotation_z(angle_rad: float) -> np.ndarray:
    cos_v = math.cos(angle_rad)
    sin_v = math.sin(angle_rad)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.array(
        [
            [cos_v, -sin_v, 0.0],
            [sin_v, cos_v, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return transform


def _translation_x(distance_mm: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = distance_mm
    return transform


def _translation_z(distance_mm: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[2, 3] = distance_mm
    return transform

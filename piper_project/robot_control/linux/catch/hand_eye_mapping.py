"""Use solved hand-eye calibration to map camera points into robot base frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_control.shared.geometry import EndPose


def load_camera_to_base_transform(path: Path) -> np.ndarray:
    """Load an eye-to-hand result transform from hand_eye_result.json."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"hand-eye result does not exist: {path}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")

    section = document.get("hand_eye_calibration")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration")
    if section.get("mode") != "eye_to_hand":
        raise ValueError(
            "catch_server requires an eye_to_hand result whose transform is "
            "camera -> base"
        )
    transform = section.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration.transform")
    if transform.get("from_frame") != "camera" or transform.get("to_frame") != "base":
        raise ValueError(f"{path} transform must be camera -> base")

    rotation = _matrix3(transform.get("rotation_matrix"), "rotation_matrix")
    translation = _vector3(transform.get("translation_m"), "translation_m")
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def camera_xyz_to_hand_eye_robot_pose(
    x_m: float,
    y_m: float,
    z_m: float,
    camera_to_base: np.ndarray,
    *,
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> EndPose:
    if camera_to_base.shape != (4, 4):
        raise ValueError("camera_to_base transform must be 4x4")
    camera_point = np.array([float(x_m), float(y_m), float(z_m), 1.0], dtype=float)
    base_point_m = camera_to_base @ camera_point
    if not np.all(np.isfinite(base_point_m[:3])):
        raise ValueError("mapped robot point is not finite")
    return EndPose(
        x_mm=float(base_point_m[0] * 1000.0),
        y_mm=float(base_point_m[1] * 1000.0),
        z_mm=float(base_point_m[2] * 1000.0),
        rx_deg=float(rx_deg),
        ry_deg=float(ry_deg),
        rz_deg=float(rz_deg),
    )


def _matrix3(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} must be a 3x3 matrix")
    return matrix


def _vector3(value: Any, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{label} must contain exactly 3 values")
    return vector

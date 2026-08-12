"""Eye-in-hand transform helpers for camera points on the gripper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_control.linux.catch.hand_eye_mapping import load_hand_eye_result_document
from robot_control.linux.hand_eye.calibration import transform_from_end_pose
from robot_control.shared.geometry import EndPose


def load_camera_to_gripper_transform(path: Path) -> np.ndarray:
    """Load the eye-in-hand transform returned by the project solver.

    The JSON metadata labels this as gripper -> camera for historical reasons,
    but the calibration math and OpenCV hand-eye API return camera -> gripper.
    This function names the matrix by how it is used for point mapping.
    """

    document = load_hand_eye_result_document(path)
    section = _eye_in_hand_section(document, path)
    transform = _gripper_to_camera_transform_section(section, path)

    rotation = _matrix3(transform.get("rotation_matrix"), "rotation_matrix")
    translation = _vector3(transform.get("translation_m"), "translation_m")
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def load_gripper_to_camera_transform(path: Path) -> np.ndarray:
    """Backward-compatible alias for load_camera_to_gripper_transform."""

    return load_camera_to_gripper_transform(path)


def camera_to_base_transform_from_gripper_pose(
    gripper_pose: EndPose,
    camera_to_gripper: np.ndarray,
) -> np.ndarray:
    """Return camera -> base for a specific robot gripper pose."""

    if camera_to_gripper.shape != (4, 4):
        raise ValueError("camera_to_gripper transform must be 4x4")
    base_to_gripper = transform_from_end_pose(gripper_pose)
    return base_to_gripper @ camera_to_gripper


def map_camera_point_to_base_xyz_m_eye_in_hand(
    x_m: float,
    y_m: float,
    z_m: float,
    gripper_pose: EndPose,
    camera_to_gripper: np.ndarray,
) -> tuple[float, float, float]:
    """Map camera XYZ to base XYZ for a camera rigidly attached to the gripper."""

    camera_to_base = camera_to_base_transform_from_gripper_pose(
        gripper_pose,
        camera_to_gripper,
    )
    camera_point = np.array([float(x_m), float(y_m), float(z_m), 1.0], dtype=float)
    base_point = camera_to_base @ camera_point
    if not np.all(np.isfinite(base_point[:3])):
        raise ValueError("mapped robot point is not finite")
    return tuple(float(value) for value in base_point[:3])


def invert_transform(transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _eye_in_hand_section(document: dict[str, Any], path: Path) -> dict[str, Any]:
    section = document.get("hand_eye_calibration")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration")
    if section.get("mode") != "eye_in_hand":
        raise ValueError(
            "catch_server_2 requires an eye_in_hand result whose transform is "
            "gripper -> camera"
        )
    return section


def _gripper_to_camera_transform_section(
    section: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    transform = section.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration.transform")
    if transform.get("from_frame") != "gripper" or transform.get("to_frame") != "camera":
        raise ValueError(f"{path} transform must be gripper -> camera")
    return transform


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

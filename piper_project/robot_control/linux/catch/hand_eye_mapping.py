"""Use solved hand-eye calibration to map camera points into robot base frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_control.shared.geometry import EndPose


def load_camera_to_base_transform(path: Path) -> np.ndarray:
    """Load an eye-to-hand result transform from hand_eye_result.json."""

    document = load_hand_eye_result_document(path)
    section = _hand_eye_section(document, path)
    transform = _camera_to_base_transform_section(section, path)

    rotation = _matrix3(transform.get("rotation_matrix"), "rotation_matrix")
    translation = _vector3(transform.get("translation_m"), "translation_m")
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def load_hand_eye_result_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"hand-eye result does not exist: {path}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def hand_eye_quality_summary(document: dict[str, Any]) -> str | None:
    section = document.get("hand_eye_calibration")
    if not isinstance(section, dict):
        return None
    quality = section.get("quality")
    if not isinstance(quality, dict):
        return None

    translation_rms_mm = _optional_float(quality.get("translation_rms_mm"))
    translation_max_mm = _optional_float(quality.get("translation_max_mm"))
    rotation_rms_deg = _optional_float(quality.get("rotation_rms_deg"))
    rotation_max_deg = _optional_float(quality.get("rotation_max_deg"))
    if all(
        value is None
        for value in (
            translation_rms_mm,
            translation_max_mm,
            rotation_rms_deg,
            rotation_max_deg,
        )
    ):
        return None

    fields = []
    if translation_rms_mm is not None:
        fields.append(f"translation_rms={translation_rms_mm:.1f} mm")
    if translation_max_mm is not None:
        fields.append(f"translation_max={translation_max_mm:.1f} mm")
    if rotation_rms_deg is not None:
        fields.append(f"rotation_rms={rotation_rms_deg:.1f} deg")
    if rotation_max_deg is not None:
        fields.append(f"rotation_max={rotation_max_deg:.1f} deg")

    warnings = []
    if translation_rms_mm is not None and translation_rms_mm > 10.0:
        warnings.append("translation RMS is high")
    if rotation_rms_deg is not None and rotation_rms_deg > 5.0:
        warnings.append("rotation RMS is high")
    suffix = f" ({'; '.join(warnings)})" if warnings else ""
    return ", ".join(fields) + suffix


def map_camera_point_to_base_xyz_m(
    x_m: float,
    y_m: float,
    z_m: float,
    camera_to_base: np.ndarray,
) -> tuple[float, float, float]:
    if camera_to_base.shape != (4, 4):
        raise ValueError("camera_to_base transform must be 4x4")
    camera_point = np.array([float(x_m), float(y_m), float(z_m), 1.0], dtype=float)
    base_point_m = camera_to_base @ camera_point
    if not np.all(np.isfinite(base_point_m[:3])):
        raise ValueError("mapped robot point is not finite")
    return tuple(float(value) for value in base_point_m[:3])



def _hand_eye_section(document: dict[str, Any], path: Path) -> dict[str, Any]:
    section = document.get("hand_eye_calibration")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration")
    if section.get("mode") != "eye_to_hand":
        raise ValueError(
            "catch_server requires an eye_to_hand result whose transform is "
            "camera -> base"
        )
    return section


def _camera_to_base_transform_section(section: dict[str, Any], path: Path) -> dict[str, Any]:
    transform = section.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} does not contain hand_eye_calibration.transform")
    if transform.get("from_frame") != "camera" or transform.get("to_frame") != "base":
        raise ValueError(f"{path} transform must be camera -> base")
    return transform


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
    base_x_m, base_y_m, base_z_m = map_camera_point_to_base_xyz_m(
        x_m,
        y_m,
        z_m,
        camera_to_base,
    )
    return EndPose(
        x_mm=float(base_x_m * 1000.0),
        y_mm=float(base_y_m * 1000.0),
        z_mm=float(base_z_m * 1000.0),
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


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

"""Hand-eye calibration math and sample-file helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from robot_control.shared.geometry import EndPose
from robot_control.shared.hand_eye_protocol import HandEyeSampleCommand, PoseVector


SCHEMA_VERSION = 1
ROTATION_CONVENTION = "extrinsic_xyz_degrees_rz_ry_rx"


@dataclass(frozen=True)
class HandEyeSample:
    index: int
    sample_id: str
    robot_base_to_gripper: EndPose
    target_to_camera: PoseVector
    image_path: str | None = None
    reprojection_error_px: float | None = None
    detected_corners: int | None = None


@dataclass(frozen=True)
class HandEyeResult:
    mode: str
    rotation: np.ndarray
    translation_m: np.ndarray
    sample_count: int


@dataclass(frozen=True)
class ResidualMetrics:
    translation_rms_m: float
    translation_max_m: float
    rotation_rms_deg: float
    rotation_max_deg: float
    per_sample_translation_m: tuple[float, ...]
    per_sample_rotation_deg: tuple[float, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rotation_matrix_xyz_deg(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """Return Rz * Ry * Rx for Piper's reported extrinsic XYZ Euler angles."""

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


def transform_from_end_pose(pose: EndPose) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_xyz_deg(
        pose.rx_deg,
        pose.ry_deg,
        pose.rz_deg,
    )
    transform[:3, 3] = np.array(
        [pose.x_mm, pose.y_mm, pose.z_mm],
        dtype=float,
    ) / 1000.0
    return transform


def transform_from_pose_vector(pose: PoseVector) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    rotation, _ = cv2.Rodrigues(np.asarray(pose.rvec, dtype=float).reshape(3, 1))
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(pose.tvec_m, dtype=float).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def pose_vector_from_transform(transform: np.ndarray) -> PoseVector:
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return PoseVector(
        rvec=tuple(float(value) for value in rvec.reshape(3)),
        tvec_m=tuple(float(value) for value in transform[:3, 3]),
    )


def make_sample(
    index: int,
    command: HandEyeSampleCommand,
    robot_pose: EndPose,
) -> HandEyeSample:
    return HandEyeSample(
        index=index,
        sample_id=command.sample_id or str(index),
        robot_base_to_gripper=robot_pose,
        target_to_camera=command.target_to_camera,
        image_path=command.image_path,
        reprojection_error_px=command.reprojection_error_px,
        detected_corners=command.detected_corners,
    )


def make_samples_document(
    samples: Sequence[HandEyeSample],
    *,
    mode: str = "eye_to_hand",
    can_name: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if mode not in ("eye_to_hand", "eye_in_hand"):
        raise ValueError("mode must be eye_to_hand or eye_in_hand")
    document: dict[str, Any] = {
        "hand_eye_samples": {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "captured_at": utc_now(),
            "robot_pose_frame": "base_to_gripper",
            "target_pose_frame": "target_to_camera",
            "robot_rotation_convention": ROTATION_CONVENTION,
            "sample_count": len(samples),
            "samples": [_sample_to_dict(sample) for sample in samples],
        }
    }
    if can_name is not None:
        document["hand_eye_samples"]["can_name"] = can_name
    if notes:
        document["hand_eye_samples"]["notes"] = notes
    return document


def load_samples(path: Path) -> list[HandEyeSample]:
    root = _load_json_mapping(path)
    section = root.get("hand_eye_samples")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain hand_eye_samples")
    raw_samples = section.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{path} hand_eye_samples.samples must be a list")
    samples: list[HandEyeSample] = []
    for fallback_index, raw_sample in enumerate(raw_samples, start=1):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"sample {fallback_index} must be an object")
        samples.append(_sample_from_dict(raw_sample, fallback_index))
    return samples


def load_mode(path: Path, default: str = "eye_to_hand") -> str:
    root = _load_json_mapping(path)
    section = root.get("hand_eye_samples")
    if not isinstance(section, dict):
        return default
    mode = str(section.get("mode", default))
    if mode not in ("eye_to_hand", "eye_in_hand"):
        raise ValueError(f"unsupported hand-eye mode in {path}: {mode}")
    return mode


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def solve_hand_eye(
    samples: Sequence[HandEyeSample],
    *,
    mode: str = "eye_to_hand",
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> HandEyeResult:
    if mode not in ("eye_to_hand", "eye_in_hand"):
        raise ValueError("mode must be eye_to_hand or eye_in_hand")
    if len(samples) < 4:
        raise ValueError("at least 4 samples are required; 20-30 are recommended")

    base_to_gripper = [
        transform_from_end_pose(sample.robot_base_to_gripper) for sample in samples
    ]
    target_to_camera = [
        transform_from_pose_vector(sample.target_to_camera) for sample in samples
    ]

    if mode == "eye_in_hand":
        r_gripper2base = [transform[:3, :3] for transform in base_to_gripper]
        t_gripper2base = [transform[:3, 3].reshape(3, 1) for transform in base_to_gripper]
        r_target2cam = [transform[:3, :3] for transform in target_to_camera]
        t_target2cam = [transform[:3, 3].reshape(3, 1) for transform in target_to_camera]
        rotation, translation = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method,
        )
        return HandEyeResult(
            mode=mode,
            rotation=np.asarray(rotation, dtype=float),
            translation_m=np.asarray(translation, dtype=float).reshape(3),
            sample_count=len(samples),
        )

    # Eye-to-hand: camera is fixed in base frame, target is moved by the gripper.
    # For OpenCV this is solved by passing base->gripper and target->camera;
    # the returned transform is camera->base.
    base_to_gripper_inv = [invert_transform(transform) for transform in base_to_gripper]
    r_base2gripper = [transform[:3, :3] for transform in base_to_gripper_inv]
    t_base2gripper = [
        transform[:3, 3].reshape(3, 1) for transform in base_to_gripper_inv
    ]
    r_target2cam = [transform[:3, :3] for transform in target_to_camera]
    t_target2cam = [transform[:3, 3].reshape(3, 1) for transform in target_to_camera]
    rotation, translation = cv2.calibrateHandEye(
        r_base2gripper,
        t_base2gripper,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    return HandEyeResult(
        mode=mode,
        rotation=np.asarray(rotation, dtype=float),
        translation_m=np.asarray(translation, dtype=float).reshape(3),
        sample_count=len(samples),
    )


def make_result_document(
    result: HandEyeResult,
    *,
    source_samples: Path,
    method_name: str,
    residuals: ResidualMetrics | None = None,
) -> dict[str, Any]:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = result.rotation
    transform[:3, 3] = result.translation_m
    pose_vector = pose_vector_from_transform(transform)
    if result.mode == "eye_in_hand":
        from_frame = "gripper"
        to_frame = "camera"
    else:
        from_frame = "camera"
        to_frame = "base"
    document = {
        "hand_eye_calibration": {
            "schema_version": SCHEMA_VERSION,
            "mode": result.mode,
            "method": method_name,
            "calibrated_at": utc_now(),
            "source_samples": str(source_samples),
            "sample_count": result.sample_count,
            "transform": {
                "from_frame": from_frame,
                "to_frame": to_frame,
                "rotation_matrix": result.rotation.tolist(),
                "translation_m": [
                    float(value) for value in result.translation_m.reshape(3)
                ],
                "translation_mm": [
                    float(value * 1000.0) for value in result.translation_m.reshape(3)
                ],
                "rvec": list(pose_vector.rvec),
                "tvec_m": list(pose_vector.tvec_m),
            },
            "notes": (
                "eye_to_hand outputs camera_to_base. eye_in_hand outputs "
                "gripper_to_camera."
            ),
        }
    }
    if residuals is not None:
        document["hand_eye_calibration"]["quality"] = {
            "translation_rms_m": residuals.translation_rms_m,
            "translation_max_m": residuals.translation_max_m,
            "translation_rms_mm": residuals.translation_rms_m * 1000.0,
            "translation_max_mm": residuals.translation_max_m * 1000.0,
            "rotation_rms_deg": residuals.rotation_rms_deg,
            "rotation_max_deg": residuals.rotation_max_deg,
            "per_sample_translation_m": list(residuals.per_sample_translation_m),
            "per_sample_rotation_deg": list(residuals.per_sample_rotation_deg),
        }
    return document


def evaluate_residuals(
    samples: Sequence[HandEyeSample],
    result: HandEyeResult,
) -> ResidualMetrics:
    if not samples:
        raise ValueError("at least one sample is required")
    solved = np.eye(4, dtype=float)
    solved[:3, :3] = result.rotation
    solved[:3, 3] = result.translation_m

    constants: list[np.ndarray] = []
    for sample in samples:
        base_to_gripper = transform_from_end_pose(sample.robot_base_to_gripper)
        target_to_camera = transform_from_pose_vector(sample.target_to_camera)
        if result.mode == "eye_to_hand":
            # bTg * gTt = bTc * cTt, so gTt should be constant.
            constants.append(
                invert_transform(base_to_gripper) @ solved @ target_to_camera
            )
        elif result.mode == "eye_in_hand":
            # bTt = bTg * gTc * cTt should be constant.
            constants.append(base_to_gripper @ solved @ target_to_camera)
        else:
            raise ValueError(f"unsupported hand-eye mode: {result.mode}")

    reference = constants[0]
    translation_errors = []
    rotation_errors = []
    for transform in constants:
        delta = invert_transform(reference) @ transform
        translation_errors.append(float(np.linalg.norm(delta[:3, 3])))
        rotation_errors.append(rotation_angle_deg(delta[:3, :3]))

    translation_array = np.asarray(translation_errors, dtype=float)
    rotation_array = np.asarray(rotation_errors, dtype=float)
    return ResidualMetrics(
        translation_rms_m=float(np.sqrt(np.mean(np.square(translation_array)))),
        translation_max_m=float(np.max(translation_array)),
        rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotation_array)))),
        rotation_max_deg=float(np.max(rotation_array)),
        per_sample_translation_m=tuple(translation_errors),
        per_sample_rotation_deg=tuple(rotation_errors),
    )


def rotation_angle_deg(rotation: np.ndarray) -> float:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return math.degrees(math.acos(value))


def method_from_name(name: str) -> int:
    normalized = name.lower().replace("-", "_")
    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    try:
        return methods[normalized]
    except KeyError:
        raise ValueError(
            f"unsupported method {name!r}; choose one of {', '.join(methods)}"
        ) from None


def format_end_pose(pose: EndPose) -> str:
    return (
        f"x={pose.x_mm:.3f} y={pose.y_mm:.3f} z={pose.z_mm:.3f} mm, "
        f"rx={pose.rx_deg:.3f} ry={pose.ry_deg:.3f} rz={pose.rz_deg:.3f} deg"
    )


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"file does not exist: {path}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _sample_to_dict(sample: HandEyeSample) -> dict[str, Any]:
    return {
        "index": sample.index,
        "sample_id": sample.sample_id,
        "robot_base_to_gripper": _end_pose_dict(sample.robot_base_to_gripper),
        "target_to_camera": _pose_vector_dict(sample.target_to_camera),
        "image_path": sample.image_path,
        "reprojection_error_px": sample.reprojection_error_px,
        "detected_corners": sample.detected_corners,
    }


def _sample_from_dict(mapping: dict[str, Any], fallback_index: int) -> HandEyeSample:
    index = int(mapping.get("index", fallback_index))
    sample_id = str(mapping.get("sample_id", index))
    robot_pose = _end_pose_from_mapping(
        _require_mapping(mapping.get("robot_base_to_gripper"), "robot_base_to_gripper"),
        "robot_base_to_gripper",
    )
    target_to_camera = _pose_vector_from_mapping(
        _require_mapping(mapping.get("target_to_camera"), "target_to_camera"),
        "target_to_camera",
    )
    return HandEyeSample(
        index=index,
        sample_id=sample_id,
        robot_base_to_gripper=robot_pose,
        target_to_camera=target_to_camera,
        image_path=(
            str(mapping["image_path"]) if mapping.get("image_path") is not None else None
        ),
        reprojection_error_px=(
            float(mapping["reprojection_error_px"])
            if mapping.get("reprojection_error_px") is not None
            else None
        ),
        detected_corners=(
            int(mapping["detected_corners"])
            if mapping.get("detected_corners") is not None
            else None
        ),
    )


def _end_pose_dict(pose: EndPose) -> dict[str, float]:
    return {
        "x_mm": float(pose.x_mm),
        "y_mm": float(pose.y_mm),
        "z_mm": float(pose.z_mm),
        "rx_deg": float(pose.rx_deg),
        "ry_deg": float(pose.ry_deg),
        "rz_deg": float(pose.rz_deg),
    }


def _end_pose_from_mapping(mapping: dict[str, Any], label: str) -> EndPose:
    names = ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")
    try:
        values = [float(mapping[name]) for name in names]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric {', '.join(names)}") from exc
    return EndPose(*values)


def _pose_vector_dict(pose: PoseVector) -> dict[str, list[float]]:
    return {
        "rvec": [float(value) for value in pose.rvec],
        "tvec_m": [float(value) for value in pose.tvec_m],
    }


def _pose_vector_from_mapping(mapping: dict[str, Any], label: str) -> PoseVector:
    return PoseVector(
        rvec=_tuple3(mapping.get("rvec"), f"{label}.rvec"),
        tvec_m=_tuple3(mapping.get("tvec_m"), f"{label}.tvec_m"),
    )


def _tuple3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numeric values")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 3 numeric values") from exc


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def compact_matrix(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def sample_indexes(samples: Iterable[HandEyeSample]) -> list[int]:
    return [sample.index for sample in samples]

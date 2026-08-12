"""Fixed-point pivot calibration helpers for a gripper TCP.

The Piper SDK reports the J6 pose in the robot base frame.  When the desired
TCP touches one fixed reference point from several J6 orientations:

    reference_base = translation_base_j6 + rotation_base_j6 * offset_j6_tcp

The unknown J6-to-TCP translation and the fixed reference point can be solved
together with least squares.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from robot_control.shared.geometry import EndPose


SCHEMA_VERSION = 1
METHOD = "fixed_point_pivot"
ROTATION_CONVENTION = "extrinsic_xyz_degrees_rz_ry_rx"


@dataclass(frozen=True)
class CalibrationSolution:
    offset_j6_tcp_mm: tuple[float, float, float]
    reference_point_base_mm: tuple[float, float, float]
    rms_error_mm: float
    max_error_mm: float
    matrix_rank: int
    condition_number: float
    per_sample_error_mm: tuple[float, ...]


@dataclass(frozen=True)
class ValidationMetrics:
    centroid_base_mm: tuple[float, float, float]
    spread_rms_mm: float
    spread_max_mm: float
    reference_rms_error_mm: float | None
    reference_max_error_mm: float | None
    per_sample_reference_error_mm: tuple[float, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rotation_matrix_xyz_deg(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> np.ndarray:
    """Return Rz * Ry * Rx for Piper's extrinsic XYZ Euler angles."""

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


def tcp_position_base_mm(
    flange_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float],
) -> np.ndarray:
    offset = _vector3(offset_j6_tcp_mm, "offset_j6_tcp_mm")
    translation = np.array(
        [flange_pose.x_mm, flange_pose.y_mm, flange_pose.z_mm],
        dtype=float,
    )
    rotation = rotation_matrix_xyz_deg(
        flange_pose.rx_deg,
        flange_pose.ry_deg,
        flange_pose.rz_deg,
    )
    return translation + rotation @ offset


def solve_fixed_point_tcp(poses: Sequence[EndPose]) -> CalibrationSolution:
    if len(poses) < 4:
        raise ValueError("at least 4 poses are required; 6 or more are recommended")

    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for pose in poses:
        rotation = rotation_matrix_xyz_deg(pose.rx_deg, pose.ry_deg, pose.rz_deg)
        translation = np.array([pose.x_mm, pose.y_mm, pose.z_mm], dtype=float)
        rows.append(np.hstack((rotation, -np.eye(3))))
        targets.append(-translation)

    matrix = np.vstack(rows)
    target = np.concatenate(targets)
    solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if rank < 6:
        raise ValueError(
            "TCP solve is underconstrained: use more poses with clearly different "
            "J6 orientations"
        )

    offset = solution[:3]
    reference = solution[3:]
    points = np.vstack([tcp_position_base_mm(pose, offset) for pose in poses])
    errors = np.linalg.norm(points - reference, axis=1)
    return CalibrationSolution(
        offset_j6_tcp_mm=_tuple3(offset),
        reference_point_base_mm=_tuple3(reference),
        rms_error_mm=float(np.sqrt(np.mean(np.square(errors)))),
        max_error_mm=float(np.max(errors)),
        matrix_rank=int(rank),
        condition_number=float(np.linalg.cond(matrix)),
        per_sample_error_mm=tuple(float(value) for value in errors),
    )


def evaluate_tcp(
    poses: Sequence[EndPose],
    offset_j6_tcp_mm: Sequence[float],
    reference_point_base_mm: Sequence[float] | None = None,
) -> ValidationMetrics:
    if not poses:
        raise ValueError("at least one validation pose is required")

    points = np.vstack(
        [tcp_position_base_mm(pose, offset_j6_tcp_mm) for pose in poses]
    )
    centroid = np.mean(points, axis=0)
    spread_errors = np.linalg.norm(points - centroid, axis=1)

    if reference_point_base_mm is None:
        reference_errors = np.array([], dtype=float)
        reference_rms = None
        reference_max = None
    else:
        reference = _vector3(reference_point_base_mm, "reference_point_base_mm")
        reference_errors = np.linalg.norm(points - reference, axis=1)
        reference_rms = float(np.sqrt(np.mean(np.square(reference_errors))))
        reference_max = float(np.max(reference_errors))

    return ValidationMetrics(
        centroid_base_mm=_tuple3(centroid),
        spread_rms_mm=float(np.sqrt(np.mean(np.square(spread_errors)))),
        spread_max_mm=float(np.max(spread_errors)),
        reference_rms_error_mm=reference_rms,
        reference_max_error_mm=reference_max,
        per_sample_reference_error_mm=tuple(float(value) for value in reference_errors),
    )


def feedback_to_end_pose(feedback: Any) -> EndPose:
    end_pose = getattr(feedback, "end_pose", None)
    if end_pose is None:
        raise ValueError("Piper feedback does not contain end_pose")
    axis_names = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")
    values = [getattr(end_pose, name, None) for name in axis_names]
    if any(value is None for value in values):
        raise ValueError("Piper end_pose feedback is incomplete")
    return EndPose(*(float(value) / 1000.0 for value in values))


def read_fresh_end_pose(
    controller: Any,
    timeout_s: float = 2.0,
    settle_s: float = 0.3,
) -> EndPose:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if settle_s < 0:
        raise ValueError("settle_s must be >= 0")
    if settle_s:
        time.sleep(settle_s)

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        feedback = controller.get_end_pose()
        try:
            pose = feedback_to_end_pose(feedback)
        except ValueError as exc:
            last_error = exc
            time.sleep(0.02)
            continue
        timestamp = float(getattr(feedback, "time_stamp", 0.0))
        hz = float(getattr(feedback, "Hz", 0.0))
        if timestamp > 0.0 or hz > 0.0:
            return pose
        last_error = ValueError("Piper end_pose feedback is not fresh")
        time.sleep(0.02)
    raise RuntimeError(f"failed to read fresh Piper end pose: {last_error}")


def collect_poses_interactively(
    controller: Any,
    initial_poses: Sequence[EndPose] = (),
    min_samples: int = 6,
    timeout_s: float = 2.0,
    settle_s: float = 0.3,
    input_fn: Callable[[str], str] = input,
) -> list[EndPose]:
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    poses = list(initial_poses)
    print("This collector does not command robot motion.")
    print("Keep the intended TCP on one fixed reference point at every pose.")
    print("Press Enter to record, 'u' to undo, or 'q' to finish.")
    while True:
        command = input_fn(f"[{len(poses)} samples] command: ").strip().lower()
        if command in ("q", "quit"):
            if len(poses) < min_samples:
                print(f"Need at least {min_samples} samples before finishing.")
                continue
            return poses
        if command in ("u", "undo"):
            if poses:
                removed = poses.pop()
                print(f"Removed sample: {format_pose(removed)}")
            else:
                print("No sample to remove.")
            continue
        if command:
            print("Unknown command. Press Enter, 'u', or 'q'.")
            continue
        pose = read_fresh_end_pose(controller, timeout_s=timeout_s, settle_s=settle_s)
        poses.append(pose)
        print(f"Recorded sample {len(poses)}: {format_pose(pose)}")


def make_samples_document(
    poses: Sequence[EndPose],
    *,
    purpose: str = "calibration",
    can_name: str | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "tcp_samples": {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "purpose": purpose,
            "captured_at": utc_now(),
            "pose_frame": "robot_base_to_j6",
            "pose_rotation_convention": ROTATION_CONVENTION,
            "samples": [
                {"index": index, "flange_pose": _end_pose_dict(pose)}
                for index, pose in enumerate(poses, start=1)
            ],
        }
    }
    if can_name is not None:
        document["tcp_samples"]["can_name"] = can_name
    return document


def make_calibration_document(
    solution: CalibrationSolution,
    *,
    source_samples: Path,
    sample_count: int,
    rms_limit_mm: float,
    max_error_limit_mm: float,
) -> dict[str, Any]:
    passed = (
        solution.rms_error_mm <= rms_limit_mm
        and solution.max_error_mm <= max_error_limit_mm
    )
    x_mm, y_mm, z_mm = solution.offset_j6_tcp_mm
    return {
        "tcp_calibration": {
            "schema_version": SCHEMA_VERSION,
            "status": "calibrated" if passed else "poor_fit",
            "method": METHOD,
            "calibrated_at": utc_now(),
            "source_samples": str(source_samples),
            "sample_count": sample_count,
            "transform": {
                "from_frame": "j6",
                "to_frame": "gripper_tcp",
                "x_mm": x_mm,
                "y_mm": y_mm,
                "z_mm": z_mm,
                "rx_deg": 0.0,
                "ry_deg": 0.0,
                "rz_deg": 0.0,
                "rotation_source": "assumed_aligned_with_j6",
            },
            "fixed_reference_point_base_mm": _xyz_dict(
                solution.reference_point_base_mm
            ),
            "quality": {
                "rms_error_mm": solution.rms_error_mm,
                "max_error_mm": solution.max_error_mm,
                "rms_limit_mm": rms_limit_mm,
                "max_error_limit_mm": max_error_limit_mm,
                "matrix_rank": solution.matrix_rank,
                "condition_number": solution.condition_number,
                "per_sample_error_mm": list(solution.per_sample_error_mm),
            },
            "notes": (
                "Fixed-point pivot calibration solves the J6-to-TCP translation "
                "only. TCP orientation is assumed to be aligned with J6."
            ),
        }
    }


def load_sample_poses(path: Path) -> list[EndPose]:
    root = _load_yaml_mapping(path)
    section = root.get("tcp_samples")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain tcp_samples")
    if section.get("method") != METHOD:
        raise ValueError(f"{path} is not a {METHOD} sample file")
    raw_samples = section.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"{path} tcp_samples.samples must be a list")

    poses: list[EndPose] = []
    for index, raw_sample in enumerate(raw_samples, start=1):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"sample {index} must be a mapping")
        raw_pose = raw_sample.get("flange_pose")
        if not isinstance(raw_pose, dict):
            raise ValueError(f"sample {index} does not contain flange_pose")
        poses.append(_end_pose_from_mapping(raw_pose, f"sample {index} flange_pose"))
    return poses


def load_calibration(path: Path) -> dict[str, Any]:
    root = _load_yaml_mapping(path)
    section = root.get("tcp_calibration")
    if not isinstance(section, dict):
        raise ValueError(f"{path} does not contain tcp_calibration")
    if section.get("method") != METHOD:
        raise ValueError(f"{path} is not a {METHOD} calibration file")
    transform = section.get("transform")
    if not isinstance(transform, dict):
        raise ValueError(f"{path} does not contain tcp_calibration.transform")
    reference = section.get("fixed_reference_point_base_mm")
    if not isinstance(reference, dict):
        raise ValueError(
            f"{path} does not contain tcp_calibration.fixed_reference_point_base_mm"
        )
    return {
        "status": str(section.get("status", "unknown")),
        "offset_j6_tcp_mm": _xyz_from_mapping(transform, "transform"),
        "reference_point_base_mm": _xyz_from_mapping(
            reference,
            "fixed_reference_point_base_mm",
        ),
    }


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def format_pose(pose: EndPose) -> str:
    return (
        f"x={pose.x_mm:.3f} y={pose.y_mm:.3f} z={pose.z_mm:.3f} mm, "
        f"rx={pose.rx_deg:.3f} ry={pose.ry_deg:.3f} rz={pose.rz_deg:.3f} deg"
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"file does not exist: {path}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return document


def _end_pose_from_mapping(mapping: dict[str, Any], label: str) -> EndPose:
    names = ("x_mm", "y_mm", "z_mm", "rx_deg", "ry_deg", "rz_deg")
    try:
        values = [float(mapping[name]) for name in names]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric {', '.join(names)}") from exc
    return EndPose(*values)


def _end_pose_dict(pose: EndPose) -> dict[str, float]:
    return {
        "x_mm": float(pose.x_mm),
        "y_mm": float(pose.y_mm),
        "z_mm": float(pose.z_mm),
        "rx_deg": float(pose.rx_deg),
        "ry_deg": float(pose.ry_deg),
        "rz_deg": float(pose.rz_deg),
    }


def _xyz_from_mapping(mapping: dict[str, Any], label: str) -> tuple[float, float, float]:
    try:
        return tuple(float(mapping[name]) for name in ("x_mm", "y_mm", "z_mm"))  # type: ignore[return-value]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric x_mm, y_mm, z_mm") from exc


def _xyz_dict(vector: Sequence[float]) -> dict[str, float]:
    x_mm, y_mm, z_mm = _vector3(vector, "xyz")
    return {"x_mm": float(x_mm), "y_mm": float(y_mm), "z_mm": float(z_mm)}


def _vector3(vector: Sequence[float], label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,):
        raise ValueError(f"{label} must contain exactly 3 values")
    return value


def _tuple3(vector: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _vector3(vector, "xyz")
    return float(x), float(y), float(z)

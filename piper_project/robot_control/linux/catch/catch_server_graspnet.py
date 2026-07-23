"""Safety-first GraspNet catch server with full 6-DOF IK feasibility checks.

The vision client proposes grasps. This server owns robot feasibility: it
converts candidates to the Piper TCP frame, solves complete IK, checks a
continuous Cartesian approach, and executes only the prevalidated joint path.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
import socket
import sys
import time
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_control.linux.arm.kinematics import (
    euler_xyz_deg_from_rotation_matrix,
    forward_kinematics_transform,
)
from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import format_pose
from robot_control.linux.catch.catch_server_1 import (
    grasp_effort_nm,
    joint_feedback_to_rad,
    maybe_prepare_gripper,
    send_gripper,
)
from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
)
from robot_control.linux.catch.tcp_offset import rotation_matrix_xyz_deg
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    CameraPointCommand,
    GraspCandidate,
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.shared.simulated_mapping import pose_to_dict


DEFAULT_HAND_EYE_RESULT = (
    Path(__file__).resolve().parents[1] / "hand_eye" / "eye_to_hand_result_auto.json"
)
JOINT_COUNT = 6


@dataclass(frozen=True)
class ResolvedCandidate:
    source_index: int | None
    source: str
    score: float
    width_m: float | None
    camera_position_m: tuple[float, float, float]
    grasp_pose: EndPose
    approach_axis: np.ndarray
    orientation_variant: str


@dataclass(frozen=True)
class IkSolution:
    joints_rad: tuple[float, ...]
    pose: EndPose
    position_error_mm: float
    orientation_error_deg: float
    condition_number: float
    joint_margin_deg: float


@dataclass(frozen=True)
class GraspPlan:
    command: CameraPointCommand
    candidate: ResolvedCandidate
    start_joints_rad: tuple[float, ...]
    pregrasp_pose: EndPose
    grasp_pose: EndPose
    transfer_path: tuple[tuple[float, ...], ...]
    approach_path: tuple[tuple[float, ...], ...]
    transfer_cost: float
    maximum_joint_delta_deg: float
    minimum_joint_margin_deg: float
    maximum_condition_number: float
    approach_tilt_deg: float
    total_cost: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive GraspNet 6D candidates and execute only candidates that pass "
            "full IK and continuous joint-trajectory safety checks."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--hand-eye-result", type=Path, default=DEFAULT_HAND_EYE_RESULT)
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--grasp-speed", type=int, default=12)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--joint-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--debug-plan", action="store_true")
    parser.add_argument("--collision-protection-level", type=int, default=2)

    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--approach-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument(
        "--approach-axis-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="sign applied to the selected GraspNet local approach axis",
    )
    parser.add_argument("--approach-distance-mm", type=float, default=90.0)
    parser.add_argument("--approach-sample-distance-mm", type=float, default=10.0)
    parser.add_argument("--max-approach-tilt-deg", type=float, default=45.0)
    parser.add_argument(
        "--allow-approach-axis-flip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also try the opposite approach direction; disabled for predictable safety",
    )
    parser.add_argument(
        "--try-finger-flip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also try the equivalent 180-degree rotation about the approach axis",
    )
    parser.add_argument("--tcp-rot-x-deg", type=float, default=0.0)
    parser.add_argument("--tcp-rot-y-deg", type=float, default=0.0)
    parser.add_argument("--tcp-rot-z-deg", type=float, default=0.0)
    parser.add_argument("--tcp-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-z-mm", type=float, default=0.0)
    parser.add_argument(
        "--fk-tool-z-offset-mm",
        type=float,
        default=82.0,
        help="tool-Z offset from the DH J6 frame to the commanded Piper TCP",
    )

    parser.add_argument("--min-target-x-mm", type=float, default=80.0)
    parser.add_argument("--max-target-x-mm", type=float, default=550.0)
    parser.add_argument("--min-target-y-mm", type=float, default=-300.0)
    parser.add_argument("--max-target-y-mm", type=float, default=300.0)
    parser.add_argument(
        "--min-grasp-z-mm",
        type=float,
        default=120.0,
        help="hard floor for the commanded grasp TCP; never silently clamped",
    )
    parser.add_argument("--max-grasp-z-mm", type=float, default=450.0)
    parser.add_argument(
        "--max-grasp-below-detected-top-mm",
        type=float,
        default=25.0,
        help="reject a TCP farther below the vision-reported object top",
    )
    parser.add_argument("--min-trajectory-tcp-z-mm", type=float, default=110.0)
    parser.add_argument("--max-gripper-width-mm", type=float, default=80.0)

    parser.add_argument("--ik-max-evaluations", type=int, default=300)
    parser.add_argument("--ik-seed-count", type=int, default=8)
    parser.add_argument("--ik-position-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--ik-orientation-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=5.0)
    parser.add_argument("--max-jacobian-condition", type=float, default=500.0)
    parser.add_argument(
        "--nominal-joints-deg",
        type=float,
        nargs=6,
        default=(0.0, 90.0, -90.0, 0.0, 0.0, 0.0),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="planning start used only for dry-run when live feedback is unavailable",
    )
    parser.add_argument(
        "--max-joint-delta-deg",
        type=float,
        nargs=6,
        default=(60.0, 70.0, 70.0, 90.0, 60.0, 90.0),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument(
        "--max-approach-joint-delta-deg",
        type=float,
        nargs=6,
        default=(25.0, 35.0, 35.0, 45.0, 35.0, 45.0),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="maximum total joint change from pregrasp to grasp",
    )
    parser.add_argument("--max-joint-waypoint-step-deg", type=float, default=8.0)
    parser.add_argument(
        "--joint-cost-weights",
        type=float,
        nargs=6,
        default=(1.2, 1.5, 1.5, 0.8, 1.0, 0.6),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--grasp-score-weight", type=float, default=2.0)
    parser.add_argument("--joint-motion-weight", type=float, default=1.0)
    parser.add_argument("--joint-margin-weight", type=float, default=0.5)
    parser.add_argument("--condition-weight", type=float, default=0.1)
    parser.add_argument(
        "--approach-tilt-weight",
        type=float,
        default=1.0,
        help="cost applied to approach tilt/maximum tilt, preferring top-down grasps",
    )

    parser.add_argument("--open-command-mm", type=float, default=0.0)
    parser.add_argument("--grasp-command-mm", type=float, default=-100.0)
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument("--grasp-effort-nm", type=float)
    parser.add_argument("--max-gripper-effort-nm", type=float, default=0.8)
    parser.add_argument(
        "--safe-gripper-stop-on-effort",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--safe-gripper-monitor-s", type=float, default=0.6)
    parser.add_argument("--safe-gripper-poll-s", type=float, default=0.02)
    parser.add_argument("--safe-gripper-release-mm", type=float)
    parser.add_argument("--gripper-settle-s", type=float, default=0.3)
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.speed <= 100 or not 0 <= args.grasp_speed <= 100:
        raise ValueError("--speed and --grasp-speed must be in [0, 100]")
    if args.max_candidates < 1 or args.ik_seed_count < 1:
        raise ValueError("--max-candidates and --ik-seed-count must be >= 1")
    positive = (
        "approach_distance_mm",
        "approach_sample_distance_mm",
        "ik_position_tolerance_mm",
        "ik_orientation_tolerance_deg",
        "joint_limit_margin_deg",
        "max_jacobian_condition",
        "max_joint_waypoint_step_deg",
        "motion_timeout_s",
        "joint_tolerance_deg",
    )
    for name in positive:
        if not math.isfinite(float(getattr(args, name))) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    finite = (
        "tcp_rot_x_deg",
        "tcp_rot_y_deg",
        "tcp_rot_z_deg",
        "tcp_offset_x_mm",
        "tcp_offset_y_mm",
        "tcp_offset_z_mm",
        "fk_tool_z_offset_mm",
        "min_target_x_mm",
        "max_target_x_mm",
        "min_target_y_mm",
        "max_target_y_mm",
        "min_grasp_z_mm",
        "max_grasp_z_mm",
        "max_grasp_below_detected_top_mm",
        "min_trajectory_tcp_z_mm",
    )
    for name in finite:
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.min_grasp_z_mm < args.min_trajectory_tcp_z_mm:
        raise ValueError("--min-grasp-z-mm must be >= --min-trajectory-tcp-z-mm")
    if args.min_grasp_z_mm > args.max_grasp_z_mm:
        raise ValueError("--min-grasp-z-mm must be <= --max-grasp-z-mm")
    if any(value <= 0 for value in args.max_joint_delta_deg):
        raise ValueError("all --max-joint-delta-deg values must be > 0")
    if any(value <= 0 for value in args.max_approach_joint_delta_deg):
        raise ValueError("all --max-approach-joint-delta-deg values must be > 0")
    if args.collision_protection_level is not None and not (
        0 <= args.collision_protection_level <= 8
    ):
        raise ValueError("--collision-protection-level must be in [0, 8]")
    if not 0.0 < args.max_approach_tilt_deg <= 180.0:
        raise ValueError("--max-approach-tilt-deg must be in (0, 180]")
    if not math.isfinite(args.approach_tilt_weight) or args.approach_tilt_weight < 0.0:
        raise ValueError("--approach-tilt-weight must be finite and >= 0")


def pose_transform(pose: EndPose) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix_xyz_deg(
        pose.rx_deg, pose.ry_deg, pose.rz_deg
    )
    transform[:3, 3] = (pose.x_mm, pose.y_mm, pose.z_mm)
    return transform


def transform_pose(transform: np.ndarray) -> EndPose:
    rx_deg, ry_deg, rz_deg = euler_xyz_deg_from_rotation_matrix(transform[:3, :3])
    return EndPose(
        x_mm=float(transform[0, 3]),
        y_mm=float(transform[1, 3]),
        z_mm=float(transform[2, 3]),
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
    )


def fk_tcp_transform(joints_rad: Sequence[float], args: argparse.Namespace) -> np.ndarray:
    transform = forward_kinematics_transform(joints_rad).copy()
    transform[:3, 3] += transform[:3, :3] @ np.array(
        [0.0, 0.0, float(args.fk_tool_z_offset_mm)], dtype=float
    )
    return transform


def fk_tcp_pose(joints_rad: Sequence[float], args: argparse.Namespace) -> EndPose:
    return transform_pose(fk_tcp_transform(joints_rad, args))


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}[axis]


def _local_axis_rotation(axis: str, angle_deg: float) -> np.ndarray:
    vector = np.zeros(3, dtype=float)
    vector[_axis_index(axis)] = math.radians(angle_deg)
    return Rotation.from_rotvec(vector).as_matrix()


def resolve_candidate_variants(
    candidate: GraspCandidate,
    command: CameraPointCommand,
    camera_to_base: np.ndarray,
    args: argparse.Namespace,
) -> tuple[ResolvedCandidate, ...]:
    if camera_to_base.shape != (4, 4) or not np.all(np.isfinite(camera_to_base)):
        raise ValueError("camera_to_base transform must be a finite 4x4 matrix")
    position_camera = np.asarray(candidate.position_camera_m, dtype=float)
    rotation_camera = np.asarray(candidate.rotation_camera, dtype=float)
    if position_camera.shape != (3,) or rotation_camera.shape != (3, 3):
        raise ValueError("GraspNet candidate pose must be 3D position plus 3x3 rotation")
    if not np.all(np.isfinite(position_camera)) or not np.all(np.isfinite(rotation_camera)):
        raise ValueError("GraspNet candidate pose must be finite")
    if not np.allclose(rotation_camera.T @ rotation_camera, np.eye(3), atol=2e-2):
        raise ValueError("GraspNet candidate rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation_camera)), 1.0, abs_tol=2e-2):
        raise ValueError("GraspNet candidate rotation must be right-handed")
    if candidate.score is not None and not math.isfinite(float(candidate.score)):
        raise ValueError("GraspNet candidate score must be finite")
    if candidate.width_m is not None and (
        not math.isfinite(float(candidate.width_m)) or candidate.width_m < 0.0
    ):
        raise ValueError("GraspNet candidate width must be finite and >= 0")

    point_base = camera_to_base @ np.array([*position_camera, 1.0], dtype=float)
    grasp_rotation_base = camera_to_base[:3, :3] @ rotation_camera
    grasp_to_tcp = rotation_matrix_xyz_deg(
        args.tcp_rot_x_deg, args.tcp_rot_y_deg, args.tcp_rot_z_deg
    )
    position_mm = point_base[:3] * 1000.0 + grasp_rotation_base @ np.array(
        [args.tcp_offset_x_mm, args.tcp_offset_y_mm, args.tcp_offset_z_mm],
        dtype=float,
    )
    if command.grasp_mode == "dynamic":
        raise SafetyError(
            "dynamic GraspNet grasp is disabled in phase 1: plan only stationary targets"
        )

    finger_angles = (0.0, 180.0) if args.try_finger_flip else (0.0,)
    axis_signs = (args.approach_axis_sign,)
    if args.allow_approach_axis_flip:
        axis_signs = (args.approach_axis_sign, -args.approach_axis_sign)
    variants: list[ResolvedCandidate] = []
    for axis_sign in axis_signs:
        for finger_angle in finger_angles:
            rotation = (
                grasp_rotation_base
                @ _local_axis_rotation(args.approach_axis, finger_angle)
                @ grasp_to_tcp
            )
            approach_axis = (
                grasp_rotation_base[:, _axis_index(args.approach_axis)]
                * float(axis_sign)
            )
            grasp_transform = np.eye(4, dtype=float)
            grasp_transform[:3, :3] = rotation
            grasp_transform[:3, 3] = position_mm
            variants.append(
                ResolvedCandidate(
                    source_index=candidate.index,
                    source=candidate.source,
                    score=0.0 if candidate.score is None else float(candidate.score),
                    width_m=candidate.width_m,
                    camera_position_m=tuple(float(value) for value in position_camera),
                    grasp_pose=transform_pose(grasp_transform),
                    approach_axis=approach_axis,
                    orientation_variant=(
                        f"axis_sign={axis_sign},finger_rotation={finger_angle:.0f}deg"
                    ),
                )
            )
    return tuple(variants)


def pregrasp_pose(candidate: ResolvedCandidate, args: argparse.Namespace) -> EndPose:
    transform = pose_transform(candidate.grasp_pose)
    transform[:3, 3] -= candidate.approach_axis * float(args.approach_distance_mm)
    return transform_pose(transform)


def approach_tilt_deg(candidate: ResolvedCandidate) -> float:
    downward = np.array([0.0, 0.0, -1.0], dtype=float)
    return math.degrees(
        math.acos(
            float(np.clip(np.dot(candidate.approach_axis, downward), -1.0, 1.0))
        )
    )


def validate_candidate_geometry(
    candidate: ResolvedCandidate,
    command: CameraPointCommand,
    controller: PiperController,
    args: argparse.Namespace,
) -> None:
    pose = candidate.grasp_pose
    checks = (
        ("x", pose.x_mm, args.min_target_x_mm, args.max_target_x_mm),
        ("y", pose.y_mm, args.min_target_y_mm, args.max_target_y_mm),
        ("grasp z", pose.z_mm, args.min_grasp_z_mm, args.max_grasp_z_mm),
    )
    for label, value, lower, upper in checks:
        if value < lower or value > upper:
            raise SafetyError(
                f"candidate {label}={value:.3f} is outside [{lower:.3f}, {upper:.3f}] mm"
            )
    if command.detected_top_z_mm is not None:
        minimum_from_top = (
            float(command.detected_top_z_mm)
            - float(args.max_grasp_below_detected_top_mm)
        )
        if pose.z_mm < minimum_from_top:
            raise SafetyError(
                f"candidate grasp z={pose.z_mm:.3f} mm is too far below detected "
                f"top z={float(command.detected_top_z_mm):.3f} mm"
            )
    if candidate.width_m is not None and candidate.width_m * 1000.0 > args.max_gripper_width_mm:
        raise SafetyError("candidate width exceeds configured gripper width")
    tilt_deg = approach_tilt_deg(candidate)
    if tilt_deg > args.max_approach_tilt_deg:
        raise SafetyError(
            f"approach tilt={tilt_deg:.3f}deg exceeds {args.max_approach_tilt_deg:.3f}deg"
        )
    controller.validate_pose(pose)
    controller.validate_pose(pregrasp_pose(candidate, args))


def _joint_bounds(
    controller: PiperController, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    margin = math.radians(float(args.joint_limit_margin_deg))
    lower = np.array(
        [limit.min_value + margin for limit in controller.config.joint_limits_rad],
        dtype=float,
    )
    upper = np.array(
        [limit.max_value - margin for limit in controller.config.joint_limits_rad],
        dtype=float,
    )
    if np.any(lower >= upper):
        raise ValueError("joint limit margin leaves no feasible joint range")
    return lower, upper


def _ik_residual(
    joints_rad: np.ndarray, target: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    current = fk_tcp_transform(joints_rad, args)
    position_residual = (current[:3, 3] - target[:3, 3]) / 100.0
    rotation_residual = Rotation.from_matrix(
        current[:3, :3].T @ target[:3, :3]
    ).as_rotvec() / math.radians(10.0)
    return np.concatenate((position_residual, rotation_residual))


def _ik_objective(
    joints_rad: np.ndarray,
    target: np.ndarray,
    args: argparse.Namespace,
    reference_joints: np.ndarray,
) -> np.ndarray:
    # A weak continuity term selects the nearby IK branch without relaxing pose checks.
    continuity = (joints_rad - reference_joints) * 0.02
    return np.concatenate((_ik_residual(joints_rad, target, args), continuity))


def _finite_difference_jacobian(
    joints_rad: Sequence[float], target: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    joints = np.asarray(joints_rad, dtype=float)
    jacobian = np.empty((6, 6), dtype=float)
    epsilon = 1e-5
    for index in range(JOINT_COUNT):
        plus = joints.copy()
        minus = joints.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        jacobian[:, index] = (
            _ik_residual(plus, target, args) - _ik_residual(minus, target, args)
        ) / (2.0 * epsilon)
    return jacobian


def _joint_margin_deg(
    joints_rad: Sequence[float], controller: PiperController
) -> float:
    return min(
        math.degrees(
            min(value - limit.min_value, limit.max_value - value)
        )
        for value, limit in zip(joints_rad, controller.config.joint_limits_rad)
    )


def solve_ik(
    target_pose: EndPose,
    seeds: Sequence[Sequence[float]],
    controller: PiperController,
    args: argparse.Namespace,
    *,
    reference_joints: Sequence[float],
    local_joint_delta_deg: float | None = None,
) -> IkSolution:
    target = pose_transform(target_pose)
    lower, upper = _joint_bounds(controller, args)
    reference = np.asarray(reference_joints, dtype=float)
    if local_joint_delta_deg is not None:
        local_delta = math.radians(float(local_joint_delta_deg))
        lower = np.maximum(lower, reference - local_delta)
        upper = np.minimum(upper, reference + local_delta)
        if np.any(lower >= upper):
            raise SafetyError("local IK continuity bounds leave no feasible range")
    unique_seeds: list[np.ndarray] = []
    for seed in seeds:
        clipped = np.clip(np.asarray(seed, dtype=float), lower, upper)
        if not any(np.max(np.abs(clipped - existing)) < 1e-5 for existing in unique_seeds):
            unique_seeds.append(clipped)

    solutions: list[IkSolution] = []
    for seed in unique_seeds:
        result = least_squares(
            _ik_objective,
            seed,
            args=(target, args, reference),
            bounds=(lower, upper),
            max_nfev=int(args.ik_max_evaluations),
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        joints = tuple(float(value) for value in result.x)
        solved = fk_tcp_transform(joints, args)
        position_error = float(np.linalg.norm(solved[:3, 3] - target[:3, 3]))
        orientation_error = math.degrees(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(solved[:3, :3].T @ target[:3, :3]).as_rotvec()
                )
            )
        )
        if (
            position_error > args.ik_position_tolerance_mm
            or orientation_error > args.ik_orientation_tolerance_deg
        ):
            continue
        condition = float(
            np.linalg.cond(_finite_difference_jacobian(joints, target, args))
        )
        if not math.isfinite(condition) or condition > args.max_jacobian_condition:
            continue
        solutions.append(
            IkSolution(
                joints_rad=joints,
                pose=fk_tcp_pose(joints, args),
                position_error_mm=position_error,
                orientation_error_deg=orientation_error,
                condition_number=condition,
                joint_margin_deg=_joint_margin_deg(joints, controller),
            )
        )
    if not solutions:
        raise SafetyError(f"full 6D IK failed for {format_pose(target_pose)}")

    weights = np.asarray(args.joint_cost_weights, dtype=float)
    return min(
        solutions,
        key=lambda solution: float(
            np.sum(weights * np.abs(np.asarray(solution.joints_rad) - reference))
        ),
    )


def planning_seeds(
    start_joints: Sequence[float],
    controller: PiperController,
    args: argparse.Namespace,
) -> tuple[tuple[float, ...], ...]:
    lower, upper = _joint_bounds(controller, args)
    start = np.clip(np.asarray(start_joints, dtype=float), lower, upper)
    midpoint = (lower + upper) * 0.5
    seeds = [start, midpoint]
    rng = np.random.default_rng(0)
    while len(seeds) < int(args.ik_seed_count):
        if len(seeds) < 8:
            perturbation = rng.normal(0.0, math.radians(25.0), JOINT_COUNT)
            seeds.append(np.clip(start + perturbation, lower, upper))
        else:
            seeds.append(rng.uniform(lower, upper))
    return tuple(tuple(float(value) for value in seed) for seed in seeds)


def interpolate_joint_path(
    start: Sequence[float], target: Sequence[float], max_step_deg: float
) -> tuple[tuple[float, ...], ...]:
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    max_delta_deg = float(np.max(np.abs(np.degrees(target_array - start_array))))
    segments = max(1, int(math.ceil(max_delta_deg / max_step_deg)))
    return tuple(
        tuple(float(value) for value in start_array + (target_array - start_array) * alpha)
        for alpha in np.linspace(0.0, 1.0, segments + 1)
    )


def interpolate_pose(start: EndPose, end: EndPose, alpha: float) -> EndPose:
    start_transform = pose_transform(start)
    end_transform = pose_transform(end)
    rotation_delta = Rotation.from_matrix(
        start_transform[:3, :3].T @ end_transform[:3, :3]
    ).as_rotvec()
    rotation = start_transform[:3, :3] @ Rotation.from_rotvec(
        rotation_delta * alpha
    ).as_matrix()
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = (
        start_transform[:3, 3]
        + (end_transform[:3, 3] - start_transform[:3, 3]) * alpha
    )
    return transform_pose(transform)


def validate_joint_deltas(
    start: Sequence[float], target: Sequence[float], args: argparse.Namespace
) -> tuple[float, float]:
    deltas_deg = np.abs(np.degrees(np.asarray(target) - np.asarray(start)))
    limits = np.asarray(args.max_joint_delta_deg, dtype=float)
    violations = [
        f"J{index + 1}={delta:.2f}deg>{limit:.2f}deg"
        for index, (delta, limit) in enumerate(zip(deltas_deg, limits))
        if delta > limit
    ]
    if violations:
        raise SafetyError("pregrasp joint change too large: " + ", ".join(violations))
    cost = float(
        np.sum(np.asarray(args.joint_cost_weights, dtype=float) * deltas_deg / limits)
    )
    return float(np.max(deltas_deg)), cost


def validate_path_height(
    path: Sequence[Sequence[float]],
    controller: PiperController,
    args: argparse.Namespace,
    *,
    label: str,
) -> tuple[float, float]:
    minimum_margin_deg = float("inf")
    maximum_condition = 0.0
    for index, joints in enumerate(path):
        pose = fk_tcp_pose(joints, args)
        controller.validate_joints(list(joints))
        controller.validate_pose(pose)
        if pose.z_mm < args.min_trajectory_tcp_z_mm:
            raise SafetyError(
                f"{label}[{index}] TCP z={pose.z_mm:.3f} mm is below trajectory "
                f"floor {args.min_trajectory_tcp_z_mm:.3f} mm"
            )
        target = fk_tcp_transform(joints, args)
        condition = float(
            np.linalg.cond(_finite_difference_jacobian(joints, target, args))
        )
        if not math.isfinite(condition) or condition > args.max_jacobian_condition:
            raise SafetyError(
                f"{label}[{index}] Jacobian condition={condition:.3f} exceeds "
                f"{args.max_jacobian_condition:.3f}"
            )
        maximum_condition = max(maximum_condition, condition)
        minimum_margin_deg = min(
            minimum_margin_deg, _joint_margin_deg(joints, controller)
        )
    return minimum_margin_deg, maximum_condition


def build_candidate_plan(
    command: CameraPointCommand,
    candidate: ResolvedCandidate,
    start_joints: tuple[float, ...],
    controller: PiperController,
    args: argparse.Namespace,
) -> GraspPlan:
    validate_candidate_geometry(candidate, command, controller, args)
    pre_pose = pregrasp_pose(candidate, args)
    seeds = planning_seeds(start_joints, controller, args)
    pre_solution = solve_ik(
        pre_pose,
        seeds,
        controller,
        args,
        reference_joints=start_joints,
    )
    max_delta, transfer_cost = validate_joint_deltas(
        start_joints, pre_solution.joints_rad, args
    )
    transfer_path = interpolate_joint_path(
        start_joints, pre_solution.joints_rad, args.max_joint_waypoint_step_deg
    )
    transfer_margin, transfer_condition = validate_path_height(
        transfer_path, controller, args, label="transfer"
    )

    distance = float(args.approach_distance_mm)
    sample_count = max(
        1, int(math.ceil(distance / float(args.approach_sample_distance_mm)))
    )
    approach_solutions = [pre_solution]
    previous = pre_solution
    for alpha in np.linspace(0.0, 1.0, sample_count + 1)[1:]:
        pose = interpolate_pose(pre_pose, candidate.grasp_pose, float(alpha))
        solution = solve_ik(
            pose,
            (previous.joints_rad,),
            controller,
            args,
            reference_joints=previous.joints_rad,
            local_joint_delta_deg=float(args.max_joint_waypoint_step_deg),
        )
        step_delta = float(
            np.max(
                np.abs(
                    np.degrees(
                        np.asarray(solution.joints_rad)
                        - np.asarray(previous.joints_rad)
                    )
                )
            )
        )
        if step_delta > args.max_joint_waypoint_step_deg:
            raise SafetyError(
                f"continuous approach IK jumps {step_delta:.3f}deg between samples"
            )
        approach_solutions.append(solution)
        previous = solution

    approach_path = tuple(solution.joints_rad for solution in approach_solutions)
    approach_total_delta = np.abs(
        np.degrees(np.asarray(approach_path[-1]) - np.asarray(approach_path[0]))
    )
    approach_limits = np.asarray(args.max_approach_joint_delta_deg, dtype=float)
    approach_violations = [
        f"J{index + 1}={delta:.2f}deg>{limit:.2f}deg"
        for index, (delta, limit) in enumerate(
            zip(approach_total_delta, approach_limits)
        )
        if delta > limit
    ]
    if approach_violations:
        raise SafetyError(
            "total approach joint change too large: "
            + ", ".join(approach_violations)
        )
    approach_margin, approach_condition = validate_path_height(
        approach_path, controller, args, label="approach"
    )
    min_margin = min(transfer_margin, approach_margin)
    max_condition = max(transfer_condition, approach_condition)
    normalized_condition = math.log10(max(1.0, max_condition)) / math.log10(
        args.max_jacobian_condition
    )
    tilt_deg = approach_tilt_deg(candidate)
    total_cost = (
        args.joint_motion_weight * transfer_cost
        + args.joint_margin_weight / max(min_margin, 0.1)
        + args.condition_weight * normalized_condition
        + args.approach_tilt_weight
        * tilt_deg
        / float(args.max_approach_tilt_deg)
        - args.grasp_score_weight * candidate.score
    )
    return GraspPlan(
        command=command,
        candidate=candidate,
        start_joints_rad=start_joints,
        pregrasp_pose=pre_pose,
        grasp_pose=candidate.grasp_pose,
        transfer_path=transfer_path,
        approach_path=approach_path,
        transfer_cost=transfer_cost,
        maximum_joint_delta_deg=max_delta,
        minimum_joint_margin_deg=min_margin,
        maximum_condition_number=max_condition,
        approach_tilt_deg=tilt_deg,
        total_cost=float(total_cost),
    )


def select_plan(
    command: CameraPointCommand,
    camera_to_base: np.ndarray,
    start_joints: tuple[float, ...],
    controller: PiperController,
    args: argparse.Namespace,
) -> tuple[GraspPlan, tuple[str, ...]]:
    if not command.grasp_candidates:
        raise ValueError("GraspNet command contains no grasp candidates")
    plans: list[GraspPlan] = []
    failures: list[str] = []
    candidates = sorted(
        command.grasp_candidates[: int(args.max_candidates)],
        key=lambda item: -float("inf") if item.score is None else item.score,
        reverse=True,
    )
    for raw_candidate in candidates:
        try:
            variants = resolve_candidate_variants(
                raw_candidate, command, camera_to_base, args
            )
        except (ValueError, SafetyError) as exc:
            failures.append(f"candidate {raw_candidate.index}: {exc}")
            continue
        for variant in variants:
            try:
                plans.append(
                    build_candidate_plan(
                        command, variant, start_joints, controller, args
                    )
                )
            except (ValueError, SafetyError) as exc:
                failures.append(
                    f"candidate {raw_candidate.index} ({variant.orientation_variant}): {exc}"
                )
    if not plans:
        detail = "; ".join(failures[:12])
        raise SafetyError(f"no feasible GraspNet candidate: {detail}")
    return min(plans, key=lambda plan: plan.total_cost), tuple(failures)


def _joint_feedback_timestamp(feedback: Any) -> float | None:
    value = getattr(feedback, "time_stamp", None)
    return None if value is None else float(value)


def read_live_joints(controller: PiperController) -> tuple[float, ...]:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        feedback = controller.get_joint_state()
        joints = joint_feedback_to_rad(feedback)
        fresh = (
            float(getattr(feedback, "time_stamp", 0.0)) > 0.0
            or float(getattr(feedback, "Hz", 0.0)) > 0.0
        )
        if joints is not None and fresh:
            return tuple(float(value) for value in joints)
        time.sleep(0.02)
    raise RuntimeError("fresh Piper joint feedback is required for GraspNet planning")


def wait_until_joints_reached(
    controller: PiperController,
    target: Sequence[float],
    args: argparse.Namespace,
    *,
    label: str,
    initial_stamp: float | None,
) -> None:
    deadline = time.monotonic() + float(args.motion_timeout_s)
    last_error = float("inf")
    while time.monotonic() < deadline:
        feedback = controller.get_joint_state()
        current = joint_feedback_to_rad(feedback)
        stamp = _joint_feedback_timestamp(feedback)
        fresh = initial_stamp is None or stamp is None or stamp > initial_stamp
        if current is not None:
            last_error = float(
                np.max(np.abs(np.degrees(np.asarray(current) - np.asarray(target))))
            )
            if fresh and last_error <= args.joint_tolerance_deg:
                return
        time.sleep(max(0.01, float(args.command_period_s)))
    raise TimeoutError(
        f"{label} joint waypoint not reached: max_error={last_error:.3f}deg"
    )


def execute_joint_path(
    controller: PiperController,
    path: Sequence[Sequence[float]],
    args: argparse.Namespace,
    *,
    label: str,
    speed: int,
    wait_each_waypoint: bool = False,
) -> int:
    count = 0
    final_stamp: float | None = None
    for index, joints in enumerate(path[1:], start=1):
        feedback = controller.get_joint_state()
        stamp = _joint_feedback_timestamp(feedback)
        controller.move_joints(list(joints), speed_percent=speed)
        count += 1
        final_stamp = stamp
        if wait_each_waypoint:
            wait_until_joints_reached(
                controller,
                joints,
                args,
                label=f"{label}[{index}]",
                initial_stamp=stamp,
            )
        elif args.command_period_s > 0:
            time.sleep(args.command_period_s)
    if len(path) > 1 and not wait_each_waypoint:
        wait_until_joints_reached(
            controller,
            path[-1],
            args,
            label=f"{label}[final]",
            initial_stamp=final_stamp,
        )
    return count


def execute_plan(
    controller: PiperController, plan: GraspPlan, args: argparse.Namespace
) -> dict[str, int]:
    current = read_live_joints(controller)
    drift_deg = float(
        np.max(np.abs(np.degrees(np.asarray(current) - np.asarray(plan.start_joints_rad))))
    )
    if drift_deg > args.joint_tolerance_deg:
        raise SafetyError(
            f"robot moved {drift_deg:.3f}deg after planning; refusing stale trajectory"
        )
    send_gripper(controller, args, "open")
    counts = {
        "transfer": execute_joint_path(
            controller, plan.transfer_path, args, label="transfer", speed=args.speed
        ),
        "approach": execute_joint_path(
            controller,
            plan.approach_path,
            args,
            label="approach",
            speed=args.grasp_speed,
            wait_each_waypoint=True,
        ),
    }
    send_gripper(controller, args, "close")
    retreat = tuple(reversed(plan.approach_path))
    counts["retreat"] = execute_joint_path(
        controller,
        retreat,
        args,
        label="retreat",
        speed=args.grasp_speed,
        wait_each_waypoint=True,
    )
    return_path = tuple(reversed(plan.transfer_path))
    counts["return"] = execute_joint_path(
        controller, return_path, args, label="return", speed=args.speed
    )
    return counts


def plan_to_dict(plan: GraspPlan) -> dict[str, Any]:
    candidate = plan.candidate
    return {
        "candidate": {
            "source_index": candidate.source_index,
            "source": candidate.source,
            "score": candidate.score,
            "width_m": candidate.width_m,
            "orientation_variant": candidate.orientation_variant,
            "approach_axis_base": candidate.approach_axis.tolist(),
            "camera_position_m": list(candidate.camera_position_m),
        },
        "pregrasp_pose": pose_to_dict(plan.pregrasp_pose),
        "grasp_pose": pose_to_dict(plan.grasp_pose),
        "start_joints_deg": list(np.degrees(plan.start_joints_rad)),
        "pregrasp_joints_deg": list(np.degrees(plan.transfer_path[-1])),
        "grasp_joints_deg": list(np.degrees(plan.approach_path[-1])),
        "transfer_waypoint_count": len(plan.transfer_path),
        "approach_waypoint_count": len(plan.approach_path),
        "maximum_joint_delta_deg": plan.maximum_joint_delta_deg,
        "minimum_joint_margin_deg": plan.minimum_joint_margin_deg,
        "maximum_jacobian_condition": plan.maximum_condition_number,
        "approach_tilt_deg": plan.approach_tilt_deg,
        "transfer_cost": plan.transfer_cost,
        "total_cost": plan.total_cost,
    }


def handle_message(
    message: dict[str, Any],
    *,
    args: argparse.Namespace,
    camera_to_base: np.ndarray,
    validator: PiperController,
    controller: PiperController | None = None,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    if command.grasp_mode == "probe":
        raise ValueError("probe mode is not supported by catch_server_graspnet")
    if command.task != "out":
        raise ValueError("phase-1 catch_server_graspnet supports only task='out'")
    if command.vision_source not in (None, "graspnet"):
        raise ValueError("catch_server_graspnet accepts only GraspNet commands")
    if controller is None:
        start_joints = tuple(math.radians(value) for value in args.nominal_joints_deg)
    else:
        start_joints = read_live_joints(controller)
    validator.validate_joints(start_joints)
    plan, failures = select_plan(
        command, camera_to_base, start_joints, controller or validator, args
    )
    if args.debug_plan or not args.execute:
        print(
            "Selected GraspNet plan: "
            f"candidate={plan.candidate.source_index} "
            f"variant={plan.candidate.orientation_variant} "
            f"pregrasp={format_pose(plan.pregrasp_pose)} "
            f"grasp={format_pose(plan.grasp_pose)} "
            f"max_joint_delta={plan.maximum_joint_delta_deg:.3f}deg "
            f"min_margin={plan.minimum_joint_margin_deg:.3f}deg "
            f"condition={plan.maximum_condition_number:.3f} "
            f"approach_tilt={plan.approach_tilt_deg:.3f}deg "
            f"cost={plan.total_cost:.6f}",
            flush=True,
        )
    counts: dict[str, int] = {}
    if args.execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        counts = execute_plan(controller, plan, args)
    response = make_ack_message(
        command.command_id, pose_to_dict(plan.grasp_pose), executed=bool(args.execute)
    )
    response["payload"].update(
        {
            "vision_source": "graspnet",
            "task": command.task,
            "grasp_mode": command.grasp_mode,
            "plan": plan_to_dict(plan),
            "rejected_variant_count": len(failures),
            "rejected_variants": list(failures) if args.debug_plan else [],
            "send_counts": counts,
            "safety": {
                "minimum_grasp_z_mm": args.min_grasp_z_mm,
                "minimum_trajectory_tcp_z_mm": args.min_trajectory_tcp_z_mm,
                "joint_limit_margin_deg": args.joint_limit_margin_deg,
                "maximum_joint_delta_deg": list(args.max_joint_delta_deg),
                "maximum_approach_joint_delta_deg": list(
                    args.max_approach_joint_delta_deg
                ),
                "maximum_joint_waypoint_step_deg": args.max_joint_waypoint_step_deg,
                "maximum_jacobian_condition": args.max_jacobian_condition,
                "collision_protection_level": args.collision_protection_level,
                "obstacle_collision_planning": False,
                "self_collision_planning": False,
            },
            "gripper": {
                "enabled": not args.no_gripper,
                "grasp_effort_nm": grasp_effort_nm(args),
                "effort_stop_enabled": bool(args.safe_gripper_stop_on_effort),
                "maximum_effort_nm": args.max_gripper_effort_nm,
            },
        }
    )
    return response


def serve(args: argparse.Namespace) -> None:
    validate_args(args)
    hand_eye_document = load_hand_eye_result_document(args.hand_eye_result)
    camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
    print(f"Using hand-eye camera->base transform: {args.hand_eye_result}", flush=True)
    quality = hand_eye_quality_summary(hand_eye_document)
    if quality is not None:
        print(f"Hand-eye quality: {quality}", flush=True)

    config = replace(ROBOT_CONFIG, can_name=args.can)
    validator = PiperController(config=config)
    controller: PiperController | None = None
    if args.execute:
        controller = PiperController(config=config)
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        if args.collision_protection_level is not None:
            controller.set_collision_protection([args.collision_protection_level] * 6)
        maybe_prepare_gripper(controller, args)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"GraspNet catch server listening on {args.host}:{args.port}, "
                f"execute={args.execute}",
                flush=True,
            )
            while True:
                conn, addr = server.accept()
                with conn:
                    print(f"Client connected: {addr}", flush=True)
                    with conn.makefile("rb") as sock_file:
                        while True:
                            message = recv_json_line(sock_file)
                            if message is None:
                                break
                            command_id = str(message.get("command_id"))
                            try:
                                response = handle_message(
                                    message,
                                    args=args,
                                    camera_to_base=camera_to_base,
                                    validator=validator,
                                    controller=controller,
                                )
                            except Exception as exc:  # noqa: BLE001 - returned to client
                                response = make_error_message(command_id, str(exc))
                                print(f"Error: {exc}", flush=True)
                            send_json_line(conn, response)
    finally:
        if controller is not None:
            controller.disconnect()


def main() -> None:
    try:
        serve(build_parser().parse_args())
    except (SafetyError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

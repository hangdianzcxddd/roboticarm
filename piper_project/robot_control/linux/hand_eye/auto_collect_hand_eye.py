"""Automatic small-random-motion hand-eye sample collector."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from robot_control.linux.arm.piper_arm import PiperController, RAD_TO_MILLI_DEG
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.hand_eye.calibration import (
    HandEyeSample,
    format_end_pose,
    load_samples,
    make_sample,
    make_samples_document,
    write_json,
)
from robot_control.linux.hand_eye.move_board_pose import execute_pose_for_duration
from robot_control.linux.tcp_calibration.calibration import (
    feedback_to_end_pose,
    read_fresh_end_pose,
)
from robot_control.shared.geometry import EndPose
from robot_control.shared.hand_eye_auto_protocol import (
    AutoHandEyeRequest,
    AutoHandEyeDetection,
    request_auto_hand_eye_detection,
)
from robot_control.shared.hand_eye_protocol import HandEyeSampleCommand


DEFAULT_OUTPUT = Path(__file__).with_name("auto_hand_eye_samples.json")
POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz")
JOINT_FIELDS = ("j1", "j2", "j3", "j4", "j5", "j6")
WRIST_JOINT_START_INDEX = 3


@dataclass(frozen=True)
class PoseStepLimits:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float


@dataclass(frozen=True)
class JointStepLimits:
    joints_rad: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class PoseBounds:
    min_x_mm: float
    max_x_mm: float
    min_y_mm: float
    max_y_mm: float
    min_z_mm: float
    max_z_mm: float
    min_rx_deg: float
    max_rx_deg: float
    min_ry_deg: float
    max_ry_deg: float
    min_rz_deg: float
    max_rz_deg: float


@dataclass(frozen=True)
class JointBounds:
    min_rad: tuple[float, float, float, float, float, float]
    max_rad: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CandidateResult:
    pose: EndPose
    selected_fields: tuple[str, ...]
    attempt: int


@dataclass(frozen=True)
class JointCandidateResult:
    joints_rad: tuple[float, float, float, float, float, float]
    selected_fields: tuple[str, ...]
    attempt: int


class JointJumpError(RuntimeError):
    """Raised when feedback shows an unsafe joint-space discontinuity."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically move Piper through small random end-pose changes, ask "
            "a Windows chessboard vision server whether the full board is visible, "
            "and save hand-eye samples. Supports eye_to_hand and eye_in_hand."
        )
    )
    parser.add_argument("--vision-host", required=True, help="Windows vision server IP")
    parser.add_argument("--vision-port", type=int, default=5011)
    parser.add_argument("--vision-timeout-s", type=float, default=5.0)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument(
        "--mode",
        choices=("eye_to_hand", "eye_in_hand"),
        default="eye_in_hand",
        help=(
            "eye_to_hand: fixed camera, robot moves the checkerboard. "
            "eye_in_hand: robot moves the camera, checkerboard is fixed."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--notes")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--move-mode", choices=("P", "L"), default="P")
    parser.add_argument(
        "--motion-space",
        choices=("joint", "pose"),
        default="joint",
        help=(
            "joint: default, randomize small JointCtrl targets around the seed pose "
            "to avoid Cartesian IK branch flips. pose: legacy EndPoseCtrl random "
            "walk with joint-jump monitoring."
        ),
    )
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument(
        "--post-move-stabilize-s",
        type=float,
        default=0.5,
        help=(
            "extra time to wait after a candidate pose is reached before reading "
            "robot feedback and requesting chessboard detection"
        ),
    )
    parser.add_argument("--settle-s", type=float, default=0.4)
    parser.add_argument("--read-timeout-s", type=float, default=2.0)
    parser.add_argument("--min-dof", type=int, default=1)
    parser.add_argument("--max-dof", type=int, default=3)
    parser.add_argument("--step-x-mm", type=float, default=20.0)
    parser.add_argument("--step-y-mm", type=float, default=20.0)
    parser.add_argument("--step-z-mm", type=float, default=15.0)
    parser.add_argument("--step-rx-deg", type=float, default=7.0)
    parser.add_argument("--step-ry-deg", type=float, default=7.0)
    parser.add_argument("--step-rz-deg", type=float, default=10.0)
    parser.add_argument(
        "--step-base-joint-deg",
        type=float,
        default=3.0,
        help="maximum random step for J1-J3 when --motion-space joint",
    )
    parser.add_argument(
        "--step-wrist-joint-deg",
        type=float,
        default=4.0,
        help="maximum random step for J4-J6 when --motion-space joint",
    )
    parser.add_argument("--range-x-mm", type=float, default=80.0)
    parser.add_argument("--range-y-mm", type=float, default=80.0)
    parser.add_argument("--range-z-mm", type=float, default=50.0)
    parser.add_argument("--range-rx-deg", type=float, default=25.0)
    parser.add_argument("--range-ry-deg", type=float, default=25.0)
    parser.add_argument("--range-rz-deg", type=float, default=35.0)
    parser.add_argument(
        "--range-base-joint-deg",
        type=float,
        default=15.0,
        help="maximum distance from seed for J1-J3 when --motion-space joint",
    )
    parser.add_argument(
        "--range-wrist-joint-deg",
        type=float,
        default=18.0,
        help="maximum distance from seed for J4-J6 when --motion-space joint",
    )
    parser.add_argument(
        "--max-joint-delta-deg",
        type=float,
        default=12.0,
        help="maximum observed J1-J3 change allowed for one guarded move segment",
    )
    parser.add_argument(
        "--max-wrist-joint-delta-deg",
        type=float,
        default=8.0,
        help="maximum observed J4-J6 change allowed for one guarded move segment",
    )
    parser.add_argument(
        "--joint-tolerance-deg",
        type=float,
        default=2.0,
        help="joint feedback tolerance for guarded JointCtrl moves",
    )
    parser.add_argument("--min-z-mm", type=float, help="absolute lower Z safety bound")
    parser.add_argument("--max-z-mm", type=float, help="absolute upper Z safety bound")
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--max-consecutive-failures", type=int, default=8)
    parser.add_argument(
        "--return-to-seed-after-failures",
        type=int,
        default=3,
        help="return to the seed pose after this many consecutive rejected moves",
    )
    parser.add_argument(
        "--shrink-after-failures",
        type=int,
        default=5,
        help="temporarily shrink random step size after this many failures",
    )
    parser.add_argument("--min-translation-delta-mm", type=float, default=15.0)
    parser.add_argument("--min-rotation-delta-deg", type=float, default=5.0)
    parser.add_argument("--min-margin-px", type=float, default=35.0)
    parser.add_argument("--max-reprojection-error-px", type=float, default=1.0)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="query the vision server at the current pose and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    if args.output.exists() and not args.append and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}. Use --append or --overwrite."
        )
    samples = load_samples(args.output) if args.append and args.output.exists() else []

    controller = PiperController(config=replace(ROBOT_CONFIG, can_name=args.can))
    controller.connect(enable=args.execute)
    try:
        seed_pose = read_fresh_end_pose(
            controller,
            timeout_s=args.read_timeout_s,
            settle_s=args.settle_s,
        )
        print(f"Seed pose: {format_end_pose(seed_pose)}", flush=True)
        precheck = query_vision(args, "precheck", capture=False)
        print(f"Vision precheck: {format_detection(precheck)}", flush=True)
        if args.precheck_only:
            return
        if not precheck.qualified:
            raise SystemExit(
                "Current pose is not a qualified starting view. Move to a pose "
                f"where the full chessboard is visible first: {precheck.reason}"
            )
        if not args.execute:
            print("Dry run only. Add --execute to move and collect samples.")
            return

        initialize_output(args, samples)
        seed_joints = read_fresh_joints_rad(
            controller,
            timeout_s=args.read_timeout_s,
            settle_s=0.0,
        )
        print(f"Seed joints: {format_joints(seed_joints)}", flush=True)

        bounds = pose_bounds_from_seed(seed_pose, args)
        step_limits = PoseStepLimits(
            x_mm=args.step_x_mm,
            y_mm=args.step_y_mm,
            z_mm=args.step_z_mm,
            rx_deg=args.step_rx_deg,
            ry_deg=args.step_ry_deg,
            rz_deg=args.step_rz_deg,
        )
        joint_bounds = joint_bounds_from_seed(seed_joints, controller.config, args)
        joint_step_limits = joint_step_limits_from_args(args)
        collect_samples(
            args,
            controller,
            rng,
            seed_pose,
            seed_joints,
            bounds,
            step_limits,
            joint_bounds,
            joint_step_limits,
            samples,
        )
    finally:
        controller.disconnect()


def validate_args(args: argparse.Namespace) -> None:
    if args.vision_port <= 0:
        raise SystemExit("--vision-port must be > 0")
    if args.vision_timeout_s <= 0:
        raise SystemExit("--vision-timeout-s must be > 0")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.speed <= 0 or args.speed > 100:
        raise SystemExit("--speed must be in [1, 100]")
    if args.command_duration_s < 0:
        raise SystemExit("--command-duration-s must be >= 0")
    if args.command_period_s <= 0:
        raise SystemExit("--command-period-s must be > 0")
    if args.motion_timeout_s <= 0:
        raise SystemExit("--motion-timeout-s must be > 0")
    if args.position_tolerance_mm <= 0:
        raise SystemExit("--position-tolerance-mm must be > 0")
    if args.orientation_tolerance_deg <= 0:
        raise SystemExit("--orientation-tolerance-deg must be > 0")
    if args.post_move_stabilize_s < 0:
        raise SystemExit("--post-move-stabilize-s must be >= 0")
    if args.settle_s < 0:
        raise SystemExit("--settle-s must be >= 0")
    if args.read_timeout_s <= 0:
        raise SystemExit("--read-timeout-s must be > 0")
    if args.min_dof < 1 or args.max_dof < args.min_dof or args.max_dof > 6:
        raise SystemExit("--min-dof/--max-dof must satisfy 1 <= min <= max <= 6")
    for name in (
        "step_x_mm",
        "step_y_mm",
        "step_z_mm",
        "step_rx_deg",
        "step_ry_deg",
        "step_rz_deg",
        "step_base_joint_deg",
        "step_wrist_joint_deg",
        "range_x_mm",
        "range_y_mm",
        "range_z_mm",
        "range_rx_deg",
        "range_ry_deg",
        "range_rz_deg",
        "range_base_joint_deg",
        "range_wrist_joint_deg",
        "max_joint_delta_deg",
        "max_wrist_joint_delta_deg",
        "joint_tolerance_deg",
    ):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 0")
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.max_consecutive_failures < 1:
        raise SystemExit("--max-consecutive-failures must be >= 1")
    if args.return_to_seed_after_failures < 1:
        raise SystemExit("--return-to-seed-after-failures must be >= 1")
    if args.shrink_after_failures < 1:
        raise SystemExit("--shrink-after-failures must be >= 1")
    if args.max_joint_delta_deg <= 0 or args.max_wrist_joint_delta_deg <= 0:
        raise SystemExit("joint jump delta limits must be > 0")
    if args.joint_tolerance_deg <= 0:
        raise SystemExit("--joint-tolerance-deg must be > 0")
    if args.min_translation_delta_mm < 0 or args.min_rotation_delta_deg < 0:
        raise SystemExit("minimum diversity deltas must be >= 0")
    if args.min_margin_px < 0:
        raise SystemExit("--min-margin-px must be >= 0")
    if args.max_reprojection_error_px <= 0:
        raise SystemExit("--max-reprojection-error-px must be > 0")
    if args.stable_frames < 1:
        raise SystemExit("--stable-frames must be >= 1")


def collect_samples(
    args: argparse.Namespace,
    controller: PiperController,
    rng: random.Random,
    seed_pose: EndPose,
    seed_joints: tuple[float, float, float, float, float, float],
    bounds: PoseBounds,
    step_limits: PoseStepLimits,
    joint_bounds: JointBounds,
    joint_step_limits: JointStepLimits,
    samples: list[HandEyeSample],
) -> None:
    last_good_pose = seed_pose
    last_good_joints = seed_joints
    consecutive_failures = 0
    attempts = 0
    while len(samples) < args.samples and attempts < args.max_attempts:
        attempts += 1
        shrink = 0.5 if consecutive_failures >= args.shrink_after_failures else 1.0
        try:
            if args.motion_space == "joint":
                candidate = random_joint_candidate(
                    rng,
                    last_good_joints,
                    joint_bounds,
                    joint_step_limits,
                    args.min_dof,
                    args.max_dof,
                    shrink=shrink,
                    attempt=attempts,
                )
            else:
                candidate = random_candidate(
                    rng,
                    last_good_pose,
                    bounds,
                    step_limits,
                    args.min_dof,
                    args.max_dof,
                    shrink=shrink,
                    attempt=attempts,
                )
        except ValueError as exc:
            consecutive_failures += 1
            print(f"[{attempts}] candidate generation failed: {exc}", flush=True)
            continue

        if (
            args.motion_space == "pose"
            and isinstance(candidate, CandidateResult)
            and not diverse_enough(
                candidate.pose,
                samples,
                min_translation_delta_mm=args.min_translation_delta_mm,
                min_rotation_delta_deg=args.min_rotation_delta_deg,
            )
        ):
            print(
                f"[{attempts}] skipped similar candidate fields={candidate.selected_fields} "
                f"pose=({format_end_pose(candidate.pose)})",
                flush=True,
            )
            continue

        if isinstance(candidate, JointCandidateResult):
            print(
                f"[{attempts}] moving joint candidate fields={candidate.selected_fields}: "
                f"{format_joints(candidate.joints_rad)}",
                flush=True,
            )
        else:
            print(
                f"[{attempts}] moving candidate fields={candidate.selected_fields}: "
                f"{format_end_pose(candidate.pose)}",
                flush=True,
            )
        try:
            actual_joints: tuple[float, float, float, float, float, float] | None = None
            if isinstance(candidate, JointCandidateResult):
                actual_joints = move_to_joints(controller, candidate.joints_rad, args)
            else:
                actual_joints = move_to_pose(controller, candidate.pose, args)
            wait_after_motion(args)
            actual_pose = read_fresh_end_pose(
                controller,
                timeout_s=args.read_timeout_s,
                settle_s=args.settle_s,
            )
            if args.motion_space == "joint" and not diverse_enough(
                actual_pose,
                samples,
                min_translation_delta_mm=args.min_translation_delta_mm,
                min_rotation_delta_deg=args.min_rotation_delta_deg,
            ):
                print(
                    f"[{attempts}] skipped similar actual joint candidate "
                    f"pose=({format_end_pose(actual_pose)})",
                    flush=True,
                )
                move_to_joints(controller, last_good_joints, args)
                continue
            status = query_vision(args, f"candidate-{attempts:03d}", capture=False)
            print(f"[{attempts}] vision status: {format_detection(status)}", flush=True)
            if not status.qualified:
                consecutive_failures += 1
                handle_rejected_motion(
                    args,
                    controller,
                    seed_pose,
                    seed_joints,
                    last_good_pose,
                    last_good_joints,
                    consecutive_failures,
                    status.reason or "unqualified",
                )
                continue

            sample_id = f"auto-{len(samples) + 1:03d}"
            capture = query_vision(args, sample_id, capture=True)
            if not capture.qualified or capture.target_to_camera is None:
                consecutive_failures += 1
                handle_rejected_motion(
                    args,
                    controller,
                    seed_pose,
                    seed_joints,
                    last_good_pose,
                    last_good_joints,
                    consecutive_failures,
                    capture.reason or "capture unqualified",
                )
                continue

            command = HandEyeSampleCommand(
                target_to_camera=capture.target_to_camera,
                sample_id=sample_id,
                image_path=capture.image_path,
                reprojection_error_px=capture.reprojection_error_px,
                detected_corners=capture.detected_corners,
                source="auto_hand_eye",
            )
            sample = make_sample(len(samples) + 1, command, actual_pose)
            samples.append(sample)
            write_json(
                args.output,
                make_samples_document(
                    samples,
                    mode=args.mode,
                    can_name=args.can,
                    notes=args.notes,
                ),
            )
            last_good_pose = actual_pose
            if actual_joints is not None:
                last_good_joints = actual_joints
            consecutive_failures = 0
            print(
                f"Saved sample {sample.index}/{args.samples}: "
                f"pose=({format_end_pose(actual_pose)}) "
                f"vision=({format_detection(capture)})",
                flush=True,
            )
        except JointJumpError as exc:
            raise SystemExit(
                f"Stopped automatic collection after unsafe joint jump: {exc}. "
                "Move the robot manually to a safe pose before resuming."
            ) from None
        except (SafetyError, RuntimeError, TimeoutError, ValueError) as exc:
            consecutive_failures += 1
            print(f"[{attempts}] rejected: {exc}", flush=True)
            handle_rejected_motion(
                args,
                controller,
                seed_pose,
                seed_joints,
                last_good_pose,
                last_good_joints,
                consecutive_failures,
                str(exc),
            )

        if consecutive_failures >= args.max_consecutive_failures:
            raise SystemExit(
                f"Stopped after {consecutive_failures} consecutive failures. "
                "Move to a better seed pose or reduce random ranges."
            )

    if len(samples) < args.samples:
        raise SystemExit(
            f"Collected {len(samples)} samples, but requested {args.samples}; "
            f"max attempts {args.max_attempts} reached."
        )
    print(f"Completed automatic hand-eye collection: {args.output}", flush=True)


def initialize_output(args: argparse.Namespace, samples: list[HandEyeSample]) -> None:
    if args.overwrite and not samples:
        write_json(
            args.output,
            make_samples_document(
                samples,
                mode=args.mode,
                can_name=args.can,
                notes=args.notes,
            ),
        )
        print(f"Initialized fresh sample file: {args.output}", flush=True)


def wait_after_motion(args: argparse.Namespace) -> None:
    if args.post_move_stabilize_s <= 0.0:
        return
    print(
        f"Waiting {args.post_move_stabilize_s:.3f}s for post-move stabilization.",
        flush=True,
    )
    time.sleep(args.post_move_stabilize_s)


def random_candidate(
    rng: random.Random,
    base_pose: EndPose,
    bounds: PoseBounds,
    step_limits: PoseStepLimits,
    min_dof: int,
    max_dof: int,
    *,
    shrink: float,
    attempt: int,
) -> CandidateResult:
    fields = tuple(rng.sample(POSE_FIELDS, rng.randint(min_dof, max_dof)))
    values = pose_to_values(base_pose)
    for field in fields:
        values[field] += rng.uniform(-step_for_field(step_limits, field), step_for_field(step_limits, field)) * shrink
    pose = values_to_pose(values)
    pose = clamp_pose(pose, bounds)
    if same_pose(pose, base_pose):
        raise ValueError("candidate was clamped back to the base pose")
    return CandidateResult(pose=pose, selected_fields=fields, attempt=attempt)


def random_joint_candidate(
    rng: random.Random,
    base_joints_rad: Sequence[float],
    bounds: JointBounds,
    step_limits: JointStepLimits,
    min_dof: int,
    max_dof: int,
    *,
    shrink: float,
    attempt: int,
) -> JointCandidateResult:
    fields = tuple(rng.sample(JOINT_FIELDS, rng.randint(min_dof, max_dof)))
    values = list(base_joints_rad)
    for field in fields:
        index = JOINT_FIELDS.index(field)
        step = step_limits.joints_rad[index]
        values[index] += rng.uniform(-step, step) * shrink
    joints = clamp_joints(values, bounds)
    if same_joints(joints, base_joints_rad):
        raise ValueError("joint candidate was clamped back to the base joints")
    return JointCandidateResult(joints_rad=joints, selected_fields=fields, attempt=attempt)


def step_for_field(step_limits: PoseStepLimits, field: str) -> float:
    return {
        "x": step_limits.x_mm,
        "y": step_limits.y_mm,
        "z": step_limits.z_mm,
        "rx": step_limits.rx_deg,
        "ry": step_limits.ry_deg,
        "rz": step_limits.rz_deg,
    }[field]


def joint_step_limits_from_args(args: argparse.Namespace) -> JointStepLimits:
    base = math.radians(args.step_base_joint_deg)
    wrist = math.radians(args.step_wrist_joint_deg)
    return JointStepLimits((base, base, base, wrist, wrist, wrist))


def pose_bounds_from_seed(seed: EndPose, args: argparse.Namespace) -> PoseBounds:
    min_z = seed.z_mm - args.range_z_mm if args.min_z_mm is None else args.min_z_mm
    max_z = seed.z_mm + args.range_z_mm if args.max_z_mm is None else args.max_z_mm
    return PoseBounds(
        min_x_mm=seed.x_mm - args.range_x_mm,
        max_x_mm=seed.x_mm + args.range_x_mm,
        min_y_mm=seed.y_mm - args.range_y_mm,
        max_y_mm=seed.y_mm + args.range_y_mm,
        min_z_mm=min_z,
        max_z_mm=max_z,
        min_rx_deg=max(-180.0, seed.rx_deg - args.range_rx_deg),
        max_rx_deg=min(180.0, seed.rx_deg + args.range_rx_deg),
        min_ry_deg=max(-180.0, seed.ry_deg - args.range_ry_deg),
        max_ry_deg=min(180.0, seed.ry_deg + args.range_ry_deg),
        min_rz_deg=max(-180.0, seed.rz_deg - args.range_rz_deg),
        max_rz_deg=min(180.0, seed.rz_deg + args.range_rz_deg),
    )


def joint_bounds_from_seed(
    seed_joints_rad: Sequence[float],
    config: RobotConfig,
    args: argparse.Namespace,
) -> JointBounds:
    base_range = math.radians(args.range_base_joint_deg)
    wrist_range = math.radians(args.range_wrist_joint_deg)
    min_values: list[float] = []
    max_values: list[float] = []
    for index, (seed, limit) in enumerate(
        zip(seed_joints_rad, config.joint_limits_rad)
    ):
        allowed_range = wrist_range if index >= WRIST_JOINT_START_INDEX else base_range
        min_values.append(max(limit.min_value, float(seed) - allowed_range))
        max_values.append(min(limit.max_value, float(seed) + allowed_range))
    return JointBounds(
        min_rad=joint_tuple(min_values),
        max_rad=joint_tuple(max_values),
    )


def clamp_pose(pose: EndPose, bounds: PoseBounds) -> EndPose:
    return EndPose(
        x_mm=clamp(pose.x_mm, bounds.min_x_mm, bounds.max_x_mm),
        y_mm=clamp(pose.y_mm, bounds.min_y_mm, bounds.max_y_mm),
        z_mm=clamp(pose.z_mm, bounds.min_z_mm, bounds.max_z_mm),
        rx_deg=clamp(pose.rx_deg, bounds.min_rx_deg, bounds.max_rx_deg),
        ry_deg=clamp(pose.ry_deg, bounds.min_ry_deg, bounds.max_ry_deg),
        rz_deg=normalize_deg(clamp(pose.rz_deg, bounds.min_rz_deg, bounds.max_rz_deg)),
    )


def clamp_joints(
    values: Sequence[float],
    bounds: JointBounds,
) -> tuple[float, float, float, float, float, float]:
    if len(values) != len(JOINT_FIELDS):
        raise ValueError(f"expected 6 joint values, got {len(values)}")
    return joint_tuple(
        clamp(float(value), low, high)
        for value, low, high in zip(values, bounds.min_rad, bounds.max_rad)
    )


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def normalize_deg(value: float) -> float:
    normalized = float(value)
    while normalized > 180.0:
        normalized -= 360.0
    while normalized < -180.0:
        normalized += 360.0
    return normalized


def same_pose(a: EndPose, b: EndPose, eps: float = 1e-6) -> bool:
    return all(abs(x - y) <= eps for x, y in zip(pose_tuple(a), pose_tuple(b)))


def same_joints(
    a: Sequence[float],
    b: Sequence[float],
    eps: float = 1e-9,
) -> bool:
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def pose_to_values(pose: EndPose) -> dict[str, float]:
    return {
        "x": pose.x_mm,
        "y": pose.y_mm,
        "z": pose.z_mm,
        "rx": pose.rx_deg,
        "ry": pose.ry_deg,
        "rz": pose.rz_deg,
    }


def values_to_pose(values: dict[str, float]) -> EndPose:
    return EndPose(
        x_mm=values["x"],
        y_mm=values["y"],
        z_mm=values["z"],
        rx_deg=normalize_deg(values["rx"]),
        ry_deg=normalize_deg(values["ry"]),
        rz_deg=normalize_deg(values["rz"]),
    )


def pose_tuple(pose: EndPose) -> tuple[float, float, float, float, float, float]:
    return (
        pose.x_mm,
        pose.y_mm,
        pose.z_mm,
        pose.rx_deg,
        pose.ry_deg,
        pose.rz_deg,
    )


def joint_tuple(
    values: Iterable[float],
) -> tuple[float, float, float, float, float, float]:
    items = tuple(float(value) for value in values)
    if len(items) != len(JOINT_FIELDS):
        raise ValueError(f"expected 6 joint values, got {len(items)}")
    return items  # type: ignore[return-value]


def move_to_pose(
    controller: PiperController,
    pose: EndPose,
    args: argparse.Namespace,
) -> tuple[float, float, float, float, float, float]:
    before_joints = read_fresh_joints_rad(
        controller,
        timeout_s=args.read_timeout_s,
        settle_s=0.0,
    )
    controller.validate_pose(pose)
    execute_pose_for_duration(
        controller,
        pose,
        speed=args.speed,
        mode=args.move_mode,
        duration_s=args.command_duration_s,
        period_s=args.command_period_s,
    )
    wait_until_pose_reached(
        controller,
        "auto_hand_eye_candidate",
        pose,
        args.motion_timeout_s,
        args.position_tolerance_mm,
        args.orientation_tolerance_deg,
    )
    after_joints = read_fresh_joints_rad(
        controller,
        timeout_s=args.read_timeout_s,
        settle_s=0.0,
    )
    assert_joint_step_is_safe(before_joints, after_joints, args)
    return after_joints


def wait_until_pose_reached(
    controller: PiperController,
    label: str,
    target: EndPose,
    timeout_s: float,
    position_tolerance_mm: float,
    orientation_tolerance_deg: float,
) -> EndPose:
    deadline = time.monotonic() + timeout_s
    last_pose: EndPose | None = None
    while timeout_s == 0 or time.monotonic() < deadline:
        pose = read_end_pose(controller)
        if pose is not None:
            last_pose = pose
            position_error, orientation_error = pose_errors(target, pose)
            if (
                position_error <= position_tolerance_mm
                and orientation_error <= orientation_tolerance_deg
            ):
                return pose
        if timeout_s == 0:
            break
        time.sleep(0.05)
    position_error, orientation_error = (
        pose_errors(target, last_pose)
        if last_pose is not None
        else (float("inf"), float("inf"))
    )
    raise RuntimeError(
        f"waypoint {label} not reached: target={format_end_pose(target)} "
        f"last={format_end_pose(last_pose) if last_pose is not None else 'unavailable'} "
        f"pos_err={position_error:.3f}mm ori_err={orientation_error:.3f}deg"
    )


def read_end_pose(controller: PiperController) -> EndPose | None:
    try:
        return feedback_to_end_pose(controller.get_end_pose())
    except ValueError:
        return None


def pose_errors(target: EndPose, actual: EndPose) -> tuple[float, float]:
    position_error = math.sqrt(
        (actual.x_mm - target.x_mm) ** 2
        + (actual.y_mm - target.y_mm) ** 2
        + (actual.z_mm - target.z_mm) ** 2
    )
    return position_error, rotation_angle_error_deg(target, actual)


def rotation_angle_error_deg(target: EndPose, actual: EndPose) -> float:
    target_rotation = rotation_matrix_xyz_deg(
        target.rx_deg,
        target.ry_deg,
        target.rz_deg,
    )
    actual_rotation = rotation_matrix_xyz_deg(
        actual.rx_deg,
        actual.ry_deg,
        actual.rz_deg,
    )
    relative_rotation = mat3_multiply(mat3_transpose(target_rotation), actual_rotation)
    cosine = (mat3_trace(relative_rotation) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def rotation_matrix_xyz_deg(
    rx_deg: float,
    ry_deg: float,
    rz_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rx, ry, rz = (math.radians(value) for value in (rx_deg, ry_deg, rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz),
        (cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz),
        (-sy, sx * cy, cx * cy),
    )


def mat3_transpose(
    matrix: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0])),
        (float(matrix[0][1]), float(matrix[1][1]), float(matrix[2][1])),
        (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2])),
    )


def mat3_multiply(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(sum(float(a[0][k]) * float(b[k][col]) for k in range(3)) for col in range(3)),
        tuple(sum(float(a[1][k]) * float(b[k][col]) for k in range(3)) for col in range(3)),
        tuple(sum(float(a[2][k]) * float(b[k][col]) for k in range(3)) for col in range(3)),
    )


def mat3_trace(matrix: Sequence[Sequence[float]]) -> float:
    return float(matrix[0][0]) + float(matrix[1][1]) + float(matrix[2][2])


def move_to_joints(
    controller: PiperController,
    target_joints_rad: Sequence[float],
    args: argparse.Namespace,
) -> tuple[float, float, float, float, float, float]:
    target = joint_tuple(target_joints_rad)
    controller.validate_joints(list(target))
    current = read_fresh_joints_rad(
        controller,
        timeout_s=args.read_timeout_s,
        settle_s=0.0,
    )
    for segment in joint_segments(current, target, args):
        execute_joints_for_duration(
            controller,
            segment,
            speed=args.speed,
            duration_s=args.command_duration_s,
            period_s=args.command_period_s,
        )
        reached = wait_until_joints_reached(
            controller,
            segment,
            timeout_s=args.motion_timeout_s,
            tolerance_rad=math.radians(args.joint_tolerance_deg),
            start_joints_rad=current,
            args=args,
        )
        assert_joint_step_is_safe(current, reached, args)
        current = reached
    return current


def execute_joints_for_duration(
    controller: PiperController,
    joints_rad: Sequence[float],
    speed: int,
    duration_s: float,
    period_s: float,
) -> int:
    deadline = time.monotonic() + duration_s
    send_count = 0
    while send_count == 0 or time.monotonic() < deadline:
        controller.move_joints(list(joints_rad), speed_percent=speed)
        send_count += 1
        if time.monotonic() >= deadline:
            break
        time.sleep(period_s)
    return send_count


def joint_segments(
    start_joints_rad: Sequence[float],
    target_joints_rad: Sequence[float],
    args: argparse.Namespace,
) -> tuple[tuple[float, float, float, float, float, float], ...]:
    start = joint_tuple(start_joints_rad)
    target = joint_tuple(target_joints_rad)
    max_step_rad = per_joint_max_delta_rad(args)
    segment_count = 1
    for start_value, target_value, max_delta in zip(start, target, max_step_rad):
        if max_delta <= 0.0:
            continue
        delta = abs(target_value - start_value)
        segment_count = max(segment_count, math.ceil(delta / max_delta))
    segments = []
    for index in range(1, segment_count + 1):
        ratio = index / segment_count
        segments.append(
            joint_tuple(
                start_value + (target_value - start_value) * ratio
                for start_value, target_value in zip(start, target)
            )
        )
    return tuple(segments)


def wait_until_joints_reached(
    controller: PiperController,
    target_joints_rad: Sequence[float],
    timeout_s: float,
    tolerance_rad: float,
    start_joints_rad: Sequence[float] | None = None,
    args: argparse.Namespace | None = None,
) -> tuple[float, float, float, float, float, float]:
    target = joint_tuple(target_joints_rad)
    start = joint_tuple(start_joints_rad) if start_joints_rad is not None else None
    deadline = time.monotonic() + timeout_s
    last_joints: tuple[float, float, float, float, float, float] | None = None
    while timeout_s == 0 or time.monotonic() < deadline:
        joints = read_joints_rad(controller)
        if joints is not None:
            last_joints = joints
            if start is not None and args is not None:
                assert_joint_step_is_safe(start, joints, args)
            if joints_are_close(joints, target, tolerance_rad):
                return joints
        if timeout_s == 0:
            break
        time.sleep(0.05)
    last = "unavailable" if last_joints is None else format_joints(last_joints)
    raise RuntimeError(
        f"joint target not reached: target={format_joints(target)} "
        f"last={last} tolerance={math.degrees(tolerance_rad):.3f}deg"
    )


def read_fresh_joints_rad(
    controller: PiperController,
    timeout_s: float = 2.0,
    settle_s: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")
    if settle_s < 0:
        raise ValueError("settle_s must be >= 0")
    if settle_s:
        time.sleep(settle_s)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        joints = read_joints_rad(controller)
        if joints is not None:
            return joints
        time.sleep(0.02)
    raise RuntimeError("failed to read fresh Piper joint feedback")


def read_joints_rad(
    controller: PiperController,
) -> tuple[float, float, float, float, float, float] | None:
    joint_feedback = controller.get_joint_state()
    joint_state = getattr(joint_feedback, "joint_state", None)
    if joint_state is None:
        return None
    values = [getattr(joint_state, f"joint_{index}", None) for index in range(1, 7)]
    if any(value is None for value in values):
        return None
    timestamp = float(getattr(joint_feedback, "time_stamp", 0.0))
    hz = float(getattr(joint_feedback, "Hz", 0.0))
    if timestamp <= 0.0 and hz <= 0.0:
        return None
    return joint_tuple(float(value) / RAD_TO_MILLI_DEG for value in values)


def assert_joint_step_is_safe(
    before: Sequence[float],
    after: Sequence[float],
    args: argparse.Namespace,
) -> None:
    max_delta = per_joint_max_delta_rad(args)
    deltas = [abs(float(a) - float(b)) for a, b in zip(before, after)]
    violations = [
        f"j{index + 1}={math.degrees(delta):.1f}deg>{math.degrees(limit):.1f}deg"
        for index, (delta, limit) in enumerate(zip(deltas, max_delta))
        if delta > limit
    ]
    if violations:
        raise JointJumpError(
            f"{', '.join(violations)} before={format_joints(before)} "
            f"after={format_joints(after)}"
        )


def per_joint_max_delta_rad(
    args: argparse.Namespace,
) -> tuple[float, float, float, float, float, float]:
    base = math.radians(args.max_joint_delta_deg)
    wrist = math.radians(args.max_wrist_joint_delta_deg)
    return (base, base, base, wrist, wrist, wrist)


def joints_are_close(
    actual: Sequence[float],
    target: Sequence[float],
    tolerance_rad: float,
) -> bool:
    return all(abs(float(a) - float(t)) <= tolerance_rad for a, t in zip(actual, target))


def handle_rejected_motion(
    args: argparse.Namespace,
    controller: PiperController,
    seed_pose: EndPose,
    seed_joints: tuple[float, float, float, float, float, float],
    last_good_pose: EndPose,
    last_good_joints: tuple[float, float, float, float, float, float],
    consecutive_failures: int,
    reason: str,
) -> None:
    if consecutive_failures % args.return_to_seed_after_failures == 0:
        return_pose = seed_pose
        return_joints = seed_joints
        label = "seed"
    else:
        return_pose = last_good_pose
        return_joints = last_good_joints
        label = "last-good"
    print(
        f"Rejected candidate ({reason}); returning to {label} "
        f"{args.motion_space} "
        f"after {consecutive_failures} consecutive failure(s).",
        flush=True,
    )
    if args.motion_space == "joint":
        move_to_joints(controller, return_joints, args)
    else:
        move_to_pose(controller, return_pose, args)


def query_vision(
    args: argparse.Namespace,
    sample_id: str,
    *,
    capture: bool,
) -> AutoHandEyeDetection:
    request = AutoHandEyeRequest(
        action="capture" if capture else "status",
        sample_id=sample_id,
        min_margin_px=args.min_margin_px,
        max_reprojection_error_px=args.max_reprojection_error_px,
        stable_frames=args.stable_frames,
        save_image=not args.no_save_images,
    )
    return request_auto_hand_eye_detection(
        args.vision_host,
        args.vision_port,
        request,
        timeout_s=args.vision_timeout_s,
    )


def format_detection(detection: AutoHandEyeDetection) -> str:
    if not detection.detected:
        return f"not detected: {detection.reason}"
    fields = [
        f"qualified={detection.qualified}",
        f"reason={detection.reason}",
        f"reproj={format_optional(detection.reprojection_error_px, '.3f')}px",
        f"margin={format_optional(detection.margin_px, '.1f')}px",
        f"center=({format_optional(detection.center_u, '.1f')},"
        f"{format_optional(detection.center_v, '.1f')})",
        f"stable={detection.stable_count}",
    ]
    return " ".join(fields)


def format_optional(value: float | None, spec: str) -> str:
    if value is None:
        return "NA"
    return format(value, spec)


def format_joints(joints_rad: Sequence[float]) -> str:
    values = [math.degrees(float(value)) for value in joints_rad]
    return " ".join(
        f"j{index}={value:.3f}deg" for index, value in enumerate(values, start=1)
    )


def diverse_enough(
    pose: EndPose,
    samples: Iterable[HandEyeSample],
    *,
    min_translation_delta_mm: float,
    min_rotation_delta_deg: float,
) -> bool:
    sample_list = list(samples)
    if not sample_list:
        return True
    for sample in sample_list:
        previous = sample.robot_base_to_gripper
        translation_delta = math.sqrt(
            (pose.x_mm - previous.x_mm) ** 2
            + (pose.y_mm - previous.y_mm) ** 2
            + (pose.z_mm - previous.z_mm) ** 2
        )
        rotation_delta = math.sqrt(
            angle_delta_deg(pose.rx_deg, previous.rx_deg) ** 2
            + angle_delta_deg(pose.ry_deg, previous.ry_deg) ** 2
            + angle_delta_deg(pose.rz_deg, previous.rz_deg) ** 2
        )
        if (
            translation_delta < min_translation_delta_mm
            and rotation_delta < min_rotation_delta_deg
        ):
            return False
    return True


def angle_delta_deg(a: float, b: float) -> float:
    return abs(normalize_deg(a - b))


if __name__ == "__main__":
    try:
        main()
    except (SafetyError, RuntimeError, TimeoutError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None

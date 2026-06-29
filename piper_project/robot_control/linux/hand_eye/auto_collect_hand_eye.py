"""Automatic small-random-motion hand-eye sample collector."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import wait_until_pose_reached
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.hand_eye.calibration import (
    HandEyeSample,
    format_end_pose,
    load_samples,
    make_sample,
    make_samples_document,
    write_json,
)
from robot_control.linux.hand_eye.move_board_pose import execute_pose_for_duration
from robot_control.linux.tcp_calibration.calibration import read_fresh_end_pose
from robot_control.shared.geometry import EndPose
from robot_control.shared.hand_eye_auto_protocol import (
    AutoHandEyeRequest,
    AutoHandEyeDetection,
    request_auto_hand_eye_detection,
)
from robot_control.shared.hand_eye_protocol import HandEyeSampleCommand


DEFAULT_OUTPUT = Path(__file__).with_name("auto_hand_eye_samples.json")
POSE_FIELDS = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(frozen=True)
class PoseStepLimits:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float


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
class CandidateResult:
    pose: EndPose
    selected_fields: tuple[str, ...]
    attempt: int


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
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0)
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
    parser.add_argument("--range-x-mm", type=float, default=80.0)
    parser.add_argument("--range-y-mm", type=float, default=80.0)
    parser.add_argument("--range-z-mm", type=float, default=50.0)
    parser.add_argument("--range-rx-deg", type=float, default=25.0)
    parser.add_argument("--range-ry-deg", type=float, default=25.0)
    parser.add_argument("--range-rz-deg", type=float, default=35.0)
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

        bounds = pose_bounds_from_seed(seed_pose, args)
        step_limits = PoseStepLimits(
            x_mm=args.step_x_mm,
            y_mm=args.step_y_mm,
            z_mm=args.step_z_mm,
            rx_deg=args.step_rx_deg,
            ry_deg=args.step_ry_deg,
            rz_deg=args.step_rz_deg,
        )
        collect_samples(
            args,
            controller,
            rng,
            seed_pose,
            bounds,
            step_limits,
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
        "range_x_mm",
        "range_y_mm",
        "range_z_mm",
        "range_rx_deg",
        "range_ry_deg",
        "range_rz_deg",
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
    bounds: PoseBounds,
    step_limits: PoseStepLimits,
    samples: list[HandEyeSample],
) -> None:
    last_good_pose = seed_pose
    consecutive_failures = 0
    attempts = 0
    while len(samples) < args.samples and attempts < args.max_attempts:
        attempts += 1
        shrink = 0.5 if consecutive_failures >= args.shrink_after_failures else 1.0
        try:
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

        if not diverse_enough(
            candidate.pose,
            samples,
            min_translation_delta_mm=args.min_translation_delta_mm,
            min_rotation_delta_deg=args.min_rotation_delta_deg,
        ):
            print(
                f"[{attempts}] skipped similar candidate fields={candidate.selected_fields} "
                f"pose=({format_end_pose(candidate.pose)})",
                flush=True,
            )
            continue

        print(
            f"[{attempts}] moving candidate fields={candidate.selected_fields}: "
            f"{format_end_pose(candidate.pose)}",
            flush=True,
        )
        try:
            move_to_pose(controller, candidate.pose, args)
            actual_pose = read_fresh_end_pose(
                controller,
                timeout_s=args.read_timeout_s,
                settle_s=args.settle_s,
            )
            status = query_vision(args, f"candidate-{attempts:03d}", capture=False)
            print(f"[{attempts}] vision status: {format_detection(status)}", flush=True)
            if not status.qualified:
                consecutive_failures += 1
                handle_rejected_pose(
                    args,
                    controller,
                    seed_pose,
                    last_good_pose,
                    consecutive_failures,
                    status.reason or "unqualified",
                )
                continue

            sample_id = f"auto-{len(samples) + 1:03d}"
            capture = query_vision(args, sample_id, capture=True)
            if not capture.qualified or capture.target_to_camera is None:
                consecutive_failures += 1
                handle_rejected_pose(
                    args,
                    controller,
                    seed_pose,
                    last_good_pose,
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
            consecutive_failures = 0
            print(
                f"Saved sample {sample.index}/{args.samples}: "
                f"pose=({format_end_pose(actual_pose)}) "
                f"vision=({format_detection(capture)})",
                flush=True,
            )
        except (SafetyError, RuntimeError, TimeoutError, ValueError) as exc:
            consecutive_failures += 1
            print(f"[{attempts}] rejected: {exc}", flush=True)
            handle_rejected_pose(
                args,
                controller,
                seed_pose,
                last_good_pose,
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
    del attempt
    fields = tuple(rng.sample(POSE_FIELDS, rng.randint(min_dof, max_dof)))
    values = pose_to_values(base_pose)
    for field in fields:
        values[field] += rng.uniform(-step_for_field(step_limits, field), step_for_field(step_limits, field)) * shrink
    pose = values_to_pose(values)
    pose = clamp_pose(pose, bounds)
    if same_pose(pose, base_pose):
        raise ValueError("candidate was clamped back to the base pose")
    return CandidateResult(pose=pose, selected_fields=fields, attempt=attempt)


def step_for_field(step_limits: PoseStepLimits, field: str) -> float:
    return {
        "x": step_limits.x_mm,
        "y": step_limits.y_mm,
        "z": step_limits.z_mm,
        "rx": step_limits.rx_deg,
        "ry": step_limits.ry_deg,
        "rz": step_limits.rz_deg,
    }[field]


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


def clamp_pose(pose: EndPose, bounds: PoseBounds) -> EndPose:
    return EndPose(
        x_mm=clamp(pose.x_mm, bounds.min_x_mm, bounds.max_x_mm),
        y_mm=clamp(pose.y_mm, bounds.min_y_mm, bounds.max_y_mm),
        z_mm=clamp(pose.z_mm, bounds.min_z_mm, bounds.max_z_mm),
        rx_deg=clamp(pose.rx_deg, bounds.min_rx_deg, bounds.max_rx_deg),
        ry_deg=clamp(pose.ry_deg, bounds.min_ry_deg, bounds.max_ry_deg),
        rz_deg=normalize_deg(clamp(pose.rz_deg, bounds.min_rz_deg, bounds.max_rz_deg)),
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


def move_to_pose(
    controller: PiperController,
    pose: EndPose,
    args: argparse.Namespace,
) -> None:
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


def handle_rejected_pose(
    args: argparse.Namespace,
    controller: PiperController,
    seed_pose: EndPose,
    last_good_pose: EndPose,
    consecutive_failures: int,
    reason: str,
) -> None:
    if consecutive_failures % args.return_to_seed_after_failures == 0:
        return_pose = seed_pose
        label = "seed"
    else:
        return_pose = last_good_pose
        label = "last-good"
    print(
        f"Rejected candidate ({reason}); returning to {label} pose "
        f"after {consecutive_failures} consecutive failure(s).",
        flush=True,
    )
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

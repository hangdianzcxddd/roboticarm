"""YOLO-driven grasp server using the fixed-pose continuous sequence style.

This server receives ``CameraPointCommand`` messages from the Windows YOLO
client, maps the camera XYZ through the solved eye-to-hand transform, and runs a
direct J6 pose sequence modeled after ``run_pose_continuous_catch_sequence``:

``out`` task:
ready/open -> target above -> target grasp/close -> lift -> fixed place/open
-> return ready.

``in`` task:
ready/open -> selected target above -> selected target grasp/close -> lift
-> return ready/open.

Unlike ``catch_server.py``, this script intentionally commands fixed J6 poses
directly and keeps one fixed wrist orientation for every waypoint. A YOLO bbox
by itself is not enough for 3D grasping; the Windows side should send the
bbox-selected point after RealSense depth deprojection as ``x_m/y_m/z_m``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Literal

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import (
    execute_pose_for_duration,
    feedback_to_pose,
    format_pose,
    pose_delta,
    wait_until_pose_reached,
)
from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
    map_camera_point_to_base_xyz_m,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    CameraPointCommand,
    CameraPointTask,
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.shared.simulated_mapping import pose_to_dict


DEFAULT_HAND_EYE_RESULT = (
    Path(__file__).resolve().parents[1] / "hand_eye" / "hand_eye_result.json"
)
Action = Literal["open", "close", None]


@dataclass(frozen=True)
class SequenceStep:
    label: str
    pose: EndPose
    mode: Literal["P", "L"]
    gripper_after: Action = None
    delay_before_s: float = 0.0


@dataclass(frozen=True)
class YoloCatchTarget:
    camera_command: CameraPointCommand
    base_xyz_m: tuple[float, float, float]
    pick_x_mm: float
    pick_y_mm: float
    grasp_z_mm: float
    raw_grasp_z_mm: float | None = None
    grasp_z_source: str = "fixed_grasp_z"


@dataclass(frozen=True)
class YoloCatchPlan:
    task: CameraPointTask
    target: YoloCatchTarget
    steps: tuple[SequenceStep, ...]


@dataclass(frozen=True)
class DynamicTiming:
    t_arrive_s: float
    t_robot_s: float
    action_time_s: float
    t_wait_s: float
    dynamic_preposition: bool = False
    final_action_time_s: float | None = None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class GuardedDescentResult:
    send_count: int
    start_z_mm: float
    target_z_mm: float
    aborted: bool = False
    reason: str | None = None
    last_feedback_z_mm: float | None = None
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None


@dataclass(frozen=True)
class SafeGripperResult:
    action: Action
    command_mm: float | None = None
    command_effort_nm: float | None = None
    max_effort_nm: float | None = None
    triggered: bool = False
    stopped: bool = False
    stop_action: str | None = None
    release_command_mm: float | None = None
    last_effort_nm: float | None = None
    last_position_mm: float | None = None
    poll_count: int = 0


@dataclass(frozen=True)
class DynamicExecutionResult:
    send_counts: dict[str, int]
    protected_abort: bool = False
    protected_abort_reason: str | None = None
    protected_abort_return: str | None = None
    protected_abort_return_label: str | None = None
    preposition_z_mm: float | None = None
    preposition_source: str | None = None
    preposition_elapsed_s: float | None = None
    final_action_due_s: float | None = None
    final_wait_s: float | None = None
    final_action_late_s: float | None = None
    guarded_descent: GuardedDescentResult | None = None
    safe_gripper_results: tuple[SafeGripperResult, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive YOLO camera XYZ over TCP and run a fixed-orientation "
            "pose-continuous out or in transfer sequence."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=40)
    parser.add_argument(
        "--startup-speed",
        type=int,
        default=10,
        help=(
            "speed percent used only for the execute-server startup ready pose; "
            "--speed remains the normal catch/transfer speed"
        ),
    )
    parser.add_argument("--transfer-mode", choices=("P", "L"), default="P")
    parser.add_argument("--vertical-mode", choices=("P", "L"), default="L")
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--no-wait-waypoints", action="store_true")
    parser.add_argument("--print-feedback", action="store_true")
    parser.add_argument("--debug-plan", action="store_true")
    parser.add_argument(
        "--t-robot-s",
        type=float,
        default=1.7,
        help=(
            "dynamic mode measured robot reaction time from grasp command to "
            "gripper closed at the pick point"
        ),
    )
    parser.add_argument(
        "--dynamic-preposition",
        action="store_true",
        help=(
            "in dynamic mode, move to ready/open and a target preposition pose "
            "immediately, then wait only for the final descent/close window"
        ),
    )
    parser.add_argument(
        "--dynamic-preposition-clearance-mm",
        type=float,
        default=20.0,
        help=(
            "dynamic preposition clearance above detected_z_mm; when detected Z "
            "is unavailable, falls back to grasp_z plus the guarded descent zone"
        ),
    )
    parser.add_argument(
        "--final-action-time-s",
        type=float,
        default=0.55,
        help=(
            "dynamic-preposition timing budget for the final descent and gripper "
            "close; this replaces --t-robot-s for the final wait calculation"
        ),
    )
    parser.add_argument(
        "--guarded-descent",
        "--descent-protection",
        action=argparse.BooleanOptionalAction,
        dest="guarded_descent",
        default=True,
        help=(
            "check J6 Z feedback during the final dynamic descent and abort if "
            "the arm stops following the commanded Z; enabled by default, use "
            "--no-guarded-descent or --no-descent-protection to disable"
        ),
    )
    parser.add_argument("--guarded-descent-zone-mm", type=float, default=30.0)
    parser.add_argument("--guarded-descent-step-mm", type=float, default=3.0)
    parser.add_argument("--guarded-descent-speed", type=int, default=20)
    parser.add_argument(
        "--guarded-follow-error-mm",
        type=float,
        default=4.0,
        help=(
            "maximum allowed Z feedback lag from each guarded descent command "
            "after the step wait; gripper close effort is not used for descent "
            "collision protection"
        ),
    )
    parser.add_argument("--guarded-stall-s", type=float, default=0.2)
    parser.add_argument(
        "--guarded-start-tolerance-mm",
        type=float,
        default=2.0,
        help=(
            "required Z feedback tolerance at the guarded descent start pose; "
            "prevents starting descent while the arm is still above guard start"
        ),
    )
    parser.add_argument(
        "--guarded-start-timeout-s",
        type=float,
        default=1.0,
        help="maximum time to wait for feedback to reach guarded descent start Z",
    )
    parser.add_argument(
        "--guarded-progress-min-mm",
        type=float,
        default=1.0,
        help=(
            "minimum observed downward J6 feedback movement required after each "
            "guarded descent command; 0 requires any measurable downward movement"
        ),
    )
    parser.add_argument(
        "--guarded-step-timeout-s",
        type=float,
        default=0.25,
        help=(
            "maximum time to wait for guarded descent feedback progress after "
            "each step; defaults to --guarded-stall-s when 0"
        ),
    )
    parser.add_argument(
        "--collision-protection-level",
        type=int,
        help=(
            "optional Piper SDK collision protection level applied to all six "
            "joints at startup; valid range is 0-8"
        ),
    )
    parser.add_argument(
        "--guarded-abort-return",
        choices=("pick_above", "preposition", "none"),
        default="pick_above",
        help=(
            "pose to return to after guarded descent abort; pick_above returns "
            "to transfer_z, preposition returns to the dynamic preposition pose"
        ),
    )
    parser.add_argument(
        "--no-start-ready",
        action="store_true",
        help=(
            "do not move to the run_pose_continuous_catch_sequence ready pose "
            "when the execute server starts"
        ),
    )

    parser.add_argument(
        "--hand-eye-result",
        type=Path,
        default=DEFAULT_HAND_EYE_RESULT,
        help="eye_to_hand result whose transform is camera -> base",
    )
    parser.add_argument("--ready-x-mm", type=float, default=343.108)
    parser.add_argument("--ready-y-mm", type=float, default=0.0)
    parser.add_argument("--transfer-z-mm", type=float, default=253.252)
    parser.add_argument(
        "--grasp-z-mm",
        "--grasp-z",
        dest="grasp_z_mm",
        type=float,
        default=133.252,
        help="fixed J6 grasp Z, matching run_pose_continuous_catch_sequence style",
    )
    parser.add_argument(
        "--use-detected-z",
        action="store_true",
        help=(
            "use hand-eye mapped base Z for J6 grasp Z instead of --grasp-z-mm. "
            "Default keeps a fixed grasp plane."
        ),
    )
    parser.add_argument(
        "--detected-z-offset-mm",
        type=float,
        default=0.0,
        help="extra J6 Z offset when --use-detected-z is enabled",
    )
    parser.add_argument("--rx-deg", type=float, default=180.0)
    parser.add_argument("--ry-deg", type=float, default=14.795)
    parser.add_argument("--rz-deg", type=float, default=180.0)
    parser.add_argument(
        "--pick-delay-s",
        type=float,
        default=0.0,
        help="pause before descending from target-above to target-grasp",
    )
    parser.add_argument("--place-x-mm", type=float, default=130.0)
    parser.add_argument("--place-y-mm", type=float, default=200.0)
    parser.add_argument(
        "--place-pick-x-mm",
        type=float,
        default=-130.0,
        help="X used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-y-mm",
        type=float,
        default=200.0,
        help="Y used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-z-mm",
        type=float,
        help="Z used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-delay-s",
        type=float,
        default=3.0,
        help="pause at ready before moving to the fixed place-pick point",
    )
    parser.add_argument(
        "--grasp-only",
        action="store_true",
        help="legacy mode: pick and lift the YOLO target without fixed place/pick-back",
    )
    parser.add_argument(
        "--no-return-ready",
        action="store_true",
        help=(
            "for task=out, stop at the task end point instead of returning to "
            "the ready pose"
        ),
    )

    parser.add_argument("--min-target-x-mm", type=float, default=80.0)
    parser.add_argument("--max-target-x-mm", type=float)
    parser.add_argument("--min-target-y-mm", type=float, default=-300.0)
    parser.add_argument("--max-target-y-mm", type=float, default=300.0)
    parser.add_argument("--min-grasp-z-mm", type=float)
    parser.add_argument("--max-grasp-z-mm", type=float)

    parser.add_argument("--open-command-mm", type=float, default=0.0)
    parser.add_argument("--grasp-command-mm", type=float, default=-100.0)
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument(
        "--grasp-effort-nm",
        type=float,
        help="gripper effort used only for close commands; defaults to --effort-nm",
    )
    parser.add_argument(
        "--max-gripper-effort-nm",
        type=float,
        default=0.8,
        help="feedback effort threshold for --safe-gripper-stop-on-effort",
    )
    parser.add_argument(
        "--safe-gripper-stop-on-effort",
        action="store_true",
        help=(
            "after close, poll gripper feedback and stop/release when "
            "grippers_effort exceeds --max-gripper-effort-nm"
        ),
    )
    parser.add_argument(
        "--safe-gripper-monitor-s",
        type=float,
        default=0.35,
        help="seconds to monitor gripper feedback after a close command",
    )
    parser.add_argument(
        "--safe-gripper-poll-s",
        type=float,
        default=0.02,
        help="poll period for --safe-gripper-stop-on-effort",
    )
    parser.add_argument(
        "--safe-gripper-release-mm",
        type=float,
        help=(
            "optional gripper command after an effort stop; defaults to disabling "
            "the gripper at the current open command"
        ),
    )
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.speed, int) or not 0 <= args.speed <= 100:
        raise ValueError("--speed must be an integer in [0, 100]")
    if not isinstance(args.startup_speed, int) or not 0 <= args.startup_speed <= 100:
        raise ValueError("--startup-speed must be an integer in [0, 100]")
    for name in (
        "command_duration_s",
        "motion_timeout_s",
        "position_tolerance_mm",
        "orientation_tolerance_deg",
        "t_robot_s",
        "final_action_time_s",
        "pick_delay_s",
        "place_pick_delay_s",
        "safe_gripper_monitor_s",
        "gripper_settle_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(args.command_period_s) or args.command_period_s <= 0:
        raise ValueError("--command-period-s must be finite and > 0")
    if not math.isfinite(args.safe_gripper_poll_s) or args.safe_gripper_poll_s <= 0:
        raise ValueError("--safe-gripper-poll-s must be finite and > 0")
    if (
        not isinstance(args.guarded_descent_speed, int)
        or not 0 <= args.guarded_descent_speed <= 100
    ):
        raise ValueError("--guarded-descent-speed must be an integer in [0, 100]")
    for name in (
        "dynamic_preposition_clearance_mm",
        "guarded_descent_zone_mm",
        "guarded_descent_step_mm",
        "guarded_follow_error_mm",
        "guarded_stall_s",
        "guarded_start_tolerance_mm",
        "guarded_start_timeout_s",
        "guarded_progress_min_mm",
        "guarded_step_timeout_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if args.dynamic_preposition_clearance_mm <= 0.0:
        raise ValueError("--dynamic-preposition-clearance-mm must be > 0")
    if args.guarded_descent_zone_mm <= 0.0:
        raise ValueError("--guarded-descent-zone-mm must be > 0")
    if args.guarded_descent_step_mm <= 0.0:
        raise ValueError("--guarded-descent-step-mm must be > 0")
    if args.guarded_follow_error_mm <= 0.0:
        raise ValueError("--guarded-follow-error-mm must be > 0")
    if args.guarded_start_tolerance_mm <= 0.0:
        raise ValueError("--guarded-start-tolerance-mm must be > 0")
    if args.collision_protection_level is not None and (
        not isinstance(args.collision_protection_level, int)
        or not 0 <= args.collision_protection_level <= 8
    ):
        raise ValueError("--collision-protection-level must be an integer in [0, 8]")
    if args.grasp_z_mm > args.transfer_z_mm:
        raise ValueError("--grasp-z-mm should not be above --transfer-z-mm")
    if args.place_pick_z_mm is not None and args.place_pick_z_mm > args.transfer_z_mm:
        raise ValueError("--place-pick-z-mm should not be above --transfer-z-mm")
    for option_name, value in (
        ("--effort-nm", args.effort_nm),
        ("--grasp-effort-nm", args.grasp_effort_nm),
        ("--max-gripper-effort-nm", args.max_gripper_effort_nm),
    ):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 5):
            raise ValueError(f"{option_name} must be finite and in [0, 5]")

    for name in (
        "ready_x_mm",
        "ready_y_mm",
        "transfer_z_mm",
        "grasp_z_mm",
        "detected_z_offset_mm",
        "place_x_mm",
        "place_y_mm",
        "place_pick_x_mm",
        "place_pick_y_mm",
        "place_pick_z_mm",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "min_target_x_mm",
        "max_target_x_mm",
        "min_target_y_mm",
        "max_target_y_mm",
        "min_grasp_z_mm",
        "max_grasp_z_mm",
        "open_command_mm",
        "grasp_command_mm",
        "safe_gripper_release_mm",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    for lower_name, upper_name in (
        ("min_target_x_mm", "max_target_x_mm"),
        ("min_target_y_mm", "max_target_y_mm"),
        ("min_grasp_z_mm", "max_grasp_z_mm"),
    ):
        lower = getattr(args, lower_name)
        upper = getattr(args, upper_name)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"--{lower_name.replace('_', '-')} must be <= "
                f"--{upper_name.replace('_', '-')}"
            )


def fixed_pose(args: argparse.Namespace, x_mm: float, y_mm: float, z_mm: float) -> EndPose:
    return EndPose(
        float(x_mm),
        float(y_mm),
        float(z_mm),
        float(args.rx_deg),
        float(args.ry_deg),
        float(args.rz_deg),
    )


def ready_pose(args: argparse.Namespace) -> EndPose:
    return fixed_pose(args, args.ready_x_mm, args.ready_y_mm, args.transfer_z_mm)


def place_pick_pose(args: argparse.Namespace) -> EndPose:
    x_mm = args.place_x_mm if args.place_pick_x_mm is None else args.place_pick_x_mm
    y_mm = args.place_y_mm if args.place_pick_y_mm is None else args.place_pick_y_mm
    z_mm = args.grasp_z_mm if args.place_pick_z_mm is None else args.place_pick_z_mm
    return fixed_pose(args, x_mm, y_mm, z_mm)


def target_from_command(
    command: CameraPointCommand,
    camera_to_base: Any,
    args: argparse.Namespace,
    *,
    validate: bool = True,
) -> YoloCatchTarget:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        command.x_m,
        command.y_m,
        command.z_m,
        camera_to_base,
    )
    if command.grasp_mode == "dynamic":
        if command.pick_x_mm is None or command.pick_y_mm is None:
            raise ValueError("dynamic grasp command requires pick_x_mm and pick_y_mm")
        pick_x_mm = float(command.pick_x_mm)
        pick_y_mm = float(command.pick_y_mm)
    else:
        pick_x_mm = base_xyz_m[0] * 1000.0
        pick_y_mm = base_xyz_m[1] * 1000.0
    raw_grasp_z_mm: float | None = None
    grasp_z_source = "fixed_grasp_z"
    if args.use_detected_z:
        detected_z_mm = base_xyz_m[2] * 1000.0
        grasp_z_source = "mapped_command_z_plus_offset"
        if (
            command.grasp_mode == "dynamic"
            and command.detected_z_mm is not None
            and math.isfinite(float(command.detected_z_mm))
        ):
            detected_z_mm = float(command.detected_z_mm)
            grasp_z_source = "dynamic_detected_z_plus_offset"
        raw_grasp_z_mm = detected_z_mm + args.detected_z_offset_mm
        grasp_z_mm = max(raw_grasp_z_mm, float(args.grasp_z_mm))
        if grasp_z_mm != raw_grasp_z_mm:
            grasp_z_source = f"{grasp_z_source}_clamped_to_fixed_grasp_z"
    else:
        grasp_z_mm = args.grasp_z_mm
    target = YoloCatchTarget(
        camera_command=command,
        base_xyz_m=base_xyz_m,
        pick_x_mm=float(pick_x_mm),
        pick_y_mm=float(pick_y_mm),
        grasp_z_mm=float(grasp_z_mm),
        raw_grasp_z_mm=raw_grasp_z_mm,
        grasp_z_source=grasp_z_source,
    )
    if validate:
        validate_target(target, args)
    return target


def validate_target(target: YoloCatchTarget, args: argparse.Namespace) -> None:
    checks = (
        ("target-x", target.pick_x_mm, args.min_target_x_mm, args.max_target_x_mm),
        ("target-y", target.pick_y_mm, args.min_target_y_mm, args.max_target_y_mm),
        ("grasp-z", target.grasp_z_mm, args.min_grasp_z_mm, args.max_grasp_z_mm),
    )
    for label, value, lower, upper in checks:
        if lower is not None and value < lower:
            raise SafetyError(f"{label} {value:.3f} mm is below limit {lower:.3f} mm")
        if upper is not None and value > upper:
            raise SafetyError(f"{label} {value:.3f} mm exceeds limit {upper:.3f} mm")
    if target.grasp_z_mm > args.transfer_z_mm:
        raise SafetyError(
            f"grasp-z {target.grasp_z_mm:.3f} mm is above transfer plane "
            f"{args.transfer_z_mm:.3f} mm"
        )


def build_out_steps(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[SequenceStep, ...]:
    ready = ready_pose(args)
    pick_above = fixed_pose(args, target.pick_x_mm, target.pick_y_mm, args.transfer_z_mm)
    pick_grasp = fixed_pose(args, target.pick_x_mm, target.pick_y_mm, target.grasp_z_mm)
    place_above = fixed_pose(args, args.place_x_mm, args.place_y_mm, args.transfer_z_mm)

    steps: list[SequenceStep] = [
        SequenceStep("ready_open", ready, args.transfer_mode, "open"),
        SequenceStep("pick_above", pick_above, args.transfer_mode),
        SequenceStep(
            "pick_grasp",
            pick_grasp,
            args.vertical_mode,
            "close",
            args.pick_delay_s,
        ),
        SequenceStep("pick_lift", pick_above, args.vertical_mode),
    ]
    if args.grasp_only:
        if not args.no_return_ready:
            steps.append(SequenceStep("return_ready", ready, args.transfer_mode))
        return tuple(steps)

    steps.extend(
        [
            SequenceStep("place_above", place_above, args.transfer_mode),
            SequenceStep("place_release", place_above, args.transfer_mode, "open"),
        ]
    )
    if not args.no_return_ready:
        steps.append(SequenceStep("return_ready", ready, args.transfer_mode))
    return tuple(steps)


def build_in_steps(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[SequenceStep, ...]:
    if args.grasp_only:
        raise ValueError("--grasp-only is only supported for task 'out'")
    if args.no_return_ready:
        raise ValueError("--no-return-ready is incompatible with task 'in'")

    ready = ready_pose(args)
    pick_above = fixed_pose(args, target.pick_x_mm, target.pick_y_mm, args.transfer_z_mm)
    pick_grasp = fixed_pose(args, target.pick_x_mm, target.pick_y_mm, target.grasp_z_mm)
    return (
        SequenceStep("ready_open", ready, args.transfer_mode, "open"),
        SequenceStep(
            "return_pick_above",
            pick_above,
            args.transfer_mode,
            None,
            args.place_pick_delay_s,
        ),
        SequenceStep("return_pick_grasp", pick_grasp, args.vertical_mode, "close"),
        SequenceStep("return_pick_lift", pick_above, args.vertical_mode),
        SequenceStep("return_ready_above", ready, args.transfer_mode),
        SequenceStep("final_release", ready, args.transfer_mode, "open"),
    )


def build_steps(args: argparse.Namespace, target: YoloCatchTarget) -> tuple[SequenceStep, ...]:
    if target.camera_command.task == "out":
        return build_out_steps(args, target)
    if target.camera_command.task == "in":
        return build_in_steps(args, target)
    raise ValueError("camera point task must be 'out' or 'in'")


def validate_steps(controller: PiperController, steps: tuple[SequenceStep, ...]) -> None:
    for step in steps:
        try:
            controller.validate_pose(step.pose)
        except SafetyError as exc:
            raise SafetyError(f"{step.label}: {exc}") from exc


def dynamic_timing_from_command(
    command: CameraPointCommand,
    args: argparse.Namespace,
) -> DynamicTiming | None:
    if command.grasp_mode != "dynamic":
        return None
    if command.t_arrive_s is None:
        raise ValueError("dynamic grasp command requires t_arrive_s")
    t_arrive_s = float(command.t_arrive_s)
    t_robot_s = float(args.t_robot_s)
    if not math.isfinite(t_arrive_s) or t_arrive_s < 0.0:
        raise ValueError("dynamic t_arrive_s must be finite and >= 0")
    if not math.isfinite(t_robot_s) or t_robot_s < 0.0:
        raise ValueError("--t-robot-s must be finite and >= 0")
    if args.dynamic_preposition:
        action_time_s = float(args.final_action_time_s)
        action_time_label = "final_action_time"
    else:
        action_time_s = t_robot_s
        action_time_label = "t_robot"
    if not math.isfinite(action_time_s) or action_time_s < 0.0:
        raise ValueError(f"--{action_time_label.replace('_', '-')} must be finite and >= 0")
    t_wait_s = t_arrive_s - action_time_s
    if t_wait_s < 0.0:
        return DynamicTiming(
            t_arrive_s=t_arrive_s,
            t_robot_s=t_robot_s,
            action_time_s=action_time_s,
            t_wait_s=t_wait_s,
            dynamic_preposition=bool(args.dynamic_preposition),
            final_action_time_s=(
                float(args.final_action_time_s) if args.dynamic_preposition else None
            ),
            skipped=True,
            skip_reason=(
                "dynamic target is too close to the pick line: "
                f"t_arrive={t_arrive_s:.3f}s < "
                f"{action_time_label}={action_time_s:.3f}s"
            ),
        )
    return DynamicTiming(
        t_arrive_s=t_arrive_s,
        t_robot_s=t_robot_s,
        action_time_s=action_time_s,
        t_wait_s=t_wait_s,
        dynamic_preposition=bool(args.dynamic_preposition),
        final_action_time_s=(
            float(args.final_action_time_s) if args.dynamic_preposition else None
        ),
    )


def wait_for_dynamic_timing(timing: DynamicTiming | None) -> None:
    if timing is None:
        return
    if timing.skipped:
        print(f"Dynamic timing: skipped: {timing.skip_reason}", flush=True)
        return
    if timing.t_wait_s <= 0.0:
        print(
            "Dynamic timing: start immediately "
            f"(t_arrive={timing.t_arrive_s:.3f}s, "
            f"t_robot={timing.t_robot_s:.3f}s)",
            flush=True,
        )
        return
    print(
        "Dynamic timing: waiting "
        f"{timing.t_wait_s:.3f}s before grasp "
        f"(t_arrive={timing.t_arrive_s:.3f}s, "
        f"t_robot={timing.t_robot_s:.3f}s)",
        flush=True,
    )
    time.sleep(timing.t_wait_s)


def dynamic_preposition_pose(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[EndPose, str]:
    command = target.camera_command
    detected_z_mm = command.detected_z_mm
    if detected_z_mm is not None and math.isfinite(float(detected_z_mm)):
        requested_z_mm = float(detected_z_mm) + float(args.dynamic_preposition_clearance_mm)
        source = "detected_z_plus_clearance"
    else:
        requested_z_mm = target.grasp_z_mm + float(args.guarded_descent_zone_mm)
        source = "grasp_z_plus_guarded_zone"

    min_descent_z_mm = min(
        float(args.transfer_z_mm),
        target.grasp_z_mm + float(args.guarded_descent_step_mm),
    )
    z_mm = min(float(args.transfer_z_mm), max(requested_z_mm, min_descent_z_mm))
    if z_mm != requested_z_mm:
        source = f"{source}_clamped"
    return fixed_pose(args, target.pick_x_mm, target.pick_y_mm, z_mm), source


def guarded_start_pose_for_grasp(
    args: argparse.Namespace,
    grasp_pose: EndPose,
) -> EndPose:
    z_mm = min(
        float(args.transfer_z_mm),
        grasp_pose.z_mm + float(args.guarded_descent_zone_mm),
    )
    return replace(grasp_pose, z_mm=z_mm)


def should_guard_step(step: SequenceStep, args: argparse.Namespace) -> bool:
    return bool(
        args.guarded_descent
        and step.gripper_after == "close"
        and step.pose.z_mm < args.transfer_z_mm
    )


def guarded_abort_return_step(
    args: argparse.Namespace,
    steps: tuple[SequenceStep, ...],
    step_index: int,
    fallback_pose: EndPose,
) -> SequenceStep | None:
    if args.guarded_abort_return == "none":
        return None
    if args.guarded_abort_return == "preposition":
        return SequenceStep(
            "protected_abort_return_preposition",
            fallback_pose,
            args.vertical_mode,
        )
    for prior_step in reversed(steps[:step_index]):
        same_xy = (
            abs(prior_step.pose.x_mm - steps[step_index].pose.x_mm) <= 1e-6
            and abs(prior_step.pose.y_mm - steps[step_index].pose.y_mm) <= 1e-6
        )
        above_target = prior_step.pose.z_mm > steps[step_index].pose.z_mm
        if same_xy and above_target:
            return SequenceStep(
                f"protected_abort_return_{prior_step.label}",
                prior_step.pose,
                args.vertical_mode,
            )
    return SequenceStep(
        "protected_abort_return_pick_above",
        replace(steps[step_index].pose, z_mm=args.transfer_z_mm),
        args.vertical_mode,
    )


def dynamic_protected_abort_return_step(
    args: argparse.Namespace,
    *,
    pick_above_pose: EndPose,
    preposition_pose: EndPose,
) -> tuple[SequenceStep | None, str | None]:
    if args.guarded_abort_return == "none":
        return None, None
    if args.guarded_abort_return == "preposition":
        return (
            SequenceStep(
                "protected_abort_return_preposition",
                preposition_pose,
                args.vertical_mode,
            ),
            "preposition",
        )
    return (
        SequenceStep(
            "protected_abort_return_pick_above",
            pick_above_pose,
            args.vertical_mode,
        ),
        "pick_above",
    )


def execute_protected_abort_return(
    controller: PiperController,
    args: argparse.Namespace,
    counts: dict[str, int],
    gripper_results: list[SafeGripperResult],
    abort_step: SequenceStep | None,
) -> tuple[str | None, str | None]:
    if abort_step is None:
        return None, None
    counts[abort_step.label] = execute_and_record_step(
        controller,
        args,
        abort_step,
        gripper_results,
    )
    abort_return = None
    if abort_step.label.startswith("protected_abort_return_"):
        abort_return = abort_step.label.removeprefix("protected_abort_return_")
    return abort_return, abort_step.label


def safe_gripper_abort_reason(result: SafeGripperResult) -> str:
    effort = (
        "unknown" if result.last_effort_nm is None else f"{result.last_effort_nm:.3f}Nm"
    )
    threshold = (
        "unknown" if result.max_effort_nm is None else f"{result.max_effort_nm:.3f}Nm"
    )
    return (
        "safe gripper effort threshold exceeded: "
        f"effort={effort} threshold={threshold}"
    )


def wait_until_guarded_start_reached(
    controller: PiperController,
    args: argparse.Namespace,
    start_pose: EndPose,
) -> tuple[bool, str | None, float | None]:
    timeout_s = float(args.guarded_start_timeout_s)
    deadline = time.monotonic() + timeout_s
    last_z_mm: float | None = None
    tolerance_mm = float(args.guarded_start_tolerance_mm)
    while timeout_s == 0.0 or time.monotonic() < deadline:
        feedback_pose = feedback_to_pose(controller.get_end_pose())
        if feedback_pose is not None:
            last_z_mm = float(feedback_pose.z_mm)
            if abs(last_z_mm - float(start_pose.z_mm)) <= tolerance_mm:
                return True, None, last_z_mm
        if timeout_s == 0.0:
            break
        time.sleep(min(max(float(args.command_period_s), 0.005), 0.05))
    return (
        False,
        "guarded descent start not reached: "
        f"target_z={start_pose.z_mm:.3f}mm "
        f"feedback_z={last_z_mm}mm "
        f"tolerance={tolerance_mm:.3f}mm "
        f"timeout={timeout_s:.3f}s",
        last_z_mm,
    )


def wait_for_dynamic_final_action(
    timing: DynamicTiming,
    *,
    elapsed_s: float,
    final_action_time_s: float,
) -> float:
    due_s = timing.t_arrive_s - final_action_time_s
    wait_s = due_s - elapsed_s
    if wait_s <= 0.0:
        late_s = elapsed_s - due_s
        print(
            "Dynamic final timing: start immediately "
            f"(t_arrive={timing.t_arrive_s:.3f}s, "
            f"elapsed={elapsed_s:.3f}s, "
            f"final_action={final_action_time_s:.3f}s, "
            f"due={due_s:.3f}s, "
            f"late={late_s:.3f}s)",
            flush=True,
        )
        return 0.0
    print(
        "Dynamic final timing: waiting "
        f"{wait_s:.3f}s at preposition "
        f"(t_arrive={timing.t_arrive_s:.3f}s, "
        f"elapsed={elapsed_s:.3f}s, "
        f"final_action={final_action_time_s:.3f}s, "
        f"due={due_s:.3f}s)",
        flush=True,
    )
    time.sleep(wait_s)
    return wait_s


def execute_guarded_descent(
    controller: PiperController,
    args: argparse.Namespace,
    *,
    start_pose: EndPose,
    target_pose: EndPose,
) -> GuardedDescentResult:
    start_z_mm = float(start_pose.z_mm)
    target_z_mm = float(target_pose.z_mm)
    if start_z_mm <= target_z_mm:
        return GuardedDescentResult(
            send_count=0,
            start_z_mm=start_z_mm,
            target_z_mm=target_z_mm,
            aborted=True,
            reason=(
                "guarded descent start is not above target: "
                f"start_z={start_z_mm:.3f} target_z={target_z_mm:.3f}"
            ),
        )
    start_reached, start_reason, start_feedback_z_mm = wait_until_guarded_start_reached(
        controller,
        args,
        start_pose,
    )
    if not start_reached:
        return GuardedDescentResult(
            send_count=0,
            start_z_mm=start_z_mm,
            target_z_mm=target_z_mm,
            aborted=True,
            reason=start_reason,
            last_feedback_z_mm=start_feedback_z_mm,
        )

    print(
        "Running guarded_descent: "
        f"z {start_z_mm:.3f}->{target_z_mm:.3f} mm "
        f"step={args.guarded_descent_step_mm:.3f}mm "
        f"speed={args.guarded_descent_speed}",
        flush=True,
    )
    send_count = 0
    commanded_z_mm = start_z_mm
    last_feedback_z_mm: float | None = start_feedback_z_mm
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None
    while commanded_z_mm > target_z_mm:
        previous_feedback_z_mm = last_feedback_z_mm
        if previous_feedback_z_mm is None:
            previous_pose = feedback_to_pose(controller.get_end_pose())
            if previous_pose is not None:
                previous_feedback_z_mm = float(previous_pose.z_mm)
        next_commanded_z_mm = max(
            target_z_mm,
            commanded_z_mm - float(args.guarded_descent_step_mm),
        )
        commanded_drop_mm = commanded_z_mm - next_commanded_z_mm
        commanded_z_mm = next_commanded_z_mm
        command_pose = replace(target_pose, z_mm=commanded_z_mm)
        controller.move_to_pose(
            command_pose,
            speed_percent=args.guarded_descent_speed,
            move_mode=args.vertical_mode,
        )
        send_count += 1
        (
            aborted,
            reason,
            last_feedback_z_mm,
            last_follow_error_mm,
            last_progress_mm,
        ) = (
            guarded_step_abort_status(
                controller,
                args,
                command_pose,
                previous_feedback_z_mm=previous_feedback_z_mm,
                commanded_drop_mm=commanded_drop_mm,
            )
        )
        if args.print_feedback:
            print(
                "Guarded step feedback: "
                f"command_z={command_pose.z_mm:.3f}mm "
                f"feedback_z={last_feedback_z_mm}mm "
                f"progress={last_progress_mm}mm "
                f"follow_error={last_follow_error_mm}mm",
                flush=True,
            )
        if aborted:
            controller.move_to_pose(
                start_pose,
                speed_percent=args.speed,
                move_mode=args.vertical_mode,
            )
            return GuardedDescentResult(
                send_count=send_count,
                start_z_mm=start_z_mm,
                target_z_mm=target_z_mm,
                aborted=True,
                reason=reason,
                last_feedback_z_mm=last_feedback_z_mm,
                last_follow_error_mm=last_follow_error_mm,
                last_progress_mm=last_progress_mm,
            )
    return GuardedDescentResult(
        send_count=send_count,
        start_z_mm=start_z_mm,
        target_z_mm=target_z_mm,
        last_feedback_z_mm=last_feedback_z_mm,
        last_follow_error_mm=last_follow_error_mm,
        last_progress_mm=last_progress_mm,
    )


def guarded_step_abort_status(
    controller: PiperController,
    args: argparse.Namespace,
    command_pose: EndPose,
    *,
    previous_feedback_z_mm: float | None = None,
    commanded_drop_mm: float | None = None,
) -> tuple[bool, str | None, float | None, float | None, float | None]:
    timeout_s = (
        float(args.guarded_step_timeout_s)
        if float(args.guarded_step_timeout_s) > 0.0
        else float(args.guarded_stall_s)
    )
    deadline = time.monotonic() + timeout_s
    last_feedback_z_mm: float | None = None
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None
    progress_min_mm = float(args.guarded_progress_min_mm)
    required_progress_mm = progress_min_mm
    if commanded_drop_mm is not None and commanded_drop_mm > 0.0 and progress_min_mm > 0.0:
        required_progress_mm = min(progress_min_mm, commanded_drop_mm)
    while True:
        feedback_pose = feedback_to_pose(controller.get_end_pose())
        if feedback_pose is not None:
            last_feedback_z_mm = float(feedback_pose.z_mm)
            last_follow_error_mm = last_feedback_z_mm - float(command_pose.z_mm)
            if previous_feedback_z_mm is not None:
                last_progress_mm = previous_feedback_z_mm - last_feedback_z_mm
            progress_ok = (
                last_progress_mm is not None
                and (
                    last_progress_mm > 0.0
                    if required_progress_mm <= 0.0
                    else last_progress_mm >= required_progress_mm
                )
            )
            follow_error_ok = last_follow_error_mm <= float(args.guarded_follow_error_mm)
            if progress_ok and follow_error_ok:
                return (
                    False,
                    None,
                    last_feedback_z_mm,
                    last_follow_error_mm,
                    last_progress_mm,
                )
        if time.monotonic() >= deadline:
            if last_feedback_z_mm is None:
                return (
                    True,
                    "no valid J6 feedback during guarded descent",
                    None,
                    None,
                    None,
                )
            progress_ok = (
                last_progress_mm is not None
                and (
                    last_progress_mm > 0.0
                    if required_progress_mm <= 0.0
                    else last_progress_mm >= required_progress_mm
                )
            )
            if progress_ok:
                return (
                    True,
                    "guarded descent feedback did not reach commanded height "
                    f"within {timeout_s:.3f}s: "
                    f"command_z={command_pose.z_mm:.3f}mm "
                    f"previous_feedback_z={previous_feedback_z_mm}mm "
                    f"feedback_z={last_feedback_z_mm:.3f}mm "
                    f"progress={last_progress_mm}mm "
                    f"follow_error={last_follow_error_mm}mm "
                    f"limit={args.guarded_follow_error_mm:.3f}mm",
                    last_feedback_z_mm,
                    last_follow_error_mm,
                    last_progress_mm,
                )
            return (
                True,
                "guarded descent height did not decrease "
                f"{required_progress_mm:.3f} mm within {timeout_s:.3f}s: "
                f"command_z={command_pose.z_mm:.3f}mm "
                f"previous_feedback_z={previous_feedback_z_mm}mm "
                f"feedback_z={last_feedback_z_mm:.3f}mm "
                f"progress={last_progress_mm}mm "
                f"follow_error={last_follow_error_mm}mm",
                last_feedback_z_mm,
                last_follow_error_mm,
                last_progress_mm,
            )
        time.sleep(min(max(float(args.command_period_s), 0.005), 0.05))


def yolo_metadata_from_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return {}

    metadata: dict[str, Any] = {}
    bbox_value = payload.get("bbox")
    bbox_format = payload.get("bbox_format")
    if bbox_value is None:
        bbox_value = payload.get("bbox_xyxy")
        if bbox_value is not None:
            bbox_format = bbox_format or "xyxy"
    if bbox_value is None:
        bbox_value = payload.get("bbox_xywh")
        if bbox_value is not None:
            bbox_format = bbox_format or "xywh"
    if bbox_value is None and all(
        key in payload for key in ("x", "y", "width", "height")
    ):
        bbox_value = [
            payload["x"],
            payload["y"],
            payload["width"],
            payload["height"],
        ]
        bbox_format = bbox_format or "xywh"
    if bbox_value is None and all(key in payload for key in ("x1", "y1", "x2", "y2")):
        bbox_value = [payload["x1"], payload["y1"], payload["x2"], payload["y2"]]
        bbox_format = bbox_format or "xyxy"

    if bbox_value is not None:
        bbox = _tuple4_float(bbox_value, "bbox")
        metadata["bbox"] = list(bbox)
        metadata["bbox_format"] = str(bbox_format or "unknown")

    for key in ("label", "source_label"):
        if payload.get(key) is not None:
            metadata["label"] = str(payload[key])
            break
    if payload.get("confidence") is not None:
        metadata["confidence"] = float(payload["confidence"])
    if payload.get("class_id") is not None:
        metadata["class_id"] = int(payload["class_id"])
    if payload.get("rank") is not None:
        metadata["rank"] = int(payload["rank"])
    if payload.get("task") is not None:
        metadata["task"] = str(payload["task"])
    return metadata


def _tuple4_float(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly 4 numeric values")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 4 numeric values") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite numeric values")
    return values


def print_plan(
    plan: YoloCatchPlan,
    args: argparse.Namespace,
    metadata: dict[str, Any] | None = None,
    timing: DynamicTiming | None = None,
) -> None:
    command = plan.target.camera_command
    base = plan.target.base_xyz_m
    print("YOLO catch plan:", flush=True)
    print(
        "  camera: "
        f"u={command.u} v={command.v} "
        f"xyz=({command.x_m:.6f}, {command.y_m:.6f}, {command.z_m:.6f}) m "
        f"source={command.source}",
        flush=True,
    )
    print(
        "  base: "
        f"x={base[0] * 1000.0:.3f} y={base[1] * 1000.0:.3f} "
        f"z={base[2] * 1000.0:.3f} mm "
        f"grasp_z={plan.target.grasp_z_mm:.3f} mm "
        f"source={plan.target.grasp_z_source}",
        flush=True,
    )
    if plan.target.raw_grasp_z_mm is not None:
        print(
            "  detected_grasp_z: "
            f"raw={plan.target.raw_grasp_z_mm:.3f}mm "
            f"fixed_floor={args.grasp_z_mm:.3f}mm "
            f"final={plan.target.grasp_z_mm:.3f}mm",
            flush=True,
        )
    if metadata:
        print(f"  yolo: {metadata}", flush=True)
    if timing is not None:
        print(
            "  dynamic_timing: "
            f"t_arrive={timing.t_arrive_s:.3f}s "
            f"t_robot={timing.t_robot_s:.3f}s "
            f"action_time={timing.action_time_s:.3f}s "
            f"t_wait={timing.t_wait_s:.3f}s "
            f"preposition={timing.dynamic_preposition} "
            f"skipped={timing.skipped}",
            flush=True,
        )
        if timing.dynamic_preposition:
            preposition_pose, preposition_source = dynamic_preposition_pose(args, plan.target)
            print(
                "  dynamic_preposition: "
                f"z={preposition_pose.z_mm:.3f}mm "
                f"source={preposition_source} "
                f"clearance={args.dynamic_preposition_clearance_mm:.3f}mm "
                f"final_action={args.final_action_time_s:.3f}s",
                flush=True,
            )
    for index, step in enumerate(plan.steps, start=1):
        gripper = "" if step.gripper_after is None else f" gripper={step.gripper_after}"
        delay = "" if step.delay_before_s <= 0 else f" delay_before={step.delay_before_s:g}s"
        print(
            f"  {index:02d}. {step.label}: mode={step.mode} "
            f"{format_pose(step.pose)}{delay}{gripper}",
            flush=True,
        )


def handle_probe_message(
    command: CameraPointCommand,
    metadata: dict[str, Any],
    *,
    camera_to_base: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        command.x_m,
        command.y_m,
        command.z_m,
        camera_to_base,
    )
    probe_pose = EndPose(
        x_mm=base_xyz_m[0] * 1000.0,
        y_mm=base_xyz_m[1] * 1000.0,
        z_mm=base_xyz_m[2] * 1000.0,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )
    label = metadata.get("label", command.label or "unknown")
    rank = metadata.get("rank", command.rank)
    bbox = metadata.get("bbox", command.bbox_xywh)
    bbox_format = metadata.get("bbox_format", "xywh" if command.bbox_xywh else None)
    print(
        "YOLO probe only: "
        f"rank={rank} label={label} "
        f"camera_pixel=({command.u},{command.v}) "
        "camera_xyz="
        f"({command.x_m:.6f}, {command.y_m:.6f}, {command.z_m:.6f}) m -> "
        "base_xyz="
        f"({base_xyz_m[0] * 1000.0:.3f}, "
        f"{base_xyz_m[1] * 1000.0:.3f}, "
        f"{base_xyz_m[2] * 1000.0:.3f}) mm "
        f"bbox={bbox} bbox_format={bbox_format}",
        flush=True,
    )
    response = make_ack_message(
        command.command_id,
        pose_to_dict(probe_pose),
        executed=False,
    )
    response["payload"].update(
        {
            "task": command.task,
            "grasp_mode": command.grasp_mode,
            "probe_only": True,
            "mapping_source": "hand_eye",
            "base_xyz_m": {
                "x_m": base_xyz_m[0],
                "y_m": base_xyz_m[1],
                "z_m": base_xyz_m[2],
            },
            "pick_x_mm": None,
            "pick_y_mm": None,
            "grasp_z_mm": None,
            "skipped": False,
        }
    )
    if metadata:
        response["payload"]["yolo"] = metadata
    return response


def grasp_effort_nm(args: argparse.Namespace) -> float:
    return float(args.effort_nm if args.grasp_effort_nm is None else args.grasp_effort_nm)


def gripper_feedback_values(feedback_msg: Any) -> tuple[float | None, float | None]:
    state = getattr(feedback_msg, "gripper_state", feedback_msg)
    position_raw = getattr(state, "grippers_angle", None)
    effort_raw = getattr(state, "grippers_effort", None)
    position_mm = None if position_raw is None else float(position_raw) * 0.001
    effort_nm = None if effort_raw is None else abs(float(effort_raw) * 0.001)
    return position_mm, effort_nm


def monitor_safe_gripper_effort(
    controller: PiperController,
    args: argparse.Namespace,
) -> SafeGripperResult:
    max_effort_nm = float(args.max_gripper_effort_nm)
    deadline = time.monotonic() + float(args.safe_gripper_monitor_s)
    poll_count = 0
    last_position_mm: float | None = None
    last_effort_nm: float | None = None
    while float(args.safe_gripper_monitor_s) == 0.0 or time.monotonic() <= deadline:
        poll_count += 1
        last_position_mm, last_effort_nm = gripper_feedback_values(
            controller.get_gripper_state()
        )
        if last_effort_nm is not None and last_effort_nm >= max_effort_nm:
            controller.set_gripper(args.open_command_mm, 0.0, enable=False)
            stop_action = "disable"
            release_command_mm = None
            if args.safe_gripper_release_mm is not None:
                release_command_mm = float(args.safe_gripper_release_mm)
                controller.set_gripper(
                    release_command_mm,
                    min(grasp_effort_nm(args), max_effort_nm),
                    enable=True,
                )
                stop_action = "disable_release"
            return SafeGripperResult(
                action="close",
                command_mm=float(args.grasp_command_mm),
                command_effort_nm=grasp_effort_nm(args),
                max_effort_nm=max_effort_nm,
                triggered=True,
                stopped=True,
                stop_action=stop_action,
                release_command_mm=release_command_mm,
                last_effort_nm=last_effort_nm,
                last_position_mm=last_position_mm,
                poll_count=poll_count,
            )
        if float(args.safe_gripper_monitor_s) == 0.0:
            break
        time.sleep(float(args.safe_gripper_poll_s))
    return SafeGripperResult(
        action="close",
        command_mm=float(args.grasp_command_mm),
        command_effort_nm=grasp_effort_nm(args),
        max_effort_nm=max_effort_nm,
        last_effort_nm=last_effort_nm,
        last_position_mm=last_position_mm,
        poll_count=poll_count,
    )


def send_gripper(
    controller: PiperController,
    args: argparse.Namespace,
    action: Action,
) -> SafeGripperResult | None:
    if action is None or args.no_gripper:
        return None
    command_mm = args.open_command_mm if action == "open" else args.grasp_command_mm
    effort_nm = float(args.effort_nm if action == "open" else grasp_effort_nm(args))
    controller.set_gripper(command_mm, effort_nm, enable=True)
    if action == "close" and args.safe_gripper_stop_on_effort:
        return monitor_safe_gripper_effort(controller, args)
    if args.gripper_settle_s > 0:
        time.sleep(args.gripper_settle_s)
    return SafeGripperResult(
        action=action,
        command_mm=float(command_mm),
        command_effort_nm=effort_nm,
        max_effort_nm=(
            float(args.max_gripper_effort_nm)
            if action == "close" and args.safe_gripper_stop_on_effort
            else None
        ),
    )


def execute_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    *,
    speed: int | None = None,
) -> tuple[int, SafeGripperResult | None]:
    if step.delay_before_s > 0:
        print(f"Waiting {step.delay_before_s:g}s before {step.label}", flush=True)
        time.sleep(step.delay_before_s)
    print(f"Running {step.label}: {format_pose(step.pose)}", flush=True)
    send_count = execute_pose_for_duration(
        controller,
        step.pose,
        args.speed if speed is None else speed,
        step.mode,
        args.command_duration_s,
        args.command_period_s,
    )
    if not args.no_wait_waypoints:
        wait_until_pose_reached(
            controller,
            step.label,
            step.pose,
            timeout_s=args.motion_timeout_s,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
        )
    gripper_result = send_gripper(controller, args, step.gripper_after)
    return send_count, gripper_result


def record_gripper_result(
    results: list[SafeGripperResult],
    result: SafeGripperResult | None,
) -> None:
    if result is not None:
        results.append(result)


def safe_gripper_result_to_dict(result: SafeGripperResult) -> dict[str, Any]:
    return {
        "action": result.action,
        "command_mm": result.command_mm,
        "command_effort_nm": result.command_effort_nm,
        "max_effort_nm": result.max_effort_nm,
        "triggered": result.triggered,
        "stopped": result.stopped,
        "stop_action": result.stop_action,
        "release_command_mm": result.release_command_mm,
        "last_effort_nm": result.last_effort_nm,
        "last_position_mm": result.last_position_mm,
        "poll_count": result.poll_count,
    }


def execute_and_record_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    gripper_results: list[SafeGripperResult],
) -> int:
    send_count, gripper_result = execute_step(controller, args, step)
    record_gripper_result(gripper_results, gripper_result)
    return send_count


def execute_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: YoloCatchPlan,
) -> DynamicExecutionResult:
    before_pose = None
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before catch J6: {format_pose(before_pose)}", flush=True)

    counts: dict[str, int] = {}
    guarded_result: GuardedDescentResult | None = None
    gripper_results: list[SafeGripperResult] = []
    for index, step in enumerate(plan.steps):
        if should_guard_step(step, args):
            guarded_start_pose = guarded_start_pose_for_grasp(args, step.pose)
            guarded_start_step = SequenceStep(
                f"{step.label}_guard_start",
                guarded_start_pose,
                args.vertical_mode,
                None,
                step.delay_before_s,
            )
            counts[guarded_start_step.label] = execute_and_record_step(
                controller,
                args,
                guarded_start_step,
                gripper_results,
            )
            guarded_result = execute_guarded_descent(
                controller,
                args,
                start_pose=guarded_start_pose,
                target_pose=step.pose,
            )
            counts[step.label] = guarded_result.send_count
            if guarded_result.aborted:
                reason = guarded_result.reason or "guarded descent aborted"
                print(f"Protected abort: {reason}", flush=True)
                abort_step = guarded_abort_return_step(
                    args,
                    plan.steps,
                    index,
                    guarded_start_pose,
                )
                abort_return, abort_return_label = execute_protected_abort_return(
                    controller,
                    args,
                    counts,
                    gripper_results,
                    abort_step,
                )
                record_gripper_result(
                    gripper_results,
                    send_gripper(controller, args, "open"),
                )
                return DynamicExecutionResult(
                    send_counts=counts,
                    protected_abort=True,
                    protected_abort_reason=reason,
                    protected_abort_return=abort_return,
                    protected_abort_return_label=abort_return_label,
                    guarded_descent=guarded_result,
                    safe_gripper_results=tuple(gripper_results),
                )
            gripper_result = send_gripper(controller, args, step.gripper_after)
            record_gripper_result(gripper_results, gripper_result)
            continue
        send_count, gripper_result = execute_step(controller, args, step)
        counts[step.label] = send_count
        record_gripper_result(gripper_results, gripper_result)

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)
    return DynamicExecutionResult(
        send_counts=counts,
        guarded_descent=guarded_result,
        safe_gripper_results=tuple(gripper_results),
    )


def execute_dynamic_preposition_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: YoloCatchPlan,
    timing: DynamicTiming,
) -> DynamicExecutionResult:
    if len(plan.steps) < 4:
        raise RuntimeError("dynamic preposition requires at least ready/above/grasp/lift steps")

    before_pose = None
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before catch J6: {format_pose(before_pose)}", flush=True)

    counts: dict[str, int] = {}
    gripper_results: list[SafeGripperResult] = []
    started_s = time.monotonic()
    ready_step = plan.steps[0]
    pick_above_step = plan.steps[1]
    grasp_step = replace(plan.steps[2], delay_before_s=0.0)
    lift_step = plan.steps[3]
    preposition_pose, preposition_source = dynamic_preposition_pose(args, plan.target)
    preposition_step = SequenceStep(
        "dynamic_preposition",
        preposition_pose,
        args.vertical_mode,
    )

    counts[ready_step.label] = execute_and_record_step(
        controller,
        args,
        ready_step,
        gripper_results,
    )
    counts[pick_above_step.label] = execute_and_record_step(
        controller,
        args,
        pick_above_step,
        gripper_results,
    )
    counts[preposition_step.label] = execute_and_record_step(
        controller,
        args,
        preposition_step,
        gripper_results,
    )

    preposition_elapsed_s = time.monotonic() - started_s
    final_action_due_s = timing.t_arrive_s - float(args.final_action_time_s)
    final_wait_s = wait_for_dynamic_final_action(
        timing,
        elapsed_s=preposition_elapsed_s,
        final_action_time_s=float(args.final_action_time_s),
    )
    final_action_late_s = max(0.0, preposition_elapsed_s - final_action_due_s)

    guarded_result: GuardedDescentResult | None = None
    if args.guarded_descent:
        guarded_result = execute_guarded_descent(
            controller,
            args,
            start_pose=preposition_pose,
            target_pose=grasp_step.pose,
        )
        counts[grasp_step.label] = guarded_result.send_count
        if guarded_result.aborted:
            reason = guarded_result.reason or "guarded descent aborted"
            print(f"Protected abort: {reason}", flush=True)
            abort_step, abort_return_label = dynamic_protected_abort_return_step(
                args,
                pick_above_pose=pick_above_step.pose,
                preposition_pose=preposition_pose,
            )
            _, abort_return_step_label = execute_protected_abort_return(
                controller,
                args,
                counts,
                gripper_results,
                abort_step,
            )
            record_gripper_result(
                gripper_results,
                send_gripper(controller, args, "open"),
            )
            return DynamicExecutionResult(
                send_counts=counts,
                protected_abort=True,
                protected_abort_reason=reason,
                protected_abort_return=abort_return_label,
                protected_abort_return_label=abort_return_step_label,
                preposition_z_mm=preposition_pose.z_mm,
                preposition_source=preposition_source,
                preposition_elapsed_s=preposition_elapsed_s,
                final_action_due_s=final_action_due_s,
                final_wait_s=final_wait_s,
                final_action_late_s=final_action_late_s,
                guarded_descent=guarded_result,
                safe_gripper_results=tuple(gripper_results),
            )
        gripper_result = send_gripper(controller, args, grasp_step.gripper_after)
        record_gripper_result(gripper_results, gripper_result)
    else:
        send_count, gripper_result = execute_step(
            controller,
            args,
            grasp_step,
        )
        counts[grasp_step.label] = send_count
        record_gripper_result(gripper_results, gripper_result)

    counts[lift_step.label] = execute_and_record_step(
        controller,
        args,
        lift_step,
        gripper_results,
    )
    for step in plan.steps[4:]:
        counts[step.label] = execute_and_record_step(
            controller,
            args,
            step,
            gripper_results,
        )

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)

    return DynamicExecutionResult(
        send_counts=counts,
        preposition_z_mm=preposition_pose.z_mm,
        preposition_source=preposition_source,
        preposition_elapsed_s=preposition_elapsed_s,
        final_action_due_s=final_action_due_s,
        final_wait_s=final_wait_s,
        final_action_late_s=final_action_late_s,
        guarded_descent=guarded_result,
        safe_gripper_results=tuple(gripper_results),
    )


def maybe_prepare_gripper(controller: PiperController, args: argparse.Namespace) -> None:
    if args.no_gripper:
        return
    if args.clear_gripper_error:
        controller.set_gripper(
            args.open_command_mm,
            args.effort_nm,
            enable=True,
            clear_error=True,
        )
        time.sleep(args.gripper_settle_s)
    if not args.no_pre_open:
        controller.set_gripper(args.open_command_mm, args.effort_nm, enable=True)
        time.sleep(args.gripper_settle_s)


def move_to_start_ready(controller: PiperController, args: argparse.Namespace) -> int:
    step = SequenceStep("startup_ready", ready_pose(args), args.transfer_mode)
    validate_steps(controller, (step,))
    print(
        "Moving to startup ready pose from run_pose_continuous_catch_sequence: "
        f"{format_pose(step.pose)}",
        flush=True,
    )
    send_count, _ = execute_step(controller, args, step, speed=args.startup_speed)
    return send_count


def handle_message(
    message: dict[str, Any],
    *,
    args: argparse.Namespace,
    camera_to_base: Any,
    validator: PiperController,
    controller: PiperController | None = None,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    metadata = yolo_metadata_from_message(message)
    if command.grasp_mode == "probe":
        return handle_probe_message(
            command,
            metadata,
            camera_to_base=camera_to_base,
            args=args,
        )
    timing = dynamic_timing_from_command(command, args)
    target = target_from_command(
        command,
        camera_to_base,
        args,
    )
    plan = YoloCatchPlan(
        task=command.task,
        target=target,
        steps=build_steps(args, target),
    )
    validate_steps(controller or validator, plan.steps)
    planned_preposition_pose: EndPose | None = None
    planned_preposition_source: str | None = None
    if timing is not None and args.dynamic_preposition:
        planned_preposition_pose, planned_preposition_source = dynamic_preposition_pose(
            args,
            target,
        )
        validate_steps(
            controller or validator,
            (
                SequenceStep(
                    "dynamic_preposition",
                    planned_preposition_pose,
                    args.vertical_mode,
                ),
            ),
        )

    if args.debug_plan or not args.execute:
        print_plan(plan, args, metadata, timing)

    execution = DynamicExecutionResult(
        send_counts={},
        preposition_z_mm=(
            None if planned_preposition_pose is None else planned_preposition_pose.z_mm
        ),
        preposition_source=planned_preposition_source,
    )
    skipped = bool(timing is not None and timing.skipped)
    if skipped and timing is not None:
        print(f"Dynamic target skipped: {timing.skip_reason}", flush=True)
    did_execute = bool(args.execute and not skipped)
    if did_execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        if timing is not None and args.dynamic_preposition:
            execution = execute_dynamic_preposition_plan(controller, args, plan, timing)
        else:
            wait_for_dynamic_timing(timing)
            execution = execute_plan(controller, args, plan)

    print(
        f"task={plan.task} "
        f"mode={command.grasp_mode} "
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"base_xy=({target.pick_x_mm:.3f}, {target.pick_y_mm:.3f}) mm "
        f"grasp_z={target.grasp_z_mm:.3f} mm "
        f"task_pose={format_pose(plan.steps[2].pose)} "
        f"place=({args.place_x_mm:.3f}, {args.place_y_mm:.3f}) mm",
        flush=True,
    )
    release_at_place = plan.task == "out" and not args.grasp_only
    pick_from_place = False
    return_ready = any(step.label in ("return_ready", "return_ready_above") for step in plan.steps)
    sequence_kind = {
        "out": "yolo_out_to_fixed_place" if not args.grasp_only else "yolo_grasp_only",
        "in": "yolo_in_to_ready",
    }[plan.task]
    response = make_ack_message(
        command.command_id,
        pose_to_dict(plan.steps[2].pose),
        executed=did_execute,
    )
    response["payload"].update(
        {
            "task": plan.task,
            "grasp_mode": command.grasp_mode,
            "mapping_source": "hand_eye",
            "base_xyz_m": {
                "x_m": target.base_xyz_m[0],
                "y_m": target.base_xyz_m[1],
                "z_m": target.base_xyz_m[2],
            },
            "pick_x_mm": target.pick_x_mm,
            "pick_y_mm": target.pick_y_mm,
            "grasp_z_mm": target.grasp_z_mm,
            "raw_grasp_z_mm": target.raw_grasp_z_mm,
            "grasp_z_source": target.grasp_z_source,
            "fixed_grasp_z_floor_mm": args.grasp_z_mm if args.use_detected_z else None,
            "place_x_mm": args.place_x_mm,
            "place_y_mm": args.place_y_mm,
            "place_pick_pose": pose_to_dict(place_pick_pose(args)),
            "steps": [
                {
                    "label": step.label,
                    "mode": step.mode,
                    "pose": pose_to_dict(step.pose),
                    "gripper_after": step.gripper_after,
                    "delay_before_s": step.delay_before_s,
                }
                for step in plan.steps
            ],
            "send_counts": execution.send_counts,
            "fixed_orientation_deg": {
                "rx_deg": args.rx_deg,
                "ry_deg": args.ry_deg,
                "rz_deg": args.rz_deg,
            },
            "sequence": {
                "kind": sequence_kind,
                "task": plan.task,
                "return_ready": return_ready,
                "release_after_grasp": False,
                "release_at_place": release_at_place,
                "pick_back_from_place": pick_from_place,
            },
            "gripper_enabled": not args.no_gripper,
            "grasp_effort_nm": grasp_effort_nm(args),
            "safe_gripper": {
                "enabled": bool(args.safe_gripper_stop_on_effort),
                "max_effort_nm": args.max_gripper_effort_nm,
                "monitor_s": args.safe_gripper_monitor_s,
                "poll_s": args.safe_gripper_poll_s,
                "release_mm": args.safe_gripper_release_mm,
                "results": [
                    safe_gripper_result_to_dict(result)
                    for result in execution.safe_gripper_results
                ],
                "triggered": any(
                    result.triggered for result in execution.safe_gripper_results
                ),
            },
            "skipped": skipped,
            "protected_abort": execution.protected_abort,
            "protected_abort_reason": execution.protected_abort_reason,
        }
    )
    guarded_execution: dict[str, Any] = {
        "guarded_descent_enabled": bool(args.guarded_descent),
        "protected_abort": execution.protected_abort,
        "protected_abort_reason": execution.protected_abort_reason,
        "protected_abort_return": execution.protected_abort_return,
        "protected_abort_return_label": execution.protected_abort_return_label,
    }
    if execution.guarded_descent is not None:
        guarded_execution["guarded_descent"] = {
            "send_count": execution.guarded_descent.send_count,
            "start_z_mm": execution.guarded_descent.start_z_mm,
            "target_z_mm": execution.guarded_descent.target_z_mm,
            "aborted": execution.guarded_descent.aborted,
            "reason": execution.guarded_descent.reason,
            "last_feedback_z_mm": execution.guarded_descent.last_feedback_z_mm,
            "last_follow_error_mm": execution.guarded_descent.last_follow_error_mm,
            "last_progress_mm": execution.guarded_descent.last_progress_mm,
            "zone_mm": args.guarded_descent_zone_mm,
            "step_mm": args.guarded_descent_step_mm,
            "speed": args.guarded_descent_speed,
            "follow_error_mm": args.guarded_follow_error_mm,
            "stall_s": args.guarded_stall_s,
            "start_tolerance_mm": args.guarded_start_tolerance_mm,
            "start_timeout_s": args.guarded_start_timeout_s,
            "progress_min_mm": args.guarded_progress_min_mm,
            "step_timeout_s": (
                args.guarded_step_timeout_s
                if args.guarded_step_timeout_s > 0.0
                else args.guarded_stall_s
            ),
        }
    response["payload"]["guarded_execution"] = guarded_execution
    if timing is not None:
        response["payload"]["dynamic_timing"] = {
            "t_arrive_s": timing.t_arrive_s,
            "t_robot_s": timing.t_robot_s,
            "action_time_s": timing.action_time_s,
            "t_wait_s": timing.t_wait_s,
            "dynamic_preposition": timing.dynamic_preposition,
            "final_action_time_s": timing.final_action_time_s,
            "skipped": timing.skipped,
            "skip_reason": timing.skip_reason,
            "conveyor_speed_mm_s": command.conveyor_speed_mm_s,
            "pick_x_mm": command.pick_x_mm,
            "pick_y_mm": command.pick_y_mm,
            "detected_x_mm": command.detected_x_mm,
            "detected_y_mm": command.detected_y_mm,
            "detected_z_mm": command.detected_z_mm,
        }
        dynamic_execution: dict[str, Any] = {
            "preposition_enabled": bool(args.dynamic_preposition),
            "guarded_descent_enabled": bool(args.guarded_descent),
            "preposition_z_mm": execution.preposition_z_mm,
            "preposition_source": execution.preposition_source,
            "preposition_clearance_mm": args.dynamic_preposition_clearance_mm,
            "preposition_elapsed_s": execution.preposition_elapsed_s,
            "final_action_due_s": execution.final_action_due_s,
            "final_wait_s": execution.final_wait_s,
            "final_action_late_s": execution.final_action_late_s,
            "protected_abort": execution.protected_abort,
            "protected_abort_reason": execution.protected_abort_reason,
            "protected_abort_return": execution.protected_abort_return,
            "protected_abort_return_label": execution.protected_abort_return_label,
        }
        if "guarded_descent" in guarded_execution:
            dynamic_execution["guarded_descent"] = guarded_execution["guarded_descent"]
        response["payload"]["dynamic_execution"] = dynamic_execution
    if metadata:
        response["payload"]["yolo"] = metadata
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
            levels = [args.collision_protection_level] * 6
            controller.set_collision_protection(levels)
            print(
                "Piper collision protection configured: "
                f"levels={levels}",
                flush=True,
            )
        maybe_prepare_gripper(controller, args)
        if not args.no_start_ready:
            move_to_start_ready(controller, args)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"Listening on {args.host}:{args.port}, execute={args.execute}, "
                f"speed={args.speed}, startup_speed={args.startup_speed}, "
                f"fixed_orientation="
                f"({args.rx_deg:.3f}, {args.ry_deg:.3f}, {args.rz_deg:.3f})",
                flush=True,
            )
            print(f"Catch server 1 args: {args}", flush=True)
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

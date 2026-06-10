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


@dataclass(frozen=True)
class YoloCatchPlan:
    task: CameraPointTask
    target: YoloCatchTarget
    steps: tuple[SequenceStep, ...]


@dataclass(frozen=True)
class DynamicTiming:
    t_arrive_s: float
    t_robot_s: float
    t_wait_s: float
    skipped: bool = False
    skip_reason: str | None = None


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
    parser.add_argument("--place-x-mm", type=float, default=100.108)
    parser.add_argument("--place-y-mm", type=float, default=200.0)
    parser.add_argument(
        "--place-pick-x-mm",
        type=float,
        help="X used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-y-mm",
        type=float,
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
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not isinstance(args.speed, int) or not 0 <= args.speed <= 100:
        raise ValueError("--speed must be an integer in [0, 100]")
    for name in (
        "command_duration_s",
        "motion_timeout_s",
        "position_tolerance_mm",
        "orientation_tolerance_deg",
        "t_robot_s",
        "pick_delay_s",
        "place_pick_delay_s",
        "gripper_settle_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(args.command_period_s) or args.command_period_s <= 0:
        raise ValueError("--command-period-s must be finite and > 0")
    if args.grasp_z_mm > args.transfer_z_mm:
        raise ValueError("--grasp-z-mm should not be above --transfer-z-mm")
    if args.place_pick_z_mm is not None and args.place_pick_z_mm > args.transfer_z_mm:
        raise ValueError("--place-pick-z-mm should not be above --transfer-z-mm")
    if not math.isfinite(args.effort_nm) or not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be finite and in [0, 5]")

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
    if args.use_detected_z:
        grasp_z_mm = base_xyz_m[2] * 1000.0 + args.detected_z_offset_mm
    else:
        grasp_z_mm = args.grasp_z_mm
    target = YoloCatchTarget(
        camera_command=command,
        base_xyz_m=base_xyz_m,
        pick_x_mm=float(pick_x_mm),
        pick_y_mm=float(pick_y_mm),
        grasp_z_mm=float(grasp_z_mm),
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
    t_wait_s = t_arrive_s - t_robot_s
    if t_wait_s < 0.0:
        return DynamicTiming(
            t_arrive_s=t_arrive_s,
            t_robot_s=t_robot_s,
            t_wait_s=t_wait_s,
            skipped=True,
            skip_reason=(
                "dynamic target is too close to the pick line: "
                f"t_arrive={t_arrive_s:.3f}s < t_robot={t_robot_s:.3f}s"
            ),
        )
    return DynamicTiming(
        t_arrive_s=t_arrive_s,
        t_robot_s=t_robot_s,
        t_wait_s=t_wait_s,
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
        f"grasp_z={plan.target.grasp_z_mm:.3f} mm",
        flush=True,
    )
    if metadata:
        print(f"  yolo: {metadata}", flush=True)
    if timing is not None:
        print(
            "  dynamic_timing: "
            f"t_arrive={timing.t_arrive_s:.3f}s "
            f"t_robot={timing.t_robot_s:.3f}s "
            f"t_wait={timing.t_wait_s:.3f}s "
            f"skipped={timing.skipped}",
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


def send_gripper(controller: PiperController, args: argparse.Namespace, action: Action) -> None:
    if action is None or args.no_gripper:
        return
    command_mm = args.open_command_mm if action == "open" else args.grasp_command_mm
    controller.set_gripper(command_mm, args.effort_nm, enable=True)
    if args.gripper_settle_s > 0:
        time.sleep(args.gripper_settle_s)


def execute_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
) -> int:
    if step.delay_before_s > 0:
        print(f"Waiting {step.delay_before_s:g}s before {step.label}", flush=True)
        time.sleep(step.delay_before_s)
    print(f"Running {step.label}: {format_pose(step.pose)}", flush=True)
    send_count = execute_pose_for_duration(
        controller,
        step.pose,
        args.speed,
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
    send_gripper(controller, args, step.gripper_after)
    return send_count


def execute_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: YoloCatchPlan,
) -> dict[str, int]:
    before_pose = None
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before catch J6: {format_pose(before_pose)}", flush=True)

    counts: dict[str, int] = {}
    for step in plan.steps:
        counts[step.label] = execute_step(controller, args, step)

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)
    return counts


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
    return execute_step(controller, args, step)


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

    if args.debug_plan or not args.execute:
        print_plan(plan, metadata, timing)

    send_counts: dict[str, int] = {}
    skipped = bool(timing is not None and timing.skipped)
    if skipped and timing is not None:
        print(f"Dynamic target skipped: {timing.skip_reason}", flush=True)
    did_execute = bool(args.execute and not skipped)
    if did_execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        wait_for_dynamic_timing(timing)
        send_counts = execute_plan(controller, args, plan)

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
            "send_counts": send_counts,
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
            "skipped": skipped,
        }
    )
    if timing is not None:
        response["payload"]["dynamic_timing"] = {
            "t_arrive_s": timing.t_arrive_s,
            "t_robot_s": timing.t_robot_s,
            "t_wait_s": timing.t_wait_s,
            "skipped": timing.skipped,
            "skip_reason": timing.skip_reason,
            "conveyor_speed_mm_s": command.conveyor_speed_mm_s,
            "pick_x_mm": command.pick_x_mm,
            "pick_y_mm": command.pick_y_mm,
            "detected_x_mm": command.detected_x_mm,
            "detected_y_mm": command.detected_y_mm,
            "detected_z_mm": command.detected_z_mm,
        }
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
                f"speed={args.speed}, fixed_orientation="
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

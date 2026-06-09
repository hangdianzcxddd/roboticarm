"""Run a fixed pose-based continuous catch sequence.

The sequence uses robot-frame J6 poses directly, matching the coordinates used
by ``python -m robot_control.linux.arm.arm_cli pose``.  It keeps one fixed wrist
orientation for every waypoint and uses a high transfer plane for horizontal
motion, dropping to the grasp plane only for vertical pick/place actions.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from typing import Literal

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import (
    execute_pose_for_duration,
    format_pose,
    wait_until_pose_reached,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose


Action = Literal["open", "close", None]


@dataclass(frozen=True)
class SequenceStep:
    label: str
    pose: EndPose
    mode: Literal["P", "L"]
    gripper_after: Action = None
    delay_before_s: float = 0.0


@dataclass(frozen=True)
class XYPoint:
    x_mm: float
    y_mm: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a continuous catch/place sequence using fixed arm_cli-style "
            "robot-frame poses."
        )
    )
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
    parser.add_argument(
        "--initial-pick-delay-s",
        type=float,
        default=5.0,
        help="pause at pick1_above before descending to the first pick grasp",
    )
    parser.add_argument(
        "--place-pick-delay-s",
        type=float,
        default=3.0,
        help="pause at ready before moving to place_pick_above",
    )
    parser.add_argument("--no-wait-waypoints", action="store_true")

    parser.add_argument("--ready-x-mm", type=float, default=343.108)
    parser.add_argument("--ready-y-mm", type=float, default=0.0)
    parser.add_argument("--transfer-z-mm", type=float, default=253.252)
    parser.add_argument("--grasp-z-mm", type=float, default=133.252)
    parser.add_argument("--place-x-mm", type=float, default=100.108)
    parser.add_argument("--place-y-mm", type=float, default=200.0)
    parser.add_argument(
        "--place-pick-x-mm",
        type=float,
        help="X used when picking the object back from the place area",
    )
    parser.add_argument(
        "--place-pick-y-mm",
        type=float,
        help="Y used when picking the object back from the place area",
    )
    parser.add_argument(
        "--place-pick-z-mm",
        type=float,
        help="Z used when picking the object back from the place area",
    )
    parser.add_argument(
        "--pick-count",
        type=int,
        help=(
            "number of objects to pick. If omitted, explicit --pick-point "
            "values decide the count; without --pick-point the default is 2. "
            "Generated points start at --first-pick-x-mm and step by "
            "--pick-x-gap-mm with shared --pick-y-mm."
        ),
    )
    parser.add_argument(
        "--pick-point",
        nargs=2,
        action="append",
        type=float,
        metavar=("X_MM", "Y_MM"),
        help=(
            "explicit object pick point. Repeat this option for multiple "
            "objects, for example: --pick-point 330 0 --pick-point 360 20"
        ),
    )
    parser.add_argument("--first-pick-x-mm", type=float, default=343.108)
    parser.add_argument(
        "--second-pick-x-mm",
        type=float,
        help=(
            "second pick X. If omitted, it is computed from "
            "--first-pick-x-mm + --pick-x-gap-mm"
        ),
    )
    parser.add_argument("--pick-x-gap-mm", type=float, default=20.0)
    parser.add_argument(
        "--pick-y-mm",
        type=float,
        default=0.0,
        help="shared Y value for the first two generated pick points",
    )

    parser.add_argument("--rx-deg", type=float, default=180.0)
    parser.add_argument("--ry-deg", type=float, default=14.795)
    parser.add_argument("--rz-deg", type=float, default=180.0)
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
        "initial_pick_delay_s",
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
    if not math.isfinite(args.effort_nm) or not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be finite and in [0, 5]")
    if args.pick_count is not None and args.pick_count < 1:
        raise ValueError("--pick-count must be >= 1")
    for name in (
        "ready_x_mm",
        "ready_y_mm",
        "transfer_z_mm",
        "grasp_z_mm",
        "place_x_mm",
        "place_y_mm",
        "place_pick_x_mm",
        "place_pick_y_mm",
        "place_pick_z_mm",
        "first_pick_x_mm",
        "pick_x_gap_mm",
        "pick_y_mm",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "open_command_mm",
        "grasp_command_mm",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.second_pick_x_mm is not None and not math.isfinite(args.second_pick_x_mm):
        raise ValueError("--second-pick-x-mm must be finite")
    pick_points_from_args(args)


def second_pick_x_mm(args: argparse.Namespace) -> float:
    if args.second_pick_x_mm is not None:
        return args.second_pick_x_mm
    return args.first_pick_x_mm + args.pick_x_gap_mm


def pick_points_from_args(args: argparse.Namespace) -> tuple[XYPoint, ...]:
    if args.pick_point:
        points = tuple(XYPoint(x_mm=float(x), y_mm=float(y)) for x, y in args.pick_point)
        if args.pick_count is not None and args.pick_count != len(points):
            raise ValueError(
                "--pick-count must match the number of --pick-point values "
                f"({args.pick_count} != {len(points)})"
            )
    else:
        pick_count = 2 if args.pick_count is None else args.pick_count
        points_list = [XYPoint(args.first_pick_x_mm, args.pick_y_mm)]
        if pick_count >= 2:
            points_list.append(XYPoint(second_pick_x_mm(args), args.pick_y_mm))
        while len(points_list) < pick_count:
            points_list.append(
                XYPoint(points_list[-1].x_mm + args.pick_x_gap_mm, args.pick_y_mm)
            )
        points = tuple(points_list)

    for index, point in enumerate(points, start=1):
        if not math.isfinite(point.x_mm) or not math.isfinite(point.y_mm):
            raise ValueError(f"pick point {index} must contain finite X/Y values")
    seen: set[tuple[float, float]] = set()
    for index, point in enumerate(points, start=1):
        key = (round(point.x_mm, 6), round(point.y_mm, 6))
        if key in seen:
            raise ValueError(f"pick point {index} duplicates an earlier pick point")
        seen.add(key)
    return points


def place_pick_pose(args: argparse.Namespace) -> EndPose:
    x_mm = args.place_x_mm if args.place_pick_x_mm is None else args.place_pick_x_mm
    y_mm = args.place_y_mm if args.place_pick_y_mm is None else args.place_pick_y_mm
    z_mm = args.grasp_z_mm if args.place_pick_z_mm is None else args.place_pick_z_mm
    return pose(args, x_mm, y_mm, z_mm)


def pose(args: argparse.Namespace, x_mm: float, y_mm: float, z_mm: float) -> EndPose:
    return EndPose(x_mm, y_mm, z_mm, args.rx_deg, args.ry_deg, args.rz_deg)


def build_steps(args: argparse.Namespace) -> tuple[SequenceStep, ...]:
    ready = pose(args, args.ready_x_mm, args.ready_y_mm, args.transfer_z_mm)
    pick_points = pick_points_from_args(args)
    place_above = pose(args, args.place_x_mm, args.place_y_mm, args.transfer_z_mm)
    place_pick = place_pick_pose(args)
    place_pick_above = pose(args, place_pick.x_mm, place_pick.y_mm, args.transfer_z_mm)

    steps: list[SequenceStep] = [
        SequenceStep("ready_open", ready, args.transfer_mode, "open"),
    ]
    for index, pick_point in enumerate(pick_points, start=1):
        pick_above = pose(args, pick_point.x_mm, pick_point.y_mm, args.transfer_z_mm)
        pick_grasp = pose(args, pick_point.x_mm, pick_point.y_mm, args.grasp_z_mm)
        steps.extend(
            [
                SequenceStep(f"pick{index}_above", pick_above, args.transfer_mode),
                SequenceStep(
                    f"pick{index}_grasp",
                    pick_grasp,
                    args.vertical_mode,
                    "close",
                    args.initial_pick_delay_s if index == 1 else 0.0,
                ),
                SequenceStep(f"pick{index}_lift", pick_above, args.vertical_mode),
                SequenceStep(f"place{index}_above", place_above, args.transfer_mode),
                SequenceStep(f"place{index}_release", place_above, args.transfer_mode, "open"),
            ]
        )
        if index < len(pick_points):
            steps.append(SequenceStep(f"ready_after_place{index}", ready, args.transfer_mode))

    steps.extend(
        [
            SequenceStep("ready_before_place_pick", ready, args.transfer_mode),
            SequenceStep(
                "place_pick_above",
                place_pick_above,
                args.transfer_mode,
                None,
                args.place_pick_delay_s,
            ),
            SequenceStep(
                "place_pick_grasp",
                place_pick,
                args.vertical_mode,
                "close",
            ),
            SequenceStep("place_pick_lift", place_pick_above, args.vertical_mode),
            SequenceStep("return_ready_above", ready, args.transfer_mode),
            SequenceStep("final_release", ready, args.transfer_mode, "open"),
        ]
    )
    return tuple(steps)


def validate_steps(controller: PiperController, steps: tuple[SequenceStep, ...]) -> None:
    for step in steps:
        try:
            controller.validate_pose(step.pose)
        except SafetyError as exc:
            raise SafetyError(f"{step.label}: {exc}") from exc


def print_steps(steps: tuple[SequenceStep, ...]) -> None:
    print("Planned pose continuous catch steps:", flush=True)
    for index, step in enumerate(steps, start=1):
        gripper = "" if step.gripper_after is None else f" gripper={step.gripper_after}"
        delay = "" if step.delay_before_s <= 0 else f" delay_before={step.delay_before_s:g}s"
        print(
            f"  {index:02d}. {step.label}: mode={step.mode} "
            f"{format_pose(step.pose)}{delay}{gripper}",
            flush=True,
        )


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


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    config = replace(ROBOT_CONFIG, can_name=args.can)
    controller = PiperController(config=config)
    steps = build_steps(args)
    validate_steps(controller, steps)
    print_steps(steps)
    if not args.execute:
        print("Dry run only. Add --execute to move the arm and gripper.", flush=True)
        return

    controller.connect(enable=True)
    try:
        if not args.no_gripper:
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

        counts: dict[str, int] = {}
        for step in steps:
            counts[step.label] = execute_step(controller, args, step)
        print(f"Sequence send counts: {counts}", flush=True)
    finally:
        controller.disconnect()


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except (SafetyError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

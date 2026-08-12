#!/usr/bin/env python3
"""Control Piper gripper position and effort without changing its zero point."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG, RobotConfig
from robot_control.linux.gripper.gripper_debug import feedback_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send ordinary Piper gripper position and torque commands. "
            "This script never changes the gripper zero point."
        )
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument(
        "--command-mm",
        type=float,
        help=(
            "Signed gripper position command in mm. After fully-open zeroing, "
            "use 0 for fully open and negative values to close."
        ),
    )
    parser.add_argument(
        "--effort-nm",
        type=float,
        default=1.0,
        help="Gripper torque in N.m, valid range 0-5.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Seconds to keep resending the command; 0 sends once.",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=0.02,
        help="Seconds between repeated commands.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.5,
        help="Seconds to wait before printing feedback.",
    )
    parser.add_argument(
        "--clear-error",
        action="store_true",
        help="Clear gripper error while sending the first ordinary control command.",
    )
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="Do not print gripper feedback after commands.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the connection open and accept repeated '<command_mm> [effort_nm]' inputs.",
    )
    return parser


def validate_args(args: argparse.Namespace, config: RobotConfig = ROBOT_CONFIG) -> None:
    if args.command_mm is None and not args.interactive:
        raise ValueError("provide --command-mm or use --interactive")
    if args.command_mm is not None:
        validate_command(args.command_mm, args.effort_nm, config)
    elif not math.isfinite(args.effort_nm) or not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be finite and in [0, 5]")
    if not math.isfinite(args.duration_s) or args.duration_s < 0:
        raise ValueError("--duration-s must be finite and >= 0")
    if not math.isfinite(args.period_s) or args.period_s <= 0:
        raise ValueError("--period-s must be finite and > 0")
    if not math.isfinite(args.settle_s) or args.settle_s < 0:
        raise ValueError("--settle-s must be finite and >= 0")


def validate_command(
    command_mm: float,
    effort_nm: float,
    config: RobotConfig = ROBOT_CONFIG,
) -> None:
    min_mm = config.gripper_command_min_m * 1000.0
    max_mm = config.gripper_command_max_m * 1000.0
    if not math.isfinite(command_mm) or not min_mm <= command_mm <= max_mm:
        raise ValueError(f"command_mm must be finite and in [{min_mm:g}, {max_mm:g}]")
    if not math.isfinite(effort_nm) or not 0 <= effort_nm <= 5:
        raise ValueError("effort_nm must be finite and in [0, 5]")


def make_controller(can_name: str) -> PiperController:
    return PiperController(config=replace(ROBOT_CONFIG, can_name=can_name))


def send_gripper_target(
    controller: PiperController,
    command_mm: float,
    effort_nm: float,
    duration_s: float = 0.0,
    period_s: float = 0.02,
    clear_error: bool = False,
) -> int:
    validate_command(command_mm, effort_nm, controller.config)
    if not math.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and >= 0")
    if not math.isfinite(period_s) or period_s <= 0:
        raise ValueError("period_s must be finite and > 0")

    controller.set_gripper(
        command_mm,
        effort_nm,
        enable=True,
        clear_error=clear_error,
    )
    send_count = 1
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        time.sleep(period_s)
        controller.set_gripper(command_mm, effort_nm, enable=True)
        send_count += 1
    return send_count


def print_result(
    controller: PiperController,
    command_mm: float,
    effort_nm: float,
    send_count: int,
    settle_s: float,
    no_feedback: bool,
) -> None:
    print(
        f"sent command_mm={command_mm:.3f} effort_nm={effort_nm:.3f} "
        f"count={send_count} set_zero=false",
        flush=True,
    )
    if settle_s > 0:
        time.sleep(settle_s)
    if not no_feedback:
        print(json.dumps(feedback_to_dict(controller.get_gripper_state()), indent=2))


def run_interactive(
    controller: PiperController,
    args: argparse.Namespace,
    clear_error_pending: bool,
) -> None:
    print("Enter: <command_mm> [effort_nm], or: status, quit", flush=True)
    while True:
        try:
            line = input("gripper> ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            return
        if line == "status":
            print(json.dumps(feedback_to_dict(controller.get_gripper_state()), indent=2))
            continue
        try:
            command_mm, effort_nm = parse_interactive_command(line, args.effort_nm)
            validate_command(command_mm, effort_nm, controller.config)
        except ValueError as exc:
            print(f"invalid command: {exc}", flush=True)
            continue
        count = send_gripper_target(
            controller,
            command_mm,
            effort_nm,
            duration_s=args.duration_s,
            period_s=args.period_s,
            clear_error=clear_error_pending,
        )
        clear_error_pending = False
        print_result(
            controller,
            command_mm,
            effort_nm,
            count,
            args.settle_s,
            args.no_feedback,
        )


def parse_interactive_command(line: str, default_effort_nm: float) -> tuple[float, float]:
    parts = line.split()
    if len(parts) not in (1, 2):
        raise ValueError("expected '<command_mm> [effort_nm]'")
    command_mm = float(parts[0])
    effort_nm = default_effort_nm if len(parts) == 1 else float(parts[1])
    return command_mm, effort_nm


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    controller = make_controller(args.can)
    controller.connect(enable=True)
    try:
        clear_error_pending = args.clear_error
        if args.command_mm is not None:
            count = send_gripper_target(
                controller,
                args.command_mm,
                args.effort_nm,
                duration_s=args.duration_s,
                period_s=args.period_s,
                clear_error=clear_error_pending,
            )
            clear_error_pending = False
            print_result(
                controller,
                args.command_mm,
                args.effort_nm,
                count,
                args.settle_s,
                args.no_feedback,
            )
        if args.interactive:
            run_interactive(controller, args, clear_error_pending)
    finally:
        controller.disconnect()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

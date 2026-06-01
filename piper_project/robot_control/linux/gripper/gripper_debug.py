#!/usr/bin/env python3
"""Manual debug tool for the official Piper gripper.

The project wrapper uses millimeters for gripper travel and N.m for effort.
The Piper SDK converts those to 0.001 mm and 0.001 N.m internally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.arm.piper_arm import PiperController


DEFAULT_CALIBRATION_PERCENTS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
DEFAULT_GRIPPER_CONFIG_PATH = Path(__file__).with_name("gripper_config.yaml")
FOC_STATUS_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "sensor_status",
    "driver_error_status",
    "driver_enable_status",
    "homing_status",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Piper official gripper communication, calibration, and feedback tests"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    communication_parser = subparsers.add_parser(
        "communication",
        aliases=("comm",),
        help="stage 1: open, close, then stop the gripper",
    )
    add_common_connection_args(communication_parser)
    add_common_motion_args(communication_parser)
    communication_parser.add_argument(
        "--open-width",
        type=float,
        default=70.0,
        help="open target width in mm",
    )
    communication_parser.add_argument(
        "--close-width",
        type=float,
        default=0.0,
        help="close target width in mm",
    )
    communication_parser.add_argument(
        "--hold",
        type=float,
        default=2.0,
        help="seconds to wait after open and close commands",
    )
    communication_parser.add_argument(
        "--stop-wait",
        type=float,
        default=0.5,
        help="seconds to wait after stop command",
    )
    communication_parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="do not print feedback after each command",
    )

    calibration_parser = subparsers.add_parser(
        "calibration",
        aliases=("calib",),
        help="stage 2: sweep widths and optionally record measured openings",
    )
    add_common_connection_args(calibration_parser)
    add_common_motion_args(calibration_parser)
    calibration_parser.add_argument(
        "--min-width",
        type=float,
        default=0.0,
        help="minimum command width in mm for percent-based sweep",
    )
    calibration_parser.add_argument(
        "--max-width",
        type=float,
        default=70.0,
        help="maximum command width in mm for percent-based sweep",
    )
    calibration_parser.add_argument(
        "--percents",
        type=float,
        nargs="+",
        default=list(DEFAULT_CALIBRATION_PERCENTS),
        help="command percentages to sweep, mapped between min-width and max-width",
    )
    calibration_parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        help="absolute command widths in mm; overrides --percents",
    )
    calibration_parser.add_argument(
        "--settle",
        type=float,
        default=2.0,
        help="seconds to wait after each width command",
    )
    calibration_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="do not ask for measured opening after each point",
    )
    calibration_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_GRIPPER_CONFIG_PATH,
        help="path for calibration result yaml",
    )
    calibration_parser.add_argument(
        "--no-save",
        action="store_true",
        help="print results only, do not write calibration yaml",
    )
    calibration_parser.add_argument(
        "--no-stop-at-end",
        action="store_true",
        help="do not send gripper stop/disable after calibration",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="stage 3: read gripper feedback",
    )
    add_common_connection_args(status_parser)
    status_parser.add_argument(
        "--enable",
        action="store_true",
        help="enable the arm before reading feedback",
    )
    status_parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="seconds between feedback reads",
    )
    status_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="number of feedback samples; 0 means run until Ctrl-C",
    )
    status_parser.add_argument(
        "--duration",
        type=float,
        help="read feedback for this many seconds; overrides --count",
    )
    status_parser.add_argument(
        "--raw",
        action="store_true",
        help="also include SDK object's string representation",
    )
    status_parser.add_argument(
        "--pretty",
        action="store_true",
        help="pretty-print JSON feedback",
    )

    command_parser = subparsers.add_parser(
        "command",
        help="send one open, close, stop, or clear-error command",
    )
    add_common_connection_args(command_parser)
    add_common_motion_args(command_parser)
    command_parser.add_argument(
        "action",
        choices=("open", "close", "stop", "clear-error", "reset-enable", "set-zero"),
        help="single gripper command to send",
    )
    command_parser.add_argument(
        "--width",
        type=float,
        help="target width in mm; defaults to 70 for open and 0 for close",
    )
    command_parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="seconds to wait before reading feedback",
    )
    command_parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds to keep resending open/close target; 0 sends once",
    )
    command_parser.add_argument(
        "--period",
        type=float,
        default=0.005,
        help="seconds between repeated gripper commands when --duration is used",
    )
    command_parser.add_argument(
        "--official-init",
        action="store_true",
        help="send disable+clear-error then enable before movement, matching SDK demo",
    )
    command_parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="do not print feedback after command",
    )
    command_parser.add_argument(
        "--show-control",
        action="store_true",
        help="also print the last gripper control command seen by SDK",
    )

    return parser


def add_common_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--can",
        default=ROBOT_CONFIG.can_name,
        help="CAN interface name",
    )


def add_common_motion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--effort",
        type=float,
        default=1.0,
        help="gripper effort in N.m, valid range 0-5",
    )
    parser.add_argument(
        "--clear-error",
        action="store_true",
        help="send enable+clear-error before movement",
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command in ("communication", "comm"):
        run_communication(args)
    elif args.command in ("calibration", "calib"):
        run_calibration(args)
    elif args.command == "status":
        run_status(args)
    elif args.command == "command":
        run_command(args)


def make_controller(can_name: str) -> PiperController:
    return PiperController(config=replace(ROBOT_CONFIG, can_name=can_name))


def run_communication(args: argparse.Namespace) -> None:
    validate_width(args.open_width, "open-width")
    validate_width(args.close_width, "close-width")

    controller = make_controller(args.can)
    controller.connect(enable=True)
    try:
        maybe_clear_error(controller, args.effort, args.clear_error)
        steps = (
            ("open", args.open_width, args.hold),
            ("close", args.close_width, args.hold),
            ("stop", None, args.stop_wait),
        )
        for step, width_mm, wait_s in steps:
            send_gripper_command(controller, step, width_mm, args.effort)
            print_command(step, width_mm, args.effort)
            time.sleep(wait_s)
            if not args.no_feedback:
                print_feedback(controller.get_gripper_state(), label=step)
    finally:
        controller.disconnect()


def run_calibration(args: argparse.Namespace) -> None:
    validate_width(args.min_width, "min-width")
    validate_width(args.max_width, "max-width")
    if args.max_width < args.min_width:
        raise SystemExit("--max-width must be greater than or equal to --min-width")

    points = calibration_points(args)
    controller = make_controller(args.can)
    prompt_measurement = should_prompt_for_measurement(args.no_prompt)
    records: list[dict[str, Any]] = []

    controller.connect(enable=True)
    try:
        maybe_clear_error(controller, args.effort, args.clear_error)
        print(
            "command_percent,command_width_mm,feedback_position_mm,"
            "feedback_effort_nm,status_code,actual_width_mm"
        )
        for command_percent, width_mm in points:
            validate_width(width_mm, "calibration width")
            send_gripper_command(controller, "open", width_mm, args.effort)
            time.sleep(args.settle)
            feedback = feedback_to_dict(controller.get_gripper_state())
            actual_width_mm = prompt_actual_width(width_mm) if prompt_measurement else None
            record = {
                "command_percent": command_percent,
                "command_width_mm": round(width_mm, 3),
                "feedback_position_mm": feedback["position_mm"],
                "feedback_effort_nm": feedback["effort_nm"],
                "status_code": feedback["status_code"],
                "actual_width_mm": actual_width_mm,
            }
            records.append(record)
            print_calibration_record(record)
    finally:
        if not args.no_stop_at_end:
            try:
                send_gripper_command(controller, "stop", None, args.effort)
            except Exception as exc:  # pragma: no cover - hardware cleanup path
                print(f"failed to stop gripper: {exc}", file=sys.stderr)
        controller.disconnect()

    summary = calibration_summary(records)
    print_calibration_summary(summary)
    if not args.no_save:
        write_calibration_yaml(args.output, records, summary)
        print(f"saved calibration: {args.output}")


def run_status(args: argparse.Namespace) -> None:
    controller = make_controller(args.can)
    controller.connect(enable=args.enable)
    try:
        for _ in feedback_sample_indexes(args.count, args.duration, args.interval):
            feedback_msg = controller.get_gripper_state()
            feedback = feedback_to_dict(feedback_msg)
            if args.raw:
                feedback["raw"] = str(feedback_msg)
            print_json(feedback, pretty=args.pretty)
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        controller.disconnect()


def run_command(args: argparse.Namespace) -> None:
    width = args.width
    if width is None:
        width = 70.0 if args.action == "open" else 0.0
    validate_width(width, "width")
    validate_duration(args.duration, "duration")
    validate_period(args.period, "period")

    controller = make_controller(args.can)
    controller.connect(enable=True)
    try:
        if args.official_init:
            reset_enable_gripper(controller, args.effort)
        maybe_clear_error(controller, args.effort, args.clear_error)
        command_width = (
            None
            if args.action in ("stop", "clear-error", "reset-enable", "set-zero")
            else width
        )
        send_gripper_command_repeated(
            controller,
            args.action,
            command_width,
            args.effort,
            args.duration,
            args.period,
        )
        print_command(args.action, command_width, args.effort)
        time.sleep(args.wait)
        if args.show_control:
            print_gripper_control(controller.get_gripper_command())
        if not args.no_feedback:
            print_feedback(controller.get_gripper_state(), label=args.action)
    finally:
        controller.disconnect()


def maybe_clear_error(controller: PiperController, effort_nm: float, enabled: bool) -> None:
    if enabled:
        controller.set_gripper(0.0, effort_nm=effort_nm, enable=True, clear_error=True)
        time.sleep(0.2)


def reset_enable_gripper(controller: PiperController, effort_nm: float) -> None:
    controller.set_gripper(0.0, effort_nm=effort_nm, enable=False, clear_error=True)
    time.sleep(0.05)
    controller.set_gripper(0.0, effort_nm=effort_nm, enable=True)
    time.sleep(0.05)


def send_gripper_command_repeated(
    controller: PiperController,
    command: str,
    width_mm: float | None,
    effort_nm: float,
    duration_s: float,
    period_s: float,
) -> None:
    if duration_s <= 0 or command not in ("open", "close"):
        send_gripper_command(controller, command, width_mm, effort_nm)
        return
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        send_gripper_command(controller, command, width_mm, effort_nm)
        time.sleep(period_s)


def send_gripper_command(
    controller: PiperController,
    command: str,
    width_mm: float | None,
    effort_nm: float,
) -> None:
    if command == "open":
        if width_mm is None:
            raise ValueError("open requires width_mm")
        controller.set_gripper(width_mm, effort_nm, enable=True)
    elif command == "close":
        if width_mm is None:
            raise ValueError("close requires width_mm")
        controller.set_gripper(width_mm, effort_nm, enable=True)
    elif command == "stop":
        controller.set_gripper(0.0, 0.0, enable=False)
    elif command == "clear-error":
        controller.set_gripper(0.0, effort_nm, enable=True, clear_error=True)
    elif command == "reset-enable":
        reset_enable_gripper(controller, effort_nm)
    elif command == "set-zero":
        controller.set_gripper(0.0, effort_nm, enable=False)
        time.sleep(1.5)
        controller.set_gripper(0.0, effort_nm, enable=False, set_zero=True)
    else:
        raise ValueError(f"unsupported gripper command: {command}")


def feedback_to_dict(feedback_msg: Any) -> dict[str, Any]:
    state = getattr(feedback_msg, "gripper_state", feedback_msg)
    status_code = getattr(state, "status_code", None)
    foc_status = getattr(state, "foc_status", None)
    return {
        "time_stamp": getattr(feedback_msg, "time_stamp", None),
        "hz": getattr(feedback_msg, "Hz", None),
        "position_mm": scaled_value(getattr(state, "grippers_angle", None), 0.001),
        "effort_nm": scaled_value(getattr(state, "grippers_effort", None), 0.001),
        "current_a": None,
        "force_n": None,
        "moving": None,
        "status_code": status_code,
        "error_code": status_code,
        "foc_status": {
            field: getattr(foc_status, field)
            for field in FOC_STATUS_FIELDS
            if foc_status is not None and hasattr(foc_status, field)
        },
    }


def scaled_value(value: Any, factor: float) -> float | None:
    if value is None:
        return None
    return round(float(value) * factor, 3)


def print_command(command: str, width_mm: float | None, effort_nm: float) -> None:
    if command == "stop":
        width_text = "disabled"
        effort_text = "0.000 N.m"
    elif width_mm is None:
        width_text = "0.000 mm"
        effort_text = f"{effort_nm:.3f} N.m"
    else:
        width_text = f"{width_mm:.3f} mm"
        effort_text = f"{effort_nm:.3f} N.m"
    print(f"{command}: width={width_text}, effort={effort_text}")


def print_feedback(feedback_msg: Any, label: str) -> None:
    payload = feedback_to_dict(feedback_msg)
    payload["step"] = label
    print_json(payload, pretty=True)


def print_gripper_control(control_msg: Any) -> None:
    payload = gripper_control_to_dict(control_msg)
    payload["step"] = "control"
    print_json(payload, pretty=True)


def gripper_control_to_dict(control_msg: Any) -> dict[str, Any]:
    control = getattr(control_msg, "gripper_ctrl", control_msg)
    return {
        "time_stamp": getattr(control_msg, "time_stamp", None),
        "hz": getattr(control_msg, "Hz", None),
        "command_width_mm": scaled_value(getattr(control, "grippers_angle", None), 0.001),
        "command_effort_nm": scaled_value(
            getattr(control, "grippers_effort", None), 0.001
        ),
        "command_status_code": getattr(control, "status_code", None),
        "set_zero": getattr(control, "set_zero", None),
    }


def print_json(payload: dict[str, Any], pretty: bool = False) -> None:
    indent = 2 if pretty else None
    print(json.dumps(payload, ensure_ascii=False, indent=indent))


def calibration_points(args: argparse.Namespace) -> list[tuple[float | None, float]]:
    if args.widths is not None:
        return [(None, width) for width in args.widths]
    points = []
    span = args.max_width - args.min_width
    for percent in args.percents:
        if percent < 0 or percent > 100:
            raise SystemExit("--percents values must be in range 0-100")
        width = args.min_width + span * (percent / 100.0)
        points.append((percent, width))
    return points


def should_prompt_for_measurement(no_prompt: bool) -> bool:
    if no_prompt:
        return False
    if sys.stdin.isatty():
        return True
    print("measurement prompt disabled because stdin is not interactive", file=sys.stderr)
    return False


def prompt_actual_width(command_width_mm: float) -> float | None:
    while True:
        raw_value = input(
            f"Measured opening for command {command_width_mm:.3f} mm "
            "(blank to skip): "
        ).strip()
        if raw_value == "":
            return None
        try:
            value = float(raw_value)
        except ValueError:
            print("please enter a number, or blank to skip")
            continue
        if value < 0:
            print("measured width must be >= 0")
            continue
        return round(value, 3)


def print_calibration_record(record: dict[str, Any]) -> None:
    print(
        f"{format_optional(record['command_percent'])},"
        f"{format_optional(record['command_width_mm'])},"
        f"{format_optional(record['feedback_position_mm'])},"
        f"{format_optional(record['feedback_effort_nm'])},"
        f"{format_optional(record['status_code'])},"
        f"{format_optional(record['actual_width_mm'])}"
    )


def calibration_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record["actual_width_mm"] for record in records]
    measured = [value for value in measured if value is not None]
    source = "actual_width_mm"
    if not measured:
        measured = [record["feedback_position_mm"] for record in records]
        measured = [value for value in measured if value is not None]
        source = "feedback_position_mm"
    if not measured:
        return {"min_width": None, "max_width": None, "source": source}
    return {
        "min_width": round(min(measured), 3),
        "max_width": round(max(measured), 3),
        "source": source,
    }


def print_calibration_summary(summary: dict[str, Any]) -> None:
    print("gripper:")
    print(f"  min_width: {format_optional(summary['min_width'])}")
    print(f"  max_width: {format_optional(summary['max_width'])}")
    print(f"  source: {summary['source']}")


def write_calibration_yaml(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "gripper:",
        f"  min_width: {format_yaml_value(summary['min_width'])}",
        f"  max_width: {format_yaml_value(summary['max_width'])}",
        f"  source: {summary['source']}",
        "calibration_points:",
    ]
    for record in records:
        lines.extend(
            [
                "  - command_percent: "
                f"{format_yaml_value(record['command_percent'])}",
                f"    command_width_mm: {format_yaml_value(record['command_width_mm'])}",
                "    feedback_position_mm: "
                f"{format_yaml_value(record['feedback_position_mm'])}",
                f"    feedback_effort_nm: {format_yaml_value(record['feedback_effort_nm'])}",
                f"    status_code: {format_yaml_value(record['status_code'])}",
                f"    actual_width_mm: {format_yaml_value(record['actual_width_mm'])}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def feedback_sample_indexes(
    count: int,
    duration: float | None,
    interval: float,
) -> Any:
    if duration is not None:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            yield None
        return
    if count == 0:
        while True:
            yield None
    if count < 0:
        raise SystemExit("--count must be >= 0")
    for index in range(count):
        yield index


def validate_width(width_mm: float, name: str) -> None:
    if width_mm < 0:
        raise SystemExit(f"--{name} must be >= 0")


def validate_duration(duration_s: float, name: str) -> None:
    if duration_s < 0:
        raise SystemExit(f"--{name} must be >= 0")


def validate_period(period_s: float, name: str) -> None:
    if period_s <= 0:
        raise SystemExit(f"--{name} must be > 0")


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()

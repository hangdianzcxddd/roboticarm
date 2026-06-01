#!/usr/bin/env python3
"""Diagnose Piper J6-to-gripper communication faults.

This tool keeps the diagnosis evidence-oriented:

1. Arm SDK connection and feedback are alive.
2. J6 motor driver feedback is visible.
3. Gripper commands are sent.
4. Gripper feedback CAN ID 0x2A8 appears and changes after commands.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import Counter
from dataclasses import replace
from typing import Any, Callable

from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG, PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.gripper.gripper_debug import (
    feedback_to_dict,
    gripper_control_to_dict,
)


ARM_STATUS_ID = 0x2A1
ARM_JOINT_FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7)
ARM_GRIPPER_FEEDBACK_ID = 0x2A8
ARM_GRIPPER_CTRL_ID = 0x159
J6_HIGH_SPEED_FEEDBACK_ID = 0x256
J6_LOW_SPEED_FEEDBACK_ID = 0x266

LOW_SPEED_FOC_FIELDS = (
    "voltage_too_low",
    "motor_overheating",
    "driver_overcurrent",
    "driver_overheating",
    "collision_status",
    "driver_error_status",
    "driver_enable_status",
    "stall_status",
)


class CanIdMonitor:
    """Best-effort raw SocketCAN frame counter."""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.error: str | None = None
        self._bus: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._counts: Counter[int] = Counter()
        self._last_messages: dict[int, dict[str, Any]] = {}

    def start(self) -> bool:
        try:
            import can

            try:
                self._bus = can.interface.Bus(
                    channel=self.channel,
                    interface="socketcan",
                    receive_own_messages=True,
                )
            except TypeError:
                self._bus = can.interface.Bus(
                    channel=self.channel,
                    interface="socketcan",
                )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

        self._thread = threading.Thread(target=self._run, name="can-id-monitor")
        self._thread.daemon = True
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._bus.recv(timeout=0.1)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return
            if msg is None:
                continue
            can_id = int(msg.arbitration_id)
            data = bytes(msg.data)
            with self._lock:
                self._counts[can_id] += 1
                self._last_messages[can_id] = {
                    "timestamp": getattr(msg, "timestamp", None),
                    "dlc": getattr(msg, "dlc", len(data)),
                    "data_hex": data.hex(" "),
                    "is_rx": bool(getattr(msg, "is_rx", True)),
                }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            last_messages = dict(self._last_messages)
        interesting_ids = (
            ARM_STATUS_ID,
            *ARM_JOINT_FEEDBACK_IDS,
            ARM_GRIPPER_FEEDBACK_ID,
            ARM_GRIPPER_CTRL_ID,
            J6_HIGH_SPEED_FEEDBACK_ID,
            J6_LOW_SPEED_FEEDBACK_ID,
        )
        return {
            "channel": self.channel,
            "error": self.error,
            "counts": {hex(key): value for key, value in sorted(counts.items())},
            "interesting_counts": {
                hex(can_id): counts.get(can_id, 0) for can_id in interesting_ids
            },
            "last_interesting_messages": {
                hex(can_id): last_messages[can_id]
                for can_id in interesting_ids
                if can_id in last_messages
            },
            "decoded_last_interesting_messages": {
                hex(can_id): decode_raw_can_message(can_id, last_messages[can_id])
                for can_id in interesting_ids
                if can_id in last_messages
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Piper J6 driver board / end-adapter / gripper CAN path"
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name, help="CAN interface name")
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=1.5,
        help="seconds to collect SDK feedback in each stage",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="seconds between SDK feedback reads",
    )
    parser.add_argument(
        "--open-width",
        type=float,
        default=50.0,
        help="gripper open target in mm",
    )
    parser.add_argument(
        "--close-width",
        type=float,
        default=0.0,
        help="gripper close target in mm",
    )
    parser.add_argument(
        "--effort",
        type=float,
        default=1.0,
        help="gripper effort in N.m, valid range 0-5",
    )
    parser.add_argument(
        "--command-wait",
        type=float,
        default=1.0,
        help="seconds to wait after each gripper command before sampling",
    )
    parser.add_argument(
        "--send-duration",
        type=float,
        default=0.5,
        help="seconds to continuously resend each gripper command; 0 sends once",
    )
    parser.add_argument(
        "--send-period",
        type=float,
        default=0.005,
        help="seconds between repeated gripper command sends",
    )
    parser.add_argument(
        "--skip-can-monitor",
        action="store_true",
        help="skip raw SocketCAN ID counting and use SDK feedback only",
    )
    parser.add_argument(
        "--skip-movej-check",
        action="store_true",
        help="do not send the safe MoveJ hold-current-position check",
    )
    parser.add_argument(
        "--official-init",
        action="store_true",
        help="send disable+clear-error then enable before movement, matching SDK demo",
    )
    parser.add_argument(
        "--skip-gripper-pre-enable",
        action="store_true",
        help="skip the default gripper clear-error + enable stage before movement",
    )
    parser.add_argument(
        "--jog-j6-deg",
        type=float,
        default=0.0,
        help="optional J6 jog angle in degrees; 0 only sends current-position hold",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=10,
        help="MoveJ speed percent for the optional joint check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print full JSON report instead of the concise text report",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    report = run_diagnosis(args)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text_report(report)


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_duration <= 0:
        raise SystemExit("--sample-duration must be > 0")
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")
    if args.open_width < 0 or args.close_width < 0:
        raise SystemExit("--open-width and --close-width must be >= 0")
    if not 0 <= args.effort <= 5:
        raise SystemExit("--effort must be in [0, 5]")
    if args.command_wait < 0:
        raise SystemExit("--command-wait must be >= 0")
    if args.send_duration < 0:
        raise SystemExit("--send-duration must be >= 0")
    if args.send_period <= 0:
        raise SystemExit("--send-period must be > 0")
    if not 1 <= args.speed <= 100:
        raise SystemExit("--speed must be in [1, 100]")


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    config = replace(ROBOT_CONFIG, can_name=args.can)
    controller = PiperController(config=config)
    monitor = None if args.skip_can_monitor else CanIdMonitor(args.can)
    can_monitor_started = monitor.start() if monitor is not None else False

    report: dict[str, Any] = {
        "can": args.can,
        "ids": {
            "gripper_ctrl": hex(ARM_GRIPPER_CTRL_ID),
            "gripper_feedback": hex(ARM_GRIPPER_FEEDBACK_ID),
            "j6_high_speed_feedback": hex(J6_HIGH_SPEED_FEEDBACK_ID),
            "j6_low_speed_feedback": hex(J6_LOW_SPEED_FEEDBACK_ID),
        },
        "parameters": {
            "sample_duration_s": args.sample_duration,
            "interval_s": args.interval,
            "open_width_mm": args.open_width,
            "close_width_mm": args.close_width,
            "effort_nm": args.effort,
            "send_duration_s": args.send_duration,
            "send_period_s": args.send_period,
            "movej_check": not args.skip_movej_check,
            "official_init": args.official_init,
            "gripper_pre_enable": not args.skip_gripper_pre_enable,
            "jog_j6_deg": args.jog_j6_deg,
        },
        "can_monitor": {
            "enabled": monitor is not None,
            "started": can_monitor_started,
            "error": monitor.error if monitor is not None else None,
        },
        "stages": [],
        "commands": [],
    }

    connected = False
    try:
        controller.connect(enable=True)
        connected = True
        report["connect"] = {"ok": True}
        report["stages"].append(
            collect_sdk_feedback(
                controller,
                "baseline_after_enable",
                args.sample_duration,
                args.interval,
            )
        )

        if not args.skip_movej_check:
            movej_result = run_movej_check(controller, args.speed, args.jog_j6_deg)
            report["movej_check"] = movej_result
            report["stages"].append(
                collect_sdk_feedback(
                    controller,
                    "after_movej_check",
                    args.sample_duration,
                    args.interval,
                )
            )

        if args.official_init:
            for label, width_mm, enable, clear_error in (
                ("official_disable_clear_error", 0.0, False, True),
                ("official_enable_after_clear_error", 0.0, True, False),
            ):
                run_gripper_stage(
                    controller,
                    report,
                    label,
                    width_mm,
                    args.effort,
                    enable,
                    clear_error,
                    args.send_duration,
                    args.send_period,
                    args.command_wait,
                    args.sample_duration,
                    args.interval,
                )

        if not args.skip_gripper_pre_enable:
            pre_enable_result = pre_enable_gripper(
                controller,
                report,
                args.effort,
                args.send_duration,
                args.send_period,
                args.command_wait,
                args.sample_duration,
                args.interval,
            )
            report["gripper_pre_enable"] = pre_enable_result

        gripper_commands = (
            ("open", args.open_width, True, False),
            ("close", args.close_width, True, False),
        )
        for label, width_mm, enable, clear_error in gripper_commands:
            run_gripper_stage(
                controller,
                report,
                label,
                width_mm,
                args.effort,
                enable,
                clear_error,
                args.send_duration,
                args.send_period,
                args.command_wait,
                args.sample_duration,
                args.interval,
            )
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if monitor is not None:
            report["can_monitor"].update(monitor.snapshot())
            report["can_monitor"]["snapshot_point"] = "before_cleanup"
        if connected:
            try:
                controller.set_gripper(0.0, 0.0, enable=False)
                report["commands"].append(
                    {
                        "label": "gripper_disable_cleanup",
                        "ok": True,
                        "width_mm": 0.0,
                        "effort_nm": 0.0,
                    }
                )
            except Exception as exc:
                report["cleanup_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            try:
                controller.disconnect()
            except Exception as exc:
                report["disconnect_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        if monitor is not None:
            monitor.stop()

    report["analysis"] = analyze_report(report)
    return report


def pre_enable_gripper(
    controller: PiperController,
    report: dict[str, Any],
    effort_nm: float,
    send_duration_s: float,
    send_period_s: float,
    command_wait_s: float,
    sample_duration_s: float,
    interval_s: float,
) -> dict[str, Any]:
    for label, width_mm, enable, clear_error in (
        ("pre_enable_clear_error", 0.0, True, True),
        ("pre_enable_enable", 0.0, True, False),
    ):
        run_gripper_stage(
            controller,
            report,
            label,
            width_mm,
            effort_nm,
            enable,
            clear_error,
            send_duration_s,
            send_period_s,
            command_wait_s,
            sample_duration_s,
            interval_s,
        )
    latest_gripper = report["stages"][-1].get("last", {}).get("gripper", {})
    foc_status = latest_gripper.get("foc_status", {})
    return {
        "driver_enabled": bool(foc_status.get("driver_enable_status")),
        "homed": bool(foc_status.get("homing_status")),
        "status_code": latest_gripper.get("status_code"),
        "position_mm": latest_gripper.get("position_mm"),
    }


def run_gripper_stage(
    controller: PiperController,
    report: dict[str, Any],
    label: str,
    width_mm: float,
    effort_nm: float,
    enable: bool,
    clear_error: bool,
    send_duration_s: float,
    send_period_s: float,
    command_wait_s: float,
    sample_duration_s: float,
    interval_s: float,
) -> None:
    command_result = send_gripper_command(
        controller,
        label,
        width_mm,
        effort_nm,
        enable=enable,
        clear_error=clear_error,
        send_duration_s=send_duration_s,
        send_period_s=send_period_s,
    )
    report["commands"].append(command_result)
    time.sleep(command_wait_s)
    report["stages"].append(
        collect_sdk_feedback(
            controller,
            f"after_gripper_{label}",
            sample_duration_s,
            interval_s,
        )
    )


def collect_sdk_feedback(
    controller: PiperController,
    label: str,
    duration_s: float,
    interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + duration_s
    stage: dict[str, Any] = {
        "label": label,
        "samples": 0,
        "max_hz": {},
        "last": {},
        "fresh": {},
    }

    readers: dict[str, Callable[[], Any]] = {
        "status": controller.get_status,
        "joint": controller.get_joint_state,
        "gripper": controller.get_gripper_state,
        "gripper_ctrl": controller.get_gripper_command,
        "high_spd": lambda: call_sdk_method(controller, "GetArmHighSpdInfoMsgs"),
        "low_spd": lambda: call_sdk_method(controller, "GetArmLowSpdInfoMsgs"),
    }

    while True:
        stage["samples"] += 1
        for name, reader in readers.items():
            try:
                message = reader()
                payload = normalize_message(name, message)
            except Exception as exc:
                payload = {"error": f"{type(exc).__name__}: {exc}"}
            stage["last"][name] = payload
            hz = numeric_value(payload.get("hz"))
            if hz is not None:
                stage["max_hz"][name] = max(float(hz), stage["max_hz"].get(name, 0.0))
            if has_fresh_feedback(payload):
                stage["fresh"][name] = True

        if time.monotonic() >= deadline:
            break
        time.sleep(interval_s)

    return stage


def call_sdk_method(controller: PiperController, method_name: str) -> Any:
    method = getattr(controller.sdk, method_name, None)
    if method is None:
        return {"missing_method": method_name}
    return method()


def normalize_message(name: str, message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if name == "gripper":
        return feedback_to_dict(message)
    if name == "gripper_ctrl":
        return gripper_control_to_dict(message)
    if name == "joint":
        return joint_feedback_to_dict(message)
    if name == "status":
        return arm_status_to_dict(message)
    if name == "high_spd":
        return high_speed_feedback_to_dict(message)
    if name == "low_spd":
        return low_speed_feedback_to_dict(message)
    return {"raw": str(message)}


def arm_status_to_dict(message: Any) -> dict[str, Any]:
    status = getattr(message, "arm_status", message)
    return {
        "time_stamp": getattr(message, "time_stamp", None),
        "hz": getattr(message, "Hz", None),
        "ctrl_mode": getattr(status, "ctrl_mode", None),
        "arm_status": getattr(status, "arm_status", None),
        "mode_feed": getattr(status, "mode_feed", None),
        "teach_status": getattr(status, "teach_status", None),
        "motion_status": getattr(status, "motion_status", None),
        "trajectory_num": getattr(status, "trajectory_num", None),
        "err_code": getattr(status, "err_code", None),
    }


def joint_feedback_to_dict(message: Any) -> dict[str, Any]:
    joints_rad = joint_feedback_to_rad(message)
    return {
        "time_stamp": getattr(message, "time_stamp", None),
        "hz": getattr(message, "Hz", None),
        "joints_rad": joints_rad,
        "joints_deg": (
            [round(math.degrees(value), 3) for value in joints_rad]
            if joints_rad is not None
            else None
        ),
    }


def joint_feedback_to_rad(message: Any) -> list[float] | None:
    joint_state = getattr(message, "joint_state", None)
    if joint_state is None:
        return None
    values = [getattr(joint_state, f"joint_{index}", None) for index in range(1, 7)]
    if any(value is None for value in values):
        return None
    return [float(value) / RAD_TO_MILLI_DEG for value in values]


def high_speed_feedback_to_dict(message: Any) -> dict[str, Any]:
    return {
        "time_stamp": getattr(message, "time_stamp", None),
        "hz": getattr(message, "Hz", None),
        "motors": {
            str(index): high_speed_motor_to_dict(
                getattr(message, f"motor_{index}", None)
            )
            for index in range(1, 7)
        },
    }


def high_speed_motor_to_dict(motor: Any) -> dict[str, Any]:
    if motor is None:
        return {}
    return {
        "can_id": int_or_none(getattr(motor, "can_id", None)),
        "can_id_hex": hex_or_none(getattr(motor, "can_id", None)),
        "motor_speed": getattr(motor, "motor_speed", None),
        "current": getattr(motor, "current", None),
        "pos": getattr(motor, "pos", None),
        "effort": getattr(motor, "effort", None),
    }


def low_speed_feedback_to_dict(message: Any) -> dict[str, Any]:
    return {
        "time_stamp": getattr(message, "time_stamp", None),
        "hz": getattr(message, "Hz", None),
        "motors": {
            str(index): low_speed_motor_to_dict(
                getattr(message, f"motor_{index}", None)
            )
            for index in range(1, 7)
        },
    }


def low_speed_motor_to_dict(motor: Any) -> dict[str, Any]:
    if motor is None:
        return {}
    foc_status = getattr(motor, "foc_status", None)
    return {
        "can_id": int_or_none(getattr(motor, "can_id", None)),
        "can_id_hex": hex_or_none(getattr(motor, "can_id", None)),
        "voltage_v": scaled_value(getattr(motor, "vol", None), 0.1),
        "foc_temp_c": getattr(motor, "foc_temp", None),
        "motor_temp_c": getattr(motor, "motor_temp", None),
        "foc_status_code": getattr(motor, "foc_status_code", None),
        "bus_current_a": scaled_value(getattr(motor, "bus_current", None), 0.001),
        "foc_status": {
            field: getattr(foc_status, field)
            for field in LOW_SPEED_FOC_FIELDS
            if foc_status is not None and hasattr(foc_status, field)
        },
    }


def scaled_value(value: Any, factor: float) -> float | None:
    if value is None:
        return None
    return round(float(value) * factor, 3)


def run_movej_check(
    controller: PiperController,
    speed_percent: int,
    jog_j6_deg: float,
) -> dict[str, Any]:
    current_joints = joint_feedback_to_rad(controller.get_joint_state())
    if current_joints is None:
        return {
            "ok": False,
            "message": "failed to read current joint feedback before MoveJ check",
        }

    result: dict[str, Any] = {
        "ok": True,
        "speed_percent": speed_percent,
        "start_joints_deg": [round(math.degrees(value), 3) for value in current_joints],
    }
    controller.move_joints(current_joints, speed_percent=speed_percent)
    result["hold_current_position_sent"] = True

    if jog_j6_deg != 0:
        jogged = list(current_joints)
        jogged[5] += math.radians(jog_j6_deg)
        controller.validate_joints(jogged)
        controller.move_joints(jogged, speed_percent=speed_percent)
        time.sleep(0.8)
        controller.move_joints(current_joints, speed_percent=speed_percent)
        result["jog_j6_deg"] = jog_j6_deg
    return result


def send_gripper_command(
    controller: PiperController,
    label: str,
    width_mm: float,
    effort_nm: float,
    enable: bool,
    clear_error: bool,
    send_duration_s: float = 0.0,
    send_period_s: float = 0.005,
) -> dict[str, Any]:
    result = {
        "label": label,
        "width_mm": width_mm,
        "effort_nm": effort_nm,
        "enable": enable,
        "clear_error": clear_error,
        "send_duration_s": send_duration_s,
        "send_period_s": send_period_s,
        "send_count": 0,
    }
    try:
        if send_duration_s <= 0:
            controller.set_gripper(
                width_mm,
                effort_nm=effort_nm,
                enable=enable,
                clear_error=clear_error,
            )
            result["send_count"] = 1
        else:
            deadline = time.monotonic() + send_duration_s
            while True:
                controller.set_gripper(
                    width_mm,
                    effort_nm=effort_nm,
                    enable=enable,
                    clear_error=clear_error,
                )
                result["send_count"] += 1
                if time.monotonic() >= deadline:
                    break
                time.sleep(send_period_s)
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        result["ok"] = True
    return result


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    stages = report.get("stages", [])
    latest = stages[-1] if stages else {}
    raw_counts = raw_interesting_counts(report)

    arm_feedback_ok = any_stage_fresh(stages, "status") and any_stage_fresh(stages, "joint")
    joint_feedback_ok = any_stage_fresh(stages, "joint")
    j6_sdk_feedback_ok = any_stage_has_motor_can_id(
        stages,
        "high_spd",
        "6",
        J6_HIGH_SPEED_FEEDBACK_ID,
    ) or any_stage_has_motor_can_id(
        stages,
        "low_spd",
        "6",
        J6_LOW_SPEED_FEEDBACK_ID,
    )
    j6_raw_feedback_ok = (
        raw_counts.get(hex(J6_HIGH_SPEED_FEEDBACK_ID), 0) > 0
        or raw_counts.get(hex(J6_LOW_SPEED_FEEDBACK_ID), 0) > 0
    )
    j6_feedback_ok = j6_sdk_feedback_ok or j6_raw_feedback_ok

    gripper_sdk_feedback_ok = any_stage_fresh(stages, "gripper")
    gripper_raw_feedback_ok = raw_counts.get(hex(ARM_GRIPPER_FEEDBACK_ID), 0) > 0
    gripper_feedback_ok = gripper_sdk_feedback_ok or gripper_raw_feedback_ok
    gripper_ctrl_seen_on_can = raw_counts.get(hex(ARM_GRIPPER_CTRL_ID), 0) > 0
    gripper_commands_ok = all(
        command.get("ok", False)
        for command in report.get("commands", [])
        if command.get("label") != "gripper_disable_cleanup"
    )
    gripper_position_changed = gripper_position_changed_across_stages(stages)
    gripper_status = latest.get("last", {}).get("gripper", {})
    gripper_foc = gripper_status.get("foc_status", {})
    gripper_driver_enabled = bool(gripper_foc.get("driver_enable_status"))
    gripper_homed = bool(gripper_foc.get("homing_status"))
    gripper_has_error = bool(
        gripper_foc.get("voltage_too_low")
        or gripper_foc.get("motor_overheating")
        or gripper_foc.get("driver_overcurrent")
        or gripper_foc.get("driver_overheating")
        or gripper_foc.get("sensor_status")
        or gripper_foc.get("driver_error_status")
    )

    findings: list[str] = []
    conclusion = "inconclusive"

    if report.get("error"):
        conclusion = "sdk_or_can_setup_failed"
        findings.append("SDK/CAN 连接或使能阶段失败，先处理基础 CAN/供电/驱动环境。")
    elif not arm_feedback_ok:
        conclusion = "base_can_or_arm_feedback_failed"
        findings.append("机械臂状态/关节反馈不稳定，不能直接判断 J6 到夹爪链路。")
    elif not j6_feedback_ok:
        conclusion = "j6_driver_or_j6_can_feedback_suspect"
        findings.append("机械臂主反馈存在，但 J6 驱动反馈没有出现，优先检查 J6 驱动板/线束。")
    elif gripper_feedback_ok and gripper_position_changed and not gripper_has_error:
        conclusion = "gripper_path_ok"
        findings.append("夹爪反馈存在且位置随命令变化，J6 到夹爪通讯链路基本正常。")
    elif gripper_feedback_ok and gripper_has_error:
        conclusion = "gripper_reports_driver_or_sensor_error"
        findings.append("夹爪节点有反馈，但状态码报告错误，优先按状态码检查夹爪本体/供电/传感器。")
    elif gripper_feedback_ok and not gripper_driver_enabled:
        conclusion = "gripper_feedback_present_but_driver_not_enabled"
        findings.append(
            "夹爪反馈 0x2A8 存在，但 driver_enable_status 为 False；命令到达 CAN 后夹爪驱动没有进入使能。"
        )
    elif gripper_commands_ok and not gripper_feedback_ok:
        conclusion = "j6_to_gripper_or_end_adapter_suspect"
        findings.append(
            "机械臂和 J6 反馈正常，夹爪命令已调用，但夹爪反馈 0x2A8 不存在；高度怀疑 J6 到夹爪链路、末端转接板或夹爪节点。"
        )
    elif gripper_feedback_ok and not gripper_position_changed:
        conclusion = "gripper_feedback_present_but_no_motion"
        findings.append("夹爪反馈存在但位置没有变化，检查夹爪使能、堵转、机械卡滞或力矩设置。")
    else:
        findings.append("证据不足，建议延长采样时间并确认 raw CAN 监视已启动。")

    if gripper_commands_ok and not gripper_ctrl_seen_on_can and report.get("can_monitor", {}).get("started"):
        findings.append("raw CAN 未看到 0x159；如果 SDK 没报错，仍需确认 SocketCAN 是否接收本机发送回环。")

    return {
        "conclusion": conclusion,
        "arm_feedback_ok": arm_feedback_ok,
        "joint_feedback_ok": joint_feedback_ok,
        "j6_feedback_ok": j6_feedback_ok,
        "j6_sdk_feedback_ok": j6_sdk_feedback_ok,
        "j6_raw_feedback_ok": j6_raw_feedback_ok,
        "gripper_commands_ok": gripper_commands_ok,
        "gripper_ctrl_seen_on_can": gripper_ctrl_seen_on_can,
        "gripper_feedback_ok": gripper_feedback_ok,
        "gripper_sdk_feedback_ok": gripper_sdk_feedback_ok,
        "gripper_raw_feedback_ok": gripper_raw_feedback_ok,
        "gripper_driver_enabled": gripper_driver_enabled,
        "gripper_homed": gripper_homed,
        "gripper_position_changed": gripper_position_changed,
        "gripper_has_error": gripper_has_error,
        "findings": findings,
    }


def raw_interesting_counts(report: dict[str, Any]) -> dict[str, int]:
    can_monitor = report.get("can_monitor", {})
    counts = can_monitor.get("interesting_counts", {})
    return {str(key): int(value) for key, value in counts.items()}


def decode_raw_can_message(can_id: int, raw_message: dict[str, Any]) -> dict[str, Any]:
    data_hex = raw_message.get("data_hex")
    if not isinstance(data_hex, str):
        return {}
    data = bytes.fromhex(data_hex)
    if can_id == ARM_GRIPPER_CTRL_ID and len(data) >= 8:
        grippers_angle = int.from_bytes(data[0:4], byteorder="big", signed=True)
        grippers_effort = int.from_bytes(data[4:6], byteorder="big", signed=False)
        return {
            "type": "gripper_ctrl",
            "command_width_mm": round(grippers_angle * 0.001, 3),
            "command_effort_nm": round(grippers_effort * 0.001, 3),
            "command_status_code": data[6],
            "command_status_hex": hex(data[6]),
            "set_zero": data[7],
            "set_zero_hex": hex(data[7]),
        }
    if can_id == ARM_GRIPPER_FEEDBACK_ID and len(data) >= 8:
        grippers_angle = int.from_bytes(data[0:4], byteorder="big", signed=True)
        grippers_effort = int.from_bytes(data[4:6], byteorder="big", signed=True)
        status_code = data[6]
        return {
            "type": "gripper_feedback",
            "position_mm": round(grippers_angle * 0.001, 3),
            "effort_nm": round(grippers_effort * 0.001, 3),
            "status_code": status_code,
            "status_hex": hex(status_code),
            "driver_enabled": bool(status_code & (1 << 6)),
            "homed": bool(status_code & (1 << 7)),
            "error_bits": {
                "voltage_too_low": bool(status_code & (1 << 0)),
                "motor_overheating": bool(status_code & (1 << 1)),
                "driver_overcurrent": bool(status_code & (1 << 2)),
                "driver_overheating": bool(status_code & (1 << 3)),
                "sensor_status": bool(status_code & (1 << 4)),
                "driver_error_status": bool(status_code & (1 << 5)),
            },
        }
    return {}


def any_stage_fresh(stages: list[dict[str, Any]], name: str) -> bool:
    return any(stage.get("fresh", {}).get(name, False) for stage in stages)


def any_stage_has_motor_can_id(
    stages: list[dict[str, Any]],
    message_name: str,
    motor_index: str,
    expected_can_id: int,
) -> bool:
    for stage in stages:
        payload = stage.get("last", {}).get(message_name, {})
        motor = payload.get("motors", {}).get(motor_index, {})
        if motor.get("can_id") == expected_can_id:
            return True
    return False


def gripper_position_changed_across_stages(stages: list[dict[str, Any]]) -> bool:
    positions = []
    for stage in stages:
        gripper = stage.get("last", {}).get("gripper", {})
        value = numeric_value(gripper.get("position_mm"))
        if value is not None:
            positions.append(value)
    if len(positions) < 2:
        return False
    return max(positions) - min(positions) >= 1.0


def has_fresh_feedback(payload: dict[str, Any]) -> bool:
    timestamp = numeric_value(payload.get("time_stamp"))
    hz = numeric_value(payload.get("hz"))
    if timestamp is not None and timestamp > 0:
        return True
    return hz is not None and hz > 0


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hex_or_none(value: Any) -> str | None:
    int_value = int_or_none(value)
    if int_value is None:
        return None
    return hex(int_value)


def print_text_report(report: dict[str, Any]) -> None:
    analysis = report["analysis"]
    print("Piper J6/Gripper diagnostic")
    print(f"CAN: {report['can']}")
    if report.get("error"):
        print(f"ERROR: {report['error']['type']}: {report['error']['message']}")

    print("\nChecks")
    print(f"- arm feedback: {format_bool(analysis['arm_feedback_ok'])}")
    print(f"- J6 feedback: {format_bool(analysis['j6_feedback_ok'])}")
    print(f"- gripper commands completed: {format_bool(analysis['gripper_commands_ok'])}")
    print(f"- gripper 0x159 seen on raw CAN: {format_bool(analysis['gripper_ctrl_seen_on_can'])}")
    print(f"- gripper 0x2A8 feedback: {format_bool(analysis['gripper_feedback_ok'])}")
    print(f"- gripper driver enabled: {format_bool(analysis['gripper_driver_enabled'])}")
    print(f"- gripper homed: {format_bool(analysis['gripper_homed'])}")
    print(f"- gripper position changed: {format_bool(analysis['gripper_position_changed'])}")
    print(f"- gripper status has error bits: {format_bool(analysis['gripper_has_error'])}")

    can_monitor = report.get("can_monitor", {})
    if can_monitor.get("enabled"):
        print("\nRaw CAN interesting counts")
        if can_monitor.get("error"):
            print(f"- monitor error: {can_monitor['error']}")
        for can_id, count in can_monitor.get("interesting_counts", {}).items():
            print(f"- {can_id}: {count}")
        decoded = can_monitor.get("decoded_last_interesting_messages", {})
        if decoded:
            print("\nRaw CAN last decoded")
            gripper_ctrl = decoded.get(hex(ARM_GRIPPER_CTRL_ID), {})
            gripper_feedback = decoded.get(hex(ARM_GRIPPER_FEEDBACK_ID), {})
            if gripper_ctrl:
                print(
                    "- 0x159 gripper ctrl: "
                    f"width={format_optional(gripper_ctrl.get('command_width_mm'))} mm, "
                    f"effort={format_optional(gripper_ctrl.get('command_effort_nm'))} N.m, "
                    f"status={format_optional(gripper_ctrl.get('command_status_hex'))}, "
                    f"set_zero={format_optional(gripper_ctrl.get('set_zero_hex'))}"
                )
            if gripper_feedback:
                print(
                    "- 0x2A8 gripper feedback: "
                    f"pos={format_optional(gripper_feedback.get('position_mm'))} mm, "
                    f"effort={format_optional(gripper_feedback.get('effort_nm'))} N.m, "
                    f"status={format_optional(gripper_feedback.get('status_hex'))}, "
                    f"enabled={format_bool(bool(gripper_feedback.get('driver_enabled')))}, "
                    f"homed={format_bool(bool(gripper_feedback.get('homed')))}"
                )

    print("\nStage summary")
    for stage in report.get("stages", []):
        gripper_stage = stage.get("last", {}).get("gripper", {})
        control_stage = stage.get("last", {}).get("gripper_ctrl", {})
        print(
            f"- {stage.get('label')}: "
            f"pos={format_optional(gripper_stage.get('position_mm'))} mm, "
            f"status={format_optional(gripper_stage.get('status_code'))}, "
            f"cmd_width={format_optional(control_stage.get('command_width_mm'))} mm, "
            f"cmd_status={format_optional(control_stage.get('command_status_code'))}"
        )

    last_stage = report.get("stages", [{}])[-1]
    gripper = last_stage.get("last", {}).get("gripper", {})
    j6_low = (
        last_stage.get("last", {})
        .get("low_spd", {})
        .get("motors", {})
        .get("6", {})
    )
    print("\nLatest SDK values")
    print(f"- gripper position mm: {format_optional(gripper.get('position_mm'))}")
    print(f"- gripper effort N.m: {format_optional(gripper.get('effort_nm'))}")
    print(f"- gripper status_code: {format_optional(gripper.get('status_code'))}")
    print(f"- J6 low-speed CAN ID: {format_optional(j6_low.get('can_id_hex'))}")
    print(f"- J6 voltage V: {format_optional(j6_low.get('voltage_v'))}")

    print("\nConclusion")
    print(f"- {analysis['conclusion']}")
    for finding in analysis["findings"]:
        print(f"- {finding}")


def format_bool(value: bool) -> str:
    return "OK" if value else "NO"


def format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


if __name__ == "__main__":
    main()

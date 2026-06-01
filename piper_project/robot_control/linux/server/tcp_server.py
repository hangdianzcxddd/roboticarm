"""Linux-side TCP server: camera XYZ command -> simulated robot pose -> move.

Run on Linux from ``piper_project``:

    python -m robot_control.linux.server.tcp_server --host 0.0.0.0 --port 5005

Default mode is dry-run and will not move the robot. Add ``--execute`` only
after confirming the generated poses are safe for the current arm setup.
"""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import replace
from typing import Any

from robot_control.shared.simulated_mapping import (
    SimulatedMappingConfig,
    camera_xyz_to_simulated_robot_pose,
    pose_to_dict,
)
from robot_control.shared.protocol import (
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose
from robot_control.linux.arm.piper_arm import PiperController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive camera XYZ over TCP and optionally move Piper arm."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--mode", choices=("P", "L"), default="P")
    parser.add_argument(
        "--command-duration-s",
        type=float,
        default=1.0,
        help="When executing, keep sending each target pose for this many seconds.",
    )
    parser.add_argument(
        "--command-period-s",
        type=float,
        default=0.02,
        help="When executing, interval between repeated pose commands.",
    )
    parser.add_argument(
        "--use-current-pose-as-base",
        action="store_true",
        help=(
            "When executing, use current end pose as the simulated mapping center. "
            "This is useful before calibration because camera XYZ only creates "
            "small offsets around the current arm pose."
        ),
    )
    parser.add_argument(
        "--print-feedback",
        action="store_true",
        help="Print arm status and end-pose feedback before and after each command.",
    )
    parser.add_argument("--base-x-mm", type=float, default=150.0)
    parser.add_argument("--base-y-mm", type=float, default=0.0)
    parser.add_argument("--base-z-mm", type=float, default=220.0)
    parser.add_argument("--scale-x", type=float, default=120.0)
    parser.add_argument("--scale-y", type=float, default=120.0)
    parser.add_argument("--scale-z", type=float, default=80.0)
    parser.add_argument("--max-delta-x-mm", type=float, default=40.0)
    parser.add_argument("--max-delta-y-mm", type=float, default=40.0)
    parser.add_argument("--max-delta-z-mm", type=float, default=30.0)
    parser.add_argument("--rx-deg", type=float, default=0.0)
    parser.add_argument("--ry-deg", type=float, default=85.0)
    parser.add_argument("--rz-deg", type=float, default=0.0)
    return parser


def mapping_config_from_args(args: argparse.Namespace) -> SimulatedMappingConfig:
    return SimulatedMappingConfig(
        base_x_mm=args.base_x_mm,
        base_y_mm=args.base_y_mm,
        base_z_mm=args.base_z_mm,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        scale_z=args.scale_z,
        max_delta_x_mm=args.max_delta_x_mm,
        max_delta_y_mm=args.max_delta_y_mm,
        max_delta_z_mm=args.max_delta_z_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )


def _feedback_to_pose(feedback: Any) -> EndPose | None:
    end_pose = getattr(feedback, "end_pose", None)
    if end_pose is None:
        return None
    axis_names = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")
    values = [getattr(end_pose, name, None) for name in axis_names]
    if any(value is None for value in values):
        return None
    return EndPose(*(float(value) / 1000.0 for value in values))


def format_pose(pose: EndPose | None) -> str:
    if pose is None:
        return "unavailable"
    return (
        f"x={pose.x_mm:.3f} y={pose.y_mm:.3f} z={pose.z_mm:.3f} "
        f"rx={pose.rx_deg:.3f} ry={pose.ry_deg:.3f} rz={pose.rz_deg:.3f}"
    )


def pose_delta_mm(before: EndPose | None, after: EndPose | None) -> str:
    if before is None or after is None:
        return "unavailable"
    return (
        f"dx={after.x_mm - before.x_mm:.3f} "
        f"dy={after.y_mm - before.y_mm:.3f} "
        f"dz={after.z_mm - before.z_mm:.3f}"
    )


def format_status(controller: PiperController) -> str:
    feedback = controller.get_status()
    status = getattr(feedback, "arm_status", None)
    if status is None:
        return "status=unavailable"
    err_status = getattr(status, "err_status", None)
    fields = (
        f"status_hz={float(getattr(feedback, 'Hz', 0.0)):.1f}",
        f"ctrl_mode={getattr(status, 'ctrl_mode', None)}",
        f"arm_status={getattr(status, 'arm_status', None)}",
        f"mode_feed={getattr(status, 'mode_feed', None)}",
        f"teach_status={getattr(status, 'teach_status', None)}",
        f"motion_status={getattr(status, 'motion_status', None)}",
        f"err_code={getattr(status, 'err_code', None)}",
    )
    problems = []
    if err_status is not None:
        for name in (
            "joint_1_angle_limit",
            "joint_2_angle_limit",
            "joint_3_angle_limit",
            "joint_4_angle_limit",
            "joint_5_angle_limit",
            "joint_6_angle_limit",
            "communication_status_joint_1",
            "communication_status_joint_2",
            "communication_status_joint_3",
            "communication_status_joint_4",
            "communication_status_joint_5",
            "communication_status_joint_6",
        ):
            if getattr(err_status, name, False):
                problems.append(name)
    error_text = ",".join(problems) if problems else "none"
    return " ".join(fields) + f" err_flags={error_text}"


def read_feedback_pose(controller: PiperController) -> EndPose | None:
    return _feedback_to_pose(controller.get_end_pose())


def read_current_end_pose(controller: PiperController, timeout_s: float = 1.0) -> EndPose:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        feedback = controller.get_end_pose()
        pose = _feedback_to_pose(feedback)
        if (
            pose is not None
            and (
                float(getattr(feedback, "time_stamp", 0.0)) > 0.0
                or float(getattr(feedback, "Hz", 0.0)) > 0.0
            )
        ):
            return pose
        time.sleep(0.02)
    raise RuntimeError("failed to read fresh current Piper end pose")


def execute_pose_for_duration(
    controller: PiperController,
    pose: EndPose,
    speed: int,
    mode: str,
    duration_s: float,
    period_s: float,
) -> int:
    if duration_s < 0:
        raise ValueError("--command-duration-s must be >= 0")
    if period_s <= 0:
        raise ValueError("--command-period-s must be > 0")

    deadline = time.monotonic() + duration_s
    send_count = 0
    while send_count == 0 or time.monotonic() < deadline:
        controller.move_to_pose(pose, speed_percent=speed, move_mode=mode)
        send_count += 1
        if time.monotonic() >= deadline:
            break
        time.sleep(period_s)
    return send_count


def validate_or_execute_pose(
    pose: EndPose,
    execute: bool,
    speed: int,
    mode: str,
    controller: PiperController | None,
    command_duration_s: float,
    command_period_s: float,
    print_feedback: bool,
) -> None:
    validator = controller or PiperController()
    validator.validate_pose(pose)
    if not execute:
        return

    if controller is None:
        raise RuntimeError("execute=True requires a connected PiperController")

    before_pose = read_feedback_pose(controller) if print_feedback else None
    if print_feedback:
        print(f"Before command: {format_status(controller)}", flush=True)
        print(f"Before pose: {format_pose(before_pose)}", flush=True)

    send_count = execute_pose_for_duration(
        controller,
        pose,
        speed=speed,
        mode=mode,
        duration_s=command_duration_s,
        period_s=command_period_s,
    )
    print(f"Sent pose command {send_count} times", flush=True)
    if print_feedback:
        time.sleep(0.1)
        after_pose = read_feedback_pose(controller)
        print(f"After command: {format_status(controller)}", flush=True)
        print(f"After pose: {format_pose(after_pose)}", flush=True)
        print(f"Feedback pose delta: {pose_delta_mm(before_pose, after_pose)}", flush=True)


def handle_message(
    message: dict[str, Any],
    mapping_config: SimulatedMappingConfig,
    execute: bool,
    speed: int,
    mode: str,
    controller: PiperController | None = None,
    command_duration_s: float = 1.0,
    command_period_s: float = 0.02,
    print_feedback: bool = False,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    pose = camera_xyz_to_simulated_robot_pose(
        command.x_m,
        command.y_m,
        command.z_m,
        mapping_config,
    )
    validate_or_execute_pose(
        pose,
        execute=execute,
        speed=speed,
        mode=mode,
        controller=controller,
        command_duration_s=command_duration_s,
        command_period_s=command_period_s,
        print_feedback=print_feedback,
    )
    print(
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"pose={pose}",
        flush=True,
    )
    return make_ack_message(command.command_id, pose_to_dict(pose), executed=execute)


def serve(args: argparse.Namespace) -> None:
    mapping_config = mapping_config_from_args(args)
    controller: PiperController | None = None
    if args.use_current_pose_as_base and not args.execute:
        raise ValueError("--use-current-pose-as-base requires --execute")

    if args.execute:
        controller = PiperController()
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        if args.use_current_pose_as_base:
            current_pose = read_current_end_pose(controller)
            mapping_config = replace(
                mapping_config,
                base_x_mm=current_pose.x_mm,
                base_y_mm=current_pose.y_mm,
                base_z_mm=current_pose.z_mm,
                rx_deg=current_pose.rx_deg,
                ry_deg=current_pose.ry_deg,
                rz_deg=current_pose.rz_deg,
            )
            print(f"Using current end pose as mapping base: {current_pose}", flush=True)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"Listening on {args.host}:{args.port}, "
                f"execute={args.execute}, speed={args.speed}, mode={args.mode}, "
                f"command_duration_s={args.command_duration_s}",
                flush=True,
            )
            print(f"Robot safety config: {ROBOT_CONFIG}", flush=True)

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
                                    mapping_config,
                                    execute=args.execute,
                                    speed=args.speed,
                                    mode=args.mode,
                                    controller=controller,
                                    command_duration_s=args.command_duration_s,
                                    command_period_s=args.command_period_s,
                                    print_feedback=args.print_feedback,
                                )
                            except Exception as exc:  # noqa: BLE001 - sent to remote client
                                response = make_error_message(command_id, str(exc))
                                print(f"Error: {exc}", flush=True)
                            send_json_line(conn, response)
    finally:
        if controller is not None:
            controller.disconnect()


def main() -> None:
    serve(build_parser().parse_args())


if __name__ == "__main__":
    main()

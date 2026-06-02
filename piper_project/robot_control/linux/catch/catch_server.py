"""Linux-side click-to-catch TCP server."""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import replace
from typing import Any, Sequence

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.catch.tcp_offset import (
    OFFICIAL_GRIPPER_TCP_OFFSET_MM,
    CatchPlan,
    build_catch_plan,
    tcp_pose_for_j6_pose,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.shared.simulated_mapping import (
    SimulatedMappingConfig,
    camera_xyz_to_simulated_robot_pose,
    pose_to_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive camera XYZ over TCP and run a gripper catch sequence."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--approach-mode", choices=("P", "L"), default="P")
    parser.add_argument("--grasp-mode", choices=("P", "L"), default="L")
    parser.add_argument("--lift-mode", choices=("P", "L"), default="L")
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument(
        "--use-current-pose-as-base",
        action="store_true",
        help="When executing, use the current gripper TCP pose as the mapping center.",
    )
    parser.add_argument("--print-feedback", action="store_true")
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
    parser.add_argument("--tcp-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--tcp-offset-z-mm",
        type=float,
        default=OFFICIAL_GRIPPER_TCP_OFFSET_MM[2],
        help="Official Piper gripper TCP offset is 145 mm, i.e. 0.145 m.",
    )
    parser.add_argument("--target-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--approach-height-mm", type=float, default=50.0)
    parser.add_argument("--lift-height-mm", type=float, default=50.0)
    parser.add_argument("--open-width-mm", type=float, default=70.0)
    parser.add_argument("--close-width-mm", type=float, default=0.0)
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
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


def tcp_offset_from_args(args: argparse.Namespace) -> tuple[float, float, float]:
    return (
        float(args.tcp_offset_x_mm),
        float(args.tcp_offset_y_mm),
        float(args.tcp_offset_z_mm),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.use_current_pose_as_base and not args.execute:
        raise ValueError("--use-current-pose-as-base requires --execute")
    if args.command_duration_s < 0:
        raise ValueError("--command-duration-s must be >= 0")
    if args.command_period_s <= 0:
        raise ValueError("--command-period-s must be > 0")
    if args.approach_height_mm < 0:
        raise ValueError("--approach-height-mm must be >= 0")
    if args.lift_height_mm < 0:
        raise ValueError("--lift-height-mm must be >= 0")
    if args.open_width_mm < 0 or args.close_width_mm < 0:
        raise ValueError("gripper widths must be >= 0")
    if args.gripper_settle_s < 0:
        raise ValueError("--gripper-settle-s must be >= 0")
    if not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be in [0, 5]")


def feedback_to_pose(feedback: Any) -> EndPose | None:
    end_pose = getattr(feedback, "end_pose", None)
    if end_pose is None:
        return None
    axis_names = ("X_axis", "Y_axis", "Z_axis", "RX_axis", "RY_axis", "RZ_axis")
    values = [getattr(end_pose, name, None) for name in axis_names]
    if any(value is None for value in values):
        return None
    return EndPose(*(float(value) / 1000.0 for value in values))


def read_current_j6_pose(controller: PiperController, timeout_s: float = 1.0) -> EndPose:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        feedback = controller.get_end_pose()
        pose = feedback_to_pose(feedback)
        if (
            pose is not None
            and (
                float(getattr(feedback, "time_stamp", 0.0)) > 0.0
                or float(getattr(feedback, "Hz", 0.0)) > 0.0
            )
        ):
            return pose
        time.sleep(0.02)
    raise RuntimeError("failed to read fresh current Piper J6 pose")


def execute_pose_for_duration(
    controller: PiperController,
    pose: EndPose,
    speed: int,
    mode: str,
    duration_s: float,
    period_s: float,
) -> int:
    deadline = time.monotonic() + duration_s
    send_count = 0
    while send_count == 0 or time.monotonic() < deadline:
        controller.move_to_pose(pose, speed_percent=speed, move_mode=mode)
        send_count += 1
        if time.monotonic() >= deadline:
            break
        time.sleep(period_s)
    return send_count


def validate_plan(controller: PiperController, plan: CatchPlan) -> None:
    controller.validate_pose(plan.approach_j6_pose)
    controller.validate_pose(plan.grasp_j6_pose)
    controller.validate_pose(plan.lift_j6_pose)


def execute_catch_plan(
    controller: PiperController,
    plan: CatchPlan,
    args: argparse.Namespace,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if args.print_feedback:
        print(f"Before catch J6: {format_pose(feedback_to_pose(controller.get_end_pose()))}")

    if not args.no_gripper:
        if args.clear_gripper_error:
            controller.set_gripper(0.0, args.effort_nm, enable=True, clear_error=True)
            time.sleep(args.gripper_settle_s)
        if not args.no_pre_open:
            controller.set_gripper(args.open_width_mm, args.effort_nm, enable=True)
            time.sleep(args.gripper_settle_s)

    counts["approach"] = execute_pose_for_duration(
        controller,
        plan.approach_j6_pose,
        args.speed,
        args.approach_mode,
        args.command_duration_s,
        args.command_period_s,
    )
    counts["grasp"] = execute_pose_for_duration(
        controller,
        plan.grasp_j6_pose,
        args.speed,
        args.grasp_mode,
        args.command_duration_s,
        args.command_period_s,
    )

    if not args.no_gripper:
        controller.set_gripper(args.close_width_mm, args.effort_nm, enable=True)
        time.sleep(args.gripper_settle_s)

    counts["lift"] = execute_pose_for_duration(
        controller,
        plan.lift_j6_pose,
        args.speed,
        args.lift_mode,
        args.command_duration_s,
        args.command_period_s,
    )

    if args.print_feedback:
        time.sleep(0.1)
        print(f"After catch J6: {format_pose(feedback_to_pose(controller.get_end_pose()))}")
    return counts


def handle_message(
    message: dict[str, Any],
    mapping_config: SimulatedMappingConfig,
    tcp_offset_j6_mm: Sequence[float],
    execute: bool,
    controller: PiperController | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    target_tcp_pose = camera_xyz_to_simulated_robot_pose(
        command.x_m,
        command.y_m,
        command.z_m,
        mapping_config,
    )
    plan = build_catch_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        approach_height_mm=args.approach_height_mm,
        lift_height_mm=args.lift_height_mm,
        target_z_offset_mm=args.target_z_offset_mm,
    )

    validator = controller or PiperController()
    validate_plan(validator, plan)
    send_counts: dict[str, int] = {}
    if execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        send_counts = execute_catch_plan(controller, plan, args)

    print(
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"target_tcp={format_pose(plan.target_tcp_pose)} "
        f"grasp_j6={format_pose(plan.grasp_j6_pose)}",
        flush=True,
    )
    response = make_ack_message(
        command.command_id,
        pose_to_dict(plan.grasp_j6_pose),
        executed=execute,
    )
    response["payload"].update(
        {
            "target_tcp_pose": pose_to_dict(plan.target_tcp_pose),
            "approach_j6_pose": pose_to_dict(plan.approach_j6_pose),
            "grasp_j6_pose": pose_to_dict(plan.grasp_j6_pose),
            "lift_j6_pose": pose_to_dict(plan.lift_j6_pose),
            "tcp_offset_j6_mm": {
                "x_mm": float(tcp_offset_j6_mm[0]),
                "y_mm": float(tcp_offset_j6_mm[1]),
                "z_mm": float(tcp_offset_j6_mm[2]),
            },
            "catch": {
                "send_counts": send_counts,
                "open_width_mm": args.open_width_mm,
                "close_width_mm": args.close_width_mm,
                "effort_nm": args.effort_nm,
                "gripper_enabled": not args.no_gripper,
            },
        }
    )
    return response


def serve(args: argparse.Namespace) -> None:
    validate_args(args)
    mapping_config = mapping_config_from_args(args)
    tcp_offset_j6_mm = tcp_offset_from_args(args)
    controller: PiperController | None = None

    if args.execute:
        config = replace(ROBOT_CONFIG, can_name=args.can)
        controller = PiperController(config=config)
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        if args.use_current_pose_as_base:
            current_j6_pose = read_current_j6_pose(controller)
            current_tcp_pose = tcp_pose_for_j6_pose(current_j6_pose, tcp_offset_j6_mm)
            mapping_config = replace(
                mapping_config,
                base_x_mm=current_tcp_pose.x_mm,
                base_y_mm=current_tcp_pose.y_mm,
                base_z_mm=current_tcp_pose.z_mm,
                rx_deg=current_tcp_pose.rx_deg,
                ry_deg=current_tcp_pose.ry_deg,
                rz_deg=current_tcp_pose.rz_deg,
            )
            print(f"Using current TCP pose as mapping base: {current_tcp_pose}", flush=True)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"Listening on {args.host}:{args.port}, execute={args.execute}, "
                f"speed={args.speed}, tcp_offset_j6_mm={tcp_offset_j6_mm}",
                flush=True,
            )
            print(f"Catch args: {args}", flush=True)

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
                                    tcp_offset_j6_mm=tcp_offset_j6_mm,
                                    execute=args.execute,
                                    controller=controller,
                                    args=args,
                                )
                            except Exception as exc:  # noqa: BLE001 - sent to client
                                response = make_error_message(command_id, str(exc))
                                print(f"Error: {exc}", flush=True)
                            send_json_line(conn, response)
    finally:
        if controller is not None:
            controller.disconnect()


def format_pose(pose: EndPose | None) -> str:
    if pose is None:
        return "unavailable"
    return (
        f"x={pose.x_mm:.3f} y={pose.y_mm:.3f} z={pose.z_mm:.3f} "
        f"rx={pose.rx_deg:.3f} ry={pose.ry_deg:.3f} rz={pose.rz_deg:.3f}"
    )


def main() -> None:
    serve(build_parser().parse_args())


if __name__ == "__main__":
    main()

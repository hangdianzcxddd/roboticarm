"""Linux-side click-to-catch TCP server."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.catch.tcp_offset import (
    OFFICIAL_GRIPPER_TCP_OFFSET_MM,
    CatchPlan,
    build_catch_plan,
    tcp_pose_for_j6_pose,
)
from robot_control.linux.catch.set_gripper_zero import set_gripper_zero
from robot_control.linux.catch.hand_eye_mapping import (
    camera_xyz_to_hand_eye_robot_pose,
    load_camera_to_base_transform,
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


DEFAULT_HAND_EYE_RESULT = (
    Path(__file__).resolve().parents[1] / "hand_eye" / "hand_eye_result.json"
)


@dataclass(frozen=True)
class PlacePlan:
    place_tcp_pose: EndPose
    approach_tcp_pose: EndPose
    retreat_tcp_pose: EndPose
    place_j6_pose: EndPose
    approach_j6_pose: EndPose
    retreat_j6_pose: EndPose


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
    parser.add_argument(
        "--hand-eye-result",
        type=Path,
        help=(
            "Use an eye_to_hand hand_eye_result.json camera->base transform. "
            f"If omitted, keep using the temporary simulated mapping. Default "
            f"path when supplied without a value is not supported; use "
            f"{DEFAULT_HAND_EYE_RESULT} explicitly."
        ),
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
    parser.add_argument("--place-x-mm", type=float, default=0.0)
    parser.add_argument("--place-y-mm", type=float, default=150.0)
    parser.add_argument("--place-z-mm", type=float, default=150.0)
    parser.add_argument("--place-rx-deg", type=float)
    parser.add_argument("--place-ry-deg", type=float)
    parser.add_argument("--place-rz-deg", type=float)
    parser.add_argument("--place-approach-height-mm", type=float, default=50.0)
    parser.add_argument(
        "--open-command-mm",
        "--open-width-mm",
        dest="open_command_mm",
        type=float,
        default=0.0,
        help="Gripper open command. With open-zero calibration, 0 mm is fully open.",
    )
    parser.add_argument(
        "--grasp-command-mm",
        "--close-command-mm",
        "--close-width-mm",
        dest="grasp_command_mm",
        type=float,
        default=-100.0,
        help=(
            "Gripper close command. With open-zero calibration, negative values "
            "move toward closing; default -100 mm."
        ),
    )
    parser.add_argument(
        "--release-command-mm",
        type=float,
        help="Gripper release command; defaults to --open-command-mm.",
    )
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument("--grasp-effort-nm", type=float)
    parser.add_argument("--release-effort-nm", type=float)
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
    parser.add_argument(
        "--set-open-zero-on-start",
        action="store_true",
        help=(
            "Set the current physical gripper position as zero at startup. "
            "Before using this, manually make the gripper fully open at 100 mm."
        ),
    )
    parser.add_argument("--zero-settle-s", type=float, default=1.5)
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
    if args.use_current_pose_as_base and args.hand_eye_result is not None:
        raise ValueError("--use-current-pose-as-base is only for simulated mapping")
    if args.command_duration_s < 0:
        raise ValueError("--command-duration-s must be >= 0")
    if args.command_period_s <= 0:
        raise ValueError("--command-period-s must be > 0")
    if args.approach_height_mm < 0:
        raise ValueError("--approach-height-mm must be >= 0")
    if args.lift_height_mm < 0:
        raise ValueError("--lift-height-mm must be >= 0")
    if args.place_approach_height_mm < 0:
        raise ValueError("--place-approach-height-mm must be >= 0")
    for option_name, value in (
        ("--open-command-mm", args.open_command_mm),
        ("--grasp-command-mm", args.grasp_command_mm),
        ("--release-command-mm", release_command_mm(args)),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{option_name} must be finite")
    if args.gripper_settle_s < 0:
        raise ValueError("--gripper-settle-s must be >= 0")
    if args.zero_settle_s < 0:
        raise ValueError("--zero-settle-s must be >= 0")
    if not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be in [0, 5]")
    for option_name, value in (
        ("--grasp-effort-nm", args.grasp_effort_nm),
        ("--release-effort-nm", args.release_effort_nm),
    ):
        if value is not None and not 0 <= value <= 5:
            raise ValueError(f"{option_name} must be in [0, 5]")


def grasp_effort_nm(args: argparse.Namespace) -> float:
    return float(args.effort_nm if args.grasp_effort_nm is None else args.grasp_effort_nm)


def release_effort_nm(args: argparse.Namespace) -> float:
    return float(
        args.effort_nm if args.release_effort_nm is None else args.release_effort_nm
    )


def release_command_mm(args: argparse.Namespace) -> float:
    return float(
        args.open_command_mm
        if args.release_command_mm is None
        else args.release_command_mm
    )


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


def build_place_plan(
    target_tcp_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float],
    args: argparse.Namespace,
) -> PlacePlan:
    rx_deg = target_tcp_pose.rx_deg if args.place_rx_deg is None else args.place_rx_deg
    ry_deg = target_tcp_pose.ry_deg if args.place_ry_deg is None else args.place_ry_deg
    rz_deg = target_tcp_pose.rz_deg if args.place_rz_deg is None else args.place_rz_deg
    place_tcp_pose = EndPose(
        args.place_x_mm,
        args.place_y_mm,
        args.place_z_mm,
        rx_deg,
        ry_deg,
        rz_deg,
    )
    approach_tcp_pose = EndPose(
        place_tcp_pose.x_mm,
        place_tcp_pose.y_mm,
        place_tcp_pose.z_mm + args.place_approach_height_mm,
        place_tcp_pose.rx_deg,
        place_tcp_pose.ry_deg,
        place_tcp_pose.rz_deg,
    )
    return PlacePlan(
        place_tcp_pose=place_tcp_pose,
        approach_tcp_pose=approach_tcp_pose,
        retreat_tcp_pose=approach_tcp_pose,
        place_j6_pose=build_catch_j6_pose(place_tcp_pose, offset_j6_tcp_mm),
        approach_j6_pose=build_catch_j6_pose(approach_tcp_pose, offset_j6_tcp_mm),
        retreat_j6_pose=build_catch_j6_pose(approach_tcp_pose, offset_j6_tcp_mm),
    )


def build_catch_j6_pose(
    tcp_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float],
) -> EndPose:
    from robot_control.linux.catch.tcp_offset import j6_pose_for_tcp_pose

    return j6_pose_for_tcp_pose(tcp_pose, offset_j6_tcp_mm)


def validate_plan(
    controller: PiperController,
    catch_plan: CatchPlan,
    place_plan: PlacePlan,
) -> None:
    controller.validate_pose(catch_plan.approach_j6_pose)
    controller.validate_pose(catch_plan.grasp_j6_pose)
    controller.validate_pose(catch_plan.lift_j6_pose)
    controller.validate_pose(place_plan.approach_j6_pose)
    controller.validate_pose(place_plan.place_j6_pose)
    controller.validate_pose(place_plan.retreat_j6_pose)


def maybe_set_open_zero(controller: PiperController, args: argparse.Namespace) -> None:
    if args.no_gripper or not args.set_open_zero_on_start:
        return
    print(
        "Setting gripper zero at current physical fully-open position "
        f"(command={args.open_command_mm:.3f} mm).",
        flush=True,
    )
    set_gripper_zero(
        controller,
        zero_command_mm=args.open_command_mm,
        effort_nm=release_effort_nm(args),
        settle_s=args.zero_settle_s,
    )


def execute_catch_plan(
    controller: PiperController,
    catch_plan: CatchPlan,
    place_plan: PlacePlan,
    args: argparse.Namespace,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if args.print_feedback:
        print(f"Before catch J6: {format_pose(feedback_to_pose(controller.get_end_pose()))}")

    if not args.no_gripper:
        if args.clear_gripper_error:
            controller.set_gripper(
                args.open_command_mm,
                release_effort_nm(args),
                enable=True,
                clear_error=True,
            )
            time.sleep(args.gripper_settle_s)
        if not args.no_pre_open:
            controller.set_gripper(
                args.open_command_mm,
                release_effort_nm(args),
                enable=True,
            )
            time.sleep(args.gripper_settle_s)

    counts["approach"] = execute_pose_for_duration(
        controller,
        catch_plan.approach_j6_pose,
        args.speed,
        args.approach_mode,
        args.command_duration_s,
        args.command_period_s,
    )
    counts["grasp"] = execute_pose_for_duration(
        controller,
        catch_plan.grasp_j6_pose,
        args.speed,
        args.grasp_mode,
        args.command_duration_s,
        args.command_period_s,
    )

    if not args.no_gripper:
        controller.set_gripper(
            args.grasp_command_mm,
            grasp_effort_nm(args),
            enable=True,
        )
        time.sleep(args.gripper_settle_s)

    counts["lift"] = execute_pose_for_duration(
        controller,
        catch_plan.lift_j6_pose,
        args.speed,
        args.lift_mode,
        args.command_duration_s,
        args.command_period_s,
    )
    counts["place_approach"] = execute_pose_for_duration(
        controller,
        place_plan.approach_j6_pose,
        args.speed,
        args.approach_mode,
        args.command_duration_s,
        args.command_period_s,
    )
    counts["place"] = execute_pose_for_duration(
        controller,
        place_plan.place_j6_pose,
        args.speed,
        args.grasp_mode,
        args.command_duration_s,
        args.command_period_s,
    )

    if not args.no_gripper:
        controller.set_gripper(
            release_command_mm(args),
            release_effort_nm(args),
            enable=True,
        )
        time.sleep(args.gripper_settle_s)

    counts["place_retreat"] = execute_pose_for_duration(
        controller,
        place_plan.retreat_j6_pose,
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
    hand_eye_camera_to_base: Any | None = None,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    if hand_eye_camera_to_base is None:
        target_tcp_pose = camera_xyz_to_simulated_robot_pose(
            command.x_m,
            command.y_m,
            command.z_m,
            mapping_config,
        )
        mapping_source = "simulated"
    else:
        target_tcp_pose = camera_xyz_to_hand_eye_robot_pose(
            command.x_m,
            command.y_m,
            command.z_m,
            hand_eye_camera_to_base,
            rx_deg=args.rx_deg,
            ry_deg=args.ry_deg,
            rz_deg=args.rz_deg,
        )
        mapping_source = "hand_eye"
    plan = build_catch_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        approach_height_mm=args.approach_height_mm,
        lift_height_mm=args.lift_height_mm,
        target_z_offset_mm=args.target_z_offset_mm,
    )
    place_plan = build_place_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        args=args,
    )

    validator = controller or PiperController()
    validate_plan(validator, plan, place_plan)
    send_counts: dict[str, int] = {}
    if execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        send_counts = execute_catch_plan(controller, plan, place_plan, args)

    print(
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"mapping={mapping_source} "
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
            "place_tcp_pose": pose_to_dict(place_plan.place_tcp_pose),
            "place_j6_pose": pose_to_dict(place_plan.place_j6_pose),
            "place_approach_j6_pose": pose_to_dict(place_plan.approach_j6_pose),
            "tcp_offset_j6_mm": {
                "x_mm": float(tcp_offset_j6_mm[0]),
                "y_mm": float(tcp_offset_j6_mm[1]),
                "z_mm": float(tcp_offset_j6_mm[2]),
            },
            "mapping_source": mapping_source,
            "catch": {
                "send_counts": send_counts,
                "open_command_mm": args.open_command_mm,
                "grasp_command_mm": args.grasp_command_mm,
                "release_command_mm": release_command_mm(args),
                "grasp_effort_nm": grasp_effort_nm(args),
                "release_effort_nm": release_effort_nm(args),
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
    hand_eye_camera_to_base = None
    if args.hand_eye_result is not None:
        hand_eye_camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
        print(
            f"Using hand-eye camera->base transform: {args.hand_eye_result}",
            flush=True,
        )

    if args.execute:
        config = replace(ROBOT_CONFIG, can_name=args.can)
        controller = PiperController(config=config)
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        maybe_set_open_zero(controller, args)
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
                                    hand_eye_camera_to_base=hand_eye_camera_to_base,
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

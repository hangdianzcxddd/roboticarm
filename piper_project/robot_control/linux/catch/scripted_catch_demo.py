"""Run a fixed catch/place sequence without camera or hand-eye input."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Sequence

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.catch.catch_server import (
    PlacePlan,
    build_catch_j6_pose,
    execute_pose_for_duration,
    feedback_to_pose,
    format_pose,
    grasp_effort_nm,
    maybe_set_open_zero,
    pose_errors,
    read_current_j6_pose,
    release_command_mm,
    release_effort_nm,
    tcp_offset_from_args,
    validate_args,
)
from robot_control.linux.catch.tcp_offset import (
    OFFICIAL_GRIPPER_TCP_OFFSET_MM,
    build_catch_plan,
    tcp_pose_for_j6_pose,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import EndPose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a scripted top-down catch/place demo from fixed robot poses. "
            "This bypasses camera clicks and hand-eye calibration."
        )
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--no-enable",
        action="store_true",
        help="connect without calling EnablePiper; use only when the arm is already enabled",
    )
    parser.add_argument("--speed", type=int, default=5)
    parser.add_argument("--approach-mode", choices=("P", "L"), default="P")
    parser.add_argument("--grasp-mode", choices=("P", "L"), default="L")
    parser.add_argument("--lift-mode", choices=("P", "L"), default="L")
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=12.0)
    parser.add_argument(
        "--continue-on-waypoint-error",
        action="store_true",
        help="continue the scripted demo when a waypoint is close but not fully reached",
    )
    parser.add_argument("--no-wait-waypoints", action="store_true")
    parser.add_argument("--print-feedback", action="store_true")
    parser.add_argument("--tcp-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--tcp-offset-z-mm",
        type=float,
        default=OFFICIAL_GRIPPER_TCP_OFFSET_MM[2],
    )
    parser.add_argument(
        "--target-x-mm",
        type=float,
        help="fixed grasp TCP X; defaults to current TCP X",
    )
    parser.add_argument(
        "--target-y-mm",
        type=float,
        help="fixed grasp TCP Y; defaults to current TCP Y",
    )
    parser.add_argument(
        "--target-z-mm",
        type=float,
        help="fixed grasp TCP Z; defaults to current TCP Z",
    )
    parser.add_argument("--rx-deg", type=float, default=180.0)
    parser.add_argument("--ry-deg", type=float, default=0.0)
    parser.add_argument(
        "--rz-deg",
        type=float,
        help="fixed grasp TCP RZ; defaults to current TCP RZ",
    )
    parser.add_argument("--target-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--approach-height-mm", type=float, default=40.0)
    parser.add_argument("--descend-mm", type=float, default=30.0)
    parser.add_argument("--min-target-z-mm", type=float, default=0.0)
    parser.add_argument("--lift-height-mm", type=float, default=50.0)
    parser.add_argument("--place-x-mm", type=float)
    parser.add_argument("--place-y-mm", type=float)
    parser.add_argument("--place-dx-mm", type=float, default=0.0)
    parser.add_argument("--place-dy-mm", type=float, default=30.0)
    parser.add_argument("--place-z-mm", type=float, default=80.0)
    parser.add_argument("--place-rx-deg", type=float)
    parser.add_argument("--place-ry-deg", type=float)
    parser.add_argument("--place-rz-deg", type=float)
    parser.add_argument("--place-approach-height-mm", type=float, default=40.0)
    parser.add_argument("--return-current-after-release", action="store_true")
    parser.add_argument("--open-command-mm", type=float, default=0.0)
    parser.add_argument("--grasp-command-mm", type=float, default=-100.0)
    parser.add_argument("--release-command-mm", type=float)
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument("--grasp-effort-nm", type=float)
    parser.add_argument("--release-effort-nm", type=float)
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
    parser.add_argument("--set-open-zero-on-start", action="store_true")
    parser.add_argument("--zero-settle-s", type=float, default=1.5)
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    return parser


def validate_demo_args(args: argparse.Namespace) -> None:
    args.use_current_pose_as_base = False
    args.hand_eye_result = None
    args.use_current_orientation = False
    args.place_rx_deg = args.place_rx_deg
    args.place_ry_deg = args.place_ry_deg
    args.place_rz_deg = args.place_rz_deg
    args.stage_height_mm = 0.0
    validate_args(args)
    if args.descend_mm < 0:
        raise ValueError("--descend-mm must be >= 0")
    if args.min_target_z_mm < 0:
        raise ValueError("--min-target-z-mm must be >= 0")


def target_pose_from_args(
    args: argparse.Namespace,
    current_j6_pose: EndPose,
    tcp_offset_j6_mm: Sequence[float],
) -> EndPose:
    rz_deg = current_j6_pose.rz_deg if args.rz_deg is None else args.rz_deg
    desired_grasp_j6_pose = EndPose(
        x_mm=current_j6_pose.x_mm,
        y_mm=current_j6_pose.y_mm,
        z_mm=current_j6_pose.z_mm - args.descend_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=rz_deg,
    )
    default_target_tcp = tcp_pose_for_j6_pose(
        desired_grasp_j6_pose,
        tcp_offset_j6_mm,
    )
    if default_target_tcp.z_mm < args.min_target_z_mm:
        desired_grasp_j6_pose = replace(
            desired_grasp_j6_pose,
            z_mm=desired_grasp_j6_pose.z_mm
            + args.min_target_z_mm
            - default_target_tcp.z_mm,
        )
        default_target_tcp = tcp_pose_for_j6_pose(
            desired_grasp_j6_pose,
            tcp_offset_j6_mm,
        )
    return EndPose(
        x_mm=default_target_tcp.x_mm if args.target_x_mm is None else args.target_x_mm,
        y_mm=default_target_tcp.y_mm if args.target_y_mm is None else args.target_y_mm,
        z_mm=default_target_tcp.z_mm if args.target_z_mm is None else args.target_z_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=rz_deg,
    )


def build_demo_place_plan(
    target_tcp_pose: EndPose,
    offset_j6_tcp_mm: Sequence[float],
    args: argparse.Namespace,
) -> PlacePlan:
    rx_deg = target_tcp_pose.rx_deg if args.place_rx_deg is None else args.place_rx_deg
    ry_deg = target_tcp_pose.ry_deg if args.place_ry_deg is None else args.place_ry_deg
    rz_deg = target_tcp_pose.rz_deg if args.place_rz_deg is None else args.place_rz_deg
    place_tcp_pose = EndPose(
        (
            target_tcp_pose.x_mm + args.place_dx_mm
            if args.place_x_mm is None
            else args.place_x_mm
        ),
        (
            target_tcp_pose.y_mm + args.place_dy_mm
            if args.place_y_mm is None
            else args.place_y_mm
        ),
        args.place_z_mm,
        rx_deg,
        ry_deg,
        rz_deg,
    )
    approach_tcp_pose = replace(
        place_tcp_pose,
        z_mm=place_tcp_pose.z_mm + args.place_approach_height_mm,
    )
    return PlacePlan(
        place_tcp_pose=place_tcp_pose,
        approach_tcp_pose=approach_tcp_pose,
        retreat_tcp_pose=approach_tcp_pose,
        place_j6_pose=build_catch_j6_pose(place_tcp_pose, offset_j6_tcp_mm),
        approach_j6_pose=build_catch_j6_pose(approach_tcp_pose, offset_j6_tcp_mm),
        retreat_j6_pose=build_catch_j6_pose(approach_tcp_pose, offset_j6_tcp_mm),
    )


def validate_plan(
    controller: PiperController,
    poses: Sequence[tuple[str, EndPose]],
) -> None:
    for label, pose in poses:
        controller.validate_pose(pose)
        print(f"{label}: {format_pose(pose)}", flush=True)


def execute_demo_waypoint(
    controller: PiperController,
    label: str,
    pose: EndPose,
    speed: int,
    mode: str,
    args: argparse.Namespace,
) -> int:
    send_count = execute_pose_for_duration(
        controller,
        pose,
        speed,
        mode,
        args.command_duration_s,
        args.command_period_s,
    )
    if args.no_wait_waypoints:
        return send_count
    try:
        from robot_control.linux.catch.catch_server import wait_until_pose_reached

        wait_until_pose_reached(
            controller,
            label,
            pose,
            timeout_s=args.motion_timeout_s,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
        )
    except RuntimeError as exc:
        if not args.continue_on_waypoint_error:
            raise
        actual_pose = feedback_to_pose(controller.get_end_pose())
        position_error, orientation_error = (
            pose_errors(pose, actual_pose)
            if actual_pose is not None
            else (float("inf"), float("inf"))
        )
        print(
            f"Warning: {exc}; continuing because --continue-on-waypoint-error "
            f"is set (pos_err={position_error:.3f} mm, "
            f"ori_err={orientation_error:.3f} deg)",
            flush=True,
        )
    return send_count


def execute_demo(
    controller: PiperController,
    target_tcp_pose: EndPose,
    tcp_offset_j6_mm: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, int]:
    catch_plan = build_catch_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        approach_height_mm=args.approach_height_mm,
        lift_height_mm=args.lift_height_mm,
        target_z_offset_mm=args.target_z_offset_mm,
    )
    place_plan = build_demo_place_plan(target_tcp_pose, tcp_offset_j6_mm, args)
    retreat_pose = feedback_to_pose(controller.get_end_pose())
    named_poses = [
        ("approach_j6", catch_plan.approach_j6_pose),
        ("grasp_j6", catch_plan.grasp_j6_pose),
        ("lift_j6", catch_plan.lift_j6_pose),
        ("place_approach_j6", place_plan.approach_j6_pose),
        ("place_j6", place_plan.place_j6_pose),
        ("place_retreat_j6", place_plan.retreat_j6_pose),
    ]
    if args.return_current_after_release and retreat_pose is not None:
        named_poses.append(("return_j6", retreat_pose))
    validate_plan(controller, named_poses)
    if not args.execute:
        return {}

    counts: dict[str, int] = {}
    if args.print_feedback:
        print(f"Before demo J6: {format_pose(retreat_pose)}", flush=True)

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

    counts["approach"] = execute_demo_waypoint(
        controller,
        "approach",
        catch_plan.approach_j6_pose,
        args.speed,
        args.approach_mode,
        args,
    )
    counts["grasp"] = execute_demo_waypoint(
        controller,
        "grasp",
        catch_plan.grasp_j6_pose,
        args.speed,
        args.grasp_mode,
        args,
    )
    if not args.no_gripper:
        controller.set_gripper(
            args.grasp_command_mm,
            grasp_effort_nm(args),
            enable=True,
        )
        time.sleep(args.gripper_settle_s)

    counts["lift"] = execute_demo_waypoint(
        controller,
        "lift",
        catch_plan.lift_j6_pose,
        args.speed,
        args.lift_mode,
        args,
    )
    counts["place_approach"] = execute_demo_waypoint(
        controller,
        "place_approach",
        place_plan.approach_j6_pose,
        args.speed,
        args.approach_mode,
        args,
    )
    counts["place"] = execute_demo_waypoint(
        controller,
        "place",
        place_plan.place_j6_pose,
        args.speed,
        args.grasp_mode,
        args,
    )
    if not args.no_gripper:
        controller.set_gripper(
            release_command_mm(args),
            release_effort_nm(args),
            enable=True,
        )
        time.sleep(args.gripper_settle_s)

    counts["place_retreat"] = execute_demo_waypoint(
        controller,
        "place_retreat",
        place_plan.retreat_j6_pose,
        args.speed,
        args.lift_mode,
        args,
    )
    if args.return_current_after_release and retreat_pose is not None:
        counts["return"] = execute_demo_waypoint(
            controller,
            "return",
            retreat_pose,
            args.speed,
            args.approach_mode,
            args,
        )

    if args.print_feedback:
        print(f"After demo J6: {format_pose(feedback_to_pose(controller.get_end_pose()))}")
        print(f"Demo send counts: {counts}")
    return counts


def main() -> None:
    args = build_parser().parse_args()
    validate_demo_args(args)
    tcp_offset_j6_mm = tcp_offset_from_args(args)
    controller = PiperController(config=replace(ROBOT_CONFIG, can_name=args.can))
    print("Connecting to Piper arm...", flush=True)
    controller.connect(enable=args.execute and not args.no_enable)
    if args.execute and not args.no_enable:
        print("Piper arm enabled", flush=True)
    elif args.execute:
        print("Skipped Piper enable; assuming arm is already enabled", flush=True)
    try:
        if args.execute:
            maybe_set_open_zero(controller, args)
        current_j6 = read_current_j6_pose(controller)
        current_tcp = tcp_pose_for_j6_pose(current_j6, tcp_offset_j6_mm)
        target_tcp = target_pose_from_args(args, current_j6, tcp_offset_j6_mm)
        print(f"Current J6: {format_pose(current_j6)}", flush=True)
        print(f"Current TCP: {format_pose(current_tcp)}", flush=True)
        print(f"Target TCP: {format_pose(target_tcp)}", flush=True)
        counts = execute_demo(controller, target_tcp, tcp_offset_j6_mm, args)
        print(f"Demo complete, execute={args.execute}, send_counts={counts}", flush=True)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()

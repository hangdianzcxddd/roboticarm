"""Run a fixed top-down catch/place sequence without camera input.

The script is intentionally conservative: all commanded J6 waypoints are
printed and validated before execution. It cannot prove that Piper IK will solve
every pose, but it catches unsafe XYZ/radius limits before anything is sent.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace
from typing import Sequence

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import (
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
    wait_until_pose_reached,
)
from robot_control.linux.catch.tcp_offset import (
    OFFICIAL_GRIPPER_TCP_OFFSET_MM,
    build_catch_plan,
    j6_pose_for_tcp_pose,
    tcp_pose_for_j6_pose,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG, RobotConfig
from robot_control.shared.geometry import EndPose


@dataclass(frozen=True)
class DemoWaypoint:
    label: str
    pose: EndPose
    mode: str


@dataclass(frozen=True)
class DemoPlan:
    current_j6_pose: EndPose
    current_tcp_pose: EndPose
    target_tcp_pose: EndPose
    place_tcp_pose: EndPose
    waypoints: tuple[DemoWaypoint, ...]

    def named_j6_poses(self) -> tuple[tuple[str, EndPose], ...]:
        return tuple((waypoint.label, waypoint.pose) for waypoint in self.waypoints)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a scripted top-down catch/place demo from fixed robot-frame "
            "poses. This bypasses camera clicks and hand-eye calibration."
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
    parser.add_argument("--return-mode", choices=("P", "L"), default="P")
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=12.0)
    parser.add_argument(
        "--continue-on-waypoint-error",
        action="store_true",
        help="continue when a waypoint wait times out after command streaming",
    )
    parser.add_argument("--no-wait-waypoints", action="store_true")
    parser.add_argument("--print-feedback", action="store_true")
    parser.add_argument("--tcp-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-y-mm", type=float, default=0.0)
    parser.add_argument(
        "--tcp-offset-z-mm",
        type=float,
        default=OFFICIAL_GRIPPER_TCP_OFFSET_MM[2],
        help="J6 to gripper TCP Z offset; official Piper gripper default is 145 mm.",
    )
    parser.add_argument(
        "--target-x-mm",
        type=float,
        help="grasp TCP X in base frame; defaults to the safe test X 300 mm",
    )
    parser.add_argument(
        "--target-y-mm",
        type=float,
        help="grasp TCP Y in base frame; defaults to the safe test Y 0 mm",
    )
    parser.add_argument(
        "--target-z-mm",
        type=float,
        help="grasp TCP Z in base frame; defaults to the safe test Z 80 mm",
    )
    # 默认保持当前机械臂姿态
    parser.add_argument(
        "--rx-deg",
        type=float,
        help="target RX; defaults to current J6 RX",
    )

    parser.add_argument(
        "--ry-deg",
        type=float,
        help="target RY; defaults to current J6 RY",
    )

    parser.add_argument(
        "--rz-deg",
        type=float,
        help="target RZ; defaults to current J6 RZ",
    )

    parser.add_argument("--target-z-offset-mm", type=float, default=0.0)
    parser.add_argument("--approach-height-mm", type=float, default=50.0)
    parser.add_argument(
        "--descend-mm",
        type=float,
        default=20.0,
        help="default grasp J6 descent from current pose when target XYZ is omitted",
    )
    parser.add_argument("--min-target-z-mm", type=float, default=0.0)
    parser.add_argument("--lift-height-mm", type=float, default=60.0)
    parser.add_argument(
        "--place-x-mm",
        type=float,
        help="place TCP X in base frame; defaults to target X plus --place-dx-mm",
    )
    parser.add_argument(
        "--place-y-mm",
        type=float,
        help="place TCP Y in base frame; defaults to target Y plus --place-dy-mm",
    )
    parser.add_argument("--place-dx-mm", type=float, default=0.0)
    parser.add_argument("--place-dy-mm", type=float, default=-80.0)
    parser.add_argument("--place-z-mm", type=float, default=80.0)
    parser.add_argument("--place-rx-deg", type=float)
    parser.add_argument("--place-ry-deg", type=float)
    parser.add_argument("--place-rz-deg", type=float)
    parser.add_argument("--place-approach-height-mm", type=float, default=50.0)
    parser.add_argument(
        "--stage-height-mm",
        type=float,
        default=60.0,
        help="initial lift above current J6 before rotating to top-down",
    )
    parser.add_argument(
        "--no-return-home",
        dest="return_home",
        action="store_false",
        help="stop after place retreat instead of restoring posture and returning home",
    )
    parser.set_defaults(return_home=True)
    parser.add_argument(
        "--return-height-mm",
        type=float,
        default=60.0,
        help="extra J6 height used for the posture-restore return waypoint",
    )
    parser.add_argument(
        "--min-horizontal-radius-mm",
        type=float,
        default=0.0,
        help="optional conservative lower bound for commanded J6 radius",
    )
    parser.add_argument(
        "--max-horizontal-radius-mm",
        type=float,
        help="optional conservative upper bound for commanded J6 radius",
    )
    parser.add_argument(
        "--min-j6-z-mm",
        type=float,
        help="optional conservative lower bound for commanded J6 Z",
    )
    parser.add_argument(
        "--max-j6-z-mm",
        type=float,
        help="optional conservative upper bound for commanded J6 Z",
    )
    parser.add_argument(
        "--warn-margin-mm",
        type=float,
        default=25.0,
        help="print a warning when a waypoint is this close to an envelope bound",
    )
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
    if not isinstance(args.speed, int) or not 0 <= args.speed <= 100:
        raise ValueError("--speed must be an integer in [0, 100]")
    if args.command_duration_s < 0:
        raise ValueError("--command-duration-s must be >= 0")
    if args.command_period_s <= 0:
        raise ValueError("--command-period-s must be > 0")
    if args.motion_timeout_s < 0:
        raise ValueError("--motion-timeout-s must be >= 0")
    if args.position_tolerance_mm < 0:
        raise ValueError("--position-tolerance-mm must be >= 0")
    if args.orientation_tolerance_deg < 0:
        raise ValueError("--orientation-tolerance-deg must be >= 0")
    for name in (
        "target_z_offset_mm",
        "approach_height_mm",
        "descend_mm",
        "min_target_z_mm",
        "lift_height_mm",
        "place_approach_height_mm",
        "stage_height_mm",
        "return_height_mm",
        "min_horizontal_radius_mm",
        "warn_margin_mm",
    ):
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
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
    if not math.isfinite(release_command_mm(args)):
        raise ValueError("--release-command-mm must be finite")
    if args.max_horizontal_radius_mm is not None and args.max_horizontal_radius_mm < 0:
        raise ValueError("--max-horizontal-radius-mm must be >= 0")
    if (
        args.max_horizontal_radius_mm is not None
        and args.min_horizontal_radius_mm > args.max_horizontal_radius_mm
    ):
        raise ValueError("--min-horizontal-radius-mm must be <= --max-horizontal-radius-mm")
    if (
        args.min_j6_z_mm is not None
        and args.max_j6_z_mm is not None
        and args.min_j6_z_mm > args.max_j6_z_mm
    ):
        raise ValueError("--min-j6-z-mm must be <= --max-j6-z-mm")


# def target_pose_from_args(
#     args: argparse.Namespace,
#     current_j6_pose: EndPose,
#     tcp_offset_j6_mm: Sequence[float],
# ) -> EndPose:

#     rx_deg = (
#         current_j6_pose.rx_deg
#         if args.rx_deg is None
#         else args.rx_deg
#     )

#     ry_deg = (
#         current_j6_pose.ry_deg
#         if args.ry_deg is None
#         else args.ry_deg
#     )

#     rz_deg = (
#         current_j6_pose.rz_deg
#         if args.rz_deg is None
#         else args.rz_deg
#     )

#     desired_grasp_j6_pose = EndPose(
#         x_mm=current_j6_pose.x_mm,
#         y_mm=current_j6_pose.y_mm,
#         z_mm=current_j6_pose.z_mm - args.descend_mm,
#         rx_deg=rx_deg,
#         ry_deg=ry_deg,
#         rz_deg=rz_deg,
#     )

#     default_target_tcp = tcp_pose_for_j6_pose(
#         desired_grasp_j6_pose,
#         tcp_offset_j6_mm,
#     )
#     if default_target_tcp.z_mm < args.min_target_z_mm:
#         desired_grasp_j6_pose = replace(
#             desired_grasp_j6_pose,
#             z_mm=desired_grasp_j6_pose.z_mm
#             + args.min_target_z_mm
#             - default_target_tcp.z_mm,
#         )
#         default_target_tcp = tcp_pose_for_j6_pose(
#             desired_grasp_j6_pose,
#             tcp_offset_j6_mm,
#         )
#     return EndPose(
#         x_mm=default_target_tcp.x_mm if args.target_x_mm is None else args.target_x_mm,
#         y_mm=default_target_tcp.y_mm if args.target_y_mm is None else args.target_y_mm,
#         z_mm=default_target_tcp.z_mm if args.target_z_mm is None else args.target_z_mm,
#         rx_deg=(
#             current_j6_pose.rx_deg
#             if args.rx_deg is None
#             else args.rx_deg
#         ),
#         ry_deg=(
#             current_j6_pose.ry_deg
#             if args.ry_deg is None
#             else args.ry_deg
#         ),
#         rz_deg=rz_deg,
#     )
def target_pose_from_args(
    args: argparse.Namespace,
    current_j6_pose: EndPose,
    tcp_offset_j6_mm: Sequence[float],
) -> EndPose:

    rx_deg = current_j6_pose.rx_deg if args.rx_deg is None else args.rx_deg
    ry_deg = current_j6_pose.ry_deg if args.ry_deg is None else args.ry_deg
    rz_deg = current_j6_pose.rz_deg if args.rz_deg is None else args.rz_deg

    # ---------------------------------------------------
    # 默认测试点（安全区域）
    # ---------------------------------------------------

    default_x_mm = 300.0
    default_y_mm = 0.0
    default_z_mm = 60.0

    target_x_mm = (
        default_x_mm
        if args.target_x_mm is None
        else args.target_x_mm
    )

    target_y_mm = (
        default_y_mm
        if args.target_y_mm is None
        else args.target_y_mm
    )

    target_z_mm = (
        default_z_mm
        if args.target_z_mm is None
        else args.target_z_mm
    )

    return EndPose(
        x_mm=target_x_mm,
        y_mm=target_y_mm,
        z_mm=target_z_mm,
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
    )

def place_pose_from_args(args: argparse.Namespace, target_tcp_pose: EndPose) -> EndPose:
    rx_deg = target_tcp_pose.rx_deg if args.place_rx_deg is None else args.place_rx_deg
    ry_deg = target_tcp_pose.ry_deg if args.place_ry_deg is None else args.place_ry_deg
    rz_deg = target_tcp_pose.rz_deg if args.place_rz_deg is None else args.place_rz_deg
    return EndPose(
        x_mm=(
            target_tcp_pose.x_mm + args.place_dx_mm
            if args.place_x_mm is None
            else args.place_x_mm
        ),
        y_mm=(
            target_tcp_pose.y_mm + args.place_dy_mm
            if args.place_y_mm is None
            else args.place_y_mm
        ),
        z_mm=args.place_z_mm,
        rx_deg=rx_deg,
        ry_deg=ry_deg,
        rz_deg=rz_deg,
    )


def build_scripted_plan(
    args: argparse.Namespace,
    current_j6_pose: EndPose,
    tcp_offset_j6_mm: Sequence[float],
    config: RobotConfig = ROBOT_CONFIG,
) -> DemoPlan:
    current_tcp_pose = tcp_pose_for_j6_pose(current_j6_pose, tcp_offset_j6_mm)
    target_tcp_pose = target_pose_from_args(args, current_j6_pose, tcp_offset_j6_mm)
    catch_plan = build_catch_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        approach_height_mm=40.0,   # 抬高幅度
        lift_height_mm=40.0,       # 抬高幅度
        target_z_offset_mm=40.0,   # 抓取点不要贴地太低
    )
    place_tcp_pose = place_pose_from_args(args, target_tcp_pose)
    place_approach_tcp_pose = replace(
        place_tcp_pose,
        z_mm=place_tcp_pose.z_mm + args.place_approach_height_mm,
    )
    place_j6_pose = j6_pose_for_tcp_pose(place_tcp_pose, tcp_offset_j6_mm)
    place_approach_j6_pose = j6_pose_for_tcp_pose(
        place_approach_tcp_pose,
        tcp_offset_j6_mm,
    )

    first_stage_z_mm = min(
        max(
            current_j6_pose.z_mm + args.stage_height_mm,
            current_j6_pose.z_mm,
            catch_plan.approach_j6_pose.z_mm,
        ),
        config.z_limit_mm.max_value,
    )
    stage_lift_pose = replace(current_j6_pose, z_mm=first_stage_z_mm)
    orient_top_down_pose = EndPose(
        current_j6_pose.x_mm,
        current_j6_pose.y_mm,
        first_stage_z_mm,
        target_tcp_pose.rx_deg,
        target_tcp_pose.ry_deg,
        target_tcp_pose.rz_deg,
    )

    # waypoints = [
    #     DemoWaypoint("stage_lift", stage_lift_pose, "P"),
    #     DemoWaypoint("orient_top_down", orient_top_down_pose, "P"),
    #     DemoWaypoint("approach", catch_plan.approach_j6_pose, args.approach_mode),
    #     DemoWaypoint("grasp", catch_plan.grasp_j6_pose, args.grasp_mode),
    #     DemoWaypoint("lift", catch_plan.lift_j6_pose, args.lift_mode),
    #     DemoWaypoint("place_approach", place_approach_j6_pose, args.approach_mode),
    #     DemoWaypoint("place", place_j6_pose, args.grasp_mode),
    #     DemoWaypoint("place_retreat", place_approach_j6_pose, args.lift_mode),
    # ]

    waypoints = [
        DemoWaypoint("stage_lift", stage_lift_pose, "P"),
    ]

    orientation_changed = (
        args.rx_deg is not None
        or args.ry_deg is not None
        or args.rz_deg is not None
    )

    if orientation_changed:
        waypoints.append(
            DemoWaypoint(
                "orient_top_down",
                orient_top_down_pose,
                "P",
            )
        )

    waypoints.extend(
        [
            DemoWaypoint(
                "approach",
                catch_plan.approach_j6_pose,
                args.approach_mode,
            ),
            DemoWaypoint(
                "grasp",
                catch_plan.grasp_j6_pose,
                args.grasp_mode,
            ),
            DemoWaypoint(
                "lift",
                catch_plan.lift_j6_pose,
                args.lift_mode,
            ),
            DemoWaypoint(
                "place_approach",
                place_approach_j6_pose,
                args.approach_mode,
            ),
            DemoWaypoint(
                "place",
                place_j6_pose,
                args.grasp_mode,
            ),
            DemoWaypoint(
                "place_retreat",
                place_approach_j6_pose,
                args.lift_mode,
            ),
        ]
    )

    if args.return_home:
        return_z_mm = min(
            max(
                current_j6_pose.z_mm + args.return_height_mm,
                catch_plan.lift_j6_pose.z_mm,
                place_approach_j6_pose.z_mm,
            ),
            config.z_limit_mm.max_value,
        )
        return_align_pose = EndPose(
            current_j6_pose.x_mm,
            current_j6_pose.y_mm,
            return_z_mm,
            place_tcp_pose.rx_deg,
            place_tcp_pose.ry_deg,
            place_tcp_pose.rz_deg,
        )
        restore_orientation_pose = EndPose(
            current_j6_pose.x_mm,
            current_j6_pose.y_mm,
            return_z_mm,
            current_j6_pose.rx_deg,
            current_j6_pose.ry_deg,
            current_j6_pose.rz_deg,
        )
        waypoints.extend(
            [
                DemoWaypoint("return_align", return_align_pose, args.return_mode),
                DemoWaypoint(
                    "restore_orientation",
                    restore_orientation_pose,
                    args.return_mode,
                ),
                DemoWaypoint("return_home", current_j6_pose, args.return_mode),
            ]
        )

    return DemoPlan(
        current_j6_pose=current_j6_pose,
        current_tcp_pose=current_tcp_pose,
        target_tcp_pose=target_tcp_pose,
        place_tcp_pose=place_tcp_pose,
        waypoints=tuple(waypoints),
    )


def validate_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: DemoPlan,
) -> None:
    print(f"Current J6: {format_pose(plan.current_j6_pose)}", flush=True)
    print(f"Current TCP: {format_pose(plan.current_tcp_pose)}", flush=True)
    print(f"Target TCP: {format_pose(plan.target_tcp_pose)}", flush=True)
    print(f"Place TCP: {format_pose(plan.place_tcp_pose)}", flush=True)
    print("Planned J6 waypoints:", flush=True)
    previous_pose = plan.current_j6_pose

    for label, pose in plan.named_j6_poses():

        validate_j6_waypoint(
            controller,
            args,
            label,
            pose,
            previous_pose,
        )

        previous_pose = pose
        print(f"  {label}: {format_pose(pose)}", flush=True)


def validate_j6_waypoint(
    controller: PiperController,
    args: argparse.Namespace,
    label: str,
    pose: EndPose,
    current_pose=None,
) -> None:
    try:
        controller.validate_pose(pose)
        if current_pose is not None:

            ori_delta = max(
                abs(pose.rx_deg - current_pose.rx_deg),
                abs(pose.ry_deg - current_pose.ry_deg),
                abs(pose.rz_deg - current_pose.rz_deg),
            )

            if ori_delta > 45.0:
                raise SafetyError(
                    f"{label}: orientation jump "
                    f"{ori_delta:.1f} deg exceeds 45 deg"
                )
    except SafetyError as exc:
        raise SafetyError(f"{label}: {exc}") from exc

    radius_mm = math.hypot(pose.x_mm, pose.y_mm)
    min_radius_mm = args.min_horizontal_radius_mm
    max_radius_mm = (
        controller.config.work_radius_mm
        if args.max_horizontal_radius_mm is None
        else args.max_horizontal_radius_mm
    )
    min_z_mm = (
        controller.config.z_limit_mm.min_value
        if args.min_j6_z_mm is None
        else args.min_j6_z_mm
    )
    max_z_mm = (
        controller.config.z_limit_mm.max_value
        if args.max_j6_z_mm is None
        else args.max_j6_z_mm
    )
    if radius_mm < min_radius_mm:
        raise SafetyError(
            f"{label}: J6 horizontal radius {radius_mm:.3f} mm is below "
            f"--min-horizontal-radius-mm {min_radius_mm:.3f}"
        )
    if radius_mm > max_radius_mm:
        raise SafetyError(
            f"{label}: J6 horizontal radius {radius_mm:.3f} mm exceeds "
            f"max radius {max_radius_mm:.3f}"
        )
    if pose.z_mm < min_z_mm:
        raise SafetyError(
            f"{label}: J6 z {pose.z_mm:.3f} mm is below min z {min_z_mm:.3f}"
        )
    if pose.z_mm > max_z_mm:
        raise SafetyError(
            f"{label}: J6 z {pose.z_mm:.3f} mm exceeds max z {max_z_mm:.3f}"
        )

    margin = args.warn_margin_mm
    warnings: list[str] = []
    if margin > 0:
        if radius_mm - min_radius_mm <= margin:
            warnings.append(f"radius is {radius_mm - min_radius_mm:.1f} mm above min")
        if max_radius_mm - radius_mm <= margin:
            warnings.append(f"radius is {max_radius_mm - radius_mm:.1f} mm below max")
        if pose.z_mm - min_z_mm <= margin:
            warnings.append(f"z is {pose.z_mm - min_z_mm:.1f} mm above min")
        if max_z_mm - pose.z_mm <= margin:
            warnings.append(f"z is {max_z_mm - pose.z_mm:.1f} mm below max")
    if warnings:
        print(f"  Warning {label}: {'; '.join(warnings)}", flush=True)


def execute_demo_waypoint(
    controller: PiperController,
    waypoint: DemoWaypoint,
    speed: int,
    args: argparse.Namespace,
) -> int:
    send_count = execute_pose_for_duration(
        controller,
        waypoint.pose,
        speed,
        waypoint.mode,
        args.command_duration_s,
        args.command_period_s,
    )
    if args.no_wait_waypoints:
        return send_count
    try:
        wait_until_pose_reached(
            controller,
            waypoint.label,
            waypoint.pose,
            timeout_s=args.motion_timeout_s,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
        )
    except RuntimeError as exc:
        if not args.continue_on_waypoint_error:
            raise
        actual_pose = feedback_to_pose(controller.get_end_pose())
        position_error, orientation_error = (
            pose_errors(waypoint.pose, actual_pose)
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
    plan: DemoPlan,
    args: argparse.Namespace,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before demo J6: {format_pose(before_pose)}", flush=True)

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

    for waypoint in plan.waypoints:
        counts[waypoint.label] = execute_demo_waypoint(
            controller,
            waypoint,
            args.speed,
            args,
        )
        if waypoint.label == "grasp" and not args.no_gripper:
            controller.set_gripper(
                args.grasp_command_mm,
                grasp_effort_nm(args),
                enable=True,
            )
            time.sleep(args.gripper_settle_s)
        elif waypoint.label == "place" and not args.no_gripper:
            controller.set_gripper(
                release_command_mm(args),
                release_effort_nm(args),
                enable=True,
            )
            time.sleep(args.gripper_settle_s)

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After demo J6: {format_pose(after_pose)}", flush=True)
        print(f"Demo send counts: {counts}", flush=True)
    return counts


def run_with_controller(
    controller: PiperController,
    args: argparse.Namespace,
    tcp_offset_j6_mm: Sequence[float],
) -> dict[str, int]:
    current_j6_pose = read_current_j6_pose(controller)
    plan = build_scripted_plan(
        args,
        current_j6_pose,
        tcp_offset_j6_mm,
        config=controller.config,
    )
    validate_plan(controller, args, plan)
    if not args.execute:
        print("Dry-run complete; add --execute to move the arm.", flush=True)
        return {}
    maybe_set_open_zero(controller, args)
    counts = execute_demo(controller, plan, args)
    print(f"Demo complete, execute={args.execute}, send_counts={counts}", flush=True)
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
        run_with_controller(controller, args, tcp_offset_j6_mm)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()

"""Inspect one camera XYZ point through hand-eye and catch planning."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_control.linux.catch.catch_server import (
    DEFAULT_HAND_EYE_RESULT,
    DEFAULT_PLACE_X_MM,
    DEFAULT_PLACE_Y_MM,
    DEFAULT_PLACE_Z_MM,
    build_place_plan,
    format_pose,
    planned_j6_waypoints,
)
from robot_control.linux.catch.hand_eye_mapping import (
    camera_xyz_to_hand_eye_robot_pose,
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
    map_camera_point_to_base_xyz_m,
)
from robot_control.linux.catch.tcp_offset import build_catch_plan
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.shared.geometry import TOP_DOWN_TCP_ORIENTATION_DEG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Map a RealSense camera XYZ point through hand-eye calibration and "
            "print the resulting TCP/J6 catch waypoints without connecting to Piper."
        )
    )
    parser.add_argument("--x-m", type=float, required=True)
    parser.add_argument("--y-m", type=float, required=True)
    parser.add_argument("--z-m", type=float, required=True)
    parser.add_argument(
        "--hand-eye-result",
        type=Path,
        default=DEFAULT_HAND_EYE_RESULT,
    )
    parser.add_argument("--rx-deg", type=float)
    parser.add_argument("--ry-deg", type=float)
    parser.add_argument("--rz-deg", type=float)
    parser.add_argument("--tcp-offset-x-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-y-mm", type=float, default=0.0)
    parser.add_argument("--tcp-offset-z-mm", type=float, default=145.0)
    parser.add_argument("--target-z-offset-mm", type=float, default=40.0)
    parser.add_argument("--approach-height-mm", type=float, default=40.0)
    parser.add_argument("--lift-height-mm", type=float, default=40.0)
    parser.add_argument(
        "--place-x-mm",
        type=float,
        help=f"place TCP X in base frame; defaults to {DEFAULT_PLACE_X_MM:.0f} mm",
    )
    parser.add_argument(
        "--place-y-mm",
        type=float,
        help=f"place TCP Y in base frame; defaults to {DEFAULT_PLACE_Y_MM:.0f} mm",
    )
    parser.add_argument(
        "--place-dx-mm",
        type=float,
        help=(
            "legacy relative place TCP X offset from the clicked target; setting "
            "either relative offset switches place planning to target-relative mode"
        ),
    )
    parser.add_argument(
        "--place-dy-mm",
        type=float,
        help=(
            "legacy relative place TCP Y offset from the clicked target; setting "
            "either relative offset switches place planning to target-relative mode"
        ),
    )
    parser.add_argument("--place-z-mm", type=float, default=DEFAULT_PLACE_Z_MM)
    parser.add_argument("--place-rx-deg", type=float)
    parser.add_argument("--place-ry-deg", type=float)
    parser.add_argument("--place-rz-deg", type=float)
    parser.add_argument("--place-approach-height-mm", type=float, default=50.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    document = load_hand_eye_result_document(args.hand_eye_result)
    camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
    quality = hand_eye_quality_summary(document)
    if quality is not None:
        print(f"Hand-eye quality: {quality}")

    default_rx, default_ry, default_rz = TOP_DOWN_TCP_ORIENTATION_DEG
    target_rx_deg = default_rx if args.rx_deg is None else args.rx_deg
    target_ry_deg = default_ry if args.ry_deg is None else args.ry_deg
    target_rz_deg = default_rz if args.rz_deg is None else args.rz_deg
    orientation_source = "args" if None not in (args.rx_deg, args.ry_deg, args.rz_deg) else "dry_run_default"
    tcp_offset_j6_mm = (
        args.tcp_offset_x_mm,
        args.tcp_offset_y_mm,
        args.tcp_offset_z_mm,
    )
    base_xyz_m = map_camera_point_to_base_xyz_m(
        args.x_m,
        args.y_m,
        args.z_m,
        camera_to_base,
    )
    target_tcp_pose = camera_xyz_to_hand_eye_robot_pose(
        args.x_m,
        args.y_m,
        args.z_m,
        camera_to_base,
        rx_deg=target_rx_deg,
        ry_deg=target_ry_deg,
        rz_deg=target_rz_deg,
    )
    catch_plan = build_catch_plan(
        target_tcp_pose,
        offset_j6_tcp_mm=tcp_offset_j6_mm,
        approach_height_mm=args.approach_height_mm,
        lift_height_mm=args.lift_height_mm,
        target_z_offset_mm=args.target_z_offset_mm,
    )
    place_plan = build_place_plan(target_tcp_pose, tcp_offset_j6_mm, args)

    print(
        "camera_xyz="
        f"({args.x_m:.6f}, {args.y_m:.6f}, {args.z_m:.6f}) m"
    )
    print(
        "base_xyz="
        f"({base_xyz_m[0] * 1000.0:.3f}, "
        f"{base_xyz_m[1] * 1000.0:.3f}, "
        f"{base_xyz_m[2] * 1000.0:.3f}) mm"
    )
    print(f"orientation={orientation_source}")
    print(f"target_tcp={format_pose(catch_plan.target_tcp_pose)}")
    print(f"grasp_tcp={format_pose(catch_plan.grasp_tcp_pose)}")
    print(f"place_tcp={format_pose(place_plan.place_tcp_pose)}")
    for label, pose in planned_j6_waypoints(catch_plan, place_plan):
        radius_mm = (pose.x_mm**2 + pose.y_mm**2) ** 0.5
        limit = "OK" if radius_mm <= ROBOT_CONFIG.work_radius_mm else "OUT_OF_RADIUS"
        print(f"{label}_j6={format_pose(pose)} radius={radius_mm:.1f}mm {limit}")


if __name__ == "__main__":
    main()

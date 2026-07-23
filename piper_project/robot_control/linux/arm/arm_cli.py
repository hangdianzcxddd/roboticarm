"""Command-line entry point for Linux-side Piper arm control."""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.arm.kinematics import (
    forward_kinematics_pose,
    forward_kinematics_transform,
)
from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.shared.geometry import EndPose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Piper robot motion control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="print current arm feedback")
    subparsers.add_parser("enable", help="connect and enable arm")
    disable_parser = subparsers.add_parser("disable", help="park and disable arm")
    disable_parser.add_argument(
        "--no-park",
        action="store_true",
        help="disable immediately without moving to the configured parking joints",
    )

    pose_parser = subparsers.add_parser("pose", help="move to end pose in robot frame")
    pose_parser.add_argument("x", type=float)
    pose_parser.add_argument("y", type=float)
    pose_parser.add_argument("z", type=float)
    pose_parser.add_argument("rx", type=float)
    pose_parser.add_argument("ry", type=float)
    pose_parser.add_argument("rz", type=float)
    pose_parser.add_argument(
        "--speed", type=int, default=ROBOT_CONFIG.default_speed_percent
    )
    pose_parser.add_argument("--mode", choices=("P", "L"), default="P")

    joints_parser = subparsers.add_parser("joints", help="move joints in radians")
    for index in range(1, 7):
        joints_parser.add_argument(f"j{index}", type=float)
    joints_parser.add_argument(
        "--speed", type=int, default=ROBOT_CONFIG.default_speed_percent
    )
    joints_parser.add_argument(
        "--max-joint-delta-deg",
        type=float,
        nargs=6,
        default=(60.0,) * 6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="maximum change per joint from live feedback, in degrees",
    )
    joints_parser.add_argument("--motion-timeout-s", type=float, default=10.0)
    joints_parser.add_argument("--joint-tolerance-deg", type=float, default=2.0)
    joints_parser.add_argument(
        "--max-joint-waypoint-step-deg", type=float, default=8.0
    )
    joints_parser.add_argument(
        "--max-runtime-path-error-deg", type=float, default=5.0
    )
    joints_parser.add_argument("--max-jacobian-condition", type=float, default=500.0)

    fk_parser = subparsers.add_parser(
        "joints-to-pose",
        help="print forward-kinematics J6 pose for joints in radians",
    )
    for index in range(1, 7):
        fk_parser.add_argument(f"j{index}", type=float)

    return parser


def read_live_joints(controller: PiperController) -> list[float]:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        feedback = controller.get_joint_state()
        joints = controller._joint_feedback_to_rad(feedback)
        fresh = (
            float(getattr(feedback, "time_stamp", 0.0)) > 0.0
            or float(getattr(feedback, "Hz", 0.0)) > 0.0
        )
        if joints is not None and fresh:
            return joints
        time.sleep(0.02)
    raise RuntimeError("fresh Piper joint feedback is required before joint motion")


def validate_joint_change(
    current: list[float],
    target: list[float],
    limits_deg: list[float] | tuple[float, ...],
) -> None:
    if len(limits_deg) != 6 or any(
        not math.isfinite(value) or value <= 0 for value in limits_deg
    ):
        raise SafetyError("all --max-joint-delta-deg values must be finite and > 0")
    violations = [
        f"J{index}={delta:.2f}deg>{limit:.2f}deg"
        for index, (start, end, limit) in enumerate(
            zip(current, target, limits_deg), start=1
        )
        if (delta := abs(math.degrees(end - start))) > limit
    ]
    if violations:
        raise SafetyError("joint change too large: " + ", ".join(violations))


def validate_joint_motion_args(args: argparse.Namespace) -> None:
    for option in (
        "motion_timeout_s",
        "joint_tolerance_deg",
        "max_joint_waypoint_step_deg",
        "max_runtime_path_error_deg",
        "max_jacobian_condition",
    ):
        value = float(getattr(args, option))
        if not math.isfinite(value) or value <= 0:
            raise SafetyError(f"--{option.replace('_', '-')} must be finite and > 0")


def joint_path(
    start: list[float], target: list[float], max_step_deg: float
) -> tuple[tuple[float, ...], ...]:
    if not math.isfinite(max_step_deg) or max_step_deg <= 0:
        raise SafetyError("--max-joint-waypoint-step-deg must be finite and > 0")
    start_array = np.asarray(start, dtype=float)
    target_array = np.asarray(target, dtype=float)
    maximum_delta = float(np.max(np.abs(np.degrees(target_array - start_array))))
    segments = max(1, math.ceil(maximum_delta / max_step_deg))
    return tuple(
        tuple(
            float(value)
            for value in start_array + alpha * (target_array - start_array)
        )
        for alpha in np.linspace(0.0, 1.0, segments + 1)
    )


def jacobian_condition(joints: list[float] | tuple[float, ...]) -> float:
    joints_array = np.asarray(joints, dtype=float)
    base = forward_kinematics_transform(joints_array)
    jacobian = np.empty((6, 6), dtype=float)
    epsilon = 1e-5
    for index in range(6):
        shifted = joints_array.copy()
        shifted[index] += epsilon
        transform = forward_kinematics_transform(shifted)
        position = (transform[:3, 3] - base[:3, 3]) / (100.0 * epsilon)
        rotation = Rotation.from_matrix(
            base[:3, :3].T @ transform[:3, :3]
        ).as_rotvec() / (math.radians(10.0) * epsilon)
        jacobian[:, index] = np.concatenate((position, rotation))
    return float(np.linalg.cond(jacobian))


def validate_joint_path(
    controller: PiperController,
    path: tuple[tuple[float, ...], ...],
    max_condition: float,
) -> None:
    if not math.isfinite(max_condition) or max_condition <= 0:
        raise SafetyError("--max-jacobian-condition must be finite and > 0")
    for index, joints in enumerate(path):
        controller.validate_joints(list(joints))
        condition = jacobian_condition(joints)
        if not math.isfinite(condition) or condition > max_condition:
            raise SafetyError(
                f"joint path[{index}] Jacobian condition={condition:.3f} exceeds "
                f"{max_condition:.3f}"
            )


def wait_for_joint_waypoint(
    controller: PiperController,
    segment_start: list[float],
    target: tuple[float, ...],
    args: argparse.Namespace,
    *,
    label: str,
) -> None:
    deadline = time.monotonic() + args.motion_timeout_s
    tolerance = math.radians(args.max_runtime_path_error_deg)
    lower = np.minimum(segment_start, target) - tolerance
    upper = np.maximum(segment_start, target) + tolerance
    last_error = float("inf")
    while time.monotonic() < deadline:
        current = np.asarray(read_live_joints(controller), dtype=float)
        if np.any(current < lower) or np.any(current > upper):
            raise SafetyError(f"{label} live joints deviated from the planned segment")
        condition = jacobian_condition(tuple(current))
        if not math.isfinite(condition) or condition > args.max_jacobian_condition:
            raise SafetyError(
                f"{label} live Jacobian condition={condition:.3f} exceeds "
                f"{args.max_jacobian_condition:.3f}"
            )
        last_error = float(np.max(np.abs(np.degrees(current - target))))
        if last_error <= args.joint_tolerance_deg:
            return
        time.sleep(0.02)
    raise TimeoutError(f"{label} not reached: max_error={last_error:.3f}deg")


def execute_joint_path(
    controller: PiperController,
    path: tuple[tuple[float, ...], ...],
    args: argparse.Namespace,
) -> None:
    for index, target in enumerate(path[1:], start=1):
        current = read_live_joints(controller)
        expected = np.asarray(path[index - 1], dtype=float)
        error = float(np.max(np.abs(np.degrees(np.asarray(current) - expected))))
        if error > args.joint_tolerance_deg:
            raise SafetyError(
                f"joint path[{index}] starts {error:.3f}deg away from planned path"
            )
        controller.move_joints(list(target), speed_percent=args.speed)
        wait_for_joint_waypoint(
            controller,
            current,
            target,
            args,
            label=f"joint path[{index}]",
        )


def main() -> None:
    args = build_parser().parse_args()
    controller = PiperController()

    try:
        if args.command == "status":
            controller.connect(enable=False)
            print(controller.get_status())
            print(controller.get_end_pose())
            print(controller.get_joint_state())
            controller.disconnect()
        elif args.command == "enable":
            controller.connect(enable=True)
            controller.disconnect()
        elif args.command == "disable":
            should_park = not args.no_park
            controller.connect(enable=should_park)
            controller.safe_disable(park=should_park)
            controller.disconnect()
        elif args.command == "pose":
            pose = EndPose(args.x, args.y, args.z, args.rx, args.ry, args.rz)
            controller.validate_pose(pose)
            with controller:
                controller.move_to_pose(
                    pose,
                    speed_percent=args.speed,
                    move_mode=args.mode,
                )
        elif args.command == "joints":
            joints = [args.j1, args.j2, args.j3, args.j4, args.j5, args.j6]
            controller.validate_joints(joints)
            validate_joint_motion_args(args)
            controller.connect()
            try:
                current = read_live_joints(controller)
                validate_joint_change(current, joints, args.max_joint_delta_deg)
                path = joint_path(
                    current,
                    joints,
                    args.max_joint_waypoint_step_deg,
                )
                validate_joint_path(
                    controller,
                    path,
                    args.max_jacobian_condition,
                )
                execute_joint_path(controller, path, args)
            except BaseException:
                controller.emergency_stop()
                raise
            finally:
                controller.disconnect()
        elif args.command == "joints-to-pose":
            joints = [args.j1, args.j2, args.j3, args.j4, args.j5, args.j6]
            controller.validate_joints(joints)
            pose = forward_kinematics_pose(joints)
            print(
                "x_mm={:.3f} y_mm={:.3f} z_mm={:.3f} "
                "rx_deg={:.6f} ry_deg={:.6f} rz_deg={:.6f}".format(
                    pose.x_mm,
                    pose.y_mm,
                    pose.z_mm,
                    pose.rx_deg,
                    pose.ry_deg,
                    pose.rz_deg,
                )
            )
    except (SafetyError, RuntimeError, TimeoutError) as exc:
        controller.disconnect()
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

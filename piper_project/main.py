"""Manual entry point for Piper motion tests."""

from __future__ import annotations

import argparse
import sys

from config.robot_config import ROBOT_CONFIG
from robot.piper_controller import EndPose, PiperController
from robot.safety import SafetyError


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
    pose_parser.add_argument("--speed", type=int, default=ROBOT_CONFIG.default_speed_percent)
    pose_parser.add_argument("--mode", choices=("P", "L"), default="P")

    joints_parser = subparsers.add_parser("joints", help="move joints in radians")
    for index in range(1, 7):
        joints_parser.add_argument(f"j{index}", type=float)
    joints_parser.add_argument("--speed", type=int, default=ROBOT_CONFIG.default_speed_percent)

    return parser


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
            with controller:
                controller.move_joints(
                    joints,
                    speed_percent=args.speed,
                )
    except SafetyError as exc:
        controller.disconnect()
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

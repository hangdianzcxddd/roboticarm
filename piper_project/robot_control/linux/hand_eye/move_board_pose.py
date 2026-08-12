"""Move the gripper-held calibration board with custom roll/pitch/yaw."""

from __future__ import annotations

import argparse
from dataclasses import replace
import time

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.hand_eye.calibration import format_end_pose
from robot_control.linux.tcp_calibration.calibration import read_fresh_end_pose
from robot_control.shared.geometry import EndPose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move Piper end pose while keeping or setting XYZ and applying custom "
            "roll/pitch/yaw. Use this to present a gripper-held checkerboard at "
            "different orientations for eye-to-hand calibration."
        )
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument(
        "--x-mm",
        type=float,
        help="target X in robot base; omitted means current X",
    )
    parser.add_argument(
        "--y-mm",
        type=float,
        help="target Y in robot base; omitted means current Y",
    )
    parser.add_argument(
        "--z-mm",
        type=float,
        help="target Z in robot base; omitted means current Z",
    )
    parser.add_argument("--roll-deg", type=float, required=True)
    parser.add_argument("--pitch-deg", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--mode", choices=("P", "L"), default="P")
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually command the robot; without this the script only validates",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    controller = PiperController(config=replace(ROBOT_CONFIG, can_name=args.can))
    controller.connect(enable=args.execute)
    try:
        current = read_fresh_end_pose(controller, settle_s=args.settle_s)
        target = EndPose(
            current.x_mm if args.x_mm is None else args.x_mm,
            current.y_mm if args.y_mm is None else args.y_mm,
            current.z_mm if args.z_mm is None else args.z_mm,
            args.roll_deg,
            args.pitch_deg,
            args.yaw_deg,
        )
        controller.validate_pose(target)
        print(f"Current pose: {format_end_pose(current)}")
        print(f"Target pose:  {format_end_pose(target)}")
        if not args.execute:
            print("Dry run only. Add --execute to move.")
            return
        count = execute_pose_for_duration(
            controller,
            target,
            speed=args.speed,
            mode=args.mode,
            duration_s=args.command_duration_s,
            period_s=args.command_period_s,
        )
        print(f"Sent target pose {count} times.")
        after = read_fresh_end_pose(controller, settle_s=args.settle_s)
        print(f"After pose:   {format_end_pose(after)}")
    finally:
        controller.disconnect()


def validate_args(args: argparse.Namespace) -> None:
    if args.speed <= 0 or args.speed > 100:
        raise SystemExit("--speed must be in [1, 100]")
    if args.command_duration_s < 0:
        raise SystemExit("--command-duration-s must be >= 0")
    if args.command_period_s <= 0:
        raise SystemExit("--command-period-s must be > 0")
    if args.settle_s < 0:
        raise SystemExit("--settle-s must be >= 0")


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


if __name__ == "__main__":
    main()

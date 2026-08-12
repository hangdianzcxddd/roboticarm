"""Print the current Piper end pose for hand-eye calibration."""

from __future__ import annotations

import argparse
from dataclasses import replace

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.hand_eye.calibration import format_end_pose
from robot_control.linux.tcp_calibration.calibration import read_fresh_end_pose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read current Piper end pose.")
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--settle-s", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    controller = PiperController(config=replace(ROBOT_CONFIG, can_name=args.can))
    controller.connect(enable=False)
    try:
        pose = read_fresh_end_pose(controller, settle_s=args.settle_s)
        print(format_end_pose(pose))
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()

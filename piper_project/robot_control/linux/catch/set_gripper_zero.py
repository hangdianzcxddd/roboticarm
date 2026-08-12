"""Set the Piper gripper zero point from the Linux catch package.

For the open-zero workflow, first move the gripper to the physical fully-open
100 mm position, then run this script. After that, command 0 mm is treated as
fully open by the gripper controller.
Closing commands should then use negative values, for example -100 mm.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from typing import Any

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Set current physical Piper gripper position as zero. "
            "Open the gripper fully to 100 mm before running."
        )
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument(
        "--zero-command-mm",
        type=float,
        default=0.0,
        help="Command value used while setting zero. For open-zero, keep 0 mm.",
    )
    parser.add_argument("--effort-nm", type=float, default=0.4)
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument(
        "--clear-error",
        action="store_true",
        help="Send enable+clear-error before setting zero.",
    )
    parser.add_argument(
        "--enable-after",
        action="store_true",
        help="Enable gripper at --zero-command-mm after setting zero.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Print gripper feedback after setting zero.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt before setting zero.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.zero_command_mm < 0:
        raise ValueError("--zero-command-mm must be >= 0")
    if not 0 <= args.effort_nm <= 5:
        raise ValueError("--effort-nm must be in [0, 5]")
    if args.settle_s < 0:
        raise ValueError("--settle-s must be >= 0")


def set_gripper_zero(
    controller: PiperController,
    zero_command_mm: float = 0.0,
    effort_nm: float = 0.4,
    settle_s: float = 1.5,
    clear_error: bool = False,
    enable_after: bool = False,
) -> None:
    if zero_command_mm < 0:
        raise ValueError("zero_command_mm must be >= 0")
    if not 0 <= effort_nm <= 5:
        raise ValueError("effort_nm must be in [0, 5]")
    if settle_s < 0:
        raise ValueError("settle_s must be >= 0")

    if clear_error:
        controller.set_gripper(
            zero_command_mm,
            effort_nm,
            enable=True,
            clear_error=True,
        )
        time.sleep(settle_s)

    controller.set_gripper(zero_command_mm, effort_nm, enable=False)
    time.sleep(settle_s)
    controller.set_gripper(
        zero_command_mm,
        effort_nm,
        enable=False,
        set_zero=True,
    )
    time.sleep(settle_s)

    if enable_after:
        controller.set_gripper(zero_command_mm, effort_nm, enable=True)
        time.sleep(settle_s)


def gripper_feedback_summary(feedback_msg: Any) -> str:
    state = getattr(feedback_msg, "gripper_state", feedback_msg)
    position = _scaled(getattr(state, "grippers_angle", None), 0.001)
    effort = _scaled(getattr(state, "grippers_effort", None), 0.001)
    status = getattr(state, "status_code", None)
    hz = getattr(feedback_msg, "Hz", None)
    return (
        f"position_mm={position} effort_nm={effort} "
        f"status_code={status} hz={hz}"
    )


def confirm_zero(args: argparse.Namespace) -> None:
    if args.yes:
        return
    print(
        "Before continuing, manually move the gripper to the physical fully-open "
        "100 mm position."
    )
    answer = input("Set this current physical gripper position as zero? [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        raise SystemExit("Aborted; gripper zero was not changed.")


def make_controller(can_name: str) -> PiperController:
    return PiperController(config=replace(ROBOT_CONFIG, can_name=can_name))


def _scaled(value: Any, factor: float) -> float | None:
    if value is None:
        return None
    return round(float(value) * factor, 3)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    confirm_zero(args)

    controller = make_controller(args.can)
    controller.connect(enable=True)
    try:
        set_gripper_zero(
            controller,
            zero_command_mm=args.zero_command_mm,
            effort_nm=args.effort_nm,
            settle_s=args.settle_s,
            clear_error=args.clear_error,
            enable_after=args.enable_after,
        )
        print(
            f"Set gripper zero at current physical position using "
            f"command={args.zero_command_mm:.3f} mm."
        )
        if args.verify:
            time.sleep(max(args.settle_s, 0.1))
            print(gripper_feedback_summary(controller.get_gripper_state()))
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()

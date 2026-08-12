"""Collect Piper J6 poses for gripper TCP fixed-point pivot calibration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.tcp_calibration.calibration import (
    collect_poses_interactively,
    load_sample_poses,
    make_samples_document,
    write_yaml,
)


DEFAULT_OUTPUT = Path(__file__).with_name("tcp_samples.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect J6 poses for gripper TCP fixed-point pivot calibration."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--read-timeout-s", type=float, default=2.0)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--append",
        action="store_true",
        help="load existing output samples before collecting more",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() and not args.append and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}. Use --append or --overwrite."
        )
    initial_poses = load_sample_poses(args.output) if args.append else []
    config = replace(ROBOT_CONFIG, can_name=args.can)
    controller = PiperController(config=config)

    print(f"Connecting to Piper feedback on {args.can}; arm motion is not enabled.")
    controller.connect(enable=False)
    try:
        poses = collect_poses_interactively(
            controller,
            initial_poses=initial_poses,
            min_samples=args.min_samples,
            timeout_s=args.read_timeout_s,
            settle_s=args.settle_s,
        )
    finally:
        controller.disconnect()

    write_yaml(args.output, make_samples_document(poses, can_name=args.can))
    print(f"Wrote {len(poses)} TCP calibration samples to {args.output}")


if __name__ == "__main__":
    main()

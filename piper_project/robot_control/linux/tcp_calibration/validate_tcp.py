"""Validate a solved gripper TCP calibration with independent pivot samples."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.tcp_calibration.calibration import (
    collect_poses_interactively,
    evaluate_tcp,
    load_calibration,
    load_sample_poses,
    make_samples_document,
    write_yaml,
)


DEFAULT_CONFIG = Path(__file__).with_name("tcp.yaml")
DEFAULT_SAMPLES = Path(__file__).with_name("tcp_validation_samples.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a gripper TCP with independent fixed-point samples."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="collect validation samples from Piper before checking them",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing validation sample file when using --collect",
    )
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--min-samples", type=int, default=4)
    parser.add_argument("--settle-s", type=float, default=0.3)
    parser.add_argument("--read-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-rms-error-mm", type=float, default=1.0)
    parser.add_argument("--max-error-mm", type=float, default=2.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    calibration = load_calibration(args.config)
    if calibration["status"] != "calibrated":
        raise SystemExit(
            f"Calibration status is {calibration['status']!r}, not 'calibrated'."
        )
    if args.max_rms_error_mm <= 0 or args.max_error_mm <= 0:
        raise SystemExit("Error limits must be > 0.")

    if args.collect:
        if args.samples.exists() and not args.overwrite:
            raise SystemExit(
                f"Validation samples already exist: {args.samples}. Use --overwrite."
            )
        config = replace(ROBOT_CONFIG, can_name=args.can)
        controller = PiperController(config=config)
        print(f"Connecting to Piper feedback on {args.can}; arm motion is not enabled.")
        controller.connect(enable=False)
        try:
            poses = collect_poses_interactively(
                controller,
                min_samples=args.min_samples,
                timeout_s=args.read_timeout_s,
                settle_s=args.settle_s,
            )
        finally:
            controller.disconnect()
        write_yaml(
            args.samples,
            make_samples_document(poses, purpose="validation", can_name=args.can),
        )
        print(f"Wrote {len(poses)} TCP validation samples to {args.samples}")
    else:
        poses = load_sample_poses(args.samples)

    metrics = evaluate_tcp(
        poses,
        calibration["offset_j6_tcp_mm"],
        calibration["reference_point_base_mm"],
    )
    print(
        f"validation RMS={metrics.reference_rms_error_mm:.3f} mm, "
        f"max={metrics.reference_max_error_mm:.3f} mm"
    )
    print(
        f"validation spread RMS={metrics.spread_rms_mm:.3f} mm, "
        f"max={metrics.spread_max_mm:.3f} mm"
    )
    if (
        metrics.reference_rms_error_mm > args.max_rms_error_mm
        or metrics.reference_max_error_mm > args.max_error_mm
    ):
        raise SystemExit("TCP validation failed: errors exceed the configured limits.")
    print("TCP validation passed.")


if __name__ == "__main__":
    main()

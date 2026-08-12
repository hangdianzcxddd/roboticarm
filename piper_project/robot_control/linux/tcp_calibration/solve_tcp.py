"""Solve a gripper TCP translation from fixed-point pivot samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_control.linux.tcp_calibration.calibration import (
    load_sample_poses,
    make_calibration_document,
    solve_fixed_point_tcp,
    write_yaml,
)


DEFAULT_INPUT = Path(__file__).with_name("tcp_samples.yaml")
DEFAULT_OUTPUT = Path(__file__).with_name("tcp.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve a gripper J6-to-TCP translation from pivot samples."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rms-error-mm", type=float, default=1.0)
    parser.add_argument("--max-error-mm", type=float, default=2.0)
    parser.add_argument(
        "--allow-poor-fit",
        action="store_true",
        help="return success even when residual errors exceed the limits",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_rms_error_mm <= 0 or args.max_error_mm <= 0:
        raise SystemExit("Error limits must be > 0.")

    poses = load_sample_poses(args.input)
    solution = solve_fixed_point_tcp(poses)
    document = make_calibration_document(
        solution,
        source_samples=args.input,
        sample_count=len(poses),
        rms_limit_mm=args.max_rms_error_mm,
        max_error_limit_mm=args.max_error_mm,
    )
    write_yaml(args.output, document)
    status = document["tcp_calibration"]["status"]
    x_mm, y_mm, z_mm = solution.offset_j6_tcp_mm
    print(f"status={status}")
    print(f"J6 -> gripper TCP offset: x={x_mm:.3f} y={y_mm:.3f} z={z_mm:.3f} mm")
    print(
        f"fit RMS={solution.rms_error_mm:.3f} mm, "
        f"max={solution.max_error_mm:.3f} mm, "
        f"rank={solution.matrix_rank}, condition={solution.condition_number:.3f}"
    )
    print(f"Wrote TCP calibration to {args.output}")
    if status != "calibrated" and not args.allow_poor_fit:
        raise SystemExit(
            "TCP fit exceeds error limits. Recollect samples or inspect the fixture. "
            "The output was written with status=poor_fit."
        )


if __name__ == "__main__":
    main()

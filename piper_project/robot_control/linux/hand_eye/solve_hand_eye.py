"""Solve hand-eye calibration from collected samples."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_control.linux.hand_eye.calibration import (
    evaluate_residuals,
    load_mode,
    load_samples,
    make_result_document,
    method_from_name,
    solve_hand_eye,
    write_json,
)


DEFAULT_SAMPLES = Path(__file__).with_name("hand_eye_samples.json")
DEFAULT_OUTPUT = Path(__file__).with_name("hand_eye_result.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenCV calibrateHandEye() on collected Piper samples."
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("eye_to_hand", "eye_in_hand"),
        help="override mode stored in the samples file",
    )
    parser.add_argument(
        "--method",
        choices=("tsai", "park", "horaud", "andreff", "daniilidis"),
        default="tsai",
    )
    parser.add_argument("--min-samples", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_samples(args.samples)
    mode = args.mode or load_mode(args.samples)
    if len(samples) < args.min_samples:
        raise SystemExit(
            f"Only {len(samples)} samples found; collect at least {args.min_samples}."
        )

    result = solve_hand_eye(
        samples,
        mode=mode,
        method=method_from_name(args.method),
    )
    residuals = evaluate_residuals(samples, result)
    write_json(
        args.output,
        make_result_document(
            result,
            source_samples=args.samples,
            method_name=args.method,
            residuals=residuals,
        ),
    )

    label = "camera -> base" if mode == "eye_to_hand" else "gripper -> camera"
    print(f"Solved {label} using {len(samples)} samples.")
    print(
        "translation_m="
        f"{tuple(round(float(value), 6) for value in result.translation_m)}"
    )
    print("rotation_matrix=")
    for row in result.rotation:
        print("  " + " ".join(f"{float(value): .9f}" for value in row))
    print(
        "residuals: "
        f"translation_rms={residuals.translation_rms_m * 1000.0:.3f} mm, "
        f"translation_max={residuals.translation_max_m * 1000.0:.3f} mm, "
        f"rotation_rms={residuals.rotation_rms_deg:.3f} deg, "
        f"rotation_max={residuals.rotation_max_deg:.3f} deg"
    )
    print(f"Wrote result: {args.output}")


if __name__ == "__main__":
    main()

"""Linux-side hand-eye sample server.

It receives target-to-camera chessboard poses from the Windows camera host,
reads the current Piper end pose, and appends one synchronized sample.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import socket
import sys
import time
from typing import Any

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.hand_eye.calibration import (
    HandEyeSample,
    format_end_pose,
    load_samples,
    make_sample,
    make_samples_document,
    write_json,
)
from robot_control.linux.tcp_calibration.calibration import read_fresh_end_pose
from robot_control.shared.hand_eye_protocol import (
    hand_eye_sample_from_message,
    make_hand_eye_ack_message,
    make_hand_eye_error_message,
    recv_json_line,
    send_json_line,
)


DEFAULT_OUTPUT = Path(__file__).with_name("hand_eye_samples.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive Windows chessboard poses and save synchronized Piper "
            "hand-eye calibration samples."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("eye_to_hand", "eye_in_hand"),
        default="eye_to_hand",
        help=(
            "eye_to_hand: camera fixed, gripper moves the checkerboard. "
            "eye_in_hand: camera on gripper, checkerboard fixed."
        ),
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--read-timeout-s", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=0.2)
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to an existing output file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    parser.add_argument(
        "--notes",
        help="optional note stored in the sample file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists() and not args.append and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}. Use --append or --overwrite."
        )
    if args.min_samples < 1:
        raise SystemExit("--min-samples must be >= 1")
    if args.read_timeout_s <= 0:
        raise SystemExit("--read-timeout-s must be > 0")
    if args.settle_s < 0:
        raise SystemExit("--settle-s must be >= 0")

    samples = load_samples(args.output) if args.append and args.output.exists() else []
    config = replace(ROBOT_CONFIG, can_name=args.can)
    controller = PiperController(config=config)

    print(f"Connecting to Piper feedback on {args.can}; arm motion is not enabled.")
    controller.connect(enable=False)
    try:
        serve(args, controller, samples)
    finally:
        controller.disconnect()


def serve(
    args: argparse.Namespace,
    controller: PiperController,
    samples: list[HandEyeSample],
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"Hand-eye sample server listening on {args.host}:{args.port}; "
            f"mode={args.mode}, existing_samples={len(samples)}",
            flush=True,
        )
        print("Move the arm, wait until stable, then press capture on Windows.")
        print("Press Ctrl-C here after collecting 20-30 valid samples.")

        try:
            while True:
                conn, addr = server.accept()
                with conn:
                    with conn.makefile("rb") as sock_file:
                        message = recv_json_line(sock_file)
                    if message is None:
                        continue
                    response = handle_message(args, controller, samples, message)
                    send_json_line(conn, response)
                    if len(samples) >= args.min_samples:
                        print(
                            f"{len(samples)} samples collected; enough for solving, "
                            "you can keep collecting or stop with Ctrl-C.",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\nStopped sample server.", flush=True)


def handle_message(
    args: argparse.Namespace,
    controller: PiperController,
    samples: list[HandEyeSample],
    message: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(message.get("sample_id")) if message.get("sample_id") else None
    try:
        command = hand_eye_sample_from_message(message)
        robot_pose = read_fresh_end_pose(
            controller,
            timeout_s=args.read_timeout_s,
            settle_s=args.settle_s,
        )
        sample = make_sample(len(samples) + 1, command, robot_pose)
        samples.append(sample)
        write_json(
            args.output,
            make_samples_document(
                samples,
                mode=args.mode,
                can_name=args.can,
                notes=args.notes,
            ),
        )
        print(
            f"Saved sample {sample.index}: id={sample.sample_id} "
            f"robot=({format_end_pose(robot_pose)}) "
            f"target_tvec_m={command.target_to_camera.tvec_m} "
            f"reproj={command.reprojection_error_px}",
            flush=True,
        )
        return make_hand_eye_ack_message(
            sample.sample_id,
            sample.index,
            saved=True,
            robot_pose={
                "x_mm": robot_pose.x_mm,
                "y_mm": robot_pose.y_mm,
                "z_mm": robot_pose.z_mm,
                "rx_deg": robot_pose.rx_deg,
                "ry_deg": robot_pose.ry_deg,
                "rz_deg": robot_pose.rz_deg,
            },
        )
    except Exception as exc:  # noqa: BLE001 - sent to the remote capture client
        print(f"Failed to save sample {sample_id}: {exc}", file=sys.stderr, flush=True)
        return make_hand_eye_error_message(sample_id, str(exc))


if __name__ == "__main__":
    main()

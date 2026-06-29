"""Eye-in-hand static grasp server.

This is the eye-in-hand companion to ``catch_server_1``. It keeps the existing
YOLO/OpenCV camera-point protocol and reuses the catch_server_1 motion sequence,
but computes a per-message camera -> base transform from:

    camera -> base = gripper -> base at capture pose @ camera -> gripper

The eye-in-hand calibration result is produced by ``solve_hand_eye --mode
eye_in_hand``. In the current project math, that solved matrix maps camera
coordinates into the gripper frame even though the JSON frame labels are kept
for backward compatibility with older docs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch import catch_server_1
from robot_control.linux.catch.eye_in_hand_mapping import (
    camera_to_base_transform_from_gripper_pose,
    load_camera_to_gripper_transform,
)
from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_hand_eye_result_document,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.hand_eye.calibration import format_end_pose
from robot_control.linux.tcp_calibration.calibration import read_fresh_end_pose
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    camera_point_from_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.shared.simulated_mapping import pose_to_dict


DEFAULT_EYE_IN_HAND_RESULT = (
    Path(__file__).resolve().parents[1]
    / "hand_eye"
    / "eye_in_hand_result_auto.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = catch_server_1.build_parser()
    parser.description = (
        "Receive camera XYZ over TCP and run catch_server_1's grasp sequence "
        "using an eye-in-hand camera mounted on the gripper."
    )
    parser.add_argument(
        "--eye-in-hand-result",
        type=Path,
        default=DEFAULT_EYE_IN_HAND_RESULT,
        help=(
            "eye_in_hand hand-eye result. The solved matrix is used as "
            "camera -> gripper for point mapping."
        ),
    )
    parser.add_argument(
        "--capture-pose-source",
        choices=("current", "ready", "configured"),
        default="current",
        help=(
            "gripper pose used for camera->base mapping. current reads robot "
            "feedback when the message arrives; ready uses --ready-* and "
            "--transfer-z-mm; configured uses --capture-* values."
        ),
    )
    parser.add_argument("--capture-read-timeout-s", type=float, default=1.0)
    parser.add_argument("--capture-settle-s", type=float, default=0.0)
    parser.add_argument("--capture-x-mm", type=float)
    parser.add_argument("--capture-y-mm", type=float)
    parser.add_argument("--capture-z-mm", type=float)
    parser.add_argument("--capture-rx-deg", type=float)
    parser.add_argument("--capture-ry-deg", type=float)
    parser.add_argument("--capture-rz-deg", type=float)
    parser.add_argument(
        "--ready-pose-source",
        choices=("args", "current", "configured"),
        default="args",
        help=(
            "pose used for startup_ready/ready_open/return_ready. args keeps "
            "catch_server_1 --ready-* plus fixed orientation; current reads the "
            "pose at server startup; configured uses --ready-pose-* values."
        ),
    )
    parser.add_argument("--ready-pose-read-timeout-s", type=float, default=1.0)
    parser.add_argument("--ready-pose-settle-s", type=float, default=0.0)
    parser.add_argument("--ready-pose-x-mm", type=float)
    parser.add_argument("--ready-pose-y-mm", type=float)
    parser.add_argument("--ready-pose-z-mm", type=float)
    parser.add_argument("--ready-pose-rx-deg", type=float)
    parser.add_argument("--ready-pose-ry-deg", type=float)
    parser.add_argument("--ready-pose-rz-deg", type=float)
    return parser


def validate_eye_in_hand_args(args: argparse.Namespace) -> None:
    catch_server_1.validate_args(args)
    if args.capture_read_timeout_s <= 0.0:
        raise ValueError("--capture-read-timeout-s must be > 0")
    if args.capture_settle_s < 0.0:
        raise ValueError("--capture-settle-s must be >= 0")
    if args.capture_pose_source == "configured":
        missing = [
            name
            for name in (
                "capture_x_mm",
                "capture_y_mm",
                "capture_z_mm",
                "capture_rx_deg",
                "capture_ry_deg",
                "capture_rz_deg",
            )
            if getattr(args, name) is None
        ]
        if missing:
            options = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise ValueError(f"--capture-pose-source configured requires {options}")
    if args.ready_pose_read_timeout_s <= 0.0:
        raise ValueError("--ready-pose-read-timeout-s must be > 0")
    if args.ready_pose_settle_s < 0.0:
        raise ValueError("--ready-pose-settle-s must be >= 0")
    if args.ready_pose_source == "configured":
        missing = [
            name
            for name in (
                "ready_pose_x_mm",
                "ready_pose_y_mm",
                "ready_pose_z_mm",
                "ready_pose_rx_deg",
                "ready_pose_ry_deg",
                "ready_pose_rz_deg",
            )
            if getattr(args, name) is None
        ]
        if missing:
            options = ", ".join("--" + name.replace("_", "-") for name in missing)
            raise ValueError(f"--ready-pose-source configured requires {options}")


def configured_ready_pose(args: argparse.Namespace) -> EndPose:
    return EndPose(
        float(args.ready_pose_x_mm),
        float(args.ready_pose_y_mm),
        float(args.ready_pose_z_mm),
        float(args.ready_pose_rx_deg),
        float(args.ready_pose_ry_deg),
        float(args.ready_pose_rz_deg),
    )


def apply_ready_pose_override(args: argparse.Namespace, ready_pose: EndPose) -> None:
    args.ready_x_mm = float(ready_pose.x_mm)
    args.ready_y_mm = float(ready_pose.y_mm)
    args.transfer_z_mm = float(ready_pose.z_mm)
    args.ready_rx_deg = float(ready_pose.rx_deg)
    args.ready_ry_deg = float(ready_pose.ry_deg)
    args.ready_rz_deg = float(ready_pose.rz_deg)
    args.eye_in_hand_ready_pose_override = True


def resolve_ready_pose_override(
    args: argparse.Namespace,
    controller: PiperController | None,
) -> EndPose | None:
    if args.ready_pose_source == "args":
        args.eye_in_hand_ready_pose_override = False
        return None
    if args.ready_pose_source == "configured":
        pose = configured_ready_pose(args)
    else:
        if controller is None:
            raise RuntimeError("--ready-pose-source current requires a controller")
        pose = read_fresh_end_pose(
            controller,
            timeout_s=float(args.ready_pose_read_timeout_s),
            settle_s=float(args.ready_pose_settle_s),
        )
    apply_ready_pose_override(args, pose)
    return pose


def ready_pose(args: argparse.Namespace) -> EndPose:
    if getattr(args, "eye_in_hand_ready_pose_override", False):
        return EndPose(
            float(args.ready_x_mm),
            float(args.ready_y_mm),
            float(args.transfer_z_mm),
            float(args.ready_rx_deg),
            float(args.ready_ry_deg),
            float(args.ready_rz_deg),
        )
    return catch_server_1.ready_pose(args)


def move_to_start_ready(controller: PiperController, args: argparse.Namespace) -> int:
    step = catch_server_1.SequenceStep(
        "startup_ready",
        ready_pose(args),
        args.transfer_mode,
    )
    catch_server_1.validate_steps(controller, (step,))
    print(
        "Moving to eye-in-hand startup ready pose: "
        f"{catch_server_1.format_pose(step.pose)}",
        flush=True,
    )
    send_count, _ = catch_server_1.execute_step(
        controller,
        args,
        step,
        speed=args.startup_speed,
    )
    return send_count


def capture_pose_for_message(
    args: argparse.Namespace,
    pose_reader: PiperController | None,
) -> tuple[EndPose, str]:
    if args.capture_pose_source == "ready":
        return catch_server_1.ready_pose(args), "ready_pose"
    if args.capture_pose_source == "configured":
        return (
            EndPose(
                float(args.capture_x_mm),
                float(args.capture_y_mm),
                float(args.capture_z_mm),
                float(args.capture_rx_deg),
                float(args.capture_ry_deg),
                float(args.capture_rz_deg),
            ),
            "configured_pose",
        )
    if pose_reader is None:
        raise RuntimeError(
            "--capture-pose-source current requires a connected Piper feedback reader"
        )
    pose = read_fresh_end_pose(
        pose_reader,
        timeout_s=float(args.capture_read_timeout_s),
        settle_s=float(args.capture_settle_s),
    )
    return pose, "current_feedback"


def camera_to_base_for_message(
    args: argparse.Namespace,
    pose_reader: PiperController | None,
    camera_to_gripper: np.ndarray,
) -> tuple[np.ndarray, EndPose, str]:
    capture_pose, capture_pose_source = capture_pose_for_message(args, pose_reader)
    camera_to_base = camera_to_base_transform_from_gripper_pose(
        capture_pose,
        camera_to_gripper,
    )
    return camera_to_base, capture_pose, capture_pose_source


def handle_message(
    message: dict[str, Any],
    *,
    args: argparse.Namespace,
    camera_to_gripper: np.ndarray,
    validator: PiperController,
    pose_reader: PiperController | None,
    controller: PiperController | None = None,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    if command.grasp_mode == "dynamic":
        raise ValueError(
            "catch_server_2 currently supports static/probe only; dynamic "
            "will be added on top of the per-message eye-in-hand mapping"
        )

    camera_to_base, capture_pose, capture_pose_source = camera_to_base_for_message(
        args,
        pose_reader,
        camera_to_gripper,
    )
    response = handle_message_with_ready_pose_override(
        message,
        args=args,
        camera_to_base=camera_to_base,
        validator=validator,
        controller=controller,
    )
    payload = response.get("payload")
    if isinstance(payload, dict):
        payload["mapping_source"] = "eye_in_hand"
        payload["eye_in_hand"] = {
            "result_path": str(args.eye_in_hand_result),
            "capture_pose_source": capture_pose_source,
            "capture_pose": pose_to_dict(capture_pose),
            "camera_to_gripper": matrix_to_nested_list(camera_to_gripper),
            "camera_to_base": matrix_to_nested_list(camera_to_base),
            "dynamic_ready": False,
        }
    return response


def handle_message_with_ready_pose_override(
    message: dict[str, Any],
    *,
    args: argparse.Namespace,
    camera_to_base: np.ndarray,
    validator: PiperController,
    controller: PiperController | None,
) -> dict[str, Any]:
    original_ready_pose = catch_server_1.ready_pose
    try:
        catch_server_1.ready_pose = ready_pose
        return catch_server_1.handle_message(
            message,
            args=args,
            camera_to_base=camera_to_base,
            validator=validator,
            controller=controller,
        )
    finally:
        catch_server_1.ready_pose = original_ready_pose


def matrix_to_nested_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def serve(args: argparse.Namespace) -> None:
    validate_eye_in_hand_args(args)
    hand_eye_document = load_hand_eye_result_document(args.eye_in_hand_result)
    camera_to_gripper = load_camera_to_gripper_transform(args.eye_in_hand_result)
    print(
        "Using eye-in-hand camera->gripper transform: "
        f"{args.eye_in_hand_result}",
        flush=True,
    )
    quality = hand_eye_quality_summary(hand_eye_document)
    if quality is not None:
        print(f"Hand-eye quality: {quality}", flush=True)

    config = replace(ROBOT_CONFIG, can_name=args.can)
    validator = PiperController(config=config)
    controller: PiperController | None = None
    feedback_reader: PiperController | None = None
    pose_reader: PiperController | None = None

    if args.execute:
        controller = PiperController(config=config)
        pose_reader = controller
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        ready_override = resolve_ready_pose_override(args, controller)
        if ready_override is not None:
            controller.validate_pose(ready_override)
            print(
                "Using eye-in-hand ready/start pose: "
                f"{format_end_pose(ready_override)}",
                flush=True,
            )
        if args.collision_protection_level is not None:
            levels = [args.collision_protection_level] * 6
            controller.set_collision_protection(levels)
            print(
                "Piper collision protection configured: "
                f"levels={levels}",
                flush=True,
            )
        catch_server_1.maybe_prepare_gripper(controller, args)
        if not args.no_start_ready:
            move_to_start_ready(controller, args)
    elif args.capture_pose_source == "current":
        feedback_reader = PiperController(config=config)
        pose_reader = feedback_reader
        print("Connecting to Piper feedback for current capture pose...", flush=True)
        feedback_reader.connect(enable=False)
        ready_override = resolve_ready_pose_override(args, feedback_reader)
        if ready_override is not None:
            print(
                "Using eye-in-hand ready pose for planning: "
                f"{format_end_pose(ready_override)}",
                flush=True,
            )
    else:
        ready_override = resolve_ready_pose_override(args, None)
        if ready_override is not None:
            print(
                "Using eye-in-hand ready pose for planning: "
                f"{format_end_pose(ready_override)}",
                flush=True,
            )

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"Listening on {args.host}:{args.port}, execute={args.execute}, "
                f"speed={args.speed}, capture_pose_source={args.capture_pose_source}, "
                f"fixed_orientation="
                f"({args.rx_deg:.3f}, {args.ry_deg:.3f}, {args.rz_deg:.3f})",
                flush=True,
            )
            print(f"Catch server 2 args: {args}", flush=True)
            while True:
                conn, addr = server.accept()
                with conn:
                    print(f"Client connected: {addr}", flush=True)
                    with conn.makefile("rb") as sock_file:
                        while True:
                            message = recv_json_line(sock_file)
                            if message is None:
                                break
                            command_id = str(message.get("command_id"))
                            try:
                                response = handle_message(
                                    message,
                                    args=args,
                                    camera_to_gripper=camera_to_gripper,
                                    validator=validator,
                                    pose_reader=pose_reader,
                                    controller=controller,
                                )
                            except Exception as exc:  # noqa: BLE001 - returned to client
                                response = make_error_message(command_id, str(exc))
                                print(f"Error: {exc}", flush=True)
                            send_json_line(conn, response)
    finally:
        if controller is not None:
            controller.disconnect()
        if feedback_reader is not None:
            feedback_reader.disconnect()


def main() -> None:
    try:
        serve(build_parser().parse_args())
    except (SafetyError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

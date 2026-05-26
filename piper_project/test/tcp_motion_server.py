"""Linux-side TCP server: camera XYZ command -> simulated robot pose -> move.

Run on Linux from ``piper_project``:

    python test/tcp_motion_server.py --host 0.0.0.0 --port 5005

Default mode is dry-run and will not move the robot. Add ``--execute`` only
after confirming the generated poses are safe for the current arm setup.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration.simulated_mapping import (
    SimulatedMappingConfig,
    camera_xyz_to_simulated_robot_pose,
    pose_to_dict,
)
from communication.protocol import (
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from config.robot_config import ROBOT_CONFIG
from robot.piper_controller import EndPose, PiperController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Receive camera XYZ over TCP and optionally move Piper arm."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=10)
    parser.add_argument("--mode", choices=("P", "L"), default="P")
    parser.add_argument("--base-x-mm", type=float, default=150.0)
    parser.add_argument("--base-y-mm", type=float, default=0.0)
    parser.add_argument("--base-z-mm", type=float, default=220.0)
    parser.add_argument("--scale-x", type=float, default=120.0)
    parser.add_argument("--scale-y", type=float, default=120.0)
    parser.add_argument("--scale-z", type=float, default=80.0)
    parser.add_argument("--max-delta-x-mm", type=float, default=40.0)
    parser.add_argument("--max-delta-y-mm", type=float, default=40.0)
    parser.add_argument("--max-delta-z-mm", type=float, default=30.0)
    parser.add_argument("--rx-deg", type=float, default=0.0)
    parser.add_argument("--ry-deg", type=float, default=85.0)
    parser.add_argument("--rz-deg", type=float, default=0.0)
    return parser


def mapping_config_from_args(args: argparse.Namespace) -> SimulatedMappingConfig:
    return SimulatedMappingConfig(
        base_x_mm=args.base_x_mm,
        base_y_mm=args.base_y_mm,
        base_z_mm=args.base_z_mm,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        scale_z=args.scale_z,
        max_delta_x_mm=args.max_delta_x_mm,
        max_delta_y_mm=args.max_delta_y_mm,
        max_delta_z_mm=args.max_delta_z_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )


def validate_or_execute_pose(
    pose: EndPose,
    execute: bool,
    speed: int,
    mode: str,
) -> None:
    controller = PiperController()
    controller.validate_pose(pose)
    if not execute:
        return

    with controller:
        controller.move_to_pose(pose, speed_percent=speed, move_mode=mode)


def handle_message(
    message: dict[str, Any],
    mapping_config: SimulatedMappingConfig,
    execute: bool,
    speed: int,
    mode: str,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    pose = camera_xyz_to_simulated_robot_pose(
        command.x_m,
        command.y_m,
        command.z_m,
        mapping_config,
    )
    validate_or_execute_pose(pose, execute=execute, speed=speed, mode=mode)
    print(
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"pose={pose}",
        flush=True,
    )
    return make_ack_message(command.command_id, pose_to_dict(pose), executed=execute)


def serve(args: argparse.Namespace) -> None:
    mapping_config = mapping_config_from_args(args)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"Listening on {args.host}:{args.port}, "
            f"execute={args.execute}, speed={args.speed}, mode={args.mode}",
            flush=True,
        )
        print(f"Robot safety config: {ROBOT_CONFIG}", flush=True)

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
                                mapping_config,
                                execute=args.execute,
                                speed=args.speed,
                                mode=args.mode,
                            )
                        except Exception as exc:  # noqa: BLE001 - sent to remote client
                            response = make_error_message(command_id, str(exc))
                            print(f"Error: {exc}", flush=True)
                        send_json_line(conn, response)


def main() -> None:
    serve(build_parser().parse_args())


if __name__ == "__main__":
    main()

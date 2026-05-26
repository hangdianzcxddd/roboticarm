"""JSON-lines protocol for sending camera points to the motion controller."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal


PROTOCOL_VERSION = 1
MessageType = Literal["camera_point", "ack", "error"]


@dataclass(frozen=True)
class CameraPointCommand:
    x_m: float
    y_m: float
    z_m: float
    u: int | None = None
    v: int | None = None
    depth_m: float | None = None
    source: str = "manual_click"
    command_id: str | None = None


def make_camera_point_message(command: CameraPointCommand) -> dict[str, Any]:
    command_id = command.command_id or str(time.time_ns())
    payload = asdict(command)
    payload["command_id"] = command_id
    return {
        "type": "camera_point",
        "version": PROTOCOL_VERSION,
        "command_id": command_id,
        "payload": payload,
    }


def make_ack_message(
    command_id: str | None,
    pose: dict[str, float],
    executed: bool,
) -> dict[str, Any]:
    return {
        "type": "ack",
        "version": PROTOCOL_VERSION,
        "command_id": command_id,
        "payload": {
            "executed": executed,
            "pose": pose,
        },
    }


def make_error_message(command_id: str | None, error: str) -> dict[str, Any]:
    return {
        "type": "error",
        "version": PROTOCOL_VERSION,
        "command_id": command_id,
        "payload": {"error": error},
    }


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(line: bytes) -> dict[str, Any]:
    message = json.loads(line.decode("utf-8"))
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    if message.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {message.get('version')!r}")
    if message.get("type") not in ("camera_point", "ack", "error"):
        raise ValueError(f"unsupported message type: {message.get('type')!r}")
    return message


def send_json_line(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def recv_json_line(sock_file: Any) -> dict[str, Any] | None:
    line = sock_file.readline()
    if not line:
        return None
    return decode_message(line)


def camera_point_from_message(message: dict[str, Any]) -> CameraPointCommand:
    if message.get("type") != "camera_point":
        raise ValueError(f"expected camera_point message, got {message.get('type')!r}")
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("camera_point payload must be an object")

    try:
        return CameraPointCommand(
            x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]),
            z_m=float(payload["z_m"]),
            u=int(payload["u"]) if payload.get("u") is not None else None,
            v=int(payload["v"]) if payload.get("v") is not None else None,
            depth_m=(
                float(payload["depth_m"])
                if payload.get("depth_m") is not None
                else None
            ),
            source=str(payload.get("source", "unknown")),
            command_id=(
                str(payload["command_id"])
                if payload.get("command_id") is not None
                else str(message.get("command_id"))
            ),
        )
    except KeyError as exc:
        raise ValueError(f"missing camera point field: {exc.args[0]}") from exc


def send_camera_point(
    host: str,
    port: int,
    command: CameraPointCommand,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    message = make_camera_point_message(command)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        send_json_line(sock, message)
        with sock.makefile("rb") as sock_file:
            response = recv_json_line(sock_file)
            if response is None:
                raise ConnectionError("server closed connection without a response")
            return response

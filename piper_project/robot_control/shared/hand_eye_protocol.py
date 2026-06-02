"""JSON-lines protocol for hand-eye calibration sample capture."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal


PROTOCOL_VERSION = 1
MessageType = Literal["hand_eye_sample", "hand_eye_ack", "hand_eye_error"]


@dataclass(frozen=True)
class PoseVector:
    """Rigid transform stored as OpenCV Rodrigues rotation plus translation."""

    rvec: tuple[float, float, float]
    tvec_m: tuple[float, float, float]


@dataclass(frozen=True)
class HandEyeSampleCommand:
    target_to_camera: PoseVector
    sample_id: str | None = None
    image_path: str | None = None
    reprojection_error_px: float | None = None
    detected_corners: int | None = None
    source: str = "windows_chessboard_capture"


def make_hand_eye_sample_message(command: HandEyeSampleCommand) -> dict[str, Any]:
    sample_id = command.sample_id or str(time.time_ns())
    payload = asdict(command)
    payload["sample_id"] = sample_id
    return {
        "type": "hand_eye_sample",
        "version": PROTOCOL_VERSION,
        "sample_id": sample_id,
        "payload": payload,
    }


def make_hand_eye_ack_message(
    sample_id: str | None,
    sample_index: int,
    saved: bool,
    robot_pose: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "type": "hand_eye_ack",
        "version": PROTOCOL_VERSION,
        "sample_id": sample_id,
        "payload": {
            "sample_index": sample_index,
            "saved": saved,
            "robot_pose": robot_pose,
        },
    }


def make_hand_eye_error_message(sample_id: str | None, error: str) -> dict[str, Any]:
    return {
        "type": "hand_eye_error",
        "version": PROTOCOL_VERSION,
        "sample_id": sample_id,
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
    if message.get("type") not in (
        "hand_eye_sample",
        "hand_eye_ack",
        "hand_eye_error",
    ):
        raise ValueError(f"unsupported message type: {message.get('type')!r}")
    return message


def send_json_line(sock: socket.socket, message: dict[str, Any]) -> None:
    sock.sendall(encode_message(message))


def recv_json_line(sock_file: Any) -> dict[str, Any] | None:
    line = sock_file.readline()
    if not line:
        return None
    return decode_message(line)


def hand_eye_sample_from_message(message: dict[str, Any]) -> HandEyeSampleCommand:
    if message.get("type") != "hand_eye_sample":
        raise ValueError(
            f"expected hand_eye_sample message, got {message.get('type')!r}"
        )
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("hand_eye_sample payload must be an object")

    target_to_camera = _pose_vector_from_payload(
        payload.get("target_to_camera"),
        "target_to_camera",
    )
    return HandEyeSampleCommand(
        target_to_camera=target_to_camera,
        sample_id=(
            str(payload["sample_id"])
            if payload.get("sample_id") is not None
            else str(message.get("sample_id"))
        ),
        image_path=(
            str(payload["image_path"]) if payload.get("image_path") is not None else None
        ),
        reprojection_error_px=(
            float(payload["reprojection_error_px"])
            if payload.get("reprojection_error_px") is not None
            else None
        ),
        detected_corners=(
            int(payload["detected_corners"])
            if payload.get("detected_corners") is not None
            else None
        ),
        source=str(payload.get("source", "unknown")),
    )


def send_hand_eye_sample(
    host: str,
    port: int,
    command: HandEyeSampleCommand,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    message = make_hand_eye_sample_message(command)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        send_json_line(sock, message)
        with sock.makefile("rb") as sock_file:
            response = recv_json_line(sock_file)
            if response is None:
                raise ConnectionError("server closed connection without a response")
            return response


def _pose_vector_from_payload(payload: Any, label: str) -> PoseVector:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return PoseVector(
        rvec=_tuple3(payload.get("rvec"), f"{label}.rvec"),
        tvec_m=_tuple3(payload.get("tvec_m"), f"{label}.tvec_m"),
    )


def _tuple3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numeric values")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 3 numeric values") from exc

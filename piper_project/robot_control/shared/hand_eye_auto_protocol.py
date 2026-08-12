"""JSON-lines protocol for automatic hand-eye chessboard capture."""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass
from typing import Any, Literal

from robot_control.shared.hand_eye_protocol import PoseVector


PROTOCOL_VERSION = 1
MessageType = Literal[
    "auto_hand_eye_request",
    "auto_hand_eye_response",
    "auto_hand_eye_error",
]
RequestAction = Literal["status", "capture"]


@dataclass(frozen=True)
class AutoHandEyeRequest:
    action: RequestAction
    sample_id: str | None = None
    min_margin_px: float = 30.0
    max_reprojection_error_px: float = 1.0
    stable_frames: int = 2
    stability_max_center_shift_px: float = 3.0
    save_image: bool = True


@dataclass(frozen=True)
class AutoHandEyeDetection:
    detected: bool
    qualified: bool
    reason: str | None = None
    target_to_camera: PoseVector | None = None
    reprojection_error_px: float | None = None
    detected_corners: int | None = None
    required_corners: int | None = None
    margin_px: float | None = None
    center_u: float | None = None
    center_v: float | None = None
    width_px: float | None = None
    height_px: float | None = None
    stable_count: int = 0
    image_path: str | None = None


def make_auto_hand_eye_request_message(
    request: AutoHandEyeRequest,
) -> dict[str, Any]:
    return {
        "type": "auto_hand_eye_request",
        "version": PROTOCOL_VERSION,
        "payload": asdict(request),
    }


def make_auto_hand_eye_response_message(
    detection: AutoHandEyeDetection,
) -> dict[str, Any]:
    return {
        "type": "auto_hand_eye_response",
        "version": PROTOCOL_VERSION,
        "payload": _detection_to_payload(detection),
    }


def make_auto_hand_eye_error_message(error: str) -> dict[str, Any]:
    return {
        "type": "auto_hand_eye_error",
        "version": PROTOCOL_VERSION,
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
        "auto_hand_eye_request",
        "auto_hand_eye_response",
        "auto_hand_eye_error",
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


def auto_hand_eye_request_from_message(message: dict[str, Any]) -> AutoHandEyeRequest:
    if message.get("type") != "auto_hand_eye_request":
        raise ValueError(
            f"expected auto_hand_eye_request, got {message.get('type')!r}"
        )
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("auto_hand_eye_request payload must be an object")
    action = str(payload.get("action", "status"))
    if action not in ("status", "capture"):
        raise ValueError("auto hand-eye action must be 'status' or 'capture'")
    stable_frames = int(payload.get("stable_frames", 2))
    if stable_frames < 1:
        raise ValueError("stable_frames must be >= 1")
    return AutoHandEyeRequest(
        action=action,  # type: ignore[arg-type]
        sample_id=(
            str(payload["sample_id"])
            if payload.get("sample_id") is not None
            else None
        ),
        min_margin_px=float(payload.get("min_margin_px", 30.0)),
        max_reprojection_error_px=float(
            payload.get("max_reprojection_error_px", 1.0)
        ),
        stable_frames=stable_frames,
        stability_max_center_shift_px=float(
            payload.get("stability_max_center_shift_px", 3.0)
        ),
        save_image=bool(payload.get("save_image", True)),
    )


def auto_hand_eye_detection_from_message(
    message: dict[str, Any],
) -> AutoHandEyeDetection:
    if message.get("type") == "auto_hand_eye_error":
        payload = message.get("payload")
        error = payload.get("error") if isinstance(payload, dict) else "unknown error"
        raise ValueError(str(error))
    if message.get("type") != "auto_hand_eye_response":
        raise ValueError(
            f"expected auto_hand_eye_response, got {message.get('type')!r}"
        )
    payload = message.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("auto_hand_eye_response payload must be an object")
    target_to_camera = (
        _pose_vector_from_payload(payload.get("target_to_camera"))
        if payload.get("target_to_camera") is not None
        else None
    )
    return AutoHandEyeDetection(
        detected=bool(payload.get("detected", False)),
        qualified=bool(payload.get("qualified", False)),
        reason=(
            str(payload["reason"]) if payload.get("reason") is not None else None
        ),
        target_to_camera=target_to_camera,
        reprojection_error_px=_optional_float(payload, "reprojection_error_px"),
        detected_corners=_optional_int(payload, "detected_corners"),
        required_corners=_optional_int(payload, "required_corners"),
        margin_px=_optional_float(payload, "margin_px"),
        center_u=_optional_float(payload, "center_u"),
        center_v=_optional_float(payload, "center_v"),
        width_px=_optional_float(payload, "width_px"),
        height_px=_optional_float(payload, "height_px"),
        stable_count=int(payload.get("stable_count", 0)),
        image_path=(
            str(payload["image_path"]) if payload.get("image_path") is not None else None
        ),
    )


def request_auto_hand_eye_detection(
    host: str,
    port: int,
    request: AutoHandEyeRequest,
    timeout_s: float = 5.0,
) -> AutoHandEyeDetection:
    message = make_auto_hand_eye_request_message(request)
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        send_json_line(sock, message)
        with sock.makefile("rb") as sock_file:
            response = recv_json_line(sock_file)
            if response is None:
                raise ConnectionError("vision server closed connection without response")
            return auto_hand_eye_detection_from_message(response)


def _detection_to_payload(detection: AutoHandEyeDetection) -> dict[str, Any]:
    payload = asdict(detection)
    if detection.target_to_camera is not None:
        payload["target_to_camera"] = asdict(detection.target_to_camera)
    return payload


def _pose_vector_from_payload(payload: Any) -> PoseVector:
    if not isinstance(payload, dict):
        raise ValueError("target_to_camera must be an object")
    return PoseVector(
        rvec=_tuple3(payload.get("rvec"), "target_to_camera.rvec"),
        tvec_m=_tuple3(payload.get("tvec_m"), "target_to_camera.tvec_m"),
    )


def _tuple3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numeric values")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 3 numeric values") from exc


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    if payload.get(key) is None:
        return None
    return float(payload[key])


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    if payload.get(key) is None:
        return None
    return int(payload[key])

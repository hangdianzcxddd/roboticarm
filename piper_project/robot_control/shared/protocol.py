"""JSON-lines protocol for sending camera points to the motion controller."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast


PROTOCOL_VERSION = 1
MessageType = Literal["camera_point", "ack", "error"]
CameraPointTask = Literal["out", "in"]
GraspMode = Literal["static", "dynamic", "probe"]


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
    task: CameraPointTask = "out"
    rank: int | None = None
    label: str | None = None
    confidence: float | None = None
    class_id: int | None = None
    bbox_xywh: tuple[float, float, float, float] | None = None
    grasp_mode: GraspMode = "static"
    t_arrive_s: float | None = None
    conveyor_speed_mm_s: float | None = None
    pick_x_mm: float | None = None
    pick_y_mm: float | None = None
    detected_x_mm: float | None = None
    detected_y_mm: float | None = None
    detected_z_mm: float | None = None
    detected_top_x_mm: float | None = None
    detected_top_y_mm: float | None = None
    detected_top_z_mm: float | None = None
    detected_top_camera_x_m: float | None = None
    detected_top_camera_y_m: float | None = None
    detected_top_camera_z_m: float | None = None
    mask_area_px: int | None = None
    mask_centroid_u: float | None = None
    mask_centroid_v: float | None = None
    mask_top_u: float | None = None
    mask_top_v: float | None = None
    mask_depth_median_m: float | None = None
    mask_depth_valid_count: int | None = None
    object_short_axis_yaw_deg: float | None = None
    object_yaw_source: str | None = None
    object_rect_width_px: float | None = None
    object_rect_height_px: float | None = None
    object_rect_angle_deg: float | None = None


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

    task = str(payload.get("task", "out")).lower()
    if task not in ("out", "in"):
        raise ValueError("camera point task must be 'out' or 'in'")
    grasp_mode = str(payload.get("grasp_mode", "static")).lower()
    if grasp_mode not in ("static", "dynamic", "probe"):
        raise ValueError("grasp mode must be 'static', 'dynamic', or 'probe'")

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
            task=cast(CameraPointTask, task),
            rank=int(payload["rank"]) if payload.get("rank") is not None else None,
            label=str(payload["label"]) if payload.get("label") is not None else None,
            confidence=(
                float(payload["confidence"])
                if payload.get("confidence") is not None
                else None
            ),
            class_id=(
                int(payload["class_id"])
                if payload.get("class_id") is not None
                else None
            ),
            bbox_xywh=(
                _tuple4_float(payload["bbox_xywh"], "bbox_xywh")
                if payload.get("bbox_xywh") is not None
                else None
            ),
            mask_area_px=_optional_int(payload, "mask_area_px"),
            mask_centroid_u=_optional_float(payload, "mask_centroid_u"),
            mask_centroid_v=_optional_float(payload, "mask_centroid_v"),
            mask_top_u=_optional_float(payload, "mask_top_u"),
            mask_top_v=_optional_float(payload, "mask_top_v"),
            mask_depth_median_m=_optional_float(payload, "mask_depth_median_m"),
            mask_depth_valid_count=_optional_int(payload, "mask_depth_valid_count"),
            object_short_axis_yaw_deg=_optional_float(
                payload,
                "object_short_axis_yaw_deg",
            ),
            object_yaw_source=(
                str(payload["object_yaw_source"])
                if payload.get("object_yaw_source") is not None
                else None
            ),
            object_rect_width_px=_optional_float(payload, "object_rect_width_px"),
            object_rect_height_px=_optional_float(payload, "object_rect_height_px"),
            object_rect_angle_deg=_optional_float(payload, "object_rect_angle_deg"),
            grasp_mode=cast(GraspMode, grasp_mode),
            t_arrive_s=_optional_float(payload, "t_arrive_s"),
            conveyor_speed_mm_s=_optional_float(payload, "conveyor_speed_mm_s"),
            pick_x_mm=_optional_float(payload, "pick_x_mm"),
            pick_y_mm=_optional_float(payload, "pick_y_mm"),
            detected_x_mm=_optional_float(payload, "detected_x_mm"),
            detected_y_mm=_optional_float(payload, "detected_y_mm"),
            detected_z_mm=_optional_float(payload, "detected_z_mm"),
            detected_top_x_mm=_optional_float(payload, "detected_top_x_mm"),
            detected_top_y_mm=_optional_float(payload, "detected_top_y_mm"),
            detected_top_z_mm=_optional_float(payload, "detected_top_z_mm"),
            detected_top_camera_x_m=_optional_float(
                payload,
                "detected_top_camera_x_m",
            ),
            detected_top_camera_y_m=_optional_float(
                payload,
                "detected_top_camera_y_m",
            ),
            detected_top_camera_z_m=_optional_float(
                payload,
                "detected_top_camera_z_m",
            ),
        )
    except KeyError as exc:
        raise ValueError(f"missing camera point field: {exc.args[0]}") from exc


def _tuple4_float(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly 4 numeric values")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 4 numeric values") from exc
    return cast(tuple[float, float, float, float], values)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    if payload.get(key) is None:
        return None
    return float(payload[key])


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    if payload.get(key) is None:
        return None
    return int(payload[key])


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

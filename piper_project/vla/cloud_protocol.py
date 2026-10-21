"""Cloud-inference protocols used by the LingBot Piper client.

The LingBot MessagePack protocol uses the RoboTwin observation keys expected
by the deployed checkpoint.  The older JSON/HTTP-compatible payload remains
available only when ``protocol=json`` is selected explicitly.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Callable

import cv2
import numpy as np
import requests

try:
    from .deploy.msgpack_numpy import Packer as LingbotPacker, unpackb as lingbot_unpackb
except ImportError:  # pragma: no cover
    LingbotPacker = None
    lingbot_unpackb = None


def _decode_lingbot_wire(raw: bytes) -> Any:
    """Decode LingBot's msgpack-numpy wire format and normalize byte keys."""
    if lingbot_unpackb is None:
        raise RuntimeError("msgpack is required for the LingBot WebSocket protocol")
    value = lingbot_unpackb(raw)

    def normalize(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                (key.decode("utf-8") if isinstance(key, bytes) else key): normalize(val)
                for key, val in item.items()
            }
        if isinstance(item, list):
            return [normalize(x) for x in item]
        if isinstance(item, tuple):
            return tuple(normalize(x) for x in item)
        return item

    return normalize(value)

try:  # websocket-client is optional until the WebSocket deployment is used.
    import websocket
except ImportError:  # pragma: no cover - exercised only on minimal installs
    websocket = None


def encode_image_to_b64(frame_bgr: np.ndarray, jpeg_quality: int = 90) -> str:
    """Encode the camera-top BGR frame as the RGB-model API's JPEG payload."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
        raise ValueError("frame_bgr must be an HxWx3 numpy array")
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("failed to JPEG encode camera_top")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def make_inference_payload(
    frame_bgr: np.ndarray,
    state: np.ndarray,
    instruction: str,
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Build the unchanged cloud request payload."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != 7:
        raise ValueError(f"Piper state must contain 7 values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Piper state contains NaN or infinity")
    return {
        "image_b64": encode_image_to_b64(frame_bgr, jpeg_quality),
        "state": values.astype(float).tolist(),
        "instruction": str(instruction),
    }


def make_lingbot_observation(
    frame_bgr: np.ndarray,
    state: np.ndarray,
    instruction: str,
    image_key: str = "observation.images.cam_high",
    mirror_wrist_images: bool = True,
) -> dict[str, Any]:
    """Build the RoboTwin-style LingBot MessagePack observation.

    When ``mirror_wrist_images`` is enabled, the one available camera frame is
    copied to both wrist-camera fields.  Images remain RGB HWC ``uint8``;
    resizing and CHW conversion are performed by the cloud FeatureTransform.
    """
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.size != 7:
        raise ValueError(f"Piper state must contain 7 values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Piper state contains NaN or infinity")
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be an HxWx3 image")
    rgb = np.ascontiguousarray(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        dtype=np.uint8,
    )
    observation: dict[str, Any] = {
        str(image_key): rgb,
        "observation.state": values,
        "task": str(instruction),
    }
    if mirror_wrist_images:
        observation.update({
            "observation.images.cam_high": rgb,
            "observation.images.cam_left_wrist": rgb.copy(),
            "observation.images.cam_right_wrist": rgb.copy(),
        })
    return observation


def request_cloud_action(
    server_url: str,
    frame_bgr: np.ndarray,
    state: np.ndarray,
    instruction: str,
    timeout_s: float,
    jpeg_quality: int = 90,
) -> tuple[np.ndarray, dict[str, Any]]:
    """POST one observation and return ``(action_chunk, raw_response)``."""
    payload = make_inference_payload(frame_bgr, state, instruction, jpeg_quality)
    response = requests.post(server_url, json=payload, timeout=float(timeout_s))
    response.raise_for_status()
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError("cloud response is not valid JSON") from exc
    if not isinstance(data, dict) or not data.get("ok", False):
        raise RuntimeError(f"cloud inference failed: {data}")
    if "action_chunk" not in data:
        raise RuntimeError("cloud response does not contain action_chunk")
    chunk = np.asarray(data["action_chunk"], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] == 0:
        raise RuntimeError(f"action_chunk must be a non-empty 2D array, got {chunk.shape}")
    if chunk.shape[1] not in (6, 7):
        raise RuntimeError(
            "LingBo Piper action_chunk must have 6 arm values or 7 arm+gripper "
            f"values, got shape {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise RuntimeError("action_chunk contains NaN or infinity")
    return chunk, data


def parse_action_response(raw: str | bytes | bytearray | dict[str, Any] | list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse one LingBot WebSocket response.

    The preferred response is ``{"ok": true, "action_chunk": [[...]]}``.
    For easier server bring-up, ``action`` and a bare JSON array are accepted
    as well.  The returned dictionary is always the decoded response (a bare
    array is wrapped as ``{"action_chunk": ...}``).
    """
    if isinstance(raw, (bytes, bytearray)):
        # LingBot's official socket protocol is MessagePack.  Decode it before
        # falling back to the older JSON-over-WebSocket contract.
        if lingbot_unpackb is not None:
            try:
                raw = _decode_lingbot_wire(bytes(raw))
            except Exception:
                try:
                    raw = bytes(raw).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("cloud WebSocket binary response is neither MessagePack nor UTF-8 JSON") from exc
        else:
            try:
                raw = bytes(raw).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RuntimeError("install msgpack to decode LingBot binary WebSocket responses") from exc
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # LingBot's server sends plain-text exception messages for some
            # application errors. Preserve the text; it is essential for
            # diagnosing a schema/model failure on the remote side.
            preview = raw if len(raw) <= 500 else raw[:500] + "..."
            raise RuntimeError(f"cloud WebSocket text response is not JSON: {preview!r}") from exc
    else:
        data = raw
    if isinstance(data, list):
        data = {"ok": True, "action_chunk": data}
    if isinstance(data, np.ndarray):
        data = {"ok": True, "action_chunk": data}
    if not isinstance(data, dict):
        raise RuntimeError(f"cloud WebSocket response must be an object or array, got {type(data).__name__}")
    if data.get("ok") is False:
        raise RuntimeError(f"cloud inference failed: {data}")
    chunk_data = data.get("action_chunk")
    if chunk_data is None:
        chunk_data = data.get("action")
    if chunk_data is None:
        chunk_data = data.get("actions")
    if chunk_data is None:
        chunk_data = data.get("pred_action")
    if chunk_data is None and isinstance(data.get("data"), dict):
        chunk_data = data["data"].get("action_chunk", data["data"].get("action"))
    if chunk_data is None:
        raise RuntimeError("cloud WebSocket response does not contain action_chunk or action")
    chunk = np.asarray(chunk_data, dtype=np.float32)
    # A single action row is a valid one-step chunk.
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] not in (6, 7):
        raise RuntimeError(f"action_chunk must be a non-empty Nx6/Nx7 array, got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise RuntimeError("action_chunk contains NaN or infinity")
    return chunk, data


class WebSocketCloudClient:
    """Synchronous request/response client for the LingBot inference socket."""

    def __init__(
        self,
        url: str,
        connect_timeout_s: float = 10.0,
        receive_timeout_s: float = 120.0,
        headers: list[str] | None = None,
        image_key: str = "observation.images.cam_high",
        mirror_wrist_images: bool = True,
        protocol: str = "lingbot_msgpack",
        websocket_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = str(url)
        self.connect_timeout_s = float(connect_timeout_s)
        self.receive_timeout_s = float(receive_timeout_s)
        self.headers = headers or []
        self.image_key = str(image_key)
        self.mirror_wrist_images = bool(mirror_wrist_images)
        self.protocol = str(protocol)
        self._factory = websocket_factory
        self._socket: Any | None = None
        self.metadata: Any = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        factory = self._factory
        if factory is None:
            if websocket is None:
                raise RuntimeError("websocket-client is required; install websocket-client>=1.8,<2")
            factory = websocket.create_connection
        try:
            self._socket = factory(
                self.url,
                timeout=self.connect_timeout_s,
                header=self.headers,
            )
            if hasattr(self._socket, "settimeout"):
                self._socket.settimeout(self.receive_timeout_s)
            if self.protocol == "lingbot_msgpack":
                self.metadata = self._recv_decoded()
        except Exception as exc:  # noqa: BLE001
            self._socket = None
            raise RuntimeError(f"WebSocket connection failed: {self.url}: {exc}") from exc

    def _recv_decoded(self) -> Any:
        raw = self._socket.recv()
        if raw == "":
            raise RuntimeError("LingBot server returned an empty text response (check server traceback/schema)")
        if raw is None:
            raise RuntimeError("LingBot server closed the WebSocket without an action response")
        if isinstance(raw, (bytes, bytearray)) and lingbot_unpackb is not None:
            try:
                return _decode_lingbot_wire(bytes(raw))
            except Exception:
                # A server-side exception may be sent as a binary UTF-8 text
                # frame by some WebSocket implementations.
                try:
                    return bytes(raw).decode("utf-8")
                except UnicodeDecodeError:
                    raise
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _send_binary(self, wire: bytes) -> None:
        """Send MessagePack as a WebSocket binary frame."""
        if hasattr(self._socket, "send_binary"):
            self._socket.send_binary(wire)
            return
        if websocket is None:
            raise RuntimeError("websocket-client is required for binary MessagePack frames")
        self._socket.send(wire, opcode=websocket.ABNF.OPCODE_BINARY)

    def send_observation(
        self,
        frame_bgr: np.ndarray,
        state: np.ndarray,
        instruction: str,
        jpeg_quality: int = 90,
        extra: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._socket is None:
            raise RuntimeError("WebSocket is not connected")
        values = np.asarray(state, dtype=np.float32).reshape(-1)
        if values.size != 7 or not np.isfinite(values).all():
            raise ValueError("Piper state must be a finite seven-value vector")
        if self.protocol == "lingbot_msgpack":
            if LingbotPacker is None:
                raise RuntimeError("install msgpack to use the LingBot WebSocket protocol")
            # The deployed RoboTwin checkpoint expects flat keys, RGB HWC
            # images, a one-dimensional state vector, and a string task.
            payload: dict[str, Any] = make_lingbot_observation(
                frame_bgr,
                values,
                instruction,
                image_key=self.image_key,
                mirror_wrist_images=self.mirror_wrist_images,
            )
            if extra:
                payload.update(extra)
            if hasattr(self, "last_payload"):
                self.last_payload = payload
            else:
                self.last_payload = payload
            wire = LingbotPacker().pack(payload)
        else:
            payload = make_inference_payload(frame_bgr, values, instruction, jpeg_quality)
            if extra:
                payload.update(extra)
            wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if extra:
            payload.update(extra)
        try:
            if self.protocol == "lingbot_msgpack":
                self._send_binary(wire)
            else:
                self._socket.send(wire)
            raw = self._recv_decoded() if self.protocol == "lingbot_msgpack" else self._socket.recv()
            if raw == "":
                raise RuntimeError("LingBot server returned an empty text response (check server traceback/schema)")
            return parse_action_response(raw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"WebSocket send/receive failed: {exc}") from exc

    def reset(self, robot_name: str) -> Any:
        """Send the optional LingBot episode-reset message."""
        if self._socket is None:
            raise RuntimeError("WebSocket is not connected")
        if self.protocol != "lingbot_msgpack" or LingbotPacker is None:
            raise RuntimeError("reset is supported only by the LingBot MessagePack protocol")
        self._send_binary(LingbotPacker().pack({
            "reset": True,
            "robo_name": str(robot_name),
            "path_to_pi_model": None,
        }))
        response = self._recv_decoded()
        if isinstance(response, str):
            raise RuntimeError(f"LingBot reset failed:\n{response}")
        if not isinstance(response, dict) or response.get("reset_ok") is not True:
            raise RuntimeError(f"unexpected LingBot reset response: {response!r}")
        return response

    def close(self) -> None:
        socket, self._socket = self._socket, None
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass

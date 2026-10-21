#!/usr/bin/env python3
"""Probe a LingBot WebSocket server with the official MessagePack schema.

This intentionally does not use Piper observation collection or the local
``image_b64/state/instruction`` protocol.  It mirrors the official deployment
client example:

    {
        "image": {"base_0_rgb": uint8 array shaped (1, 3, H, W)},
        "state": float32 array shaped (1, state_dim),
        "prompt": ["..."],
    }

Run from ``piper_project`` after the SSH tunnel is active::

    python -m vla.deploy.test_lingbot_official_ws \\
        --url ws://127.0.0.1:9330 --state-dim 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websocket
from websocket import WebSocketConnectionClosedException

try:
    from .msgpack_numpy import Packer, unpackb
except ImportError:  # support direct execution from this directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vla.deploy.msgpack_numpy import Packer, unpackb


def decode_wire(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray)):
        try:
            return unpackb(bytes(raw), raw=False)
        except Exception as exc:  # noqa: BLE001
            try:
                return bytes(raw).decode("utf-8")
            except UnicodeDecodeError:
                return f"<undecodable binary: {exc}>"
    return raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LingBot official WebSocket schema probe")
    parser.add_argument("--url", default="ws://127.0.0.1:9330")
    parser.add_argument("--state-dim", type=int, default=8,
                        help="state width to test; official example uses 8")
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--image-key", default="base_0_rgb")
    parser.add_argument("--prompt", default="测试 LingBot 官方输入格式")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--robot-name", default="piper")
    parser.add_argument("--api-key")
    parser.add_argument("--save-request", type=Path,
                        help="optional .npz path for inspecting generated arrays")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.state_dim <= 0 or args.height <= 0 or args.width <= 0:
        raise SystemExit("state-dim, height and width must be positive")

    # Official sample uses uint8 RGB BCHW input and a batched float32 state.
    image = np.zeros((1, 3, args.height, args.width), dtype=np.uint8)
    state = np.zeros((1, args.state_dim), dtype=np.float32)
    # Make the values non-trivial while keeping them finite.
    state[0, :] = np.linspace(0.0, 0.1, args.state_dim, dtype=np.float32)
    observation = {
        "image": {str(args.image_key): image},
        "state": state,
        "prompt": [str(args.prompt)],
    }

    if args.save_request:
        np.savez(args.save_request, image=image, state=state)

    headers = [f"Authorization: Api-Key {args.api_key}"] if args.api_key else []
    print("connecting:", args.url)
    print("request schema:")
    print("  image:", args.image_key, image.shape, image.dtype)
    print("  state:", state.shape, state.dtype)
    print("  prompt:", observation["prompt"])

    ws = websocket.create_connection(args.url, timeout=float(args.timeout), header=headers)
    try:
        metadata = decode_wire(ws.recv())
        print("metadata:", repr(metadata))

        packer = Packer()
        if args.reset:
            reset = {"reset": True, "robo_name": str(args.robot_name)}
            ws.send_binary(Packer().pack(request))
            try:
                reset_raw = ws.recv()
            except WebSocketConnectionClosedException as exc:
                raise RuntimeError(
                    "server closed the WebSocket after reset; reset schema is not accepted "
                    "or the server crashed (check server logs)"
                ) from exc
            reset_reply = decode_wire(reset_raw)
            print("reset reply:", repr(reset_reply))
            if reset_raw in (b"", "", None):
                raise RuntimeError(
                    "server returned an empty reset response; do not continue with this reset "
                    "message. Retry without --reset or inspect the server traceback."
                )

        request_bytes = packer.pack(observation)
        print("sending MessagePack bytes:", len(request_bytes))
        ws.send(request_bytes)

        try:
            raw_reply = ws.recv()
        except WebSocketConnectionClosedException as exc:
            raise RuntimeError(
                "server closed the WebSocket before returning an action; inspect the server "
                "traceback for the observation schema/model error"
            ) from exc
        print("reply wire type:", type(raw_reply).__name__)
        print("reply raw preview:", repr(raw_reply[:200] if isinstance(raw_reply, (bytes, str)) else raw_reply))
        reply = decode_wire(raw_reply)
        print("reply decoded:", repr(reply))
        if isinstance(reply, dict):
            print("reply keys:", list(reply.keys()))
        if isinstance(reply, np.ndarray):
            print("reply array:", reply.shape, reply.dtype)
        return 0
    finally:
        ws.close()


if __name__ == "__main__":
    raise SystemExit(main())

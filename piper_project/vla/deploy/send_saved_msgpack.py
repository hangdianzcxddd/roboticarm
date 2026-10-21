#!/usr/bin/env python3
"""Send the exact MessagePack bytes captured from the Piper client."""

from __future__ import annotations

import argparse
from pathlib import Path

import websocket

from .msgpack_numpy import unpackb


def decode(raw):
    if isinstance(raw, (bytes, bytearray)):
        try:
            return unpackb(bytes(raw), raw=False)
        except Exception:
            return bytes(raw)
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a saved LingBot MessagePack request")
    parser.add_argument("--msgpack", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:9330")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    wire = args.msgpack.read_bytes()
    print("request bytes:", len(wire))
    local = decode(wire)
    print("local decoded request:", repr(local))

    ws = websocket.create_connection(args.url, timeout=args.timeout)
    try:
        print("server metadata:", repr(decode(ws.recv())))
        ws.send(wire)
        reply = ws.recv()
        print("reply type:", type(reply).__name__)
        print("reply:", repr(decode(reply)))
    finally:
        ws.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

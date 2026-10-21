#!/usr/bin/env python3
"""Validate a saved LingBot observation and its MessagePack round trip."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .msgpack_numpy import Packer, unpackb
except ImportError:  # direct execution
    from vla.deploy.msgpack_numpy import Packer, unpackb


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def check_observation(obs: dict[str, Any], state_dim: int | None = None) -> list[str]:
    errors: list[str] = []
    if set(obs) != {"image", "state", "prompt"}:
        errors.append(f"top-level keys are {list(obs)}, expected image/state/prompt")
    images = obs.get("image")
    if not isinstance(images, dict):
        errors.append("image must be a mapping")
        images = {}
    for key in IMAGE_KEYS:
        if key not in images:
            errors.append(f"missing image key: {key}")
            continue
        value = np.asarray(images[key])
        if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
            errors.append(f"{key} shape {value.shape}, expected (1,3,H,W)")
        if value.dtype != np.uint8:
            errors.append(f"{key} dtype {value.dtype}, expected uint8")
        if value.size and (int(value.min()) < 0 or int(value.max()) > 255):
            errors.append(f"{key} values outside [0,255]")
    state = np.asarray(obs.get("state"))
    if state.ndim != 2 or state.shape[0] != 1:
        errors.append(f"state shape {state.shape}, expected (1,D)")
    if state.dtype != np.float32:
        errors.append(f"state dtype {state.dtype}, expected float32")
    if state_dim is not None and state.ndim == 2 and state.shape[1] != state_dim:
        errors.append(f"state dim {state.shape[1]}, expected {state_dim}")
    if not np.isfinite(state).all():
        errors.append("state contains NaN or infinity")
    prompt = obs.get("prompt")
    if not isinstance(prompt, list) or not prompt or not all(isinstance(x, str) for x in prompt):
        errors.append("prompt must be a non-empty list[str]")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LingBot observation MessagePack")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--msgpack", type=Path)
    parser.add_argument("--state-dim", type=int)
    args = parser.parse_args()

    data = np.load(args.npz)
    image_keys = [key[len("image_"):] for key in data.files if key.startswith("image_")]
    obs: dict[str, Any] = {
        "image": {key: data[f"image_{key}"] for key in image_keys},
        "state": data["state"],
        "prompt": data["prompt"].astype(str).tolist(),
    }
    errors = check_observation(obs, args.state_dim)
    print("local observation:")
    print("  image keys:", list(obs["image"]))
    for key, value in obs["image"].items():
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print("  state:", obs["state"].shape, obs["state"].dtype)
    print("  prompt:", obs["prompt"])
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(" -", error)
        return 2
    print("schema validation: OK")

    wire = Packer().pack(obs)
    decoded = unpackb(wire, raw=False)
    roundtrip_errors = check_observation(decoded, args.state_dim)
    if roundtrip_errors:
        print("MessagePack round-trip FAILED:")
        for error in roundtrip_errors:
            print(" -", error)
        return 3
    print("MessagePack round-trip: OK; bytes:", len(wire))

    if args.msgpack:
        saved = args.msgpack.read_bytes()
        if saved != wire:
            print("saved MessagePack bytes differ from re-encoded NPZ observation")
            print("saved bytes:", len(saved), "re-encoded bytes:", len(wire))
            return 4
        print("saved MessagePack byte comparison: OK")
        saved_decoded = unpackb(saved, raw=False)
        saved_errors = check_observation(saved_decoded, args.state_dim)
        if saved_errors:
            print("saved MessagePack validation FAILED:")
            for error in saved_errors:
                print(" -", error)
            return 5
        print("saved MessagePack decode: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

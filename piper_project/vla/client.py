#!/usr/bin/env python3
"""Send Piper observations to LingBot and receive cloud action chunks.

The MessagePack observation follows the deployed RoboTwin schema: three flat
image keys, ``observation.state`` and ``task``.  A single top camera is copied
to the two wrist fields.  The state is ``[j1..j6, gripper]``.

Run from ``piper_project`` after creating the SSH tunnel:

    ssh -N -L 9330:127.0.0.1:9330 user@cloud-host
    python -m vla.client --ws-url ws://127.0.0.1:9330

The default is dry-run.  Add ``--execute`` only after the printed actions and
limits have been checked with the arm in a safe, empty workspace.
"""

from __future__ import annotations

import argparse
import base64
import os
import json
from dataclasses import replace
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml
import cv2

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG

from .cloud_protocol import WebSocketCloudClient, make_lingbot_observation
from .lingbo_action_adapter import ActionAdapterConfig
from .piper_observation import CameraTop, capture_observation
from .piper_safety_filter import SafetyFilterConfig
from .static_grasp_executor import StaticGraspExecutor


DEFAULT_CONFIG = Path(__file__).with_name("config_piper.yaml")


def parse_float_list(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated values, got {len(values)}")
    return values


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"VLA config must be a mapping: {path}")
    return dict(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LingBo-VLA cloud client for Piper")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ws-url", "--server-url", dest="ws_url", help="WebSocket URL (server-url is a backwards-compatible alias)")
    parser.add_argument("--instruction")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--hz", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--timeout", type=float, help="WebSocket receive timeout in seconds")
    parser.add_argument("--connect-timeout", type=float)
    parser.add_argument("--reconnect", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-reconnects", type=int)
    parser.add_argument("--global-step", type=int)
    parser.add_argument("--robot-id")
    parser.add_argument(
        "--reset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reset the LingBot Piper FeatureTransform after connecting (default: enabled)",
    )
    parser.add_argument("--robot-name", default="piper")
    parser.add_argument("--protocol", choices=("lingbot_msgpack", "json"))
    parser.add_argument("--image-key", help="LingBot top-camera field (normally observation.images.cam_high)")
    parser.add_argument(
        "--mirror-wrist-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="copy camera_top into both wrist image fields",
    )
    parser.add_argument("--ws-user", help="optional WebSocket Basic Auth username")
    parser.add_argument("--ws-password", help="optional WebSocket Basic Auth password")
    parser.add_argument("--speed-percent", type=int)
    parser.add_argument("--execute-period-s", type=float)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--capture-only", action="store_true", help="capture and print observation, then exit")
    parser.add_argument("--dump-observation", type=Path, help="save observation arrays to this .npz path")
    parser.add_argument("--dump-image", type=Path, help="save captured camera_top image as .jpg")
    parser.add_argument("--dump-msgpack", type=Path, help="save the exact LingBot MessagePack request bytes")
    parser.add_argument("--can", default=None)
    parser.add_argument("--action-semantics", choices=("absolute", "delta"))
    parser.add_argument("--arm-unit", choices=("rad",))
    parser.add_argument("--gripper-unit", choices=("mm", "normalized"))
    parser.add_argument("--max-joint-delta-rad")
    return parser


def merged_options(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    # argparse uses underscores while YAML uses the same names.
    defaults = {
        "ws_url": "ws://127.0.0.1:9330",
        "instruction": "抓取目标物体",
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30,
        "jpeg_quality": 90,
        "hz": 2.0,
        "max_steps": 20,
        "connect_timeout_s": 10.0,
        # The first LingBot inference may include torch.compile/autotune work
        # and can take several minutes on a smaller GPU.
        "receive_timeout_s": 600.0,
        "reconnect": True,
        "max_reconnects": 3,
        "protocol": "lingbot_msgpack",
        "image_key": "observation.images.cam_high",
        "mirror_wrist_images": True,
        "reset_on_connect": True,
        "speed_percent": 5,
        "execute_period_s": 0.5,
        "action_semantics": "absolute",
        "arm_unit": "rad",
        "gripper_unit": "mm",
        "arm_scale": 1.0,
        "arm_bias": 0.0,
        "gripper_scale": 1.0,
        "gripper_bias": 0.0,
        "open_mm": 0.0,
        "close_mm": -100.0,
        "max_joint_delta_rad": (0.035, 0.035, 0.035, 0.05, 0.05, 0.08),
        "max_gripper_delta_mm": 10.0,
        "min_gripper_mm": -100.0,
        "max_gripper_mm": 0.0,
    }
    merged = {**defaults, **config}
    for name in (
        "ws_url", "instruction", "camera_index", "camera_width", "camera_height",
        "camera_fps", "hz", "max_steps", "speed_percent", "execute_period_s",
        "action_semantics", "arm_unit", "gripper_unit", "protocol", "image_key",
    ):
        value = getattr(args, name, None)
        if value is not None:
            merged[name] = value
    if args.timeout is not None:
        merged["receive_timeout_s"] = args.timeout
    if args.connect_timeout is not None:
        merged["connect_timeout_s"] = args.connect_timeout
    if args.reconnect is not None:
        merged["reconnect"] = args.reconnect
    if args.max_reconnects is not None:
        merged["max_reconnects"] = args.max_reconnects
    if args.mirror_wrist_images is not None:
        merged["mirror_wrist_images"] = args.mirror_wrist_images
    if args.reset is not None:
        merged["reset_on_connect"] = args.reset
    if args.max_joint_delta_rad is not None:
        merged["max_joint_delta_rad"] = parse_float_list(args.max_joint_delta_rad, 6)
    else:
        merged["max_joint_delta_rad"] = tuple(float(x) for x in merged["max_joint_delta_rad"])
    return merged


def make_adapters(options: dict[str, Any]) -> tuple[ActionAdapterConfig, SafetyFilterConfig]:
    adapter = ActionAdapterConfig(
        semantics=str(options["action_semantics"]),
        arm_unit=str(options["arm_unit"]),
        gripper_unit=str(options["gripper_unit"]),
        arm_scale=float(options["arm_scale"]),
        arm_bias=float(options["arm_bias"]),
        gripper_scale=float(options["gripper_scale"]),
        gripper_bias=float(options["gripper_bias"]),
        open_mm=float(options["open_mm"]),
        close_mm=float(options["close_mm"]),
    )
    safety = SafetyFilterConfig(
        max_joint_delta_rad=tuple(options["max_joint_delta_rad"]),
        max_gripper_delta_mm=float(options["max_gripper_delta_mm"]),
        min_gripper_mm=float(options["min_gripper_mm"]),
        max_gripper_mm=float(options["max_gripper_mm"]),
    )
    return adapter, safety


def run(args: argparse.Namespace) -> None:
    options = merged_options(load_config(args.config), args)
    can_name = ROBOT_CONFIG.can_name if args.can is None else args.can
    robot_config = ROBOT_CONFIG if can_name == ROBOT_CONFIG.can_name else replace(ROBOT_CONFIG, can_name=can_name)
    adapter_config, safety_config = make_adapters(options)
    controller = PiperController(config=robot_config)
    camera: CameraTop | None = None
    arm_enabled_for_execution = False
    period_s = 1.0 / max(float(options["hz"]), 1e-6)
    try:
        # Establish SDK feedback first, but do not release/enable the arm until
        # the cloud connection and episode reset have both succeeded.
        controller.connect(enable=False)
        camera = CameraTop(
            int(options["camera_index"]),
            int(options["camera_width"]),
            int(options["camera_height"]),
            int(options["camera_fps"]),
        )
        print(f"LingBo WebSocket: {options['ws_url']}")
        print("Observation: camera_top + Piper [j1..j6, gripper]")
        print(f"Action semantics: {adapter_config.semantics}; execute={args.execute}")
        if args.capture_only or args.dump_observation or args.dump_image or args.dump_msgpack:
            observation = capture_observation(controller, camera)
            lingbot_obs = make_lingbot_observation(
                observation.frame_bgr,
                observation.state_vector,
                str(options["instruction"]),
                image_key=str(options["image_key"]),
                mirror_wrist_images=bool(options["mirror_wrist_images"]),
            )
            image_keys = [key for key in lingbot_obs if key.startswith("observation.images.")]
            image_array = lingbot_obs["observation.images.cam_high"]
            state_array = lingbot_obs["observation.state"]
            print("observation fields:", list(lingbot_obs))
            print("image fields:", image_keys)
            print("image shape/dtype/min/max:", image_array.shape, image_array.dtype, int(image_array.min()), int(image_array.max()))
            print("state shape/dtype/values:", state_array.shape, state_array.dtype, state_array.tolist())
            print("task:", lingbot_obs["task"])
            if args.dump_image:
                args.dump_image.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.dump_image), observation.frame_bgr):
                    raise RuntimeError(f"failed to save image: {args.dump_image}")
                print("saved image:", args.dump_image)
            if args.dump_observation:
                args.dump_observation.parent.mkdir(parents=True, exist_ok=True)
                arrays = {key: lingbot_obs[key] for key in image_keys}
                arrays.update({
                    "observation.state": state_array,
                    "task": np.asarray(lingbot_obs["task"], dtype=str),
                })
                np.savez_compressed(args.dump_observation, **arrays)
                print("saved observation arrays:", args.dump_observation)
            if args.dump_msgpack:
                if str(options["protocol"]) != "lingbot_msgpack":
                    raise ValueError("--dump-msgpack requires --protocol lingbot_msgpack")
                from .deploy.msgpack_numpy import Packer

                args.dump_msgpack.parent.mkdir(parents=True, exist_ok=True)
                wire = Packer().pack(lingbot_obs)
                args.dump_msgpack.write_bytes(wire)
                print("saved exact MessagePack request:", args.dump_msgpack, "bytes:", len(wire))
            if args.capture_only:
                return
        executor = StaticGraspExecutor(
            controller,
            safety_config,
            speed_percent=int(options["speed_percent"]),
            period_s=float(options["execute_period_s"]),
            execute=args.execute,
        )
        cloud = WebSocketCloudClient(
            str(options["ws_url"]),
            connect_timeout_s=float(options["connect_timeout_s"]),
            receive_timeout_s=float(options["receive_timeout_s"]),
            protocol=str(options["protocol"]),
            image_key=str(options["image_key"]),
            mirror_wrist_images=bool(options["mirror_wrist_images"]),
            headers=(
                ["Authorization: Basic " + base64.b64encode(
                    f"{args.ws_user or os.getenv('LINGBO_API_USER')}:{args.ws_password or os.getenv('LINGBO_API_PASSWORD')}".encode()
                ).decode()]
                if (args.ws_user or os.getenv("LINGBO_API_USER")) and (args.ws_password or os.getenv("LINGBO_API_PASSWORD"))
                else []
            ),
        )
        cloud.connect()
        print("LingBot metadata:", cloud.metadata)
        if bool(options["reset_on_connect"]):
            print("LingBot reset reply:", cloud.reset(args.robot_name))
        if args.execute:
            controller.enable()
            arm_enabled_for_execution = True
            print("Piper enabled after LingBot connection and reset succeeded.")
        try:
            reconnects = 0
            for step in range(int(options["max_steps"])):
                loop_started = time.monotonic()
                observation = capture_observation(controller, camera)
                state = observation.state_vector
                print(f"\n===== step {step} =====")
                print("state:", state.astype(float).tolist())
                extra: dict[str, Any] = {}
                if args.global_step is not None:
                    extra["global_step"] = int(args.global_step)
                if args.robot_id:
                    extra["robot_id"] = str(args.robot_id)
                try:
                    chunk, response = cloud.send_observation(
                        observation.frame_bgr,
                        state,
                        str(options["instruction"]),
                        int(options["jpeg_quality"]),
                        extra=extra,
                    )
                    reconnects = 0
                except RuntimeError as exc:
                    if not bool(options["reconnect"]) or reconnects >= int(options["max_reconnects"]):
                        raise
                    reconnects += 1
                    print(f"WebSocket error ({exc}); reconnecting {reconnects}/{options['max_reconnects']}...")
                    cloud.close()
                    time.sleep(min(2.0 * reconnects, 5.0))
                    cloud.connect()
                    continue
                print("cloud action_chunk:", tuple(chunk.shape), "response_keys:", tuple(response))
                results = executor.execute_chunk(chunk, adapter_config)
                for result in results:
                    print(
                        f"chunk[{result['index']}] "
                        f"safe_joints_rad={result['safe_joints_rad']} "
                        f"gripper_mm={result['safe_gripper_mm']} "
                        f"execute={result['executed']}"
                    )
                elapsed = time.monotonic() - loop_started
                if elapsed < period_s:
                    time.sleep(period_s - elapsed)
        finally:
            cloud.close()
    except KeyboardInterrupt:
        print("Stopping LingBo Piper client.")
    finally:
        if camera is not None:
            camera.close()
        if arm_enabled_for_execution:
            try:
                controller.hold_current_position(speed_percent=int(options["speed_percent"]), command_duration_s=0.5)
            except Exception as exc:  # noqa: BLE001
                print(f"Could not hold current position during shutdown: {exc}")
        controller.disconnect()


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

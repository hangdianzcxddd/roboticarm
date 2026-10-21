#!/usr/bin/env python3
import argparse
import base64
import time
from typing import List

import cv2
import numpy as np
import requests

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# 根据你 shake_hands 数据集统计出来的 action 范围
ACTION_MIN = np.array(
    [-19.717787, -100.000000, 2.268603, -4.490501, -5.310200, 0.000000],
    dtype=np.float32,
)
ACTION_MAX = np.array(
    [2.487932, 11.504425, 99.818512, 79.274612, 14.879075, 37.289494],
    dtype=np.float32,
)


DEFAULT_MAX_DELTA = np.array(
    [0.5, 2.0, 2.0, 2.0, 0.5, 1.0],
    dtype=np.float32,
)


DEFAULT_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_float_list(text: str, expected_len: int) -> np.ndarray:
    values = [float(x.strip()) for x in text.split(",")]
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float32)


def encode_image_to_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def infer_joint_keys(robot, obs: dict) -> List[str]:
    # 优先使用常见 SO-101/SO-100 关节名
    if all(k in obs for k in DEFAULT_JOINT_KEYS):
        return DEFAULT_JOINT_KEYS

    # 其次尝试使用 robot.action_features
    if hasattr(robot, "action_features"):
        keys = list(robot.action_features.keys())
        pos_keys = [k for k in keys if k.endswith(".pos")]
        if len(pos_keys) >= 6:
            return pos_keys[:6]
        if len(keys) >= 6:
            return keys[:6]

    # 最后从 observation 里找 .pos
    pos_keys = [k for k in obs.keys() if k.endswith(".pos")]
    if len(pos_keys) >= 6:
        return pos_keys[:6]

    raise RuntimeError(
        "Cannot infer 6 joint keys from robot observation. "
        f"Available observation keys: {list(obs.keys())}"
    )


def read_state_6d(robot, joint_keys: List[str]) -> np.ndarray:
    obs = robot.get_observation()
    state = np.asarray([float(obs[k]) for k in joint_keys], dtype=np.float32)
    return state


def request_cloud_action(
    server_url: str,
    frame_bgr: np.ndarray,
    state: np.ndarray,
    task: str,
    timeout: float,
) -> np.ndarray:
    payload = {
        "image_b64": encode_image_to_b64(frame_bgr),
        "state": state.astype(float).tolist(),
        "instruction": task,
    }

    resp = requests.post(server_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok", False):
        raise RuntimeError(f"Cloud inference failed: {data}")

    action_chunk = np.asarray(data["action_chunk"], dtype=np.float32)

    if action_chunk.ndim != 2:
        raise RuntimeError(f"Expected action_chunk to be 2D, got shape {action_chunk.shape}")

    if action_chunk.shape[1] != 6:
        raise RuntimeError(f"Expected action_dim=6, got shape {action_chunk.shape}")

    return action_chunk


def safety_filter(
    target_action: np.ndarray,
    current_state: np.ndarray,
    max_delta: np.ndarray,
) -> np.ndarray:
    target_action = np.asarray(target_action, dtype=np.float32).reshape(6)
    current_state = np.asarray(current_state, dtype=np.float32).reshape(6)

    # 1. 限制到数据集动作范围内
    target_action = np.clip(target_action, ACTION_MIN, ACTION_MAX)

    # 2. 限制单步变化量，防止突然大幅运动
    delta = target_action - current_state
    delta = np.clip(delta, -max_delta, max_delta)

    safe_action = current_state + delta

    # 3. 再次限幅
    safe_action = np.clip(safe_action, ACTION_MIN, ACTION_MAX)

    return safe_action.astype(np.float32)


def send_action_6d(robot, joint_keys: List[str], action_6d: np.ndarray):
    action_6d = np.asarray(action_6d, dtype=np.float32).reshape(6)
    action_dict = {k: float(v) for k, v in zip(joint_keys, action_6d)}
    return robot.send_action(action_dict)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:8000/infer")
    parser.add_argument("--robot_port", type=str, required=True)
    parser.add_argument("--robot_id", type=str, default="follower_so101")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--task", type=str, default="shake hands")

    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)

    # 默认不执行，只打印。确认安全后再加 --execute
    parser.add_argument("--execute", action="store_true")

    # 一次请求云端会返回 50 个 action，这里先只执行前几个
    parser.add_argument("--exec_chunk_steps", type=int, default=1)

    # 单步最大变化量
    parser.add_argument(
        "--max_delta",
        type=str,
        default="0.5,2.0,2.0,2.0,0.5,1.0",
        help="Comma-separated 6 values, e.g. 0.5,2,2,2,0.5,1",
    )

    args = parser.parse_args()

    max_delta = parse_float_list(args.max_delta, 6)

    print("Connecting SO-101 follower...")
    robot_cfg = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
    )
    robot = SO101Follower(robot_cfg)
    robot.connect()

    print("Opening camera...")
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        robot.disconnect()
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    # 读一次 observation，自动推断关节 key
    obs = robot.get_observation()
    print("Observation keys:", list(obs.keys()))

    joint_keys = infer_joint_keys(robot, obs)
    print("Using joint keys:", joint_keys)

    print("Server URL:", args.server_url)
    print("Task:", args.task)
    print("Execute:", args.execute)
    print("Max delta:", max_delta.tolist())
    print("Action min:", ACTION_MIN.tolist())
    print("Action max:", ACTION_MAX.tolist())

    if not args.execute:
        print("\n[DRY RUN] 当前是只打印，不会发送动作给机械臂。")
        print("确认 first_action / safe_action 合理后，再加 --execute 执行。\n")
    else:
        print("\n[EXECUTE MODE] 将会向机械臂发送动作。请确保周围安全，手放急停。\n")
        time.sleep(3)

    dt = 1.0 / max(args.hz, 1e-6)

    try:
        for step in range(args.max_steps):
            loop_t0 = time.time()

            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read camera frame, skipping.")
                time.sleep(dt)
                continue

            current_state = read_state_6d(robot, joint_keys)

            print(f"\n===== Step {step} =====")
            print("current_state:", current_state.tolist())

            action_chunk = request_cloud_action(
                server_url=args.server_url,
                frame_bgr=frame_bgr,
                state=current_state,
                task=args.task,
                timeout=args.timeout,
            )

            print("cloud action_chunk shape:", action_chunk.shape)
            print("cloud first_action:", action_chunk[0].tolist())

            n_exec = min(args.exec_chunk_steps, len(action_chunk))

            for j in range(n_exec):
                # 每执行一帧前重新读当前状态，做安全限幅
                current_state = read_state_6d(robot, joint_keys)
                target_action = action_chunk[j]
                safe_action = safety_filter(target_action, current_state, max_delta)

                print(f"chunk[{j}] target:", target_action.tolist())
                print(f"chunk[{j}] safe  :", safe_action.tolist())

                if args.execute:
                    sent_action = send_action_6d(robot, joint_keys, safe_action)
                    print("sent_action:", sent_action)

                time.sleep(dt)

            elapsed = time.time() - loop_t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping robot client.")

    finally:
        print("Closing camera and disconnecting robot.")
        cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
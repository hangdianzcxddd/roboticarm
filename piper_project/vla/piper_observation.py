"""Piper and camera-top observation collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG, PiperController
from robot_control.linux.catch.catch_server import feedback_to_pose


@dataclass(frozen=True)
class PiperObservation:
    frame_bgr: np.ndarray
    arm_position_rad: np.ndarray
    effector_position_mm: float
    tcp_pose: Any | None
    timestamp_s: float

    @property
    def state_vector(self) -> np.ndarray:
        """LingBo Piper state: six joints followed by one gripper value."""
        return np.concatenate(
            (self.arm_position_rad, np.asarray([self.effector_position_mm], dtype=np.float32))
        ).astype(np.float32)


def gripper_position_mm(feedback: Any) -> float:
    state = getattr(feedback, "gripper_state", feedback)
    raw = getattr(state, "grippers_angle", None)
    if raw is None:
        raise RuntimeError("Piper gripper feedback has no grippers_angle")
    return float(raw) * 0.001


def joint_positions_rad(feedback: Any) -> tuple[float, ...]:
    state = getattr(feedback, "joint_state", None)
    if state is None:
        raise RuntimeError("Piper joint feedback has no joint_state")
    raw = tuple(getattr(state, f"joint_{index}", None) for index in range(1, 7))
    if any(value is None for value in raw):
        raise RuntimeError("Piper joint feedback does not contain all six joints")
    return tuple(float(value) / RAD_TO_MILLI_DEG for value in raw)


def read_piper_state(controller: PiperController) -> tuple[np.ndarray, float, Any | None]:
    joints = joint_positions_rad(controller.get_joint_state())
    gripper = gripper_position_mm(controller.get_gripper_state())
    tcp_pose = feedback_to_pose(controller.get_end_pose())
    values = np.asarray(joints, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("Piper joint feedback contains NaN or infinity")
    return values, gripper, tcp_pose


class CameraTop:
    """OpenCV camera wrapper for the only enabled LingBo camera."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.cap = cv2.VideoCapture(int(index))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera_top index {index}")

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("failed to read camera_top frame")
        return frame

    def close(self) -> None:
        self.cap.release()


def capture_observation(controller: PiperController, camera: CameraTop) -> PiperObservation:
    frame = camera.read()
    joints, gripper, tcp_pose = read_piper_state(controller)
    import time

    return PiperObservation(frame, joints, gripper, tcp_pose, time.time())

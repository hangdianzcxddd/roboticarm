"""Minimal, deterministic safety filter for LingBo joint actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_control.linux.config.robot_config import ROBOT_CONFIG

from .lingbo_action_adapter import PiperAction


@dataclass(frozen=True)
class SafetyFilterConfig:
    max_joint_delta_rad: tuple[float, ...] = (0.035, 0.035, 0.035, 0.05, 0.05, 0.08)
    max_gripper_delta_mm: float = 10.0
    action_min_rad: tuple[float, ...] | None = None
    action_max_rad: tuple[float, ...] | None = None
    min_gripper_mm: float = -100.0
    max_gripper_mm: float = 0.0


def _joint_limits(config: SafetyFilterConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.action_min_rad is not None and config.action_max_rad is not None:
        return np.asarray(config.action_min_rad), np.asarray(config.action_max_rad)
    return (
        np.asarray([x.min_value for x in ROBOT_CONFIG.joint_limits_rad], dtype=np.float32),
        np.asarray([x.max_value for x in ROBOT_CONFIG.joint_limits_rad], dtype=np.float32),
    )


def filter_action(
    action: PiperAction,
    current_joints_rad: np.ndarray,
    config: SafetyFilterConfig,
    current_gripper_mm: float | None = None,
) -> PiperAction:
    current = np.asarray(current_joints_rad, dtype=np.float32).reshape(6)
    target = np.asarray(action.joints_rad, dtype=np.float32).reshape(6)
    if not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Piper action or feedback contains NaN/infinity")
    lower, upper = _joint_limits(config)
    if lower.shape != (6,) or upper.shape != (6,):
        raise ValueError("Piper joint limits must contain six values")
    target = np.clip(target, lower, upper)
    max_delta = np.asarray(config.max_joint_delta_rad, dtype=np.float32).reshape(6)
    safe = current + np.clip(target - current, -max_delta, max_delta)
    safe = np.clip(safe, lower, upper).astype(np.float32)
    gripper = action.gripper_mm
    if gripper is not None:
        gripper = float(np.clip(gripper, config.min_gripper_mm, config.max_gripper_mm))
        if current_gripper_mm is not None:
            delta = float(
                np.clip(
                    gripper - float(current_gripper_mm),
                    -float(config.max_gripper_delta_mm),
                    float(config.max_gripper_delta_mm),
                )
            )
            gripper = float(current_gripper_mm) + delta
            gripper = float(np.clip(gripper, config.min_gripper_mm, config.max_gripper_mm))
    return PiperAction(safe, gripper)

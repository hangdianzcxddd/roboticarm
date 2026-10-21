"""Convert a LingBo action row into the Piper action representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gripper_adapter import to_piper_gripper_mm


@dataclass(frozen=True)
class PiperAction:
    joints_rad: np.ndarray
    gripper_mm: float | None


@dataclass(frozen=True)
class ActionAdapterConfig:
    semantics: str = "absolute"
    arm_unit: str = "rad"
    gripper_unit: str = "mm"
    arm_scale: float = 1.0
    arm_bias: float = 0.0
    gripper_scale: float = 1.0
    gripper_bias: float = 0.0
    open_mm: float = 0.0
    close_mm: float = -100.0


def adapt_action_row(
    row: np.ndarray,
    current_joints_rad: np.ndarray,
    config: ActionAdapterConfig,
) -> PiperAction:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    current = np.asarray(current_joints_rad, dtype=np.float32).reshape(-1)
    if current.size != 6:
        raise ValueError(f"current Piper arm state must have 6 values, got {current.size}")
    if values.size not in (6, 7):
        raise ValueError(f"LingBo action row must have 6 or 7 values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("LingBo action row contains NaN or infinity")
    if config.arm_unit != "rad":
        raise ValueError("only arm_unit='rad' is supported by this Piper adapter")
    arm = values[:6] * float(config.arm_scale) + float(config.arm_bias)
    if config.semantics == "delta":
        arm = current + arm
    elif config.semantics != "absolute":
        raise ValueError(f"unsupported action semantics: {config.semantics}")
    gripper = None
    if values.size == 7:
        raw = float(values[6]) * float(config.gripper_scale) + float(config.gripper_bias)
        gripper = to_piper_gripper_mm(
            raw,
            unit=config.gripper_unit,
            open_mm=config.open_mm,
            close_mm=config.close_mm,
        )
    return PiperAction(arm.astype(np.float32), gripper)


def adapt_action_chunk(
    chunk: np.ndarray,
    current_joints_rad: np.ndarray,
    config: ActionAdapterConfig,
) -> list[PiperAction]:
    values = np.asarray(chunk, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("action chunk must be a non-empty 2D array")
    return [adapt_action_row(row, current_joints_rad, config) for row in values]


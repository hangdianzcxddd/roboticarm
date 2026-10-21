"""Simple LingBo effector-to-Piper gripper conversion."""

from __future__ import annotations

import numpy as np


def to_piper_gripper_mm(
    value: float,
    *,
    unit: str = "mm",
    open_mm: float = 0.0,
    close_mm: float = -100.0,
) -> float:
    """Convert one model effector value to Piper's calibrated command in mm.

    ``unit=mm`` is the default for the current action contract.  For a model
    effector in [0, 1], use ``unit=normalized``; 0 means open and 1 means
    closed under the current Piper gripper convention.
    """
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("gripper action must be finite")
    if unit == "normalized":
        value = float(np.clip(value, 0.0, 1.0))
        value = open_mm + value * (close_mm - open_mm)
    elif unit != "mm":
        raise ValueError(f"unsupported gripper unit: {unit}")
    return float(np.clip(value, min(open_mm, close_mm), max(open_mm, close_mm)))


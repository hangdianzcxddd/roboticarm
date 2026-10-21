"""Execute a complete cloud action chunk through Piper SDK.

There is deliberately no local grasp state machine in this first version. The
cloud model owns the action chunk; this class only applies one already-filtered
action at a time and refreshes feedback before each action.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController

from .lingbo_action_adapter import ActionAdapterConfig, adapt_action_row
from .piper_observation import read_piper_state
from .piper_safety_filter import SafetyFilterConfig, filter_action


class StaticGraspExecutor:
    def __init__(
        self,
        controller: PiperController,
        safety_config: SafetyFilterConfig,
        speed_percent: int = 5,
        period_s: float = 0.5,
        execute: bool = False,
    ) -> None:
        self.controller = controller
        self.safety_config = safety_config
        self.speed_percent = int(speed_percent)
        self.period_s = float(period_s)
        self.execute = bool(execute)

    def execute_chunk(
        self,
        chunk: np.ndarray,
        adapter_config: ActionAdapterConfig,
    ) -> list[dict[str, Any]]:
        values = np.asarray(chunk, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("action chunk must be a non-empty 2D array")
        results: list[dict[str, Any]] = []
        for index, row in enumerate(values):
            current, current_gripper, _ = read_piper_state(self.controller)
            action = adapt_action_row(row, current, adapter_config)
            safe = filter_action(
                action,
                current,
                self.safety_config,
                current_gripper_mm=current_gripper,
            )
            result = {
                "index": index,
                "target_joints_rad": action.joints_rad.astype(float).tolist(),
                "safe_joints_rad": safe.joints_rad.astype(float).tolist(),
                "target_gripper_mm": action.gripper_mm,
                "safe_gripper_mm": safe.gripper_mm,
                "executed": self.execute,
            }
            if self.execute:
                self.controller.move_joints(
                    tuple(float(value) for value in safe.joints_rad),
                    speed_percent=self.speed_percent,
                )
                if safe.gripper_mm is not None:
                    self.controller.set_gripper(safe.gripper_mm, effort_nm=0.8, enable=True)
            results.append(result)
            if self.execute and self.period_s > 0 and index + 1 < len(values):
                time.sleep(self.period_s)
        return results

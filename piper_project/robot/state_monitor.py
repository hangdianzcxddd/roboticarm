"""State read helpers for Piper feedback messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot.piper_controller import PiperController


@dataclass
class StateMonitor:
    controller: PiperController

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.controller.get_status(),
            "end_pose": self.controller.get_end_pose(),
            "joint_state": self.controller.get_joint_state(),
            "gripper_state": self.controller.get_gripper_state(),
        }

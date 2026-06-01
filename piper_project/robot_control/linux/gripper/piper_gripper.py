"""Piper gripper convenience wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from robot_control.linux.arm.piper_arm import PiperController


@dataclass
class Gripper:
    controller: PiperController
    open_width_mm: float = 70.0
    close_width_mm: float = 0.0
    effort_nm: float = 1.0

    def open(self) -> None:
        self.controller.set_gripper(self.open_width_mm, self.effort_nm, enable=True)

    def close(self) -> None:
        self.controller.set_gripper(self.close_width_mm, self.effort_nm, enable=True)

    def disable(self) -> None:
        self.controller.set_gripper(0.0, 0.0, enable=False)

    def clear_error(self) -> None:
        self.controller.set_gripper(0.0, self.effort_nm, enable=True, clear_error=True)

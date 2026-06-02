from __future__ import annotations

import unittest

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.catch.set_gripper_zero import set_gripper_zero
from robot_control.linux.config.robot_config import RobotConfig


def make_controller() -> tuple[PiperController, FakePiperSdk]:
    sdk = FakePiperSdk()
    config = RobotConfig(command_interval_s=0, connect_timeout_s=0.1)
    return PiperController(config=config, sdk=sdk), sdk


class TestSetGripperZero(unittest.TestCase):
    def test_set_gripper_zero_uses_disable_then_set_zero(self):
        controller, sdk = make_controller()

        set_gripper_zero(controller, zero_command_mm=0.0, effort_nm=0.4, settle_s=0.0)

        self.assertEqual(
            sdk.calls,
            [
                ("GripperCtrl", (0, 400, 0x00, 0)),
                ("GripperCtrl", (0, 400, 0x00, 0xAE)),
            ],
        )

    def test_set_gripper_zero_can_clear_error_and_enable_after(self):
        controller, sdk = make_controller()

        set_gripper_zero(
            controller,
            zero_command_mm=2.5,
            effort_nm=0.7,
            settle_s=0.0,
            clear_error=True,
            enable_after=True,
        )

        self.assertEqual(
            sdk.calls,
            [
                ("GripperCtrl", (2500, 700, 0x03, 0)),
                ("GripperCtrl", (2500, 700, 0x00, 0)),
                ("GripperCtrl", (2500, 700, 0x00, 0xAE)),
                ("GripperCtrl", (2500, 700, 0x01, 0)),
            ],
        )

    def test_invalid_values_are_rejected(self):
        controller, sdk = make_controller()

        with self.assertRaises(ValueError):
            set_gripper_zero(controller, zero_command_mm=-1.0)
        with self.assertRaises(ValueError):
            set_gripper_zero(controller, effort_nm=6.0)
        with self.assertRaises(ValueError):
            set_gripper_zero(controller, settle_s=-0.1)

        self.assertEqual(sdk.calls, [])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.gripper.control_gripper import (
    parse_interactive_command,
    send_gripper_target,
    validate_command,
)


def make_controller() -> tuple[PiperController, FakePiperSdk]:
    sdk = FakePiperSdk()
    config = RobotConfig(command_interval_s=0, connect_timeout_s=0.1)
    return PiperController(config=config, sdk=sdk), sdk


class TestControlGripper(unittest.TestCase):
    def test_send_gripper_target_allows_open_zero_negative_close_command(self):
        controller, sdk = make_controller()

        count = send_gripper_target(controller, -35.0, 1.2)

        self.assertEqual(count, 1)
        self.assertEqual(sdk.calls, [("GripperCtrl", (-35000, 1200, 0x01, 0))])

    def test_clear_error_does_not_set_zero(self):
        controller, sdk = make_controller()

        send_gripper_target(controller, -20.0, 0.4, clear_error=True)

        self.assertEqual(sdk.calls, [("GripperCtrl", (-20000, 400, 0x03, 0))])

    def test_command_outside_software_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\[-100, 100\]"):
            validate_command(-100.1, 1.0)

    def test_parse_interactive_command_uses_optional_effort(self):
        self.assertEqual(parse_interactive_command("-45", 1.1), (-45.0, 1.1))
        self.assertEqual(parse_interactive_command("-45 1.8", 1.1), (-45.0, 1.8))


if __name__ == "__main__":
    unittest.main()

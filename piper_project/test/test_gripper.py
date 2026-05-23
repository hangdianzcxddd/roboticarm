from __future__ import annotations

import unittest

from config.robot_config import RobotConfig
from robot.gripper import Gripper
from robot.piper_controller import PiperController
from test.test_move import FakePiperSdk


def make_controller() -> tuple[PiperController, FakePiperSdk]:
    sdk = FakePiperSdk()
    config = RobotConfig(command_interval_s=0, connect_timeout_s=0.1)
    return PiperController(config=config, sdk=sdk), sdk


class TestGripper(unittest.TestCase):
    def test_open_and_close_gripper_send_width_in_micrometers(self):
        controller, sdk = make_controller()
        gripper = Gripper(
            controller, open_width_mm=70.0, close_width_mm=3.5, effort_nm=1.2
        )

        gripper.open()
        gripper.close()

        self.assertEqual(
            sdk.calls,
            [
                ("GripperCtrl", (70000, 1200, 0x01, 0)),
                ("GripperCtrl", (3500, 1200, 0x01, 0)),
            ],
        )

    def test_clear_error_uses_enable_clear_error_code(self):
        controller, sdk = make_controller()

        controller.set_gripper(0.0, effort_nm=1.0, enable=True, clear_error=True)

        self.assertEqual(sdk.calls, [("GripperCtrl", (0, 1000, 0x03, 0))])

    def test_invalid_gripper_effort_is_rejected(self):
        controller, sdk = make_controller()

        with self.assertRaises(ValueError):
            controller.set_gripper(10.0, effort_nm=6.0)

        self.assertEqual(sdk.calls, [])


if __name__ == "__main__":
    unittest.main()

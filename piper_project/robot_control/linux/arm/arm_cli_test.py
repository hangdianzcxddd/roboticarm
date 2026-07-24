from __future__ import annotations

import unittest
from unittest.mock import patch

from robot_control.linux.arm.arm_cli import build_parser, main


class TestArmCli(unittest.TestCase):
    def test_resume_command_is_available(self):
        args = build_parser().parse_args(["resume"])

        self.assertEqual(args.command, "resume")

    @patch("robot_control.linux.arm.arm_cli.PiperController")
    @patch("sys.argv", ["arm_cli.py", "resume"])
    def test_resume_releases_stop_without_enabling_or_moving(self, controller_type):
        controller = controller_type.return_value

        with patch("builtins.print"):
            main()

        controller.connect.assert_called_once_with(enable=False)
        controller.resume_from_emergency_stop.assert_called_once_with()
        controller.disconnect.assert_called_once_with()
        controller.enable.assert_not_called()
        controller.move_joints.assert_not_called()

    def test_joint_motion_has_no_added_safety_limits(self):
        args = build_parser().parse_args(
            ["joints", "0", "1", "-1", "0", "0.5", "0"]
        )

        for name in (
            "max_joint_delta_deg",
            "motion_timeout_s",
            "joint_tolerance_deg",
            "max_joint_waypoint_step_deg",
            "max_runtime_path_error_deg",
            "max_jacobian_condition",
        ):
            self.assertFalse(hasattr(args, name))


if __name__ == "__main__":
    unittest.main()

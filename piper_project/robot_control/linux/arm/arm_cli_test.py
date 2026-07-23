from __future__ import annotations

import math
import unittest

from robot_control.linux.arm.arm_cli import (
    build_parser,
    joint_path,
    validate_joint_change,
)
from robot_control.linux.arm.safety import SafetyError


class TestArmCli(unittest.TestCase):
    def test_joint_motion_defaults_to_sixty_degrees_per_joint(self):
        args = build_parser().parse_args(
            ["joints", "0", "1", "-1", "0", "0.5", "0"]
        )

        self.assertEqual(tuple(args.max_joint_delta_deg), (60,) * 6)

    def test_joint_change_over_limit_is_rejected(self):
        current = [0.0] * 6
        target = [0.0, math.radians(61.0), 0.0, 0.0, 0.0, 0.0]

        with self.assertRaisesRegex(SafetyError, "J2=61.00deg>60.00deg"):
            validate_joint_change(current, target, (60.0,) * 6)

    def test_joint_change_limit_is_adjustable(self):
        current = [0.0] * 6
        target = [0.0, math.radians(61.0), 0.0, 0.0, 0.0, 0.0]

        validate_joint_change(
            current,
            target,
            (60.0, 62.0, 60.0, 60.0, 60.0, 60.0),
        )

    def test_joint_path_respects_waypoint_limit(self):
        path = joint_path(
            [0.0] * 6,
            [math.radians(19.0), math.radians(-11.0), 0.0, 0.0, 0.0, 0.0],
            5.0,
        )

        maximum_step = max(
            max(abs(math.degrees(end - start)) for start, end in zip(left, right))
            for left, right in zip(path, path[1:])
        )
        self.assertLessEqual(maximum_step, 5.0)


if __name__ == "__main__":
    unittest.main()

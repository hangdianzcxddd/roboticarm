from __future__ import annotations

import argparse
import unittest

from robot_control.linux.catch.scripted_catch_demo import build_parser, target_pose_from_args
from robot_control.linux.catch.tcp_offset import build_catch_plan
from robot_control.shared.geometry import EndPose


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "target_x_mm": None,
        "target_y_mm": None,
        "target_z_mm": None,
        "rx_deg": 180.0,
        "ry_deg": 0.0,
        "rz_deg": None,
        "approach_height_mm": 40.0,
        "descend_mm": 30.0,
        "min_target_z_mm": 0.0,
        "target_z_offset_mm": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestScriptedCatchDemo(unittest.TestCase):
    def test_parser_accepts_no_enable_for_already_enabled_arm(self):
        args = build_parser().parse_args(["--execute", "--no-enable"])

        self.assertTrue(args.execute)
        self.assertTrue(args.no_enable)

    def test_default_target_keeps_current_j6_xy_for_reachable_flow_demo(self):
        current_j6_pose = EndPose(48.835, 0.0, 168.840, 180.0, 66.356, 180.0)
        offset = (0.0, 0.0, 145.0)

        target_tcp_pose = target_pose_from_args(make_args(), current_j6_pose, offset)
        plan = build_catch_plan(
            target_tcp_pose,
            offset_j6_tcp_mm=offset,
            approach_height_mm=40.0,
            lift_height_mm=50.0,
            target_z_offset_mm=0.0,
        )

        self.assertAlmostEqual(plan.approach_j6_pose.x_mm, current_j6_pose.x_mm)
        self.assertAlmostEqual(plan.approach_j6_pose.y_mm, current_j6_pose.y_mm)
        self.assertAlmostEqual(plan.approach_j6_pose.z_mm, 185.0)
        self.assertAlmostEqual(plan.grasp_j6_pose.x_mm, current_j6_pose.x_mm)
        self.assertAlmostEqual(plan.grasp_j6_pose.y_mm, current_j6_pose.y_mm)
        self.assertAlmostEqual(plan.grasp_j6_pose.z_mm, 145.0)
        self.assertEqual(plan.grasp_j6_pose.rx_deg, 180.0)
        self.assertEqual(plan.grasp_j6_pose.ry_deg, 0.0)
        self.assertEqual(plan.grasp_j6_pose.rz_deg, 180.0)


if __name__ == "__main__":
    unittest.main()

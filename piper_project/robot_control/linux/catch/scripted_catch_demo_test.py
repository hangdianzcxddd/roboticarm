from __future__ import annotations

import argparse
import contextlib
import io
import unittest

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.catch.scripted_catch_demo import (
    build_parser,
    build_scripted_plan,
    run_with_controller,
    target_pose_from_args,
    validate_j6_waypoint,
)
from robot_control.linux.catch.tcp_offset import build_catch_plan
from robot_control.shared.geometry import EndPose


class FakeEndPose:
    def __init__(self, pose: EndPose) -> None:
        self.X_axis = round(pose.x_mm * 1000)
        self.Y_axis = round(pose.y_mm * 1000)
        self.Z_axis = round(pose.z_mm * 1000)
        self.RX_axis = round(pose.rx_deg * 1000)
        self.RY_axis = round(pose.ry_deg * 1000)
        self.RZ_axis = round(pose.rz_deg * 1000)


class FakeEndPoseFeedback:
    def __init__(self, pose: EndPose) -> None:
        self.end_pose = FakeEndPose(pose)
        self.time_stamp = 1.0
        self.Hz = 50.0


class FakeCatchPiperSdk(FakePiperSdk):
    def __init__(self, pose: EndPose) -> None:
        super().__init__()
        self.end_pose_feedback = FakeEndPoseFeedback(pose)

    def GetArmEndPoseMsgs(self):
        return self.end_pose_feedback


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "target_x_mm": None,
        "target_y_mm": None,
        "target_z_mm": None,
        "rx_deg": None,
        "ry_deg": None,
        "rz_deg": None,
        "target_z_offset_mm": 0.0,
        "approach_height_mm": 40.0,
        "descend_mm": 30.0,
        "min_target_z_mm": 0.0,
        "lift_height_mm": 50.0,
        "place_x_mm": None,
        "place_y_mm": None,
        "place_dx_mm": 0.0,
        "place_dy_mm": 30.0,
        "place_z_mm": 80.0,
        "place_rx_deg": None,
        "place_ry_deg": None,
        "place_rz_deg": None,
        "place_approach_height_mm": 40.0,
        "stage_height_mm": 60.0,
        "return_home": True,
        "return_height_mm": 60.0,
        "approach_mode": "P",
        "grasp_mode": "L",
        "lift_mode": "L",
        "return_mode": "P",
        "execute": False,
        "speed": 5,
        "command_duration_s": 0.0,
        "command_period_s": 0.02,
        "motion_timeout_s": 0.0,
        "position_tolerance_mm": 8.0,
        "orientation_tolerance_deg": 12.0,
        "continue_on_waypoint_error": False,
        "no_wait_waypoints": True,
        "print_feedback": False,
        "min_horizontal_radius_mm": 0.0,
        "max_horizontal_radius_mm": None,
        "min_j6_z_mm": None,
        "max_j6_z_mm": None,
        "warn_margin_mm": 0.0,
        "open_command_mm": 0.0,
        "grasp_command_mm": -100.0,
        "release_command_mm": 0.0,
        "effort_nm": 1.0,
        "grasp_effort_nm": 1.5,
        "release_effort_nm": 0.4,
        "gripper_settle_s": 0.0,
        "set_open_zero_on_start": False,
        "zero_settle_s": 0.0,
        "clear_gripper_error": False,
        "no_pre_open": False,
        "no_gripper": False,
        "tcp_offset_x_mm": 0.0,
        "tcp_offset_y_mm": 0.0,
        "tcp_offset_z_mm": 145.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestScriptedCatchDemo(unittest.TestCase):
    def test_parser_accepts_no_enable_for_already_enabled_arm(self):
        args = build_parser().parse_args(["--execute", "--no-enable"])

        self.assertTrue(args.execute)
        self.assertTrue(args.no_enable)

    def test_default_target_uses_safe_tcp_point_and_current_orientation(self):
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

        self.assertEqual(target_tcp_pose.x_mm, 300.0)
        self.assertEqual(target_tcp_pose.y_mm, 0.0)
        self.assertEqual(target_tcp_pose.z_mm, 80.0)
        self.assertEqual(target_tcp_pose.rx_deg, current_j6_pose.rx_deg)
        self.assertEqual(target_tcp_pose.ry_deg, current_j6_pose.ry_deg)
        self.assertEqual(target_tcp_pose.rz_deg, current_j6_pose.rz_deg)
        self.assertEqual(plan.grasp_j6_pose.rx_deg, 180.0)
        self.assertEqual(plan.grasp_j6_pose.ry_deg, 66.356)
        self.assertEqual(plan.grasp_j6_pose.rz_deg, 180.0)

    def test_scripted_plan_contains_full_return_chain(self):
        current_j6_pose = EndPose(100.0, 0.0, 200.0, 0.0, 85.0, 0.0)
        plan = build_scripted_plan(
            make_args(rx_deg=180.0, ry_deg=0.0, rz_deg=0.0),
            current_j6_pose,
            (0.0, 0.0, 145.0),
        )

        self.assertEqual(
            [waypoint.label for waypoint in plan.waypoints],
            [
                "stage_lift",
                "orient_top_down",
                "approach",
                "grasp",
                "lift",
                "place_approach",
                "place",
                "place_retreat",
                "return_align",
                "restore_orientation",
                "return_home",
            ],
        )
        self.assertEqual(plan.waypoints[1].pose.rx_deg, 180.0)
        self.assertEqual(plan.waypoints[-2].pose.rx_deg, current_j6_pose.rx_deg)
        self.assertEqual(plan.waypoints[-1].pose, current_j6_pose)

    def test_workspace_envelope_rejects_out_of_radius_waypoint(self):
        controller = PiperController(
            config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
            sdk=FakePiperSdk(),
        )

        with self.assertRaisesRegex(ValueError, "exceeds max radius"):
            validate_j6_waypoint(
                controller,
                make_args(max_horizontal_radius_mm=100.0),
                "too_far",
                EndPose(120.0, 0.0, 100.0, 0.0, 0.0, 0.0),
            )

    def test_execute_runs_gripper_actions_and_all_waypoints(self):
        current_j6_pose = EndPose(180.0, 0.0, 220.0, 180.0, 0.0, 0.0)
        sdk = FakeCatchPiperSdk(current_j6_pose)
        controller = PiperController(
            config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
            sdk=sdk,
        )
        args = make_args(
            execute=True,
            target_x_mm=180.0,
            target_y_mm=0.0,
            target_z_mm=40.0,
            rx_deg=180.0,
            ry_deg=0.0,
            rz_deg=0.0,
            place_x_mm=180.0,
            place_y_mm=40.0,
            place_z_mm=70.0,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            counts = run_with_controller(controller, args, (0.0, 0.0, 145.0))

        self.assertEqual(
            sorted(counts),
            [
                "approach",
                "grasp",
                "lift",
                "orient_top_down",
                "place",
                "place_approach",
                "place_retreat",
                "restore_orientation",
                "return_align",
                "return_home",
                "stage_lift",
            ],
        )
        gripper_calls = [call for call in sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 400, 0x01, 0)),
                ("GripperCtrl", (-100000, 1500, 0x01, 0)),
                ("GripperCtrl", (0, 400, 0x01, 0)),
            ],
        )
        end_pose_calls = [call for call in sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(len(end_pose_calls), 11)


if __name__ == "__main__":
    unittest.main()

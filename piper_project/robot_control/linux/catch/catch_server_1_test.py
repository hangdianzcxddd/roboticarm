from __future__ import annotations

import argparse
import contextlib
import io
import math
import unittest
from unittest import mock

import numpy as np

from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG
from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.arm.piper_arm_test import FakeJointFeedback, FakePiperSdk
from robot_control.linux.catch.catch_server_1 import (
    build_parser,
    build_steps,
    dynamic_preposition_pose,
    dynamic_timing_from_command,
    apply_grasp_yaw_adjustment,
    handle_message,
    move_to_start_ready,
    PlanarIkSolution,
    target_from_command,
    validate_args,
    validate_planar_ik_joint_delta_limits,
    wait_for_dynamic_timing,
    yolo_metadata_from_message,
)
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    CameraPointCommand,
    GraspCandidate,
    make_camera_point_message,
)


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "speed": 40,
        "startup_speed": 10,
        "transfer_mode": "P",
        "vertical_mode": "L",
        "lift_mode": "L",
        "command_duration_s": 0.0,
        "command_period_s": 0.02,
        "motion_timeout_s": 0.0,
        "position_tolerance_mm": 8.0,
        "orientation_tolerance_deg": 8.0,
        "no_wait_waypoints": True,
        "print_feedback": False,
        "debug_plan": False,
        "t_robot_s": 1.7,
        "dynamic_t_robot_by_x": False,
        "dynamic_t_robot_reference_x_mm": None,
        "dynamic_t_robot_s_per_mm": 0.0,
        "dynamic_t_robot_planar_far_s_per_mm": 0.0,
        "dynamic_t_robot_planar_near_s_per_mm": 0.0,
        "dynamic_t_robot_margin_s": 0.0,
        "dynamic_t_robot_min_s": None,
        "dynamic_t_robot_max_s": None,
        "dynamic_too_close_skip": False,
        "dynamic_preposition": False,
        "dynamic_wait_placement": "wait",
        "dynamic_preposition_clearance_mm": 20.0,
        "final_action_time_s": 0.55,
        "collision_protection_level": None,
        "no_start_ready": False,
        "vision_source": "opencv",
        "graspnet_candidate_policy": "best-reachable-score",
        "graspnet_max_candidates": 20,
        "graspnet_approach_distance_mm": 60.0,
        "graspnet_lift_height_mm": 60.0,
        "graspnet_allow_fixed_fallback": False,
        "graspnet_tcp_rot_x_deg": 0.0,
        "graspnet_tcp_rot_y_deg": 0.0,
        "graspnet_tcp_rot_z_deg": 0.0,
        "graspnet_max_rz_delta_deg": None,
        "ready_x_mm": 350.0,
        "ready_y_mm": 0.0,
        "transfer_z_mm": 253.252,
        "grasp_z_mm": 133.252,
        "use_detected_z": "off",
        "detected_z_offset_mm": 0.0,
        "rx_deg": -176.794223,
        "ry_deg": 0.889438,
        "rz_deg": -105.490597,
        "grasp_yaw_by_object_short_edge": False,
        "grasp_yaw_mode": "integrated-planar-ik",
        "gripper_close_axis_offset_deg": 0.0,
        "grasp_yaw_max_delta_deg": 45.0,
        "grasp_yaw_deadband_deg": 5.0,
        "grasp_yaw_speed": 20,
        "force_pick_x_mm": None,
        "pick_x_offset_mm": 0.0,
        "force_pick_y_mm": None,
        "dynamic_pitch_by_x": False,
        "dynamic_pitch_mode": "joint",
        "pitch_x_ref_mm": 350.0,
        "pitch_deg_per_mm": None,
        "keep_pitch_min_x_mm": 300.0,
        "keep_pitch_max_x_mm": 380.0,
        "pitch_deg_per_mm_far": None,
        "pitch_deg_per_mm_near": None,
        "min_dynamic_pitch_deg": -12.0,
        "max_dynamic_pitch_deg": 12.0,
        "dynamic_pitch_j2_ratio": 0.0,
        "dynamic_pitch_j3_ratio": 0.0,
        "dynamic_pitch_j5_ratio": 1.0,
        "dynamic_pitch_speed": 20,
        "planar_ik_max_iterations": 40,
        "planar_ik_position_tolerance_mm": 3.0,
        "planar_ik_rx_tolerance_deg": 5.0,
        "planar_ik_ry_tolerance_deg": 1.0,
        "planar_ik_rz_tolerance_deg": 3.0,
        "planar_ik_max_step_deg": 8.0,
        "planar_ik_damping": 1e-3,
        "planar_ik_include_j6_yaw": True,
        "planar_ik_speed": None,
        "planar_ik_grasp_z_comp_mm_per_deg": 0.0,
        "reuse_planar_ik_above_target": True,
        "planar_ik_joint_tolerance_deg": 2.0,
        "planar_ik_fk_tool_z_offset_mm": 82.0,
        "pick_delay_s": 0.0,
        "place_x_mm": 60.0,
        "place_y_mm": 270.0,
        "place_pick_x_mm": 60.0,
        "place_pick_y_mm": -270.0,
        "place_pick_z_mm": None,
        "place_pick_delay_s": 3.0,
        "grasp_only": False,
        "no_return_ready": False,
        "min_target_x_mm": 80.0,
        "max_target_x_mm": None,
        "min_target_y_mm": -300.0,
        "max_target_y_mm": 300.0,
        "min_grasp_z_mm": None,
        "max_grasp_z_mm": None,
        "open_command_mm": 0.0,
        "grasp_command_mm": -100.0,
        "effort_nm": 1.0,
        "grasp_effort_nm": None,
        "max_gripper_effort_nm": 0.8,
        "safe_gripper_stop_on_effort": False,
        "safe_gripper_monitor_s": 0.0,
        "safe_gripper_poll_s": 0.02,
        "safe_gripper_release_mm": None,
        "gripper_settle_s": 0.0,
        "clear_gripper_error": False,
        "no_pre_open": False,
        "no_gripper": False,
        "execute": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def make_controller() -> PiperController:
    return PiperController(
        config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
        sdk=FakePiperSdk(),
    )


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


class FeedbackPiperSdk(FakePiperSdk):
    def __init__(
        self,
        pose: EndPose,
        *,
        freeze_pose: bool = False,
        freeze_below_z_mm: float | None = None,
        stall_z_mm: float | None = None,
    ) -> None:
        super().__init__()
        self.pose = pose
        self.freeze_pose = freeze_pose
        self.freeze_below_z_mm = freeze_below_z_mm
        self.stall_z_mm = stall_z_mm
        self.gripper_feedbacks: list[FakeGripperFeedback] = []

    def EndPoseCtrl(self, X, Y, Z, RX, RY, RZ):
        super().EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        if self.freeze_pose:
            return
        next_z_mm = Z / 1000.0
        if self.freeze_below_z_mm is not None and next_z_mm < self.freeze_below_z_mm:
            return
        if self.stall_z_mm is not None and next_z_mm < self.stall_z_mm:
            next_z_mm = self.stall_z_mm
        self.pose = EndPose(
            X / 1000.0,
            Y / 1000.0,
            next_z_mm,
            RX / 1000.0,
            RY / 1000.0,
            RZ / 1000.0,
        )

    def JointCtrl(self, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6):
        super().JointCtrl(joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
        self.joint_feedback = FakeJointFeedback(
            (joint_1, joint_2, joint_3, joint_4, joint_5, joint_6)
        )
        self.pose = EndPose(
            self.pose.x_mm,
            self.pose.y_mm,
            self.pose.z_mm,
            self.pose.rx_deg,
            self.pose.ry_deg,
            joint_6 / 1000.0,
        )

    def GetArmEndPoseMsgs(self):
        return FakeEndPoseFeedback(self.pose)

    def GetArmGripperMsgs(self):
        if self.gripper_feedbacks:
            return self.gripper_feedbacks.pop(0)
        return FakeGripperFeedback(position_mm=0.0, effort_nm=0.0)


class SequencedFeedbackPiperSdk(FakePiperSdk):
    def __init__(self, poses: list[EndPose]) -> None:
        super().__init__()
        if not poses:
            raise ValueError("poses must not be empty")
        self.poses = poses
        self.last_pose = poses[-1]

    def GetArmEndPoseMsgs(self):
        if self.poses:
            self.last_pose = self.poses.pop(0)
        return FakeEndPoseFeedback(self.last_pose)


class FakeGripperState:
    def __init__(self, *, position_mm: float, effort_nm: float) -> None:
        self.grippers_angle = round(position_mm * 1000)
        self.grippers_effort = round(effort_nm * 1000)


class FakeGripperFeedback:
    def __init__(self, *, position_mm: float, effort_nm: float) -> None:
        self.gripper_state = FakeGripperState(
            position_mm=position_mm,
            effort_nm=effort_nm,
        )
        self.time_stamp = 1.0
        self.Hz = 50.0


def make_feedback_controller(
    pose: EndPose,
    *,
    freeze_pose: bool = False,
    freeze_below_z_mm: float | None = None,
    stall_z_mm: float | None = None,
    gripper_efforts_nm: list[float] | None = None,
) -> PiperController:
    sdk = FeedbackPiperSdk(
        pose,
        freeze_pose=freeze_pose,
        freeze_below_z_mm=freeze_below_z_mm,
        stall_z_mm=stall_z_mm,
    )
    sdk.gripper_feedbacks = [
        FakeGripperFeedback(position_mm=0.0, effort_nm=effort)
        for effort in (gripper_efforts_nm or [])
    ]
    return PiperController(
        config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
        sdk=sdk,
    )


def make_sequenced_feedback_controller(poses: list[EndPose]) -> PiperController:
    return PiperController(
        config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
        sdk=SequencedFeedbackPiperSdk(poses),
    )


def joint_feedback_from_degrees(*joints_deg: float) -> FakeJointFeedback:
    if len(joints_deg) != 6:
        raise ValueError("expected six joint values")
    return FakeJointFeedback(
        tuple(round(math.radians(value) * RAD_TO_MILLI_DEG) for value in joints_deg)
    )


class TestCatchServer1(unittest.TestCase):
    def test_parser_rejects_removed_descent_protection_options(self):
        args = build_parser().parse_args([])

        self.assertFalse(hasattr(args, "guarded_descent"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--no-descent-protection"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--guarded-descent"])
        self.assertEqual(args.startup_speed, 10)
        self.assertAlmostEqual(args.place_x_mm, 60.0)
        self.assertAlmostEqual(args.place_y_mm, 270.0)
        self.assertAlmostEqual(args.place_pick_x_mm, 60.0)
        self.assertAlmostEqual(args.place_pick_y_mm, -270.0)
        self.assertEqual(args.lift_mode, "L")
        self.assertFalse(args.dynamic_pitch_by_x)
        self.assertEqual(args.dynamic_pitch_mode, "joint")
        self.assertEqual(args.dynamic_wait_placement, "wait")
        self.assertEqual(args.vision_source, "opencv")
        self.assertEqual(args.graspnet_candidate_policy, "best-reachable-score")
        self.assertEqual(args.graspnet_max_candidates, 20)
        self.assertAlmostEqual(args.graspnet_approach_distance_mm, 60.0)
        self.assertAlmostEqual(args.graspnet_lift_height_mm, 60.0)
        self.assertFalse(args.graspnet_allow_fixed_fallback)
        self.assertIsNone(args.graspnet_max_rz_delta_deg)
        self.assertAlmostEqual(args.keep_pitch_min_x_mm, 300.0)
        self.assertAlmostEqual(args.keep_pitch_max_x_mm, 380.0)
        self.assertIsNone(args.pitch_deg_per_mm)
        self.assertIsNone(args.pitch_deg_per_mm_far)
        self.assertIsNone(args.pitch_deg_per_mm_near)
        self.assertAlmostEqual(args.min_dynamic_pitch_deg, -12.0)
        self.assertAlmostEqual(args.max_dynamic_pitch_deg, 12.0)
        self.assertEqual(args.planar_ik_max_iterations, 40)
        self.assertAlmostEqual(args.planar_ik_position_tolerance_mm, 3.0)
        self.assertAlmostEqual(args.planar_ik_rx_tolerance_deg, 5.0)
        self.assertAlmostEqual(args.planar_ik_ry_tolerance_deg, 1.0)
        self.assertAlmostEqual(args.planar_ik_rz_tolerance_deg, 3.0)
        self.assertTrue(args.planar_ik_include_j6_yaw)
        self.assertIsNone(args.planar_ik_speed)
        self.assertTrue(args.reuse_planar_ik_above_target)
        self.assertAlmostEqual(args.planar_ik_joint_tolerance_deg, 2.0)
        self.assertAlmostEqual(args.planar_ik_fk_tool_z_offset_mm, 82.0)
        self.assertEqual(args.use_detected_z, "off")
        self.assertFalse(args.grasp_yaw_by_object_short_edge)
        self.assertEqual(args.grasp_yaw_mode, "integrated-planar-ik")
        self.assertAlmostEqual(args.gripper_close_axis_offset_deg, 0.0)
        self.assertAlmostEqual(args.grasp_yaw_max_delta_deg, 45.0)
        self.assertAlmostEqual(args.grasp_yaw_deadband_deg, 5.0)
        self.assertEqual(args.grasp_yaw_speed, 20)
        self.assertAlmostEqual(args.pick_x_offset_mm, 0.0)
        self.assertFalse(args.dynamic_t_robot_by_x)
        self.assertIsNone(args.dynamic_t_robot_reference_x_mm)
        self.assertAlmostEqual(args.dynamic_t_robot_s_per_mm, 0.0)
        self.assertAlmostEqual(args.dynamic_t_robot_planar_far_s_per_mm, 0.0)
        self.assertAlmostEqual(args.dynamic_t_robot_planar_near_s_per_mm, 0.0)
        self.assertAlmostEqual(args.dynamic_t_robot_margin_s, 0.0)
        self.assertIsNone(args.dynamic_t_robot_min_s)
        self.assertIsNone(args.dynamic_t_robot_max_s)
        self.assertFalse(args.dynamic_too_close_skip)

        args = build_parser().parse_args(["--use-detected-z"])
        self.assertEqual(args.use_detected_z, "center")

        args = build_parser().parse_args(["--use-detected-z", "top"])
        self.assertEqual(args.use_detected_z, "top")

        args = build_parser().parse_args(["--grasp-yaw-mode", "separate-j6"])
        self.assertEqual(args.grasp_yaw_mode, "separate-j6")

        args = build_parser().parse_args(
            [
                "--startup-speed",
                "9",
                "--planar-ik-speed",
                "25",
                "--dynamic-wait-placement",
                "nowait",
                "--dynamic-t-robot-by-x",
                "--dynamic-t-robot-reference-x-mm",
                "300",
                "--dynamic-t-robot-s-per-mm",
                "0.002",
                "--dynamic-t-robot-planar-far-s-per-mm",
                "0.006",
                "--dynamic-t-robot-planar-near-s-per-mm",
                "0.004",
                "--dynamic-t-robot-margin-s",
                "0.1",
                "--dynamic-t-robot-min-s",
                "1.4",
                "--dynamic-t-robot-max-s",
                "3.2",
                "--dynamic-too-close-skip",
            ]
        )

        self.assertEqual(args.startup_speed, 9)
        self.assertEqual(args.planar_ik_speed, 25)
        self.assertEqual(args.dynamic_wait_placement, "nowait")
        self.assertTrue(args.dynamic_t_robot_by_x)
        self.assertAlmostEqual(args.dynamic_t_robot_reference_x_mm, 300.0)
        self.assertAlmostEqual(args.dynamic_t_robot_s_per_mm, 0.002)
        self.assertAlmostEqual(args.dynamic_t_robot_planar_far_s_per_mm, 0.006)
        self.assertAlmostEqual(args.dynamic_t_robot_planar_near_s_per_mm, 0.004)
        self.assertAlmostEqual(args.dynamic_t_robot_margin_s, 0.1)
        self.assertAlmostEqual(args.dynamic_t_robot_min_s, 1.4)
        self.assertAlmostEqual(args.dynamic_t_robot_max_s, 3.2)
        self.assertTrue(args.dynamic_too_close_skip)

    def test_parser_accepts_adaptive_joint_pitch_alias_and_slope_defaults(self):
        args = build_parser().parse_args(["--adaptive-joint-pitch-by-x"])

        validate_args(args)

        self.assertTrue(args.dynamic_pitch_by_x)
        self.assertAlmostEqual(args.pitch_deg_per_mm_far, 0.04)
        self.assertAlmostEqual(args.pitch_deg_per_mm_near, 0.03)
        self.assertAlmostEqual(args.pitch_deg_per_mm, 0.04)

        args = build_parser().parse_args(
            ["--dynamic-pitch-by-x", "--pitch-deg-per-mm", "0.07"]
        )

        validate_args(args)

        self.assertTrue(args.dynamic_pitch_by_x)
        self.assertAlmostEqual(args.pitch_deg_per_mm_far, 0.07)
        self.assertAlmostEqual(args.pitch_deg_per_mm_near, 0.07)

    def test_parser_accepts_grasp_z_alias(self):
        args = build_parser().parse_args(["--grasp-z", "133"])

        self.assertAlmostEqual(args.grasp_z_mm, 133.0)

    def test_build_steps_out_places_detected_target_without_pick_back(self):
        args = make_args(pick_delay_s=1.25, place_pick_delay_s=2.5)
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.30, 0.02, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertEqual(
            [step.label for step in steps],
            [
                "ready_open",
                "pick_above",
                "pick_grasp",
                "pick_lift",
                "place_above",
                "place_release",
                "return_ready",
            ],
        )
        self.assertEqual(
            [step.gripper_after for step in steps],
            ["open", None, "close", None, None, "open", None],
        )
        self.assertEqual(
            [step.mode for step in steps],
            ["P", "L", "L", args.lift_mode, "P", "P", "P"],
        )
        self.assertTrue(steps[1].require_reached)
        self.assertFalse(steps[2].require_reached)
        self.assertAlmostEqual(steps[1].pose.x_mm, 300.0)
        self.assertAlmostEqual(steps[1].pose.y_mm, 20.0)
        self.assertAlmostEqual(steps[1].pose.z_mm, args.transfer_z_mm)
        self.assertAlmostEqual(steps[2].pose.z_mm, args.grasp_z_mm)
        self.assertAlmostEqual(steps[4].pose.x_mm, args.place_x_mm)
        self.assertAlmostEqual(steps[4].pose.y_mm, args.place_y_mm)
        self.assertAlmostEqual(steps[4].pose.z_mm, args.transfer_z_mm)
        self.assertEqual(
            (steps[2].pose.rx_deg, steps[2].pose.ry_deg, steps[2].pose.rz_deg),
            (-176.794223, 0.889438, -105.490597),
        )
        self.assertEqual(steps[2].delay_before_s, 1.25)
        self.assertTrue(all(step.delay_before_s == 0.0 for step in steps[3:]))
        self.assertEqual(steps[2].mode, args.vertical_mode)
        self.assertEqual(steps[3].mode, args.lift_mode)

    def test_lift_mode_can_be_overridden_to_point(self):
        args = make_args(lift_mode="P")
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.30, 0.02, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertEqual(steps[3].label, "pick_lift")
        self.assertEqual(steps[3].mode, "P")

    def test_force_pick_y_and_adaptive_pitch_marks_only_outside_keep_band(self):
        args = make_args(
            force_pick_x_mm=350.0,
            force_pick_y_mm=0.0,
            dynamic_pitch_by_x=True,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            pitch_deg_per_mm_near=0.03,
            ry_deg=1.353477,
        )
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.40, 0.06, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertAlmostEqual(target.raw_pick_y_mm, 60.0)
        self.assertAlmostEqual(target.raw_pick_x_mm, 400.0)
        self.assertAlmostEqual(target.pick_x_mm, 350.0)
        self.assertAlmostEqual(target.pick_y_mm, 0.0)
        self.assertEqual(target.pick_x_source, "forced_pick_x")
        self.assertEqual(target.pick_y_source, "forced_pick_y")
        for index in (1, 2, 3):
            self.assertAlmostEqual(steps[index].pose.x_mm, 350.0)
            self.assertAlmostEqual(steps[index].pose.y_mm, 0.0)
            self.assertAlmostEqual(steps[index].pose.ry_deg, 1.353477)
            self.assertAlmostEqual(steps[index].pitch_source_x_mm, 400.0)
        self.assertTrue(steps[1].joint_pitch_before_pose)
        self.assertTrue(steps[1].joint_pitch_after_pose)
        self.assertFalse(steps[2].joint_pitch_before_pose)
        self.assertTrue(steps[2].joint_pitch_after_pose)
        self.assertFalse(steps[3].joint_pitch_before_pose)
        self.assertTrue(steps[3].joint_pitch_after_pose)
        self.assertFalse(steps[0].joint_pitch_after_pose)
        self.assertFalse(steps[4].joint_pitch_after_pose)
        self.assertFalse(steps[6].joint_pitch_after_pose)

    def test_pick_x_offset_only_applies_to_planar_ik_adjust_targets(self):
        base_args = dict(
            pick_x_offset_mm=-10.0,
            dynamic_pitch_by_x=True,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            pitch_deg_per_mm_near=0.03,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.46, 0.06, 0.10])
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")

        joint_args = make_args(**base_args, dynamic_pitch_mode="joint")
        joint_target = target_from_command(command, camera_to_base, joint_args)

        self.assertAlmostEqual(joint_target.raw_pick_x_mm, 460.0)
        self.assertAlmostEqual(joint_target.pick_x_mm, 460.0)
        self.assertEqual(joint_target.pick_x_source, "mapped_command_x")

        planar_args = make_args(**base_args, dynamic_pitch_mode="planar-ik")
        planar_target = target_from_command(command, camera_to_base, planar_args)
        planar_steps = build_steps(planar_args, planar_target)

        self.assertAlmostEqual(planar_target.raw_pick_x_mm, 460.0)
        self.assertAlmostEqual(planar_target.pick_x_mm, 450.0)
        self.assertEqual(
            planar_target.pick_x_source,
            "mapped_command_x_planar_ik_offset",
        )
        for index in (1, 2, 3):
            self.assertAlmostEqual(planar_steps[index].pose.x_mm, 450.0)
            self.assertAlmostEqual(planar_steps[index].pitch_source_x_mm, 460.0)
            self.assertTrue(planar_steps[index].planar_ik)

    def test_pick_x_offset_does_not_apply_inside_planar_ik_keep_band(self):
        args = make_args(
            pick_x_offset_mm=-10.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.35, 0.06, 0.10])
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertAlmostEqual(target.raw_pick_x_mm, 350.0)
        self.assertAlmostEqual(target.pick_x_mm, 350.0)
        self.assertEqual(target.pick_x_source, "mapped_command_x")
        for index in (1, 2, 3):
            self.assertAlmostEqual(steps[index].pose.x_mm, 350.0)
            self.assertFalse(steps[index].planar_ik)

    def test_adaptive_pitch_keep_band_keeps_cartesian_orientation_and_no_joint_pitch(self):
        args = make_args(
            force_pick_x_mm=350.0,
            force_pick_y_mm=0.0,
            dynamic_pitch_by_x=True,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            ry_deg=1.353477,
        )
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.35, 0.06, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertAlmostEqual(target.raw_pick_x_mm, 350.0)
        for index in (1, 2, 3):
            self.assertAlmostEqual(steps[index].pose.x_mm, 350.0)
            self.assertAlmostEqual(steps[index].pose.ry_deg, 1.353477)
            self.assertAlmostEqual(steps[index].pitch_source_x_mm, 350.0)
            self.assertFalse(steps[index].joint_pitch_before_pose)
            self.assertFalse(steps[index].joint_pitch_after_pose)

    def test_adaptive_pitch_by_x_clamps_business_pitch_delta(self):
        args = make_args(
            dynamic_pitch_by_x=True,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.5,
            min_dynamic_pitch_deg=-10.0,
            max_dynamic_pitch_deg=10.0,
            ry_deg=1.0,
        )
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.50, 0.0, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertAlmostEqual(steps[2].pose.ry_deg, 1.0)
        self.assertTrue(steps[2].joint_pitch_after_pose)

    def test_planar_ik_dynamic_pitch_returns_ready_before_place(self):
        args = make_args(
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            force_pick_y_mm=0.0,
            ry_deg=1.353477,
        )
        command = CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.0, command_id="cmd")
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.40, 0.06, 0.10])

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertEqual(
            [step.label for step in steps],
            [
                "ready_open",
                "pick_above",
                "pick_grasp",
                "pick_lift",
                "return_ready_above",
                "place_above",
                "place_release",
                "return_ready",
            ],
        )
        for index in (1, 2, 3):
            self.assertAlmostEqual(steps[index].pose.ry_deg, 1.353477)
            self.assertAlmostEqual(steps[index].pitch_source_x_mm, 400.0)
            self.assertFalse(steps[index].joint_pitch_before_pose)
            self.assertFalse(steps[index].joint_pitch_after_pose)
            self.assertTrue(steps[index].planar_ik)
        self.assertTrue(steps[4].planar_ik)
        self.assertAlmostEqual(steps[4].pose.x_mm, args.ready_x_mm)
        self.assertAlmostEqual(steps[4].pose.y_mm, args.ready_y_mm)
        self.assertAlmostEqual(steps[4].pose.z_mm, args.transfer_z_mm)
        self.assertEqual(steps[4].pitch_source_x_mm, args.ready_x_mm)
        for index in (5, 6, 7):
            self.assertFalse(steps[index].planar_ik)

    def test_build_steps_in_picks_selected_target_and_releases_at_ready(self):
        args = make_args(place_pick_delay_s=2.5)
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd",
            task="in",
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.22, -0.04, 0.10])
        target = target_from_command(command, camera_to_base, args)

        steps = build_steps(args, target)

        self.assertEqual(
            [step.label for step in steps],
            [
                "ready_open",
                "return_pick_above",
                "return_pick_grasp",
                "return_pick_lift",
                "return_ready_above",
                "final_release",
            ],
        )
        self.assertEqual([step.gripper_after for step in steps], ["open", None, "close", None, None, "open"])
        self.assertEqual(steps[1].mode, "L")
        self.assertTrue(steps[1].require_reached)
        self.assertEqual(steps[1].delay_before_s, 2.5)
        self.assertAlmostEqual(steps[2].pose.x_mm, 220.0)
        self.assertAlmostEqual(steps[2].pose.y_mm, -40.0)
        self.assertAlmostEqual(steps[2].pose.z_mm, args.grasp_z_mm)
        self.assertEqual(steps[2].mode, args.vertical_mode)
        self.assertEqual(steps[3].mode, args.lift_mode)
        self.assertAlmostEqual(steps[5].pose.x_mm, args.ready_x_mm)
        self.assertAlmostEqual(steps[5].pose.y_mm, args.ready_y_mm)
        self.assertAlmostEqual(steps[5].pose.z_mm, args.transfer_z_mm)

    def test_handle_message_dry_run_returns_out_plan(self):
        args = make_args(debug_plan=True)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.20, 0.10, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.30,
                u=321,
                v=123,
                depth_m=0.30,
                source="unit_yolo",
                command_id="cmd-1",
            )
        )
        message["payload"].update(
            {
                "bbox_xyxy": [300, 100, 340, 180],
                "label": "bottle",
                "confidence": 0.91,
                "class_id": 39,
                "rank": 1,
            }
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
            )

        self.assertEqual(response["type"], "ack")
        self.assertFalse(response["payload"]["executed"])
        self.assertEqual(response["payload"]["task"], "out")
        self.assertEqual(response["payload"]["mapping_source"], "hand_eye")
        self.assertEqual(response["payload"]["sequence"]["kind"], "yolo_out_to_fixed_place")
        self.assertFalse(response["payload"]["sequence"]["release_after_grasp"])
        self.assertTrue(response["payload"]["sequence"]["release_at_place"])
        self.assertFalse(response["payload"]["sequence"]["pick_back_from_place"])
        self.assertAlmostEqual(response["payload"]["pick_x_mm"], 300.0)
        self.assertAlmostEqual(response["payload"]["pick_y_mm"], 50.0)
        self.assertAlmostEqual(response["payload"]["pose"]["z_mm"], args.grasp_z_mm)
        self.assertAlmostEqual(response["payload"]["place_x_mm"], args.place_x_mm)
        self.assertAlmostEqual(response["payload"]["place_y_mm"], args.place_y_mm)
        self.assertAlmostEqual(
            response["payload"]["place_pick_pose"]["z_mm"],
            args.grasp_z_mm,
        )
        self.assertEqual(
            [step["label"] for step in response["payload"]["steps"]],
            [
                "ready_open",
                "pick_above",
                "pick_grasp",
                "pick_lift",
                "place_above",
                "place_release",
                "return_ready",
            ],
        )
        self.assertEqual(
            [step["gripper_after"] for step in response["payload"]["steps"]],
            ["open", None, "close", None, None, "open", None],
        )
        self.assertEqual(response["payload"]["yolo"]["bbox_format"], "xyxy")
        self.assertEqual(response["payload"]["yolo"]["label"], "bottle")
        self.assertEqual(response["payload"]["yolo"]["class_id"], 39)

    def test_handle_message_graspnet_uses_candidate_pose(self):
        args = make_args(
            debug_plan=True,
            vision_source="graspnet",
            graspnet_approach_distance_mm=50.0,
            graspnet_lift_height_mm=40.0,
            grasp_yaw_by_object_short_edge=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.20,
                y_m=0.01,
                z_m=0.12,
                command_id="cmd-graspnet",
                vision_source="graspnet",
                object_short_axis_yaw_deg=45.0,
                grasp_candidates=(
                    GraspCandidate(
                        position_camera_m=(0.20, 0.01, 0.12),
                        rotation_camera=(
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        score=0.8,
                        width_m=0.04,
                        approach_depth_m=0.05,
                        index=4,
                    ),
                ),
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
            )

        self.assertEqual(response["type"], "ack")
        self.assertEqual(response["payload"]["vision_source"], "graspnet")
        self.assertEqual(response["payload"]["graspnet_candidate"]["source_index"], 4)
        self.assertAlmostEqual(response["payload"]["pose"]["x_mm"], 200.0)
        self.assertAlmostEqual(response["payload"]["pose"]["y_mm"], 10.0)
        self.assertAlmostEqual(response["payload"]["pose"]["z_mm"], 120.0)
        self.assertEqual(
            [step["label"] for step in response["payload"]["steps"][:4]],
            ["ready_open", "pick_above", "pick_grasp", "pick_lift"],
        )
        self.assertAlmostEqual(response["payload"]["steps"][1]["pose"]["z_mm"], 70.0)
        self.assertAlmostEqual(response["payload"]["steps"][3]["pose"]["z_mm"], 160.0)
        self.assertEqual(response["payload"]["grasp_yaw_adjustments"], [])

    def test_handle_message_graspnet_rejects_large_rz_delta(self):
        args = make_args(
            debug_plan=True,
            vision_source="graspnet",
            rz_deg=0.0,
            graspnet_max_rz_delta_deg=20.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        rotation_z_90 = (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.20,
                y_m=0.01,
                z_m=0.12,
                command_id="cmd-graspnet-rz-limit",
                vision_source="graspnet",
                grasp_candidates=(
                    GraspCandidate(
                        position_camera_m=(0.20, 0.01, 0.12),
                        rotation_camera=rotation_z_90,
                        score=0.8,
                        index=1,
                    ),
                ),
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(SafetyError, "rz delta"):
                handle_message(
                    message,
                    args=args,
                    camera_to_base=camera_to_base,
                    validator=make_controller(),
                )

    def test_handle_message_dry_run_returns_in_plan(self):
        args = make_args(debug_plan=True)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.23, -0.05, 0.0])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-in",
                task="in",
                rank=2,
                label="placed_block",
                bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
            )

        self.assertEqual(response["type"], "ack")
        self.assertEqual(response["payload"]["task"], "in")
        self.assertEqual(response["payload"]["sequence"]["kind"], "yolo_in_to_ready")
        self.assertTrue(response["payload"]["sequence"]["return_ready"])
        self.assertFalse(response["payload"]["sequence"]["release_at_place"])
        self.assertFalse(response["payload"]["sequence"]["pick_back_from_place"])
        self.assertEqual(
            [step["label"] for step in response["payload"]["steps"]],
            [
                "ready_open",
                "return_pick_above",
                "return_pick_grasp",
                "return_pick_lift",
                "return_ready_above",
                "final_release",
            ],
        )
        self.assertEqual(
            [step["gripper_after"] for step in response["payload"]["steps"]],
            ["open", None, "close", None, None, "open"],
        )
        self.assertAlmostEqual(response["payload"]["pose"]["x_mm"], 230.0)
        self.assertAlmostEqual(response["payload"]["pose"]["y_mm"], -50.0)
        self.assertAlmostEqual(response["payload"]["pose"]["z_mm"], args.grasp_z_mm)
        self.assertEqual(response["payload"]["yolo"]["rank"], 2)
        self.assertEqual(response["payload"]["yolo"]["label"], "placed_block")
        self.assertEqual(response["payload"]["yolo"]["bbox_format"], "xywh")

    def test_handle_dynamic_message_uses_fixed_pick_point_and_reports_timing(self):
        args = make_args(debug_plan=True, t_robot_s=0.30)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.20, 0.10, 0.0])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.30,
                command_id="cmd-dynamic",
                grasp_mode="dynamic",
                t_arrive_s=0.80,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_x_mm=330.0,
                detected_y_mm=20.0,
                detected_z_mm=120.0,
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
            )

        self.assertEqual(response["type"], "ack")
        self.assertEqual(response["payload"]["grasp_mode"], "dynamic")
        self.assertAlmostEqual(response["payload"]["pick_x_mm"], 350.0)
        self.assertAlmostEqual(response["payload"]["pick_y_mm"], -30.0)
        self.assertAlmostEqual(response["payload"]["pose"]["x_mm"], 350.0)
        self.assertAlmostEqual(response["payload"]["pose"]["y_mm"], -30.0)
        self.assertAlmostEqual(response["payload"]["dynamic_timing"]["t_arrive_s"], 0.80)
        self.assertAlmostEqual(response["payload"]["dynamic_timing"]["t_robot_s"], 0.30)
        self.assertAlmostEqual(response["payload"]["dynamic_timing"]["action_time_s"], 0.30)
        self.assertAlmostEqual(response["payload"]["dynamic_timing"]["t_wait_s"], 0.50)
        self.assertEqual(response["payload"]["dynamic_timing"]["t_robot_source"], "constant")

    def test_handle_dynamic_message_reports_x_based_robot_timing_from_raw_x(self):
        args = make_args(
            debug_plan=True,
            t_robot_s=1.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=300.0,
            dynamic_t_robot_s_per_mm=0.002,
            dynamic_t_robot_planar_far_s_per_mm=0.006,
            dynamic_t_robot_margin_s=0.1,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=220.0,
            keep_pitch_max_x_mm=355.0,
            pick_x_offset_mm=-30.0,
            min_target_x_mm=200.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-dynamic-x-time",
                grasp_mode="dynamic",
                t_arrive_s=4.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=420.0,
                pick_y_mm=0.0,
                detected_z_mm=120.0,
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
        )

        timing = response["payload"]["dynamic_timing"]
        expected = 1.8 + (300.0 - 420.0) * 0.006 + 0.1
        self.assertAlmostEqual(response["payload"]["raw_pick_x_mm"], 420.0)
        self.assertAlmostEqual(response["payload"]["pick_x_mm"], 390.0)
        self.assertAlmostEqual(timing["t_robot_s"], expected)
        self.assertEqual(timing["t_robot_source"], "x_linear")
        self.assertEqual(timing["t_robot_segment"], "far")
        self.assertAlmostEqual(timing["t_robot_slope_s_per_mm"], 0.006)
        self.assertAlmostEqual(timing["t_robot_source_x_mm"], 420.0)
        self.assertAlmostEqual(timing["t_robot_reference_x_mm"], 300.0)
        self.assertAlmostEqual(timing["t_robot_base_s"], 1.8)
        self.assertAlmostEqual(timing["t_robot_margin_s"], 0.1)

    def test_dynamic_preposition_pose_uses_detected_object_height_clearance(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_preposition_clearance_mm=20.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.30,
            command_id="cmd-dynamic-prepose",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=120.0,
        )
        target = target_from_command(command, camera_to_base, args)

        pose, source = dynamic_preposition_pose(args, target)

        self.assertEqual(source, "detected_z_plus_clearance")
        self.assertAlmostEqual(pose.x_mm, 350.0)
        self.assertAlmostEqual(pose.y_mm, -30.0)
        self.assertAlmostEqual(pose.z_mm, 140.0)

    def test_dynamic_preposition_pose_uses_top_height_in_top_detected_mode(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_preposition_clearance_mm=20.0,
            use_detected_z="top",
        )
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.30,
            command_id="cmd-dynamic-prepose-top",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=120.0,
            detected_top_z_mm=150.0,
        )
        target = target_from_command(command, camera_to_base, args)

        pose, source = dynamic_preposition_pose(args, target)

        self.assertEqual(source, "detected_top_z_plus_clearance")
        self.assertAlmostEqual(pose.x_mm, 350.0)
        self.assertAlmostEqual(pose.y_mm, -30.0)
        self.assertAlmostEqual(pose.z_mm, 170.0)

    def test_use_detected_z_clamps_low_dynamic_grasp_to_fixed_safe_floor(self):
        args = make_args(use_detected_z=True, detected_z_offset_mm=130.0)
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=-0.50,
            command_id="cmd-dynamic-low-z",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=-20.0,
        )

        target = target_from_command(command, camera_to_base, args)

        self.assertAlmostEqual(target.raw_grasp_z_mm, 110.0)
        self.assertAlmostEqual(target.grasp_z_mm, args.grasp_z_mm)
        self.assertEqual(
            target.grasp_z_source,
            "dynamic_detected_z_plus_offset_clamped_to_fixed_grasp_z",
        )

    def test_use_detected_z_prefers_dynamic_detected_z_over_command_point_z(self):
        args = make_args(use_detected_z=True, detected_z_offset_mm=10.0)
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=-0.50,
            command_id="cmd-dynamic-detected-z",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=150.0,
        )

        target = target_from_command(command, camera_to_base, args)

        self.assertAlmostEqual(target.raw_grasp_z_mm, 160.0)
        self.assertAlmostEqual(target.grasp_z_mm, 160.0)
        self.assertEqual(target.grasp_z_source, "dynamic_detected_z_plus_offset")

    def test_use_detected_z_top_uses_command_top_z(self):
        args = make_args(use_detected_z="top", detected_z_offset_mm=8.0)
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=-0.50,
            command_id="cmd-top-detected-z",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=120.0,
            detected_top_z_mm=160.0,
        )

        target = target_from_command(command, camera_to_base, args)

        self.assertAlmostEqual(target.raw_grasp_z_mm, 168.0)
        self.assertAlmostEqual(target.grasp_z_mm, 168.0)
        self.assertEqual(target.grasp_z_source, "detected_top_z_plus_offset")

    def test_use_detected_z_top_requires_command_top_z(self):
        args = make_args(use_detected_z="top", detected_z_offset_mm=0.0)
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.12,
            command_id="cmd-missing-top-z",
        )

        with self.assertRaisesRegex(ValueError, "detected_top_z_mm"):
            target_from_command(command, camera_to_base, args)

    def test_grasp_yaw_adjustment_is_disabled_by_default(self):
        args = make_args(grasp_yaw_by_object_short_edge=False)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-yaw-disabled",
            object_short_axis_yaw_deg=80.0,
        )
        target = target_from_command(command, np.eye(4, dtype=float), args)
        controller = make_controller()

        result = apply_grasp_yaw_adjustment(controller, args, target)

        self.assertFalse(result.enabled)
        self.assertFalse(result.applied)
        self.assertEqual(
            [call for call in controller.sdk.calls if call[0] == "JointCtrl"],
            [],
        )

    def test_grasp_yaw_adjustment_rotates_only_j6_and_clamps_delta(self):
        args = make_args(
            grasp_yaw_by_object_short_edge=True,
            gripper_close_axis_offset_deg=0.0,
            grasp_yaw_max_delta_deg=30.0,
            grasp_yaw_deadband_deg=0.0,
            grasp_yaw_speed=20,
            command_duration_s=0.0,
        )
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-yaw-adjust",
            object_short_axis_yaw_deg=80.0,
        )
        target = target_from_command(command, np.eye(4, dtype=float), args)
        controller = make_controller()

        result = apply_grasp_yaw_adjustment(controller, args, target)

        self.assertTrue(result.enabled)
        self.assertTrue(result.applied)
        self.assertTrue(result.clamped)
        self.assertAlmostEqual(result.delta_deg, 30.0)
        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), 1)
        self.assertEqual(joint_calls[0][1][:5], (0, 0, 0, 0, 0))
        self.assertAlmostEqual(joint_calls[0][1][5], 30000, delta=1)

    def test_grasp_yaw_adjustment_uses_nearest_180_equivalent_axis(self):
        args = make_args(
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=90.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
        )
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-yaw-equivalent",
            object_short_axis_yaw_deg=170.0,
        )
        target = target_from_command(command, np.eye(4, dtype=float), args)
        controller = make_controller()

        result = apply_grasp_yaw_adjustment(controller, args, target)

        self.assertTrue(result.applied)
        self.assertAlmostEqual(result.delta_deg, -10.0)

    def test_grasp_yaw_adjustment_clamps_to_joint6_limits(self):
        args = make_args(
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=45.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
        )
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-yaw-limit",
            object_short_axis_yaw_deg=-150.0,
        )
        target = target_from_command(command, np.eye(4, dtype=float), args)
        controller = make_controller()
        controller.sdk.joint_feedback = joint_feedback_from_degrees(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -119.0,
        )

        result = apply_grasp_yaw_adjustment(controller, args, target)

        self.assertTrue(result.applied)
        self.assertTrue(result.clamped)
        self.assertTrue(result.limit_clamped)
        self.assertAlmostEqual(result.delta_deg, -0.9)
        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), 1)
        self.assertAlmostEqual(joint_calls[0][1][5], -119900, delta=1)

    def test_grasp_yaw_adjustment_chooses_reachable_180_equivalent_axis(self):
        args = make_args(
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=90.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
        )
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-yaw-reachable-equivalent",
            object_short_axis_yaw_deg=170.0,
        )
        target = target_from_command(command, np.eye(4, dtype=float), args)
        controller = make_controller()
        controller.sdk.joint_feedback = joint_feedback_from_degrees(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,
        )

        result = apply_grasp_yaw_adjustment(controller, args, target)

        self.assertTrue(result.applied)
        self.assertFalse(result.limit_clamped)
        self.assertAlmostEqual(result.delta_deg, -90.0)
        self.assertAlmostEqual(result.target_joints_deg[5], -10.0)  # type: ignore[index]

    def test_execute_plan_applies_grasp_yaw_before_grasp_descent_and_preserves_orientation(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            no_gripper=False,
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=30.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-yaw-execute",
                object_short_axis_yaw_deg=20.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 20.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        yaw_results = response["payload"]["grasp_yaw_adjustments"]
        self.assertEqual(len(yaw_results), 1)
        self.assertTrue(yaw_results[0]["applied"])
        calls = controller.sdk.calls
        pick_above_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "EndPoseCtrl" and call[1][2] == round(args.transfer_z_mm * 1000)
        )
        pick_grasp_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "EndPoseCtrl" and call[1][2] == round(args.grasp_z_mm * 1000)
        )
        yaw_joint_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "JointCtrl" and call[1][5] != 0
        )
        close_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "GripperCtrl" and call[1][0] == -100000
        )
        self.assertLess(pick_above_index, yaw_joint_index)
        self.assertLess(yaw_joint_index, pick_grasp_index)
        self.assertLess(yaw_joint_index, close_index)
        self.assertEqual(calls[pick_grasp_index][1][5], 20000)

    def test_planar_ik_grasp_z_compensation_lowers_far_grasp_only(self):
        args = make_args(
            use_detected_z=True,
            detected_z_offset_mm=190.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            planar_ik_grasp_z_comp_mm_per_deg=2.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.30,
            command_id="cmd-far-z-comp",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=430.0,
            pick_y_mm=0.0,
            detected_z_mm=10.0,
        )

        target = target_from_command(command, camera_to_base, args)
        steps = build_steps(args, target)

        self.assertAlmostEqual(target.raw_grasp_z_mm, 200.0)
        self.assertAlmostEqual(target.uncompensated_grasp_z_mm, 200.0)
        self.assertAlmostEqual(target.planar_ik_grasp_z_comp_pitch_delta_deg, 2.0)
        self.assertAlmostEqual(target.planar_ik_grasp_z_comp_mm, 4.0)
        self.assertAlmostEqual(target.grasp_z_mm, 196.0)
        self.assertEqual(
            target.grasp_z_source,
            "dynamic_detected_z_plus_offset_planar_ik_pitch_z_comp",
        )
        self.assertAlmostEqual(steps[2].pose.z_mm, 196.0)
        self.assertAlmostEqual(steps[1].pose.z_mm, args.transfer_z_mm)
        self.assertAlmostEqual(steps[3].pose.z_mm, args.transfer_z_mm)

    def test_planar_ik_grasp_z_compensation_keeps_near_target_unchanged(self):
        args = make_args(
            use_detected_z=True,
            detected_z_offset_mm=190.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            planar_ik_grasp_z_comp_mm_per_deg=2.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.30,
            command_id="cmd-near-z-no-comp",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            pick_x_mm=350.0,
            pick_y_mm=0.0,
            detected_z_mm=10.0,
        )

        target = target_from_command(command, camera_to_base, args)

        self.assertAlmostEqual(target.raw_grasp_z_mm, 200.0)
        self.assertAlmostEqual(target.grasp_z_mm, 200.0)
        self.assertAlmostEqual(target.planar_ik_grasp_z_comp_mm, 0.0)
        self.assertEqual(target.grasp_z_source, "dynamic_detected_z_plus_offset")

    def test_dynamic_preposition_timing_uses_final_action_time(self):
        args = make_args(
            debug_plan=True,
            dynamic_preposition=True,
            t_robot_s=2.80,
            final_action_time_s=0.55,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-final-time",
                grasp_mode="dynamic",
                t_arrive_s=0.80,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=make_controller(),
            )

        timing = response["payload"]["dynamic_timing"]
        self.assertFalse(timing["skipped"])
        self.assertTrue(timing["dynamic_preposition"])
        self.assertAlmostEqual(timing["t_robot_s"], 2.80)
        self.assertAlmostEqual(timing["action_time_s"], 0.55)
        self.assertAlmostEqual(timing["t_wait_s"], 0.25)
        self.assertEqual(timing["wait_placement"], "wait")
        self.assertEqual(
            response["payload"]["dynamic_execution"]["preposition_enabled"],
            True,
        )
        self.assertEqual(
            response["payload"]["dynamic_execution"]["wait_placement"],
            "wait",
        )

    def test_dynamic_preposition_nowait_timing_uses_robot_time(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=2.80,
            final_action_time_s=0.55,
        )
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-dynamic-nowait-timing",
            grasp_mode="dynamic",
            t_arrive_s=3.20,
            conveyor_speed_mm_s=100.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_z_mm=120.0,
        )

        timing = dynamic_timing_from_command(command, args)

        assert timing is not None
        self.assertFalse(timing.skipped)
        self.assertAlmostEqual(timing.t_robot_s, 2.80)
        self.assertAlmostEqual(timing.action_time_s, 2.80)
        self.assertAlmostEqual(timing.t_wait_s, 0.40)
        self.assertEqual(timing.wait_placement, "nowait")
        self.assertAlmostEqual(timing.final_action_time_s, 0.55)

    def test_dynamic_t_robot_by_x_disabled_keeps_constant_robot_time(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=2.0,
            dynamic_t_robot_by_x=False,
            dynamic_t_robot_s_per_mm=0.01,
            dynamic_t_robot_planar_far_s_per_mm=0.02,
        )
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-constant-time",
            grasp_mode="dynamic",
            t_arrive_s=3.0,
            pick_x_mm=450.0,
            pick_y_mm=0.0,
        )

        timing = dynamic_timing_from_command(command, args, source_x_mm=450.0)

        assert timing is not None
        self.assertAlmostEqual(timing.t_robot_s, 2.0)
        self.assertEqual(timing.t_robot_source, "constant")
        self.assertIsNone(timing.t_robot_segment)
        self.assertIsNone(timing.t_robot_slope_s_per_mm)
        self.assertAlmostEqual(timing.action_time_s, 2.0)

    def test_dynamic_t_robot_by_x_linear_uses_reference_distance(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=1.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=300.0,
            dynamic_t_robot_s_per_mm=0.002,
            dynamic_t_robot_margin_s=0.1,
        )
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-ordinary-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=340.0,
            pick_y_mm=0.0,
        )

        timing = dynamic_timing_from_command(command, args, source_x_mm=340.0)

        assert timing is not None
        self.assertAlmostEqual(timing.t_robot_s, 1.8 + (300.0 - 340.0) * 0.002 + 0.1)
        self.assertEqual(timing.t_robot_source, "x_linear")
        self.assertEqual(timing.t_robot_segment, "far")
        self.assertAlmostEqual(timing.t_robot_slope_s_per_mm, 0.002)
        self.assertAlmostEqual(timing.t_robot_reference_x_mm, 300.0)
        self.assertAlmostEqual(timing.t_robot_base_s, 1.8)
        self.assertAlmostEqual(timing.t_wait_s, 4.0 - timing.t_robot_s)

    def test_dynamic_t_robot_by_x_positive_slope_makes_smaller_x_slower(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=2.1,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=300.0,
            dynamic_t_robot_s_per_mm=0.002,
        )
        near_command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-near-x-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=260.0,
            pick_y_mm=0.0,
        )
        far_command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-far-x-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=340.0,
            pick_y_mm=0.0,
        )

        near_timing = dynamic_timing_from_command(
            near_command,
            args,
            source_x_mm=260.0,
        )
        far_timing = dynamic_timing_from_command(
            far_command,
            args,
            source_x_mm=340.0,
        )

        assert near_timing is not None
        assert far_timing is not None
        self.assertAlmostEqual(near_timing.t_robot_s, 2.1 + 40.0 * 0.002)
        self.assertAlmostEqual(far_timing.t_robot_s, 2.1 - 40.0 * 0.002)
        self.assertEqual(near_timing.t_robot_segment, "near")
        self.assertEqual(far_timing.t_robot_segment, "far")
        self.assertAlmostEqual(near_timing.t_robot_slope_s_per_mm, 0.002)
        self.assertAlmostEqual(far_timing.t_robot_slope_s_per_mm, 0.002)
        self.assertGreater(near_timing.t_robot_s, far_timing.t_robot_s)

    def test_dynamic_t_robot_by_x_uses_side_specific_slopes(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=1.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=300.0,
            dynamic_t_robot_s_per_mm=0.002,
            dynamic_t_robot_planar_far_s_per_mm=0.006,
            dynamic_t_robot_planar_near_s_per_mm=0.004,
            dynamic_t_robot_margin_s=0.1,
        )
        near_command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-near-side-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=260.0,
            pick_y_mm=0.0,
        )
        far_command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-far-side-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=420.0,
            pick_y_mm=0.0,
        )

        near_timing = dynamic_timing_from_command(near_command, args, source_x_mm=260.0)
        far_timing = dynamic_timing_from_command(far_command, args, source_x_mm=420.0)

        assert near_timing is not None
        assert far_timing is not None
        self.assertAlmostEqual(near_timing.t_robot_s, 1.8 + 40.0 * 0.004 + 0.1)
        self.assertAlmostEqual(far_timing.t_robot_s, 1.8 - 120.0 * 0.006 + 0.1)
        self.assertEqual(near_timing.t_robot_source, "x_linear")
        self.assertEqual(far_timing.t_robot_source, "x_linear")
        self.assertEqual(near_timing.t_robot_segment, "near")
        self.assertEqual(far_timing.t_robot_segment, "far")
        self.assertAlmostEqual(near_timing.t_robot_slope_s_per_mm, 0.004)
        self.assertAlmostEqual(far_timing.t_robot_slope_s_per_mm, 0.006)
        self.assertAlmostEqual(far_timing.t_robot_source_x_mm, 420.0)

    def test_dynamic_t_robot_by_x_reference_uses_base_time(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=2.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=340.0,
            dynamic_t_robot_s_per_mm=0.002,
            dynamic_t_robot_planar_far_s_per_mm=0.010,
            dynamic_t_robot_planar_near_s_per_mm=0.017,
        )
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            command_id="cmd-dynamic-reference-time",
            grasp_mode="dynamic",
            t_arrive_s=4.0,
            pick_x_mm=340.0,
            pick_y_mm=0.0,
        )

        timing = dynamic_timing_from_command(command, args, source_x_mm=340.0)

        assert timing is not None
        self.assertAlmostEqual(timing.t_robot_s, 2.8)
        self.assertEqual(timing.t_robot_segment, "reference")
        self.assertAlmostEqual(timing.t_robot_slope_s_per_mm, 0.0)

    def test_execute_dynamic_preposition_direct_descent_success(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-direct-descent",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        self.assertAlmostEqual(
            response["payload"]["dynamic_execution"]["preposition_z_mm"],
            140.0,
        )
        self.assertNotIn("guarded_descent", response["payload"]["dynamic_execution"])
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
            ],
        )
        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertGreaterEqual(len(end_pose_calls), 5)

    def test_execute_dynamic_preposition_reports_final_timing_window(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.05,
            t_robot_s=2.80,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-final-window",
                grasp_mode="dynamic",
                t_arrive_s=0.05,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        execution = response["payload"]["dynamic_execution"]
        self.assertAlmostEqual(execution["final_action_due_s"], 0.0)
        self.assertEqual(execution["final_wait_s"], 0.0)
        self.assertGreaterEqual(execution["final_action_late_s"], 0.0)

    def test_dynamic_preposition_nowait_waits_at_ready_before_target_moves(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            final_action_time_s=0.25,
            t_robot_s=2.80,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-nowait",
                grasp_mode="dynamic",
                t_arrive_s=3.20,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        call_order: list[str] = []

        original_move_to_pose = controller.move_to_pose

        def record_move_to_pose(pose, *args_, **kwargs_):
            if math.isclose(pose.x_mm, 350.0) and math.isclose(pose.y_mm, 0.0):
                call_order.append("ready_open")
            elif math.isclose(pose.z_mm, 253.252):
                call_order.append("pick_above")
            elif math.isclose(pose.z_mm, 140.0):
                call_order.append("dynamic_preposition")
            elif math.isclose(pose.z_mm, 133.252):
                call_order.append("pick_grasp")
            return original_move_to_pose(pose, *args_, **kwargs_)

        def record_wait(*args_, **kwargs_):
            call_order.append("wait")
            return 0.0

        with mock.patch.object(
            controller,
            "move_to_pose",
            side_effect=record_move_to_pose,
        ), mock.patch(
            "robot_control.linux.catch.catch_server_1.wait_for_dynamic_final_action",
            side_effect=record_wait,
        ) as wait_for_final_action, contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        self.assertEqual(response["payload"]["dynamic_timing"]["wait_placement"], "nowait")
        self.assertEqual(
            response["payload"]["dynamic_execution"]["wait_placement"],
            "nowait",
        )
        self.assertEqual(wait_for_final_action.call_count, 2)
        self.assertAlmostEqual(
            wait_for_final_action.call_args_list[0].kwargs["final_action_time_s"],
            2.80,
        )
        self.assertEqual(wait_for_final_action.call_args_list[0].kwargs["placement"], "nowait")
        self.assertAlmostEqual(
            wait_for_final_action.call_args_list[1].kwargs["final_action_time_s"],
            0.25,
        )
        self.assertEqual(
            wait_for_final_action.call_args_list[1].kwargs["placement"],
            "preposition_after_nowait",
        )
        self.assertEqual(
            [item for item in call_order if item in ("ready_open", "wait", "pick_above")][:3],
            ["ready_open", "wait", "pick_above"],
        )
        self.assertLess(call_order.index("wait"), call_order.index("dynamic_preposition"))
        self.assertLess(call_order.index("dynamic_preposition"), call_order.index("pick_grasp"))

    def test_dynamic_preposition_applies_grasp_yaw_after_pick_above_before_preposition(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=30.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-yaw-before-descent",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
                object_short_axis_yaw_deg=20.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        yaw_results = response["payload"]["grasp_yaw_adjustments"]
        self.assertEqual(len(yaw_results), 1)
        self.assertTrue(yaw_results[0]["applied"])
        calls = controller.sdk.calls
        pick_above_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "EndPoseCtrl"
            and call[1][1] == -30000
            and call[1][2] == round(args.transfer_z_mm * 1000)
        )
        dynamic_preposition_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "EndPoseCtrl" and call[1][2] == 140000
        )
        yaw_joint_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "JointCtrl" and call[1][5] != 0
        )
        pick_grasp_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "EndPoseCtrl" and call[1][2] == round(args.grasp_z_mm * 1000)
        )
        close_index = next(
            index
            for index, call in enumerate(calls)
            if call[0] == "GripperCtrl" and call[1][0] == -100000
        )
        self.assertLess(pick_above_index, yaw_joint_index)
        self.assertLess(yaw_joint_index, dynamic_preposition_index)
        self.assertLess(yaw_joint_index, pick_grasp_index)
        self.assertLess(pick_grasp_index, close_index)
        self.assertEqual(calls[dynamic_preposition_index][1][5], 20000)
        self.assertEqual(calls[pick_grasp_index][1][5], 20000)

    def test_handle_probe_message_only_reports_base_coordinates(self):
        args = make_args(execute=True)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.20, 0.10, 0.0])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.30,
                u=1138,
                v=287,
                depth_m=0.521,
                command_id="cmd-probe",
                label="cup",
                rank=1,
                bbox_xywh=(1049.0, 218.0, 179.0, 198.0),
                grasp_mode="probe",
            )
        )
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertEqual(response["type"], "ack")
        self.assertFalse(response["payload"]["executed"])
        self.assertTrue(response["payload"]["probe_only"])
        self.assertEqual(response["payload"]["grasp_mode"], "probe")
        self.assertAlmostEqual(response["payload"]["base_xyz_m"]["x_m"], 0.30)
        self.assertAlmostEqual(response["payload"]["base_xyz_m"]["y_m"], 0.05)
        self.assertAlmostEqual(response["payload"]["base_xyz_m"]["z_m"], 0.30)
        self.assertEqual(response["payload"]["yolo"]["label"], "cup")
        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(end_pose_calls, [])

    def test_handle_dynamic_message_skips_when_target_is_too_close_if_enabled(self):
        args = make_args(execute=True, t_robot_s=0.30, dynamic_too_close_skip=True)
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-late",
                grasp_mode="dynamic",
                t_arrive_s=0.10,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
            )
        )
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertEqual(response["type"], "ack")
        self.assertFalse(response["payload"]["executed"])
        self.assertTrue(response["payload"]["skipped"])
        self.assertTrue(response["payload"]["dynamic_timing"]["skipped"])
        self.assertLess(response["payload"]["dynamic_timing"]["t_wait_s"], 0.0)
        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(end_pose_calls, [])

    def test_handle_dynamic_message_accepts_late_target_when_skip_disabled(self):
        args = make_args(
            execute=False,
            t_robot_s=0.30,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-late-allowed",
                grasp_mode="dynamic",
                t_arrive_s=0.10,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
            )
        )
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertEqual(response["type"], "ack")
        self.assertFalse(response["payload"]["executed"])
        self.assertFalse(response["payload"]["skipped"])
        timing = response["payload"]["dynamic_timing"]
        self.assertFalse(timing["skipped"])
        self.assertFalse(timing["too_close_skip_enabled"])
        self.assertLess(timing["t_wait_s"], 0.0)

    def test_wait_for_dynamic_timing_starts_immediately_when_late_skip_disabled(self):
        args = make_args(t_robot_s=0.30, dynamic_too_close_skip=False)
        command = CameraPointCommand(
            x_m=0.35,
            y_m=0.02,
            z_m=0.30,
            command_id="cmd-dynamic-late-allowed",
            grasp_mode="dynamic",
            t_arrive_s=0.10,
            conveyor_speed_mm_s=100.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
        )
        timing = dynamic_timing_from_command(command, args, source_x_mm=350.0)

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertFalse(timing.skipped)
        self.assertLess(timing.t_wait_s, 0.0)
        with mock.patch("robot_control.linux.catch.catch_server_1.time.sleep") as sleep:
            with contextlib.redirect_stdout(io.StringIO()):
                wait_for_dynamic_timing(timing)

        sleep.assert_not_called()

    def test_execute_default_runs_out_only(self):
        args = make_args(
            execute=True,
            place_pick_delay_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-2")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        sdk = controller.sdk
        gripper_calls = [call for call in sdk.calls if call[0] == "GripperCtrl"]
        end_pose_calls = [call for call in sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
                ("GripperCtrl", (0, 1000, 0x01, 0)),
            ],
        )
        self.assertEqual(len(end_pose_calls), 7)
        self.assertEqual(response["payload"]["sequence"]["kind"], "yolo_out_to_fixed_place")
        self.assertEqual(
            [step["label"] for step in response["payload"]["steps"]],
            [
                "ready_open",
                "pick_above",
                "pick_grasp",
                "pick_lift",
                "place_above",
                "place_release",
                "return_ready",
            ],
        )

    def test_pick_above_requires_reached_even_when_no_wait_waypoints(self):
        args = make_args(
            execute=True,
            no_wait_waypoints=True,
            motion_timeout_s=0.0,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.30, 0.02, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-stop")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597),
            freeze_pose=True,
        )

        with self.assertRaisesRegex(RuntimeError, "waypoint pick_above not reached"):
            with contextlib.redirect_stdout(io.StringIO()):
                handle_message(
                    message,
                    args=args,
                    camera_to_base=camera_to_base,
                    validator=controller,
                    controller=controller,
                )

        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(
            end_pose_calls,
            [
                ("EndPoseCtrl", (350000, 0, 253252, -176794, 889, -105491)),
                ("EndPoseCtrl", (300000, 20000, 253252, -176794, 889, -105491)),
            ],
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(gripper_calls, [("GripperCtrl", (0, 1000, 0x01, 0))])

    def test_dynamic_pitch_execution_only_changes_allowed_joints(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_pitch_by_x=True,
            force_pick_x_mm=350.0,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.125,
            dynamic_pitch_speed=12,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.40, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-pitch")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 5.0, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 5.0, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 5.0, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 5.0, 60.0),
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual(
            joint_calls,
            [
                ("JointCtrl", (10000, 20000, -30000, 40000, 7500, 60000)),
                ("JointCtrl", (10000, 20000, -30000, 40000, 7500, 60000)),
                ("JointCtrl", (10000, 20000, -30000, 40000, 7500, 60000)),
                ("JointCtrl", (10000, 20000, -30000, 40000, 7500, 60000)),
            ],
        )
        motion_j_calls = [
            call for call in controller.sdk.calls
            if call == ("MotionCtrl_2", (0x01, 0x01, 12, 0x00, 0, 0x00))
        ]
        self.assertEqual(len(motion_j_calls), 4)
        end_pose_calls = [
            call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"
        ]
        self.assertEqual(end_pose_calls[1][1][0], 350000)
        self.assertAlmostEqual(response["payload"]["raw_pick_x_mm"], 400.0)
        self.assertAlmostEqual(response["payload"]["pick_x_mm"], 350.0)
        self.assertEqual(response["payload"]["pick_x_source"], "forced_pick_x")
        self.assertEqual(
            response["payload"]["dynamic_pitch"]["method"],
            "joint_space_j2_j3_j5",
        )
        self.assertAlmostEqual(
            response["payload"]["dynamic_pitch"]["pitch_delta_deg"],
            2.5,
        )
        self.assertEqual(response["payload"]["dynamic_pitch"]["mode"], "adjust")
        self.assertAlmostEqual(response["payload"]["dynamic_pitch"]["over_mm"], 20.0)
        self.assertAlmostEqual(
            response["payload"]["dynamic_pitch"]["raw_pitch_delta_deg"],
            2.5,
        )
        self.assertAlmostEqual(
            response["payload"]["dynamic_pitch"]["requested_pitch_delta_deg"],
            2.5,
        )
        self.assertEqual(len(response["payload"]["joint_pitch_adjustments"]), 4)
        self.assertEqual(
            [
                adjustment["phase"]
                for adjustment in response["payload"]["joint_pitch_adjustments"]
            ],
            ["before_pose", "after_pose", "after_pose", "after_pose"],
        )
        for adjustment in response["payload"]["joint_pitch_adjustments"]:
            self.assertAlmostEqual(adjustment["pitch_source_x_mm"], 400.0)
            self.assertAlmostEqual(adjustment["j2_delta_deg"], 0.0)
            self.assertAlmostEqual(adjustment["j3_delta_deg"], 0.0)
            self.assertAlmostEqual(adjustment["j5_delta_deg"], 2.5)

    def test_adaptive_pitch_keep_band_execution_sends_no_joint_commands(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_pitch_by_x=True,
            force_pick_x_mm=350.0,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.35, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-pitch-keep")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual(joint_calls, [])
        self.assertEqual(response["payload"]["dynamic_pitch"]["mode"], "keep_fixed")
        self.assertAlmostEqual(
            response["payload"]["dynamic_pitch"]["pitch_delta_deg"],
            0.0,
        )
        self.assertEqual(response["payload"]["joint_pitch_adjustments"], [])
        for step in response["payload"]["steps"][1:4]:
            self.assertFalse(step["joint_pitch_before_pose"])
            self.assertFalse(step["joint_pitch_after_pose"])

    def test_planar_ik_dynamic_pitch_sends_only_joint_commands_for_pick_steps(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            force_pick_x_mm=350.0,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.125,
            speed=50,
            planar_ik_speed=25,
            command_duration_s=0.0,
            no_wait_waypoints=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.40, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-cart-pitch")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(0.0, 114.592, -94.538, 0.0, 68.755, -77.349),
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual(len(joint_calls), 3)
        for _, joints in joint_calls:
            self.assertEqual(joints[0], 0)
            self.assertEqual(joints[3], 0)
            self.assertEqual(joints[5], -77349)
        motion_j_calls = [
            call for call in controller.sdk.calls
            if call == ("MotionCtrl_2", (0x01, 0x01, 25, 0x00, 0, 0x00))
        ]
        self.assertEqual(len(motion_j_calls), 3)
        end_pose_calls = [
            call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"
        ]
        self.assertEqual(len(end_pose_calls), 1)
        self.assertEqual(response["payload"]["dynamic_pitch"]["method"], "planar_ik_j2_j3_j5")
        self.assertEqual(response["payload"]["dynamic_pitch"]["control_mode"], "planar_ik")
        self.assertEqual(response["payload"]["joint_pitch_adjustments"], [])
        self.assertEqual(len(response["payload"]["planar_ik_results"]), 3)
        first_result = response["payload"]["planar_ik_results"][0]
        self.assertAlmostEqual(first_result["target_pose"]["x_mm"], 350.0)
        self.assertLessEqual(
            abs(first_result["ry_error_deg"]),
            args.planar_ik_ry_tolerance_deg,
        )

    def test_dynamic_preposition_reuses_pick_above_planar_ik_for_lift_then_places(self):
        args = make_args(
            execute=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            command_duration_s=0.0,
            no_wait_waypoints=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-dynamic-planar-reuse",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=457.775,
                pick_y_mm=0.0,
                detected_z_mm=184.737,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(0.0, 10.0, -20.0, 0.0, 30.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 12.0, -22.0, 0.0, 32.0, -70.0),
            joint_feedback_from_degrees(0.0, 13.0, -23.0, 0.0, 33.0, -70.0),
            joint_feedback_from_degrees(0.0, 14.0, -24.0, 0.0, 34.0, -70.0),
        ]
        solutions = [
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 11.0, -21.0, 0.0, 31.0, -70.0)),
                solved_pose=EndPose(457.775, 0.0, 253.252, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                solved_fk_pose=EndPose(445.0, 0.0, 323.0, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 0.0, 32.0, -70.0)),
                solved_pose=EndPose(457.775, 0.0, 204.737, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                solved_fk_pose=EndPose(445.0, 0.0, 274.0, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 13.0, -23.0, 0.0, 33.0, -70.0)),
                solved_pose=EndPose(457.775, 0.0, 204.737, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                solved_fk_pose=EndPose(445.0, 0.0, 274.0, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 14.0, -24.0, 0.0, 34.0, -70.0)),
                solved_pose=EndPose(350.0, 0.0, 253.252, args.rx_deg, args.ry_deg, args.rz_deg),
                solved_fk_pose=EndPose(338.0, 0.0, 323.0, args.rx_deg, args.ry_deg, args.rz_deg),
                pitch_delta_deg=0.0,
                requested_pitch_delta_deg=0.0,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
        ]

        with mock.patch(
            "robot_control.linux.catch.catch_server_1.solve_planar_ik_j2_j3_j5",
            side_effect=solutions,
        ) as solve_ik, contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        self.assertEqual(solve_ik.call_count, 4)
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "dynamic_preposition",
                "pick_grasp",
                "pick_lift",
                "return_ready_above",
                "place_above",
                "place_release",
                "return_ready",
            },
        )
        results = response["payload"]["planar_ik_results"]
        self.assertEqual([result["step_label"] for result in results], [
            "pick_above",
            "dynamic_preposition",
            "pick_grasp",
            "pick_lift",
            "return_ready_above",
        ])
        self.assertTrue(results[3]["reused"])
        self.assertEqual(results[3]["reused_from_step_label"], "pick_above")
        self.assertAlmostEqual(results[-1]["target_pose"]["x_mm"], args.ready_x_mm)
        self.assertAlmostEqual(results[-1]["requested_pitch_delta_deg"], 0.0)
        end_pose_calls = [
            call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"
        ]
        self.assertIn(
            (
                "EndPoseCtrl",
                (
                    round(args.place_x_mm * 1000),
                    round(args.place_y_mm * 1000),
                    round(args.transfer_z_mm * 1000),
                    round(args.rx_deg * 1000),
                    round(args.ry_deg * 1000),
                    round(args.rz_deg * 1000),
                ),
            ),
            end_pose_calls,
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
                ("GripperCtrl", (0, 1000, 0x01, 0)),
            ],
        )

    def test_dynamic_preposition_planar_ik_yaw_after_pick_above_is_preserved_through_lift(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_max_delta_deg=90.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
            no_wait_waypoints=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-dynamic-planar-yaw",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=457.775,
                pick_y_mm=0.0,
                detected_z_mm=184.737,
                object_short_axis_yaw_deg=20.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(0.0, 10.0, -20.0, 0.0, 30.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 12.0, -22.0, 0.0, 32.0, 20.0),
            joint_feedback_from_degrees(0.0, 13.0, -23.0, 0.0, 33.0, 20.0),
        ]
        solutions = [
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 11.0, -21.0, 0.0, 31.0, -70.0)),
                solved_pose=EndPose(457.775, 0.0, 253.252, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                solved_fk_pose=EndPose(445.0, 0.0, 323.0, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 0.0, 32.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 204.737, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 274.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 13.0, -23.0, 0.0, 33.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 184.737, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 254.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 14.0, -24.0, 0.0, 34.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 253.252, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 323.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
        ]

        with mock.patch(
            "robot_control.linux.catch.catch_server_1.solve_planar_ik_j2_j3_j5",
            side_effect=solutions,
        ) as solve_ik, contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        self.assertEqual(solve_ik.call_count, 4)
        results = response["payload"]["planar_ik_results"]
        self.assertEqual(
            [result["step_label"] for result in results],
            ["pick_above", "dynamic_preposition", "pick_grasp", "pick_lift"],
        )
        self.assertFalse(results[3]["reused"])
        self.assertAlmostEqual(results[1]["target_pose"]["rz_deg"], 20.0)
        self.assertAlmostEqual(results[2]["target_pose"]["rz_deg"], 20.0)
        self.assertAlmostEqual(results[3]["target_pose"]["rz_deg"], 20.0)
        self.assertEqual(results[0]["joint_indices"], [2, 3, 5])
        self.assertEqual(results[1]["joint_indices"], [2, 3, 4, 5, 6])
        self.assertEqual(results[2]["joint_indices"], [2, 3, 4, 5, 6])
        self.assertEqual(results[3]["joint_indices"], [2, 3, 4, 5, 6])
        self.assertEqual(
            [call.kwargs["joint_indices"] for call in solve_ik.call_args_list],
            [(1, 2, 4), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)],
        )
        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual([call[1][5] for call in joint_calls], [-70000, 20000, 20000, 20000])

    def test_dynamic_preposition_separate_j6_yaw_keeps_planar_ik_on_j2_j3_j5(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            dynamic_pitch_by_x=True,
            dynamic_pitch_mode="planar-ik",
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.04,
            grasp_yaw_by_object_short_edge=True,
            grasp_yaw_mode="separate-j6",
            grasp_yaw_max_delta_deg=90.0,
            grasp_yaw_deadband_deg=0.0,
            command_duration_s=0.0,
            no_wait_waypoints=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-dynamic-separate-j6-yaw",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=457.775,
                pick_y_mm=0.0,
                detected_z_mm=184.737,
                object_short_axis_yaw_deg=20.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(0.0, 10.0, -20.0, 0.0, 30.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 11.0, -21.0, 0.0, 31.0, -70.0),
            joint_feedback_from_degrees(0.0, 12.0, -22.0, 0.0, 32.0, 20.0),
            joint_feedback_from_degrees(0.0, 13.0, -23.0, 0.0, 33.0, 20.0),
            joint_feedback_from_degrees(0.0, 14.0, -24.0, 0.0, 34.0, 20.0),
        ]
        solutions = [
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 11.0, -21.0, 0.0, 31.0, -70.0)),
                solved_pose=EndPose(457.775, 0.0, 253.252, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                solved_fk_pose=EndPose(445.0, 0.0, 323.0, args.rx_deg, args.ry_deg + 3.111, args.rz_deg),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 0.0, 32.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 204.737, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 274.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 13.0, -23.0, 0.0, 33.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 184.737, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 254.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
            PlanarIkSolution(
                target_joints_rad=tuple(math.radians(value) for value in (0.0, 14.0, -24.0, 0.0, 34.0, 20.0)),
                solved_pose=EndPose(457.775, 0.0, 253.252, args.rx_deg, args.ry_deg + 3.111, 20.0),
                solved_fk_pose=EndPose(445.0, 0.0, 323.0, args.rx_deg, args.ry_deg + 3.111, 20.0),
                pitch_delta_deg=3.111,
                requested_pitch_delta_deg=3.111,
                iterations=4,
                x_error_mm=0.0,
                y_error_mm=0.0,
                z_error_mm=0.0,
                position_error_mm=0.0,
                ry_error_deg=0.0,
            ),
        ]

        with mock.patch(
            "robot_control.linux.catch.catch_server_1.solve_planar_ik_j2_j3_j5",
            side_effect=solutions,
        ) as solve_ik, contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["executed"])
        self.assertEqual(solve_ik.call_count, 4)
        results = response["payload"]["planar_ik_results"]
        self.assertEqual(
            [result["step_label"] for result in results],
            ["pick_above", "dynamic_preposition", "pick_grasp", "pick_lift"],
        )
        self.assertTrue(response["payload"]["grasp_yaw_adjustments"][0]["applied"])
        self.assertEqual(
            [result["joint_indices"] for result in results],
            [[2, 3, 5], [2, 3, 5], [2, 3, 5], [2, 3, 5]],
        )
        self.assertEqual(
            [call.kwargs["joint_indices"] for call in solve_ik.call_args_list],
            [(1, 2, 4), (1, 2, 4), (1, 2, 4), (1, 2, 4)],
        )
        self.assertAlmostEqual(results[1]["target_pose"]["rz_deg"], 20.0)
        self.assertAlmostEqual(results[2]["target_pose"]["rz_deg"], 20.0)
        self.assertAlmostEqual(results[3]["target_pose"]["rz_deg"], 20.0)
        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertEqual([call[1][5] for call in joint_calls], [-70000, 20000, 20000, 20000, 20000])

    def test_planar_ik_yaw_rejects_large_j4_or_j6_delta(self):
        current = tuple(math.radians(value) for value in (0.0, 10.0, -20.0, 0.0, 30.0, -70.0))
        safe_target = tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 45.0, 32.0, 20.0))
        validate_planar_ik_joint_delta_limits(
            current,
            safe_target,
            (1, 2, 3, 4, 5),
            label="test",
        )

        with self.assertRaisesRegex(SafetyError, r"J4 delta .*90"):
            validate_planar_ik_joint_delta_limits(
                current,
                tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 91.0, 32.0, 20.0)),
                (1, 2, 3, 4, 5),
                label="test",
            )

        with self.assertRaisesRegex(SafetyError, r"J6 delta .*90"):
            validate_planar_ik_joint_delta_limits(
                current,
                tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 45.0, 32.0, 21.0)),
                (1, 2, 3, 4, 5),
                label="test",
            )

        validate_planar_ik_joint_delta_limits(
            current,
            tuple(math.radians(value) for value in (0.0, 12.0, -22.0, 91.0, 32.0, 21.0)),
            (1, 2, 4),
            label="ordinary-planar-ik",
        )

    def test_dynamic_pitch_clamps_to_joint_limits_before_joint_command(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_pitch_by_x=True,
            force_pick_x_mm=350.0,
            keep_pitch_min_x_mm=300.0,
            keep_pitch_max_x_mm=380.0,
            pitch_deg_per_mm_far=0.125,
            dynamic_pitch_j2_ratio=0.5,
            dynamic_pitch_j3_ratio=-1.0,
            dynamic_pitch_j5_ratio=0.5,
            dynamic_pitch_speed=12,
            command_duration_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.40, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-pitch-limit")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )
        controller.sdk.joint_feedbacks = [
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 69.5, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 69.5, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 69.5, 60.0),
            joint_feedback_from_degrees(10.0, 20.0, -30.0, 40.0, 69.5, 60.0),
        ]

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        joint_calls = [call for call in controller.sdk.calls if call[0] == "JointCtrl"]
        self.assertTrue(joint_calls)
        for _, joints in joint_calls:
            self.assertLessEqual(joints[4], 69900)
        first_adjustment = response["payload"]["joint_pitch_adjustments"][0]
        self.assertTrue(first_adjustment["limit_clamped"])
        self.assertLess(first_adjustment["pitch_delta_deg"], 2.5)
        self.assertAlmostEqual(first_adjustment["j5_delta_deg"], 0.4)

    def test_execute_grasp_only_sends_open_close_only_and_no_release(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-2")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        sdk = controller.sdk
        gripper_calls = [call for call in sdk.calls if call[0] == "GripperCtrl"]
        end_pose_calls = [call for call in sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
            ],
        )
        self.assertEqual(len(end_pose_calls), 4)
        self.assertEqual(
            [step["label"] for step in response["payload"]["steps"]],
            ["ready_open", "pick_above", "pick_grasp", "pick_lift"],
        )
        self.assertFalse(response["payload"]["sequence"]["return_ready"])

    def test_execute_uses_grasp_effort_for_close_only(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            effort_nm=1.0,
            grasp_effort_nm=0.4,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-effort")
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 400, 0x01, 0)),
            ],
        )
        self.assertAlmostEqual(response["payload"]["grasp_effort_nm"], 0.4)

    def test_safe_gripper_stop_on_effort_disables_after_close_threshold(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            grasp_effort_nm=0.4,
            max_gripper_effort_nm=0.8,
            safe_gripper_stop_on_effort=True,
            safe_gripper_monitor_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-safe-gripper",
            )
        )
        controller = make_feedback_controller(
            EndPose(350.0, 0.0, 253.252, -176.794223, 0.889438, -105.490597),
            gripper_efforts_nm=[0.9],
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 400, 0x01, 0)),
                ("GripperCtrl", (0, 0, 0x00, 0)),
            ],
        )
        self.assertTrue(response["payload"]["safe_gripper"]["triggered"])
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "pick_grasp",
                "pick_lift",
            },
        )
        result = response["payload"]["safe_gripper"]["results"][1]
        self.assertTrue(result["triggered"])
        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_action"], "disable")
        self.assertAlmostEqual(result["last_effort_nm"], 0.9)

    def test_startup_ready_uses_continuous_sequence_initial_pose(self):
        args = make_args()
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            count = move_to_start_ready(controller, args)

        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        motion_calls = [call for call in controller.sdk.calls if call[0] == "MotionCtrl_2"]
        self.assertEqual(count, 1)
        self.assertEqual(motion_calls[0], ("MotionCtrl_2", (0x01, 0x00, 10, 0x00, 0, 0x00)))
        self.assertEqual(
            end_pose_calls,
            [
                (
                    "EndPoseCtrl",
                    (350000, 0, 253252, -176794, 889, -105491),
                )
            ],
        )

    def test_startup_ready_speed_is_separate_from_catch_speed(self):
        args = make_args(speed=40, startup_speed=10)
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            move_to_start_ready(controller, args)

        motion_calls = [call for call in controller.sdk.calls if call[0] == "MotionCtrl_2"]
        self.assertEqual(motion_calls[0], ("MotionCtrl_2", (0x01, 0x00, 10, 0x00, 0, 0x00)))

    def test_startup_speed_is_validated(self):
        validate_args(make_args(startup_speed=0))
        with self.assertRaisesRegex(ValueError, "--startup-speed"):
            validate_args(make_args(startup_speed=101))

    def test_keep_pitch_band_is_validated(self):
        validate_args(make_args(keep_pitch_min_x_mm=300.0, keep_pitch_max_x_mm=300.0))
        with self.assertRaisesRegex(ValueError, "--keep-pitch-min-x-mm"):
            validate_args(make_args(keep_pitch_min_x_mm=381.0, keep_pitch_max_x_mm=380.0))

    def test_collision_protection_level_is_validated(self):
        validate_args(make_args(collision_protection_level=8))
        with self.assertRaisesRegex(ValueError, "--collision-protection-level"):
            validate_args(make_args(collision_protection_level=9))

    def test_yolo_metadata_accepts_xywh_fields(self):
        message = {
            "payload": {
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
                "source_label": "cube",
            }
        }

        metadata = yolo_metadata_from_message(message)

        self.assertEqual(metadata["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(metadata["bbox_format"], "xywh")
        self.assertEqual(metadata["label"], "cube")

    def test_yolo_metadata_accepts_segmentation_mask_fields(self):
        message = {
            "payload": {
                "label": "cup",
                "mask_area_px": 1280,
                "mask_centroid_u": 321.5,
                "mask_centroid_v": 123.25,
                "mask_top_u": 320.0,
                "mask_top_v": 80.0,
                "mask_depth_median_m": 0.456,
                "mask_depth_valid_count": 976,
                "detected_top_x_mm": 331.0,
                "detected_top_y_mm": 16.0,
                "detected_top_z_mm": 145.0,
                "object_short_axis_yaw_deg": -35.2,
                "object_yaw_source": "opencv_min_area_rect_short_edge",
                "object_rect_width_px": 123.0,
                "object_rect_height_px": 58.0,
                "object_rect_angle_deg": 12.5,
            }
        }

        metadata = yolo_metadata_from_message(message)

        self.assertEqual(metadata["label"], "cup")
        self.assertEqual(metadata["mask_area_px"], 1280)
        self.assertAlmostEqual(metadata["mask_centroid_u"], 321.5)
        self.assertAlmostEqual(metadata["mask_centroid_v"], 123.25)
        self.assertAlmostEqual(metadata["mask_top_u"], 320.0)
        self.assertAlmostEqual(metadata["mask_top_v"], 80.0)
        self.assertAlmostEqual(metadata["mask_depth_median_m"], 0.456)
        self.assertEqual(metadata["mask_depth_valid_count"], 976)
        self.assertAlmostEqual(metadata["detected_top_x_mm"], 331.0)
        self.assertAlmostEqual(metadata["detected_top_y_mm"], 16.0)
        self.assertAlmostEqual(metadata["detected_top_z_mm"], 145.0)
        self.assertAlmostEqual(metadata["object_short_axis_yaw_deg"], -35.2)
        self.assertEqual(metadata["object_yaw_source"], "opencv_min_area_rect_short_edge")
        self.assertAlmostEqual(metadata["object_rect_width_px"], 123.0)
        self.assertAlmostEqual(metadata["object_rect_height_px"], 58.0)
        self.assertAlmostEqual(metadata["object_rect_angle_deg"], 12.5)

    def test_use_detected_z_rejects_grasp_above_transfer_plane(self):
        args = make_args(use_detected_z=True, detected_z_offset_mm=0.0)
        camera_to_base = np.eye(4, dtype=float)
        command = CameraPointCommand(x_m=0.30, y_m=0.0, z_m=0.30, command_id="cmd")

        with self.assertRaisesRegex(ValueError, "above transfer plane"):
            target_from_command(command, camera_to_base, args)

    def test_validate_args_rejects_negative_delay(self):
        with self.assertRaisesRegex(ValueError, "--pick-delay-s"):
            validate_args(make_args(pick_delay_s=-0.1))


if __name__ == "__main__":
    unittest.main()

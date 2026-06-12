from __future__ import annotations

import argparse
import contextlib
import io
import unittest

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.catch.catch_server_1 import (
    build_parser,
    build_steps,
    dynamic_preposition_pose,
    guarded_step_abort_status,
    handle_message,
    move_to_start_ready,
    target_from_command,
    validate_args,
    yolo_metadata_from_message,
)
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import CameraPointCommand, make_camera_point_message


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "speed": 40,
        "startup_speed": 10,
        "transfer_mode": "P",
        "vertical_mode": "L",
        "command_duration_s": 0.0,
        "command_period_s": 0.02,
        "motion_timeout_s": 0.0,
        "position_tolerance_mm": 8.0,
        "orientation_tolerance_deg": 8.0,
        "no_wait_waypoints": True,
        "print_feedback": False,
        "debug_plan": False,
        "t_robot_s": 1.7,
        "dynamic_preposition": False,
        "dynamic_preposition_clearance_mm": 20.0,
        "final_action_time_s": 0.55,
        "guarded_descent": True,
        "guarded_descent_zone_mm": 30.0,
        "guarded_descent_step_mm": 3.0,
        "guarded_descent_speed": 20,
        "guarded_follow_error_mm": 4.0,
        "guarded_stall_s": 0.2,
        "guarded_start_tolerance_mm": 2.0,
        "guarded_start_timeout_s": 1.0,
        "guarded_progress_min_mm": 1.0,
        "guarded_step_timeout_s": 0.25,
        "guarded_abort_return": "pick_above",
        "collision_protection_level": None,
        "no_start_ready": False,
        "ready_x_mm": 343.108,
        "ready_y_mm": 0.0,
        "transfer_z_mm": 253.252,
        "grasp_z_mm": 133.252,
        "use_detected_z": False,
        "detected_z_offset_mm": 0.0,
        "rx_deg": 180.0,
        "ry_deg": 14.795,
        "rz_deg": 180.0,
        "pick_delay_s": 0.0,
        "place_x_mm": 130.0,
        "place_y_mm": 200.0,
        "place_pick_x_mm": -130.0,
        "place_pick_y_mm": 200.0,
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
        stall_z_mm: float | None = None,
    ) -> None:
        super().__init__()
        self.pose = pose
        self.freeze_pose = freeze_pose
        self.stall_z_mm = stall_z_mm
        self.gripper_feedbacks: list[FakeGripperFeedback] = []

    def EndPoseCtrl(self, X, Y, Z, RX, RY, RZ):
        super().EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        if self.freeze_pose:
            return
        next_z_mm = Z / 1000.0
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
    stall_z_mm: float | None = None,
    gripper_efforts_nm: list[float] | None = None,
) -> PiperController:
    sdk = FeedbackPiperSdk(pose, freeze_pose=freeze_pose, stall_z_mm=stall_z_mm)
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


class TestCatchServer1(unittest.TestCase):
    def test_parser_defaults_enable_guarded_descent_with_overridable_values(self):
        args = build_parser().parse_args([])

        self.assertTrue(args.guarded_descent)
        self.assertAlmostEqual(args.guarded_descent_zone_mm, 30.0)
        self.assertAlmostEqual(args.guarded_descent_step_mm, 3.0)
        self.assertEqual(args.guarded_descent_speed, 20)
        self.assertAlmostEqual(args.guarded_follow_error_mm, 4.0)
        self.assertAlmostEqual(args.guarded_stall_s, 0.2)
        self.assertAlmostEqual(args.guarded_start_tolerance_mm, 2.0)
        self.assertAlmostEqual(args.guarded_start_timeout_s, 1.0)
        self.assertAlmostEqual(args.guarded_progress_min_mm, 1.0)
        self.assertAlmostEqual(args.guarded_step_timeout_s, 0.25)
        self.assertEqual(args.guarded_abort_return, "pick_above")
        self.assertEqual(args.startup_speed, 10)
        self.assertAlmostEqual(args.place_x_mm, 130.0)
        self.assertAlmostEqual(args.place_y_mm, 200.0)
        self.assertAlmostEqual(args.place_pick_x_mm, -130.0)
        self.assertAlmostEqual(args.place_pick_y_mm, 200.0)

        args = build_parser().parse_args(
            [
                "--no-guarded-descent",
                "--guarded-descent-speed",
                "12",
                "--guarded-follow-error-mm",
                "6",
                "--startup-speed",
                "9",
            ]
        )

        self.assertFalse(args.guarded_descent)
        self.assertEqual(args.guarded_descent_speed, 12)
        self.assertAlmostEqual(args.guarded_follow_error_mm, 6.0)
        self.assertEqual(args.startup_speed, 9)

    def test_parser_accepts_descent_protection_alias(self):
        args = build_parser().parse_args(["--no-descent-protection"])
        self.assertFalse(args.guarded_descent)

        args = build_parser().parse_args(
            ["--no-guarded-descent", "--descent-protection"]
        )
        self.assertTrue(args.guarded_descent)

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
            ["P", "P", "L", "L", "P", "P", "P"],
        )
        self.assertAlmostEqual(steps[1].pose.x_mm, 300.0)
        self.assertAlmostEqual(steps[1].pose.y_mm, 20.0)
        self.assertAlmostEqual(steps[1].pose.z_mm, args.transfer_z_mm)
        self.assertAlmostEqual(steps[2].pose.z_mm, args.grasp_z_mm)
        self.assertAlmostEqual(steps[4].pose.x_mm, args.place_x_mm)
        self.assertAlmostEqual(steps[4].pose.y_mm, args.place_y_mm)
        self.assertAlmostEqual(steps[4].pose.z_mm, args.transfer_z_mm)
        self.assertEqual(
            (steps[2].pose.rx_deg, steps[2].pose.ry_deg, steps[2].pose.rz_deg),
            (180.0, 14.795, 180.0),
        )
        self.assertEqual(steps[2].delay_before_s, 1.25)
        self.assertTrue(all(step.delay_before_s == 0.0 for step in steps[3:]))

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
        self.assertEqual(steps[1].delay_before_s, 2.5)
        self.assertAlmostEqual(steps[2].pose.x_mm, 220.0)
        self.assertAlmostEqual(steps[2].pose.y_mm, -40.0)
        self.assertAlmostEqual(steps[2].pose.z_mm, args.grasp_z_mm)
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

    def test_dynamic_preposition_pose_uses_detected_object_height_clearance(self):
        args = make_args(
            dynamic_preposition=True,
            dynamic_preposition_clearance_mm=20.0,
            guarded_descent_zone_mm=30.0,
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
        self.assertEqual(
            response["payload"]["dynamic_execution"]["preposition_enabled"],
            True,
        )

    def test_execute_dynamic_preposition_guarded_descent_success(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            guarded_descent=True,
            guarded_descent_step_mm=5.0,
            guarded_descent_speed=15,
            guarded_follow_error_mm=8.0,
            guarded_stall_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-guard-ok",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0)
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
        self.assertFalse(response["payload"]["protected_abort"])
        self.assertAlmostEqual(
            response["payload"]["dynamic_execution"]["preposition_z_mm"],
            140.0,
        )
        self.assertEqual(
            response["payload"]["dynamic_execution"]["guarded_descent"]["send_count"],
            2,
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
            ],
        )
        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertGreaterEqual(len(end_pose_calls), 6)

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
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0)
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

    def test_execute_dynamic_preposition_guarded_descent_abort_returns_above(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            guarded_descent=True,
            guarded_descent_step_mm=3.0,
            guarded_descent_speed=15,
            guarded_follow_error_mm=2.0,
            guarded_stall_s=0.0,
            guarded_progress_min_mm=0.0,
            guarded_abort_return="pick_above",
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-guard-abort",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0),
            stall_z_mm=140.0,
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
        self.assertTrue(response["payload"]["protected_abort"])
        self.assertIn(
            "height did not decrease",
            response["payload"]["protected_abort_reason"],
        )
        self.assertEqual(
            response["payload"]["dynamic_execution"]["protected_abort_return"],
            "pick_above",
        )
        self.assertEqual(
            response["payload"]["dynamic_execution"]["guarded_descent"]["send_count"],
            1,
        )
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "dynamic_preposition",
                "pick_grasp",
                "protected_abort_return_pick_above",
            },
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (0, 1000, 0x01, 0)),
            ],
        )
        abort_index = controller.sdk.calls.index(
            ("EndPoseCtrl", (350000, -30000, 253252, 180000, 14795, 180000))
        )
        second_open_index = [
            index
            for index, call in enumerate(controller.sdk.calls)
            if call == ("GripperCtrl", (0, 1000, 0x01, 0))
        ][1]
        self.assertLess(abort_index, second_open_index)

    def test_guarded_descent_progress_check_aborts_when_feedback_z_stalls(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            dynamic_preposition=True,
            final_action_time_s=0.0,
            guarded_descent=True,
            guarded_descent_step_mm=3.0,
            guarded_descent_speed=25,
            guarded_follow_error_mm=8.0,
            guarded_stall_s=0.0,
            guarded_progress_min_mm=1.0,
            guarded_abort_return="pick_above",
        )
        camera_to_base = np.eye(4, dtype=float)
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.35,
                y_m=0.02,
                z_m=0.30,
                command_id="cmd-dynamic-progress-abort",
                grasp_mode="dynamic",
                t_arrive_s=0.0,
                conveyor_speed_mm_s=100.0,
                pick_x_mm=350.0,
                pick_y_mm=-30.0,
                detected_z_mm=120.0,
            )
        )
        controller = make_feedback_controller(
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0),
            stall_z_mm=140.0,
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
        self.assertTrue(response["payload"]["protected_abort"])
        self.assertIn(
            "height did not decrease",
            response["payload"]["protected_abort_reason"],
        )
        guarded = response["payload"]["dynamic_execution"]["guarded_descent"]
        self.assertEqual(guarded["send_count"], 1)
        self.assertEqual(guarded["progress_min_mm"], 1.0)
        self.assertLess(guarded["last_progress_mm"], 1.0)
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "dynamic_preposition",
                "pick_grasp",
                "protected_abort_return_pick_above",
            },
        )

    def test_guarded_step_waits_for_follow_error_to_recover_within_timeout(self):
        args = make_args(
            guarded_follow_error_mm=10.0,
            guarded_progress_min_mm=0.0,
            guarded_step_timeout_s=0.2,
            command_period_s=0.005,
        )
        command_pose = EndPose(350.0, -30.0, 154.0, 180.0, 14.795, 168.0)
        controller = make_sequenced_feedback_controller(
            [
                EndPose(350.0, -30.0, 165.478, 180.0, 14.795, 168.0),
                EndPose(350.0, -30.0, 154.0, 180.0, 14.795, 168.0),
            ]
        )

        aborted, reason, feedback_z, follow_error, progress = guarded_step_abort_status(
            controller,
            args,
            command_pose,
            previous_feedback_z_mm=165.478,
        )

        self.assertFalse(aborted)
        self.assertIsNone(reason)
        self.assertAlmostEqual(feedback_z, 154.0)
        self.assertAlmostEqual(follow_error, 0.0)
        self.assertAlmostEqual(progress, 11.478)

    def test_guarded_step_aborts_when_progress_does_not_reach_commanded_height(self):
        args = make_args(
            guarded_follow_error_mm=6.0,
            guarded_progress_min_mm=0.5,
            guarded_step_timeout_s=0.0,
        )
        command_pose = EndPose(350.0, -30.0, 154.0, 180.0, 14.795, 168.0)
        controller = make_sequenced_feedback_controller(
            [
                EndPose(350.0, -30.0, 165.0, 180.0, 14.795, 168.0),
                EndPose(350.0, -30.0, 164.4, 180.0, 14.795, 168.0),
            ]
        )

        aborted, reason, feedback_z, follow_error, progress = guarded_step_abort_status(
            controller,
            args,
            command_pose,
            previous_feedback_z_mm=165.0,
            commanded_drop_mm=3.0,
        )

        self.assertTrue(aborted)
        self.assertIn("did not reach commanded height", reason)
        self.assertAlmostEqual(feedback_z, 164.4)
        self.assertGreater(follow_error, args.guarded_follow_error_mm)
        self.assertGreaterEqual(progress, args.guarded_progress_min_mm)

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

    def test_handle_dynamic_message_skips_when_target_is_too_close(self):
        args = make_args(execute=True, t_robot_s=0.30)
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

    def test_execute_default_runs_out_only(self):
        args = make_args(
            execute=True,
            place_pick_delay_s=0.0,
            guarded_descent=False,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-2")
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

    def test_execute_grasp_only_sends_open_close_only_and_no_release(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            guarded_descent=False,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-2")
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
            guarded_descent=False,
            effort_nm=1.0,
            grasp_effort_nm=0.4,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-effort")
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
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0),
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
        self.assertFalse(response["payload"]["protected_abort"])
        self.assertIsNone(response["payload"]["protected_abort_reason"])
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "pick_grasp_guard_start",
                "pick_grasp",
                "pick_lift",
            },
        )
        result = response["payload"]["safe_gripper"]["results"][1]
        self.assertTrue(result["triggered"])
        self.assertTrue(result["stopped"])
        self.assertEqual(result["stop_action"], "disable")
        self.assertAlmostEqual(result["last_effort_nm"], 0.9)

    def test_execute_static_guarded_descent_success(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            guarded_descent=True,
            guarded_descent_zone_mm=30.0,
            guarded_descent_step_mm=10.0,
            guarded_descent_speed=15,
            guarded_follow_error_mm=8.0,
            guarded_stall_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-static-guard")
        )
        controller = make_feedback_controller(
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0)
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
        self.assertFalse(response["payload"]["protected_abort"])
        self.assertTrue(response["payload"]["guarded_execution"]["guarded_descent_enabled"])
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "pick_grasp_guard_start",
                "pick_grasp",
                "pick_lift",
            },
        )
        self.assertEqual(
            response["payload"]["guarded_execution"]["guarded_descent"]["start_z_mm"],
            args.grasp_z_mm + 30.0,
        )
        self.assertEqual(
            response["payload"]["guarded_execution"]["guarded_descent"]["send_count"],
            3,
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (-100000, 1000, 0x01, 0)),
            ],
        )

    def test_guarded_descent_waits_for_start_feedback_before_first_step(self):
        args = make_args(
            execute=True,
            grasp_only=True,
            no_return_ready=True,
            guarded_descent=True,
            guarded_descent_zone_mm=30.0,
            guarded_descent_step_mm=3.0,
            guarded_start_tolerance_mm=2.0,
            guarded_start_timeout_s=0.0,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.30, command_id="cmd-guard-start")
        )
        controller = make_feedback_controller(
            EndPose(250.0, 0.0, 180.0, 180.0, 14.795, 180.0),
            freeze_pose=True,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                args=args,
                camera_to_base=camera_to_base,
                validator=controller,
                controller=controller,
            )

        self.assertTrue(response["payload"]["protected_abort"])
        self.assertIn(
            "guarded descent start not reached",
            response["payload"]["protected_abort_reason"],
        )
        self.assertEqual(
            response["payload"]["guarded_execution"]["guarded_descent"]["send_count"],
            0,
        )
        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertNotIn(
            ("EndPoseCtrl", (250000, 0, 160252, 180000, 14795, 180000)),
            end_pose_calls,
        )

    def test_execute_static_guarded_descent_abort_skips_lift_and_place(self):
        args = make_args(
            execute=True,
            grasp_only=False,
            guarded_descent=True,
            guarded_descent_zone_mm=30.0,
            guarded_descent_step_mm=3.0,
            guarded_descent_speed=15,
            guarded_follow_error_mm=2.0,
            guarded_stall_s=0.0,
            guarded_progress_min_mm=0.0,
            guarded_abort_return="pick_above",
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.25, 0.00, 0.00])
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.0,
                y_m=0.0,
                z_m=0.30,
                command_id="cmd-static-guard-abort",
            )
        )
        controller = make_feedback_controller(
            EndPose(343.108, 0.0, 253.252, 180.0, 14.795, 180.0),
            stall_z_mm=args.grasp_z_mm + 30.0,
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
        self.assertTrue(response["payload"]["protected_abort"])
        self.assertIn(
            "height did not decrease",
            response["payload"]["protected_abort_reason"],
        )
        self.assertEqual(
            response["payload"]["guarded_execution"]["protected_abort_return"],
            "pick_above",
        )
        self.assertEqual(
            set(response["payload"]["send_counts"].keys()),
            {
                "ready_open",
                "pick_above",
                "pick_grasp_guard_start",
                "pick_grasp",
                "protected_abort_return_pick_above",
            },
        )
        gripper_calls = [call for call in controller.sdk.calls if call[0] == "GripperCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 1000, 0x01, 0)),
                ("GripperCtrl", (0, 1000, 0x01, 0)),
            ],
        )
        abort_index = controller.sdk.calls.index(
            ("EndPoseCtrl", (250000, 0, 253252, 180000, 14795, 180000))
        )
        second_open_index = [
            index
            for index, call in enumerate(controller.sdk.calls)
            if call == ("GripperCtrl", (0, 1000, 0x01, 0))
        ][1]
        self.assertLess(abort_index, second_open_index)

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
                    (343108, 0, 253252, 180000, 14795, 180000),
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

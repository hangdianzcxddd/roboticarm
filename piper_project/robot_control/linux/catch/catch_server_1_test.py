from __future__ import annotations

import argparse
import contextlib
import io
import unittest

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.catch.catch_server_1 import (
    build_steps,
    handle_message,
    move_to_start_ready,
    target_from_command,
    validate_args,
    yolo_metadata_from_message,
)
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.shared.protocol import CameraPointCommand, make_camera_point_message


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "speed": 40,
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
        "place_x_mm": 100.108,
        "place_y_mm": 200.0,
        "place_pick_x_mm": None,
        "place_pick_y_mm": None,
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


class TestCatchServer1(unittest.TestCase):
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
        self.assertAlmostEqual(response["payload"]["dynamic_timing"]["t_wait_s"], 0.50)

    def test_execute_default_runs_out_only(self):
        args = make_args(execute=True, place_pick_delay_s=0.0)
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
        args = make_args(execute=True, grasp_only=True, no_return_ready=True)
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

    def test_startup_ready_uses_continuous_sequence_initial_pose(self):
        args = make_args()
        controller = make_controller()

        with contextlib.redirect_stdout(io.StringIO()):
            count = move_to_start_ready(controller, args)

        end_pose_calls = [call for call in controller.sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(count, 1)
        self.assertEqual(
            end_pose_calls,
            [
                (
                    "EndPoseCtrl",
                    (343108, 0, 253252, 180000, 14795, 180000),
                )
            ],
        )

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

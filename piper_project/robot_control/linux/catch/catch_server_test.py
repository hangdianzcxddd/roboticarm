from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_control.linux.catch.catch_server import handle_message
from robot_control.linux.catch.hand_eye_mapping import load_camera_to_base_transform
from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.catch.tcp_offset import (
    build_catch_plan,
    j6_pose_for_tcp_pose,
    tcp_pose_for_j6_pose,
)
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import CameraPointCommand, make_camera_point_message
from robot_control.shared.simulated_mapping import SimulatedMappingConfig


def make_args(**overrides: object) -> argparse.Namespace:
    values = {
        "approach_height_mm": 40.0,
        "lift_height_mm": 60.0,
        "target_z_offset_mm": -5.0,
        "rx_deg": 0.0,
        "ry_deg": 85.0,
        "rz_deg": 0.0,
        "speed": 10,
        "approach_mode": "P",
        "grasp_mode": "L",
        "lift_mode": "L",
        "command_duration_s": 0.0,
        "command_period_s": 0.02,
        "print_feedback": False,
        "clear_gripper_error": False,
        "no_pre_open": False,
        "no_gripper": False,
        "open_width_mm": 70.0,
        "close_width_mm": 0.0,
        "open_command_mm": 0.0,
        "grasp_command_mm": -100.0,
        "release_command_mm": None,
        "effort_nm": 1.0,
        "grasp_effort_nm": None,
        "release_effort_nm": None,
        "gripper_settle_s": 0.0,
        "place_x_mm": 0.0,
        "place_y_mm": 150.0,
        "place_z_mm": 150.0,
        "place_rx_deg": None,
        "place_ry_deg": None,
        "place_rz_deg": None,
        "place_approach_height_mm": 50.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestTcpOffset(unittest.TestCase):
    def test_j6_pose_for_tcp_pose_subtracts_offset_in_j6_frame(self):
        tcp_pose = EndPose(200.0, 10.0, 120.0, 0.0, 90.0, 0.0)

        j6_pose = j6_pose_for_tcp_pose(tcp_pose, (0.0, 0.0, 145.0))

        self.assertAlmostEqual(j6_pose.x_mm, 55.0, places=9)
        self.assertAlmostEqual(j6_pose.y_mm, 10.0, places=9)
        self.assertAlmostEqual(j6_pose.z_mm, 120.0, places=9)
        self.assertEqual((j6_pose.rx_deg, j6_pose.ry_deg, j6_pose.rz_deg), (0.0, 90.0, 0.0))

    def test_tcp_pose_round_trip(self):
        tcp_pose = EndPose(180.0, -20.0, 220.0, 10.0, 80.0, -15.0)

        j6_pose = j6_pose_for_tcp_pose(tcp_pose, (0.0, 0.0, 145.0))
        round_trip = tcp_pose_for_j6_pose(j6_pose, (0.0, 0.0, 145.0))

        np.testing.assert_allclose(
            [
                round_trip.x_mm,
                round_trip.y_mm,
                round_trip.z_mm,
                round_trip.rx_deg,
                round_trip.ry_deg,
                round_trip.rz_deg,
            ],
            [
                tcp_pose.x_mm,
                tcp_pose.y_mm,
                tcp_pose.z_mm,
                tcp_pose.rx_deg,
                tcp_pose.ry_deg,
                tcp_pose.rz_deg,
            ],
            atol=1e-9,
        )

    def test_build_catch_plan_offsets_tcp_waypoints(self):
        target = EndPose(220.0, 0.0, 200.0, 0.0, 0.0, 0.0)

        plan = build_catch_plan(
            target,
            (0.0, 0.0, 145.0),
            approach_height_mm=40.0,
            lift_height_mm=60.0,
            target_z_offset_mm=-5.0,
        )

        self.assertEqual(plan.grasp_tcp_pose.z_mm, 195.0)
        self.assertEqual(plan.approach_tcp_pose.z_mm, 235.0)
        self.assertEqual(plan.lift_tcp_pose.z_mm, 255.0)
        self.assertEqual(plan.grasp_j6_pose.z_mm, 50.0)


class TestCatchServer(unittest.TestCase):
    def test_handle_message_dry_run_returns_tcp_and_j6_targets(self):
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.1, y_m=-0.1, z_m=0.3, command_id="cmd-1")
        )
        mapping = SimulatedMappingConfig(
            base_x_mm=300.0,
            base_y_mm=0.0,
            base_z_mm=220.0,
            scale_x=100.0,
            scale_y=100.0,
            scale_z=100.0,
            max_delta_x_mm=20.0,
            max_delta_y_mm=20.0,
            max_delta_z_mm=20.0,
            rx_deg=0.0,
            ry_deg=90.0,
            rz_deg=0.0,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                mapping,
                tcp_offset_j6_mm=(0.0, 0.0, 145.0),
                execute=False,
                controller=None,
                args=make_args(target_z_offset_mm=0.0),
            )

        self.assertEqual(response["type"], "ack")
        self.assertFalse(response["payload"]["executed"])
        self.assertEqual(response["payload"]["target_tcp_pose"]["x_mm"], 310.0)
        self.assertEqual(response["payload"]["target_tcp_pose"]["y_mm"], -10.0)
        self.assertAlmostEqual(response["payload"]["grasp_j6_pose"]["x_mm"], 165.0)
        self.assertEqual(response["payload"]["catch"]["gripper_enabled"], True)
        self.assertEqual(response["payload"]["place_tcp_pose"]["y_mm"], 150.0)
        self.assertEqual(response["payload"]["place_tcp_pose"]["z_mm"], 150.0)
        self.assertEqual(response["payload"]["catch"]["open_command_mm"], 0.0)
        self.assertEqual(response["payload"]["catch"]["grasp_command_mm"], -100.0)

    def test_handle_message_can_use_hand_eye_camera_to_base_mapping(self):
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.10, y_m=-0.05, z_m=0.30, command_id="cmd-he")
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.20, 0.10, 0.00])

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                SimulatedMappingConfig(),
                tcp_offset_j6_mm=(0.0, 0.0, 145.0),
                execute=False,
                controller=None,
                args=make_args(
                    target_z_offset_mm=0.0,
                    rx_deg=0.0,
                    ry_deg=90.0,
                    rz_deg=0.0,
                ),
                hand_eye_camera_to_base=camera_to_base,
            )

        self.assertEqual(response["payload"]["mapping_source"], "hand_eye")
        self.assertAlmostEqual(response["payload"]["target_tcp_pose"]["x_mm"], 300.0)
        self.assertAlmostEqual(response["payload"]["target_tcp_pose"]["y_mm"], 50.0)
        self.assertAlmostEqual(response["payload"]["target_tcp_pose"]["z_mm"], 300.0)
        self.assertAlmostEqual(response["payload"]["grasp_j6_pose"]["x_mm"], 155.0)

    def test_load_camera_to_base_transform_reads_hand_eye_result(self):
        document = {
            "hand_eye_calibration": {
                "mode": "eye_to_hand",
                "transform": {
                    "from_frame": "camera",
                    "to_frame": "base",
                    "rotation_matrix": np.eye(3).tolist(),
                    "translation_m": [0.1, 0.2, 0.3],
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hand_eye_result.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            transform = load_camera_to_base_transform(path)

        np.testing.assert_allclose(transform[:3, :3], np.eye(3))
        np.testing.assert_allclose(transform[:3, 3], [0.1, 0.2, 0.3])

    def test_execute_uses_open_zero_gripper_sequence_and_place_motion(self):
        message = make_camera_point_message(
            CameraPointCommand(x_m=0.0, y_m=0.0, z_m=0.3, command_id="cmd-2")
        )
        mapping = SimulatedMappingConfig(
            base_x_mm=250.0,
            base_y_mm=0.0,
            base_z_mm=250.0,
            rx_deg=0.0,
            ry_deg=85.0,
            rz_deg=0.0,
        )
        sdk = FakePiperSdk()
        controller = PiperController(
            config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
            sdk=sdk,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = handle_message(
                message,
                mapping,
                tcp_offset_j6_mm=(0.0, 0.0, 145.0),
                execute=True,
                controller=controller,
                args=make_args(
                    command_duration_s=0.0,
                    open_command_mm=0.0,
                    grasp_command_mm=-100.0,
                    release_command_mm=0.0,
                    effort_nm=1.0,
                    grasp_effort_nm=1.5,
                    release_effort_nm=0.4,
                    place_x_mm=0.0,
                    place_y_mm=150.0,
                    place_z_mm=150.0,
                ),
            )

        gripper_calls = [call for call in sdk.calls if call[0] == "GripperCtrl"]
        end_pose_calls = [call for call in sdk.calls if call[0] == "EndPoseCtrl"]
        self.assertEqual(
            gripper_calls,
            [
                ("GripperCtrl", (0, 400, 0x01, 0)),
                ("GripperCtrl", (-100000, 1500, 0x01, 0)),
                ("GripperCtrl", (0, 400, 0x01, 0)),
            ],
        )
        self.assertEqual(len(end_pose_calls), 6)
        self.assertEqual(
            sorted(response["payload"]["catch"]["send_counts"]),
            ["approach", "grasp", "lift", "place", "place_approach", "place_retreat"],
        )


if __name__ == "__main__":
    unittest.main()

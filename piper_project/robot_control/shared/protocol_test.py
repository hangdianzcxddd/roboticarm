from __future__ import annotations

import unittest

from robot_control.shared.geometry import TOP_DOWN_TCP_ORIENTATION_DEG
from robot_control.shared.simulated_mapping import camera_xyz_to_simulated_robot_pose
from robot_control.shared.protocol import (
    CameraPointCommand,
    camera_point_from_message,
    decode_message,
    encode_message,
    make_camera_point_message,
)


class TestTcpProtocol(unittest.TestCase):
    def test_camera_point_message_round_trip(self):
        command = CameraPointCommand(
            x_m=0.10,
            y_m=-0.05,
            z_m=0.40,
            u=320,
            v=240,
            depth_m=0.40,
            source="unit_test",
            command_id="cmd-1",
            task="in",
            rank=2,
            label="cup",
            confidence=0.82,
            class_id=41,
            bbox_xywh=(10.0, 20.0, 30.0, 40.0),
            grasp_mode="dynamic",
            t_arrive_s=0.75,
            conveyor_speed_mm_s=100.0,
            pick_x_mm=350.0,
            pick_y_mm=-30.0,
            detected_x_mm=330.0,
            detected_y_mm=15.0,
            detected_z_mm=120.0,
            detected_top_x_mm=331.0,
            detected_top_y_mm=16.0,
            detected_top_z_mm=145.0,
            detected_top_camera_x_m=0.012,
            detected_top_camera_y_m=-0.034,
            detected_top_camera_z_m=0.456,
            mask_area_px=1280,
            mask_centroid_u=321.5,
            mask_centroid_v=123.25,
            mask_top_u=320.0,
            mask_top_v=80.0,
            mask_depth_median_m=0.456,
            mask_depth_valid_count=976,
            object_short_axis_yaw_deg=-35.2,
            object_yaw_source="opencv_min_area_rect_short_edge",
            object_rect_width_px=123.0,
            object_rect_height_px=58.0,
            object_rect_angle_deg=12.5,
        )

        decoded = decode_message(encode_message(make_camera_point_message(command)))
        parsed = camera_point_from_message(decoded)

        self.assertEqual(parsed.command_id, "cmd-1")
        self.assertAlmostEqual(parsed.x_m, 0.10)
        self.assertAlmostEqual(parsed.y_m, -0.05)
        self.assertAlmostEqual(parsed.z_m, 0.40)
        self.assertEqual((parsed.u, parsed.v), (320, 240))
        self.assertEqual(parsed.source, "unit_test")
        self.assertEqual(parsed.task, "in")
        self.assertEqual(parsed.rank, 2)
        self.assertEqual(parsed.label, "cup")
        self.assertAlmostEqual(parsed.confidence, 0.82)
        self.assertEqual(parsed.class_id, 41)
        self.assertEqual(parsed.bbox_xywh, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(parsed.grasp_mode, "dynamic")
        self.assertAlmostEqual(parsed.t_arrive_s, 0.75)
        self.assertAlmostEqual(parsed.conveyor_speed_mm_s, 100.0)
        self.assertAlmostEqual(parsed.pick_x_mm, 350.0)
        self.assertAlmostEqual(parsed.pick_y_mm, -30.0)
        self.assertAlmostEqual(parsed.detected_x_mm, 330.0)
        self.assertAlmostEqual(parsed.detected_y_mm, 15.0)
        self.assertAlmostEqual(parsed.detected_z_mm, 120.0)
        self.assertAlmostEqual(parsed.detected_top_x_mm, 331.0)
        self.assertAlmostEqual(parsed.detected_top_y_mm, 16.0)
        self.assertAlmostEqual(parsed.detected_top_z_mm, 145.0)
        self.assertAlmostEqual(parsed.detected_top_camera_x_m, 0.012)
        self.assertAlmostEqual(parsed.detected_top_camera_y_m, -0.034)
        self.assertAlmostEqual(parsed.detected_top_camera_z_m, 0.456)
        self.assertEqual(parsed.mask_area_px, 1280)
        self.assertAlmostEqual(parsed.mask_centroid_u, 321.5)
        self.assertAlmostEqual(parsed.mask_centroid_v, 123.25)
        self.assertAlmostEqual(parsed.mask_top_u, 320.0)
        self.assertAlmostEqual(parsed.mask_top_v, 80.0)
        self.assertAlmostEqual(parsed.mask_depth_median_m, 0.456)
        self.assertEqual(parsed.mask_depth_valid_count, 976)
        self.assertAlmostEqual(parsed.object_short_axis_yaw_deg, -35.2)
        self.assertEqual(parsed.object_yaw_source, "opencv_min_area_rect_short_edge")
        self.assertAlmostEqual(parsed.object_rect_width_px, 123.0)
        self.assertAlmostEqual(parsed.object_rect_height_px, 58.0)
        self.assertAlmostEqual(parsed.object_rect_angle_deg, 12.5)

    def test_camera_point_task_defaults_to_out(self):
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.40,
                command_id="cmd-1",
            )
        )
        del message["payload"]["task"]

        parsed = camera_point_from_message(message)

        self.assertEqual(parsed.task, "out")

    def test_camera_point_accepts_probe_grasp_mode(self):
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.40,
                command_id="cmd-probe",
                grasp_mode="probe",
            )
        )

        parsed = camera_point_from_message(message)

        self.assertEqual(parsed.grasp_mode, "probe")

    def test_camera_point_rejects_unknown_task(self):
        message = make_camera_point_message(
            CameraPointCommand(
                x_m=0.10,
                y_m=-0.05,
                z_m=0.40,
                command_id="cmd-1",
            )
        )
        message["payload"]["task"] = "sideways"

        with self.assertRaisesRegex(ValueError, "task"):
            camera_point_from_message(message)

    def test_simulated_mapping_is_bounded(self):
        pose = camera_xyz_to_simulated_robot_pose(10.0, -10.0, 10.0)

        self.assertEqual(pose.x_mm, 190.0)
        self.assertEqual(pose.y_mm, -40.0)
        self.assertEqual(pose.z_mm, 250.0)
        self.assertEqual(
            (pose.rx_deg, pose.ry_deg, pose.rz_deg),
            TOP_DOWN_TCP_ORIENTATION_DEG,
        )


if __name__ == "__main__":
    unittest.main()

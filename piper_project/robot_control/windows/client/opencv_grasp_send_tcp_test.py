from __future__ import annotations

from argparse import Namespace
import unittest

from robot_control.windows.client.opencv_grasp_send_tcp import (
    DetectionStability,
    OpenCvGraspSender,
    SelectedTarget,
    camera_point_command_from_target,
    can_auto_send,
    select_detection,
    validate_args,
)
from robot_control.windows.vision.depth import CameraPoint
from robot_control.windows.vision.detector import Detection


def detection(u: int, v: int, area: float) -> Detection:
    return Detection(
        u=u,
        v=v,
        x=u - 5,
        y=v - 5,
        width=10,
        height=10,
        area=area,
    )


class TestOpenCvGraspSendTcp(unittest.TestCase):
    def test_select_detection_uses_largest_area(self):
        detections = [
            detection(20, 30, 100.0),
            detection(120, 70, 500.0),
            detection(80, 40, 300.0),
        ]

        selected = select_detection(detections, (100, 160, 3), "largest")

        self.assertEqual(selected, detections[1])

    def test_select_detection_can_choose_nearest_image_center(self):
        detections = [
            detection(10, 10, 1000.0),
            detection(82, 52, 100.0),
        ]

        selected = select_detection(detections, (100, 160, 3), "center")

        self.assertEqual(selected, detections[1])

    def test_select_detection_left_and_right_policies(self):
        detections = [
            detection(20, 40, 100.0),
            detection(140, 40, 100.0),
            detection(80, 40, 100.0),
        ]

        self.assertEqual(
            select_detection(detections, (100, 160, 3), "left-to-right"),
            detections[0],
        )
        self.assertEqual(
            select_detection(detections, (100, 160, 3), "right-to-left"),
            detections[1],
        )

    def test_detection_stability_requires_consecutive_near_centers(self):
        stability = DetectionStability(required_frames=3, max_center_shift_px=5.0)

        self.assertFalse(stability.update(detection(50, 50, 100.0)))
        self.assertFalse(stability.update(detection(52, 51, 100.0)))
        self.assertTrue(stability.update(detection(53, 49, 100.0)))
        self.assertFalse(stability.update(detection(80, 80, 100.0)))

    def test_can_auto_send_respects_stability_limit_and_cooldown(self):
        self.assertFalse(
            can_auto_send(
                auto_send=True,
                is_stable=False,
                sent_count=0,
                max_sends=1,
                last_send_time=None,
                cooldown_s=10.0,
                now=100.0,
            )
        )
        self.assertTrue(
            can_auto_send(
                auto_send=True,
                is_stable=True,
                sent_count=0,
                max_sends=1,
                last_send_time=None,
                cooldown_s=10.0,
                now=100.0,
            )
        )
        self.assertFalse(
            can_auto_send(
                auto_send=True,
                is_stable=True,
                sent_count=1,
                max_sends=1,
                last_send_time=100.0,
                cooldown_s=10.0,
                now=120.0,
            )
        )
        self.assertFalse(
            can_auto_send(
                auto_send=True,
                is_stable=True,
                sent_count=0,
                max_sends=0,
                last_send_time=100.0,
                cooldown_s=10.0,
                now=105.0,
            )
        )
        self.assertTrue(
            can_auto_send(
                auto_send=True,
                is_stable=True,
                sent_count=0,
                max_sends=0,
                last_send_time=100.0,
                cooldown_s=10.0,
                now=111.0,
            )
        )

    def test_target_converts_to_same_camera_point_command_protocol_as_click_client(self):
        target = SelectedTarget(
            detection=detection(101, 202, 300.0),
            point=CameraPoint(
                u=101,
                v=202,
                depth_m=0.55,
                x_m=0.01,
                y_m=-0.02,
                z_m=0.55,
            ),
        )

        command = camera_point_command_from_target(target, source="opencv_background")

        self.assertEqual(command.u, 101)
        self.assertEqual(command.v, 202)
        self.assertEqual(command.depth_m, 0.55)
        self.assertEqual(command.x_m, 0.01)
        self.assertEqual(command.y_m, -0.02)
        self.assertEqual(command.z_m, 0.55)
        self.assertEqual(command.source, "opencv_background")

    def test_sender_dry_run_returns_payload_without_network(self):
        target = SelectedTarget(
            detection=detection(10, 20, 300.0),
            point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        sender = OpenCvGraspSender(
            host=None,
            port=5005,
            timeout_s=1.0,
            dry_run=True,
        )

        response = sender.send_target(target, source="opencv_test")

        self.assertEqual(response["type"], "dry_run")
        self.assertEqual(response["payload"]["u"], 10)
        self.assertEqual(response["payload"]["x_m"], 0.1)

    def test_no_display_requires_auto_send_exit_after_send(self):
        args = Namespace(
            dry_run=True,
            host=None,
            timeout_s=1.0,
            sample_radius=0,
            min_area=0.0,
            max_area=0.0,
            max_results=1,
            stable_frames=1,
            max_center_shift_px=0.0,
            send_cooldown_s=0.0,
            max_sends=1,
            no_display=True,
            auto_send=False,
            exit_after_send=False,
        )

        with self.assertRaisesRegex(ValueError, "--no-display requires"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()

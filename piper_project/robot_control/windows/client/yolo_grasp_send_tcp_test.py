from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from robot_control.windows.client.yolo_grasp_send_tcp import (
    RankedTarget,
    YoloGraspSender,
    apply_base_z_offset,
    camera_point_command_from_target,
    detections_to_ranked_targets,
    grasp_pixel_from_detection,
    offset_camera_point,
    validate_args,
)
from robot_control.windows.vision.depth import CameraPoint
from robot_control.windows.vision.detector import Detection


def detection(
    *,
    u: int = 50,
    v: int = 60,
    x: int = 20,
    y: int = 30,
    width: int = 60,
    height: int = 80,
    label: str = "target",
    confidence: float = 0.5,
) -> Detection:
    return Detection(
        u=u,
        v=v,
        x=x,
        y=y,
        width=width,
        height=height,
        area=float(width * height),
        label=label,
        confidence=confidence,
    )


def camera_point(u: int, v: int, depth_m: float) -> CameraPoint:
    return CameraPoint(
        u=u,
        v=v,
        depth_m=depth_m,
        x_m=0.01,
        y_m=0.02,
        z_m=depth_m,
    )


class TestYoloGraspSendTcp(unittest.TestCase):
    def test_grasp_pixel_moves_up_from_bbox_center(self):
        selected = grasp_pixel_from_detection(
            detection(x=10, y=20, width=100, height=80),
            bbox_up_ratio=0.25,
            bbox_up_px=5.0,
        )

        self.assertEqual(selected, (60, 35))

    def test_offset_camera_point_moves_in_image_up_direction(self):
        point = camera_point(10, 20, 0.50)

        shifted = offset_camera_point(
            point,
            camera_up_offset_mm=20.0,
            point_z_offset_mm=5.0,
        )

        self.assertEqual((shifted.u, shifted.v), (10, 20))
        self.assertAlmostEqual(shifted.depth_m, 0.50)
        self.assertAlmostEqual(shifted.x_m, 0.01)
        self.assertAlmostEqual(shifted.y_m, 0.00)
        self.assertAlmostEqual(shifted.z_m, 0.505)

    def test_apply_base_z_offset_adjusts_camera_xyz_through_hand_eye_transform(self):
        point = camera_point(10, 20, 0.50)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.10, -0.20, 0.30])

        shifted = apply_base_z_offset(
            point,
            camera_to_base=camera_to_base,
            base_z_offset_mm=20.0,
        )

        self.assertAlmostEqual(shifted.x_m, point.x_m)
        self.assertAlmostEqual(shifted.y_m, point.y_m)
        self.assertAlmostEqual(shifted.z_m, point.z_m + 0.02)

    def test_detections_are_ranked_nearest_depth_first(self):
        detections = [
            detection(x=-20, width=60, label="far"),
            detection(x=-10, width=60, label="near"),
            detection(x=0, width=60, label="middle"),
        ]
        depths_by_u = {10: 0.30, 20: 0.10, 30: 0.20}

        def fake_deproject(depth_frame, u, v, sample_radius, intrinsics):
            del depth_frame, v, sample_radius, intrinsics
            return camera_point(u, 99, depths_by_u[u])

        with patch(
            "robot_control.windows.client.yolo_grasp_send_tcp.deproject_pixel_to_camera_xyz",
            side_effect=fake_deproject,
        ):
            targets, errors = detections_to_ranked_targets(
                detections,
                depth_frame=object(),
                sample_radius=3,
                intrinsics=None,
                bbox_up_ratio=0.0,
                bbox_up_px=0.0,
                camera_up_offset_mm=0.0,
                point_z_offset_mm=0.0,
                base_z_offset_mm=0.0,
            )

        self.assertEqual(errors, {})
        self.assertEqual([target.rank for target in targets], [1, 2, 3])
        self.assertEqual([target.detection.label for target in targets], ["near", "middle", "far"])
        self.assertEqual([target.depth_mm for target in targets], [100.0, 200.0, 300.0])

    def test_base_z_offset_requires_hand_eye_transform(self):
        class Args:
            dry_run = True
            host = None
            timeout_s = 1.0
            sample_radius = 0
            max_results = 1
            bbox_up_ratio = 0.0
            bbox_up_px = 0.0
            print_interval = 1.0
            base_z_offset_mm = 20.0
            hand_eye_result = None

        with self.assertRaisesRegex(ValueError, "--base-z-offset-mm requires"):
            validate_args(Args())

    def test_target_converts_to_click_compatible_camera_point_command(self):
        target = RankedTarget(
            rank=2,
            detection=detection(label="bottle"),
            pixel=(11, 22),
            raw_point=camera_point(11, 22, 0.55),
            send_point=CameraPoint(
                u=11,
                v=22,
                depth_m=0.55,
                x_m=0.10,
                y_m=-0.02,
                z_m=0.55,
            ),
        )

        command = camera_point_command_from_target(target, source="yolo_test")

        self.assertEqual(command.u, 11)
        self.assertEqual(command.v, 22)
        self.assertEqual(command.depth_m, 0.55)
        self.assertEqual(command.x_m, 0.10)
        self.assertEqual(command.y_m, -0.02)
        self.assertEqual(command.z_m, 0.55)
        self.assertEqual(command.source, "yolo_test")

    def test_sender_dry_run_returns_payload_without_network(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        sender = YoloGraspSender(
            host=None,
            port=5005,
            timeout_s=1.0,
            dry_run=True,
        )

        response = sender.send_target(target, source="yolo_test")

        self.assertEqual(response["type"], "dry_run")
        self.assertEqual(response["payload"]["rank"], 1)
        self.assertEqual(response["payload"]["label"], "cup")
        self.assertEqual(response["payload"]["u"], 10)
        self.assertEqual(response["payload"]["x_m"], 0.1)


if __name__ == "__main__":
    unittest.main()

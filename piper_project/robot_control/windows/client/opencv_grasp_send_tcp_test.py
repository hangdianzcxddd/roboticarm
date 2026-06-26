from __future__ import annotations

import unittest

import cv2
import numpy as np

from robot_control.windows.client.opencv_grasp_send_tcp import (
    GreenConveyorDetectorConfig,
    GreenConveyorOpenCVDetector,
    OpenCVGraspSender,
    attach_object_short_axis_yaw,
    build_parser,
    hsv_in_range,
    validate_args,
)
from robot_control.windows.client.yolo_seg_grasp_send_tcp import segments_to_ranked_targets
from robot_control.windows.vision.d435i_intrinsics import PinholeIntrinsics


class FakeDepthFrame:
    def __init__(self, depth_m: np.ndarray) -> None:
        self.depth_m = np.asarray(depth_m, dtype=np.float32)

    def get_width(self) -> int:
        return int(self.depth_m.shape[1])

    def get_height(self) -> int:
        return int(self.depth_m.shape[0])

    def get_data(self):
        return np.round(self.depth_m / 0.001).astype(np.uint16)

    def get_units(self) -> float:
        return 0.001

    def get_distance(self, u: int, v: int) -> float:
        return float(self.depth_m[v, u])


def green_frame(width: int = 160, height: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (0, 180, 0)
    return frame


class TestOpenCVGraspSendTcp(unittest.TestCase):
    def test_detector_finds_non_green_object_on_green_belt(self):
        frame = green_frame()
        frame[40:80, 60:100] = (0, 0, 255)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=100,
                morphology_kernel_size=3,
            )
        )

        segments = detector.detect(frame)

        self.assertEqual(len(segments), 1)
        detection = segments[0].detection
        self.assertEqual((detection.x, detection.y, detection.width, detection.height), (60, 40, 40, 40))
        self.assertAlmostEqual(segments[0].centroid_u, 79.5, places=1)
        self.assertAlmostEqual(segments[0].centroid_v, 59.5, places=1)

    def test_detector_restricts_to_roi_and_can_reject_border_touching_components(self):
        frame = green_frame()
        frame[25:45, 25:45] = (0, 0, 255)
        frame[70:90, 120:140] = (255, 0, 0)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=50,
                morphology_kernel_size=1,
                roi_xywh=(20, 20, 40, 40),
                reject_roi_border_touching=True,
            )
        )

        segments = detector.detect(frame)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].detection.x, segments[0].detection.y), (25, 25))

    def test_hsv_range_supports_hue_wraparound(self):
        hsv = np.array([[[178, 200, 200], [5, 200, 200], [90, 200, 200]]], dtype=np.uint8)

        mask = hsv_in_range(hsv, (170, 100, 100), (10, 255, 255))

        self.assertEqual(mask[0, 0], 255)
        self.assertEqual(mask[0, 1], 255)
        self.assertEqual(mask[0, 2], 0)

    def test_segments_from_detector_convert_to_payload_compatible_target(self):
        frame = green_frame(20, 20)
        frame[5:10, 6:12] = (0, 0, 255)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=10,
                morphology_kernel_size=1,
            )
        )
        segments = detector.detect(frame)
        depth = np.full((20, 20), 0.45, dtype=np.float32)

        targets, errors = segments_to_ranked_targets(
            segments,
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=20,
                height=20,
                fx=100.0,
                fy=100.0,
                ppx=10.0,
                ppy=10.0,
            ),
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            camera_up_offset_mm=0.0,
            point_z_offset_mm=0.0,
            base_z_offset_mm=0.0,
        )
        sender = OpenCVGraspSender(host=None, port=5005, timeout_s=1.0, dry_run=True)

        response = sender.send_target(targets[0], source="opencv_test", task="in")

        self.assertEqual(errors, {})
        payload = response["payload"]
        self.assertEqual(payload["source"], "opencv_test")
        self.assertEqual(payload["task"], "in")
        self.assertEqual(payload["label"], "opencv_object")
        self.assertGreater(payload["mask_area_px"], 0)
        self.assertAlmostEqual(payload["mask_depth_median_m"], 0.45, places=3)
        self.assertAlmostEqual(payload["mask_top_u"], 8.5)
        self.assertAlmostEqual(payload["mask_top_v"], 5.0)

    def test_segments_include_top_base_z_when_hand_eye_is_available(self):
        frame = green_frame(20, 20)
        frame[5:10, 6:12] = (0, 0, 255)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=10,
                morphology_kernel_size=1,
            )
        )
        segments = detector.detect(frame)
        depth = np.full((20, 20), 0.45, dtype=np.float32)

        targets, errors = segments_to_ranked_targets(
            segments,
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=20,
                height=20,
                fx=100.0,
                fy=100.0,
                ppx=10.0,
                ppy=10.0,
            ),
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            camera_up_offset_mm=0.0,
            point_z_offset_mm=0.0,
            camera_to_base=np.eye(4, dtype=float),
            base_z_offset_mm=0.0,
        )
        sender = OpenCVGraspSender(host=None, port=5005, timeout_s=1.0, dry_run=True)

        response = sender.send_target(targets[0], source="opencv_test", task="in")

        self.assertEqual(errors, {})
        payload = response["payload"]
        self.assertAlmostEqual(payload["mask_top_u"], 8.5)
        self.assertAlmostEqual(payload["mask_top_v"], 5.0)
        self.assertAlmostEqual(payload["detected_top_z_mm"], 450.0, places=3)

    def test_opencv_targets_include_object_short_axis_yaw(self):
        frame = green_frame(80, 80)
        rect = ((40.0, 40.0), (36.0, 14.0), 30.0)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(frame, [box], (0, 0, 255))
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=50,
                morphology_kernel_size=1,
            )
        )
        segments = detector.detect(frame)
        depth = np.full((80, 80), 0.50, dtype=np.float32)
        targets, errors = segments_to_ranked_targets(
            segments,
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=80,
                height=80,
                fx=100.0,
                fy=100.0,
                ppx=40.0,
                ppy=40.0,
            ),
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            camera_up_offset_mm=0.0,
            point_z_offset_mm=0.0,
            camera_to_base=np.eye(4, dtype=float),
            base_z_offset_mm=0.0,
        )
        targets = attach_object_short_axis_yaw(
            targets,
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=80,
                height=80,
                fx=100.0,
                fy=100.0,
                ppx=40.0,
                ppy=40.0,
            ),
            camera_to_base=np.eye(4, dtype=float),
        )
        sender = OpenCVGraspSender(host=None, port=5005, timeout_s=1.0, dry_run=True)

        response = sender.send_target(targets[0], source="opencv_test", task="in")

        self.assertEqual(errors, {})
        payload = response["payload"]
        self.assertIsNotNone(payload["object_short_axis_yaw_deg"])
        self.assertEqual(
            payload["object_yaw_source"],
            "opencv_min_area_rect_short_edge",
        )
        self.assertGreater(payload["object_rect_width_px"], 0.0)
        self.assertGreater(payload["object_rect_height_px"], 0.0)

    def test_dynamic_mode_defaults_to_hand_eye_result_when_not_explicitly_disabled(self):
        args = build_parser().parse_args(["dynamic", "--dry-run", "--conveyor-speed-mm-s", "80"])

        validate_args(args)

        self.assertIsNotNone(args.hand_eye_result)

    def test_static_mode_defaults_to_hand_eye_result_for_top_z_metadata(self):
        args = build_parser().parse_args(["--dry-run"])

        validate_args(args)

        self.assertIsNotNone(args.hand_eye_result)

    def test_invalid_green_hsv_is_rejected(self):
        args = build_parser().parse_args(["--dry-run", "--green-lower-hsv", "180", "0", "0"])

        with self.assertRaisesRegex(ValueError, "H must be in"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()

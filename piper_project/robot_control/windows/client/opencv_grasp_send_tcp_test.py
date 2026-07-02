from __future__ import annotations

import unittest

import cv2
import numpy as np

from robot_control.windows.client.opencv_grasp_send_tcp import (
    GreenConveyorDetectorConfig,
    GreenConveyorOpenCVDetector,
    OpenCVGraspSender,
    apply_belt_depth_plane_compensation,
    attach_object_short_axis_yaw,
    build_parser,
    detector_config_from_args,
    hsv_in_range,
    validate_args,
)
from robot_control.windows.client.yolo_seg_grasp_send_tcp import segments_to_ranked_targets
from robot_control.windows.vision.belt_depth_plane import BeltDepthPlane
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


def black_frame(width: int = 160, height: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (8, 8, 8)
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

    def test_detector_finds_non_black_object_on_black_belt(self):
        frame = black_frame()
        frame[40:80, 60:100] = (0, 0, 255)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                lower_belt_hsv=(0, 0, 0),
                upper_belt_hsv=(179, 255, 60),
                min_area_px=100,
                morphology_kernel_size=3,
            )
        )

        segments = detector.detect(frame)

        self.assertEqual(len(segments), 1)
        detection = segments[0].detection
        self.assertEqual((detection.x, detection.y, detection.width, detection.height), (60, 40, 40, 40))

    def test_detector_can_reject_low_saturation_high_value_highlight(self):
        frame = green_frame()
        frame[20:50, 20:50] = (245, 245, 245)
        frame[60:95, 70:110] = (0, 0, 255)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=100,
                morphology_kernel_size=1,
                reject_highlight=True,
                highlight_max_s=40,
                highlight_min_v=220,
            )
        )

        segments = detector.detect(frame)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].detection.x, segments[0].detection.y), (70, 60))
        self.assertIsNotNone(detector.last_highlight_mask)
        self.assertEqual(int(detector.last_highlight_mask[25, 25]), 255)

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

    def test_detector_entry_edge_only_rejects_objects_touching_that_edge(self):
        frame = green_frame()
        frame[20:40, 40:60] = (0, 0, 255)
        frame[30:50, 100:120] = (255, 0, 0)
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=50,
                morphology_kernel_size=1,
                roi_xywh=(20, 20, 100, 60),
                entry_edge="right",
            )
        )

        segments = detector.detect(frame)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].detection.x, segments[0].detection.y), (40, 20))

    def test_detector_entry_clear_frames_requires_consecutive_clear_frames(self):
        detector = GreenConveyorOpenCVDetector(
            GreenConveyorDetectorConfig(
                min_area_px=50,
                morphology_kernel_size=1,
                entry_edge="right",
                entry_clear_frames=2,
            )
        )
        frame_touching = green_frame()
        frame_touching[40:60, 140:160] = (0, 0, 255)
        frame_clear_once = green_frame()
        frame_clear_once[40:60, 120:140] = (0, 0, 255)
        frame_clear_twice = green_frame()
        frame_clear_twice[40:60, 100:120] = (0, 0, 255)

        self.assertEqual(detector.detect(frame_touching), [])
        self.assertEqual(detector.detect(frame_clear_once), [])
        segments = detector.detect(frame_clear_twice)

        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].detection.x, segments[0].detection.y), (100, 40))

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
        self.assertIsNone(payload["detected_top_z_mm"])
        self.assertAlmostEqual(payload["detected_top_camera_z_m"], 0.45, places=3)

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

    def test_belt_hsv_arguments_override_green_compatibility_alias(self):
        args = build_parser().parse_args(
            [
                "--dry-run",
                "--green-lower-hsv",
                "35",
                "40",
                "40",
                "--green-upper-hsv",
                "90",
                "255",
                "255",
                "--belt-lower-hsv",
                "0",
                "0",
                "0",
                "--belt-upper-hsv",
                "179",
                "255",
                "70",
                "--reject-highlight",
                "--highlight-max-s",
                "80",
                "--highlight-min-v",
                "190",
                "--reject-shadow",
                "--shadow-max-v",
                "20",
            ]
        )

        validate_args(args)
        config = detector_config_from_args(args)

        self.assertEqual(config.lower_belt_hsv, (0, 0, 0))
        self.assertEqual(config.upper_belt_hsv, (179, 255, 70))
        self.assertTrue(config.reject_highlight)
        self.assertEqual(config.highlight_max_s, 80)
        self.assertEqual(config.highlight_min_v, 190)
        self.assertTrue(config.reject_shadow)
        self.assertEqual(config.shadow_max_v, 20)

    def test_entry_edge_arguments_are_stored_in_detector_config(self):
        args = build_parser().parse_args(
            [
                "--dry-run",
                "--entry-edge",
                "right",
                "--entry-clear-margin-px",
                "12",
                "--entry-clear-frames",
                "3",
            ]
        )

        validate_args(args)
        config = detector_config_from_args(args)

        self.assertEqual(config.entry_edge, "right")
        self.assertEqual(config.entry_clear_margin_px, 12)
        self.assertEqual(config.entry_clear_frames, 3)

    def test_belt_depth_plane_compensation_updates_top_depth(self):
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

        updated = apply_belt_depth_plane_compensation(
            targets,
            FakeDepthFrame(depth),
            plane=BeltDepthPlane(
                width=20,
                height=20,
                a=0.001,
                b=0.0,
                c=0.45,
                roi_xywh=(0, 0, 20, 20),
            ),
            top_percentile=20.0,
            max_depth_m=1.5,
            intrinsics=PinholeIntrinsics(
                width=20,
                height=20,
                fx=100.0,
                fy=100.0,
                ppx=10.0,
                ppy=10.0,
            ),
            camera_to_base=np.eye(4, dtype=float),
        )

        self.assertEqual(errors, {})
        self.assertAlmostEqual(updated[0].top_point.depth_m, 0.452, places=3)
        self.assertAlmostEqual(updated[0].top_base_xyz_mm[2], 452.0, places=1)

    def test_belt_depth_plane_arguments_are_validated(self):
        args = build_parser().parse_args(
            [
                "--dry-run",
                "--belt-depth-top-percentile",
                "20",
            ]
        )

        validate_args(args)
        self.assertEqual(args.belt_depth_top_percentile, 20.0)


if __name__ == "__main__":
    unittest.main()

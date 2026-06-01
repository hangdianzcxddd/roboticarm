
from __future__ import annotations

import unittest

import cv2
import numpy as np

from robot_control.windows.vision.detector import (
    BackgroundSubtractorDetector,
    ColorThresholdDetector,
    ContourDetectionConfig,
    detections_from_mask,
)


class TestOpenCVDetectors(unittest.TestCase):
    def test_detections_from_mask_returns_largest_contour_center(self):
        mask = np.zeros((160, 220), dtype=np.uint8)
        mask[40:80, 50:110] = 255
        mask[100:112, 10:22] = 255

        detections = detections_from_mask(
            mask,
            ContourDetectionConfig(min_area=100, max_results=1, morphology_kernel_size=1),
        )

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(
            (detection.x, detection.y, detection.width, detection.height),
            (50, 40, 60, 40),
        )
        self.assertAlmostEqual(detection.u, 80, delta=1)
        self.assertAlmostEqual(detection.v, 60, delta=1)

    def test_detections_from_mask_filters_small_noise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:13, 10:13] = 255

        detections = detections_from_mask(
            mask,
            ContourDetectionConfig(min_area=100, morphology_kernel_size=1),
        )

        self.assertEqual(detections, [])

    def test_background_subtractor_detector_finds_new_foreground_object(self):
        detector = BackgroundSubtractorDetector(
            config=ContourDetectionConfig(
                min_area=200,
                max_results=1,
                morphology_kernel_size=3,
            ),
            history=20,
            var_threshold=16,
            detect_shadows=False,
        )
        background = np.zeros((120, 160, 3), dtype=np.uint8)
        for _ in range(20):
            detector.apply(background, learning_rate=0.5)

        frame = background.copy()
        cv2.rectangle(frame, (60, 35), (100, 75), (255, 255, 255), -1)

        detections = detector.detect(frame, learning_rate=0.0)

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].u, 80, delta=2)
        self.assertAlmostEqual(detections[0].v, 55, delta=2)

    def test_color_threshold_detector_finds_hsv_target(self):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (30, 20), (80, 70), (0, 255, 0), -1)
        detector = ColorThresholdDetector(
            lower_hsv=(35, 80, 80),
            upper_hsv=(85, 255, 255),
            config=ContourDetectionConfig(
                min_area=200,
                max_results=1,
                morphology_kernel_size=3,
            ),
        )

        detections = detector.detect(frame)

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].u, 55, delta=2)
        self.assertAlmostEqual(detections[0].v, 45, delta=2)


if __name__ == "__main__":
    unittest.main()

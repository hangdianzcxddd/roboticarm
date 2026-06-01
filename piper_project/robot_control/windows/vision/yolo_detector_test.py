from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from robot_control.shared.paths import PROJECT_DIR
from robot_control.windows.vision.yolo_detector import YOLOObjectDetector


class FakeTensor:
    def __init__(self, values):
        self._values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.0, 20.0, 50.0, 80.0]])
        self.conf = FakeTensor([0.91])
        self.cls = FakeTensor([39])

    def __len__(self):
        return 1


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()


class TestYOLOObjectDetector(unittest.TestCase):
    @patch("robot_control.windows.vision.yolo_detector.require_yolo_model")
    @patch("robot_control.windows.vision.yolo_detector._load_ultralytics_yolo")
    def test_detect_returns_bbox_center_and_metadata(self, load_yolo, require_model):
        fake_model = MagicMock()
        fake_model.names = {39: "bottle"}
        fake_model.predict.return_value = [FakeResult()]
        load_yolo.return_value = MagicMock(return_value=fake_model)
        require_model.return_value = PROJECT_DIR / "models" / "yolo26n.pt"

        detector = YOLOObjectDetector(max_results=1)
        detections = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual((detection.u, detection.v), (30, 50))
        self.assertEqual((detection.x, detection.y, detection.width, detection.height), (10, 20, 40, 60))
        self.assertEqual(detection.label, "bottle")
        self.assertEqual(detection.class_id, 39)
        self.assertAlmostEqual(detection.confidence, 0.91)


if __name__ == "__main__":
    unittest.main()

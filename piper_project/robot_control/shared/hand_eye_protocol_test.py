from __future__ import annotations

import unittest

from robot_control.shared.hand_eye_protocol import (
    HandEyeSampleCommand,
    PoseVector,
    decode_message,
    encode_message,
    hand_eye_sample_from_message,
    make_hand_eye_sample_message,
)


class TestHandEyeProtocol(unittest.TestCase):
    def test_hand_eye_sample_round_trip(self):
        command = HandEyeSampleCommand(
            target_to_camera=PoseVector(
                rvec=(0.1, -0.2, 0.3),
                tvec_m=(0.4, 0.01, 0.8),
            ),
            sample_id="sample-001",
            image_path="captures/sample.png",
            reprojection_error_px=0.42,
            detected_corners=88,
        )

        decoded = decode_message(encode_message(make_hand_eye_sample_message(command)))
        parsed = hand_eye_sample_from_message(decoded)

        self.assertEqual(parsed.sample_id, "sample-001")
        self.assertEqual(parsed.image_path, "captures/sample.png")
        self.assertEqual(parsed.target_to_camera.rvec, (0.1, -0.2, 0.3))
        self.assertEqual(parsed.target_to_camera.tvec_m, (0.4, 0.01, 0.8))
        self.assertAlmostEqual(parsed.reprojection_error_px, 0.42)
        self.assertEqual(parsed.detected_corners, 88)


if __name__ == "__main__":
    unittest.main()

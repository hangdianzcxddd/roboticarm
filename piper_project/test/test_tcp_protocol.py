from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calibration.simulated_mapping import camera_xyz_to_simulated_robot_pose
from communication.protocol import (
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
        )

        decoded = decode_message(encode_message(make_camera_point_message(command)))
        parsed = camera_point_from_message(decoded)

        self.assertEqual(parsed.command_id, "cmd-1")
        self.assertAlmostEqual(parsed.x_m, 0.10)
        self.assertAlmostEqual(parsed.y_m, -0.05)
        self.assertAlmostEqual(parsed.z_m, 0.40)
        self.assertEqual((parsed.u, parsed.v), (320, 240))
        self.assertEqual(parsed.source, "unit_test")

    def test_simulated_mapping_is_bounded(self):
        pose = camera_xyz_to_simulated_robot_pose(10.0, -10.0, 10.0)

        self.assertEqual(pose.x_mm, 190.0)
        self.assertEqual(pose.y_mm, -40.0)
        self.assertEqual(pose.z_mm, 250.0)
        self.assertEqual((pose.rx_deg, pose.ry_deg, pose.rz_deg), (0.0, 85.0, 0.0))


if __name__ == "__main__":
    unittest.main()

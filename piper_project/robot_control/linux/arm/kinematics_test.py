from __future__ import annotations

import unittest

from robot_control.linux.arm.kinematics import forward_kinematics_pose


class TestPiperForwardKinematics(unittest.TestCase):
    def test_known_continuous_catch_ready_joint_orientation(self):
        pose = forward_kinematics_pose((0.0, 1.5, -1.3, 0.0, 1.2, 0.0))

        self.assertAlmostEqual(pose.rx_deg, 180.0, places=6)
        self.assertAlmostEqual(pose.ry_deg, 14.7859086817, places=6)
        self.assertAlmostEqual(pose.rz_deg, 180.0, places=6)

    def test_requested_initial_joint_orientation(self):
        pose = forward_kinematics_pose((0.0, 2.0, -1.65, 0.0, 1.2, -1.35))

        self.assertAlmostEqual(pose.x_mm, 401.4317488185, places=6)
        self.assertAlmostEqual(pose.z_mm, 310.8876909530, places=6)
        self.assertAlmostEqual(pose.rx_deg, -173.9576420309, places=6)
        self.assertAlmostEqual(pose.ry_deg, 1.3534770142, places=6)
        self.assertAlmostEqual(pose.rz_deg, -102.5792601461, places=6)


if __name__ == "__main__":
    unittest.main()

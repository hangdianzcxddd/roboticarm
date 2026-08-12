from __future__ import annotations

import unittest

import numpy as np

from robot_control.windows.hand_eye.validate_hand_eye_click import (
    make_click_validation_result,
    project_base_point_to_pixel,
)
from robot_control.windows.vision.d435i_intrinsics import PinholeIntrinsics
from robot_control.windows.vision.depth import CameraPoint


class TestValidateHandEyeClick(unittest.TestCase):
    def test_maps_camera_point_to_base_and_reports_error(self):
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.1, -0.2, 0.03])
        point = CameraPoint(
            u=100,
            v=200,
            depth_m=0.5,
            x_m=0.02,
            y_m=0.01,
            z_m=0.5,
        )

        result = make_click_validation_result(
            point,
            camera_to_base,
            expected_base_xyz_m=(0.125, -0.190, 0.520),
        )

        np.testing.assert_allclose(result.base_xyz_m, [0.12, -0.19, 0.53])
        np.testing.assert_allclose(result.error_xyz_m, [-0.005, 0.0, 0.01])
        self.assertAlmostEqual(result.error_norm_m, np.hypot(0.005, 0.01))

    def test_projects_base_point_to_pixel(self):
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.10, 0.20, 0.30])
        intrinsics = PinholeIntrinsics(
            width=640,
            height=480,
            fx=400.0,
            fy=500.0,
            ppx=320.0,
            ppy=240.0,
        )

        projected = project_base_point_to_pixel(
            camera_to_base,
            base_xyz_m=(0.20, 0.10, 0.80),
            intrinsics=intrinsics,
        )

        self.assertEqual(projected, (400, 140, 0.5))

    def test_project_base_point_returns_none_when_behind_camera(self):
        camera_to_base = np.eye(4, dtype=float)
        intrinsics = PinholeIntrinsics(
            width=640,
            height=480,
            fx=400.0,
            fy=400.0,
            ppx=320.0,
            ppy=240.0,
        )

        projected = project_base_point_to_pixel(
            camera_to_base,
            base_xyz_m=(0.0, 0.0, -0.1),
            intrinsics=intrinsics,
        )

        self.assertIsNone(projected)


if __name__ == "__main__":
    unittest.main()

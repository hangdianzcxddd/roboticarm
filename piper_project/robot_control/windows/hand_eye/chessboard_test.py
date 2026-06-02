from __future__ import annotations

import unittest

import numpy as np

from robot_control.windows.hand_eye.chessboard import (
    ChessboardConfig,
    intrinsics_from_profile,
    make_object_points,
)


class TestChessboard(unittest.TestCase):
    def test_default_8x11_board_object_points_use_15mm_spacing(self):
        points = make_object_points(
            ChessboardConfig(corners_cols=11, corners_rows=8, checker_size_m=0.015)
        )

        self.assertEqual(points.shape, (88, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[1], [0.015, 0.0, 0.0])
        np.testing.assert_allclose(points[11], [0.0, 0.015, 0.0])
        np.testing.assert_allclose(points[-1], [0.150, 0.105, 0.0])

    def test_640x480_intrinsic_profile_matches_supplied_values(self):
        intrinsics = intrinsics_from_profile("640x480")

        self.assertEqual((intrinsics.width, intrinsics.height), (640, 480))
        np.testing.assert_allclose(
            intrinsics.camera_matrix,
            np.array(
                [
                    [391.185, 0.0, 319.123],
                    [0.0, 391.185, 244.756],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
        np.testing.assert_allclose(intrinsics.dist_coeffs, np.zeros((5, 1)))


if __name__ == "__main__":
    unittest.main()

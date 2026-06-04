from __future__ import annotations

import unittest

from robot_control.windows.vision.d435i_intrinsics import PinholeIntrinsics
from robot_control.windows.vision.depth import deproject_pixel_to_camera_xyz


class FakeDepthFrame:
    def __init__(self, width: int, height: int, depth_m: float) -> None:
        self._width = width
        self._height = height
        self._depth_m = depth_m

    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_distance(self, u: int, v: int) -> float:
        del u, v
        return self._depth_m


class TestDepthDeprojection(unittest.TestCase):
    def test_deprojects_with_supplied_pinhole_intrinsics(self):
        point = deproject_pixel_to_camera_xyz(
            FakeDepthFrame(width=640, height=480, depth_m=0.5),
            419,
            145,
            intrinsics=PinholeIntrinsics(
                width=640,
                height=480,
                fx=400.0,
                fy=500.0,
                ppx=319.0,
                ppy=245.0,
                source="test",
            ),
        )

        self.assertAlmostEqual(point.x_m, 0.125)
        self.assertAlmostEqual(point.y_m, -0.1)
        self.assertAlmostEqual(point.z_m, 0.5)

    def test_rejects_intrinsics_resolution_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match depth frame"):
            deproject_pixel_to_camera_xyz(
                FakeDepthFrame(width=640, height=480, depth_m=0.5),
                100,
                100,
                intrinsics=PinholeIntrinsics(
                    width=848,
                    height=480,
                    fx=400.0,
                    fy=400.0,
                    ppx=0.0,
                    ppy=0.0,
                ),
            )


if __name__ == "__main__":
    unittest.main()

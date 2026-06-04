from __future__ import annotations

import unittest

from robot_control.windows.vision.d435i_intrinsics import (
    find_d435i_intrinsics,
    load_d435i_rectified_intrinsics,
)


class TestD435iIntrinsics(unittest.TestCase):
    def test_loads_usable_rectified_intrinsics(self):
        intrinsics = load_d435i_rectified_intrinsics()

        resolutions = {(item.width, item.height) for item in intrinsics}
        self.assertIn((640, 480), resolutions)
        self.assertIn((848, 480), resolutions)
        self.assertNotIn((0, 0), resolutions)

    def test_finds_intrinsics_by_video_resolution(self):
        intrinsics = find_d435i_intrinsics(640, 480)

        self.assertEqual(intrinsics.width, 640)
        self.assertEqual(intrinsics.height, 480)
        self.assertAlmostEqual(intrinsics.fx, 391.185)
        self.assertAlmostEqual(intrinsics.fy, 391.185)
        self.assertAlmostEqual(intrinsics.ppx, 319.123)
        self.assertAlmostEqual(intrinsics.ppy, 244.756)


if __name__ == "__main__":
    unittest.main()

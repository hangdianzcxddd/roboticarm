from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_control.windows.vision.belt_depth_plane import (
    fit_belt_depth_plane,
    load_belt_depth_plane,
    mask_depth_percentile,
    patch_depth_median,
    save_belt_depth_plane,
)


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


class TestBeltDepthPlane(unittest.TestCase):
    def test_fit_plane_recovers_image_depth_coefficients(self):
        samples = []
        for u, v in [(0, 0), (10, 0), (0, 10), (10, 10), (5, 7)]:
            samples.append((u, v, 0.5 + 0.001 * u + 0.002 * v))

        plane = fit_belt_depth_plane(samples, width=20, height=20, patch_size=9)

        self.assertAlmostEqual(plane.a, 0.001, places=8)
        self.assertAlmostEqual(plane.b, 0.002, places=8)
        self.assertAlmostEqual(plane.c, 0.5, places=8)
        self.assertLess(plane.residuals.max_abs_mm, 0.001)

    def test_save_and_load_plane_round_trip(self):
        plane = fit_belt_depth_plane(
            [(0, 0, 0.5), (10, 0, 0.51), (0, 10, 0.52)],
            width=20,
            height=20,
            roi_xywh=(1, 2, 10, 12),
            patch_size=11,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "belt_depth_plane.json"
            save_belt_depth_plane(path, plane)
            loaded = load_belt_depth_plane(path)

        self.assertEqual(loaded.width, 20)
        self.assertEqual(loaded.height, 20)
        self.assertEqual(loaded.roi_xywh, (1, 2, 10, 12))
        self.assertEqual(loaded.patch_size, 11)
        self.assertAlmostEqual(loaded.depth_at(5, 5), plane.depth_at(5, 5))

    def test_mask_depth_percentile_uses_valid_depth_pixels(self):
        depth = np.array(
            [
                [0.0, 0.40, 0.50],
                [0.45, 0.60, 0.70],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )

        depth_m, count = mask_depth_percentile(FakeDepthFrame(depth), mask, percentile=25)

        self.assertEqual(count, 3)
        self.assertAlmostEqual(depth_m, 0.45, places=3)

    def test_patch_depth_median_samples_neighborhood(self):
        depth = np.full((5, 5), 0.5, dtype=np.float32)
        depth[2, 2] = 0.8

        self.assertAlmostEqual(
            patch_depth_median(FakeDepthFrame(depth), 2, 2, patch_size=3),
            0.5,
            places=3,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from graspnet_backend import candidate_dicts_from_grasp_array, sample_points_for_graspnet


class TestGraspNetBackend(unittest.TestCase):
    def test_sample_points_for_graspnet_pads_small_cloud(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.6],
            ],
            dtype=np.float32,
        )

        sampled = sample_points_for_graspnet(points, 8)

        self.assertEqual(sampled.shape, (8, 3))
        self.assertTrue(np.all(np.isfinite(sampled)))
        self.assertTrue(
            all(
                np.isclose(depth, 0.5) or np.isclose(depth, 0.6)
                for depth in np.unique(sampled[:, 2])
            )
        )

    def test_sample_points_for_graspnet_is_deterministic(self):
        points = np.arange(90, dtype=np.float32).reshape(30, 3)

        first = sample_points_for_graspnet(points, 12)
        second = sample_points_for_graspnet(points, 12)

        np.testing.assert_array_equal(first, second)

    def test_candidate_dicts_from_grasp_array_maps_official_row_layout(self):
        row = np.zeros(17, dtype=np.float32)
        row[0] = 0.91
        row[1] = 0.08
        row[3] = 0.03
        row[4:13] = np.eye(3, dtype=np.float32).reshape(-1)
        row[13:16] = [0.1, -0.2, 0.55]

        candidates = candidate_dicts_from_grasp_array(row.reshape(1, -1))

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["score"], 0.91, places=5)
        self.assertTrue(
            np.allclose(candidates[0]["position_camera_m"], [0.1, -0.2, 0.55])
        )
        self.assertEqual(candidates[0]["rotation_camera"], np.eye(3).tolist())
        self.assertAlmostEqual(candidates[0]["width_m"], 0.08, places=5)


if __name__ == "__main__":
    unittest.main()

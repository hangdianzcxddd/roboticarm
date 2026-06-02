from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_control.linux.tcp_calibration.calibration import (
    evaluate_tcp,
    feedback_to_end_pose,
    load_calibration,
    load_sample_poses,
    make_calibration_document,
    make_samples_document,
    rotation_matrix_xyz_deg,
    solve_fixed_point_tcp,
    tcp_position_base_mm,
    write_yaml,
)
from robot_control.shared.geometry import EndPose


TRUE_OFFSET = np.array([12.5, -4.0, 138.0])
REFERENCE_POINT = np.array([320.0, -25.0, 185.0])
ORIENTATIONS_DEG = (
    (0.0, 0.0, 0.0),
    (18.0, 0.0, 0.0),
    (-20.0, 12.0, 0.0),
    (0.0, -24.0, 15.0),
    (14.0, 20.0, -18.0),
    (-16.0, -18.0, 22.0),
    (24.0, -10.0, -20.0),
)


def make_pivot_poses(noise_mm: float = 0.0) -> list[EndPose]:
    poses = []
    for index, (rx_deg, ry_deg, rz_deg) in enumerate(ORIENTATIONS_DEG):
        rotation = rotation_matrix_xyz_deg(rx_deg, ry_deg, rz_deg)
        translation = REFERENCE_POINT - rotation @ TRUE_OFFSET
        if noise_mm:
            translation = translation + np.array(
                [noise_mm * index, -noise_mm * index / 2.0, noise_mm / 3.0]
            )
        poses.append(EndPose(*translation, rx_deg, ry_deg, rz_deg))
    return poses


class FakeEndPose:
    X_axis = 123456
    Y_axis = -23456
    Z_axis = 300125
    RX_axis = 10250
    RY_axis = -20500
    RZ_axis = 90000


class FakeFeedback:
    end_pose = FakeEndPose()


class TestCalibration(unittest.TestCase):
    def test_rotation_matrix_matches_expected_xyz_rotation_order(self):
        rotation = rotation_matrix_xyz_deg(90.0, 0.0, 90.0)

        np.testing.assert_allclose(
            rotation @ np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            atol=1e-12,
        )

    def test_solve_recovers_tcp_offset_and_reference_point(self):
        solution = solve_fixed_point_tcp(make_pivot_poses())

        np.testing.assert_allclose(solution.offset_j6_tcp_mm, TRUE_OFFSET, atol=1e-9)
        np.testing.assert_allclose(
            solution.reference_point_base_mm,
            REFERENCE_POINT,
            atol=1e-9,
        )
        self.assertAlmostEqual(solution.rms_error_mm, 0.0, places=9)
        self.assertAlmostEqual(solution.max_error_mm, 0.0, places=9)
        self.assertEqual(solution.matrix_rank, 6)

    def test_solve_rejects_repeated_orientation(self):
        poses = [
            EndPose(100.0 + index, 50.0, 200.0, 0.0, 0.0, 0.0)
            for index in range(6)
        ]

        with self.assertRaisesRegex(ValueError, "underconstrained"):
            solve_fixed_point_tcp(poses)

    def test_evaluate_tcp_reports_independent_validation_error(self):
        poses = make_pivot_poses(noise_mm=0.05)

        metrics = evaluate_tcp(poses, TRUE_OFFSET, REFERENCE_POINT)

        self.assertGreater(metrics.reference_rms_error_mm, 0.0)
        self.assertLess(metrics.reference_max_error_mm, 0.4)

    def test_tcp_position_applies_offset_in_rotated_j6_frame(self):
        pose = EndPose(10.0, 20.0, 30.0, 0.0, 0.0, 90.0)

        point = tcp_position_base_mm(pose, (5.0, 0.0, 0.0))

        np.testing.assert_allclose(point, np.array([10.0, 25.0, 30.0]), atol=1e-9)

    def test_feedback_to_end_pose_converts_sdk_units(self):
        pose = feedback_to_end_pose(FakeFeedback())

        self.assertEqual(
            pose,
            EndPose(123.456, -23.456, 300.125, 10.25, -20.5, 90.0),
        )

    def test_yaml_round_trip(self):
        poses = make_pivot_poses()
        solution = solve_fixed_point_tcp(poses)
        with tempfile.TemporaryDirectory() as directory:
            samples_path = Path(directory) / "samples.yaml"
            config_path = Path(directory) / "tcp.yaml"
            write_yaml(samples_path, make_samples_document(poses))
            write_yaml(
                config_path,
                make_calibration_document(
                    solution,
                    source_samples=samples_path,
                    sample_count=len(poses),
                    rms_limit_mm=1.0,
                    max_error_limit_mm=2.0,
                ),
            )

            loaded_poses = load_sample_poses(samples_path)
            calibration = load_calibration(config_path)

        self.assertEqual(loaded_poses, poses)
        self.assertEqual(calibration["status"], "calibrated")
        np.testing.assert_allclose(calibration["offset_j6_tcp_mm"], TRUE_OFFSET)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from robot_control.linux.hand_eye.calibration import (
    HandEyeSample,
    evaluate_residuals,
    load_samples,
    make_samples_document,
    solve_hand_eye,
    transform_from_end_pose,
    write_json,
)
from robot_control.shared.geometry import EndPose
from robot_control.shared.hand_eye_protocol import PoseVector


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=3)
    vector = vector / np.linalg.norm(vector) * rng.uniform(0.3, 1.0)
    rotation, _ = cv2.Rodrigues(vector.reshape(3, 1))
    return rotation


def end_pose_from_transform(transform: np.ndarray) -> EndPose:
    rotation = transform[:3, :3]
    sy = -rotation[2, 0]
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy < 1e-9:
        rx = 0.0
        ry = math.atan2(sy, cy)
        rz = math.atan2(-rotation[0, 1], rotation[1, 1])
    else:
        rx = math.atan2(rotation[2, 1], rotation[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(rotation[1, 0], rotation[0, 0])
    return EndPose(
        *(transform[:3, 3] * 1000.0),
        math.degrees(rx),
        math.degrees(ry),
        math.degrees(rz),
    )


def pose_vector_from_transform(transform: np.ndarray) -> PoseVector:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return PoseVector(
        rvec=tuple(float(value) for value in rvec.reshape(3)),
        tvec_m=tuple(float(value) for value in transform[:3, 3]),
    )


class TestHandEyeCalibration(unittest.TestCase):
    def test_eye_in_hand_recovers_gripper_to_camera(self):
        base_to_target = make_transform(
            random_rotation(1),
            np.array([0.30, -0.10, 0.50]),
        )
        gripper_to_camera = make_transform(
            random_rotation(2),
            np.array([0.02, 0.01, 0.08]),
        )
        samples = []
        for index in range(25):
            base_to_gripper = make_transform(
                random_rotation(20 + index),
                np.array(
                    [
                        0.18 + 0.01 * index,
                        -0.08 + 0.02 * math.sin(index),
                        0.36 + 0.02 * math.cos(index),
                    ]
                ),
            )
            base_to_camera = base_to_gripper @ gripper_to_camera
            camera_to_target = invert(base_to_camera) @ base_to_target
            samples.append(
                HandEyeSample(
                    index=index + 1,
                    sample_id=f"s{index + 1}",
                    robot_base_to_gripper=end_pose_from_transform(base_to_gripper),
                    target_to_camera=pose_vector_from_transform(camera_to_target),
                )
            )

        result = solve_hand_eye(samples, mode="eye_in_hand")

        np.testing.assert_allclose(result.rotation, gripper_to_camera[:3, :3], atol=1e-9)
        np.testing.assert_allclose(result.translation_m, gripper_to_camera[:3, 3], atol=1e-9)
        residuals = evaluate_residuals(samples, result)
        self.assertLess(residuals.translation_rms_m, 1e-9)
        self.assertLess(residuals.rotation_rms_deg, 1e-5)

    def test_eye_to_hand_recovers_camera_to_base(self):
        camera_to_base = make_transform(
            random_rotation(3),
            np.array([0.30, -0.20, 0.80]),
        )
        base_to_camera = invert(camera_to_base)
        gripper_to_target = make_transform(
            random_rotation(4),
            np.array([0.05, 0.00, 0.12]),
        )
        samples = []
        for index in range(25):
            base_to_gripper = make_transform(
                random_rotation(40 + index),
                np.array(
                    [
                        0.20 + 0.015 * index,
                        -0.10 + 0.02 * math.sin(index),
                        0.42 + 0.02 * math.cos(index),
                    ]
                ),
            )
            target_to_camera = base_to_camera @ base_to_gripper @ gripper_to_target
            samples.append(
                HandEyeSample(
                    index=index + 1,
                    sample_id=f"s{index + 1}",
                    robot_base_to_gripper=end_pose_from_transform(base_to_gripper),
                    target_to_camera=pose_vector_from_transform(target_to_camera),
                )
            )

        result = solve_hand_eye(samples, mode="eye_to_hand")

        np.testing.assert_allclose(result.rotation, camera_to_base[:3, :3], atol=1e-9)
        np.testing.assert_allclose(result.translation_m, camera_to_base[:3, 3], atol=1e-9)
        residuals = evaluate_residuals(samples, result)
        self.assertLess(residuals.translation_rms_m, 1e-9)
        self.assertLess(residuals.rotation_rms_deg, 1e-5)

    def test_json_round_trip(self):
        sample = HandEyeSample(
            index=1,
            sample_id="s1",
            robot_base_to_gripper=EndPose(100.0, 20.0, 300.0, 10.0, 20.0, 30.0),
            target_to_camera=PoseVector((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
            image_path="sample.png",
            reprojection_error_px=0.5,
            detected_corners=88,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            write_json(path, make_samples_document([sample], mode="eye_to_hand"))

            loaded = load_samples(path)

        self.assertEqual(loaded, [sample])

    def test_transform_from_end_pose_uses_meters(self):
        transform = transform_from_end_pose(
            EndPose(100.0, -200.0, 300.0, 0.0, 0.0, 0.0)
        )

        np.testing.assert_allclose(transform[:3, 3], np.array([0.1, -0.2, 0.3]))


if __name__ == "__main__":
    unittest.main()

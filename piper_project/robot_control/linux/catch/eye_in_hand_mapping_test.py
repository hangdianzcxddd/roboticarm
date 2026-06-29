from __future__ import annotations

import numpy as np

from robot_control.linux.catch.eye_in_hand_mapping import (
    camera_to_base_transform_from_gripper_pose,
    map_camera_point_to_base_xyz_m_eye_in_hand,
)
from robot_control.shared.geometry import EndPose


def test_eye_in_hand_mapping_uses_camera_to_gripper_transform():
    camera_to_gripper = np.eye(4, dtype=float)
    camera_to_gripper[:3, 3] = np.array([0.02, -0.03, 0.10])
    gripper_pose = EndPose(
        x_mm=100.0,
        y_mm=200.0,
        z_mm=300.0,
        rx_deg=0.0,
        ry_deg=0.0,
        rz_deg=0.0,
    )

    camera_to_base = camera_to_base_transform_from_gripper_pose(
        gripper_pose,
        camera_to_gripper,
    )
    mapped = map_camera_point_to_base_xyz_m_eye_in_hand(
        0.01,
        0.02,
        0.03,
        gripper_pose,
        camera_to_gripper,
    )

    np.testing.assert_allclose(
        camera_to_base[:3, 3],
        np.array([0.12, 0.17, 0.40]),
    )
    np.testing.assert_allclose(mapped, np.array([0.13, 0.19, 0.43]))

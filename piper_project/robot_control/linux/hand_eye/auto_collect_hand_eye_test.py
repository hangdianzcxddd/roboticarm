from __future__ import annotations

import argparse
import math
import random
import sys
from types import ModuleType
import unittest

cv2_stub = ModuleType("cv2")
cv2_stub.CALIB_HAND_EYE_TSAI = 0  # type: ignore[attr-defined]
cv2_stub.CALIB_HAND_EYE_PARK = 1  # type: ignore[attr-defined]
cv2_stub.CALIB_HAND_EYE_HORAUD = 2  # type: ignore[attr-defined]
cv2_stub.CALIB_HAND_EYE_ANDREFF = 3  # type: ignore[attr-defined]
cv2_stub.CALIB_HAND_EYE_DANIILIDIS = 4  # type: ignore[attr-defined]
sys.modules.setdefault("cv2", cv2_stub)
sys.modules.setdefault("numpy", ModuleType("numpy"))
sys.modules.setdefault("yaml", ModuleType("yaml"))

from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.hand_eye.auto_collect_hand_eye import (
    JointBounds,
    JointJumpError,
    JointStepLimits,
    assert_joint_step_is_safe,
    joint_bounds_from_seed,
    joint_segments,
    random_joint_candidate,
    wait_until_joints_reached,
)
from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG


class FakeJointState:
    def __init__(self, joints: tuple[int, int, int, int, int, int]) -> None:
        (
            self.joint_1,
            self.joint_2,
            self.joint_3,
            self.joint_4,
            self.joint_5,
            self.joint_6,
        ) = joints


class FakeJointFeedback:
    def __init__(self, joints_deg: tuple[float, float, float, float, float, float]) -> None:
        joints = tuple(
            round(math.radians(value) * RAD_TO_MILLI_DEG) for value in joints_deg
        )
        self.joint_state = FakeJointState(joints)
        self.time_stamp = 1.0


class FakeController:
    def __init__(self, feedback: FakeJointFeedback) -> None:
        self.feedback = feedback

    def get_joint_state(self) -> FakeJointFeedback:
        return self.feedback


def make_args(**overrides: float) -> argparse.Namespace:
    values = {
        "max_joint_delta_deg": 12.0,
        "max_wrist_joint_delta_deg": 8.0,
        "range_base_joint_deg": 15.0,
        "range_wrist_joint_deg": 18.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestAutoCollectJointSafety(unittest.TestCase):
    def test_joint_bounds_stay_inside_seed_range_and_robot_limits(self) -> None:
        seed = (
            math.radians(145.0),
            math.radians(175.0),
            math.radians(-165.0),
            math.radians(95.0),
            math.radians(65.0),
            math.radians(115.0),
        )

        bounds = joint_bounds_from_seed(seed, RobotConfig(), make_args())

        self.assertAlmostEqual(math.degrees(bounds.max_rad[0]), 150.0)
        self.assertAlmostEqual(math.degrees(bounds.max_rad[1]), 180.0)
        self.assertAlmostEqual(math.degrees(bounds.min_rad[2]), -170.0)
        self.assertAlmostEqual(math.degrees(bounds.max_rad[3]), 100.0)
        self.assertAlmostEqual(math.degrees(bounds.max_rad[4]), 70.0)
        self.assertAlmostEqual(math.degrees(bounds.max_rad[5]), 120.0)

    def test_random_joint_candidate_respects_bounds(self) -> None:
        rng = random.Random(3)
        base = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        bounds = JointBounds(
            min_rad=tuple(math.radians(-1.0) for _ in range(6)),  # type: ignore[arg-type]
            max_rad=tuple(math.radians(1.0) for _ in range(6)),  # type: ignore[arg-type]
        )
        steps = JointStepLimits(tuple(math.radians(10.0) for _ in range(6)))  # type: ignore[arg-type]

        candidate = random_joint_candidate(
            rng,
            base,
            bounds,
            steps,
            min_dof=6,
            max_dof=6,
            shrink=1.0,
            attempt=1,
        )

        self.assertEqual(set(candidate.selected_fields), {"j1", "j2", "j3", "j4", "j5", "j6"})
        for joint in candidate.joints_rad:
            self.assertLessEqual(abs(math.degrees(joint)), 1.0)

    def test_joint_segments_split_large_wrist_motion(self) -> None:
        args = make_args(max_joint_delta_deg=12.0, max_wrist_joint_delta_deg=8.0)
        start = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        target = (
            0.0,
            0.0,
            0.0,
            math.radians(20.0),
            math.radians(2.0),
            0.0,
        )

        segments = joint_segments(start, target, args)

        self.assertEqual(len(segments), 3)
        self.assertAlmostEqual(math.degrees(segments[0][3]), 20.0 / 3.0)
        self.assertAlmostEqual(math.degrees(segments[-1][3]), 20.0)

    def test_joint_jump_detection_rejects_wrist_flip(self) -> None:
        args = make_args(max_joint_delta_deg=12.0, max_wrist_joint_delta_deg=8.0)
        before = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        after = (0.0, 0.0, 0.0, math.radians(180.0), 0.0, 0.0)

        with self.assertRaisesRegex(JointJumpError, "j4=180.0deg>8.0deg"):
            assert_joint_step_is_safe(before, after, args)

    def test_joint_jump_detection_allows_small_move(self) -> None:
        args = make_args(max_joint_delta_deg=12.0, max_wrist_joint_delta_deg=8.0)
        before = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        after = (
            math.radians(5.0),
            math.radians(10.0),
            math.radians(1.0),
            math.radians(7.5),
            0.0,
            0.0,
        )

        assert_joint_step_is_safe(before, after, args)

    def test_wait_until_joints_reached_monitors_jump_before_arrival(self) -> None:
        args = make_args(max_joint_delta_deg=12.0, max_wrist_joint_delta_deg=8.0)
        controller = FakeController(
            FakeJointFeedback((0.0, 0.0, 0.0, 30.0, 0.0, 0.0))
        )

        with self.assertRaisesRegex(JointJumpError, "j4=30.0deg>8.0deg"):
            wait_until_joints_reached(
                controller,  # type: ignore[arg-type]
                (0.0, 0.0, 0.0, math.radians(30.0), 0.0, 0.0),
                timeout_s=0.0,
                tolerance_rad=math.radians(1.0),
                start_joints_rad=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                args=args,
            )


if __name__ == "__main__":
    unittest.main()

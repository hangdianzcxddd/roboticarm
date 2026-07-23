from __future__ import annotations

import math
import unittest

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server_graspnet import (
    DEFAULT_HAND_EYE_RESULT,
    ResolvedCandidate,
    approach_tilt_deg,
    build_parser,
    fk_tcp_pose,
    fk_tcp_transform,
    interpolate_joint_path,
    resolve_candidate_variants,
    solve_ik,
    validate_args,
    validate_candidate_geometry,
    validate_joint_deltas,
)
from robot_control.shared.protocol import CameraPointCommand, GraspCandidate


KNOWN_JOINTS_DEG = (
    45.04309486,
    79.06897299,
    -47.83154153,
    -7.57847344,
    -10.26872747,
    5.56229599,
)


def known_joints_rad() -> tuple[float, ...]:
    return tuple(math.radians(value) for value in KNOWN_JOINTS_DEG)


def command_from_fk(args, *, z_offset_mm: float = 0.0) -> CameraPointCommand:
    transform = fk_tcp_transform(known_joints_rad(), args)
    position_m = transform[:3, 3] / 1000.0
    position_m[2] += z_offset_mm / 1000.0
    rotation = tuple(tuple(float(value) for value in row) for row in transform[:3, :3])
    return CameraPointCommand(
        x_m=float(position_m[0]),
        y_m=float(position_m[1]),
        z_m=float(position_m[2]),
        vision_source="graspnet",
        detected_top_z_mm=float(transform[2, 3] + 5.0),
        grasp_candidates=(
            GraspCandidate(
                position_camera_m=tuple(float(value) for value in position_m),
                rotation_camera=rotation,
                score=0.9,
                width_m=0.04,
                index=3,
            ),
        ),
    )


class TestGraspNetCatchServer(unittest.TestCase):
    def test_parser_defaults_are_conservative(self):
        args = build_parser().parse_args([])

        validate_args(args)
        self.assertEqual(args.approach_axis, "x")
        self.assertEqual(args.hand_eye_result, DEFAULT_HAND_EYE_RESULT)
        self.assertEqual(args.hand_eye_result.name, "eye_to_hand_result_auto.json")
        self.assertAlmostEqual(args.max_approach_tilt_deg, 45.0)
        self.assertAlmostEqual(args.approach_tilt_weight, 1.0)
        self.assertFalse(args.allow_approach_axis_flip)
        self.assertTrue(args.try_finger_flip)
        self.assertAlmostEqual(args.min_grasp_z_mm, 120.0)
        self.assertAlmostEqual(args.min_trajectory_tcp_z_mm, 110.0)
        self.assertEqual(tuple(args.max_joint_delta_deg), (60, 70, 70, 90, 60, 90))
        self.assertEqual(
            tuple(args.max_approach_joint_delta_deg),
            (25, 35, 35, 45, 35, 45),
        )
        self.assertTrue(args.safe_gripper_stop_on_effort)
        self.assertEqual(args.collision_protection_level, 2)

    def test_approach_tilt_is_measured_from_base_downward_axis(self):
        candidate = ResolvedCandidate(
            source_index=0,
            source="test",
            score=1.0,
            width_m=0.04,
            camera_position_m=(0.0, 0.0, 0.0),
            grasp_pose=fk_tcp_pose(known_joints_rad(), build_parser().parse_args([])),
            approach_axis=np.array([0.0, 0.0, -1.0]),
            orientation_variant="test",
        )

        self.assertAlmostEqual(approach_tilt_deg(candidate), 0.0)

    def test_full_ik_recovers_known_reachable_pose(self):
        args = build_parser().parse_args([])
        target_joints = known_joints_rad()
        target_pose = fk_tcp_pose(target_joints, args)

        solution = solve_ik(
            target_pose,
            (target_joints,),
            PiperController(),
            args,
            reference_joints=target_joints,
        )

        self.assertLess(solution.position_error_mm, 1e-5)
        self.assertLess(solution.orientation_error_deg, 1e-5)
        self.assertLess(solution.condition_number, args.max_jacobian_condition)

    def test_candidate_below_grasp_floor_is_rejected_not_clamped(self):
        args = build_parser().parse_args([])
        command = command_from_fk(args, z_offset_mm=-200.0)
        candidate = resolve_candidate_variants(
            command.grasp_candidates[0], command, np.eye(4), args
        )[0]

        with self.assertRaisesRegex(SafetyError, "grasp z"):
            validate_candidate_geometry(
                candidate, command, PiperController(), args
            )

    def test_large_joint_change_is_rejected_per_joint(self):
        args = build_parser().parse_args([])
        start = np.radians((0.0, 90.0, -90.0, 0.0, 0.0, 0.0))
        target = start.copy()
        target[0] += math.radians(61.0)

        with self.assertRaisesRegex(SafetyError, "J1=61.00deg"):
            validate_joint_deltas(start, target, args)

    def test_joint_interpolation_respects_waypoint_step(self):
        path = interpolate_joint_path(
            np.zeros(6),
            np.radians((19.0, -11.0, 0.0, 0.0, 0.0, 0.0)),
            5.0,
        )

        maximum_step = max(
            float(np.max(np.abs(np.degrees(np.asarray(right) - np.asarray(left)))))
            for left, right in zip(path, path[1:])
        )
        self.assertLessEqual(maximum_step, 5.0)

    def test_dynamic_candidate_is_rejected_in_phase_one(self):
        args = build_parser().parse_args([])
        command = command_from_fk(args)
        command = CameraPointCommand(
            **{
                **command.__dict__,
                "grasp_mode": "dynamic",
                "t_arrive_s": 3.0,
                "pick_x_mm": 350.0,
                "pick_y_mm": 0.0,
            }
        )

        with self.assertRaisesRegex(SafetyError, "dynamic GraspNet grasp is disabled"):
            resolve_candidate_variants(
                command.grasp_candidates[0], command, np.eye(4), args
            )


if __name__ == "__main__":
    unittest.main()

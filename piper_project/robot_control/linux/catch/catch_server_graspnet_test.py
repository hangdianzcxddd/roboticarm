from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server_graspnet import (
    DEFAULT_HAND_EYE_RESULT,
    ResolvedCandidate,
    approach_tilt_deg,
    build_parser,
    emergency_stop_on_failure,
    fk_tcp_pose,
    fk_tcp_transform,
    interpolate_joint_path,
    move_to_nominal_joints,
    resolve_candidate_variants,
    solve_ik,
    validate_args,
    validate_candidate_geometry,
    validate_joint_change,
    validate_joint_deltas,
    validate_runtime_joint_state,
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
        self.assertAlmostEqual(args.grasp_z_offset_mm, 0.0)
        self.assertAlmostEqual(args.min_target_y_mm, -550.0)
        self.assertAlmostEqual(args.max_target_y_mm, -350.0)
        self.assertAlmostEqual(args.min_grasp_z_mm, 120.0)
        self.assertAlmostEqual(args.min_trajectory_tcp_z_mm, 110.0)
        self.assertEqual(tuple(args.nominal_joints_deg), (0, 90, -90, 0, 30, 0))
        self.assertEqual(tuple(args.max_joint_delta_deg), (60,) * 6)
        self.assertEqual(tuple(args.max_startup_joint_delta_deg), (60,) * 6)
        self.assertAlmostEqual(args.max_runtime_path_error_deg, 5.0)
        self.assertEqual(
            tuple(args.max_approach_joint_delta_deg),
            (25, 35, 35, 45, 35, 45),
        )
        self.assertTrue(args.safe_gripper_stop_on_effort)
        self.assertEqual(args.collision_protection_level, 2)

    def test_startup_move_converts_nominal_degrees_to_radians(self):
        args = build_parser().parse_args(
            [
                "--nominal-joints-deg",
                "0",
                "90",
                "-90",
                "0",
                "30",
                "0",
                "--max-startup-joint-delta-deg",
                "100",
                "100",
                "100",
                "100",
                "100",
                "100",
            ]
        )

        class FakeController:
            def __init__(self):
                self.validated = None

            def validate_joints(self, joints):
                self.validated = tuple(joints)

        controller = FakeController()
        module = "robot_control.linux.catch.catch_server_graspnet"
        with (
            patch(f"{module}.read_live_joints", return_value=(0.0,) * 6),
            patch(f"{module}.validate_path_height") as validate_path,
            patch(f"{module}.execute_joint_path") as execute_path,
        ):
            target = move_to_nominal_joints(controller, args)

        expected = (0.0, math.pi / 2, -math.pi / 2, 0.0, math.pi / 6, 0.0)
        np.testing.assert_allclose(target, expected)
        np.testing.assert_allclose(controller.validated, expected)
        path = validate_path.call_args.args[0]
        np.testing.assert_allclose(path[0], (0.0,) * 6)
        np.testing.assert_allclose(path[-1], expected)
        self.assertEqual(execute_path.call_args.kwargs["speed"], args.speed)

    def test_startup_change_over_configured_limit_is_rejected(self):
        args = build_parser().parse_args([])

        with self.assertRaisesRegex(SafetyError, "startup.*J2=61.00deg>60.00deg"):
            validate_joint_change(
                np.zeros(6),
                np.radians((0, 61, 0, 0, 0, 0)),
                args.max_startup_joint_delta_deg,
                label="startup",
            )

    def test_runtime_path_deviation_is_rejected(self):
        args = build_parser().parse_args(["--min-trajectory-tcp-z-mm", "1"])
        controller = PiperController()
        start = known_joints_rad()
        current = list(start)
        current[0] += math.radians(args.max_runtime_path_error_deg + 1.0)

        with self.assertRaisesRegex(SafetyError, "live joint path deviation"):
            validate_runtime_joint_state(
                current,
                start,
                start,
                controller,
                args,
                label="test",
            )

    def test_failure_handler_sends_emergency_stop(self):
        class FakeController:
            def __init__(self):
                self.stop_count = 0

            def emergency_stop(self):
                self.stop_count += 1

        controller = FakeController()

        emergency_stop_on_failure(controller, context="test failure")

        self.assertEqual(controller.stop_count, 1)

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

    def test_reversed_target_y_bounds_are_rejected(self):
        args = build_parser().parse_args(
            ["--min-target-y-mm", "-350", "--max-target-y-mm", "-550"]
        )

        with self.assertRaisesRegex(ValueError, "min-target-y-mm"):
            validate_args(args)

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
        args = build_parser().parse_args(
            ["--min-target-y-mm", "-1000", "--max-target-y-mm", "1000"]
        )
        command = command_from_fk(args, z_offset_mm=-200.0)
        candidate = resolve_candidate_variants(
            command.grasp_candidates[0], command, np.eye(4), args
        )[0]

        with self.assertRaisesRegex(SafetyError, "grasp z"):
            validate_candidate_geometry(
                candidate, command, PiperController(), args
            )

    def test_candidate_far_from_detected_object_center_is_rejected(self):
        args = build_parser().parse_args(
            [
                "--min-target-y-mm",
                "-1000",
                "--max-target-y-mm",
                "1000",
                "--min-grasp-z-mm",
                "1",
                "--min-trajectory-tcp-z-mm",
                "1",
                "--max-candidate-center-distance-mm",
                "100",
            ]
        )
        command = command_from_fk(args)
        candidate = resolve_candidate_variants(
            command.grasp_candidates[0], command, np.eye(4), args
        )[0]
        command = CameraPointCommand(
            **{
                **command.__dict__,
                "detected_x_mm": candidate.grasp_pose.x_mm + 250.0,
                "detected_y_mm": candidate.grasp_pose.y_mm,
            }
        )

        with self.assertRaisesRegex(SafetyError, "does not belong to selected object"):
            validate_candidate_geometry(candidate, command, PiperController(), args)

    def test_grasp_z_offset_is_applied_along_base_z(self):
        default_args = build_parser().parse_args([])
        offset_args = build_parser().parse_args(["--grasp-z-offset-mm", "35"])
        command = command_from_fk(default_args)

        unadjusted = resolve_candidate_variants(
            command.grasp_candidates[0], command, np.eye(4), default_args
        )[0]
        adjusted = resolve_candidate_variants(
            command.grasp_candidates[0], command, np.eye(4), offset_args
        )[0]

        self.assertAlmostEqual(adjusted.grasp_pose.x_mm, unadjusted.grasp_pose.x_mm)
        self.assertAlmostEqual(adjusted.grasp_pose.y_mm, unadjusted.grasp_pose.y_mm)
        self.assertAlmostEqual(
            adjusted.grasp_pose.z_mm - unadjusted.grasp_pose.z_mm,
            35.0,
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

from __future__ import annotations

import argparse
import math
import unittest
from unittest import mock

import numpy as np

from robot_control.linux.arm.kinematics import forward_kinematics_pose
from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.config.robot_config import ROBOT_CONFIG
from robot_control.linux.catch.catch_server_1_1 import (
    CartesianJointMotionGuard,
    DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
    SOFTWARE_JOINT_LIMITS_DEG,
    YoloCatchTarget,
    apply_tool_z_offset_to_fk_pose,
    build_out_steps,
    build_parser,
    cartesian_only_robot_config,
    dynamic_preposition_pose,
    dynamic_timing_from_command,
    execute_step,
    object_grasp_orientation,
    pick_above_lift_z_mm,
    ready_pose_ik_seed,
    target_from_command,
    validate_cartesian_joint_limits,
    validate_multistart_cartesian_joint_limits,
    _multi_start_cartesian_ik_seeds,
    validate_joint_vector_limits,
    validate_or_record_ready_joint_anchor,
    validate_planned_ready_joint_anchor,
    validate_runtime_joint_limits,
    validate_args,
    validate_target,
)
from robot_control.linux.arm.safety import SafetyError
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import CameraPointCommand


def _args(**overrides: object) -> argparse.Namespace:
    args = build_parser().parse_args([])
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _target(yaw_deg: float = 20.0) -> YoloCatchTarget:
    command = CameraPointCommand(
        x_m=0.0,
        y_m=0.0,
        z_m=0.0,
        object_short_axis_yaw_deg=yaw_deg,
    )
    return YoloCatchTarget(command, (0.0, 0.0, 0.0), 345.0, -25.0, 160.0)


class CartesianCatchServer11Test(unittest.TestCase):
    def test_wait_preposition_ignores_x_dependent_robot_timing(self) -> None:
        args = _args(
            dynamic_preposition=True,
            dynamic_wait_placement="wait",
            final_action_time_s=0.5,
            t_robot_s=2.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=340.0,
            dynamic_t_robot_planar_near_s_per_mm=0.017,
            dynamic_t_robot_planar_far_s_per_mm=0.010,
        )
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            grasp_mode="dynamic",
            t_arrive_s=3.311,
            pick_x_mm=428.432,
            pick_y_mm=0.0,
        )

        timing = dynamic_timing_from_command(
            command,
            args,
            source_x_mm=command.pick_x_mm,
        )

        assert timing is not None
        self.assertEqual(timing.t_robot_source, "disabled_preposition_wait")
        self.assertIsNone(timing.t_robot_source_x_mm)
        self.assertIsNone(timing.t_robot_slope_s_per_mm)
        self.assertAlmostEqual(timing.t_robot_s, 2.8)
        self.assertAlmostEqual(timing.action_time_s, 0.5)
        self.assertAlmostEqual(timing.t_wait_s, 3.311 - 0.5)

    def test_nowait_preposition_uses_x_dependent_robot_timing(self) -> None:
        args = _args(
            dynamic_preposition=True,
            dynamic_wait_placement="nowait",
            t_robot_s=2.8,
            dynamic_t_robot_by_x=True,
            dynamic_t_robot_reference_x_mm=340.0,
            dynamic_t_robot_planar_near_s_per_mm=0.017,
            dynamic_t_robot_planar_far_s_per_mm=0.010,
        )
        command = CameraPointCommand(
            x_m=0.0,
            y_m=0.0,
            z_m=0.0,
            grasp_mode="dynamic",
            t_arrive_s=3.311,
            pick_x_mm=428.432,
            pick_y_mm=0.0,
        )

        timing = dynamic_timing_from_command(
            command,
            args,
            source_x_mm=command.pick_x_mm,
        )

        assert timing is not None
        self.assertEqual(timing.t_robot_source, "x_linear")
        self.assertEqual(timing.t_robot_segment, "far")
        self.assertAlmostEqual(timing.t_robot_slope_s_per_mm, 0.010)
        self.assertAlmostEqual(
            timing.t_robot_s,
            2.8 + (340.0 - 428.432) * 0.010,
        )
        self.assertAlmostEqual(timing.action_time_s, timing.t_robot_s)

    def test_pick_above_and_lift_z_follow_forward_x_gradient(self) -> None:
        args = _args(
            ready_x_mm=350.0,
            transfer_z_mm=253.252,
            pick_above_lift_z_gradient=1.0,
        )
        target = YoloCatchTarget(
            _target().camera_command,
            (0.0, 0.0, 0.0),
            390.684,
            0.0,
            180.579,
        )

        steps = build_out_steps(args, target)

        expected_z = 253.252 - (390.684 - 350.0)
        self.assertAlmostEqual(pick_above_lift_z_mm(args, target), expected_z)
        self.assertAlmostEqual(steps[1].pose.z_mm, expected_z)
        self.assertAlmostEqual(steps[3].pose.z_mm, expected_z)
        self.assertAlmostEqual(steps[0].pose.z_mm, 253.252)
        self.assertAlmostEqual(steps[2].pose.z_mm, 180.579)

    def test_pick_above_lift_z_gradient_is_adjustable_and_forward_only(self) -> None:
        args = _args(
            ready_x_mm=350.0,
            transfer_z_mm=253.252,
            pick_above_lift_z_gradient=0.5,
        )
        forward = YoloCatchTarget(
            _target().camera_command, (0.0, 0.0, 0.0), 370.0, 0.0, 160.0
        )
        behind = YoloCatchTarget(
            _target().camera_command, (0.0, 0.0, 0.0), 330.0, 0.0, 160.0
        )

        self.assertAlmostEqual(pick_above_lift_z_mm(args, forward), 243.252)
        self.assertAlmostEqual(pick_above_lift_z_mm(args, behind), 253.252)

    def test_dynamic_preposition_is_clamped_to_adjusted_pick_above_z(self) -> None:
        args = _args(
            ready_x_mm=350.0,
            transfer_z_mm=253.252,
            pick_above_lift_z_gradient=1.0,
            dynamic_preposition_clearance_mm=155.0,
        )
        target = YoloCatchTarget(
            _target().camera_command,
            (0.0, 0.0, 0.0),
            390.684,
            0.0,
            180.579,
        )

        pose, source = dynamic_preposition_pose(args, target)

        self.assertAlmostEqual(pose.z_mm, 212.568)
        self.assertTrue(source.endswith("_clamped"))

    def test_gradient_rejects_pick_above_below_grasp(self) -> None:
        args = _args(
            ready_x_mm=350.0,
            transfer_z_mm=253.252,
            pick_above_lift_z_gradient=2.0,
        )
        target = YoloCatchTarget(
            _target().camera_command, (0.0, 0.0, 0.0), 400.0, 0.0, 160.0
        )

        with self.assertRaisesRegex(SafetyError, "X-adjusted.*Z"):
            validate_target(target, args)

    def test_gradient_must_be_nonnegative(self) -> None:
        with self.assertRaisesRegex(ValueError, "pick-above-lift-z-gradient"):
            validate_args(_args(pick_above_lift_z_gradient=-0.1))

    def test_only_grasp_uses_object_orientation_and_lift_restores_pregrasp(self) -> None:
        args = _args(grasp_yaw_by_object_short_edge=True)
        steps = build_out_steps(args, _target())

        prepared = (args.rx_deg, args.ry_deg, args.rz_deg)
        object_orientation = object_grasp_orientation(args, _target())
        for index in (0, 1, 3, 4, 5, 6):
            pose = steps[index].pose
            self.assertEqual((pose.rx_deg, pose.ry_deg, pose.rz_deg), prepared)
        grasp_pose = steps[2].pose
        self.assertEqual(
            (grasp_pose.rx_deg, grasp_pose.ry_deg, grasp_pose.rz_deg),
            object_orientation,
        )
        self.assertEqual(steps[3].pose, steps[1].pose)

    def test_every_planned_waypoint_is_cartesian_move_p(self) -> None:
        steps = build_out_steps(_args(), _target())
        self.assertTrue(all(step.mode == "P" for step in steps))
        self.assertTrue(all(not step.planar_ik for step in steps))
        self.assertTrue(all(not step.joint_pitch_before_pose for step in steps))
        self.assertTrue(all(not step.joint_pitch_after_pose for step in steps))

    def test_only_j6_uses_a_software_angle_limit(self) -> None:
        config = cartesian_only_robot_config("can0")
        self.assertFalse(config.start_sdk_joint_limit)
        self.assertEqual(
            SOFTWARE_JOINT_LIMITS_DEG,
            (
                None,
                None,
                None,
                None,
                None,
                (-180.0, 180.0),
            ),
        )
        self.assertEqual(
            config.joint_software_limit_enabled,
            (False, False, False, False, False, True),
        )
        self.assertAlmostEqual(
            math.degrees(config.joint_limits_rad[5].min_value),
            -180.0,
        )
        self.assertAlmostEqual(
            math.degrees(config.joint_limits_rad[5].max_value),
            180.0,
        )

    def test_j1_to_j5_command_limits_are_disabled_but_j6_is_retained(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        controller.validate_joints(
            tuple(math.radians(value) for value in (151.0, -2.097, 1.0, 101.0, 71.0, 179.0))
        )

        with self.assertRaisesRegex(SafetyError, "joint_6"):
            controller.validate_joints(
                tuple(math.radians(value) for value in (0.0, 0.0, 0.0, 0.0, 0.0, 181.0))
            )

    def test_runtime_feedback_allows_negative_j2_but_rejects_j6_limit(self) -> None:
        controller = mock.Mock()
        controller.get_joint_state.return_value = object()
        with mock.patch(
            "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
            return_value=tuple(
                math.radians(value)
                for value in (0.0, -2.097, 0.0, 0.0, 0.0, 179.0)
            ),
        ):
            joints = validate_runtime_joint_limits(controller, label="startup")
        self.assertAlmostEqual(math.degrees(joints[1]), -2.097)

        with mock.patch(
            "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
            return_value=tuple(
                math.radians(value)
                for value in (0.0, -2.097, 0.0, 0.0, 0.0, 181.0)
            ),
        ), self.assertRaisesRegex(SafetyError, "feedback J6"):
            validate_runtime_joint_limits(controller, label="startup")

    def test_disabled_angle_limit_still_rejects_nonfinite_feedback(self) -> None:
        controller = mock.Mock()
        controller.get_joint_state.return_value = object()
        with mock.patch(
            "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
            return_value=(float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0),
        ), self.assertRaisesRegex(SafetyError, "feedback J1 must be finite"):
            validate_runtime_joint_limits(controller, label="startup")

    def test_reported_startup_ready_pose_is_reachable(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        pose = EndPose(350.0, 0.0, 253.252, -175.958, 1.353, -102.579)

        solution = validate_cartesian_joint_limits(
            controller,
            pose,
            seed_joints=(0.0,) * 6,
            label="startup_ready",
        )

        self.assertGreater(math.degrees(solution[1]), 90.0)
        self.assertLessEqual(math.degrees(solution[1]), 180.0)
        self.assertAlmostEqual(math.degrees(solution[4]), 69.908, places=2)

    def test_cartesian_ik_default_offset_matches_piper_d6_91_model(self) -> None:
        self.assertEqual(DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM, 82.0)
        self.assertEqual(
            ROBOT_CONFIG.modified_dh_params[-1].d_mm
            + DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
            91.0,
        )
        pose = EndPose(350.0, 0.0, 253.252, -175.958, 1.353, -102.579)
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )

        solution = validate_cartesian_joint_limits(
            controller,
            pose,
            seed_joints=(0.0,) * 6,
            label="corrected_ready",
            joint_limit_margin_deg=0.0,
        )
        corrected_fk = apply_tool_z_offset_to_fk_pose(
            forward_kinematics_pose(solution),
            DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
        )

        self.assertLess(np.linalg.norm(
            np.array((corrected_fk.x_mm, corrected_fk.y_mm, corrected_fk.z_mm))
            - np.array((pose.x_mm, pose.y_mm, pose.z_mm))
        ), 1e-4)

    def test_cartesian_ik_rejects_limit_clamped_target_with_residual(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        pose = EndPose(360.600, 0.0, 253.252, -175.958, 1.353, -102.579)
        seed = tuple(
            math.radians(value)
            for value in (-0.077, 101.854, -80.989, 0.456, 69.927, -77.573)
        )

        with self.assertRaisesRegex(
            SafetyError,
            "target is outside Piper joint limits.*J5=70.000deg",
        ):
            validate_cartesian_joint_limits(
                controller,
                pose,
                seed_joints=seed,
                label="pick_above",
                fk_tool_z_offset_mm=82.0,
                joint_limit_margin_deg=0.0,
            )

    def test_multistart_cartesian_ik_defaults_to_four_deterministic_seeds(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        seed = tuple(
            math.radians(value)
            for value in (0.0, 102.0, -81.0, 0.5, 69.0, -77.0)
        )
        seeds = _multi_start_cartesian_ik_seeds(controller, seed)
        self.assertEqual(len(seeds), 4)

        pose = EndPose(350.0, 0.0, 253.252, -175.958, 1.353, -102.579)
        with mock.patch(
            "robot_control.linux.catch.catch_server_1_1.validate_cartesian_joint_limits",
            return_value=seed,
        ) as solve:
            selected = validate_multistart_cartesian_joint_limits(
                controller,
                pose,
                seed_joints=seed,
                label="pick_above",
                fk_tool_z_offset_mm=82.0,
            )
        self.assertEqual(selected, seed)
        self.assertEqual(solve.call_count, 4)

    def test_multistart_cartesian_ik_seed_count_is_adjustable(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        seed = tuple(
            math.radians(value)
            for value in (0.0, 102.0, -81.0, 0.5, 69.0, -77.0)
        )

        self.assertEqual(
            len(_multi_start_cartesian_ik_seeds(controller, seed, seed_count=8)),
            8,
        )
        args = _args(multi_start_cartesian_ik_seed_count=8)
        validate_args(args)
        with self.assertRaisesRegex(ValueError, "seed-count"):
            validate_args(_args(multi_start_cartesian_ik_seed_count=0))

    def test_dynamic_preposition_uses_corrected_cartesian_ik_wrist_solution(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        ready = tuple(
            math.radians(value)
            for value in (-0.116, 101.915, -81.061, 0.475, 69.908, -77.636)
        )
        pose = EndPose(289.751, 0.0, 199.575, -175.958, 1.353, -102.579)

        solution = validate_cartesian_joint_limits(
            controller,
            pose,
            seed_joints=ready,
            label="dynamic_preposition",
        )

        self.assertAlmostEqual(math.degrees(solution[4]), 54.373, places=2)

    def test_logged_cartesian_path_is_reachable_with_corrected_model(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        poses = (
            EndPose(350.000, 0.000, 253.252, -175.958, 1.353, -102.579),
            EndPose(289.751, 0.000, 253.252, -175.958, 1.353, -102.579),
            EndPose(289.751, 0.000, 199.575, -175.958, 1.353, -102.579),
            EndPose(289.751, 0.000, 159.575, -175.958, 1.353, -91.996),
            EndPose(289.751, 0.000, 253.252, -175.958, 1.353, -91.996),
            EndPose(60.000, 270.000, 253.252, -175.958, 1.353, -102.579),
            EndPose(350.000, 0.000, 253.252, -175.958, 1.353, -102.579),
        )
        seed = (0.0,) * 6
        for pose in poses:
            seed = validate_cartesian_joint_limits(
                controller,
                pose,
                seed_joints=seed,
                label="logged_path",
            )

    def test_cartesian_ik_compensation_parameters_are_exposed_and_validated(self) -> None:
        args = _args()
        self.assertEqual(args.cartesian_ik_fk_tool_z_offset_mm, 82.0)
        self.assertEqual(args.cartesian_ik_joint_target_tolerance_deg, 20.0)

    def test_j2_parked_zero_is_valid_for_startup_motion_guard(self) -> None:
        joints = validate_joint_vector_limits(
            (0.0,) * 6,
            label="before startup_ready",
        )

        self.assertEqual(joints, (0.0,) * 6)

    def test_cartesian_precheck_solves_pose_inside_all_joint_limits(self) -> None:
        controller = PiperController(
            config=cartesian_only_robot_config("can0"),
            sdk=object(),
        )
        seed = tuple(
            math.radians(value)
            for value in (20.0, 30.0, -20.0, 20.0, -30.0, 40.0)
        )
        pose = apply_tool_z_offset_to_fk_pose(
            forward_kinematics_pose(seed),
            DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
        )
        solution = validate_cartesian_joint_limits(
            controller,
            pose,
            seed_joints=seed,
            label="test",
        )
        for index, limit in enumerate(controller.config.joint_limits_rad):
            self.assertGreaterEqual(math.degrees(solution[index]), math.degrees(limit.min_value))
            self.assertLessEqual(math.degrees(solution[index]), math.degrees(limit.max_value))

    def test_dynamic_y_is_not_forced_by_default(self) -> None:
        self.assertIsNone(build_parser().parse_args([]).force_pick_y_mm)

    def test_cartesian_pick_x_offset_is_applied_without_planar_ik(self) -> None:
        args = _args(pick_x_offset_mm=-45.0)
        command = CameraPointCommand(0.0, 0.0, 0.0)
        transform = np.eye(4, dtype=float)
        transform[:3, 3] = np.array([0.40, -0.10, 0.10])

        target = target_from_command(command, transform, args)

        self.assertAlmostEqual(target.raw_pick_x_mm, 400.0)
        self.assertAlmostEqual(target.pick_x_mm, 355.0)
        self.assertEqual(target.pick_x_source, "mapped_command_x_cartesian_offset")

    def test_cartesian_guard_rejects_feedback_wrap_or_branch_jump(self) -> None:
        guard = CartesianJointMotionGuard.create(
            "return_ready",
            tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 100)),
            tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 0)),
        )
        with self.assertRaisesRegex(SafetyError, "feedback jumped"):
            guard.observe(
                tuple(math.radians(value) for value in (0, 0, 0, 0, 0, -170))
            )

    def test_cartesian_guard_accepts_continuous_firmware_ik_solution_by_fk(self) -> None:
        actual = tuple(
            math.radians(value)
            for value in (-0.141, 93.863, -57.474, 0.551, 54.373, -77.818)
        )
        planned = list(actual)
        planned[4] -= math.radians(25.0)
        target_pose = apply_tool_z_offset_to_fk_pose(
            forward_kinematics_pose(actual),
            DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
        )
        guard = CartesianJointMotionGuard.create(
            "dynamic_preposition",
            actual,
            tuple(planned),
        )

        guard.validate_target_branch(
            target_pose,
            fk_tool_z_offset_mm=DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
            joint_tolerance_deg=20.0,
            position_tolerance_mm=8.0,
            orientation_tolerance_deg=8.0,
        )

    def test_cartesian_guard_rejects_joint_mismatch_when_corrected_fk_disagrees(self) -> None:
        actual = tuple(
            math.radians(value)
            for value in (-0.141, 93.863, -57.474, 0.551, 54.373, -77.818)
        )
        planned = list(actual)
        planned[4] -= math.radians(25.0)
        actual_pose = apply_tool_z_offset_to_fk_pose(
            forward_kinematics_pose(actual),
            DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
        )
        target_pose = EndPose(
            actual_pose.x_mm + 100.0,
            actual_pose.y_mm,
            actual_pose.z_mm,
            actual_pose.rx_deg,
            actual_pose.ry_deg,
            actual_pose.rz_deg,
        )
        guard = CartesianJointMotionGuard.create(
            "dynamic_preposition",
            actual,
            tuple(planned),
        )

        with self.assertRaisesRegex(
            SafetyError,
            "joint feedback disagrees with corrected Cartesian IK model",
        ):
            guard.validate_target_branch(
                target_pose,
                fk_tool_z_offset_mm=DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
                joint_tolerance_deg=20.0,
                position_tolerance_mm=8.0,
                orientation_tolerance_deg=8.0,
            )

    def test_ready_pose_rejects_a_different_joint_branch(self) -> None:
        controller = argparse.Namespace()
        step = build_out_steps(_args(), _target())[-1]
        first_branch = tuple(
            math.radians(value) for value in (10, 20, -30, 40, -20, 50)
        )
        validate_or_record_ready_joint_anchor(controller, step, first_branch)

        changed_branch = list(first_branch)
        changed_branch[5] += math.radians(10.0)
        with self.assertRaisesRegex(SafetyError, "different joint branch"):
            validate_or_record_ready_joint_anchor(
                controller,
                step,
                tuple(changed_branch),
            )

    def test_ready_pose_rejects_different_planned_branch_before_motion(self) -> None:
        controller = argparse.Namespace()
        step = build_out_steps(_args(), _target())[-1]
        actual_branch = tuple(
            math.radians(value) for value in (10, 20, -30, 40, -20, 50)
        )
        startup_ik_branch = tuple(
            math.radians(value) for value in (10, 20, -39, 40, -20, 50)
        )
        validate_or_record_ready_joint_anchor(controller, step, actual_branch)
        validate_planned_ready_joint_anchor(controller, step, startup_ik_branch)
        changed_branch = list(startup_ik_branch)
        changed_branch[3] -= math.radians(10.0)
        with self.assertRaisesRegex(SafetyError, "planned.*different joint branch"):
            validate_planned_ready_joint_anchor(
                controller,
                step,
                tuple(changed_branch),
            )

    def test_ready_pose_ik_reuses_model_counterpart_of_startup_branch(self) -> None:
        controller = argparse.Namespace()
        step = build_out_steps(_args(), _target())[-1]
        actual_branch = tuple(
            math.radians(value) for value in (10, 20, -30, 40, -20, 50)
        )
        startup_ik_branch = tuple(
            math.radians(value) for value in (10, 20, -39, 40, -20, 50)
        )
        validate_or_record_ready_joint_anchor(controller, step, actual_branch)
        validate_planned_ready_joint_anchor(controller, step, startup_ik_branch)

        seed = ready_pose_ik_seed(controller, step, (0.0,) * 6)

        self.assertEqual(seed, startup_ik_branch)

    def test_ready_open_already_at_pose_skips_cartesian_move(self) -> None:
        args = _args(gripper_settle_s=0.0)
        step = build_out_steps(args, _target())[0]
        current_pose = EndPose(
            349.917,
            0.058,
            253.151,
            -175.988,
            1.293,
            -102.535,
        )
        actual_joints = tuple(
            math.radians(value) for value in (10, 20, -30, 40, -20, 50)
        )
        controller = mock.Mock()
        controller._catch_server_1_1_ready_joint_anchor = None
        controller._catch_server_1_1_ready_ik_joint_anchor = None
        controller.get_end_pose.return_value = object()
        controller.get_joint_state.return_value = object()

        with (
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.feedback_to_pose",
                return_value=current_pose,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
                return_value=actual_joints,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.validate_cartesian_joint_limits",
            ) as solve_ik,
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.execute_cartesian_pose_guarded",
            ) as move_cartesian,
        ):
            send_count, _gripper_result = execute_step(controller, args, step)

        self.assertEqual(send_count, 0)
        solve_ik.assert_not_called()
        move_cartesian.assert_not_called()
        controller.set_gripper.assert_called_once()

    def test_return_ready_solves_from_startup_ik_and_checks_actual_branch(self) -> None:
        args = _args(command_duration_s=0.0, no_wait_waypoints=True)
        step = build_out_steps(args, _target())[-1]
        current_pose = EndPose(
            step.pose.x_mm - 100.0,
            step.pose.y_mm,
            step.pose.z_mm,
            step.pose.rx_deg,
            step.pose.ry_deg,
            step.pose.rz_deg,
        )
        current_joints = tuple(
            math.radians(value) for value in (20, 60, -50, 30, -10, 40)
        )
        actual_anchor = tuple(
            math.radians(value) for value in (10, 20, -30, 40, -20, 50)
        )
        startup_ik_anchor = tuple(
            math.radians(value) for value in (10, 20, -39, 40, -20, 50)
        )
        controller = mock.Mock()
        controller._catch_server_1_1_ready_joint_anchor = actual_anchor
        controller._catch_server_1_1_ready_ik_joint_anchor = startup_ik_anchor
        controller.get_end_pose.return_value = object()
        controller.get_joint_state.return_value = object()

        with (
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.feedback_to_pose",
                return_value=current_pose,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
                return_value=current_joints,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.validate_runtime_joint_limits",
                side_effect=(current_joints, actual_anchor),
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.validate_cartesian_joint_limits",
                return_value=startup_ik_anchor,
            ) as solve_ik,
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.execute_cartesian_pose_guarded",
                return_value=1,
            ) as move_cartesian,
        ):
            send_count, _gripper_result = execute_step(controller, args, step)

        self.assertEqual(send_count, 1)
        self.assertEqual(solve_ik.call_args.kwargs["seed_joints"], startup_ik_anchor)
        move_cartesian.assert_called_once()

    def test_ready_guard_rejects_continued_rotation_away_from_target(self) -> None:
        guard = CartesianJointMotionGuard.create(
            "return_ready",
            tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 60)),
            tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 0)),
        )
        guard.observe(tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 68)))
        with self.assertRaisesRegex(SafetyError, "away from.*ready-pose"):
            guard.observe(
                tuple(math.radians(value) for value in (0, 0, 0, 0, 0, 72))
            )

    def test_failed_cartesian_step_rolls_back_to_previous_joint_state(self) -> None:
        args = _args(command_duration_s=0.0, no_wait_waypoints=True)
        previous = EndPose(350.0, 0.0, 253.252, args.rx_deg, args.ry_deg, args.rz_deg)
        previous_joints = (0.0,) * 6
        step = build_out_steps(args, _target())[1]
        controller = mock.Mock()
        controller.get_end_pose.return_value = object()
        controller.get_joint_state.return_value = object()

        with (
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.feedback_to_pose",
                return_value=previous,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.joint_feedback_to_rad",
                return_value=previous_joints,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.validate_runtime_joint_limits",
                return_value=previous_joints,
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.validate_cartesian_joint_limits",
                side_effect=SafetyError("unreachable"),
            ),
            mock.patch(
                "robot_control.linux.catch.catch_server_1_1.rollback_to_joint_state",
            ) as rollback,
            self.assertRaisesRegex(SafetyError, "unreachable"),
        ):
            execute_step(controller, args, step)

        controller.hold_current_position.assert_called_once_with(
            speed_percent=10,
            command_duration_s=0.05,
        )
        rollback.assert_called_once_with(
            controller,
            args,
            previous_joints,
            label=step.label,
        )


if __name__ == "__main__":
    unittest.main()

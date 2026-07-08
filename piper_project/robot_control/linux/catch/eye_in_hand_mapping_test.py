from __future__ import annotations

from argparse import Namespace
import contextlib
import io

import numpy as np

from robot_control.linux.arm.piper_arm import PiperController
from robot_control.linux.arm.piper_arm_test import FakePiperSdk
from robot_control.linux.catch.catch_server_2 import (
    apply_target_offset_to_camera_to_base,
    build_parser,
    handle_message,
    move_to_start_ready,
    ready_pose,
    resolve_ready_pose_override,
    resolve_initial_pose,
    server_1_motion_args,
    validate_eye_in_hand_args,
)
from robot_control.linux.catch import catch_server_1
from robot_control.linux.config.robot_config import RobotConfig
from robot_control.linux.catch.eye_in_hand_mapping import (
    camera_to_base_transform_from_gripper_pose,
    map_camera_point_to_base_xyz_m_eye_in_hand,
)
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import CameraPointCommand, make_camera_point_message


def make_controller() -> PiperController:
    return PiperController(
        config=RobotConfig(command_interval_s=0, connect_timeout_s=0.1),
        sdk=FakePiperSdk(),
    )


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


def test_target_offset_adjusts_camera_to_base_translation():
    camera_to_base = np.eye(4, dtype=float)
    args = Namespace(
        target_x_offset_mm=20.0,
        target_y_offset_mm=-5.0,
        target_z_offset_mm=10.0,
    )

    adjusted = apply_target_offset_to_camera_to_base(args, camera_to_base)

    np.testing.assert_allclose(adjusted[:3, 3], np.array([0.02, -0.005, 0.01]))
    np.testing.assert_allclose(camera_to_base[:3, 3], np.zeros(3))


def test_initial_pose_defaults_to_resolved_ready_pose():
    args = build_parser().parse_args([])
    controller = make_controller()

    pose, source = resolve_initial_pose(args, controller)

    assert source == "ready_pose"
    assert pose == EndPose(
        x_mm=args.ready_x_mm,
        y_mm=args.ready_y_mm,
        z_mm=args.transfer_z_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )


def test_ready_pose_args_source_does_not_recurse_when_patched_into_server_1():
    args = build_parser().parse_args(["--ready-pose-source", "args"])
    original_ready_pose = catch_server_1.ready_pose

    try:
        catch_server_1.ready_pose = ready_pose
        pose = ready_pose(args)
    finally:
        catch_server_1.ready_pose = original_ready_pose

    assert pose == EndPose(
        x_mm=args.ready_x_mm,
        y_mm=args.ready_y_mm,
        z_mm=args.transfer_z_mm,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )


def test_ready_pose_source_initial_reuses_configured_initial_pose_for_sequence_ready():
    args = build_parser().parse_args(
        [
            "--ready-pose-source",
            "initial",
            "--initial-pose-source",
            "configured",
            "--initial-pose-x-mm",
            "272.949",
            "--initial-pose-y-mm",
            "0",
            "--initial-pose-z-mm",
            "341.089",
            "--initial-pose-rx-deg",
            "-160.249",
            "--initial-pose-ry-deg",
            "5.355",
            "--initial-pose-rz-deg",
            "-104.572",
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "350",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "0",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
        ]
    )
    validate_eye_in_hand_args(args)
    ready_override = resolve_ready_pose_override(args, None)

    assert ready_override == EndPose(
        x_mm=272.949,
        y_mm=0.0,
        z_mm=341.089,
        rx_deg=-160.249,
        ry_deg=5.355,
        rz_deg=-104.572,
    )
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-ready-initial",
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    by_label = {step["label"]: step for step in response["payload"]["steps"]}
    assert by_label["ready_open"]["pose"] == {
        "x_mm": 272.949,
        "y_mm": 0.0,
        "z_mm": 341.089,
        "rx_deg": -160.249,
        "ry_deg": 5.355,
        "rz_deg": -104.572,
    }
    assert by_label["return_ready"]["pose"] == by_label["ready_open"]["pose"]
    assert response["payload"]["target_orientation_deg"] == {
        "rx_deg": args.rx_deg,
        "ry_deg": args.ry_deg,
        "rz_deg": args.rz_deg,
    }


def test_ready_pose_source_initial_rejects_initial_pose_source_ready_cycle():
    args = build_parser().parse_args(
        [
            "--ready-pose-source",
            "initial",
            "--initial-pose-source",
            "ready",
        ]
    )

    try:
        validate_eye_in_hand_args(args)
    except ValueError as exc:
        assert "--ready-pose-source initial" in str(exc)
    else:
        raise AssertionError("expected ready/initial cycle validation error")


def test_eye_in_hand_parser_rejects_removed_descent_protection_options():
    parser = build_parser()

    with contextlib.redirect_stderr(io.StringIO()):
        try:
            parser.parse_args(["--guarded-descent"])
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("expected --guarded-descent to be removed")


def test_eye_in_hand_server_1_motion_args_keep_ready_planar_ik_return_disabled():
    args = build_parser().parse_args([])

    assert not hasattr(args, "guarded_descent")
    motion_args = server_1_motion_args(args)

    assert not hasattr(args, "guarded_descent")
    assert not hasattr(motion_args, "guarded_descent")
    assert motion_args.disable_ready_planar_ik_return is True


def test_eye_in_hand_planar_ik_validates_without_extra_descent_protection_flags():
    args = build_parser().parse_args(
        [
            "--adaptive-joint-pitch-by-x",
        ]
    )

    validate_eye_in_hand_args(args)

    assert args.dynamic_pitch_mode == "planar-ik"


def test_eye_in_hand_preserves_planar_ik_speed_for_server_1_motion_args():
    args = build_parser().parse_args(
        [
            "--speed",
            "8",
            "--planar-ik-speed",
            "20",
        ]
    )

    validate_eye_in_hand_args(args)
    motion_args = server_1_motion_args(args)

    assert args.speed == 8
    assert args.planar_ik_speed == 20
    assert motion_args.speed == 8
    assert motion_args.planar_ik_speed == 20


def test_eye_in_hand_planar_ik_keep_band_uses_fixed_grasp_orientation():
    args = build_parser().parse_args(
        [
            "--adaptive-joint-pitch-by-x",
            "--keep-pitch-min-x-mm",
            "300",
            "--keep-pitch-max-x-mm",
            "380",
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "350",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "0",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
        ]
    )
    validate_eye_in_hand_args(args)
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-keep-pitch",
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    payload = response["payload"]
    assert payload["dynamic_pitch"]["control_mode"] == "planar_ik"
    assert payload["dynamic_pitch"]["mode"] == "keep_fixed"
    pick_steps = [
        step
        for step in payload["steps"]
        if step["label"] in ("pick_above", "pick_grasp", "pick_lift")
    ]
    assert pick_steps
    assert all(not step["planar_ik"] for step in pick_steps)
    assert payload["target_orientation_deg"] == {
        "rx_deg": args.rx_deg,
        "ry_deg": args.ry_deg,
        "rz_deg": args.rz_deg,
    }
    assert "guarded_execution" not in payload


def test_eye_in_hand_planar_ik_far_x_uses_planar_ik_pick_steps():
    args = build_parser().parse_args(
        [
            "--adaptive-joint-pitch-by-x",
            "--planar-ik-speed",
            "22",
            "--keep-pitch-min-x-mm",
            "300",
            "--keep-pitch-max-x-mm",
            "380",
            "--max-target-x-mm",
            "450",
            "--grasp-only",
            "--no-return-ready",
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "420",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "0",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
        ]
    )
    validate_eye_in_hand_args(args)
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-far-planar-ik",
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    payload = response["payload"]
    assert payload["dynamic_pitch"]["mode"] == "adjust"
    assert payload["dynamic_pitch"]["method"] == "planar_ik_j2_j3_j5"
    assert payload["dynamic_pitch"]["planar_ik_speed"] == 22
    assert payload["dynamic_pitch"]["target_ry_deg"] != args.ry_deg
    by_label = {step["label"]: step for step in payload["steps"]}
    assert by_label["ready_open"]["planar_ik"] is False
    assert by_label["pick_above"]["planar_ik"] is True
    assert by_label["pick_grasp"]["planar_ik"] is True
    assert by_label["pick_lift"]["planar_ik"] is True
    assert all(
        not step["planar_ik"]
        for step in payload["steps"]
        if step["label"] in ("return_ready", "return_ready_above")
    )
    assert "pick_grasp_guard_start" not in by_label
    assert "guarded_execution" not in payload


def test_eye_in_hand_far_planar_ik_does_not_use_ik_for_initial_ready_return():
    args = build_parser().parse_args(
        [
            "--ready-pose-source",
            "initial",
            "--initial-pose-source",
            "configured",
            "--initial-pose-x-mm",
            "272.949",
            "--initial-pose-y-mm",
            "0",
            "--initial-pose-z-mm",
            "341.089",
            "--initial-pose-rx-deg",
            "-160.249",
            "--initial-pose-ry-deg",
            "5.355",
            "--initial-pose-rz-deg",
            "-104.572",
            "--adaptive-joint-pitch-by-x",
            "--keep-pitch-min-x-mm",
            "300",
            "--keep-pitch-max-x-mm",
            "380",
            "--max-target-x-mm",
            "450",
            "--grasp-only",
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "400",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "0",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
        ]
    )
    validate_eye_in_hand_args(args)
    resolve_ready_pose_override(args, None)
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-far-ready-return",
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    by_label = {step["label"]: step for step in response["payload"]["steps"]}
    assert by_label["pick_above"]["planar_ik"] is True
    assert by_label["pick_grasp"]["planar_ik"] is True
    assert by_label["pick_lift"]["planar_ik"] is True
    assert by_label["return_ready"]["planar_ik"] is False
    assert by_label["return_ready"]["pose"] == by_label["ready_open"]["pose"]


def test_eye_in_hand_use_detected_z_top_uses_command_top_z_plus_offset():
    args = build_parser().parse_args(
        [
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "350",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "300",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
            "--use-detected-z",
            "top",
            "--detected-z-offset-mm",
            "150",
            "--grasp-z",
            "180",
            "--min-grasp-z-mm",
            "180",
            "--max-grasp-z-mm",
            "310",
        ]
    )
    validate_eye_in_hand_args(args)
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-top-z",
                    detected_top_z_mm=40.0,
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    payload = response["payload"]
    assert payload["detected_z_mode"] == "top"
    assert payload["raw_grasp_z_mm"] == 190.0
    assert payload["grasp_z_mm"] == 190.0
    assert payload["grasp_z_source"] == "detected_top_z_plus_offset"
    assert payload["pose"]["z_mm"] == 190.0


def test_eye_in_hand_use_detected_z_top_maps_camera_top_point_when_base_top_missing():
    args = build_parser().parse_args(
        [
            "--capture-pose-source",
            "configured",
            "--capture-x-mm",
            "350",
            "--capture-y-mm",
            "0",
            "--capture-z-mm",
            "0",
            "--capture-rx-deg",
            "0",
            "--capture-ry-deg",
            "0",
            "--capture-rz-deg",
            "0",
            "--use-detected-z",
            "top",
            "--detected-z-offset-mm",
            "150",
            "--grasp-z",
            "180",
            "--min-grasp-z-mm",
            "180",
            "--max-grasp-z-mm",
            "310",
        ]
    )
    validate_eye_in_hand_args(args)
    camera_to_gripper = np.eye(4, dtype=float)
    validator = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        response = handle_message(
            make_camera_point_message(
                CameraPointCommand(
                    x_m=0.0,
                    y_m=0.0,
                    z_m=0.0,
                    command_id="cmd-camera-top-z",
                    detected_top_camera_x_m=0.01,
                    detected_top_camera_y_m=0.02,
                    detected_top_camera_z_m=0.04,
                )
            ),
            args=args,
            camera_to_gripper=camera_to_gripper,
            validator=validator,
            pose_reader=None,
            controller=None,
        )

    payload = response["payload"]
    assert payload["detected_z_mode"] == "top"
    assert payload["raw_grasp_z_mm"] == 190.0
    assert payload["grasp_z_mm"] == 190.0
    assert payload["grasp_z_source"] == "detected_top_z_plus_offset"
    assert payload["pose"]["z_mm"] == 190.0
    assert payload["eye_in_hand"]["mapped_top_from_camera"] == {
        "detected_top_x_mm": 360.0,
        "detected_top_y_mm": 20.0,
        "detected_top_z_mm": 40.0,
    }


def test_configured_initial_pose_requires_complete_pose():
    args = build_parser().parse_args(["--initial-pose-source", "configured"])

    try:
        validate_eye_in_hand_args(args)
    except ValueError as exc:
        assert "--initial-pose-source configured requires" in str(exc)
        assert "--initial-pose-x-mm" in str(exc)
    else:
        raise AssertionError("expected configured initial pose validation error")


def test_move_to_start_ready_uses_configured_initial_pose():
    args = build_parser().parse_args(
        [
            "--initial-pose-source",
            "configured",
            "--initial-pose-x-mm",
            "310",
            "--initial-pose-y-mm",
            "20",
            "--initial-pose-z-mm",
            "280",
            "--initial-pose-rx-deg",
            "-170",
            "--initial-pose-ry-deg",
            "2",
            "--initial-pose-rz-deg",
            "-100",
            "--command-duration-s",
            "0",
            "--no-wait-waypoints",
        ]
    )
    validate_eye_in_hand_args(args)
    controller = make_controller()

    with contextlib.redirect_stdout(io.StringIO()):
        count = move_to_start_ready(controller, args)

    assert count == 1
    assert controller.sdk.calls[-1] == (
        "EndPoseCtrl",
        (310000, 20000, 280000, -170000, 2000, -100000),
    )

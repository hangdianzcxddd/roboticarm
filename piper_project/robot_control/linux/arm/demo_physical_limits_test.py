from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_control.linux.arm.demo_physical_limits import (
    DEFAULT_PROFILE,
    DemoSafetyError,
    JOINT_NAMES,
    PhysicalLimitsDemo,
    RunOptions,
    build_selected_multi_scenario,
    load_profile,
    main,
)


class JointState:
    def __init__(self, degrees: tuple[float, ...]) -> None:
        for index, value in enumerate(degrees, start=1):
            setattr(self, f"joint_{index}", round(value * 1000))


class Feedback:
    def __init__(self, degrees: tuple[float, ...], stamp: float = 1.0) -> None:
        self.joint_state = JointState(degrees)
        self.time_stamp = stamp


class NormalStatus:
    class ArmStatus:
        arm_status = 0

    arm_status = ArmStatus()


class FakeController:
    def __init__(self, feedback: Feedback) -> None:
        self.feedback = feedback
        self.emergency_stop_calls = 0
        self.hold_calls: list[dict[str, object]] = []
        self.motion_modes: list[tuple[str, int]] = []
        self.moves: list[tuple[list[float], int]] = []

    def get_status(self):
        return NormalStatus()

    def get_enable_status(self):
        return (True,) * 6

    def get_joint_state(self):
        return self.feedback

    def validate_joints(self, joints):
        self.validated = list(joints)

    def set_motion_mode(self, move_mode, speed_percent):
        self.motion_modes.append((move_mode, speed_percent))

    def move_joints(self, joints, speed_percent):
        self.moves.append((list(joints), speed_percent))

    def send_joint_target(self, joints):
        self.moves.append((list(joints), self.motion_modes[-1][1]))

    def emergency_stop(self):
        self.emergency_stop_calls += 1

    def hold_current_position(self, **kwargs):
        self.hold_calls.append(kwargs)
        return tuple(math.radians(value / 1000) for value in (
            self.feedback.joint_state.joint_1,
            self.feedback.joint_state.joint_2,
            self.feedback.joint_state.joint_3,
            self.feedback.joint_state.joint_4,
            self.feedback.joint_state.joint_5,
            self.feedback.joint_state.joint_6,
        ))


def options() -> RunOptions:
    return RunOptions(
        speed=10,
        home_speed=10,
        tolerance_deg=2.0,
        timeout_s=1.0,
        feedback_timeout_s=1.0,
        max_range_speed=20,
        max_observed_speed_factor=1.2,
        abort_hold_speed=5,
        abort_hold_duration_s=2.0,
        target_command_interval_s=0.2,
        transition_step_deg=5.0,
    )


class TestProfile(unittest.TestCase):
    def test_default_j6_envelope_is_plus_minus_50_degrees(self):
        profile = load_profile(None)

        self.assertEqual(profile.demo_ranges_deg["j6"], (-50.0, 50.0))
        self.assertEqual(profile.safe_motion_ranges_deg["j6"], (-50.0, 50.0))

    def test_zero_home_is_allowed_by_transit_envelope_not_demo_boundary(self):
        profile = load_profile(None)

        self.assertEqual(profile.safe_home_deg, (0.0,) * 6)
        self.assertEqual(profile.demo_ranges_deg["j2"], (10.0, 170.0))
        self.assertEqual(profile.safe_motion_ranges_deg["j2"], (0.0, 170.0))

    def test_profile_rejects_speed_zone_outside_demo_range(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            text = str(DEFAULT_PROFILE).replace("'", '"').replace("True", "true")
            profile_path.write_text(text.replace('"j6": [-30.0, 30.0]', '"j6": [-51.0, 30.0]'))

            with self.assertRaisesRegex(ValueError, "speed_zones_deg.j6"):
                load_profile(profile_path)

    def test_profile_accepts_approved_multi_axis_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            document = dict(DEFAULT_PROFILE)
            document["multi_speed_scenarios"] = [
                {
                    "name": "j1_j4_j6_small",
                    "start_deg": [-20, 80, -80, -15, 0, -15],
                    "target_deg": [20, 80, -80, 15, 0, 15],
                }
            ]
            profile_path.write_text(json.dumps(document))

            profile = load_profile(profile_path)

        self.assertEqual(profile.multi_speed_scenarios[0]["name"], "j1_j4_j6_small")

    def test_generated_multi_axis_selection_moves_only_requested_joints(self):
        scenario = build_selected_multi_scenario(
            load_profile(None), ("j1", "j4", "j6"), "small", __import__("random").Random(7)
        )

        self.assertEqual(scenario["name"], "selected_j1_j4_j6_small")
        self.assertTrue(-15.0 <= scenario["start_deg"][0] <= -3.75)
        self.assertTrue(3.75 <= scenario["target_deg"][0] <= 15.0)
        self.assertEqual(scenario["start_deg"][1:3], (0.0, 0.0))
        self.assertEqual(scenario["target_deg"][4], 0.0)

    def test_normal_generated_multi_axis_selection_is_larger_than_small(self):
        profile = load_profile(None)
        rng = __import__("random").Random(7)
        small = build_selected_multi_scenario(profile, ("j2",), "small", rng)
        normal = build_selected_multi_scenario(profile, ("j2",), "normal", __import__("random").Random(7))

        self.assertLess(normal["start_deg"][1], small["start_deg"][1])
        self.assertGreater(normal["target_deg"][1], small["target_deg"][1])

    def test_same_seed_generates_the_same_route(self):
        profile = load_profile(None)
        first = build_selected_multi_scenario(profile, ("j1", "j6"), "small", __import__("random").Random(123))
        second = build_selected_multi_scenario(profile, ("j1", "j6"), "small", __import__("random").Random(123))

        self.assertEqual(first["start_deg"], second["start_deg"])
        self.assertEqual(first["target_deg"], second["target_deg"])


class TestRuntimeSafety(unittest.TestCase):
    def test_feedback_outside_safe_motion_envelope_aborts(self):
        profile = load_profile(None)
        controller = FakeController(Feedback((0.0, 0.0, 0.0, 0.0, 0.0, 51.0)))
        demo = PhysicalLimitsDemo(controller, profile, options())

        with self.assertRaisesRegex(DemoSafetyError, "j6 feedback"):
            demo.move_and_monitor((0.0,) * 6, speed=10, stage="test")

    def test_abort_holds_position_without_software_emergency_stop(self):
        profile = load_profile(None)
        controller = FakeController(Feedback((0.0,) * 6))
        demo = PhysicalLimitsDemo(controller, profile, options())

        held = demo.hold_after_abort("test abort")

        self.assertTrue(held)
        self.assertEqual(controller.emergency_stop_calls, 0)
        self.assertEqual(controller.hold_calls, [{"speed_percent": 5, "command_duration_s": 2.0, "feedback_timeout_s": 1.0}])

    def test_move_to_zero_uses_low_home_speed(self):
        profile = load_profile(None)
        controller = FakeController(Feedback((0.0,) * 6))
        demo = PhysicalLimitsDemo(controller, profile, options())

        demo.move_home("startup_zero_home")

        self.assertEqual(controller.moves[0][1], 10)
        self.assertEqual(controller.moves[0][0], [0.0] * 6)

    def test_range_test_moves_only_the_requested_joint(self):
        profile = load_profile(None)
        # Each completed target needs a matching feedback sample.  Update it in
        # the fake move method to behave like a responsive arm.
        controller = FakeController(Feedback((0.0,) * 6))

        def move(joints, speed_percent):
            controller.moves.append((list(joints), speed_percent))
            controller.feedback = Feedback(
                tuple(math.degrees(value) for value in joints),
                len(controller.moves),
            )

        controller.send_joint_target = lambda joints: move(joints, controller.motion_modes[-1][1])
        demo = PhysicalLimitsDemo(controller, profile, options())
        demo.run_range(("j6",))

        targets = [tuple(round(math.degrees(value)) for value in move[0]) for move in controller.moves]
        self.assertIn((0, 0, 0, 0, 0, -50), targets)
        self.assertIn((0, 0, 0, 0, 0, 50), targets)
        self.assertTrue(all(all(angle == 0 for angle in target[:5]) for target in targets))

    def test_multi_axis_motion_uses_approved_start_target_and_return(self):
        profile = load_profile(None)
        profile = profile.__class__(
            profile.safe_home_deg,
            profile.demo_ranges_deg,
            profile.safe_motion_ranges_deg,
            profile.speed_zones_deg,
            (
                {
                    "name": "small_combo",
                    "start_deg": (-20.0, 80.0, -80.0, -15.0, 0.0, -15.0),
                    "target_deg": (20.0, 80.0, -80.0, 15.0, 0.0, 15.0),
                },
            ),
        )
        controller = FakeController(Feedback((0.0,) * 6))

        def move(joints, speed_percent):
            controller.moves.append((list(joints), speed_percent))
            controller.feedback = Feedback(
                tuple(math.degrees(value) for value in joints), len(controller.moves) + 2
            )

        controller.send_joint_target = lambda joints: move(joints, controller.motion_modes[-1][1])
        demo = PhysicalLimitsDemo(controller, profile, options())
        demo.run_multi_speed(profile.multi_speed_scenarios)

        targets = [tuple(round(math.degrees(value)) for value in move[0]) for move in controller.moves]
        self.assertIn((-20, 80, -80, -15, 0, -15), targets)
        self.assertIn((20, 80, -80, 15, 0, 15), targets)
        self.assertEqual(targets[-1], (0, 0, 0, 0, 0, 0))


class TestDryRun(unittest.TestCase):
    @patch("robot_control.linux.arm.demo_physical_limits.PiperController")
    @patch("sys.argv", ["demo_physical_limits.py", "--joint", "j6"])
    def test_default_mode_does_not_construct_or_connect_controller(self, controller_type):
        self.assertEqual(main(), 0)

        controller_type.assert_not_called()

    @patch("robot_control.linux.arm.demo_physical_limits.PiperController")
    @patch(
        "sys.argv",
        [
            "demo_physical_limits.py",
            "--phase",
            "multi-speed",
            "--multi-joint",
            "j1",
            "j4",
            "j6",
            "--multi-size",
            "small",
        ],
    )
    def test_generated_multi_axis_dry_run_does_not_construct_controller(self, controller_type):
        self.assertEqual(main(), 0)

        controller_type.assert_not_called()

    @patch("robot_control.linux.arm.demo_physical_limits.PiperController")
    @patch(
        "sys.argv",
        ["demo_physical_limits.py", "--phase", "multi-speed", "--scenario", "not_in_profile", "--execute"],
    )
    def test_unknown_scenario_is_rejected_before_controller_is_constructed(self, controller_type):
        self.assertEqual(main(), 2)

        controller_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()

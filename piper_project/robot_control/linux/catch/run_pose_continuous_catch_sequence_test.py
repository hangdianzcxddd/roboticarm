from __future__ import annotations

import unittest

from robot_control.linux.catch.run_pose_continuous_catch_sequence import (
    build_parser,
    build_steps,
    validate_args,
)


def step_delay_by_label(label: str, argv: list[str] | None = None) -> float:
    args = build_parser().parse_args([] if argv is None else argv)
    validate_args(args)
    steps = build_steps(args)
    return next(step.delay_before_s for step in steps if step.label == label)


class TestRunPoseContinuousCatchSequence(unittest.TestCase):
    def test_default_delays_first_pick_and_place_pick_grasp(self):
        self.assertEqual(step_delay_by_label("pick1_grasp"), 5.0)
        self.assertEqual(step_delay_by_label("pick2_grasp"), 0.0)
        self.assertEqual(step_delay_by_label("place_pick_grasp"), 3.0)

    def test_delay_args_override_defaults(self):
        argv = [
            "--initial-pick-delay-s",
            "1.25",
            "--place-pick-delay-s",
            "0",
        ]

        self.assertEqual(step_delay_by_label("pick1_grasp", argv), 1.25)
        self.assertEqual(step_delay_by_label("place_pick_grasp", argv), 0.0)

    def test_rejects_negative_delay(self):
        args = build_parser().parse_args(["--initial-pick-delay-s", "-0.1"])

        with self.assertRaisesRegex(ValueError, "--initial-pick-delay-s"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()

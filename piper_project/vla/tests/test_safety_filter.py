import unittest

import numpy as np

from vla.lingbo_action_adapter import PiperAction
from vla.piper_safety_filter import SafetyFilterConfig, filter_action


class SafetyFilterTest(unittest.TestCase):
    def test_limits_single_step_and_gripper(self):
        result = filter_action(
            PiperAction(np.ones(6) * 10.0, -100.0),
            np.zeros(6),
            SafetyFilterConfig(
                max_joint_delta_rad=(0.1,) * 6,
                action_min_rad=(-1.0,) * 6,
                action_max_rad=(1.0,) * 6,
            ),
            current_gripper_mm=0.0,
        )
        np.testing.assert_allclose(result.joints_rad, np.ones(6) * 0.1)
        self.assertEqual(result.gripper_mm, -10.0)

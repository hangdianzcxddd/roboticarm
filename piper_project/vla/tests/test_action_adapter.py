import unittest

import numpy as np

from vla.lingbo_action_adapter import ActionAdapterConfig, adapt_action_row


class ActionAdapterTest(unittest.TestCase):
    def test_absolute_arm_and_mm_gripper(self):
        action = adapt_action_row(
            np.array([0.1, 0.2, -0.3, 0.4, 0.5, -0.6, -40.0]),
            np.zeros(6),
            ActionAdapterConfig(),
        )
        np.testing.assert_allclose(action.joints_rad, [0.1, 0.2, -0.3, 0.4, 0.5, -0.6])
        self.assertEqual(action.gripper_mm, -40.0)

    def test_delta_arm(self):
        action = adapt_action_row(
            np.ones(7) * 0.1,
            np.ones(6) * 0.5,
            ActionAdapterConfig(semantics="delta", gripper_unit="normalized"),
        )
        np.testing.assert_allclose(action.joints_rad, np.ones(6) * 0.6)
        self.assertAlmostEqual(action.gripper_mm, -10.0, places=6)

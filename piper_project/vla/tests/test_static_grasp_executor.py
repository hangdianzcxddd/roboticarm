import unittest
from unittest.mock import patch

import numpy as np

from vla.lingbo_action_adapter import ActionAdapterConfig
from vla.piper_safety_filter import SafetyFilterConfig
from vla.static_grasp_executor import StaticGraspExecutor


class FakeController:
    def __init__(self):
        self.joint_commands = []
        self.gripper_commands = []

    def move_joints(self, joints, speed_percent):
        self.joint_commands.append((tuple(joints), speed_percent))

    def set_gripper(self, value, effort_nm, enable):
        self.gripper_commands.append((value, effort_nm, enable))


class StaticExecutorContractTest(unittest.TestCase):
    @patch("vla.static_grasp_executor.read_piper_state")
    def test_executes_complete_chunk(self, read_state):
        read_state.return_value = (np.zeros(6, dtype=np.float32), 0.0, None)
        controller = FakeController()
        executor = StaticGraspExecutor(
            controller,
            SafetyFilterConfig(
                max_joint_delta_rad=(0.1,) * 6,
                action_min_rad=(-1.0,) * 6,
                action_max_rad=(1.0,) * 6,
            ),
            speed_percent=5,
            period_s=0.0,
            execute=True,
        )
        results = executor.execute_chunk(
            np.asarray([[0.2] * 6 + [-20.0], [0.3] * 6 + [-40.0]]),
            ActionAdapterConfig(),
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(len(controller.joint_commands), 2)
        self.assertEqual(len(controller.gripper_commands), 2)
        np.testing.assert_allclose(controller.joint_commands[0][0], [0.1] * 6)

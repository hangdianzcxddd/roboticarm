from __future__ import annotations

import math
import unittest

from config.robot_config import RobotConfig
from robot.piper_controller import EndPose, PiperController
from robot.safety import SafetyError


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
    def __init__(
        self,
        joints: tuple[int, int, int, int, int, int],
        time_stamp: float | None = None,
    ) -> None:
        self.joint_state = FakeJointState(joints)
        if time_stamp is not None:
            self.time_stamp = time_stamp


class FakePiperSdk:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.joint_feedbacks: list[FakeJointFeedback] = []
        self.joint_feedback = FakeJointFeedback((0, 0, 0, 0, 0, 0))

    def ConnectPort(self, can_init=False, piper_init=True, start_thread=True):
        self.calls.append(("ConnectPort", (can_init, piper_init, start_thread)))

    def DisconnectPort(self, thread_timeout=0.1):
        self.calls.append(("DisconnectPort", (thread_timeout,)))

    def EnablePiper(self):
        self.calls.append(("EnablePiper", ()))
        return True

    def DisablePiper(self):
        self.calls.append(("DisablePiper", ()))
        return False

    def EmergencyStop(self, emergency_stop=0):
        self.calls.append(("EmergencyStop", (emergency_stop,)))

    def MotionCtrl_2(
        self,
        ctrl_mode=0x01,
        move_mode=0x01,
        move_spd_rate_ctrl=50,
        is_mit_mode=0x00,
        residence_time=0,
        installation_pos=0x00,
    ):
        self.calls.append(
            (
                "MotionCtrl_2",
                (
                    ctrl_mode,
                    move_mode,
                    move_spd_rate_ctrl,
                    is_mit_mode,
                    residence_time,
                    installation_pos,
                ),
            )
        )

    def EndPoseCtrl(self, X, Y, Z, RX, RY, RZ):
        self.calls.append(("EndPoseCtrl", (X, Y, Z, RX, RY, RZ)))

    def JointCtrl(self, joint_1, joint_2, joint_3, joint_4, joint_5, joint_6):
        self.calls.append(
            ("JointCtrl", (joint_1, joint_2, joint_3, joint_4, joint_5, joint_6))
        )

    def GripperCtrl(
        self,
        gripper_angle=0,
        gripper_effort=0,
        gripper_code=0,
        set_zero=0,
    ):
        self.calls.append(
            ("GripperCtrl", (gripper_angle, gripper_effort, gripper_code, set_zero))
        )

    def GetArmStatus(self):
        return "status"

    def GetArmEndPoseMsgs(self):
        return "pose"

    def GetArmJointMsgs(self):
        if self.joint_feedbacks:
            return self.joint_feedbacks.pop(0)
        return self.joint_feedback

    def GetArmGripperMsgs(self):
        return "gripper"

    def SetSDKJointLimitParam(self, joint_name, min_val, max_val):
        self.calls.append(("SetSDKJointLimitParam", (joint_name, min_val, max_val)))

    def CrashProtectionConfig(
        self,
        joint_1_protection_level,
        joint_2_protection_level,
        joint_3_protection_level,
        joint_4_protection_level,
        joint_5_protection_level,
        joint_6_protection_level,
    ):
        self.calls.append(
            (
                "CrashProtectionConfig",
                (
                    joint_1_protection_level,
                    joint_2_protection_level,
                    joint_3_protection_level,
                    joint_4_protection_level,
                    joint_5_protection_level,
                    joint_6_protection_level,
                ),
            )
        )


def make_controller() -> tuple[PiperController, FakePiperSdk]:
    sdk = FakePiperSdk()
    config = RobotConfig(command_interval_s=0, connect_timeout_s=0.1)
    return PiperController(config=config, sdk=sdk), sdk


class TestMove(unittest.TestCase):
    def test_move_to_pose_uses_move_p_and_sdk_units(self):
        controller, sdk = make_controller()

        controller.move_to_pose(
            EndPose(150.1234, -50.0, 215.0, -179.9, 0.0, 90.5),
            speed_percent=30,
            move_mode="P",
        )

        self.assertEqual(
            sdk.calls,
            [
                ("MotionCtrl_2", (0x01, 0x00, 30, 0x00, 0, 0x00)),
                ("EndPoseCtrl", (150123, -50000, 215000, -179900, 0, 90500)),
            ],
        )

    def test_linear_pose_uses_move_l(self):
        controller, sdk = make_controller()

        controller.move_to_pose(
            EndPose(150.0, 50.0, 180.0, -180.0, 0.0, -180.0),
            speed_percent=25,
            move_mode="L",
        )

        self.assertEqual(sdk.calls[0], ("MotionCtrl_2", (0x01, 0x02, 25, 0x00, 0, 0x00)))

    def test_move_joints_converts_radians_to_milli_degrees(self):
        controller, sdk = make_controller()

        controller.move_joints([0.0, 1.0, -1.0, 0.5, -0.5, 0.25], speed_percent=40)

        self.assertEqual(
            sdk.calls,
            [
                ("MotionCtrl_2", (0x01, 0x01, 40, 0x00, 0, 0x00)),
                ("JointCtrl", (0, 57296, -57296, 28648, -28648, 14324)),
            ],
        )

    def test_move_joints_accepts_configured_full_joint_range(self):
        controller, sdk = make_controller()

        controller.move_joints(
            [
                math.radians(150.0),
                math.radians(180.0),
                math.radians(-170.0),
                math.radians(100.0),
                math.radians(70.0),
                math.radians(120.0),
            ],
            speed_percent=20,
        )

        self.assertEqual(
            sdk.calls[-1],
            ("JointCtrl", (150000, 180000, -170000, 100000, 70000, 120000)),
        )

    def test_joint_outside_configured_limit_is_rejected_before_sdk_call(self):
        controller, sdk = make_controller()

        with self.assertRaisesRegex(
            SafetyError,
            "Motion rejected: joint_6=.*outside the allowed range",
        ):
            controller.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(121.0)])

        self.assertEqual(sdk.calls, [])

    def test_pose_outside_safety_limits_is_rejected_before_sdk_call(self):
        controller, sdk = make_controller()

        with self.assertRaises(SafetyError):
            controller.move_to_pose(EndPose(0.0, 0.0, -1.0, 0.0, 0.0, 0.0))

        self.assertEqual(sdk.calls, [])

    def test_pose_outside_work_radius_is_rejected_before_sdk_call(self):
        controller, sdk = make_controller()

        with self.assertRaises(SafetyError):
            controller.move_to_pose(EndPose(626.75, 1.0, 100.0, 0.0, 0.0, 0.0))

        self.assertEqual(sdk.calls, [])

    def test_joint_count_is_validated(self):
        controller, sdk = make_controller()

        with self.assertRaises(SafetyError):
            controller.move_joints([0.0, 0.0, 0.0])

        self.assertEqual(sdk.calls, [])

    def test_connect_enables_arm_and_disconnect_closes_port(self):
        controller, sdk = make_controller()

        controller.connect()
        controller.disconnect()

        self.assertEqual(
            sdk.calls,
            [
                ("ConnectPort", (False, True, True)),
                ("EnablePiper", ()),
                ("DisconnectPort", (0.1,)),
            ],
        )

    def test_configure_sdk_limits_sends_configured_joint_limits(self):
        controller, sdk = make_controller()

        controller._configure_sdk_limits(sdk)

        self.assertEqual(len(sdk.calls), 6)
        self.assertEqual(sdk.calls[0][0], "SetSDKJointLimitParam")
        self.assertEqual(sdk.calls[0][1][0], "j1")
        self.assertAlmostEqual(sdk.calls[0][1][1], math.radians(-150.0))
        self.assertAlmostEqual(sdk.calls[0][1][2], math.radians(150.0))
        self.assertEqual(sdk.calls[-1][1][0], "j6")
        self.assertAlmostEqual(sdk.calls[-1][1][1], math.radians(-120.0))
        self.assertAlmostEqual(sdk.calls[-1][1][2], math.radians(120.0))

    def test_safe_disable_parks_before_disabling(self):
        config = RobotConfig(
            command_interval_s=0,
            connect_timeout_s=0.1,
            disable_park_joints_rad=(0.0, 0.0, -0.5, 0.0, 0.5, 0.0),
            disable_park_speed_percent=10,
            disable_park_settle_s=0,
        )
        sdk = FakePiperSdk()
        sdk.joint_feedbacks = [
            FakeJointFeedback((0, 0, 0, 0, 0, 0), time_stamp=1.0),
            FakeJointFeedback((0, 0, -28648, 0, 28648, 0), time_stamp=2.0),
        ]
        controller = PiperController(config=config, sdk=sdk)

        controller.safe_disable()

        self.assertEqual(
            sdk.calls,
            [
                ("MotionCtrl_2", (0x01, 0x01, 10, 0x00, 0, 0x00)),
                ("JointCtrl", (0, 0, -28648, 0, 28648, 0)),
                ("DisablePiper", ()),
            ],
        )

    def test_set_collision_protection_sends_valid_levels(self):
        controller, sdk = make_controller()

        controller.set_collision_protection([1, 2, 3, 4, 5, 6])

        self.assertEqual(sdk.calls, [("CrashProtectionConfig", (1, 2, 3, 4, 5, 6))])

    def test_set_collision_protection_rejects_invalid_levels(self):
        controller, sdk = make_controller()

        with self.assertRaises(ValueError):
            controller.set_collision_protection([1, 2, 3, 4, 5, 9])

        self.assertEqual(sdk.calls, [])


if __name__ == "__main__":
    unittest.main()

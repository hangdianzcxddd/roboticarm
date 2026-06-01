from __future__ import annotations

import unittest

from robot_control.linux.gripper.gripper_debug import calibration_summary, feedback_to_dict


class FakeFocStatus:
    voltage_too_low = False
    motor_overheating = False
    driver_overcurrent = False
    driver_overheating = False
    sensor_status = False
    driver_error_status = False
    driver_enable_status = True
    homing_status = True


class FakeGripperState:
    grippers_angle = 52345
    grippers_effort = 1200
    status_code = 0b11000000
    foc_status = FakeFocStatus()


class FakeGripperMessage:
    time_stamp = 12.5
    Hz = 200
    gripper_state = FakeGripperState()


class TestGripperDebug(unittest.TestCase):
    def test_feedback_to_dict_converts_sdk_units(self):
        feedback = feedback_to_dict(FakeGripperMessage())

        self.assertEqual(feedback["position_mm"], 52.345)
        self.assertEqual(feedback["effort_nm"], 1.2)
        self.assertEqual(feedback["status_code"], 0b11000000)
        self.assertEqual(feedback["error_code"], 0b11000000)
        self.assertTrue(feedback["foc_status"]["driver_enable_status"])
        self.assertTrue(feedback["foc_status"]["homing_status"])

    def test_calibration_summary_prefers_manual_measurements(self):
        summary = calibration_summary(
            [
                {
                    "actual_width_mm": 0.5,
                    "feedback_position_mm": 0.0,
                },
                {
                    "actual_width_mm": 63.2,
                    "feedback_position_mm": 70.0,
                },
            ]
        )

        self.assertEqual(
            summary,
            {
                "min_width": 0.5,
                "max_width": 63.2,
                "source": "actual_width_mm",
            },
        )

    def test_calibration_summary_uses_feedback_when_no_measurements(self):
        summary = calibration_summary(
            [
                {
                    "actual_width_mm": None,
                    "feedback_position_mm": 0.0,
                },
                {
                    "actual_width_mm": None,
                    "feedback_position_mm": 70.0,
                },
            ]
        )

        self.assertEqual(
            summary,
            {
                "min_width": 0.0,
                "max_width": 70.0,
                "source": "feedback_position_mm",
            },
        )


if __name__ == "__main__":
    unittest.main()

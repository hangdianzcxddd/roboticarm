from __future__ import annotations

import unittest

from robot_control.linux.gripper.diagnose_j6_gripper import (
    ARM_GRIPPER_CTRL_ID,
    ARM_GRIPPER_FEEDBACK_ID,
    J6_LOW_SPEED_FEEDBACK_ID,
    analyze_report,
    gripper_position_changed_across_stages,
    has_fresh_feedback,
    low_speed_motor_to_dict,
)


class FakeFocStatus:
    voltage_too_low = False
    motor_overheating = False
    driver_overcurrent = False
    driver_overheating = False
    collision_status = False
    driver_error_status = False
    driver_enable_status = True
    stall_status = False


class FakeLowSpeedMotor:
    can_id = J6_LOW_SPEED_FEEDBACK_ID
    vol = 245
    foc_temp = 31
    motor_temp = 29
    foc_status_code = 0b01000000
    bus_current = 500
    foc_status = FakeFocStatus()


def make_stage(
    *,
    gripper_fresh: bool = False,
    gripper_position_mm: float | None = None,
    gripper_error: bool = False,
    gripper_driver_enabled: bool = True,
    gripper_homed: bool = True,
    j6_can_id: int = J6_LOW_SPEED_FEEDBACK_ID,
) -> dict[str, object]:
    foc_status = {
        "voltage_too_low": False,
        "motor_overheating": False,
        "driver_overcurrent": False,
        "driver_overheating": False,
        "sensor_status": False,
        "driver_error_status": gripper_error,
        "driver_enable_status": gripper_driver_enabled,
        "homing_status": gripper_homed,
    }
    return {
        "fresh": {
            "status": True,
            "joint": True,
            "gripper": gripper_fresh,
        },
        "last": {
            "gripper": {
                "time_stamp": 10.0 if gripper_fresh else 0,
                "hz": 100 if gripper_fresh else 0,
                "position_mm": gripper_position_mm,
                "status_code": 0b01100000 if gripper_error else 0b11000000,
                "foc_status": foc_status,
            },
            "low_spd": {
                "motors": {
                    "6": {
                        "can_id": j6_can_id,
                        "can_id_hex": hex(j6_can_id),
                    }
                }
            },
        },
    }


class TestDiagnoseJ6Gripper(unittest.TestCase):
    def test_analyze_report_flags_j6_to_gripper_path_when_gripper_feedback_missing(self):
        report = {
            "stages": [make_stage(gripper_fresh=False, gripper_position_mm=None)],
            "commands": [
                {"label": "clear_error_enable", "ok": True},
                {"label": "open", "ok": True},
                {"label": "close", "ok": True},
            ],
            "can_monitor": {
                "started": True,
                "interesting_counts": {
                    hex(ARM_GRIPPER_CTRL_ID): 3,
                    hex(ARM_GRIPPER_FEEDBACK_ID): 0,
                    hex(J6_LOW_SPEED_FEEDBACK_ID): 10,
                },
            },
        }

        analysis = analyze_report(report)

        self.assertEqual(
            analysis["conclusion"], "j6_to_gripper_or_end_adapter_suspect"
        )
        self.assertTrue(analysis["arm_feedback_ok"])
        self.assertTrue(analysis["j6_feedback_ok"])
        self.assertFalse(analysis["gripper_feedback_ok"])

    def test_analyze_report_accepts_working_gripper_path(self):
        report = {
            "stages": [
                make_stage(gripper_fresh=True, gripper_position_mm=0.0),
                make_stage(gripper_fresh=True, gripper_position_mm=50.0),
            ],
            "commands": [
                {"label": "open", "ok": True},
                {"label": "close", "ok": True},
            ],
            "can_monitor": {
                "started": True,
                "interesting_counts": {
                    hex(ARM_GRIPPER_CTRL_ID): 2,
                    hex(ARM_GRIPPER_FEEDBACK_ID): 20,
                    hex(J6_LOW_SPEED_FEEDBACK_ID): 20,
                },
            },
        }

        analysis = analyze_report(report)

        self.assertEqual(analysis["conclusion"], "gripper_path_ok")
        self.assertTrue(analysis["gripper_position_changed"])

    def test_analyze_report_reports_gripper_error_bits(self):
        report = {
            "stages": [
                make_stage(
                    gripper_fresh=True,
                    gripper_position_mm=10.0,
                    gripper_error=True,
                )
            ],
            "commands": [{"label": "open", "ok": True}],
            "can_monitor": {
                "started": True,
                "interesting_counts": {
                    hex(ARM_GRIPPER_CTRL_ID): 1,
                    hex(ARM_GRIPPER_FEEDBACK_ID): 10,
                    hex(J6_LOW_SPEED_FEEDBACK_ID): 10,
                },
            },
        }

        analysis = analyze_report(report)

        self.assertEqual(
            analysis["conclusion"], "gripper_reports_driver_or_sensor_error"
        )
        self.assertTrue(analysis["gripper_has_error"])

    def test_analyze_report_reports_feedback_present_but_driver_not_enabled(self):
        report = {
            "stages": [
                make_stage(
                    gripper_fresh=True,
                    gripper_position_mm=0.0,
                    gripper_driver_enabled=False,
                    gripper_homed=False,
                ),
            ],
            "commands": [
                {"label": "clear_error_enable", "ok": True},
                {"label": "open", "ok": True},
            ],
            "can_monitor": {
                "started": True,
                "interesting_counts": {
                    hex(ARM_GRIPPER_CTRL_ID): 2,
                    hex(ARM_GRIPPER_FEEDBACK_ID): 20,
                    hex(J6_LOW_SPEED_FEEDBACK_ID): 20,
                },
            },
        }

        analysis = analyze_report(report)

        self.assertEqual(
            analysis["conclusion"],
            "gripper_feedback_present_but_driver_not_enabled",
        )
        self.assertFalse(analysis["gripper_driver_enabled"])

    def test_low_speed_motor_to_dict_converts_units_and_foc_status(self):
        payload = low_speed_motor_to_dict(FakeLowSpeedMotor())

        self.assertEqual(payload["can_id"], J6_LOW_SPEED_FEEDBACK_ID)
        self.assertEqual(payload["can_id_hex"], hex(J6_LOW_SPEED_FEEDBACK_ID))
        self.assertEqual(payload["voltage_v"], 24.5)
        self.assertEqual(payload["bus_current_a"], 0.5)
        self.assertTrue(payload["foc_status"]["driver_enable_status"])

    def test_feedback_freshness_and_position_change_helpers(self):
        self.assertTrue(has_fresh_feedback({"time_stamp": 1.0, "hz": 0}))
        self.assertTrue(has_fresh_feedback({"time_stamp": 0, "hz": 50}))
        self.assertFalse(has_fresh_feedback({"time_stamp": 0, "hz": 0}))
        self.assertTrue(
            gripper_position_changed_across_stages(
                [
                    make_stage(gripper_fresh=True, gripper_position_mm=12.0),
                    make_stage(gripper_fresh=True, gripper_position_mm=13.5),
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()

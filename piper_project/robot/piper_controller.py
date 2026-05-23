"""Thin, unit-safe wrapper around AgileX Piper SDK V2."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

from config.robot_config import MoveMode, RobotConfig, ROBOT_CONFIG
from robot.safety import SafetyChecker


SDK_DEG_FACTOR = 1000
RAD_TO_MILLI_DEG = 1000 * 180.0 / 3.141592653589793
MOVE_MODE_CODES: dict[MoveMode, int] = {"P": 0x00, "J": 0x01, "L": 0x02}


class PiperSdkLike(Protocol):
    def ConnectPort(
        self,
        can_init: bool = False,
        piper_init: bool = True,
        start_thread: bool = True,
    ) -> None: ...

    def DisconnectPort(self, thread_timeout: float = 0.1) -> None: ...

    def EnablePiper(self) -> bool: ...

    def DisablePiper(self) -> bool: ...

    def EmergencyStop(self, emergency_stop: int = 0) -> None: ...

    def MotionCtrl_2(
        self,
        ctrl_mode: int = 0x01,
        move_mode: int = 0x01,
        move_spd_rate_ctrl: int = 50,
        is_mit_mode: int = 0x00,
        residence_time: int = 0,
        installation_pos: int = 0x00,
    ) -> None: ...

    def EndPoseCtrl(
        self,
        X: int,
        Y: int,
        Z: int,
        RX: int,
        RY: int,
        RZ: int,
    ) -> None: ...

    def JointCtrl(
        self,
        joint_1: int,
        joint_2: int,
        joint_3: int,
        joint_4: int,
        joint_5: int,
        joint_6: int,
    ) -> None: ...

    def GripperCtrl(
        self,
        gripper_angle: int = 0,
        gripper_effort: int = 0,
        gripper_code: int = 0,
        set_zero: int = 0,
    ) -> None: ...

    def GetArmStatus(self) -> Any: ...

    def GetArmEndPoseMsgs(self) -> Any: ...

    def GetArmJointMsgs(self) -> Any: ...

    def GetArmGripperMsgs(self) -> Any: ...

    def SetSDKJointLimitParam(
        self,
        joint_name: str,
        min_val: float,
        max_val: float,
    ) -> None: ...


@dataclass(frozen=True)
class EndPose:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_deg: float
    ry_deg: float
    rz_deg: float

    def as_sdk_units(self) -> tuple[int, int, int, int, int, int]:
        return (
            round(self.x_mm * SDK_DEG_FACTOR),
            round(self.y_mm * SDK_DEG_FACTOR),
            round(self.z_mm * SDK_DEG_FACTOR),
            round(self.rx_deg * SDK_DEG_FACTOR),
            round(self.ry_deg * SDK_DEG_FACTOR),
            round(self.rz_deg * SDK_DEG_FACTOR),
        )


class PiperController:
    """Controller for the SDK segment: robot frame command -> Piper SDK -> motion."""

    def __init__(
        self,
        config: RobotConfig = ROBOT_CONFIG,
        sdk: PiperSdkLike | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._sdk = sdk
        self._logger = logger or logging.getLogger(__name__)
        self._safety = SafetyChecker(config)
        self._connected = False
        self._enabled = False

    @property
    def sdk(self) -> PiperSdkLike:
        if self._sdk is None:
            self._sdk = self._create_sdk()
        return self._sdk

    @staticmethod
    def _ensure_local_sdk_on_path() -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sdk_root = repo_root / "sdk" / "piper_sdk"
        if sdk_root.exists():
            sys.path.insert(0, str(sdk_root))

    def _create_sdk(self) -> PiperSdkLike:
        self._ensure_local_sdk_on_path()
        from piper_sdk import C_PiperInterface_V2

        sdk = C_PiperInterface_V2(
            self.config.can_name,
            judge_flag=self.config.judge_flag,
            can_auto_init=self.config.can_auto_init,
            start_sdk_joint_limit=self.config.start_sdk_joint_limit,
            start_sdk_gripper_limit=self.config.start_sdk_gripper_limit,
        )
        self._configure_sdk_limits(sdk)
        return sdk

    def _configure_sdk_limits(self, sdk: PiperSdkLike) -> None:
        if not self.config.start_sdk_joint_limit:
            return
        for index, limit in enumerate(self.config.joint_limits_rad, start=1):
            sdk.SetSDKJointLimitParam(f"j{index}", limit.min_value, limit.max_value)

    def connect(self, enable: bool | None = None) -> None:
        self.sdk.ConnectPort()
        self._connected = True
        if self.config.command_interval_s > 0:
            time.sleep(self.config.command_interval_s)
        should_enable = self.config.auto_enable if enable is None else enable
        if should_enable:
            self.enable()

    def disconnect(self, disable: bool = False) -> None:
        if disable and self._enabled:
            self.disable()
        if self._connected:
            self.sdk.DisconnectPort()
        self._connected = False

    def enable(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            if self.sdk.EnablePiper():
                self._enabled = True
                return
            time.sleep(self.config.command_interval_s)
        raise TimeoutError("Piper arm enable timed out")

    def disable(self) -> None:
        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            still_enabled = self.sdk.DisablePiper()
            if not still_enabled:
                self._enabled = False
                return
            time.sleep(self.config.command_interval_s)
        raise TimeoutError("Piper arm disable timed out")

    def emergency_stop(self) -> None:
        self.sdk.EmergencyStop(0x01)

    def resume_from_emergency_stop(self) -> None:
        self.sdk.EmergencyStop(0x02)

    def set_motion_mode(self, move_mode: MoveMode, speed_percent: int | None = None) -> None:
        speed = self.config.default_speed_percent if speed_percent is None else speed_percent
        self._safety.validate_speed(speed)
        self.sdk.MotionCtrl_2(0x01, MOVE_MODE_CODES[move_mode], speed, 0x00)

    def move_to_pose(
        self,
        pose: EndPose,
        speed_percent: int | None = None,
        move_mode: MoveMode | None = None,
    ) -> None:
        mode = self.config.default_move_mode if move_mode is None else move_mode
        if mode not in ("P", "L"):
            raise ValueError("Cartesian pose control supports MOVE P or MOVE L")
        self._safety.validate_pose_mm_deg(
            pose.x_mm,
            pose.y_mm,
            pose.z_mm,
            pose.rx_deg,
            pose.ry_deg,
            pose.rz_deg,
        )
        self.set_motion_mode(mode, speed_percent)
        self.sdk.EndPoseCtrl(*pose.as_sdk_units())

    def move_joints(
        self,
        joints_rad: tuple[float, float, float, float, float, float]
        | list[float],
        speed_percent: int | None = None,
    ) -> None:
        self._safety.validate_joints_rad(joints_rad)
        self.set_motion_mode("J", speed_percent)
        sdk_joints = tuple(round(value * RAD_TO_MILLI_DEG) for value in joints_rad)
        self.sdk.JointCtrl(*sdk_joints)

    def set_gripper(
        self,
        width_mm: float,
        effort_nm: float = 1.0,
        enable: bool = True,
        clear_error: bool = False,
    ) -> None:
        if width_mm < 0:
            raise ValueError("width_mm must be >= 0")
        if not 0 <= effort_nm <= 5:
            raise ValueError("effort_nm must be in [0, 5]")
        if clear_error:
            code = 0x03 if enable else 0x02
        else:
            code = 0x01 if enable else 0x00
        self.sdk.GripperCtrl(
            round(width_mm * SDK_DEG_FACTOR),
            round(effort_nm * SDK_DEG_FACTOR),
            code,
            0,
        )

    def get_status(self) -> Any:
        return self.sdk.GetArmStatus()

    def get_end_pose(self) -> Any:
        return self.sdk.GetArmEndPoseMsgs()

    def get_joint_state(self) -> Any:
        return self.sdk.GetArmJointMsgs()

    def get_gripper_state(self) -> Any:
        return self.sdk.GetArmGripperMsgs()

    def __enter__(self) -> "PiperController":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.emergency_stop()
        self.disconnect()

"""Cartesian YOLO-driven grasp server with object-oriented grasp poses.

This server receives ``CameraPointCommand`` messages from the Windows YOLO
client, maps the camera XYZ through the solved eye-to-hand transform, and runs a
direct J6 pose sequence modeled after ``run_pose_continuous_catch_sequence``:

``out`` task:
ready/open -> target above -> target grasp/close -> lift -> fixed place/open
-> return ready.

``in`` task:
ready/open -> selected target above -> selected target grasp/close -> lift
-> return ready/open.

Home, pregrasp, place and back-home keep the configured prepared orientation.
The object-short-edge correction can be performed either after reaching
``pick_above`` (the default) or after descending to ``pick_grasp``. The
resulting feedback orientation is inherited by the remaining grasp/lift steps
where appropriate. Every Cartesian waypoint is sent through Piper Cartesian
MOVE P; no joint-space waypoint is used. A YOLO bbox by itself is not enough
for 3D grasping; the Windows side should send the bbox-selected point after
RealSense depth deprojection as ``x_m/y_m/z_m``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket
import sys
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_control.linux.arm.kinematics import (
    euler_xyz_deg_from_rotation_matrix,
    forward_kinematics_pose,
    normalize_degrees,
)
from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG, PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import (
    feedback_to_pose,
    format_pose,
    format_status,
    pose_delta,
    pose_errors,
)
from robot_control.linux.catch.tcp_offset import rotation_matrix_xyz_deg
from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
    map_camera_point_to_base_xyz_m,
)
from robot_control.linux.config.robot_config import PoseLimit, ROBOT_CONFIG
from robot_control.shared.geometry import EndPose
from robot_control.shared.protocol import (
    CameraPointCommand,
    CameraPointTask,
    camera_point_from_message,
    make_ack_message,
    make_error_message,
    recv_json_line,
    send_json_line,
)
from robot_control.shared.simulated_mapping import pose_to_dict


DEFAULT_HAND_EYE_RESULT = (
    Path(__file__).resolve().parents[1] / "hand_eye" / "hand_eye_result.json"
)
Action = Literal["open", "close", None]
JointPitchMode = Literal["disabled", "keep_fixed", "adjust"]
DynamicPitchControlMode = Literal["joint", "planar_ik"]
GraspYawMode = Literal["integrated-planar-ik", "separate-j6"]
GraspYawTiming = Literal["pick_above", "pick_grasp"]
DEFAULT_GRASP_YAW_TIMING: GraspYawTiming = "pick_above"
DEFAULT_KEEP_PITCH_MIN_X_MM = 300.0
DEFAULT_KEEP_PITCH_MAX_X_MM = 380.0
DEFAULT_PITCH_DEG_PER_MM_FAR = 0.04
DEFAULT_PITCH_DEG_PER_MM_NEAR = 0.03
PLANAR_IK_JOINT_INDICES = (1, 2, 4)
PLANAR_IK_YAW_JOINT_INDICES = (1, 2, 3, 4, 5)
PLANAR_IK_YAW_DELTA_LIMIT_JOINT_INDICES = (3, 5)
PLANAR_IK_YAW_DELTA_LIMIT_DEG = 90.0

# ``catch_server_1_1`` uses Cartesian commands for every waypoint.  J1-J5
# temporarily rely on Piper's controller/mechanical limit positions instead of
# an additional software-angle rejection.  The Cartesian IK still stays inside
# the arm's absolute ranges so it cannot deliberately plan through a limit.
J6_LIMIT_DEG = 180.0
SOFTWARE_JOINT_LIMITS_DEG: tuple[tuple[float, float] | None, ...] = (
    None,  # J1: controller/mechanical limit only
    None,  # J2: controller/mechanical limit only
    None,  # J3: controller/mechanical limit only
    None,  # J4: controller/mechanical limit only
    None,  # J5: controller/mechanical limit only
    (-J6_LIMIT_DEG, J6_LIMIT_DEG),
)
DEFAULT_MAX_JACOBIAN_CONDITION = 500.0
DEFAULT_JOINT_LIMIT_STOP_MARGIN_DEG = 2.0
DEFAULT_MAX_CARTESIAN_JOINT_TRAVEL_DEG = (170.0, 170.0, 170.0, 170.0, 170.0, 120.0)
DEFAULT_MAX_JOINT_FEEDBACK_JUMP_DEG = 20.0
DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM = 82.0
DEFAULT_CARTESIAN_JOINT_TARGET_TOLERANCE_DEG = 20.0
DEFAULT_CARTESIAN_IK_LIMIT_RESIDUAL_POSITION_MM = 0.2
DEFAULT_CARTESIAN_IK_LIMIT_RESIDUAL_ORIENTATION_DEG = 0.2
# The three outbound grasp waypoints are the places where a different IK
# branch can make the difference between a valid and a J5-limit-clamped
# Cartesian target.  Keep the command mode Cartesian (MOVE P/L); this setting
# only changes the software pre-check and branch prediction.
MULTI_START_CARTESIAN_IK_STEP_LABELS = frozenset(
    ("pick_above", "pick_grasp", "pick_lift")
)
DEFAULT_MULTI_START_CARTESIAN_IK_SEED_COUNT = 4
MAX_MULTI_START_CARTESIAN_IK_SEED_COUNT = 8
PIPER_TARGET_POS_EXCEEDS_LIMIT = 0x04
DEFAULT_READY_JOINT_ANCHOR_TOLERANCE_DEG = 8.0
DEFAULT_READY_WRONG_WAY_TRAVEL_DEG = 10.0
ROLLBACK_MAX_JOINT_STEP_DEG = 5.0
READY_JOINT_ANCHOR_ATTR = "_catch_server_1_1_ready_joint_anchor"
READY_IK_JOINT_ANCHOR_ATTR = "_catch_server_1_1_ready_ik_joint_anchor"
READY_POSE_STEP_LABELS = frozenset(
    ("startup_ready", "ready_open", "return_ready", "return_ready_above", "final_release")
)


def _is_piper_target_limit_status(status: Any) -> bool:
    arm_status = getattr(status, "arm_status", None)
    try:
        return arm_status is not None and int(arm_status) == PIPER_TARGET_POS_EXCEEDS_LIMIT
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class SequenceStep:
    label: str
    pose: EndPose
    mode: Literal["P", "L"]
    gripper_after: Action = None
    delay_before_s: float = 0.0
    joint_pitch_before_pose: bool = False
    joint_pitch_after_pose: bool = False
    pitch_source_x_mm: float | None = None
    planar_ik: bool = False
    require_reached: bool = False
    planar_ik_include_j6_yaw: bool = False


@dataclass(frozen=True)
class YoloCatchTarget:
    camera_command: CameraPointCommand
    base_xyz_m: tuple[float, float, float]
    pick_x_mm: float
    pick_y_mm: float
    grasp_z_mm: float
    raw_pick_x_mm: float | None = None
    raw_pick_y_mm: float | None = None
    pick_x_source: str = "mapped_command_x"
    pick_y_source: str = "mapped_command_y"
    raw_grasp_z_mm: float | None = None
    grasp_z_source: str = "fixed_grasp_z"
    uncompensated_grasp_z_mm: float | None = None
    planar_ik_grasp_z_comp_mm: float = 0.0
    planar_ik_grasp_z_comp_pitch_delta_deg: float = 0.0
    vision_source: Literal["opencv"] = "opencv"
    pick_x_perspective_reference_mm: float | None = None
    pick_x_perspective_camera_z_mm: float | None = None
    pick_x_perspective_source_z_mm: float | None = None
    pick_x_perspective_target_z_mm: float | None = None
    pick_x_perspective_scale: float | None = None
    pick_x_perspective_compensation_mm: float = 0.0


@dataclass(frozen=True)
class YoloCatchPlan:
    task: CameraPointTask
    target: YoloCatchTarget
    steps: tuple[SequenceStep, ...]


@dataclass(frozen=True)
class DynamicTiming:
    t_arrive_s: float
    t_robot_s: float
    action_time_s: float
    t_wait_s: float
    t_robot_source: str = "constant"
    t_robot_source_x_mm: float | None = None
    t_robot_reference_x_mm: float | None = None
    t_robot_segment: str | None = None
    t_robot_slope_s_per_mm: float | None = None
    t_robot_base_s: float | None = None
    t_robot_unclamped_s: float | None = None
    t_robot_margin_s: float = 0.0
    t_robot_min_s: float | None = None
    t_robot_max_s: float | None = None
    dynamic_preposition: bool = False
    final_action_time_s: float | None = None
    wait_placement: Literal["wait", "nowait"] = "wait"
    too_close_skip_enabled: bool = True
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(frozen=True)
class SafeGripperResult:
    action: Action
    command_mm: float | None = None
    command_effort_nm: float | None = None
    max_effort_nm: float | None = None
    triggered: bool = False
    stopped: bool = False
    stop_action: str | None = None
    release_command_mm: float | None = None
    last_effort_nm: float | None = None
    last_position_mm: float | None = None
    poll_count: int = 0


@dataclass(frozen=True)
class JointPitchDecision:
    enabled: bool
    mode: JointPitchMode
    source_x_mm: float
    keep_min_x_mm: float
    keep_max_x_mm: float
    over_mm: float
    raw_pitch_delta_deg: float
    requested_pitch_delta_deg: float
    pitch_delta_deg: float
    clamped: bool
    limit_clamped: bool
    j2_ratio: float
    j3_ratio: float
    j5_ratio: float
    j2_delta_deg: float
    j3_delta_deg: float
    j5_delta_deg: float


@dataclass(frozen=True)
class JointPitchAdjustmentResult:
    step_label: str
    phase: str
    pick_x_mm: float
    pitch_source_x_mm: float
    raw_pitch_delta_deg: float
    pitch_delta_deg: float
    requested_pitch_delta_deg: float
    clamped: bool
    limit_clamped: bool
    j2_delta_deg: float
    j3_delta_deg: float
    j5_delta_deg: float
    start_joints_deg: tuple[float, float, float, float, float, float]
    target_joints_deg: tuple[float, float, float, float, float, float]
    send_count: int


@dataclass(frozen=True)
class GraspYawAdjustmentResult:
    enabled: bool
    applied: bool = False
    reason: str | None = None
    source_yaw_deg: float | None = None
    target_yaw_deg: float | None = None
    current_j6_deg: float | None = None
    delta_deg: float | None = None
    clamped: bool = False
    limit_clamped: bool = False
    start_joints_deg: tuple[float, float, float, float, float, float] | None = None
    target_joints_deg: tuple[float, float, float, float, float, float] | None = None
    send_count: int = 0


@dataclass(frozen=True)
class GraspYawPlan:
    raw_delta_deg: float
    delta_deg: float
    target_j6_deg: float
    axis_error_deg: float
    delta_clamped: bool
    limit_clamped: bool


@dataclass(frozen=True)
class PlanarIkExecutionResult:
    step_label: str
    pitch_source_x_mm: float
    target_pose: EndPose
    solved_pose: EndPose
    start_joints_deg: tuple[float, float, float, float, float, float]
    target_joints_deg: tuple[float, float, float, float, float, float]
    pitch_delta_deg: float
    requested_pitch_delta_deg: float
    iterations: int
    x_error_mm: float
    y_error_mm: float
    z_error_mm: float
    position_error_mm: float
    ry_error_deg: float
    send_count: int
    rx_error_deg: float = 0.0
    rz_error_deg: float = 0.0
    solved_fk_pose: EndPose | None = None
    final_joints_deg: tuple[float, float, float, float, float, float] | None = None
    joint_errors_deg: tuple[float, ...] | None = None
    max_joint_error_deg: float | None = None
    final_pose: EndPose | None = None
    reused: bool = False
    reused_from_step_label: str | None = None
    joint_indices: tuple[int, ...] = PLANAR_IK_JOINT_INDICES


@dataclass(frozen=True)
class PlanarIkSolution:
    target_joints_rad: tuple[float, float, float, float, float, float]
    solved_pose: EndPose
    solved_fk_pose: EndPose
    pitch_delta_deg: float
    requested_pitch_delta_deg: float
    iterations: int
    x_error_mm: float
    y_error_mm: float
    z_error_mm: float
    position_error_mm: float
    ry_error_deg: float
    rx_error_deg: float = 0.0
    rz_error_deg: float = 0.0
    joint_indices: tuple[int, ...] = PLANAR_IK_JOINT_INDICES


PlanarIkCacheKey = tuple[float, ...]


@dataclass(frozen=True)
class PlanarIkCachedTarget:
    step_label: str
    target_joints_rad: tuple[float, ...]
    result: PlanarIkExecutionResult


@dataclass
class PlanarIkReuseState:
    targets: dict[PlanarIkCacheKey, PlanarIkCachedTarget] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicExecutionResult:
    send_counts: dict[str, int]
    preposition_z_mm: float | None = None
    preposition_source: str | None = None
    preposition_elapsed_s: float | None = None
    final_action_due_s: float | None = None
    final_wait_s: float | None = None
    final_action_late_s: float | None = None
    safe_gripper_results: tuple[SafeGripperResult, ...] = ()
    joint_pitch_adjustments: tuple[JointPitchAdjustmentResult, ...] = ()
    grasp_yaw_adjustments: tuple[GraspYawAdjustmentResult, ...] = ()
    planar_ik_results: tuple[PlanarIkExecutionResult, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Receive YOLO camera XYZ over TCP and run a fixed-orientation "
            "pose-continuous out or in transfer sequence."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--can", default=ROBOT_CONFIG.can_name)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--speed", type=int, default=40)
    parser.add_argument(
        "--j6-speed",
        type=int,
        default=None,
        help=(
            "speed percent for J6-only grasp-yaw JointCtrl moves; when omitted, "
            "J6 uses --speed"
        ),
    )
    parser.add_argument(
        "--startup-speed",
        type=int,
        default=10,
        help=(
            "speed percent used only for the execute-server startup ready pose; "
            "--speed remains the normal catch/transfer speed"
        ),
    )
    parser.add_argument("--transfer-mode", choices=("P", "L"), default="P")
    parser.add_argument("--vertical-mode", choices=("P", "L"), default="L")
    parser.add_argument(
        "--lift-mode",
        choices=("P", "L"),
        default="L",
        help=(
            "motion mode for pick_lift/return_pick_lift after grasping; defaults "
            "to L to lift along the same Cartesian line used for descent"
        ),
    )
    parser.add_argument("--command-duration-s", type=float, default=1.0)
    parser.add_argument("--command-period-s", type=float, default=0.02)
    parser.add_argument("--motion-timeout-s", type=float, default=8.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=8.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--no-wait-waypoints", action="store_true")
    parser.add_argument("--print-feedback", action="store_true")
    parser.add_argument("--debug-plan", action="store_true")
    parser.add_argument(
        "--t-robot-s",
        type=float,
        default=1.7,
        help=(
            "dynamic mode measured robot reaction time from grasp command to "
            "gripper closed at the pick point"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-by-x",
        action="store_true",
        help=(
            "in dynamic mode, replace the constant --t-robot-s with a linear "
            "estimate based on raw pick X. With dynamic preposition this is "
            "used only by --dynamic-wait-placement nowait; wait schedules the "
            "final descent from --final-action-time-s"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-reference-x-mm",
        type=float,
        help=(
            "raw X where --t-robot-s is measured; defaults to --ready-x-mm"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-s-per-mm",
        type=float,
        default=0.0,
        help=(
            "default linear timing slope in seconds per raw-X millimeter when "
            "--dynamic-t-robot-by-x is enabled. Used for both sides unless "
            "the near/far side-specific slope is provided"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-planar-far-s-per-mm",
        type=float,
        default=0.0,
        help=(
            "X-based timing slope for raw X greater than "
            "--dynamic-t-robot-reference-x-mm; 0 uses --dynamic-t-robot-s-per-mm"
            "; used only by nowait when dynamic preposition is enabled"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-planar-near-s-per-mm",
        type=float,
        default=0.0,
        help=(
            "X-based timing slope for raw X smaller than "
            "--dynamic-t-robot-reference-x-mm; 0 uses --dynamic-t-robot-s-per-mm"
            "; used only by nowait when dynamic preposition is enabled"
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-margin-s",
        type=float,
        default=0.0,
        help="extra safety margin added after X-based dynamic robot timing",
    )
    parser.add_argument(
        "--dynamic-t-robot-min-s",
        type=float,
        help="optional lower clamp for X-based dynamic robot timing",
    )
    parser.add_argument(
        "--dynamic-t-robot-max-s",
        type=float,
        help="optional upper clamp for X-based dynamic robot timing",
    )
    parser.add_argument(
        "--dynamic-too-close-skip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "skip dynamic targets whose estimated arrival time is earlier than "
            "the robot action time; disabled by default, so late targets start "
            "immediately"
        ),
    )
    parser.add_argument(
        "--dynamic-preposition",
        action="store_true",
        help=(
            "in dynamic mode, move to ready/open and a target preposition pose "
            "immediately, then wait only for the final descent/close window"
        ),
    )
    parser.add_argument(
        "--dynamic-wait-placement",
        choices=("wait", "nowait"),
        default="wait",
        help=(
            "dynamic-preposition timing mode: wait keeps the existing wait at "
            "the target preposition pose; nowait waits at the initial ready pose "
            "then runs target approach and grasp without the final target wait"
        ),
    )
    parser.add_argument(
        "--dynamic-preposition-clearance-mm",
        type=float,
        default=20.0,
        help=(
            "dynamic preposition clearance above detected_z_mm; when detected Z "
            "is unavailable, falls back to grasp_z plus this clearance"
        ),
    )
    parser.add_argument(
        "--final-action-time-s",
        type=float,
        default=0.55,
        help=(
            "dynamic-preposition timing budget for the final descent and gripper "
            "close; this replaces --t-robot-s for the final wait calculation"
        ),
    )
    parser.add_argument(
        "--collision-protection-level",
        type=int,
        help=(
            "optional Piper SDK collision protection level applied to all six "
            "joints at startup; valid range is 0-8"
        ),
    )
    parser.add_argument(
        "--no-start-ready",
        action="store_true",
        help=(
            "do not move to the run_pose_continuous_catch_sequence ready pose "
            "when the execute server starts"
        ),
    )

    parser.add_argument(
        "--hand-eye-result",
        type=Path,
        default=DEFAULT_HAND_EYE_RESULT,
        help="eye_to_hand result whose transform is camera -> base",
    )
    parser.add_argument("--ready-x-mm", type=float, default=350.0)
    parser.add_argument("--ready-y-mm", type=float, default=0.0)
    parser.add_argument("--transfer-z-mm", type=float, default=253.252)
    parser.add_argument(
        "--pick-above-lift-z-gradient",
        type=float,
        default=1.0,
        help=(
            "lower pick_above and pick_lift Z by this many millimeters for "
            "each millimeter that pick X exceeds --ready-x-mm; default 1.0 "
            "mm/mm, and X at or behind ready X adds no Z offset"
        ),
    )
    parser.add_argument(
        "--grasp-z-mm",
        "--grasp-z",
        dest="grasp_z_mm",
        type=float,
        default=133.252,
        help="fixed J6 grasp Z, matching run_pose_continuous_catch_sequence style",
    )
    parser.add_argument(
        "--use-detected-z",
        nargs="?",
        const="center",
        default="off",
        choices=("center", "top"),
        help=(
            "use detected base Z for J6 grasp Z instead of --grasp-z-mm. "
            "Without a value, uses center for backward compatibility; top uses "
            "the Windows-provided object top Z. Default keeps a fixed grasp plane."
        ),
    )
    parser.add_argument(
        "--detected-z-offset-mm",
        type=float,
        default=0.0,
        help="extra J6 Z offset when --use-detected-z is enabled",
    )
    parser.add_argument("--rx-deg", type=float, default=-176.794223)
    parser.add_argument("--ry-deg", type=float, default=0.889438)
    parser.add_argument("--rz-deg", type=float, default=-105.490597)
    parser.add_argument(
        "--grasp-yaw-by-object-short-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "align the complete Cartesian grasp orientation to the object "
            "short-edge yaw reported by vision; outbound pick_lift returns to "
            "the pregrasp orientation; enabled by default in 1_1"
        ),
    )
    parser.add_argument(
        "--grasp-yaw-mode",
        choices=("integrated-planar-ik", "separate-j6"),
        default="integrated-planar-ik",
        help=(
            "object-short-edge yaw strategy when --grasp-yaw-by-object-short-edge "
            "is enabled. integrated-planar-ik keeps the existing J2/J3/J4/J5/J6 "
            "planar-IK yaw solve; separate-j6 rotates J6 first, then keeps planar "
            "IK on J2/J3/J5 only"
        ),
    )
    parser.add_argument(
        "--gripper-close-axis-offset-deg",
        type=float,
        default=0.0,
        help="calibration offset from reported object short-edge yaw to desired J6 angle",
    )
    parser.add_argument(
        "--grasp-yaw-max-delta-deg",
        type=float,
        default=45.0,
        help="maximum absolute J6 yaw correction applied per grasp",
    )
    parser.add_argument(
        "--grasp-yaw-deadband-deg",
        type=float,
        default=5.0,
        help="skip J6 yaw correction when the required delta is within this band",
    )
    parser.add_argument(
        "--grasp-yaw-timing",
        "--gripper-rotation-timing",
        dest="grasp_yaw_timing",
        choices=("pick_above", "pick_grasp"),
        default=DEFAULT_GRASP_YAW_TIMING,
        help=(
            "when to apply the object-short-edge J6 rotation: pick_above "
            "rotates at the safe height before descent (default), while "
            "pick_grasp descends with the prepared orientation, rotates at "
            "the grasp height, then closes the gripper"
        ),
    )
    parser.add_argument(
        "--grasp-yaw-speed",
        type=int,
        default=None,
        help=(
            "legacy alias for --j6-speed; when neither option is supplied, "
            "J6 uses --speed"
        ),
    )
    parser.add_argument(
        "--force-pick-x-mm",
        type=float,
        help=(
            "force the Cartesian pick X coordinate while keeping the raw mapped "
            "or dynamic-command X for dynamic joint pitch calculation"
        ),
    )
    parser.add_argument(
        "--pick-x-perspective-reference-mm",
        "--pick-x-reference-mm",
        "--pick-x-offset-mm",
        type=float,
        help=(
            "enable perspective X compensation using this base-frame X as the "
            "zero-error reference (for example 300). The old "
            "--pick-x-offset-mm spelling is retained as an alias but now also "
            "expects this reference X, not a fixed offset. The conveyor plane is "
            "assumed to be base Z=0; camera Z comes from the hand-eye transform "
            "and target Z comes from vision. The corrected X contracts toward "
            "the reference as target height and distance from the reference grow"
        ),
    )
    parser.add_argument(
        "--force-pick-y-mm",
        type=float,
        help=(
            "force the final pick line Y coordinate; use 0 for a y=0 catch line "
            "while still using vision/dynamic timing for X"
        ),
    )
    parser.add_argument(
        "--dynamic-pitch-by-x",
        "--adaptive-joint-pitch-by-x",
        action="store_true",
        dest="dynamic_pitch_by_x",
        help=(
            "after reaching pick waypoints outside the keep-pitch X band, apply "
            "a joint-space pitch trim using only J2/J3/J5; Cartesian RX/RY/RZ "
            "stay fixed"
        ),
    )
    parser.add_argument(
        "--dynamic-pitch-mode",
        choices=("joint", "planar-ik"),
        default="joint",
        help=(
            "joint keeps the legacy J2/J3/J5 trim after Cartesian waypoints; "
            "planar-ik normally fixes J1/J4/J6 and solves J2/J3/J5 for target "
            "X/Z/RY; with object-short-edge yaw it solves J2/J3/J4/J5/J6 "
            "for X/Z/RX/RY/RZ"
        ),
    )
    parser.add_argument("--planar-ik-max-iterations", type=int, default=40)
    parser.add_argument("--planar-ik-position-tolerance-mm", type=float, default=3.0)
    parser.add_argument(
        "--planar-ik-rx-tolerance-deg",
        type=float,
        default=5.0,
        help="RX tolerance for planar IK when J4 wrist roll is included",
    )
    parser.add_argument("--planar-ik-ry-tolerance-deg", type=float, default=1.0)
    parser.add_argument(
        "--planar-ik-rz-tolerance-deg",
        type=float,
        default=3.0,
        help="RZ tolerance for planar IK when J6 yaw is included",
    )
    parser.add_argument("--planar-ik-max-step-deg", type=float, default=8.0)
    parser.add_argument("--planar-ik-damping", type=float, default=1e-3)
    parser.add_argument(
        "--planar-ik-include-j6-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "when object-short-edge yaw is enabled, solve planar IK with "
            "J2/J3/J4/J5/J6 for X/Z/RX/RY/RZ instead of rotating J6 before the "
            "planar-IK descent"
        ),
    )
    parser.add_argument(
        "--planar-ik-speed",
        type=int,
        help=(
            "speed percent for planar-IK JointCtrl moves; defaults to --speed "
            "when omitted"
        ),
    )
    parser.add_argument(
        "--planar-ik-grasp-z-comp-mm-per-deg",
        type=float,
        default=0.0,
        help=(
            "lower only far planar-IK grasp Z by abs(pitch_delta_deg) times "
            "this coefficient; default 0 disables pitch-based height compensation"
        ),
    )
    parser.add_argument(
        "--reuse-planar-ik-above-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse a previously solved planar-IK joint target for repeated "
            "same-pose above/lift steps; enabled by default"
        ),
    )
    parser.add_argument(
        "--planar-ik-joint-tolerance-deg",
        type=float,
        default=2.0,
        help="joint feedback tolerance for J2/J3/J5 planar-IK waypoints",
    )
    parser.add_argument(
        "--cartesian-ik-fk-tool-z-offset-mm",
        type=float,
        default=DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
        help=(
            "tool-frame Z offset added only to the raw d6=9 mm DH model while "
            "checking Cartesian MOVE P IK; Piper SDK EndPose uses d6=91 mm, "
            "so the default compensation is 82 mm"
        ),
    )
    parser.add_argument(
        "--multi-start-cartesian-ik-seed-count",
        type=int,
        default=DEFAULT_MULTI_START_CARTESIAN_IK_SEED_COUNT,
        help=(
            "number of deterministic software IK seeds tried for pick_above, "
            "pick_grasp and pick_lift; valid range is 1 to 8, default 4. "
            "This does not change the Cartesian MOVE P/L command mode"
        ),
    )
    parser.add_argument(
        "--cartesian-ik-joint-target-tolerance-deg",
        type=float,
        default=DEFAULT_CARTESIAN_JOINT_TARGET_TOLERANCE_DEG,
        help=(
            "diagnostic per-joint difference between software IK and the "
            "firmware-selected Cartesian IK solution; a larger difference is "
            "accepted only when corrected FK still reaches the commanded pose"
        ),
    )
    parser.add_argument(
        "--planar-ik-fk-tool-z-offset-mm",
        type=float,
        default=82.0,
        help=(
            "tool-frame Z offset from the raw DH FK frame to the Piper SDK "
            "EndPose frame used by planar IK"
        ),
    )
    parser.add_argument(
        "--pitch-x-ref-mm",
        type=float,
        default=350.0,
        help=(
            "legacy dynamic pitch neutral X; kept for old --pitch-deg-per-mm "
            "setups but not used by adaptive keep-band pitch"
        ),
    )
    parser.add_argument(
        "--pitch-deg-per-mm",
        type=float,
        default=None,
        help=(
            "legacy single slope; when provided, sets both "
            "--pitch-deg-per-mm-near and --pitch-deg-per-mm-far"
        ),
    )
    parser.add_argument(
        "--keep-pitch-min-x-mm",
        type=float,
        default=DEFAULT_KEEP_PITCH_MIN_X_MM,
        help="minimum X in the fixed-gripper-pitch keep band",
    )
    parser.add_argument(
        "--keep-pitch-max-x-mm",
        type=float,
        default=DEFAULT_KEEP_PITCH_MAX_X_MM,
        help="maximum X in the fixed-gripper-pitch keep band",
    )
    parser.add_argument(
        "--pitch-deg-per-mm-far",
        type=float,
        default=None,
        help=(
            "joint-space pitch degrees per millimeter beyond "
            "--keep-pitch-max-x-mm"
        ),
    )
    parser.add_argument(
        "--pitch-deg-per-mm-near",
        type=float,
        default=None,
        help=(
            "joint-space pitch degrees per millimeter below "
            "--keep-pitch-min-x-mm; over_mm is negative on the near side"
        ),
    )
    parser.add_argument(
        "--min-dynamic-pitch-deg",
        "--min-required-pitch-deg",
        "--min-dynamic-ry-deg",
        dest="min_dynamic_pitch_deg",
        type=float,
        default=-12.0,
    )
    parser.add_argument(
        "--max-dynamic-pitch-deg",
        "--max-required-pitch-deg",
        "--max-dynamic-ry-deg",
        dest="max_dynamic_pitch_deg",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--dynamic-pitch-j2-ratio",
        type=float,
        default=0.0,
        help="share of dynamic pitch delta applied to J2",
    )
    parser.add_argument(
        "--dynamic-pitch-j3-ratio",
        type=float,
        default=0.0,
        help="share of dynamic pitch delta applied to J3",
    )
    parser.add_argument(
        "--dynamic-pitch-j5-ratio",
        type=float,
        default=1.0,
        help="share of dynamic pitch delta applied to J5",
    )
    parser.add_argument(
        "--dynamic-pitch-speed",
        type=int,
        default=20,
        help="speed percent for the J2/J3/J5 dynamic pitch trim",
    )
    parser.add_argument(
        "--pick-delay-s",
        type=float,
        default=0.0,
        help="pause before descending from target-above to target-grasp",
    )
    parser.add_argument("--place-x-mm", type=float, default=60.0)
    parser.add_argument("--place-y-mm", type=float, default=270.0)
    parser.add_argument(
        "--place-pick-x-mm",
        type=float,
        default=60.0,
        help="X used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-y-mm",
        type=float,
        default=-270.0,
        help="Y used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-z-mm",
        type=float,
        help="Z used when picking the object back from the fixed place area",
    )
    parser.add_argument(
        "--place-pick-delay-s",
        type=float,
        default=3.0,
        help="pause at ready before moving to the fixed place-pick point",
    )
    parser.add_argument(
        "--grasp-only",
        action="store_true",
        help="legacy mode: pick and lift the YOLO target without fixed place/pick-back",
    )
    parser.add_argument(
        "--no-return-ready",
        action="store_true",
        help=(
            "for task=out, stop at the task end point instead of returning to "
            "the ready pose"
        ),
    )

    parser.add_argument("--min-target-x-mm", type=float, default=80.0)
    parser.add_argument("--max-target-x-mm", type=float)
    parser.add_argument("--min-target-y-mm", type=float, default=-300.0)
    parser.add_argument("--max-target-y-mm", type=float, default=300.0)
    parser.add_argument("--min-grasp-z-mm", type=float)
    parser.add_argument("--max-grasp-z-mm", type=float)

    parser.add_argument("--open-command-mm", type=float, default=0.0)
    parser.add_argument("--grasp-command-mm", type=float, default=-100.0)
    parser.add_argument("--effort-nm", type=float, default=1.0)
    parser.add_argument(
        "--grasp-effort-nm",
        type=float,
        help="gripper effort used only for close commands; defaults to --effort-nm",
    )
    parser.add_argument(
        "--max-gripper-effort-nm",
        type=float,
        default=0.8,
        help="feedback effort threshold for --safe-gripper-stop-on-effort",
    )
    parser.add_argument(
        "--safe-gripper-stop-on-effort",
        action="store_true",
        help=(
            "after close, poll gripper feedback and stop/release when "
            "grippers_effort exceeds --max-gripper-effort-nm"
        ),
    )
    parser.add_argument(
        "--safe-gripper-monitor-s",
        type=float,
        default=0.35,
        help="seconds to monitor gripper feedback after a close command",
    )
    parser.add_argument(
        "--safe-gripper-poll-s",
        type=float,
        default=0.02,
        help="poll period for --safe-gripper-stop-on-effort",
    )
    parser.add_argument(
        "--safe-gripper-release-mm",
        type=float,
        help=(
            "optional gripper command after an effort stop; defaults to disabling "
            "the gripper at the current open command"
        ),
    )
    parser.add_argument(
        "--grasp-settle-s",
        type=float,
        default=0.0,
        help=(
            "pause after closing the gripper at pick_grasp before starting the "
            "lift; default 0 sends the lift immediately after the close command"
        ),
    )
    parser.add_argument(
        "--gripper-settle-s",
        type=float,
        default=0.6,
        help="pause after opening the gripper; close uses --grasp-settle-s",
    )
    parser.add_argument("--clear-gripper-error", action="store_true")
    parser.add_argument("--no-pre-open", action="store_true")
    parser.add_argument("--no-gripper", action="store_true")
    return parser


def normalize_dynamic_pitch_args(args: argparse.Namespace) -> None:
    if not hasattr(args, "keep_pitch_min_x_mm"):
        args.keep_pitch_min_x_mm = DEFAULT_KEEP_PITCH_MIN_X_MM
    if not hasattr(args, "keep_pitch_max_x_mm"):
        args.keep_pitch_max_x_mm = DEFAULT_KEEP_PITCH_MAX_X_MM
    legacy_slope = getattr(args, "pitch_deg_per_mm", None)
    if getattr(args, "pitch_deg_per_mm_far", None) is None:
        args.pitch_deg_per_mm_far = (
            float(legacy_slope)
            if legacy_slope is not None
            else DEFAULT_PITCH_DEG_PER_MM_FAR
        )
    if getattr(args, "pitch_deg_per_mm_near", None) is None:
        args.pitch_deg_per_mm_near = (
            float(legacy_slope)
            if legacy_slope is not None
            else DEFAULT_PITCH_DEG_PER_MM_NEAR
        )
    if legacy_slope is None:
        # Keep legacy summaries/tests from seeing None after validation.
        args.pitch_deg_per_mm = float(args.pitch_deg_per_mm_far)


def validate_args(args: argparse.Namespace) -> None:
    normalize_dynamic_pitch_args(args)
    grasp_yaw_timing_value(args)
    if not isinstance(args.speed, int) or not 0 <= args.speed <= 100:
        raise ValueError("--speed must be an integer in [0, 100]")
    for name in ("j6_speed", "grasp_yaw_speed"):
        value = getattr(args, name, None)
        if value is not None and (
            not isinstance(value, int) or not 0 <= value <= 100
        ):
            raise ValueError(f"--{name.replace('_', '-')} must be an integer in [0, 100]")
    if not isinstance(args.startup_speed, int) or not 0 <= args.startup_speed <= 100:
        raise ValueError("--startup-speed must be an integer in [0, 100]")
    if args.planar_ik_speed is not None and (
        not isinstance(args.planar_ik_speed, int) or not 0 <= args.planar_ik_speed <= 100
    ):
        raise ValueError("--planar-ik-speed must be an integer in [0, 100]")
    for name in (
        "command_duration_s",
        "motion_timeout_s",
        "position_tolerance_mm",
        "orientation_tolerance_deg",
        "t_robot_s",
        "dynamic_t_robot_margin_s",
        "final_action_time_s",
        "pick_delay_s",
        "place_pick_delay_s",
        "safe_gripper_monitor_s",
        "grasp_settle_s",
        "gripper_settle_s",
        "planar_ik_position_tolerance_mm",
        "planar_ik_rx_tolerance_deg",
        "planar_ik_ry_tolerance_deg",
        "planar_ik_rz_tolerance_deg",
        "planar_ik_max_step_deg",
        "planar_ik_damping",
        "planar_ik_grasp_z_comp_mm_per_deg",
        "planar_ik_joint_tolerance_deg",
        "planar_ik_fk_tool_z_offset_mm",
        "cartesian_ik_fk_tool_z_offset_mm",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not isinstance(args.planar_ik_max_iterations, int) or args.planar_ik_max_iterations < 1:
        raise ValueError("--planar-ik-max-iterations must be an integer >= 1")
    if args.planar_ik_position_tolerance_mm <= 0.0:
        raise ValueError("--planar-ik-position-tolerance-mm must be > 0")
    if args.planar_ik_rx_tolerance_deg <= 0.0:
        raise ValueError("--planar-ik-rx-tolerance-deg must be > 0")
    if args.planar_ik_ry_tolerance_deg <= 0.0:
        raise ValueError("--planar-ik-ry-tolerance-deg must be > 0")
    if args.planar_ik_rz_tolerance_deg <= 0.0:
        raise ValueError("--planar-ik-rz-tolerance-deg must be > 0")
    if args.planar_ik_max_step_deg <= 0.0:
        raise ValueError("--planar-ik-max-step-deg must be > 0")
    if args.planar_ik_damping <= 0.0:
        raise ValueError("--planar-ik-damping must be > 0")
    if args.planar_ik_joint_tolerance_deg <= 0.0:
        raise ValueError("--planar-ik-joint-tolerance-deg must be > 0")
    if (
        not math.isfinite(float(args.cartesian_ik_joint_target_tolerance_deg))
        or float(args.cartesian_ik_joint_target_tolerance_deg) <= 0.0
    ):
        raise ValueError(
            "--cartesian-ik-joint-target-tolerance-deg must be finite and > 0"
        )
    if (
        not isinstance(args.multi_start_cartesian_ik_seed_count, int)
        or isinstance(args.multi_start_cartesian_ik_seed_count, bool)
        or not 1
        <= args.multi_start_cartesian_ik_seed_count
        <= MAX_MULTI_START_CARTESIAN_IK_SEED_COUNT
    ):
        raise ValueError(
            "--multi-start-cartesian-ik-seed-count must be an integer in [1, 8]"
        )
    if not math.isfinite(args.command_period_s) or args.command_period_s <= 0:
        raise ValueError("--command-period-s must be finite and > 0")
    if not math.isfinite(args.safe_gripper_poll_s) or args.safe_gripper_poll_s <= 0:
        raise ValueError("--safe-gripper-poll-s must be finite and > 0")
    if (
        not isinstance(args.dynamic_pitch_speed, int)
        or not 0 <= args.dynamic_pitch_speed <= 100
    ):
        raise ValueError("--dynamic-pitch-speed must be an integer in [0, 100]")
    for name in (
        "dynamic_preposition_clearance_mm",
        "pick_above_lift_z_gradient",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if args.dynamic_preposition_clearance_mm <= 0.0:
        raise ValueError("--dynamic-preposition-clearance-mm must be > 0")
    if args.collision_protection_level is not None and (
        not isinstance(args.collision_protection_level, int)
        or not 0 <= args.collision_protection_level <= 8
    ):
        raise ValueError("--collision-protection-level must be an integer in [0, 8]")
    if args.grasp_z_mm > args.transfer_z_mm:
        raise ValueError("--grasp-z-mm should not be above --transfer-z-mm")
    if args.place_pick_z_mm is not None and args.place_pick_z_mm > args.transfer_z_mm:
        raise ValueError("--place-pick-z-mm should not be above --transfer-z-mm")
    for option_name, value in (
        ("--effort-nm", args.effort_nm),
        ("--grasp-effort-nm", args.grasp_effort_nm),
        ("--max-gripper-effort-nm", args.max_gripper_effort_nm),
    ):
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 5):
            raise ValueError(f"{option_name} must be finite and in [0, 5]")

    for name in (
        "ready_x_mm",
        "ready_y_mm",
        "transfer_z_mm",
        "pick_above_lift_z_gradient",
        "grasp_z_mm",
        "detected_z_offset_mm",
        "place_x_mm",
        "place_y_mm",
        "place_pick_x_mm",
        "place_pick_y_mm",
        "place_pick_z_mm",
        "rx_deg",
        "ry_deg",
        "rz_deg",
        "gripper_close_axis_offset_deg",
        "grasp_yaw_max_delta_deg",
        "grasp_yaw_deadband_deg",
        "force_pick_x_mm",
        "pick_x_perspective_reference_mm",
        "force_pick_y_mm",
        "dynamic_t_robot_reference_x_mm",
        "dynamic_t_robot_s_per_mm",
        "dynamic_t_robot_planar_far_s_per_mm",
        "dynamic_t_robot_planar_near_s_per_mm",
        "dynamic_t_robot_min_s",
        "dynamic_t_robot_max_s",
        "pitch_x_ref_mm",
        "pitch_deg_per_mm",
        "keep_pitch_min_x_mm",
        "keep_pitch_max_x_mm",
        "pitch_deg_per_mm_far",
        "pitch_deg_per_mm_near",
        "min_dynamic_pitch_deg",
        "max_dynamic_pitch_deg",
        "dynamic_pitch_j2_ratio",
        "dynamic_pitch_j3_ratio",
        "dynamic_pitch_j5_ratio",
        "min_target_x_mm",
        "max_target_x_mm",
        "min_target_y_mm",
        "max_target_y_mm",
        "min_grasp_z_mm",
        "max_grasp_z_mm",
        "open_command_mm",
        "grasp_command_mm",
        "safe_gripper_release_mm",
    ):
        value = getattr(args, name)
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if args.planar_ik_grasp_z_comp_mm_per_deg < 0.0:
        raise ValueError("--planar-ik-grasp-z-comp-mm-per-deg must be >= 0")
    if args.dynamic_t_robot_margin_s < 0.0:
        raise ValueError("--dynamic-t-robot-margin-s must be >= 0")
    if (
        args.dynamic_t_robot_min_s is not None
        and args.dynamic_t_robot_max_s is not None
        and args.dynamic_t_robot_min_s > args.dynamic_t_robot_max_s
    ):
        raise ValueError("--dynamic-t-robot-min-s must be <= --dynamic-t-robot-max-s")
    if args.grasp_yaw_max_delta_deg < 0.0:
        raise ValueError("--grasp-yaw-max-delta-deg must be >= 0")
    if args.grasp_yaw_deadband_deg < 0.0:
        raise ValueError("--grasp-yaw-deadband-deg must be >= 0")
    if args.min_dynamic_pitch_deg > args.max_dynamic_pitch_deg:
        raise ValueError(
            "--min-dynamic-pitch-deg must be <= --max-dynamic-pitch-deg"
        )
    if args.keep_pitch_min_x_mm > args.keep_pitch_max_x_mm:
        raise ValueError(
            "--keep-pitch-min-x-mm must be <= --keep-pitch-max-x-mm"
        )
    if (
        args.dynamic_pitch_by_x
        and args.dynamic_pitch_j2_ratio == 0.0
        and args.dynamic_pitch_j3_ratio == 0.0
        and args.dynamic_pitch_j5_ratio == 0.0
    ):
        raise ValueError(
            "at least one dynamic pitch joint ratio among J2/J3/J5 must be non-zero"
        )
    for lower_name, upper_name in (
        ("min_target_x_mm", "max_target_x_mm"),
        ("min_target_y_mm", "max_target_y_mm"),
        ("min_grasp_z_mm", "max_grasp_z_mm"),
    ):
        lower = getattr(args, lower_name)
        upper = getattr(args, upper_name)
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"--{lower_name.replace('_', '-')} must be <= "
                f"--{upper_name.replace('_', '-')}"
            )


def fixed_pose(args: argparse.Namespace, x_mm: float, y_mm: float, z_mm: float) -> EndPose:
    return EndPose(
        float(x_mm),
        float(y_mm),
        float(z_mm),
        float(args.rx_deg),
        float(args.ry_deg),
        float(args.rz_deg),
    )


def cartesian_only_robot_config(can_name: str) -> Any:
    """Return the safety configuration used by the 1_1 Cartesian server.

    J1-J5 command and feedback software-angle checks are disabled temporarily;
    the arm controller still enforces its limit positions.  Their configured
    absolute ranges remain available as Cartesian IK bounds.  J6 keeps the
    explicit +/-180-degree software rotation limit.
    """

    limits = list(ROBOT_CONFIG.joint_limits_rad)
    limits[5] = PoseLimit(math.radians(-J6_LIMIT_DEG), math.radians(J6_LIMIT_DEG))
    return replace(
        ROBOT_CONFIG,
        can_name=can_name,
        joint_limits_rad=tuple(limits),
        joint_software_limit_enabled=(False, False, False, False, False, True),
        # Piper SDK applies its configured joint limits to feedback values as
        # well as JointCtrl commands.  Keep raw feedback for windup detection;
        # PiperController still checks finite values on every command and
        # enforces the retained J6 software limit before sending it.
        start_sdk_joint_limit=False,
    )


def object_grasp_orientation(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[float, float, float]:
    """Return a complete object-aligned end-pose orientation for grasp.

    Vision reports the object's short edge as a base-frame yaw.  Rotate the
    complete prepared orientation around the base Z axis by the same shortest
    axis correction used by the legacy yaw planner.  This changes RX/RY/RZ as a
    coupled Cartesian orientation; it does not command J6 independently.
    """

    if not getattr(args, "grasp_yaw_by_object_short_edge", False):
        return float(args.rx_deg), float(args.ry_deg), float(args.rz_deg)
    source_yaw = target.camera_command.object_short_axis_yaw_deg
    if source_yaw is None or not math.isfinite(float(source_yaw)):
        raise SafetyError(
            "object Cartesian orientation requires object_short_axis_yaw_deg"
        )
    target_yaw = normalize_degrees(
        float(source_yaw) + float(args.gripper_close_axis_offset_deg)
    )
    prepared = rotation_matrix_xyz_deg(args.rx_deg, args.ry_deg, args.rz_deg)
    # The physical closing direction is the tool-frame Y axis for the Piper
    # gripper.  Align that complete Cartesian axis to the object's short edge.
    closing_axis_base = prepared[:, 1]
    current_axis_yaw = normalize_degrees(
        math.degrees(
            math.atan2(float(closing_axis_base[1]), float(closing_axis_base[0]))
        )
    )
    yaw_plan = plan_grasp_yaw_delta(
        target_yaw,
        current_axis_yaw,
        max_delta_deg=180.0,
        # This legacy helper operates on the Cartesian closing-axis yaw here,
        # not directly on J6.  Actual joints are checked by Cartesian IK below.
        min_j6_deg=-360.0,
        max_j6_deg=360.0,
    )
    delta_rad = math.radians(float(yaw_plan.delta_deg))
    c, s = math.cos(delta_rad), math.sin(delta_rad)
    base_z_rotation = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return euler_xyz_deg_from_rotation_matrix(base_z_rotation @ prepared)


def grasp_oriented_pose(
    args: argparse.Namespace,
    target: YoloCatchTarget,
    pose: EndPose,
) -> EndPose:
    rx_deg, ry_deg, rz_deg = object_grasp_orientation(args, target)
    return replace(pose, rx_deg=rx_deg, ry_deg=ry_deg, rz_deg=rz_deg)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def detected_z_mode(args: argparse.Namespace) -> Literal["off", "center", "top"]:
    value = getattr(args, "use_detected_z", False)
    if value is True:
        return "center"
    if value in (False, None, "off"):
        return "off"
    if value in ("center", "top"):
        return value
    raise ValueError("--use-detected-z must be 'center', 'top', or disabled")


def dynamic_pitch_control_mode(args: argparse.Namespace) -> DynamicPitchControlMode:
    value = getattr(args, "dynamic_pitch_mode", "joint")
    return "planar_ik" if value == "planar-ik" else value


def joint_pitch_decision(
    args: argparse.Namespace,
    source_x_mm: float,
    *,
    feasible_pitch_delta_deg: float | None = None,
) -> JointPitchDecision:
    normalize_dynamic_pitch_args(args)
    source_x = float(source_x_mm)
    keep_min_x = float(args.keep_pitch_min_x_mm)
    keep_max_x = float(args.keep_pitch_max_x_mm)
    j2_ratio = float(args.dynamic_pitch_j2_ratio)
    j3_ratio = float(args.dynamic_pitch_j3_ratio)
    j5_ratio = float(args.dynamic_pitch_j5_ratio)
    if not args.dynamic_pitch_by_x:
        return JointPitchDecision(
            enabled=False,
            mode="disabled",
            source_x_mm=source_x,
            keep_min_x_mm=keep_min_x,
            keep_max_x_mm=keep_max_x,
            over_mm=0.0,
            raw_pitch_delta_deg=0.0,
            requested_pitch_delta_deg=0.0,
            pitch_delta_deg=0.0,
            clamped=False,
            limit_clamped=False,
            j2_ratio=j2_ratio,
            j3_ratio=j3_ratio,
            j5_ratio=j5_ratio,
            j2_delta_deg=0.0,
            j3_delta_deg=0.0,
            j5_delta_deg=0.0,
        )
    over_mm = 0.0
    raw_pitch_deg = 0.0
    mode: JointPitchMode = "keep_fixed"
    if source_x > keep_max_x:
        over_mm = source_x - keep_max_x
        raw_pitch_deg = over_mm * float(args.pitch_deg_per_mm_far)
        mode = "adjust"
    elif source_x < keep_min_x:
        over_mm = source_x - keep_min_x
        raw_pitch_deg = over_mm * float(args.pitch_deg_per_mm_near)
        mode = "adjust"

    requested_pitch_deg = (
        0.0
        if mode == "keep_fixed"
        else clamp(
            raw_pitch_deg,
            float(args.min_dynamic_pitch_deg),
            float(args.max_dynamic_pitch_deg),
        )
    )
    pitch_delta_deg = (
        requested_pitch_deg
        if feasible_pitch_delta_deg is None
        else float(feasible_pitch_delta_deg)
    )
    if mode != "adjust":
        pitch_delta_deg = 0.0
    return JointPitchDecision(
        enabled=True,
        mode=mode,
        source_x_mm=source_x,
        keep_min_x_mm=keep_min_x,
        keep_max_x_mm=keep_max_x,
        over_mm=over_mm,
        raw_pitch_delta_deg=raw_pitch_deg,
        requested_pitch_delta_deg=requested_pitch_deg,
        pitch_delta_deg=pitch_delta_deg,
        clamped=not math.isclose(
            raw_pitch_deg,
            requested_pitch_deg,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        limit_clamped=not math.isclose(
            requested_pitch_deg,
            pitch_delta_deg,
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
        j2_ratio=j2_ratio,
        j3_ratio=j3_ratio,
        j5_ratio=j5_ratio,
        j2_delta_deg=pitch_delta_deg * j2_ratio,
        j3_delta_deg=pitch_delta_deg * j3_ratio,
        j5_delta_deg=pitch_delta_deg * j5_ratio,
    )


def target_pose(
    args: argparse.Namespace,
    target: YoloCatchTarget,
    z_mm: float,
) -> EndPose:
    return fixed_pose(args, target.pick_x_mm, target.pick_y_mm, z_mm)


def use_joint_pitch_adjustment(args: argparse.Namespace, pitch_source_x_mm: float) -> bool:
    return False


def use_planar_ik(args: argparse.Namespace, pitch_source_x_mm: float) -> bool:
    return False


def fixed_pitch_source_x_mm(args: argparse.Namespace) -> float:
    return clamp(
        float(args.ready_x_mm),
        float(args.keep_pitch_min_x_mm),
        float(args.keep_pitch_max_x_mm),
    )


def far_planar_transfer_via_ready(args: argparse.Namespace, pitch_source_x_mm: float) -> bool:
    if getattr(args, "disable_ready_planar_ik_return", False):
        return False
    return use_planar_ik(args, pitch_source_x_mm)


def use_planar_ik_yaw(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "planar_ik_include_j6_yaw", False)
        and getattr(args, "grasp_yaw_by_object_short_edge", False)
        and getattr(args, "grasp_yaw_mode", "integrated-planar-ik")
        == "integrated-planar-ik"
    )


def grasp_yaw_timing_value(args: argparse.Namespace) -> GraspYawTiming:
    """Return the configured phase for the object-short-edge J6 rotation."""

    value = getattr(args, "grasp_yaw_timing", DEFAULT_GRASP_YAW_TIMING)
    if value not in ("pick_above", "pick_grasp"):
        raise ValueError(
            "--grasp-yaw-timing must be either 'pick_above' or 'pick_grasp'"
        )
    return value  # type: ignore[return-value]


def should_apply_grasp_yaw_after_step(
    args: argparse.Namespace,
    step_label: str,
) -> bool:
    timing = grasp_yaw_timing_value(args)
    if timing == "pick_above":
        return step_label in ("pick_above", "return_pick_above")
    return step_label in ("pick_grasp", "return_pick_grasp")


def dynamic_pitch_summary(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> dict[str, Any]:
    pitch_x_mm = pitch_source_x_mm(target)
    decision = joint_pitch_decision(args, pitch_x_mm)
    control_mode = dynamic_pitch_control_mode(args)
    target_ry_deg = float(args.ry_deg)
    if control_mode == "planar_ik" and decision.mode == "adjust":
        target_ry_deg += float(decision.pitch_delta_deg)
    return {
        "enabled": decision.enabled,
        "mode": decision.mode,
        "method": (
            (
                "planar_ik_j2_j3_j4_j5_j6_yaw"
                if use_planar_ik_yaw(args)
                else "planar_ik_j2_j3_j5"
            )
            if control_mode == "planar_ik"
            else "joint_space_j2_j3_j5"
        ),
        "control_mode": control_mode,
        "fixed_pose_ry_deg": float(args.ry_deg),
        "target_ry_deg": target_ry_deg,
        "pick_x_mm": float(target.pick_x_mm),
        "raw_pick_x_mm": target.raw_pick_x_mm,
        "pitch_source_x_mm": pitch_x_mm,
        "source_x_mm": pitch_x_mm,
        "keep_min_x_mm": decision.keep_min_x_mm,
        "keep_max_x_mm": decision.keep_max_x_mm,
        "over_mm": decision.over_mm,
        "x_ref_mm": float(args.pitch_x_ref_mm),
        "deg_per_mm": float(args.pitch_deg_per_mm),
        "pitch_deg_per_mm_far": float(args.pitch_deg_per_mm_far),
        "pitch_deg_per_mm_near": float(args.pitch_deg_per_mm_near),
        "raw_pitch_delta_deg": decision.raw_pitch_delta_deg,
        "requested_pitch_delta_deg": decision.requested_pitch_delta_deg,
        "pitch_delta_deg": decision.pitch_delta_deg,
        "min_pitch_delta_deg": float(args.min_dynamic_pitch_deg),
        "max_pitch_delta_deg": float(args.max_dynamic_pitch_deg),
        "j2_ratio": decision.j2_ratio,
        "j3_ratio": decision.j3_ratio,
        "j5_ratio": decision.j5_ratio,
        "j2_delta_deg": decision.j2_delta_deg,
        "j3_delta_deg": decision.j3_delta_deg,
        "j5_delta_deg": decision.j5_delta_deg,
        "speed": int(args.dynamic_pitch_speed),
        "planar_ik_speed": planar_ik_speed(args),
        "clamped": decision.clamped,
        "limit_clamped": decision.limit_clamped,
    }


def ready_pose(args: argparse.Namespace) -> EndPose:
    return fixed_pose(args, args.ready_x_mm, args.ready_y_mm, args.transfer_z_mm)


def place_pick_pose(args: argparse.Namespace) -> EndPose:
    x_mm = args.place_x_mm if args.place_pick_x_mm is None else args.place_pick_x_mm
    y_mm = args.place_y_mm if args.place_pick_y_mm is None else args.place_pick_y_mm
    z_mm = args.grasp_z_mm if args.place_pick_z_mm is None else args.place_pick_z_mm
    return fixed_pose(args, x_mm, y_mm, z_mm)


def pitch_source_x_mm(target: YoloCatchTarget) -> float:
    return (
        float(target.raw_pick_x_mm)
        if target.raw_pick_x_mm is not None
        else float(target.pick_x_mm)
    )


def planar_ik_grasp_z_compensation(
    args: argparse.Namespace,
    pitch_source_x_mm: float,
) -> tuple[float, float]:
    coefficient = float(getattr(args, "planar_ik_grasp_z_comp_mm_per_deg", 0.0))
    if coefficient <= 0.0 or not use_planar_ik(args, pitch_source_x_mm):
        return 0.0, 0.0
    decision = joint_pitch_decision(args, pitch_source_x_mm)
    pitch_delta_deg = float(decision.pitch_delta_deg)
    return abs(pitch_delta_deg) * coefficient, pitch_delta_deg


def dynamic_t_robot_reference_x_mm(args: argparse.Namespace) -> float:
    explicit = getattr(args, "dynamic_t_robot_reference_x_mm", None)
    if explicit is not None:
        return float(explicit)
    return float(args.ready_x_mm)


def dynamic_t_robot_s_for_x(
    args: argparse.Namespace,
    source_x_mm: float | None,
) -> tuple[float, dict[str, Any]]:
    base_s = float(args.t_robot_s)
    source = "constant"
    if not getattr(args, "dynamic_t_robot_by_x", False) or source_x_mm is None:
        return base_s, {
            "source": source,
            "source_x_mm": source_x_mm,
            "reference_x_mm": None,
            "segment": None,
            "slope_s_per_mm": None,
            "base_s": base_s,
            "unclamped_s": base_s,
            "margin_s": 0.0,
            "min_s": None,
            "max_s": None,
        }

    source_x = float(source_x_mm)
    reference_x = dynamic_t_robot_reference_x_mm(args)
    default_slope_s_per_mm = float(args.dynamic_t_robot_s_per_mm)
    if source_x < reference_x:
        segment = "near"
        side_slope = float(getattr(args, "dynamic_t_robot_planar_near_s_per_mm", 0.0))
    elif source_x > reference_x:
        segment = "far"
        side_slope = float(getattr(args, "dynamic_t_robot_planar_far_s_per_mm", 0.0))
    else:
        segment = "reference"
        side_slope = 0.0
    slope_s_per_mm = (
        side_slope
        if segment != "reference"
        and not math.isclose(side_slope, 0.0, rel_tol=0.0, abs_tol=1e-12)
        else default_slope_s_per_mm
    )
    if segment == "reference":
        slope_s_per_mm = 0.0
    margin_s = float(args.dynamic_t_robot_margin_s)
    t_robot_s = base_s + (reference_x - source_x) * slope_s_per_mm
    t_robot_s += margin_s
    unclamped_s = t_robot_s
    min_s = getattr(args, "dynamic_t_robot_min_s", None)
    max_s = getattr(args, "dynamic_t_robot_max_s", None)
    if min_s is not None:
        t_robot_s = max(t_robot_s, float(min_s))
    if max_s is not None:
        t_robot_s = min(t_robot_s, float(max_s))
    return t_robot_s, {
        "source": "x_linear",
        "source_x_mm": source_x,
        "reference_x_mm": reference_x,
        "segment": segment,
        "slope_s_per_mm": slope_s_per_mm,
        "base_s": base_s,
        "unclamped_s": unclamped_s,
        "margin_s": margin_s,
        "min_s": min_s,
        "max_s": max_s,
    }


def detected_target_z_mm_for_pick_x_perspective(
    command: CameraPointCommand,
    base_xyz_m: tuple[float, float, float],
) -> float:
    """Return the best available measured source height for X parallax."""

    for value in (
        command.detected_top_z_mm,
        command.detected_z_mm,
        float(base_xyz_m[2]) * 1000.0,
    ):
        if value is not None and math.isfinite(float(value)):
            return float(value)
    raise ValueError("perspective X compensation requires a finite detected source Z")


def compensate_pick_x_perspective_mm(
    raw_x_mm: float,
    *,
    reference_x_mm: float,
    camera_z_mm: float,
    source_z_mm: float,
    target_z_mm: float,
) -> tuple[float, float, float]:
    """Map X between two heights using the similar-triangle relation.

    ``raw_x_mm`` is the mapped detection position at ``source_z_mm`` and
    ``target_z_mm`` is the final pick-plane height.  The position relative to
    ``reference_x_mm`` scales by
    ``(camera_z-target_z)/(camera_z-source_z)``.  This is the same similar-
    triangle relation that makes a shadow stretch with distance from a light;
    depending on the two heights it may expand or contract. The reference
    itself always has zero compensation.
    """

    values = {
        "raw_x_mm": raw_x_mm,
        "reference_x_mm": reference_x_mm,
        "camera_z_mm": camera_z_mm,
        "source_z_mm": source_z_mm,
        "target_z_mm": target_z_mm,
    }
    for label, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"perspective X compensation {label} must be finite")

    source_camera_distance_mm = float(camera_z_mm) - float(source_z_mm)
    target_camera_distance_mm = float(camera_z_mm) - float(target_z_mm)
    if source_camera_distance_mm <= 0.0 or target_camera_distance_mm <= 0.0:
        raise ValueError(
            "perspective X compensation requires source and target Z below camera Z: "
            f"source={source_z_mm:.3f}mm target={target_z_mm:.3f}mm "
            f"camera={camera_z_mm:.3f}mm"
        )

    scale = target_camera_distance_mm / source_camera_distance_mm
    corrected_x_mm = float(reference_x_mm) + scale * (
        float(raw_x_mm) - float(reference_x_mm)
    )
    compensation_mm = corrected_x_mm - float(raw_x_mm)
    return corrected_x_mm, compensation_mm, scale


def target_from_command(
    command: CameraPointCommand,
    camera_to_base: Any,
    args: argparse.Namespace,
    *,
    validate: bool = True,
) -> YoloCatchTarget:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        command.x_m,
        command.y_m,
        command.z_m,
        camera_to_base,
    )
    if command.grasp_mode == "dynamic":
        if command.pick_x_mm is None or command.pick_y_mm is None:
            raise ValueError("dynamic grasp command requires pick_x_mm and pick_y_mm")
        pick_x_mm = float(command.pick_x_mm)
        pick_y_mm = float(command.pick_y_mm)
        pick_x_source = "dynamic_command_x"
        pick_y_source = "dynamic_command_y"
    else:
        pick_x_mm = base_xyz_m[0] * 1000.0
        pick_y_mm = base_xyz_m[1] * 1000.0
        pick_x_source = "mapped_command_x"
        pick_y_source = "mapped_command_y"
    raw_pick_x_mm = float(pick_x_mm)
    raw_pick_y_mm = float(pick_y_mm)
    if args.force_pick_x_mm is not None:
        pick_x_mm = float(args.force_pick_x_mm)
        pick_x_source = "forced_pick_x"
    perspective_reference_x_mm = getattr(
        args,
        "pick_x_perspective_reference_mm",
        None,
    )
    perspective_camera_z_mm: float | None = None
    perspective_source_z_mm: float | None = None
    perspective_target_z_mm: float | None = None
    perspective_scale: float | None = None
    perspective_compensation_mm = 0.0
    if args.force_pick_y_mm is not None:
        pick_y_mm = float(args.force_pick_y_mm)
        pick_y_source = "forced_pick_y"
    raw_grasp_z_mm: float | None = None
    grasp_z_source = "fixed_grasp_z"
    z_mode = detected_z_mode(args)
    if z_mode != "off":
        if z_mode == "top":
            if command.detected_top_z_mm is None or not math.isfinite(
                float(command.detected_top_z_mm)
            ):
                raise ValueError(
                    "--use-detected-z top requires detected_top_z_mm in the command payload"
                )
            detected_z_mm = float(command.detected_top_z_mm)
            grasp_z_source = "detected_top_z_plus_offset"
        else:
            detected_z_mm = base_xyz_m[2] * 1000.0
            grasp_z_source = "mapped_command_z_plus_offset"
            if (
                command.grasp_mode == "dynamic"
                and command.detected_z_mm is not None
                and math.isfinite(float(command.detected_z_mm))
            ):
                detected_z_mm = float(command.detected_z_mm)
                grasp_z_source = "dynamic_detected_z_plus_offset"
        raw_grasp_z_mm = detected_z_mm + args.detected_z_offset_mm
        grasp_z_mm = max(raw_grasp_z_mm, float(args.grasp_z_mm))
        if grasp_z_mm != raw_grasp_z_mm:
            grasp_z_source = f"{grasp_z_source}_clamped_to_fixed_grasp_z"
    else:
        grasp_z_mm = args.grasp_z_mm

    uncompensated_grasp_z_mm = float(grasp_z_mm)
    grasp_z_comp_mm, grasp_z_comp_pitch_delta_deg = planar_ik_grasp_z_compensation(
        args,
        float(raw_pick_x_mm),
    )
    if grasp_z_comp_mm > 0.0:
        grasp_z_mm = float(grasp_z_mm) - grasp_z_comp_mm
        grasp_z_source = f"{grasp_z_source}_planar_ik_pitch_z_comp"

    if perspective_reference_x_mm is not None:
        perspective_reference_x_mm = float(perspective_reference_x_mm)
        perspective_camera_z_mm = float(camera_to_base[2, 3]) * 1000.0
        perspective_source_z_mm = detected_target_z_mm_for_pick_x_perspective(
            command,
            base_xyz_m,
        )
        perspective_target_z_mm = float(grasp_z_mm)
        (
            pick_x_mm,
            perspective_compensation_mm,
            perspective_scale,
        ) = compensate_pick_x_perspective_mm(
            float(pick_x_mm),
            reference_x_mm=perspective_reference_x_mm,
            camera_z_mm=perspective_camera_z_mm,
            source_z_mm=perspective_source_z_mm,
            target_z_mm=perspective_target_z_mm,
        )
        pick_x_source = f"{pick_x_source}_perspective_compensated"
    target = YoloCatchTarget(
        camera_command=command,
        base_xyz_m=base_xyz_m,
        pick_x_mm=float(pick_x_mm),
        pick_y_mm=float(pick_y_mm),
        grasp_z_mm=float(grasp_z_mm),
        raw_pick_x_mm=raw_pick_x_mm,
        raw_pick_y_mm=raw_pick_y_mm,
        pick_x_source=pick_x_source,
        pick_y_source=pick_y_source,
        raw_grasp_z_mm=raw_grasp_z_mm,
        grasp_z_source=grasp_z_source,
        uncompensated_grasp_z_mm=uncompensated_grasp_z_mm,
        planar_ik_grasp_z_comp_mm=grasp_z_comp_mm,
        planar_ik_grasp_z_comp_pitch_delta_deg=grasp_z_comp_pitch_delta_deg,
        vision_source="opencv",
        pick_x_perspective_reference_mm=perspective_reference_x_mm,
        pick_x_perspective_camera_z_mm=perspective_camera_z_mm,
        pick_x_perspective_source_z_mm=perspective_source_z_mm,
        pick_x_perspective_target_z_mm=perspective_target_z_mm,
        pick_x_perspective_scale=perspective_scale,
        pick_x_perspective_compensation_mm=perspective_compensation_mm,
    )
    if validate:
        validate_target(target, args)
    return target


def validate_target(target: YoloCatchTarget, args: argparse.Namespace) -> None:
    checks = (
        ("target-x", target.pick_x_mm, args.min_target_x_mm, args.max_target_x_mm),
        ("target-y", target.pick_y_mm, args.min_target_y_mm, args.max_target_y_mm),
        ("grasp-z", target.grasp_z_mm, args.min_grasp_z_mm, args.max_grasp_z_mm),
    )
    for label, value, lower, upper in checks:
        if lower is not None and value < lower:
            raise SafetyError(f"{label} {value:.3f} mm is below limit {lower:.3f} mm")
        if upper is not None and value > upper:
            raise SafetyError(f"{label} {value:.3f} mm exceeds limit {upper:.3f} mm")
    above_z_mm = pick_above_lift_z_mm(args, target)
    if target.grasp_z_mm > above_z_mm:
        raise SafetyError(
            f"grasp-z {target.grasp_z_mm:.3f} mm is above the X-adjusted "
            f"pick_above/pick_lift Z {above_z_mm:.3f} mm. Reduce "
            "--pick-above-lift-z-gradient, reduce pick X, or increase the "
            "ready/transfer Z."
        )


def pick_above_lift_z_mm(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> float:
    """Return the X-dependent Z shared by pick_above and pick_lift.

    ``transfer_z_mm`` is also the ready-open Z in this Cartesian sequence.
    Only targets farther forward than ``ready_x_mm`` lower the waypoint;
    targets at or behind ready X retain the ready Z.
    """

    forward_x_mm = max(0.0, float(target.pick_x_mm) - float(args.ready_x_mm))
    return float(args.transfer_z_mm) - (
        float(args.pick_above_lift_z_gradient) * forward_x_mm
    )


def build_out_steps(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[SequenceStep, ...]:
    ready = ready_pose(args)
    pick_above = target_pose(args, target, pick_above_lift_z_mm(args, target))
    # The object yaw is applied after this waypoint is reached (see
    # ``execute_plan``). Keep all planned Cartesian orientations on the
    # prepared branch so the software pre-check does not ask IK to solve an
    # orientation that is only reachable after a J6-only rotation.
    pick_grasp = target_pose(args, target, target.grasp_z_mm)
    pick_lift = target_pose(args, target, pick_above_lift_z_mm(args, target))
    place_above = fixed_pose(args, args.place_x_mm, args.place_y_mm, args.transfer_z_mm)
    # 1_1 is Cartesian-only.  The legacy joint-pitch/planar-IK branches are
    # intentionally disabled so no JointCtrl command can be emitted.

    steps: list[SequenceStep] = [
        SequenceStep("ready_open", ready, "P", "open"),
        SequenceStep(
            "pick_above",
            pick_above,
            "P",
            require_reached=True,
        ),
        SequenceStep(
            label="pick_grasp",
            pose=pick_grasp,
            mode="P",
            gripper_after="close",
            delay_before_s=args.pick_delay_s,
        ),
        SequenceStep(
            "pick_lift",
            pick_lift,
            "P",
        ),
    ]
    if args.grasp_only:
        if not args.no_return_ready:
            steps.append(
                SequenceStep(
                    "return_ready",
                    ready,
                    "P",
                )
            )
        return tuple(steps)

    steps.extend(
        [
            SequenceStep("place_above", place_above, "P"),
            SequenceStep("place_release", place_above, "P", "open"),
        ]
    )
    if not args.no_return_ready:
        steps.append(SequenceStep("return_ready", ready, "P"))
    return tuple(steps)


def build_in_steps(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[SequenceStep, ...]:
    if args.grasp_only:
        raise ValueError("--grasp-only is only supported for task 'out'")
    if args.no_return_ready:
        raise ValueError("--no-return-ready is incompatible with task 'in'")

    ready = ready_pose(args)
    pick_above = target_pose(args, target, pick_above_lift_z_mm(args, target))
    # As with the outbound sequence, rotate J6 only after reaching the safe
    # above waypoint. Runtime execution replaces the descent/lift orientation
    # with the post-rotation feedback pose.
    pick_grasp = target_pose(args, target, target.grasp_z_mm)
    pick_lift = target_pose(args, target, pick_above_lift_z_mm(args, target))
    return (
        SequenceStep("ready_open", ready, "P", "open"),
        SequenceStep(
            label="return_pick_above",
            pose=pick_above,
            mode="P",
            delay_before_s=args.place_pick_delay_s,
            require_reached=True,
        ),
        SequenceStep(
            label="return_pick_grasp",
            pose=pick_grasp,
            mode="P",
            gripper_after="close",
        ),
        SequenceStep(
            "return_pick_lift",
            pick_lift,
            "P",
        ),
        SequenceStep("return_ready_above", ready, "P"),
        SequenceStep("final_release", ready, "P", "open"),
    )


def build_steps(args: argparse.Namespace, target: YoloCatchTarget) -> tuple[SequenceStep, ...]:
    if target.camera_command.task == "out":
        return build_out_steps(args, target)
    if target.camera_command.task == "in":
        return build_in_steps(args, target)
    raise ValueError("camera point task must be 'out' or 'in'")


def validate_steps(controller: PiperController, steps: tuple[SequenceStep, ...]) -> None:
    for step in steps:
        try:
            controller.validate_pose(step.pose)
        except SafetyError as exc:
            raise SafetyError(f"{step.label}: {exc}") from exc


def _cartesian_ik_residual(
    joints_rad: np.ndarray,
    target_pose: EndPose,
    fk_tool_z_offset_mm: float = DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
) -> np.ndarray:
    solved = apply_tool_z_offset_to_fk_pose(
        forward_kinematics_pose(joints_rad),
        fk_tool_z_offset_mm,
    )
    target_rotation = rotation_matrix_xyz_deg(
        target_pose.rx_deg,
        target_pose.ry_deg,
        target_pose.rz_deg,
    )
    solved_rotation = rotation_matrix_xyz_deg(
        solved.rx_deg,
        solved.ry_deg,
        solved.rz_deg,
    )
    orientation_error = Rotation.from_matrix(
        solved_rotation.T @ target_rotation
    ).as_rotvec()
    return np.concatenate(
        (
            np.array(
                [
                    (solved.x_mm - target_pose.x_mm) / 100.0,
                    (solved.y_mm - target_pose.y_mm) / 100.0,
                    (solved.z_mm - target_pose.z_mm) / 100.0,
                ],
                dtype=float,
            ),
            orientation_error,
        )
    )


def _cartesian_ik_joint_bounds(controller: PiperController) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [limit.min_value for limit in controller.config.joint_limits_rad],
        dtype=float,
    )
    upper = np.array(
        [limit.max_value for limit in controller.config.joint_limits_rad],
        dtype=float,
    )
    return lower, upper


def _cartesian_ik_jacobian_condition(
    joints: tuple[float, ...] | np.ndarray,
    pose: EndPose,
    fk_tool_z_offset_mm: float,
) -> float:
    joint_vector = np.asarray(joints, dtype=float)
    jacobian = np.empty((6, 6), dtype=float)
    epsilon = 1e-5
    for index in range(6):
        plus = joint_vector.copy()
        minus = joint_vector.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        jacobian[:, index] = (
            _cartesian_ik_residual(plus, pose, float(fk_tool_z_offset_mm))
            - _cartesian_ik_residual(minus, pose, float(fk_tool_z_offset_mm))
        ) / (2.0 * epsilon)
    return float(np.linalg.cond(jacobian))


def validate_cartesian_joint_limits(
    controller: PiperController,
    pose: EndPose,
    *,
    seed_joints: tuple[float, ...] | None = None,
    label: str = "cartesian waypoint",
    fk_tool_z_offset_mm: float = DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
    joint_limit_margin_deg: float = 0.0,
) -> tuple[float, ...]:
    """Check Cartesian reachability, all joint limits and singularity."""

    lower, upper = _cartesian_ik_joint_bounds(controller)
    margin_rad = math.radians(float(joint_limit_margin_deg))
    if margin_rad < 0.0:
        raise ValueError("joint_limit_margin_deg must be >= 0")
    if margin_rad:
        lower = lower + margin_rad
        upper = upper - margin_rad
        if np.any(lower >= upper):
            raise ValueError("joint_limit_margin_deg leaves no feasible joint interval")
    if seed_joints is None:
        seed = (lower + upper) * 0.5
    else:
        seed = np.clip(np.asarray(seed_joints, dtype=float), lower, upper)
    # scipy.optimize.least_squares can remain stuck when a seed is exactly on
    # a one-sided bound (notably Piper's parked J2=0 state).  Move only exact
    # boundary values a negligible distance into the feasible interval; this
    # preserves the feedback branch while giving the optimizer a search
    # direction.
    seed_epsilon_rad = 1e-6
    seed = np.maximum(seed, lower + seed_epsilon_rad)
    seed = np.minimum(seed, upper - seed_epsilon_rad)
    result = least_squares(
        _cartesian_ik_residual,
        seed,
        args=(pose, float(fk_tool_z_offset_mm)),
        bounds=(lower, upper),
        max_nfev=300,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    joints = tuple(float(value) for value in result.x)
    residual = _cartesian_ik_residual(
        np.asarray(joints),
        pose,
        float(fk_tool_z_offset_mm),
    )
    position_error_mm = float(np.linalg.norm(residual[:3]) * 100.0)
    orientation_error_deg = math.degrees(float(np.linalg.norm(residual[3:])))
    # A bounded least-squares solve can stop exactly on a joint limit while
    # still leaving a small Cartesian residual.  The broad 8 mm/8 deg
    # reachability tolerance is useful for model calibration, but must not
    # turn a limit-clamped, physically unreachable target into a MOVE P
    # command that Piper will reject with TARGET_POS_EXCEEDS_LIMIT.
    limit_hits: list[str] = []
    limit_epsilon_rad = 1e-5
    for index, (value, lower_value, upper_value) in enumerate(
        zip(joints, lower, upper),
    ):
        if abs(value - lower_value) <= limit_epsilon_rad:
            limit_hits.append(f"J{index + 1}={math.degrees(value):.3f}deg(min)")
        elif abs(value - upper_value) <= limit_epsilon_rad:
            limit_hits.append(f"J{index + 1}={math.degrees(value):.3f}deg(max)")
    if limit_hits and (
        position_error_mm > DEFAULT_CARTESIAN_IK_LIMIT_RESIDUAL_POSITION_MM
        or orientation_error_deg > DEFAULT_CARTESIAN_IK_LIMIT_RESIDUAL_ORIENTATION_DEG
    ):
        raise SafetyError(
            f"{label}: target is outside Piper joint limits; Cartesian IK is "
            f"clamped at {', '.join(limit_hits)} with residual "
            f"position_error={position_error_mm:.3f}mm, "
            f"orientation_error={orientation_error_deg:.3f}deg. "
            "Reduce pick X/offset, lower the waypoint Z, or change the prepared "
            "orientation."
        )
    if position_error_mm > 8.0 or orientation_error_deg > 8.0:
        raise SafetyError(
            f"{label}: Cartesian IK failed to reach {format_pose(pose)} "
            f"(position_error={position_error_mm:.3f}mm, "
            f"orientation_error={orientation_error_deg:.3f}deg)"
        )
    condition = _cartesian_ik_jacobian_condition(
        joints,
        pose,
        float(fk_tool_z_offset_mm),
    )
    if not math.isfinite(condition) or condition > DEFAULT_MAX_JACOBIAN_CONDITION:
        raise SafetyError(
            f"{label}: Cartesian IK is singular or ill-conditioned "
            f"(condition={condition:.3f})"
        )
    return joints


def _multi_start_cartesian_ik_seeds(
    controller: PiperController,
    seed_joints: tuple[float, ...] | None,
    seed_count: int = DEFAULT_MULTI_START_CARTESIAN_IK_SEED_COUNT,
) -> tuple[tuple[float, ...], ...]:
    """Build deterministic IK seeds spanning common Piper branches.

    The solver is still a bounded local least-squares IK.  Mirroring selected
    joints about their configured intervals provides deterministic elbow/wrist
    alternatives without introducing random behaviour into a catch command.
    The returned list is always exactly ``seed_count`` entries. If a
    reflection happens to coincide with another seed after clipping, it is
    retained so the diagnostic count still reflects the requested search
    budget.
    """

    if (
        not isinstance(seed_count, int)
        or isinstance(seed_count, bool)
        or not 1 <= seed_count <= MAX_MULTI_START_CARTESIAN_IK_SEED_COUNT
    ):
        raise ValueError("multi-start Cartesian IK seed_count must be in [1, 8]")

    try:
        lower, upper = _cartesian_ik_joint_bounds(controller)
    except (AttributeError, TypeError):
        # Keep this helper mock-friendly for controller-level unit tests.  A
        # real PiperController always exposes finite six-joint bounds; the
        # fallback still exercises the configured number of calls to the
        # patched solver.
        if seed_joints is None:
            raise
        return tuple(
            tuple(float(value) for value in seed_joints) for _ in range(seed_count)
        )
    if seed_joints is None:
        base = (lower + upper) * 0.5
    else:
        base = np.asarray(seed_joints, dtype=float)
    base = np.clip(base, lower + 1e-6, upper - 1e-6)
    # J1, J2/J3 (elbow), J4/J5/J6 (wrist), and combinations cover the usual
    # analytic Piper branches while keeping the search deterministic.  The
    # interval-centre seed is deliberately included: a J1 reflection of a
    # near-zero current posture would otherwise be identical to the base seed.
    reflected_masks = (
        None,
        "center",
        (1, 2),
        (3, 4, 5),
        (1, 2, 3, 4, 5),
        "j1_plus",
        "j1_minus_elbow",
        "j1_plus_wrist",
    )
    seeds: list[tuple[float, ...]] = []
    for mask in reflected_masks[:seed_count]:
        candidate = (lower + upper) * 0.5 if mask == "center" else base.copy()
        if mask == "j1_plus":
            candidate[0] += math.radians(90.0)
        elif mask == "j1_minus_elbow":
            candidate[0] -= math.radians(90.0)
            indices = np.asarray((1, 2), dtype=int)
            candidate[indices] = lower[indices] + upper[indices] - candidate[indices]
        elif mask == "j1_plus_wrist":
            candidate[0] += math.radians(90.0)
            indices = np.asarray((3, 4, 5), dtype=int)
            candidate[indices] = lower[indices] + upper[indices] - candidate[indices]
        elif mask is not None and mask != "center":
            indices = np.asarray(mask, dtype=int)
            candidate[indices] = lower[indices] + upper[indices] - candidate[indices]
        candidate = np.clip(candidate, lower + 1e-6, upper - 1e-6)
        seeds.append(tuple(float(value) for value in candidate))
    while len(seeds) < seed_count:
        seeds.append(seeds[-1])
    return tuple(seeds)


def validate_multistart_cartesian_joint_limits(
    controller: PiperController,
    pose: EndPose,
    *,
    seed_joints: tuple[float, ...] | None = None,
    label: str = "cartesian waypoint",
    fk_tool_z_offset_mm: float = DEFAULT_CARTESIAN_IK_FK_TOOL_Z_OFFSET_MM,
    seed_count: int = DEFAULT_MULTI_START_CARTESIAN_IK_SEED_COUNT,
) -> tuple[float, ...]:
    """Solve a grasp waypoint from multiple seeds and select the nearest valid one.

    This affects only the software IK pre-check/branch prediction.  The
    caller still sends ``pose`` through the existing Cartesian MOVE P/L path;
    no MOVE J command is emitted here.
    """

    seeds = _multi_start_cartesian_ik_seeds(
        controller,
        seed_joints,
        seed_count=seed_count,
    )
    reference = np.asarray(seed_joints, dtype=float) if seed_joints is not None else None
    solutions: list[tuple[float, ...]] = []
    failures: list[tuple[int, Exception]] = []
    for index, candidate_seed in enumerate(seeds, start=1):
        try:
            solution = validate_cartesian_joint_limits(
                controller,
                pose,
                seed_joints=candidate_seed,
                label=f"{label} [IK seed {index}/{len(seeds)}]",
                fk_tool_z_offset_mm=fk_tool_z_offset_mm,
            )
        except (SafetyError, ValueError, RuntimeError) as exc:
            failures.append((index, exc))
            continue
        # De-duplicate numerically identical local solutions.  We still run
        # all configured solves so diagnostics are comparable across waypoints.
        if not any(
            max(abs(a - b) for a, b in zip(solution, known))
            <= math.radians(0.05)
            for known in solutions
        ):
            solutions.append(solution)

    if not solutions:
        if failures:
            # Preserve the detailed single-start error (including J5 and FK
            # residuals) when no branch is valid.
            failed_text = "; ".join(
                f"seed {index}: {exc}" for index, exc in failures
            )
            raise SafetyError(
                f"{label}: all {len(seeds)} multi-start Cartesian IK seeds "
                f"failed; {failed_text}"
            ) from failures[0][1]
        raise SafetyError(f"{label}: multi-start Cartesian IK found no solution")

    if reference is None:
        selected = solutions[0]
    else:
        selected = min(
            solutions,
            key=lambda solution: float(
                np.linalg.norm(np.asarray(solution, dtype=float) - reference)
            ),
        )
    selected_delta_deg = (
        0.0
        if reference is None
        else math.degrees(
            float(np.linalg.norm(np.asarray(selected, dtype=float) - reference))
        )
    )
    branch_text = "; ".join(
        "(" + ", ".join(f"{math.degrees(value):.3f}" for value in solution) + ")"
        for solution in solutions
    )
    print(
        f"{label}: multi-start Cartesian IK seeds={len(seeds)} "
        f"valid_branches={len(solutions)} failed_seeds={len(failures)} "
        f"selected_delta_deg={selected_delta_deg:.3f} "
        f"valid_joint_branches_deg=[{branch_text}]",
        flush=True,
    )
    return selected


def validate_runtime_joint_limits(
    controller: PiperController,
    *,
    label: str,
) -> tuple[float, ...]:
    joints = joint_feedback_to_rad(controller.get_joint_state())
    if joints is None:
        raise RuntimeError(f"{label}: valid joint feedback is required for safety")
    for index, software_limit in enumerate(SOFTWARE_JOINT_LIMITS_DEG):
        joint_deg = math.degrees(float(joints[index]))
        if not math.isfinite(joint_deg):
            raise SafetyError(f"{label}: feedback J{index + 1} must be finite")
        if software_limit is None:
            continue
        min_deg, max_deg = software_limit
        if not min_deg <= joint_deg <= max_deg:
            raise SafetyError(
                f"{label}: feedback J{index + 1}={joint_deg:.3f}deg exceeds "
                f"software range [{min_deg:.1f}, {max_deg:.1f}]deg"
            )
    return joints


def validate_joint_vector_limits(
    joints: tuple[float, ...] | list[float] | np.ndarray,
    *,
    label: str,
    stop_margin_deg: float = DEFAULT_JOINT_LIMIT_STOP_MARGIN_DEG,
) -> tuple[float, ...]:
    if len(joints) != 6:
        raise SafetyError(f"{label}: expected 6 joints, got {len(joints)}")
    checked = tuple(float(value) for value in joints)
    for index, (joint_rad, software_limit) in enumerate(
        zip(checked, SOFTWARE_JOINT_LIMITS_DEG),
    ):
        joint_deg = math.degrees(joint_rad)
        if not math.isfinite(joint_deg):
            raise SafetyError(f"{label}: J{index + 1} must be finite")
        if software_limit is None:
            continue
        min_deg, max_deg = software_limit
        safe_min_deg = float(min_deg) + float(stop_margin_deg)
        safe_max_deg = float(max_deg) - float(stop_margin_deg)
        if joint_deg < safe_min_deg or joint_deg > safe_max_deg:
            raise SafetyError(
                f"{label}: J{index + 1}={joint_deg:.3f}deg reaches the "
                f"protected [{safe_min_deg:.1f}, {safe_max_deg:.1f}]deg boundary "
                f"(software range [{min_deg:.1f}, {max_deg:.1f}]deg)"
            )
    return checked


@dataclass
class CartesianJointMotionGuard:
    """Reject unsafe Cartesian joint discontinuities and long-way rotations."""

    label: str
    start_joints: tuple[float, ...]
    target_joints: tuple[float, ...]
    previous_joints: tuple[float, ...]
    cumulative_travel_deg: list[float]
    wrong_way_travel_deg: list[float]
    travel_budget_deg: tuple[float, ...]

    @classmethod
    def create(
        cls,
        label: str,
        start_joints: tuple[float, ...],
        target_joints: tuple[float, ...],
    ) -> "CartesianJointMotionGuard":
        start = validate_joint_vector_limits(
            start_joints,
            label=f"before {label}",
        )
        target = validate_joint_vector_limits(
            target_joints,
            label=f"planned {label}",
        )
        requested_delta_deg = tuple(
            abs(math.degrees(target_value - start_value))
            for start_value, target_value in zip(start, target)
        )
        for index, (delta_deg, maximum_deg) in enumerate(
            zip(requested_delta_deg, DEFAULT_MAX_CARTESIAN_JOINT_TRAVEL_DEG),
        ):
            if delta_deg > maximum_deg:
                raise SafetyError(
                    f"{label}: planned J{index + 1} change {delta_deg:.3f}deg "
                    f"exceeds {maximum_deg:.1f}deg; refusing a possible "
                    "long-way rotation or IK branch switch"
                )
        budgets = tuple(
            min(maximum_deg, max(30.0, delta_deg + 30.0))
            for delta_deg, maximum_deg in zip(
                requested_delta_deg,
                DEFAULT_MAX_CARTESIAN_JOINT_TRAVEL_DEG,
            )
        )
        return cls(label, start, target, start, [0.0] * 6, [0.0] * 6, budgets)

    def observe(self, joints: tuple[float, ...]) -> tuple[float, ...]:
        current = validate_joint_vector_limits(
            joints,
            label=f"during {self.label}",
        )
        for index, (previous, value, budget_deg) in enumerate(
            zip(self.previous_joints, current, self.travel_budget_deg),
        ):
            increment_deg = abs(math.degrees(value - previous))
            if increment_deg > DEFAULT_MAX_JOINT_FEEDBACK_JUMP_DEG:
                raise SafetyError(
                    f"{self.label}: J{index + 1} feedback jumped "
                    f"{increment_deg:.3f}deg; possible angle wrap or IK branch switch"
                )
            self.cumulative_travel_deg[index] += increment_deg
            target_delta = self.target_joints[index] - self.start_joints[index]
            observed_delta = value - previous
            if (
                self.label in READY_POSE_STEP_LABELS
                and (
                    (target_delta > 0.0 and observed_delta < 0.0)
                    or (target_delta < 0.0 and observed_delta > 0.0)
                    or (math.isclose(target_delta, 0.0, abs_tol=math.radians(1.0)))
                )
            ):
                self.wrong_way_travel_deg[index] += increment_deg
                if self.wrong_way_travel_deg[index] > DEFAULT_READY_WRONG_WAY_TRAVEL_DEG:
                    raise SafetyError(
                        f"{self.label}: J{index + 1} moved "
                        f"{self.wrong_way_travel_deg[index]:.3f}deg away from the "
                        "fixed ready-pose joint branch; stopping possible windup"
                    )
            if self.cumulative_travel_deg[index] > budget_deg:
                raise SafetyError(
                    f"{self.label}: J{index + 1} accumulated "
                    f"{self.cumulative_travel_deg[index]:.3f}deg travel, exceeding "
                    f"the {budget_deg:.1f}deg anti-windup budget"
                )
        self.previous_joints = current
        return current

    def validate_target_branch(
        self,
        target_pose: EndPose,
        *,
        fk_tool_z_offset_mm: float,
        joint_tolerance_deg: float,
        position_tolerance_mm: float,
        orientation_tolerance_deg: float,
    ) -> None:
        signed_errors_deg = tuple(
            math.degrees(actual - target)
            for actual, target in zip(self.previous_joints, self.target_joints)
        )
        maximum_error_deg = max(abs(value) for value in signed_errors_deg)
        if maximum_error_deg <= float(joint_tolerance_deg):
            return

        # EndPoseCtrl performs IK inside Piper firmware and cannot receive our
        # scipy seed.  A different joint vector is therefore not, by itself,
        # evidence of an unsafe branch switch.  Accept a continuous, bounded
        # firmware solution when the same corrected FK model proves that it
        # reaches the commanded pose.  Feedback jumps, limit crossings and
        # excessive cumulative travel have already been rejected by observe().
        actual_fk_pose = apply_tool_z_offset_to_fk_pose(
            forward_kinematics_pose(self.previous_joints),
            float(fk_tool_z_offset_mm),
        )
        fk_position_error_mm, fk_orientation_error_deg = pose_errors(
            target_pose,
            actual_fk_pose,
        )
        actual_condition = _cartesian_ik_jacobian_condition(
            self.previous_joints,
            target_pose,
            float(fk_tool_z_offset_mm),
        )
        diagnostic = (
            f"planned_deg={tuple(round(math.degrees(value), 3) for value in self.target_joints)} "
            f"actual_deg={tuple(round(math.degrees(value), 3) for value in self.previous_joints)} "
            f"joint_error_deg={tuple(round(value, 3) for value in signed_errors_deg)} "
            f"corrected_fk_pos_err={fk_position_error_mm:.3f}mm "
            f"corrected_fk_ori_err={fk_orientation_error_deg:.3f}deg "
            f"corrected_fk_jacobian_condition={actual_condition:.3f}"
        )
        if (
            fk_position_error_mm <= float(position_tolerance_mm)
            and fk_orientation_error_deg <= float(orientation_tolerance_deg)
            and math.isfinite(actual_condition)
            and actual_condition <= DEFAULT_MAX_JACOBIAN_CONDITION
        ):
            print(
                f"{self.label}: accepting continuous Piper firmware IK solution; "
                + diagnostic,
                flush=True,
            )
            return
        raise SafetyError(
            f"{self.label}: endpoint joint feedback disagrees with corrected "
            f"Cartesian IK model; {diagnostic}"
        )


def execute_cartesian_pose_guarded(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    *,
    speed: int,
    initial_joints: tuple[float, ...],
    target_joints: tuple[float, ...],
    should_wait: bool,
) -> int:
    """Send MOVE P while continuously rejecting joint windup and branch flips."""

    guard = CartesianJointMotionGuard.create(
        step.label,
        initial_joints,
        target_joints,
    )
    send_deadline = time.monotonic() + float(args.command_duration_s)
    send_count = 0
    while send_count == 0 or time.monotonic() < send_deadline:
        controller.move_to_pose(step.pose, speed_percent=speed, move_mode=step.mode)
        send_count += 1
        if _is_piper_target_limit_status(controller.get_status()):
            raise SafetyError(
                f"{step.label}: Piper rejected Cartesian target with "
                "TARGET_POS_EXCEEDS_LIMIT (0x04); target is outside the "
                "controller's joint/workspace limits"
            )
        feedback = joint_feedback_to_rad(controller.get_joint_state())
        if feedback is None:
            raise RuntimeError(
                f"{step.label}: valid joint feedback is required during Cartesian motion"
            )
        guard.observe(feedback)
        if time.monotonic() >= send_deadline:
            break
        time.sleep(float(args.command_period_s))

    if not should_wait:
        return send_count

    wait_deadline = time.monotonic() + float(args.motion_timeout_s)
    last_pose: EndPose | None = None
    while args.motion_timeout_s == 0 or time.monotonic() < wait_deadline:
        feedback = joint_feedback_to_rad(controller.get_joint_state())
        if feedback is None:
            raise RuntimeError(
                f"{step.label}: valid joint feedback is required while waiting"
            )
        if _is_piper_target_limit_status(controller.get_status()):
            raise SafetyError(
                f"{step.label}: Piper reported TARGET_POS_EXCEEDS_LIMIT "
                "(0x04) while waiting for the Cartesian target"
            )
        guard.observe(feedback)
        last_pose = feedback_to_pose(controller.get_end_pose())
        if last_pose is not None:
            position_error, orientation_error = pose_errors(step.pose, last_pose)
            if (
                position_error <= args.position_tolerance_mm
                and orientation_error <= args.orientation_tolerance_deg
            ):
                guard.validate_target_branch(
                    step.pose,
                    fk_tool_z_offset_mm=float(
                        args.cartesian_ik_fk_tool_z_offset_mm
                    ),
                    joint_tolerance_deg=float(
                        args.cartesian_ik_joint_target_tolerance_deg
                    ),
                    position_tolerance_mm=float(args.position_tolerance_mm),
                    orientation_tolerance_deg=float(args.orientation_tolerance_deg),
                )
                return send_count
        if args.motion_timeout_s == 0:
            break
        time.sleep(min(0.05, float(args.command_period_s)))

    position_error, orientation_error = (
        pose_errors(step.pose, last_pose)
        if last_pose is not None
        else (float("inf"), float("inf"))
    )
    raise RuntimeError(
        f"waypoint {step.label} not reached: target={format_pose(step.pose)} "
        f"last={format_pose(last_pose)} pos_err={position_error:.3f} mm "
        f"ori_err={orientation_error:.3f} deg {format_status(controller)}"
    )


def rollback_to_joint_state(
    controller: PiperController,
    args: argparse.Namespace,
    target_joints: tuple[float, ...],
    *,
    label: str,
) -> None:
    """Recover by commanding the direct joint direction in small bounded steps."""

    target = validate_joint_vector_limits(target_joints, label=f"rollback {label} target")
    current_feedback = joint_feedback_to_rad(controller.get_joint_state())
    if current_feedback is None:
        raise RuntimeError(f"rollback {label}: valid joint feedback is required")
    current = tuple(float(value) for value in current_feedback)
    maximum_delta_deg = max(
        abs(math.degrees(target_value - current_value))
        for current_value, target_value in zip(current, target)
    )
    waypoint_count = max(1, math.ceil(maximum_delta_deg / ROLLBACK_MAX_JOINT_STEP_DEG))
    speed = max(1, min(int(args.speed), 20))
    for waypoint_index in range(1, waypoint_count + 1):
        ratio = waypoint_index / waypoint_count
        waypoint = tuple(
            current_value + (target_value - current_value) * ratio
            for current_value, target_value in zip(current, target)
        )
        validate_joint_vector_limits(
            waypoint,
            label=f"rollback {label}[{waypoint_index}]",
        )
        controller.move_joints_and_wait(
            waypoint,
            speed_percent=speed,
            timeout_s=max(1.0, float(args.motion_timeout_s)),
            command_duration_s=max(0.05, float(args.command_period_s)),
        )
        validate_runtime_joint_limits(
            controller,
            label=f"after rollback {label}[{waypoint_index}]",
        )


def validate_or_record_ready_joint_anchor(
    controller: PiperController,
    step: SequenceStep,
    joints: tuple[float, ...],
) -> None:
    """Ensure every fixed ready pose resolves to the original joint branch."""

    if step.label not in READY_POSE_STEP_LABELS:
        return
    current = validate_joint_vector_limits(
        joints,
        label=f"ready branch {step.label}",
    )
    anchor = getattr(controller, READY_JOINT_ANCHOR_ATTR, None)
    if anchor is None:
        setattr(controller, READY_JOINT_ANCHOR_ATTR, current)
        print(
            "Recorded fixed ready-pose joint branch: "
            + " ".join(
                f"J{index + 1}={math.degrees(value):.3f}deg"
                for index, value in enumerate(current)
            ),
            flush=True,
        )
        return
    for index, (value, expected) in enumerate(zip(current, anchor)):
        error_deg = abs(math.degrees(value - expected))
        if error_deg > DEFAULT_READY_JOINT_ANCHOR_TOLERANCE_DEG:
            raise SafetyError(
                f"{step.label}: fixed ready pose reached through a different joint "
                f"branch; J{index + 1} differs from startup by {error_deg:.3f}deg"
            )


def ready_pose_ik_seed(
    controller: PiperController,
    step: SequenceStep,
    current_joints: tuple[float, ...],
) -> tuple[float, ...]:
    """Reuse the startup ready-pose branch when solving later ready moves."""

    if step.label not in READY_POSE_STEP_LABELS:
        return current_joints
    ik_anchor = getattr(controller, READY_IK_JOINT_ANCHOR_ATTR, None)
    if ik_anchor is not None:
        return tuple(float(value) for value in ik_anchor)
    actual_anchor = getattr(controller, READY_JOINT_ANCHOR_ATTR, None)
    if actual_anchor is not None:
        return tuple(float(value) for value in actual_anchor)
    return current_joints


def validate_planned_ready_joint_anchor(
    controller: PiperController,
    step: SequenceStep,
    joints: tuple[float, ...],
) -> None:
    """Record or validate the model-side IK counterpart of the startup branch."""

    if step.label not in READY_POSE_STEP_LABELS:
        return
    planned = validate_joint_vector_limits(
        joints,
        label=f"planned ready branch {step.label}",
    )
    anchor = getattr(controller, READY_IK_JOINT_ANCHOR_ATTR, None)
    if anchor is None:
        setattr(controller, READY_IK_JOINT_ANCHOR_ATTR, planned)
        print(
            "Recorded fixed ready-pose IK branch: "
            + " ".join(
                f"J{index + 1}={math.degrees(value):.3f}deg"
                for index, value in enumerate(planned)
            ),
            flush=True,
        )
        return
    for index, (value, expected) in enumerate(zip(planned, anchor)):
        error_deg = abs(math.degrees(value - expected))
        if error_deg > DEFAULT_READY_JOINT_ANCHOR_TOLERANCE_DEG:
            raise SafetyError(
                f"{step.label}: planned fixed ready pose uses a different joint "
                f"branch; model J{index + 1} differs from the startup IK branch "
                f"by {error_deg:.3f}deg"
            )


def ready_pose_is_already_reached(
    step: SequenceStep,
    current_pose: EndPose | None,
    args: argparse.Namespace,
) -> tuple[bool, float, float]:
    """Return whether a fixed ready step needs no Cartesian command."""

    if step.label not in READY_POSE_STEP_LABELS or current_pose is None:
        return False, float("inf"), float("inf")
    position_error, orientation_error = pose_errors(step.pose, current_pose)
    reached = (
        position_error <= float(args.position_tolerance_mm)
        and orientation_error <= float(args.orientation_tolerance_deg)
    )
    return reached, position_error, orientation_error


def build_plan_for_target(
    command: CameraPointCommand,
    target: YoloCatchTarget,
    args: argparse.Namespace,
) -> YoloCatchPlan:
    return YoloCatchPlan(
        task=command.task,
        target=target,
        steps=build_steps(args, target),
    )


def dynamic_timing_from_command(
    command: CameraPointCommand,
    args: argparse.Namespace,
    *,
    source_x_mm: float | None = None,
) -> DynamicTiming | None:
    if command.grasp_mode != "dynamic":
        return None
    if command.t_arrive_s is None:
        raise ValueError("dynamic grasp command requires t_arrive_s")
    t_arrive_s = float(command.t_arrive_s)
    wait_placement = str(getattr(args, "dynamic_wait_placement", "wait"))
    # In preposition+wait mode the arm first reaches the target-side waiting
    # pose. Only the final descent/close is scheduled, using the fixed
    # final_action_time_s budget. The full ready-to-grasp t_robot estimate and
    # its X slopes do not affect that deadline, so do not apply or advertise an
    # unused X timing correction. Nowait still needs the X-dependent estimate
    # because it schedules departure from the initial ready pose.
    x_timing_is_used = not (
        bool(args.dynamic_preposition) and wait_placement == "wait"
    )
    t_robot_s, t_robot_meta = dynamic_t_robot_s_for_x(
        args,
        source_x_mm if x_timing_is_used else None,
    )
    if not x_timing_is_used:
        t_robot_meta["source"] = "disabled_preposition_wait"
    if not math.isfinite(t_arrive_s) or t_arrive_s < 0.0:
        raise ValueError("dynamic t_arrive_s must be finite and >= 0")
    if not math.isfinite(t_robot_s) or t_robot_s < 0.0:
        raise ValueError("dynamic t_robot_s must be finite and >= 0")
    if args.dynamic_preposition and wait_placement == "nowait":
        action_time_s = t_robot_s
        action_time_label = "t_robot"
    elif args.dynamic_preposition:
        action_time_s = float(args.final_action_time_s)
        action_time_label = "final_action_time"
    else:
        action_time_s = t_robot_s
        action_time_label = "t_robot"
    if not math.isfinite(action_time_s) or action_time_s < 0.0:
        raise ValueError(f"--{action_time_label.replace('_', '-')} must be finite and >= 0")
    t_wait_s = t_arrive_s - action_time_s
    too_close_skip_enabled = bool(getattr(args, "dynamic_too_close_skip", True))
    if t_wait_s < 0.0 and too_close_skip_enabled:
        return DynamicTiming(
            t_arrive_s=t_arrive_s,
            t_robot_s=t_robot_s,
            action_time_s=action_time_s,
            t_wait_s=t_wait_s,
            t_robot_source=str(t_robot_meta["source"]),
            t_robot_source_x_mm=t_robot_meta["source_x_mm"],
            t_robot_reference_x_mm=t_robot_meta["reference_x_mm"],
            t_robot_segment=t_robot_meta["segment"],
            t_robot_slope_s_per_mm=t_robot_meta["slope_s_per_mm"],
            t_robot_base_s=t_robot_meta["base_s"],
            t_robot_unclamped_s=t_robot_meta["unclamped_s"],
            t_robot_margin_s=t_robot_meta["margin_s"],
            t_robot_min_s=t_robot_meta["min_s"],
            t_robot_max_s=t_robot_meta["max_s"],
            dynamic_preposition=bool(args.dynamic_preposition),
            final_action_time_s=(
                float(args.final_action_time_s) if args.dynamic_preposition else None
            ),
            wait_placement=wait_placement,
            too_close_skip_enabled=too_close_skip_enabled,
            skipped=True,
            skip_reason=(
                "dynamic target is too close to the pick line: "
                f"t_arrive={t_arrive_s:.3f}s < "
                f"{action_time_label}={action_time_s:.3f}s"
            ),
        )
    return DynamicTiming(
        t_arrive_s=t_arrive_s,
        t_robot_s=t_robot_s,
        action_time_s=action_time_s,
        t_wait_s=t_wait_s,
        t_robot_source=str(t_robot_meta["source"]),
        t_robot_source_x_mm=t_robot_meta["source_x_mm"],
        t_robot_reference_x_mm=t_robot_meta["reference_x_mm"],
        t_robot_segment=t_robot_meta["segment"],
        t_robot_slope_s_per_mm=t_robot_meta["slope_s_per_mm"],
        t_robot_base_s=t_robot_meta["base_s"],
        t_robot_unclamped_s=t_robot_meta["unclamped_s"],
        t_robot_margin_s=t_robot_meta["margin_s"],
        t_robot_min_s=t_robot_meta["min_s"],
        t_robot_max_s=t_robot_meta["max_s"],
        dynamic_preposition=bool(args.dynamic_preposition),
        final_action_time_s=(
            float(args.final_action_time_s) if args.dynamic_preposition else None
        ),
        wait_placement=wait_placement,
        too_close_skip_enabled=too_close_skip_enabled,
    )


def wait_for_dynamic_timing(timing: DynamicTiming | None) -> None:
    if timing is None:
        return
    if timing.skipped:
        print(f"Dynamic timing: skipped: {timing.skip_reason}", flush=True)
        return
    if timing.t_wait_s <= 0.0:
        print(
            "Dynamic timing: start immediately "
            f"(t_arrive={timing.t_arrive_s:.3f}s, "
            f"t_robot={timing.t_robot_s:.3f}s)",
            flush=True,
        )
        return
    print(
        "Dynamic timing: waiting "
        f"{timing.t_wait_s:.3f}s before grasp "
        f"(t_arrive={timing.t_arrive_s:.3f}s, "
        f"t_robot={timing.t_robot_s:.3f}s)",
        flush=True,
    )
    time.sleep(timing.t_wait_s)


def dynamic_preposition_pose(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[EndPose, str]:
    command = target.camera_command
    if detected_z_mode(args) == "top":
        detected_z_mm = command.detected_top_z_mm
        detected_source = "detected_top_z_plus_clearance"
    else:
        detected_z_mm = command.detected_z_mm
        detected_source = "detected_z_plus_clearance"
    if detected_z_mm is not None and math.isfinite(float(detected_z_mm)):
        requested_z_mm = float(detected_z_mm) + float(args.dynamic_preposition_clearance_mm)
        source = detected_source
    else:
        requested_z_mm = target.grasp_z_mm + float(
            args.dynamic_preposition_clearance_mm
        )
        source = "grasp_z_plus_clearance"

    above_z_mm = pick_above_lift_z_mm(args, target)
    min_descent_z_mm = min(above_z_mm, target.grasp_z_mm + 1.0)
    z_mm = min(above_z_mm, max(requested_z_mm, min_descent_z_mm))
    if z_mm != requested_z_mm:
        source = f"{source}_clamped"
    return target_pose(args, target, z_mm), source


def wait_for_dynamic_final_action(
    timing: DynamicTiming,
    *,
    elapsed_s: float,
    final_action_time_s: float,
    placement: str = "wait",
) -> float:
    due_s = timing.t_arrive_s - final_action_time_s
    wait_s = due_s - elapsed_s
    placement_text = {
        "nowait": "at initial ready",
        "preposition_after_nowait": "at preposition",
    }.get(placement, "at preposition")
    if wait_s <= 0.0:
        late_s = elapsed_s - due_s
        print(
            "Dynamic final timing: start immediately "
            f"(t_arrive={timing.t_arrive_s:.3f}s, "
            f"elapsed={elapsed_s:.3f}s, "
            f"final_action={final_action_time_s:.3f}s, "
            f"placement={placement} "
            f"due={due_s:.3f}s, "
            f"late={late_s:.3f}s)",
            flush=True,
        )
        return 0.0
    print(
        "Dynamic final timing: waiting "
        f"{wait_s:.3f}s {placement_text} "
        f"(t_arrive={timing.t_arrive_s:.3f}s, "
        f"elapsed={elapsed_s:.3f}s, "
        f"final_action={final_action_time_s:.3f}s, "
        f"placement={placement} "
        f"due={due_s:.3f}s)",
        flush=True,
    )
    time.sleep(wait_s)
    return wait_s


def use_dynamic_nowait(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "dynamic_preposition", False)
        and getattr(args, "dynamic_wait_placement", "wait") == "nowait"
    )


def yolo_metadata_from_message(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return {}

    metadata: dict[str, Any] = {}
    bbox_value = payload.get("bbox")
    bbox_format = payload.get("bbox_format")
    if bbox_value is None:
        bbox_value = payload.get("bbox_xyxy")
        if bbox_value is not None:
            bbox_format = bbox_format or "xyxy"
    if bbox_value is None:
        bbox_value = payload.get("bbox_xywh")
        if bbox_value is not None:
            bbox_format = bbox_format or "xywh"
    if bbox_value is None and all(
        key in payload for key in ("x", "y", "width", "height")
    ):
        bbox_value = [
            payload["x"],
            payload["y"],
            payload["width"],
            payload["height"],
        ]
        bbox_format = bbox_format or "xywh"
    if bbox_value is None and all(key in payload for key in ("x1", "y1", "x2", "y2")):
        bbox_value = [payload["x1"], payload["y1"], payload["x2"], payload["y2"]]
        bbox_format = bbox_format or "xyxy"

    if bbox_value is not None:
        bbox = _tuple4_float(bbox_value, "bbox")
        metadata["bbox"] = list(bbox)
        metadata["bbox_format"] = str(bbox_format or "unknown")

    for key in ("label", "source_label"):
        if payload.get(key) is not None:
            metadata["label"] = str(payload[key])
            break
    if payload.get("confidence") is not None:
        metadata["confidence"] = float(payload["confidence"])
    if payload.get("class_id") is not None:
        metadata["class_id"] = int(payload["class_id"])
    if payload.get("rank") is not None:
        metadata["rank"] = int(payload["rank"])
    if payload.get("task") is not None:
        metadata["task"] = str(payload["task"])
    for key in (
        "mask_area_px",
        "mask_centroid_u",
        "mask_centroid_v",
        "mask_top_u",
        "mask_top_v",
        "mask_depth_median_m",
        "mask_depth_valid_count",
        "detected_top_x_mm",
        "detected_top_y_mm",
        "detected_top_z_mm",
        "object_short_axis_yaw_deg",
        "object_rect_width_px",
        "object_rect_height_px",
        "object_rect_angle_deg",
    ):
        if payload.get(key) is not None:
            value = payload[key]
            metadata[key] = (
                int(value)
                if key in ("mask_area_px", "mask_depth_valid_count")
                else float(value)
            )
    if payload.get("object_yaw_source") is not None:
        metadata["object_yaw_source"] = str(payload["object_yaw_source"])
    return metadata


def _tuple4_float(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must contain exactly 4 numeric values")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 4 numeric values") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{label} must contain finite numeric values")
    return values


def print_plan(
    plan: YoloCatchPlan,
    args: argparse.Namespace,
    metadata: dict[str, Any] | None = None,
    timing: DynamicTiming | None = None,
) -> None:
    command = plan.target.camera_command
    base = plan.target.base_xyz_m
    print("YOLO catch plan:", flush=True)
    print(
        "  cartesian_ik_model: "
        f"raw_d6={ROBOT_CONFIG.modified_dh_params[-1].d_mm:.3f}mm "
        f"tool_z_compensation={float(args.cartesian_ik_fk_tool_z_offset_mm):.3f}mm "
        f"effective_d6="
        f"{ROBOT_CONFIG.modified_dh_params[-1].d_mm + float(args.cartesian_ik_fk_tool_z_offset_mm):.3f}mm",
        flush=True,
    )
    print(
        "  grasp_multistart_ik: "
        f"seed_count={int(args.multi_start_cartesian_ik_seed_count)} "
        f"steps={','.join(sorted(MULTI_START_CARTESIAN_IK_STEP_LABELS))}",
        flush=True,
    )
    print(f"  speed: sequence={int(args.speed)}% j6={j6_speed(args)}%", flush=True)
    print(
        "  camera: "
        f"u={command.u} v={command.v} "
        f"xyz=({command.x_m:.6f}, {command.y_m:.6f}, {command.z_m:.6f}) m "
        f"source={command.source}",
        flush=True,
    )
    print(
        "  base: "
        f"x={base[0] * 1000.0:.3f} y={base[1] * 1000.0:.3f} "
        f"z={base[2] * 1000.0:.3f} mm "
        f"grasp_z={plan.target.grasp_z_mm:.3f} mm "
        f"source={plan.target.grasp_z_source}",
        flush=True,
    )
    if plan.target.raw_grasp_z_mm is not None:
        print(
            "  detected_grasp_z: "
            f"raw={plan.target.raw_grasp_z_mm:.3f}mm "
            f"fixed_floor={args.grasp_z_mm:.3f}mm "
            f"final={plan.target.grasp_z_mm:.3f}mm",
            flush=True,
        )
    if plan.target.planar_ik_grasp_z_comp_mm > 0.0:
        print(
            "  planar_ik_grasp_z_comp: "
            f"pitch_delta={plan.target.planar_ik_grasp_z_comp_pitch_delta_deg:.3f}deg "
            f"k={float(args.planar_ik_grasp_z_comp_mm_per_deg):.3f}mm/deg "
            f"comp={plan.target.planar_ik_grasp_z_comp_mm:.3f}mm "
            f"original_z={plan.target.uncompensated_grasp_z_mm:.3f}mm "
            f"adjusted_z={plan.target.grasp_z_mm:.3f}mm",
            flush=True,
        )
    above_z_mm = pick_above_lift_z_mm(args, plan.target)
    forward_x_mm = max(
        0.0,
        float(plan.target.pick_x_mm) - float(args.ready_x_mm),
    )
    print(
        "  pick_above_lift_z: "
        f"ready_x={float(args.ready_x_mm):.3f}mm "
        f"ready_z={float(args.transfer_z_mm):.3f}mm "
        f"pick_x={float(plan.target.pick_x_mm):.3f}mm "
        f"forward_x={forward_x_mm:.3f}mm "
        f"gradient={float(args.pick_above_lift_z_gradient):.6f}mm/mm "
        f"final_z={above_z_mm:.3f}mm",
        flush=True,
    )
    if (
        plan.target.raw_pick_x_mm is not None
        and (
            plan.target.pick_x_source != "mapped_command_x"
            or plan.target.pick_y_source != "mapped_command_y"
        )
    ):
        print(
            "  pick_line: "
            f"raw=({plan.target.raw_pick_x_mm:.3f}, "
            f"{plan.target.raw_pick_y_mm:.3f})mm "
            f"final=({plan.target.pick_x_mm:.3f}, "
            f"{plan.target.pick_y_mm:.3f})mm "
            f"x_source={plan.target.pick_x_source} "
            f"y_source={plan.target.pick_y_source}",
            flush=True,
        )
    if plan.target.pick_x_perspective_reference_mm is not None:
        print(
            "  pick_x_perspective: "
            f"reference={plan.target.pick_x_perspective_reference_mm:.3f}mm "
            f"camera_z={plan.target.pick_x_perspective_camera_z_mm:.3f}mm "
            f"source_z={plan.target.pick_x_perspective_source_z_mm:.3f}mm "
            f"target_z={plan.target.pick_x_perspective_target_z_mm:.3f}mm "
            f"scale={plan.target.pick_x_perspective_scale:.6f} "
            f"compensation={plan.target.pick_x_perspective_compensation_mm:.3f}mm",
            flush=True,
        )
    print(
        "  grasp_yaw_timing: "
        f"{grasp_yaw_timing_value(args)}",
        flush=True,
    )
    pitch = dynamic_pitch_summary(args, plan.target)
    if pitch["enabled"]:
        print(
            "  dynamic_pitch: "
            f"method={pitch['method']} "
            f"mode={pitch['mode']} "
            f"source_x={pitch['source_x_mm']:.3f}mm "
            f"keep=[{pitch['keep_min_x_mm']:.3f},"
            f"{pitch['keep_max_x_mm']:.3f}]mm "
            f"over={pitch['over_mm']:.3f}mm "
            f"k_near={pitch['pitch_deg_per_mm_near']:.6f}deg/mm "
            f"k_far={pitch['pitch_deg_per_mm_far']:.6f}deg/mm "
            f"raw_delta={pitch['raw_pitch_delta_deg']:.3f}deg "
            f"requested={pitch['requested_pitch_delta_deg']:.3f}deg "
            f"delta={pitch['pitch_delta_deg']:.3f}deg "
            f"j2={pitch['j2_ratio']:.3f} "
            f"j3={pitch['j3_ratio']:.3f} "
            f"j5={pitch['j5_ratio']:.3f} "
            f"clamped={pitch['clamped']} "
            f"limit_clamped={pitch['limit_clamped']}",
            flush=True,
        )
    if metadata:
        print(f"  yolo: {metadata}", flush=True)
    if timing is not None:
        print(
            "  dynamic_timing: "
            f"t_arrive={timing.t_arrive_s:.3f}s "
            f"t_robot={timing.t_robot_s:.3f}s "
            f"action_time={timing.action_time_s:.3f}s "
            f"t_wait={timing.t_wait_s:.3f}s "
            f"t_robot_source={timing.t_robot_source} "
            f"source_x={timing.t_robot_source_x_mm}mm "
            f"segment={timing.t_robot_segment} "
            f"preposition={timing.dynamic_preposition} "
            f"skipped={timing.skipped}",
            flush=True,
        )
        if timing.dynamic_preposition:
            preposition_pose, preposition_source = dynamic_preposition_pose(args, plan.target)
            print(
                "  dynamic_preposition: "
                f"z={preposition_pose.z_mm:.3f}mm "
                f"source={preposition_source} "
                f"clearance={args.dynamic_preposition_clearance_mm:.3f}mm "
                f"final_action={args.final_action_time_s:.3f}s "
                f"wait_placement={args.dynamic_wait_placement}",
                flush=True,
            )
    for index, step in enumerate(plan.steps, start=1):
        gripper = "" if step.gripper_after is None else f" gripper={step.gripper_after}"
        delay = "" if step.delay_before_s <= 0 else f" delay_before={step.delay_before_s:g}s"
        print(
            f"  {index:02d}. {step.label}: mode={step.mode} "
            f"{format_pose(step.pose)}{delay}{gripper}",
            flush=True,
        )


def handle_probe_message(
    command: CameraPointCommand,
    metadata: dict[str, Any],
    *,
    camera_to_base: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        command.x_m,
        command.y_m,
        command.z_m,
        camera_to_base,
    )
    probe_pose = EndPose(
        x_mm=base_xyz_m[0] * 1000.0,
        y_mm=base_xyz_m[1] * 1000.0,
        z_mm=base_xyz_m[2] * 1000.0,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )
    label = metadata.get("label", command.label or "unknown")
    rank = metadata.get("rank", command.rank)
    bbox = metadata.get("bbox", command.bbox_xywh)
    bbox_format = metadata.get("bbox_format", "xywh" if command.bbox_xywh else None)
    print(
        "YOLO probe only: "
        f"rank={rank} label={label} "
        f"camera_pixel=({command.u},{command.v}) "
        "camera_xyz="
        f"({command.x_m:.6f}, {command.y_m:.6f}, {command.z_m:.6f}) m -> "
        "base_xyz="
        f"({base_xyz_m[0] * 1000.0:.3f}, "
        f"{base_xyz_m[1] * 1000.0:.3f}, "
        f"{base_xyz_m[2] * 1000.0:.3f}) mm "
        f"bbox={bbox} bbox_format={bbox_format}",
        flush=True,
    )
    response = make_ack_message(
        command.command_id,
        pose_to_dict(probe_pose),
        executed=False,
    )
    response["payload"].update(
        {
            "task": command.task,
            "grasp_mode": command.grasp_mode,
            "probe_only": True,
            "mapping_source": "hand_eye",
            "base_xyz_m": {
                "x_m": base_xyz_m[0],
                "y_m": base_xyz_m[1],
                "z_m": base_xyz_m[2],
            },
            "pick_x_mm": None,
            "pick_y_mm": None,
            "grasp_z_mm": None,
            "skipped": False,
        }
    )
    if metadata:
        response["payload"]["yolo"] = metadata
    return response


def grasp_effort_nm(args: argparse.Namespace) -> float:
    return float(args.effort_nm if args.grasp_effort_nm is None else args.grasp_effort_nm)


def gripper_feedback_values(feedback_msg: Any) -> tuple[float | None, float | None]:
    state = getattr(feedback_msg, "gripper_state", feedback_msg)
    position_raw = getattr(state, "grippers_angle", None)
    effort_raw = getattr(state, "grippers_effort", None)
    position_mm = None if position_raw is None else float(position_raw) * 0.001
    effort_nm = None if effort_raw is None else abs(float(effort_raw) * 0.001)
    return position_mm, effort_nm


def monitor_safe_gripper_effort(
    controller: PiperController,
    args: argparse.Namespace,
) -> SafeGripperResult:
    max_effort_nm = float(args.max_gripper_effort_nm)
    deadline = time.monotonic() + float(args.safe_gripper_monitor_s)
    poll_count = 0
    last_position_mm: float | None = None
    last_effort_nm: float | None = None
    while float(args.safe_gripper_monitor_s) == 0.0 or time.monotonic() <= deadline:
        poll_count += 1
        last_position_mm, last_effort_nm = gripper_feedback_values(
            controller.get_gripper_state()
        )
        if last_effort_nm is not None and last_effort_nm >= max_effort_nm:
            controller.set_gripper(args.open_command_mm, 0.0, enable=False)
            stop_action = "disable"
            release_command_mm = None
            if args.safe_gripper_release_mm is not None:
                release_command_mm = float(args.safe_gripper_release_mm)
                controller.set_gripper(
                    release_command_mm,
                    min(grasp_effort_nm(args), max_effort_nm),
                    enable=True,
                )
                stop_action = "disable_release"
            return SafeGripperResult(
                action="close",
                command_mm=float(args.grasp_command_mm),
                command_effort_nm=grasp_effort_nm(args),
                max_effort_nm=max_effort_nm,
                triggered=True,
                stopped=True,
                stop_action=stop_action,
                release_command_mm=release_command_mm,
                last_effort_nm=last_effort_nm,
                last_position_mm=last_position_mm,
                poll_count=poll_count,
            )
        if float(args.safe_gripper_monitor_s) == 0.0:
            break
        time.sleep(float(args.safe_gripper_poll_s))
    return SafeGripperResult(
        action="close",
        command_mm=float(args.grasp_command_mm),
        command_effort_nm=grasp_effort_nm(args),
        max_effort_nm=max_effort_nm,
        last_effort_nm=last_effort_nm,
        last_position_mm=last_position_mm,
        poll_count=poll_count,
    )


def send_gripper(
    controller: PiperController,
    args: argparse.Namespace,
    action: Action,
) -> SafeGripperResult | None:
    if action is None or args.no_gripper:
        return None
    command_mm = args.open_command_mm if action == "open" else args.grasp_command_mm
    effort_nm = float(args.effort_nm if action == "open" else grasp_effort_nm(args))
    controller.set_gripper(command_mm, effort_nm, enable=True)
    if action == "close" and args.safe_gripper_stop_on_effort:
        return monitor_safe_gripper_effort(controller, args)
    settle_s = (
        float(getattr(args, "grasp_settle_s", 0.0))
        if action == "close"
        else float(args.gripper_settle_s)
    )
    if settle_s > 0:
        time.sleep(settle_s)
    return SafeGripperResult(
        action=action,
        command_mm=float(command_mm),
        command_effort_nm=effort_nm,
        max_effort_nm=(
            float(args.max_gripper_effort_nm)
            if action == "close" and args.safe_gripper_stop_on_effort
            else None
        ),
    )


def execute_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    *,
    speed: int | None = None,
    joint_pitch_adjustments: list[JointPitchAdjustmentResult] | None = None,
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult] | None = None,
    planar_ik_results: list[PlanarIkExecutionResult] | None = None,
    planar_ik_reuse_state: PlanarIkReuseState | None = None,
) -> tuple[int, SafeGripperResult | None]:
    if step.planar_ik or step.joint_pitch_before_pose or step.joint_pitch_after_pose:
        raise RuntimeError(
            f"{step.label}: catch_server_1_1 is Cartesian-only; "
            "joint/planar-IK waypoints are disabled"
        )
    previous_pose = feedback_to_pose(controller.get_end_pose())
    previous_joints = joint_feedback_to_rad(controller.get_joint_state())
    try:
        if step.delay_before_s > 0:
            print(f"Waiting {step.delay_before_s:g}s before {step.label}", flush=True)
            time.sleep(step.delay_before_s)
        current_joints = validate_runtime_joint_limits(
            controller,
            label=f"before {step.label}",
        )
        current_pose = feedback_to_pose(controller.get_end_pose())
        already_reached, position_error, orientation_error = ready_pose_is_already_reached(
            step,
            current_pose,
            args,
        )
        if already_reached:
            validate_or_record_ready_joint_anchor(controller, step, current_joints)
            print(
                f"Skipping Cartesian MOVE P for {step.label}: fixed ready pose "
                f"already reached (position_error={position_error:.3f}mm, "
                f"orientation_error={orientation_error:.3f}deg)",
                flush=True,
            )
            gripper_result = send_gripper(controller, args, step.gripper_after)
            return 0, gripper_result
        ik_seed = ready_pose_ik_seed(controller, step, current_joints)
        ik_solver = (
            validate_multistart_cartesian_joint_limits
            if step.label in MULTI_START_CARTESIAN_IK_STEP_LABELS
            else validate_cartesian_joint_limits
        )
        ik_kwargs: dict[str, Any] = {}
        if step.label in MULTI_START_CARTESIAN_IK_STEP_LABELS:
            ik_kwargs["seed_count"] = int(args.multi_start_cartesian_ik_seed_count)
        planned_joints = ik_solver(
            controller,
            step.pose,
            seed_joints=ik_seed,
            label=step.label,
            fk_tool_z_offset_mm=float(args.cartesian_ik_fk_tool_z_offset_mm),
            **ik_kwargs,
        )
        validate_planned_ready_joint_anchor(controller, step, planned_joints)
        print(
            f"Running {step.label}: {format_pose(step.pose)} mode={step.mode} "
            f"cartesian_ik_fk_tool_z_offset="
            f"{float(args.cartesian_ik_fk_tool_z_offset_mm):.3f}mm "
            f"ik_seed_deg={tuple(round(math.degrees(value), 3) for value in ik_seed)} "
            f"planned_joints_deg="
            f"{tuple(round(math.degrees(value), 3) for value in planned_joints)}",
            flush=True,
        )
        should_wait = step.require_reached or not args.no_wait_waypoints
        send_count = execute_cartesian_pose_guarded(
            controller,
            args,
            step,
            speed=args.speed if speed is None else speed,
            initial_joints=current_joints,
            target_joints=planned_joints,
            should_wait=should_wait,
        )
        final_joints = validate_runtime_joint_limits(
            controller,
            label=f"after {step.label}",
        )
        final_joint_errors_deg = tuple(
            math.degrees(actual - planned)
            for actual, planned in zip(final_joints, planned_joints)
        )
        print(
            f"Cartesian endpoint joints: step={step.label} "
            f"planned_deg="
            f"{tuple(round(math.degrees(value), 3) for value in planned_joints)} "
            f"actual_deg="
            f"{tuple(round(math.degrees(value), 3) for value in final_joints)} "
            f"joint_error_deg="
            f"{tuple(round(value, 3) for value in final_joint_errors_deg)}",
            flush=True,
        )
        validate_or_record_ready_joint_anchor(controller, step, final_joints)
        gripper_result = send_gripper(controller, args, step.gripper_after)
        return send_count, gripper_result
    except Exception:
        if previous_joints is not None:
            print(
                f"{step.label} failed; stopping Cartesian motion and returning "
                "to the previous joint state"
                + (
                    f" at {format_pose(previous_pose)}"
                    if previous_pose is not None
                    else ""
                ),
                flush=True,
            )
            try:
                controller.hold_current_position(
                    speed_percent=max(1, min(int(args.speed), 10)),
                    command_duration_s=max(0.05, float(args.command_period_s)),
                )
                rollback_to_joint_state(
                    controller,
                    args,
                    previous_joints,
                    label=step.label,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                print(f"Joint rollback failed: {rollback_exc}", flush=True)
        raise


def record_gripper_result(
    results: list[SafeGripperResult],
    result: SafeGripperResult | None,
) -> None:
    if result is not None:
        results.append(result)


def safe_gripper_result_to_dict(result: SafeGripperResult) -> dict[str, Any]:
    return {
        "action": result.action,
        "command_mm": result.command_mm,
        "command_effort_nm": result.command_effort_nm,
        "max_effort_nm": result.max_effort_nm,
        "triggered": result.triggered,
        "stopped": result.stopped,
        "stop_action": result.stop_action,
        "release_command_mm": result.release_command_mm,
        "last_effort_nm": result.last_effort_nm,
        "last_position_mm": result.last_position_mm,
        "poll_count": result.poll_count,
    }


def joint_pitch_adjustment_to_dict(
    result: JointPitchAdjustmentResult,
) -> dict[str, Any]:
    return {
        "step_label": result.step_label,
        "phase": result.phase,
        "pick_x_mm": result.pick_x_mm,
        "pitch_source_x_mm": result.pitch_source_x_mm,
        "raw_pitch_delta_deg": result.raw_pitch_delta_deg,
        "pitch_delta_deg": result.pitch_delta_deg,
        "requested_pitch_delta_deg": result.requested_pitch_delta_deg,
        "clamped": result.clamped,
        "limit_clamped": result.limit_clamped,
        "j2_delta_deg": result.j2_delta_deg,
        "j3_delta_deg": result.j3_delta_deg,
        "j5_delta_deg": result.j5_delta_deg,
        "start_joints_deg": list(result.start_joints_deg),
        "target_joints_deg": list(result.target_joints_deg),
        "send_count": result.send_count,
    }


def grasp_yaw_adjustment_to_dict(result: GraspYawAdjustmentResult) -> dict[str, Any]:
    return {
        "enabled": result.enabled,
        "applied": result.applied,
        "reason": result.reason,
        "source_yaw_deg": result.source_yaw_deg,
        "target_yaw_deg": result.target_yaw_deg,
        "current_j6_deg": result.current_j6_deg,
        "delta_deg": result.delta_deg,
        "clamped": result.clamped,
        "limit_clamped": result.limit_clamped,
        "start_joints_deg": (
            None if result.start_joints_deg is None else list(result.start_joints_deg)
        ),
        "target_joints_deg": (
            None if result.target_joints_deg is None else list(result.target_joints_deg)
        ),
        "send_count": result.send_count,
    }


def joint_feedback_to_rad(joint_feedback: Any) -> tuple[float, ...] | None:
    joint_state = getattr(joint_feedback, "joint_state", None)
    if joint_state is None:
        return None
    values = [getattr(joint_state, f"joint_{index}", None) for index in range(1, 7)]
    if any(value is None for value in values):
        return None
    return tuple(float(value) / RAD_TO_MILLI_DEG for value in values)


def joints_rad_to_deg(joints_rad: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    return tuple(math.degrees(value) for value in joints_rad)


def joint_feedback_timestamp(joint_feedback: Any) -> float | None:
    timestamp = getattr(joint_feedback, "time_stamp", None)
    if timestamp is None:
        return None
    return float(timestamp)


def apply_tool_z_offset_to_fk_pose(pose: EndPose, offset_mm: float) -> EndPose:
    if math.isclose(offset_mm, 0.0, rel_tol=0.0, abs_tol=1e-9):
        return pose
    offset_base = rotation_matrix_xyz_deg(
        pose.rx_deg,
        pose.ry_deg,
        pose.rz_deg,
    ) @ np.array([0.0, 0.0, float(offset_mm)], dtype=float)
    return EndPose(
        x_mm=float(pose.x_mm + offset_base[0]),
        y_mm=float(pose.y_mm + offset_base[1]),
        z_mm=float(pose.z_mm + offset_base[2]),
        rx_deg=pose.rx_deg,
        ry_deg=pose.ry_deg,
        rz_deg=pose.rz_deg,
    )


def forward_kinematics_planar_ik_pose(
    joints_rad: tuple[float, ...] | list[float],
    args: argparse.Namespace,
) -> EndPose:
    fk_pose = forward_kinematics_pose(joints_rad)
    return apply_tool_z_offset_to_fk_pose(
        fk_pose,
        float(args.planar_ik_fk_tool_z_offset_mm),
    )


def joint_errors_deg(
    current_joints_rad: tuple[float, ...] | list[float],
    target_joints_rad: tuple[float, ...] | list[float],
    joint_indices: tuple[int, ...] = PLANAR_IK_JOINT_INDICES,
) -> tuple[float, ...]:
    return tuple(
        math.degrees(float(current_joints_rad[index]) - float(target_joints_rad[index]))
        for index in joint_indices
    )


def wait_until_planar_ik_joints_reached(
    controller: PiperController,
    label: str,
    target_joints_rad: tuple[float, ...] | list[float],
    timeout_s: float,
    joint_tolerance_deg: float,
    *,
    joint_indices: tuple[int, ...] = PLANAR_IK_JOINT_INDICES,
    initial_feedback_stamp: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    deadline = time.monotonic() + timeout_s
    last_joints: tuple[float, ...] | None = None
    last_errors: tuple[float, ...] | None = None
    max_error = float("inf")
    while timeout_s == 0 or time.monotonic() < deadline:
        feedback = controller.get_joint_state()
        current_joints = joint_feedback_to_rad(feedback)
        timestamp = joint_feedback_timestamp(feedback)
        fresh_feedback = (
            initial_feedback_stamp is None
            or timestamp is None
            or timestamp > initial_feedback_stamp
        )
        if current_joints is not None:
            last_joints = current_joints
            last_errors = joint_errors_deg(
                current_joints,
                target_joints_rad,
                joint_indices,
            )
            max_error = max(abs(value) for value in last_errors)
            if fresh_feedback and max_error <= joint_tolerance_deg:
                return current_joints, last_errors, max_error
        if timeout_s == 0:
            break
        time.sleep(0.05)
    target_deg = joints_rad_to_deg(target_joints_rad)
    last_deg = None if last_joints is None else joints_rad_to_deg(last_joints)
    raise RuntimeError(
        f"planar IK joints for {label} not reached: "
        f"target_{planar_ik_joint_label(joint_indices).lower()}="
        f"{tuple(round(target_deg[index], 3) for index in joint_indices)} "
        f"last_{planar_ik_joint_label(joint_indices).lower()}="
        f"{'unavailable' if last_deg is None else tuple(round(last_deg[index], 3) for index in joint_indices)} "
        f"max_joint_err={max_error:.3f}deg "
        f"tol={joint_tolerance_deg:.3f}deg "
        f"{format_status(controller)}"
    )


def planar_ik_result_to_dict(result: PlanarIkExecutionResult) -> dict[str, Any]:
    payload = {
        "step_label": result.step_label,
        "pitch_source_x_mm": result.pitch_source_x_mm,
        "target_pose": pose_to_dict(result.target_pose),
        "solved_pose": pose_to_dict(result.solved_pose),
        "solved_fk_pose": (
            None if result.solved_fk_pose is None else pose_to_dict(result.solved_fk_pose)
        ),
        "start_joints_deg": list(result.start_joints_deg),
        "target_joints_deg": list(result.target_joints_deg),
        "final_joints_deg": (
            None if result.final_joints_deg is None else list(result.final_joints_deg)
        ),
        "joint_errors_deg": (
            None if result.joint_errors_deg is None else list(result.joint_errors_deg)
        ),
        "max_joint_error_deg": result.max_joint_error_deg,
        "final_pose": None if result.final_pose is None else pose_to_dict(result.final_pose),
        "reused": result.reused,
        "reused_from_step_label": result.reused_from_step_label,
        "pitch_delta_deg": result.pitch_delta_deg,
        "requested_pitch_delta_deg": result.requested_pitch_delta_deg,
        "iterations": result.iterations,
        "x_error_mm": result.x_error_mm,
        "y_error_mm": result.y_error_mm,
        "z_error_mm": result.z_error_mm,
        "position_error_mm": result.position_error_mm,
        "rx_error_deg": result.rx_error_deg,
        "ry_error_deg": result.ry_error_deg,
        "rz_error_deg": result.rz_error_deg,
        "send_count": result.send_count,
        "joint_indices": [index + 1 for index in result.joint_indices],
    }
    return payload


def execute_joints_for_duration(
    controller: PiperController,
    joints_rad: tuple[float, ...] | list[float],
    speed: int,
    duration_s: float,
    period_s: float,
) -> int:
    deadline = time.monotonic() + duration_s
    send_count = 0
    while send_count == 0 or time.monotonic() < deadline:
        controller.move_joints(list(joints_rad), speed_percent=speed)
        send_count += 1
        if time.monotonic() >= deadline:
            break
        time.sleep(period_s)
    return send_count


def j6_speed(args: argparse.Namespace) -> int:
    """Return the effective speed for a J6-only rotation.

    ``--j6-speed`` is the explicit option.  ``--grasp-yaw-speed`` remains
    accepted as a compatibility alias for callers using the older name.  If
    neither is provided, J6 follows the normal sequence ``--speed``.
    """

    value = getattr(args, "j6_speed", None)
    if value is None:
        value = getattr(args, "grasp_yaw_speed", None)
    return int(args.speed if value is None else value)


def planar_ik_speed(args: argparse.Namespace) -> int:
    value = getattr(args, "planar_ik_speed", None)
    return int(args.speed if value is None else value)


def planar_ik_joint_indices_for_step(step: SequenceStep) -> tuple[int, ...]:
    return (
        PLANAR_IK_YAW_JOINT_INDICES
        if getattr(step, "planar_ik_include_j6_yaw", False)
        else PLANAR_IK_JOINT_INDICES
    )


def planar_ik_joint_label(joint_indices: tuple[int, ...]) -> str:
    return "/".join(f"J{index + 1}" for index in joint_indices)


def planar_ik_cache_key(
    args: argparse.Namespace,
    step: SequenceStep,
    pitch_source_x_mm: float,
) -> PlanarIkCacheKey:
    decision = joint_pitch_decision(args, pitch_source_x_mm)
    return (
        round(float(step.pose.x_mm), 3),
        round(float(step.pose.y_mm), 3),
        round(float(step.pose.z_mm), 3),
        round(float(step.pose.rx_deg), 3),
        round(float(step.pose.ry_deg), 3),
        round(float(step.pose.rz_deg), 3),
        round(float(pitch_source_x_mm), 3),
        round(float(decision.requested_pitch_delta_deg), 6),
        round(float(args.planar_ik_fk_tool_z_offset_mm), 3),
        *tuple(float(index) for index in planar_ik_joint_indices_for_step(step)),
    )


def can_reuse_planar_ik_target(
    args: argparse.Namespace,
    step: SequenceStep,
    reuse_state: PlanarIkReuseState | None,
) -> bool:
    return bool(
        getattr(args, "reuse_planar_ik_above_target", True)
        and reuse_state is not None
        and step.planar_ik
        and step.label.endswith("_lift")
        and step.gripper_after is None
    )


def execute_reused_planar_ik_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    cached: PlanarIkCachedTarget,
) -> tuple[int, PlanarIkExecutionResult]:
    initial_joint_feedback = controller.get_joint_state()
    initial_feedback_stamp = joint_feedback_timestamp(initial_joint_feedback)
    joint_indices = planar_ik_joint_indices_for_step(step)
    print(
        f"Reusing planar IK {planar_ik_joint_label(joint_indices)}: "
        f"step={step.label} from={cached.step_label} "
        f"target={format_pose(step.pose)} "
        f"speed={planar_ik_speed(args)}",
        flush=True,
    )
    send_count = execute_joints_for_duration(
        controller,
        cached.target_joints_rad,
        planar_ik_speed(args),
        float(args.command_duration_s),
        float(args.command_period_s),
    )
    final_joints: tuple[float, ...] | None = None
    final_joint_errors: tuple[float, ...] | None = None
    max_joint_error: float | None = None
    if not args.no_wait_waypoints:
        final_joints, final_joint_errors, max_joint_error = (
            wait_until_planar_ik_joints_reached(
                controller,
                step.label,
                cached.target_joints_rad,
                timeout_s=args.motion_timeout_s,
                joint_tolerance_deg=float(args.planar_ik_joint_tolerance_deg),
                joint_indices=joint_indices,
                initial_feedback_stamp=initial_feedback_stamp,
            )
        )
    else:
        feedback_joints = joint_feedback_to_rad(controller.get_joint_state())
        if feedback_joints is not None:
            final_joints = feedback_joints
            final_joint_errors = joint_errors_deg(
                feedback_joints,
                cached.target_joints_rad,
                joint_indices,
            )
            max_joint_error = max(abs(value) for value in final_joint_errors)
    final_pose = feedback_to_pose(controller.get_end_pose())
    if final_joints is not None and final_joint_errors is not None:
        print(
            "Planar IK reuse feedback: "
            f"step={step.label} "
            f"joint_errors={tuple(round(value, 3) for value in final_joint_errors)}deg "
            f"max_err={max_joint_error:.3f}deg "
            f"final_pose={format_pose(final_pose)}",
            flush=True,
        )
    reused = replace(
        cached.result,
        step_label=step.label,
        target_pose=step.pose,
        send_count=send_count,
        final_joints_deg=(
            None if final_joints is None else joints_rad_to_deg(final_joints)
        ),
        joint_errors_deg=(
            None
            if final_joint_errors is None
            else (
                *(float(value) for value in final_joint_errors),
            )
        ),
        max_joint_error_deg=max_joint_error,
        final_pose=final_pose,
        reused=True,
        reused_from_step_label=cached.step_label,
        joint_indices=joint_indices,
    )
    return send_count, reused


def clamp_pitch_delta_to_joint_limits(
    controller: PiperController,
    current_joints: tuple[float, ...],
    requested_pitch_delta_deg: float,
    args: argparse.Namespace,
) -> float:
    lower = -float("inf")
    upper = float("inf")
    ratios = (
        (1, float(args.dynamic_pitch_j2_ratio)),
        (2, float(args.dynamic_pitch_j3_ratio)),
        (4, float(args.dynamic_pitch_j5_ratio)),
    )
    # Keep a small margin so rounding to SDK milli-degrees cannot land just outside limits.
    margin_rad = math.radians(0.1)
    for joint_index, ratio in ratios:
        if math.isclose(ratio, 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        limit = controller.config.joint_limits_rad[joint_index]
        current_deg = math.degrees(float(current_joints[joint_index]))
        min_delta_deg = math.degrees(limit.min_value + margin_rad) - current_deg
        max_delta_deg = math.degrees(limit.max_value - margin_rad) - current_deg
        joint_lower = min_delta_deg / ratio
        joint_upper = max_delta_deg / ratio
        if joint_lower > joint_upper:
            joint_lower, joint_upper = joint_upper, joint_lower
        lower = max(lower, joint_lower)
        upper = min(upper, joint_upper)
    if lower > upper:
        raise SafetyError(
            "dynamic joint pitch has no feasible delta within J2/J3/J5 limits: "
            f"requested={requested_pitch_delta_deg:.3f}deg "
            f"feasible=[{lower:.3f},{upper:.3f}]deg"
        )
    return min(upper, max(lower, requested_pitch_delta_deg))


def validate_planar_ik_joint_delta_limits(
    current_joints: tuple[float, ...] | list[float],
    target_joints: tuple[float, ...] | list[float],
    joint_indices: tuple[int, ...],
    *,
    label: str,
) -> None:
    for joint_index in PLANAR_IK_YAW_DELTA_LIMIT_JOINT_INDICES:
        if joint_index not in joint_indices:
            continue
        delta_deg = normalize_degrees(
            math.degrees(float(target_joints[joint_index]) - float(current_joints[joint_index]))
        )
        if abs(delta_deg) > PLANAR_IK_YAW_DELTA_LIMIT_DEG:
            raise SafetyError(
                f"planar IK {label} requires J{joint_index + 1} delta "
                f"{delta_deg:.3f}deg, exceeding +/-"
                f"{PLANAR_IK_YAW_DELTA_LIMIT_DEG:.1f}deg limit"
            )


def _planar_ik_residual(
    joints_rad: tuple[float, ...] | list[float],
    args: argparse.Namespace,
    target_pose: EndPose,
    target_ry_deg: float,
    *,
    include_rx: bool = False,
    include_rz: bool = False,
) -> np.ndarray:
    pose = forward_kinematics_planar_ik_pose(joints_rad, args)
    residual = [
        (pose.x_mm - target_pose.x_mm) / 100.0,
        (pose.z_mm - target_pose.z_mm) / 100.0,
    ]
    if include_rx:
        residual.append(normalize_degrees(pose.rx_deg - target_pose.rx_deg) / 10.0)
    residual.append(normalize_degrees(pose.ry_deg - target_ry_deg) / 10.0)
    if include_rz:
        residual.append(normalize_degrees(pose.rz_deg - target_pose.rz_deg) / 10.0)
    return np.array(residual, dtype=float)


def solve_planar_ik_j2_j3_j5(
    controller: PiperController,
    args: argparse.Namespace,
    current_joints: tuple[float, ...],
    target_pose: EndPose,
    pitch_source_x_mm: float,
    joint_indices: tuple[int, ...] = PLANAR_IK_JOINT_INDICES,
) -> PlanarIkSolution:
    decision = joint_pitch_decision(args, pitch_source_x_mm)
    target_ry_deg = float(args.ry_deg) + float(decision.requested_pitch_delta_deg)
    include_rx = 3 in joint_indices
    include_rz = 5 in joint_indices
    joints = np.array(current_joints, dtype=float)
    max_step_rad = math.radians(float(args.planar_ik_max_step_deg))
    damping = float(args.planar_ik_damping)
    iterations = 0

    for iteration in range(int(args.planar_ik_max_iterations)):
        iterations = iteration + 1
        residual = _planar_ik_residual(
            joints,
            args,
            target_pose,
            target_ry_deg,
            include_rx=include_rx,
            include_rz=include_rz,
        )
        solved_pose = forward_kinematics_planar_ik_pose(joints, args)
        position_error_mm = math.sqrt(
            (solved_pose.x_mm - target_pose.x_mm) ** 2
            + (solved_pose.z_mm - target_pose.z_mm) ** 2
        )
        rx_error_deg = abs(normalize_degrees(solved_pose.rx_deg - target_pose.rx_deg))
        ry_error_deg = abs(normalize_degrees(solved_pose.ry_deg - target_ry_deg))
        rz_error_deg = abs(normalize_degrees(solved_pose.rz_deg - target_pose.rz_deg))
        if (
            position_error_mm <= float(args.planar_ik_position_tolerance_mm)
            and (
                not include_rx
                or rx_error_deg <= float(args.planar_ik_rx_tolerance_deg)
            )
            and ry_error_deg <= float(args.planar_ik_ry_tolerance_deg)
            and (
                not include_rz
                or rz_error_deg <= float(args.planar_ik_rz_tolerance_deg)
            )
        ):
            break

        jacobian = np.zeros((len(residual), len(joint_indices)), dtype=float)
        epsilon = 1e-4
        for column, joint_index in enumerate(joint_indices):
            plus = joints.copy()
            plus[joint_index] += epsilon
            minus = joints.copy()
            minus[joint_index] -= epsilon
            jacobian[:, column] = (
                _planar_ik_residual(
                    plus,
                    args,
                    target_pose,
                    target_ry_deg,
                    include_rx=include_rx,
                    include_rz=include_rz,
                )
                - _planar_ik_residual(
                    minus,
                    args,
                    target_pose,
                    target_ry_deg,
                    include_rx=include_rx,
                    include_rz=include_rz,
                )
            ) / (2.0 * epsilon)

        try:
            delta = np.linalg.solve(
                jacobian.T @ jacobian
                + damping * np.eye(len(joint_indices), dtype=float),
                -jacobian.T @ residual,
            )
        except np.linalg.LinAlgError as exc:
            raise SafetyError(
                f"planar IK failed: singular {planar_ik_joint_label(joint_indices)} Jacobian"
            ) from exc
        max_abs_delta = float(np.max(np.abs(delta)))
        if max_abs_delta > max_step_rad:
            delta *= max_step_rad / max_abs_delta

        for delta_value, joint_index in zip(delta, joint_indices):
            limit = controller.config.joint_limits_rad[joint_index]
            joints[joint_index] = min(
                limit.max_value - math.radians(0.1),
                max(limit.min_value + math.radians(0.1), joints[joint_index] + delta_value),
            )

    target_joints = tuple(float(value) for value in joints)
    controller.validate_joints(target_joints)
    validate_planar_ik_joint_delta_limits(
        current_joints,
        target_joints,
        joint_indices,
        label=format_pose(target_pose),
    )
    solved_fk_pose = forward_kinematics_pose(target_joints)
    solved_pose = apply_tool_z_offset_to_fk_pose(
        solved_fk_pose,
        float(args.planar_ik_fk_tool_z_offset_mm),
    )
    x_error_mm = solved_pose.x_mm - target_pose.x_mm
    y_error_mm = solved_pose.y_mm - target_pose.y_mm
    z_error_mm = solved_pose.z_mm - target_pose.z_mm
    position_error_mm = math.sqrt(x_error_mm**2 + z_error_mm**2)
    rx_error_deg = normalize_degrees(solved_pose.rx_deg - target_pose.rx_deg)
    ry_error_deg = normalize_degrees(solved_pose.ry_deg - target_ry_deg)
    rz_error_deg = normalize_degrees(solved_pose.rz_deg - target_pose.rz_deg)
    if (
        position_error_mm > float(args.planar_ik_position_tolerance_mm)
        or (
            include_rx
            and abs(rx_error_deg) > float(args.planar_ik_rx_tolerance_deg)
        )
        or abs(ry_error_deg) > float(args.planar_ik_ry_tolerance_deg)
        or (
            include_rz
            and abs(rz_error_deg) > float(args.planar_ik_rz_tolerance_deg)
        )
    ):
        rz_error_text = (
            f" rz_err={rz_error_deg:.3f}deg"
            if include_rz
            else ""
        )
        rx_error_text = (
            f"rx_err={rx_error_deg:.3f}deg "
            if include_rx
            else ""
        )
        raise SafetyError(
            "planar IK did not converge within tolerance: "
            f"target={format_pose(target_pose)} "
            f"target_ry={target_ry_deg:.3f}deg "
            f"solved={format_pose(solved_pose)} "
            f"x_err={x_error_mm:.3f}mm z_err={z_error_mm:.3f}mm "
            f"{rx_error_text}"
            f"ry_err={ry_error_deg:.3f}deg "
            f"{rz_error_text}"
            f"iterations={iterations}"
        )
    return PlanarIkSolution(
        target_joints_rad=target_joints,  # type: ignore[arg-type]
        solved_pose=solved_pose,
        solved_fk_pose=solved_fk_pose,
        pitch_delta_deg=normalize_degrees(target_ry_deg - float(args.ry_deg)),
        requested_pitch_delta_deg=decision.requested_pitch_delta_deg,
        iterations=iterations,
        x_error_mm=x_error_mm,
        y_error_mm=y_error_mm,
        z_error_mm=z_error_mm,
        position_error_mm=position_error_mm,
        ry_error_deg=ry_error_deg,
        rx_error_deg=rx_error_deg if include_rx else 0.0,
        rz_error_deg=rz_error_deg if include_rz else 0.0,
        joint_indices=joint_indices,
    )


def execute_planar_ik_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
) -> tuple[int, PlanarIkExecutionResult]:
    pitch_source_x = (
        float(step.pitch_source_x_mm)
        if step.pitch_source_x_mm is not None
        else float(step.pose.x_mm)
    )
    initial_joint_feedback = controller.get_joint_state()
    initial_feedback_stamp = joint_feedback_timestamp(initial_joint_feedback)
    current_joints = joint_feedback_to_rad(initial_joint_feedback)
    if current_joints is None:
        raise RuntimeError("planar IK requires valid joint feedback before JointCtrl")
    joint_indices = planar_ik_joint_indices_for_step(step)
    solution = solve_planar_ik_j2_j3_j5(
        controller,
        args,
        current_joints,
        step.pose,
        pitch_source_x,
        joint_indices=joint_indices,
    )
    print(
        f"Running planar IK {planar_ik_joint_label(joint_indices)}: "
        f"step={step.label} "
        f"target={format_pose(step.pose)} "
        f"solved={format_pose(solution.solved_pose)} "
        f"raw_fk={format_pose(solution.solved_fk_pose)} "
        f"fk_tool_z_offset={float(args.planar_ik_fk_tool_z_offset_mm):.3f}mm "
        f"pitch_delta={solution.pitch_delta_deg:.3f}deg "
        f"x_err={solution.x_error_mm:.3f}mm "
        f"z_err={solution.z_error_mm:.3f}mm "
        f"rx_err={solution.rx_error_deg:.3f}deg "
        f"ry_err={solution.ry_error_deg:.3f}deg "
        f"rz_err={solution.rz_error_deg:.3f}deg "
        f"speed={planar_ik_speed(args)}",
        flush=True,
    )
    send_count = execute_joints_for_duration(
        controller,
        solution.target_joints_rad,
        planar_ik_speed(args),
        float(args.command_duration_s),
        float(args.command_period_s),
    )
    final_joints: tuple[float, ...] | None = None
    final_joint_errors: tuple[float, ...] | None = None
    max_joint_error: float | None = None
    should_wait = step.require_reached or not args.no_wait_waypoints
    if should_wait:
        final_joints, final_joint_errors, max_joint_error = (
            wait_until_planar_ik_joints_reached(
                controller,
                step.label,
                solution.target_joints_rad,
                timeout_s=args.motion_timeout_s,
                joint_tolerance_deg=float(args.planar_ik_joint_tolerance_deg),
                joint_indices=joint_indices,
                initial_feedback_stamp=initial_feedback_stamp,
            )
        )
    else:
        feedback_joints = joint_feedback_to_rad(controller.get_joint_state())
        if feedback_joints is not None:
            final_joints = feedback_joints
            final_joint_errors = joint_errors_deg(
                feedback_joints,
                solution.target_joints_rad,
                joint_indices,
            )
            max_joint_error = max(abs(value) for value in final_joint_errors)
    final_pose = feedback_to_pose(controller.get_end_pose())
    if final_joints is not None and final_joint_errors is not None:
        print(
            "Planar IK joint feedback: "
            f"step={step.label} "
            f"joint_errors={tuple(round(value, 3) for value in final_joint_errors)}deg "
            f"max_err={max_joint_error:.3f}deg "
            f"final_pose={format_pose(final_pose)}",
            flush=True,
        )
    elif final_pose is not None:
        print(
            "Planar IK feedback pose: "
            f"step={step.label} final_pose={format_pose(final_pose)}",
            flush=True,
        )
    return send_count, PlanarIkExecutionResult(
        step_label=step.label,
        pitch_source_x_mm=pitch_source_x,
        target_pose=step.pose,
        solved_pose=solution.solved_pose,
        start_joints_deg=joints_rad_to_deg(current_joints),  # type: ignore[arg-type]
        target_joints_deg=joints_rad_to_deg(solution.target_joints_rad),
        pitch_delta_deg=solution.pitch_delta_deg,
        requested_pitch_delta_deg=solution.requested_pitch_delta_deg,
        iterations=solution.iterations,
        x_error_mm=solution.x_error_mm,
        y_error_mm=solution.y_error_mm,
        z_error_mm=solution.z_error_mm,
        position_error_mm=solution.position_error_mm,
        rx_error_deg=solution.rx_error_deg,
        ry_error_deg=solution.ry_error_deg,
        rz_error_deg=solution.rz_error_deg,
        send_count=send_count,
        solved_fk_pose=solution.solved_fk_pose,
        final_joints_deg=(
            None if final_joints is None else joints_rad_to_deg(final_joints)
        ),
        joint_errors_deg=(
            None
            if final_joint_errors is None
            else (
                *(float(value) for value in final_joint_errors),
            )
        ),
        max_joint_error_deg=max_joint_error,
        final_pose=final_pose,
        joint_indices=joint_indices,
    )


def apply_joint_pitch_adjustment(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    *,
    phase: str,
) -> JointPitchAdjustmentResult | None:
    pitch_source_x = (
        float(step.pitch_source_x_mm)
        if step.pitch_source_x_mm is not None
        else float(step.pose.x_mm)
    )
    requested_decision = joint_pitch_decision(args, pitch_source_x)
    if (
        requested_decision.mode != "adjust"
        or math.isclose(
            requested_decision.requested_pitch_delta_deg,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return None

    current_joints = joint_feedback_to_rad(controller.get_joint_state())
    if current_joints is None:
        raise RuntimeError(
            "dynamic joint pitch requires valid joint feedback before JointCtrl"
        )
    pitch_delta_deg = clamp_pitch_delta_to_joint_limits(
        controller,
        current_joints,
        requested_decision.requested_pitch_delta_deg,
        args,
    )
    decision = joint_pitch_decision(
        args,
        pitch_source_x,
        feasible_pitch_delta_deg=pitch_delta_deg,
    )
    if math.isclose(decision.pitch_delta_deg, 0.0, rel_tol=0.0, abs_tol=1e-9):
        return None
    target_joints = list(current_joints)
    j2_delta_deg = decision.j2_delta_deg
    j3_delta_deg = decision.j3_delta_deg
    j5_delta_deg = decision.j5_delta_deg
    target_joints[1] += math.radians(j2_delta_deg)
    target_joints[2] += math.radians(j3_delta_deg)
    target_joints[4] += math.radians(j5_delta_deg)

    print(
        "Applying dynamic joint pitch: "
        f"step={step.label} phase={phase} "
        f"pick_x={step.pose.x_mm:.3f}mm "
        f"source_x={pitch_source_x:.3f}mm "
        f"mode={decision.mode} "
        f"over={decision.over_mm:.3f}mm "
        f"pitch_delta={decision.pitch_delta_deg:.3f}deg "
        f"requested={decision.requested_pitch_delta_deg:.3f}deg "
        f"j2={j2_delta_deg:.3f}deg "
        f"j3={j3_delta_deg:.3f}deg "
        f"j5={j5_delta_deg:.3f}deg",
        flush=True,
    )
    send_count = execute_joints_for_duration(
        controller,
        target_joints,
        int(args.dynamic_pitch_speed),
        float(args.command_duration_s),
        float(args.command_period_s),
    )
    return JointPitchAdjustmentResult(
        step_label=step.label,
        phase=phase,
        pick_x_mm=float(step.pose.x_mm),
        pitch_source_x_mm=pitch_source_x,
        raw_pitch_delta_deg=decision.raw_pitch_delta_deg,
        pitch_delta_deg=decision.pitch_delta_deg,
        requested_pitch_delta_deg=decision.requested_pitch_delta_deg,
        clamped=decision.clamped,
        limit_clamped=decision.limit_clamped,
        j2_delta_deg=j2_delta_deg,
        j3_delta_deg=j3_delta_deg,
        j5_delta_deg=j5_delta_deg,
        start_joints_deg=joints_rad_to_deg(current_joints),  # type: ignore[arg-type]
        target_joints_deg=joints_rad_to_deg(target_joints),
        send_count=send_count,
    )


def nearest_axis_equivalent_delta_deg(target_deg: float, current_deg: float) -> float:
    """Return nearest delta where target and target+180 describe same gripper axis."""

    delta = normalize_degrees(float(target_deg) - float(current_deg))
    if delta > 90.0:
        delta -= 180.0
    elif delta < -90.0:
        delta += 180.0
    return delta


def plan_grasp_yaw_delta(
    target_deg: float,
    current_deg: float,
    *,
    max_delta_deg: float,
    min_j6_deg: float,
    max_j6_deg: float,
) -> GraspYawPlan:
    """Choose the shortest reachable rotation for the gripper's axis.

    The closing axis is unoriented, so both ``target`` and ``target + 180°``
    describe the same grasp.  Each equivalent target is evaluated in its
    shortest signed direction (positive is counter-clockwise, negative is
    clockwise), then the candidate with the smallest resulting axis error is
    selected.  If both directions are equally good, clockwise is selected
    deterministically (the requested policy is "counter-clockwise only when
    it is strictly shorter; otherwise clockwise").
    """

    candidates: list[GraspYawPlan] = []
    target_axis_deg = normalize_degrees(target_deg)
    for equivalent_target_deg in (
        target_axis_deg,
        normalize_degrees(target_axis_deg + 180.0),
    ):
        raw_delta_deg = normalize_degrees(equivalent_target_deg - float(current_deg))
        delta_limited_deg = min(
            float(max_delta_deg),
            max(-float(max_delta_deg), raw_delta_deg),
        )
        requested_j6_deg = float(current_deg) + delta_limited_deg
        target_j6_deg = min(float(max_j6_deg), max(float(min_j6_deg), requested_j6_deg))
        delta_deg = target_j6_deg - float(current_deg)
        reached_axis_delta = nearest_axis_equivalent_delta_deg(target_axis_deg, target_j6_deg)
        candidates.append(
            GraspYawPlan(
                raw_delta_deg=raw_delta_deg,
                delta_deg=delta_deg,
                target_j6_deg=target_j6_deg,
                axis_error_deg=abs(reached_axis_delta),
                delta_clamped=not math.isclose(
                    delta_limited_deg,
                    raw_delta_deg,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                limit_clamped=not math.isclose(
                    target_j6_deg,
                    requested_j6_deg,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
            )
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.axis_error_deg,
            candidate.limit_clamped,
            abs(candidate.delta_deg),
            # On an exact tie prefer clockwise (negative delta).  This also
            # makes the 90°/90° axis case follow the documented policy.
            candidate.delta_deg >= 0.0,
        ),
    )


def compute_grasp_yaw_adjustment_result(
    controller: PiperController,
    args: argparse.Namespace,
    target: YoloCatchTarget,
    *,
    execute: bool,
) -> tuple[GraspYawAdjustmentResult, tuple[float, ...] | None]:
    if not getattr(args, "grasp_yaw_by_object_short_edge", False):
        return GraspYawAdjustmentResult(enabled=False, reason="disabled"), None

    source_yaw = target.camera_command.object_short_axis_yaw_deg
    if source_yaw is None or not math.isfinite(float(source_yaw)):
        return (
            GraspYawAdjustmentResult(
                enabled=True,
                reason="missing_object_short_axis_yaw_deg",
            ),
            None,
        )

    current_joints = joint_feedback_to_rad(controller.get_joint_state())
    if current_joints is None:
        raise RuntimeError("grasp yaw adjustment requires valid joint feedback")

    start_joints_deg = joints_rad_to_deg(current_joints)  # type: ignore[arg-type]
    current_j6_deg = float(start_joints_deg[5])
    target_yaw_deg = normalize_degrees(
        float(source_yaw) + float(args.gripper_close_axis_offset_deg)
    )
    joint6_limit = controller.config.joint_limits_rad[5]
    margin_rad = math.radians(0.1)
    min_j6_deg = math.degrees(joint6_limit.min_value + margin_rad)
    max_j6_deg = math.degrees(joint6_limit.max_value - margin_rad)
    yaw_plan = plan_grasp_yaw_delta(
        target_yaw_deg,
        current_j6_deg,
        max_delta_deg=float(args.grasp_yaw_max_delta_deg),
        min_j6_deg=min_j6_deg,
        max_j6_deg=max_j6_deg,
    )
    raw_delta_deg = yaw_plan.raw_delta_deg
    if abs(yaw_plan.delta_deg) <= float(args.grasp_yaw_deadband_deg):
        return (
            GraspYawAdjustmentResult(
                enabled=True,
                applied=False,
                reason=(
                    "within_joint6_limit_deadband"
                    if yaw_plan.limit_clamped
                    else "within_deadband"
                ),
                source_yaw_deg=float(source_yaw),
                target_yaw_deg=target_yaw_deg,
                current_j6_deg=current_j6_deg,
                delta_deg=0.0,
                clamped=yaw_plan.delta_clamped or yaw_plan.limit_clamped,
                limit_clamped=yaw_plan.limit_clamped,
                start_joints_deg=start_joints_deg,  # type: ignore[arg-type]
                target_joints_deg=start_joints_deg,  # type: ignore[arg-type]
            ),
            current_joints,
        )

    delta_deg = yaw_plan.delta_deg
    clamped = yaw_plan.delta_clamped or yaw_plan.limit_clamped
    target_joints = list(current_joints)
    target_joints[5] += math.radians(delta_deg)
    controller.validate_joints(target_joints)

    print(
        (
            "Applying object short-edge grasp yaw: "
            if execute
            else "Planning integrated planar-IK object short-edge grasp yaw: "
        )
        + f"source_yaw={float(source_yaw):.3f}deg "
        f"offset={float(args.gripper_close_axis_offset_deg):.3f}deg "
        f"target_yaw={target_yaw_deg:.3f}deg "
        f"current_j6={current_j6_deg:.3f}deg "
        f"delta={delta_deg:.3f}deg "
        f"raw_delta={raw_delta_deg:.3f}deg "
        f"axis_err={yaw_plan.axis_error_deg:.3f}deg "
        f"clamped={clamped} "
        f"limit_clamped={yaw_plan.limit_clamped}",
        flush=True,
    )
    send_count = 0
    if execute:
        send_count = execute_joints_for_duration(
            controller,
            target_joints,
            j6_speed(args),
            float(args.command_duration_s),
            float(args.command_period_s),
        )
        if not args.no_wait_waypoints:
            wait_until_planar_ik_joints_reached(
                controller,
                "grasp_yaw",
                target_joints,
                timeout_s=args.motion_timeout_s,
                joint_tolerance_deg=float(args.planar_ik_joint_tolerance_deg),
                joint_indices=(5,),
            )
    return (
        GraspYawAdjustmentResult(
            enabled=True,
            applied=True,
            source_yaw_deg=float(source_yaw),
            target_yaw_deg=target_yaw_deg,
            current_j6_deg=current_j6_deg,
            delta_deg=delta_deg,
            clamped=clamped,
            limit_clamped=yaw_plan.limit_clamped,
            start_joints_deg=start_joints_deg,  # type: ignore[arg-type]
            target_joints_deg=joints_rad_to_deg(target_joints),
            send_count=send_count,
        ),
        tuple(float(value) for value in target_joints),
    )


def apply_grasp_yaw_adjustment(
    controller: PiperController,
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> GraspYawAdjustmentResult:
    result, _ = compute_grasp_yaw_adjustment_result(
        controller,
        args,
        target,
        execute=True,
    )
    return result


def should_inherit_post_yaw_orientation(result: GraspYawAdjustmentResult) -> bool:
    return bool(
        result.applied
        or result.reason in ("within_deadband", "within_joint6_limit_deadband")
    )


def with_current_orientation(pose: EndPose, current_pose: EndPose) -> EndPose:
    return replace(
        pose,
        rx_deg=current_pose.rx_deg,
        ry_deg=current_pose.ry_deg,
        rz_deg=current_pose.rz_deg,
    )


def post_grasp_yaw_orientation(
    before_pose: EndPose | None,
    after_pose: EndPose | None,
    result: GraspYawAdjustmentResult,
) -> EndPose | None:
    if not should_inherit_post_yaw_orientation(result):
        return None
    current_pose = after_pose
    if (
        result.applied
        and result.delta_deg is not None
        and before_pose is not None
        and (
            after_pose is None
            or abs(normalize_degrees(after_pose.rz_deg - before_pose.rz_deg))
            < min(1.0, max(0.1, abs(float(result.delta_deg)) * 0.25))
        )
    ):
        current_pose = replace(
            before_pose,
            rz_deg=normalize_degrees(before_pose.rz_deg + float(result.delta_deg)),
        )
    return current_pose


def apply_grasp_yaw_and_get_orientation(
    controller: PiperController,
    args: argparse.Namespace,
    target: YoloCatchTarget,
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult],
    *,
    phase: str,
) -> EndPose | None:
    before_pose = feedback_to_pose(controller.get_end_pose())
    result = apply_grasp_yaw_adjustment(controller, args, target)
    grasp_yaw_adjustments.append(result)
    if not should_inherit_post_yaw_orientation(result):
        return None
    current_pose = post_grasp_yaw_orientation(
        before_pose,
        feedback_to_pose(controller.get_end_pose()),
        result,
    )
    if current_pose is None:
        raise RuntimeError(
            f"grasp yaw {phase} requires valid end-pose feedback "
            "to preserve the rotated orientation"
        )
    return current_pose


def plan_grasp_yaw_orientation_for_planar_ik(
    controller: PiperController,
    args: argparse.Namespace,
    target: YoloCatchTarget,
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult],
    *,
    phase: str,
) -> EndPose | None:
    before_pose = feedback_to_pose(controller.get_end_pose())
    result, _target_joints = compute_grasp_yaw_adjustment_result(
        controller,
        args,
        target,
        execute=False,
    )
    grasp_yaw_adjustments.append(result)
    if not should_inherit_post_yaw_orientation(result):
        return None
    if before_pose is None:
        raise RuntimeError(
            f"grasp yaw {phase} requires valid end-pose feedback "
            "to plan the integrated planar-IK yaw orientation"
        )
    if result.applied and result.delta_deg is not None:
        return replace(
            before_pose,
            rz_deg=normalize_degrees(before_pose.rz_deg + float(result.delta_deg)),
        )
    return before_pose


def apply_grasp_yaw_before_descent(
    controller: PiperController,
    args: argparse.Namespace,
    target: YoloCatchTarget,
    descent_step: SequenceStep,
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult],
) -> SequenceStep:
    current_pose = apply_grasp_yaw_and_get_orientation(
        controller,
        args,
        target,
        grasp_yaw_adjustments,
        phase="before descent",
    )
    if current_pose is None:
        return descent_step
    updated_pose = with_current_orientation(descent_step.pose, current_pose)
    print(
        "Using post-yaw orientation for grasp descent: "
        f"step={descent_step.label} orientation="
        f"({updated_pose.rx_deg:.3f}, {updated_pose.ry_deg:.3f}, "
        f"{updated_pose.rz_deg:.3f})",
        flush=True,
    )
    return replace(descent_step, pose=updated_pose)


def execute_and_record_step(
    controller: PiperController,
    args: argparse.Namespace,
    step: SequenceStep,
    gripper_results: list[SafeGripperResult],
    joint_pitch_adjustments: list[JointPitchAdjustmentResult] | None = None,
    planar_ik_results: list[PlanarIkExecutionResult] | None = None,
    planar_ik_reuse_state: PlanarIkReuseState | None = None,
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult] | None = None,
) -> int:
    send_count, gripper_result = execute_step(
        controller,
        args,
        step,
        joint_pitch_adjustments=joint_pitch_adjustments,
        grasp_yaw_adjustments=grasp_yaw_adjustments,
        planar_ik_results=planar_ik_results,
        planar_ik_reuse_state=planar_ik_reuse_state,
    )
    record_gripper_result(gripper_results, gripper_result)
    return send_count


def execute_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: YoloCatchPlan,
) -> DynamicExecutionResult:
    before_pose = None
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before catch J6: {format_pose(before_pose)}", flush=True)

    counts: dict[str, int] = {}
    gripper_results: list[SafeGripperResult] = []
    joint_pitch_adjustments: list[JointPitchAdjustmentResult] = []
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult] = []
    planar_ik_results: list[PlanarIkExecutionResult] = []
    planar_ik_reuse_state = PlanarIkReuseState()
    post_yaw_orientation: EndPose | None = None
    yaw_timing = grasp_yaw_timing_value(args)
    for step in plan.steps:
        active_step = step
        if (
            post_yaw_orientation is not None
            and (
                active_step.label in ("pick_grasp", "return_pick_grasp")
                or (plan.task == "in" and active_step.label == "return_pick_lift")
            )
        ):
            active_step = replace(
                active_step,
                pose=with_current_orientation(active_step.pose, post_yaw_orientation),
            )
        rotate_at_grasp = (
            yaw_timing == "pick_grasp"
            and active_step.label in ("pick_grasp", "return_pick_grasp")
        )
        # A pick_grasp step normally closes immediately after the Cartesian
        # descent.  When rotation is requested at that phase, defer the close
        # until after the J6 move so the gripper rotates while still open at
        # the grasp height.
        step_for_execution = (
            replace(active_step, gripper_after=None)
            if rotate_at_grasp
            else active_step
        )
        send_count, gripper_result = execute_step(
            controller,
            args,
            step_for_execution,
            joint_pitch_adjustments=joint_pitch_adjustments,
            grasp_yaw_adjustments=grasp_yaw_adjustments,
            planar_ik_results=planar_ik_results,
            planar_ik_reuse_state=planar_ik_reuse_state,
        )
        counts[active_step.label] = send_count
        record_gripper_result(gripper_results, gripper_result)

        # Apply the object yaw at the configured phase.  At pick_above the
        # result is inherited by the descent; at pick_grasp it is applied after
        # the descent and before the close command.
        if should_apply_grasp_yaw_after_step(args, active_step.label):
            post_yaw_orientation = apply_grasp_yaw_and_get_orientation(
                controller,
                args,
                plan.target,
                grasp_yaw_adjustments,
                phase=f"after {active_step.label}",
            )
            if post_yaw_orientation is not None:
                print(
                    "Using post-yaw orientation: "
                    f"orientation=({post_yaw_orientation.rx_deg:.3f}, "
                    f"{post_yaw_orientation.ry_deg:.3f}, "
                    f"{post_yaw_orientation.rz_deg:.3f})",
                    flush=True,
                )
        if rotate_at_grasp:
            close_result = send_gripper(controller, args, active_step.gripper_after)
            record_gripper_result(gripper_results, close_result)
    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)
    return DynamicExecutionResult(
        send_counts=counts,
        safe_gripper_results=tuple(gripper_results),
        joint_pitch_adjustments=tuple(joint_pitch_adjustments),
        grasp_yaw_adjustments=tuple(grasp_yaw_adjustments),
        planar_ik_results=tuple(planar_ik_results),
    )


def execute_dynamic_preposition_plan(
    controller: PiperController,
    args: argparse.Namespace,
    plan: YoloCatchPlan,
    timing: DynamicTiming,
) -> DynamicExecutionResult:
    if len(plan.steps) < 4:
        raise RuntimeError("dynamic preposition requires at least ready/above/grasp/lift steps")

    before_pose = None
    if args.print_feedback:
        before_pose = feedback_to_pose(controller.get_end_pose())
        print(f"Before catch J6: {format_pose(before_pose)}", flush=True)

    counts: dict[str, int] = {}
    gripper_results: list[SafeGripperResult] = []
    joint_pitch_adjustments: list[JointPitchAdjustmentResult] = []
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult] = []
    planar_ik_results: list[PlanarIkExecutionResult] = []
    planar_ik_reuse_state = PlanarIkReuseState()
    started_s = time.monotonic()
    yaw_timing = grasp_yaw_timing_value(args)
    ready_step = plan.steps[0]
    pick_above_step = plan.steps[1]
    grasp_step = replace(plan.steps[2], delay_before_s=0.0)
    lift_step = plan.steps[3]
    preposition_pose, preposition_source = dynamic_preposition_pose(args, plan.target)
    preposition_step = SequenceStep(
        "dynamic_preposition",
        preposition_pose,
        "P",
    )
    nowait = use_dynamic_nowait(args)

    counts[ready_step.label] = execute_and_record_step(
        controller,
        args,
        ready_step,
        gripper_results,
        joint_pitch_adjustments,
        planar_ik_results,
        planar_ik_reuse_state,
    )
    ready_elapsed_s = time.monotonic() - started_s
    final_action_due_s = timing.t_arrive_s - float(args.final_action_time_s)
    ready_departure_due_s = timing.t_arrive_s - float(timing.t_robot_s)
    final_wait_s = 0.0
    final_action_late_s = 0.0
    if nowait:
        wait_for_dynamic_final_action(
            timing,
            elapsed_s=ready_elapsed_s,
            final_action_time_s=float(timing.t_robot_s),
            placement="nowait",
        )
        final_action_late_s = max(0.0, ready_elapsed_s - ready_departure_due_s)

    counts[pick_above_step.label] = execute_and_record_step(
        controller,
        args,
        pick_above_step,
        gripper_results,
        joint_pitch_adjustments,
        planar_ik_results,
        planar_ik_reuse_state,
    )
    # With the default timing, rotate at the safe pick_above height and let
    # the dynamic preposition/descent inherit the feedback orientation.  With
    # pick_grasp timing, keep the prepared orientation through the descent and
    # rotate only after reaching the grasp pose.
    if yaw_timing == "pick_above":
        post_yaw_orientation = apply_grasp_yaw_and_get_orientation(
            controller,
            args,
            plan.target,
            grasp_yaw_adjustments,
            phase="after pick_above",
        )
        if post_yaw_orientation is not None:
            preposition_step = replace(
                preposition_step,
                pose=with_current_orientation(preposition_step.pose, post_yaw_orientation),
            )
            grasp_step = replace(
                grasp_step,
                pose=with_current_orientation(grasp_step.pose, post_yaw_orientation),
            )
            print(
                "Using post-yaw orientation for dynamic descent: "
                f"orientation=({post_yaw_orientation.rx_deg:.3f}, "
                f"{post_yaw_orientation.ry_deg:.3f}, "
                f"{post_yaw_orientation.rz_deg:.3f})",
                flush=True,
            )
    counts[preposition_step.label] = execute_and_record_step(
        controller,
        args,
        preposition_step,
        gripper_results,
        joint_pitch_adjustments,
        planar_ik_results,
        planar_ik_reuse_state,
    )
    preposition_elapsed_s = time.monotonic() - started_s
    final_wait_s = wait_for_dynamic_final_action(
        timing,
        elapsed_s=preposition_elapsed_s,
        final_action_time_s=float(args.final_action_time_s),
        placement="wait" if not nowait else "preposition_after_nowait",
    )
    final_action_late_s = max(0.0, preposition_elapsed_s - final_action_due_s)

    rotate_at_grasp = yaw_timing == "pick_grasp"
    grasp_step_for_execution = (
        replace(grasp_step, gripper_after=None)
        if rotate_at_grasp
        else grasp_step
    )
    send_count, gripper_result = execute_step(
        controller,
        args,
        grasp_step_for_execution,
        joint_pitch_adjustments=joint_pitch_adjustments,
        grasp_yaw_adjustments=grasp_yaw_adjustments,
        planar_ik_results=planar_ik_results,
        planar_ik_reuse_state=planar_ik_reuse_state,
    )
    counts[grasp_step.label] = send_count
    record_gripper_result(gripper_results, gripper_result)
    if rotate_at_grasp:
        post_yaw_orientation = apply_grasp_yaw_and_get_orientation(
            controller,
            args,
            plan.target,
            grasp_yaw_adjustments,
            phase="after pick_grasp",
        )
        if post_yaw_orientation is not None:
            print(
                "Using post-yaw orientation after dynamic grasp: "
                f"orientation=({post_yaw_orientation.rx_deg:.3f}, "
                f"{post_yaw_orientation.ry_deg:.3f}, "
                f"{post_yaw_orientation.rz_deg:.3f})",
                flush=True,
            )
        close_result = send_gripper(controller, args, grasp_step.gripper_after)
        record_gripper_result(gripper_results, close_result)

    counts[lift_step.label] = execute_and_record_step(
        controller,
        args,
        lift_step,
        gripper_results,
        joint_pitch_adjustments,
        planar_ik_results,
        planar_ik_reuse_state,
    )
    for step in plan.steps[4:]:
        counts[step.label] = execute_and_record_step(
            controller,
            args,
            step,
            gripper_results,
            joint_pitch_adjustments,
            planar_ik_results,
            planar_ik_reuse_state,
        )

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)

    return DynamicExecutionResult(
        send_counts=counts,
        preposition_z_mm=preposition_pose.z_mm,
        preposition_source=preposition_source,
        preposition_elapsed_s=preposition_elapsed_s,
        final_action_due_s=final_action_due_s,
        final_wait_s=final_wait_s,
        final_action_late_s=final_action_late_s,
        safe_gripper_results=tuple(gripper_results),
        joint_pitch_adjustments=tuple(joint_pitch_adjustments),
        grasp_yaw_adjustments=tuple(grasp_yaw_adjustments),
        planar_ik_results=tuple(planar_ik_results),
    )


def maybe_prepare_gripper(controller: PiperController, args: argparse.Namespace) -> None:
    if args.no_gripper:
        return
    if args.clear_gripper_error:
        controller.set_gripper(
            args.open_command_mm,
            args.effort_nm,
            enable=True,
            clear_error=True,
        )
        time.sleep(args.gripper_settle_s)
    if not args.no_pre_open:
        controller.set_gripper(args.open_command_mm, args.effort_nm, enable=True)
        time.sleep(args.gripper_settle_s)


def move_to_start_ready(controller: PiperController, args: argparse.Namespace) -> int:
    step = SequenceStep("startup_ready", ready_pose(args), "P")
    validate_steps(controller, (step,))
    print(
        "Moving to startup ready pose from run_pose_continuous_catch_sequence: "
        f"{format_pose(step.pose)}",
        flush=True,
    )
    send_count, _ = execute_step(controller, args, step, speed=args.startup_speed)
    return send_count


def handle_message(
    message: dict[str, Any],
    *,
    args: argparse.Namespace,
    camera_to_base: Any,
    validator: PiperController,
    controller: PiperController | None = None,
) -> dict[str, Any]:
    command = camera_point_from_message(message)
    metadata = yolo_metadata_from_message(message)
    if command.grasp_mode == "probe":
        return handle_probe_message(
            command,
            metadata,
            camera_to_base=camera_to_base,
            args=args,
        )
    timing = dynamic_timing_from_command(
        command,
        args,
        source_x_mm=command.pick_x_mm,
    )
    if command.vision_source == "graspnet" or command.grasp_candidates:
        raise ValueError(
            "catch_server_1 no longer accepts GraspNet commands; "
            "use catch_server_graspnet.py"
        )
    active_validator = controller or validator
    target = target_from_command(
        command,
        camera_to_base,
        args,
    )
    plan = build_plan_for_target(command, target, args)
    validate_steps(active_validator, plan.steps)
    target = plan.target
    planned_preposition_pose: EndPose | None = None
    planned_preposition_source: str | None = None
    if timing is not None and args.dynamic_preposition:
        planned_preposition_pose, planned_preposition_source = dynamic_preposition_pose(
            args,
            target,
        )
        validate_steps(
            controller or validator,
            (
                SequenceStep(
                    "dynamic_preposition",
                    planned_preposition_pose,
                    "P",
                ),
            ),
        )

    if args.debug_plan or not args.execute:
        print_plan(plan, args, metadata, timing)

    execution = DynamicExecutionResult(
        send_counts={},
        preposition_z_mm=(
            None if planned_preposition_pose is None else planned_preposition_pose.z_mm
        ),
        preposition_source=planned_preposition_source,
    )
    skipped = bool(timing is not None and timing.skipped)
    if skipped and timing is not None:
        print(f"Dynamic target skipped: {timing.skip_reason}", flush=True)
    did_execute = bool(args.execute and not skipped)
    if did_execute:
        if controller is None:
            raise RuntimeError("execute=True requires a connected PiperController")
        if timing is not None and args.dynamic_preposition:
            execution = execute_dynamic_preposition_plan(controller, args, plan, timing)
        else:
            wait_for_dynamic_timing(timing)
            execution = execute_plan(controller, args, plan)

    print(
        f"task={plan.task} "
        f"mode={command.grasp_mode} "
        "camera_xyz="
        f"({command.x_m:.4f}, {command.y_m:.4f}, {command.z_m:.4f}) m -> "
        f"base_xy=({target.pick_x_mm:.3f}, {target.pick_y_mm:.3f}) mm "
        f"grasp_z={target.grasp_z_mm:.3f} mm "
        f"grasp_z_comp={target.planar_ik_grasp_z_comp_mm:.3f} mm "
        f"task_pose={format_pose(plan.steps[2].pose)} "
        f"place=({args.place_x_mm:.3f}, {args.place_y_mm:.3f}) mm",
        flush=True,
    )
    release_at_place = plan.task == "out" and not args.grasp_only
    pick_from_place = False
    return_ready = any(step.label in ("return_ready", "return_ready_above") for step in plan.steps)
    sequence_kind = {
        "out": "yolo_out_to_fixed_place" if not args.grasp_only else "yolo_grasp_only",
        "in": "yolo_in_to_ready",
    }[plan.task]
    response = make_ack_message(
        command.command_id,
        pose_to_dict(plan.steps[2].pose),
        executed=did_execute,
    )
    response["payload"].update(
        {
            "task": plan.task,
            "grasp_mode": command.grasp_mode,
            "vision_source": target.vision_source,
            "mapping_source": "hand_eye",
            "base_xyz_m": {
                "x_m": target.base_xyz_m[0],
                "y_m": target.base_xyz_m[1],
                "z_m": target.base_xyz_m[2],
            },
            "pick_x_mm": target.pick_x_mm,
            "pick_y_mm": target.pick_y_mm,
            "raw_pick_x_mm": target.raw_pick_x_mm,
            "raw_pick_y_mm": target.raw_pick_y_mm,
            "pick_x_source": target.pick_x_source,
            "pick_y_source": target.pick_y_source,
            "pick_x_perspective_reference_mm": (
                target.pick_x_perspective_reference_mm
            ),
            "pick_x_perspective_camera_z_mm": (
                target.pick_x_perspective_camera_z_mm
            ),
            "pick_x_perspective_source_z_mm": (
                target.pick_x_perspective_source_z_mm
            ),
            "pick_x_perspective_target_z_mm": (
                target.pick_x_perspective_target_z_mm
            ),
            "pick_x_perspective_scale": target.pick_x_perspective_scale,
            "pick_x_perspective_compensation_mm": (
                target.pick_x_perspective_compensation_mm
            ),
            "grasp_z_mm": target.grasp_z_mm,
            "raw_grasp_z_mm": target.raw_grasp_z_mm,
            "grasp_z_source": target.grasp_z_source,
            "pick_above_lift_z_mm": pick_above_lift_z_mm(args, target),
            "pick_above_lift_z_gradient": args.pick_above_lift_z_gradient,
            "multi_start_cartesian_ik_seed_count": (
                args.multi_start_cartesian_ik_seed_count
            ),
            "uncompensated_grasp_z_mm": target.uncompensated_grasp_z_mm,
            "planar_ik_grasp_z_comp_mm": target.planar_ik_grasp_z_comp_mm,
            "planar_ik_grasp_z_comp_pitch_delta_deg": (
                target.planar_ik_grasp_z_comp_pitch_delta_deg
            ),
            "planar_ik_grasp_z_comp_mm_per_deg": (
                args.planar_ik_grasp_z_comp_mm_per_deg
            ),
            "detected_z_mode": detected_z_mode(args),
            "fixed_grasp_z_floor_mm": (
                args.grasp_z_mm if detected_z_mode(args) != "off" else None
            ),
            "place_x_mm": args.place_x_mm,
            "place_y_mm": args.place_y_mm,
            "place_pick_pose": pose_to_dict(place_pick_pose(args)),
            "steps": [
                {
                    "label": step.label,
                    "mode": step.mode,
                    "pose": pose_to_dict(step.pose),
                    "gripper_after": step.gripper_after,
                    "delay_before_s": step.delay_before_s,
                    "joint_pitch_before_pose": step.joint_pitch_before_pose,
                    "joint_pitch_after_pose": step.joint_pitch_after_pose,
                    "pitch_source_x_mm": step.pitch_source_x_mm,
                    "planar_ik": step.planar_ik,
                    "planar_ik_include_j6_yaw": step.planar_ik_include_j6_yaw,
                    "require_reached": step.require_reached,
                }
                for step in plan.steps
            ],
            "send_counts": execution.send_counts,
            "fixed_orientation_deg": {
                "rx_deg": args.rx_deg,
                "ry_deg": args.ry_deg,
                "rz_deg": args.rz_deg,
            },
            "target_orientation_deg": {
                "rx_deg": plan.steps[2].pose.rx_deg,
                "ry_deg": plan.steps[2].pose.ry_deg,
                "rz_deg": plan.steps[2].pose.rz_deg,
            },
            "grasp_yaw_timing": grasp_yaw_timing_value(args),
            "dynamic_pitch": dynamic_pitch_summary(args, target),
            "joint_pitch_adjustments": [
                joint_pitch_adjustment_to_dict(result)
                for result in execution.joint_pitch_adjustments
            ],
            "grasp_yaw_adjustments": [
                grasp_yaw_adjustment_to_dict(result)
                for result in execution.grasp_yaw_adjustments
            ],
            "planar_ik_results": [
                planar_ik_result_to_dict(result)
                for result in execution.planar_ik_results
            ],
            "sequence": {
                "kind": sequence_kind,
                "task": plan.task,
                "return_ready": return_ready,
                "release_after_grasp": False,
                "release_at_place": release_at_place,
                "pick_back_from_place": pick_from_place,
            },
            "gripper_enabled": not args.no_gripper,
            "speed_percent": int(args.speed),
            "j6_speed_percent": j6_speed(args),
            "grasp_effort_nm": grasp_effort_nm(args),
            "safe_gripper": {
                "enabled": bool(args.safe_gripper_stop_on_effort),
                "max_effort_nm": args.max_gripper_effort_nm,
                "monitor_s": args.safe_gripper_monitor_s,
                "poll_s": args.safe_gripper_poll_s,
                "release_mm": args.safe_gripper_release_mm,
                "results": [
                    safe_gripper_result_to_dict(result)
                    for result in execution.safe_gripper_results
                ],
                "triggered": any(
                    result.triggered for result in execution.safe_gripper_results
                ),
            },
            "skipped": skipped,
        }
    )
    if timing is not None:
        response["payload"]["dynamic_timing"] = {
            "t_arrive_s": timing.t_arrive_s,
            "t_robot_s": timing.t_robot_s,
            "t_robot_source": timing.t_robot_source,
            "t_robot_source_x_mm": timing.t_robot_source_x_mm,
            "t_robot_reference_x_mm": timing.t_robot_reference_x_mm,
            "t_robot_segment": timing.t_robot_segment,
            "t_robot_slope_s_per_mm": timing.t_robot_slope_s_per_mm,
            "t_robot_base_s": timing.t_robot_base_s,
            "t_robot_unclamped_s": timing.t_robot_unclamped_s,
            "t_robot_margin_s": timing.t_robot_margin_s,
            "t_robot_min_s": timing.t_robot_min_s,
            "t_robot_max_s": timing.t_robot_max_s,
            "action_time_s": timing.action_time_s,
            "t_wait_s": timing.t_wait_s,
            "dynamic_preposition": timing.dynamic_preposition,
            "final_action_time_s": timing.final_action_time_s,
            "wait_placement": timing.wait_placement,
            "too_close_skip_enabled": timing.too_close_skip_enabled,
            "skipped": timing.skipped,
            "skip_reason": timing.skip_reason,
            "conveyor_speed_mm_s": command.conveyor_speed_mm_s,
            "pick_x_mm": command.pick_x_mm,
            "pick_y_mm": command.pick_y_mm,
            "detected_x_mm": command.detected_x_mm,
            "detected_y_mm": command.detected_y_mm,
            "detected_z_mm": command.detected_z_mm,
            "detected_top_x_mm": command.detected_top_x_mm,
            "detected_top_y_mm": command.detected_top_y_mm,
            "detected_top_z_mm": command.detected_top_z_mm,
        }
        dynamic_execution: dict[str, Any] = {
            "preposition_enabled": bool(args.dynamic_preposition),
            "wait_placement": args.dynamic_wait_placement,
            "preposition_z_mm": execution.preposition_z_mm,
            "preposition_source": execution.preposition_source,
            "preposition_clearance_mm": args.dynamic_preposition_clearance_mm,
            "preposition_elapsed_s": execution.preposition_elapsed_s,
            "final_action_due_s": execution.final_action_due_s,
            "final_wait_s": execution.final_wait_s,
            "final_action_late_s": execution.final_action_late_s,
        }
        response["payload"]["dynamic_execution"] = dynamic_execution
    if metadata:
        response["payload"]["yolo"] = metadata
    return response


def serve(args: argparse.Namespace) -> None:
    validate_args(args)
    hand_eye_document = load_hand_eye_result_document(args.hand_eye_result)
    camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
    print(f"Using hand-eye camera->base transform: {args.hand_eye_result}", flush=True)
    quality = hand_eye_quality_summary(hand_eye_document)
    if quality is not None:
        print(f"Hand-eye quality: {quality}", flush=True)

    config = cartesian_only_robot_config(args.can)
    validator = PiperController(config=config)
    controller: PiperController | None = None
    if args.execute:
        controller = PiperController(config=config)
        print("Connecting and enabling Piper arm...", flush=True)
        controller.connect(enable=True)
        print("Piper arm enabled", flush=True)
        if args.collision_protection_level is not None:
            levels = [args.collision_protection_level] * 6
            controller.set_collision_protection(levels)
            print(
                "Piper collision protection configured: "
                f"levels={levels}",
                flush=True,
            )
        maybe_prepare_gripper(controller, args)
        if not args.no_start_ready:
            move_to_start_ready(controller, args)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((args.host, args.port))
            server.listen(1)
            print(
                f"Listening on {args.host}:{args.port}, execute={args.execute}, "
                f"speed={args.speed}, j6_speed={j6_speed(args)}, "
                f"startup_speed={args.startup_speed}, "
                f"fixed_orientation="
                f"({args.rx_deg:.3f}, {args.ry_deg:.3f}, {args.rz_deg:.3f})",
                flush=True,
            )
            print(f"Catch server 1 args: {args}", flush=True)
            while True:
                conn, addr = server.accept()
                with conn:
                    print(f"Client connected: {addr}", flush=True)
                    with conn.makefile("rb") as sock_file:
                        while True:
                            message = recv_json_line(sock_file)
                            if message is None:
                                break
                            command_id = str(message.get("command_id"))
                            try:
                                response = handle_message(
                                    message,
                                    args=args,
                                    camera_to_base=camera_to_base,
                                    validator=validator,
                                    controller=controller,
                                )
                            except Exception as exc:  # noqa: BLE001 - returned to client
                                response = make_error_message(command_id, str(exc))
                                print(f"Error: {exc}", flush=True)
                            send_json_line(conn, response)
    finally:
        if controller is not None:
            controller.disconnect()


def main() -> None:
    try:
        serve(build_parser().parse_args())
    except (SafetyError, ValueError, RuntimeError, TimeoutError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

"""YOLO-driven grasp server using the fixed-pose continuous sequence style.

This server receives ``CameraPointCommand`` messages from the Windows YOLO
client, maps the camera XYZ through the solved eye-to-hand transform, and runs a
direct J6 pose sequence modeled after ``run_pose_continuous_catch_sequence``:

``out`` task:
ready/open -> target above -> target grasp/close -> lift -> fixed place/open
-> return ready.

``in`` task:
ready/open -> selected target above -> selected target grasp/close -> lift
-> return ready/open.

Unlike ``catch_server.py``, this script intentionally commands fixed J6 poses
directly and keeps one fixed wrist orientation for every waypoint. A YOLO bbox
by itself is not enough for 3D grasping; the Windows side should send the
bbox-selected point after RealSense depth deprojection as ``x_m/y_m/z_m``.
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

from robot_control.linux.arm.kinematics import (
    forward_kinematics_pose,
    normalize_degrees,
)
from robot_control.linux.arm.piper_arm import RAD_TO_MILLI_DEG, PiperController
from robot_control.linux.arm.safety import SafetyError
from robot_control.linux.catch.catch_server import (
    execute_pose_for_duration,
    feedback_to_pose,
    format_pose,
    format_status,
    pose_delta,
    wait_until_pose_reached,
)
from robot_control.linux.catch.tcp_offset import rotation_matrix_xyz_deg
from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
    map_camera_point_to_base_xyz_m,
)
from robot_control.linux.config.robot_config import ROBOT_CONFIG
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
DEFAULT_KEEP_PITCH_MIN_X_MM = 300.0
DEFAULT_KEEP_PITCH_MAX_X_MM = 380.0
DEFAULT_PITCH_DEG_PER_MM_FAR = 0.04
DEFAULT_PITCH_DEG_PER_MM_NEAR = 0.03
PLANAR_IK_JOINT_INDICES = (1, 2, 4)
PLANAR_IK_YAW_JOINT_INDICES = (1, 2, 3, 4, 5)
PLANAR_IK_YAW_DELTA_LIMIT_JOINT_INDICES = (3, 5)
PLANAR_IK_YAW_DELTA_LIMIT_DEG = 90.0


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
class GuardedDescentResult:
    send_count: int
    start_z_mm: float
    target_z_mm: float
    aborted: bool = False
    reason: str | None = None
    last_feedback_z_mm: float | None = None
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None


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
    protected_abort: bool = False
    protected_abort_reason: str | None = None
    protected_abort_return: str | None = None
    protected_abort_return_label: str | None = None
    preposition_z_mm: float | None = None
    preposition_source: str | None = None
    preposition_elapsed_s: float | None = None
    final_action_due_s: float | None = None
    final_wait_s: float | None = None
    final_action_late_s: float | None = None
    guarded_descent: GuardedDescentResult | None = None
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
            "estimate based on raw pick X"
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
        ),
    )
    parser.add_argument(
        "--dynamic-t-robot-planar-near-s-per-mm",
        type=float,
        default=0.0,
        help=(
            "X-based timing slope for raw X smaller than "
            "--dynamic-t-robot-reference-x-mm; 0 uses --dynamic-t-robot-s-per-mm"
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
            "is unavailable, falls back to grasp_z plus the guarded descent zone"
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
        "--guarded-descent",
        "--descent-protection",
        action=argparse.BooleanOptionalAction,
        dest="guarded_descent",
        default=True,
        help=(
            "check J6 Z feedback during the final dynamic descent and abort if "
            "the arm stops following the commanded Z; enabled by default, use "
            "--no-guarded-descent or --no-descent-protection to disable"
        ),
    )
    parser.add_argument("--guarded-descent-zone-mm", type=float, default=30.0)
    parser.add_argument("--guarded-descent-step-mm", type=float, default=3.0)
    parser.add_argument("--guarded-descent-speed", type=int, default=20)
    parser.add_argument(
        "--guarded-follow-error-mm",
        type=float,
        default=4.0,
        help=(
            "maximum allowed Z feedback lag from each guarded descent command "
            "after the step wait; gripper close effort is not used for descent "
            "collision protection"
        ),
    )
    parser.add_argument("--guarded-stall-s", type=float, default=0.2)
    parser.add_argument(
        "--guarded-start-tolerance-mm",
        type=float,
        default=2.0,
        help=(
            "required Z feedback tolerance at the guarded descent start pose; "
            "prevents starting descent while the arm is still above guard start"
        ),
    )
    parser.add_argument(
        "--guarded-start-timeout-s",
        type=float,
        default=1.0,
        help="maximum time to wait for feedback to reach guarded descent start Z",
    )
    parser.add_argument(
        "--guarded-progress-min-mm",
        type=float,
        default=1.0,
        help=(
            "minimum observed downward J6 feedback movement required after each "
            "guarded descent command; 0 requires any measurable downward movement"
        ),
    )
    parser.add_argument(
        "--guarded-step-timeout-s",
        type=float,
        default=0.25,
        help=(
            "maximum time to wait for guarded descent feedback progress after "
            "each step; defaults to --guarded-stall-s when 0"
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
        "--guarded-abort-return",
        choices=("pick_above", "preposition", "none"),
        default="pick_above",
        help=(
            "legacy abort-return selection. Guarded descent aborts now stop "
            "the sequence without issuing an automatic return pose."
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
        action="store_true",
        help=(
            "align grasp yaw to the object short-edge yaw reported by the vision "
            "payload before final grasp descent; disabled by default"
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
        "--grasp-yaw-speed",
        type=int,
        default=20,
        help="speed percent for object-short-edge J6 yaw alignment",
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
        "--pick-x-offset-mm",
        type=float,
        default=0.0,
        help=(
            "add this X offset to the final pick coordinate only when the target "
            "uses planar-IK dynamic pitch; raw X is still used for pitch selection"
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
    parser.add_argument("--gripper-settle-s", type=float, default=0.6)
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
    if not isinstance(args.speed, int) or not 0 <= args.speed <= 100:
        raise ValueError("--speed must be an integer in [0, 100]")
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
    if not math.isfinite(args.command_period_s) or args.command_period_s <= 0:
        raise ValueError("--command-period-s must be finite and > 0")
    if not math.isfinite(args.safe_gripper_poll_s) or args.safe_gripper_poll_s <= 0:
        raise ValueError("--safe-gripper-poll-s must be finite and > 0")
    if (
        not isinstance(args.guarded_descent_speed, int)
        or not 0 <= args.guarded_descent_speed <= 100
    ):
        raise ValueError("--guarded-descent-speed must be an integer in [0, 100]")
    if (
        not isinstance(args.dynamic_pitch_speed, int)
        or not 0 <= args.dynamic_pitch_speed <= 100
    ):
        raise ValueError("--dynamic-pitch-speed must be an integer in [0, 100]")
    if (
        not isinstance(args.grasp_yaw_speed, int)
        or not 0 <= args.grasp_yaw_speed <= 100
    ):
        raise ValueError("--grasp-yaw-speed must be an integer in [0, 100]")
    for name in (
        "dynamic_preposition_clearance_mm",
        "guarded_descent_zone_mm",
        "guarded_descent_step_mm",
        "guarded_follow_error_mm",
        "guarded_stall_s",
        "guarded_start_tolerance_mm",
        "guarded_start_timeout_s",
        "guarded_progress_min_mm",
        "guarded_step_timeout_s",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if args.dynamic_preposition_clearance_mm <= 0.0:
        raise ValueError("--dynamic-preposition-clearance-mm must be > 0")
    if args.guarded_descent_zone_mm <= 0.0:
        raise ValueError("--guarded-descent-zone-mm must be > 0")
    if args.guarded_descent_step_mm <= 0.0:
        raise ValueError("--guarded-descent-step-mm must be > 0")
    if args.guarded_follow_error_mm <= 0.0:
        raise ValueError("--guarded-follow-error-mm must be > 0")
    if args.guarded_start_tolerance_mm <= 0.0:
        raise ValueError("--guarded-start-tolerance-mm must be > 0")
    if args.collision_protection_level is not None and (
        not isinstance(args.collision_protection_level, int)
        or not 0 <= args.collision_protection_level <= 8
    ):
        raise ValueError("--collision-protection-level must be an integer in [0, 8]")
    if getattr(args, "dynamic_pitch_mode", "joint") == "planar-ik" and args.guarded_descent:
        raise ValueError("--dynamic-pitch-mode planar-ik requires --no-guarded-descent")
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
        "pick_x_offset_mm",
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
    return (
        dynamic_pitch_control_mode(args) == "joint"
        and joint_pitch_decision(args, pitch_source_x_mm).mode == "adjust"
    )


def use_planar_ik(args: argparse.Namespace, pitch_source_x_mm: float) -> bool:
    return (
        dynamic_pitch_control_mode(args) == "planar_ik"
        and joint_pitch_decision(args, pitch_source_x_mm).mode == "adjust"
    )


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
    pick_x_offset_mm = float(getattr(args, "pick_x_offset_mm", 0.0))
    if (
        not math.isclose(pick_x_offset_mm, 0.0, rel_tol=0.0, abs_tol=1e-9)
        and use_planar_ik(args, float(raw_pick_x_mm))
    ):
        pick_x_mm = float(pick_x_mm) + pick_x_offset_mm
        pick_x_source = f"{pick_x_source}_planar_ik_offset"
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
    if target.grasp_z_mm > args.transfer_z_mm:
        raise SafetyError(
            f"grasp-z {target.grasp_z_mm:.3f} mm is above transfer plane "
            f"{args.transfer_z_mm:.3f} mm"
        )


def build_out_steps(
    args: argparse.Namespace,
    target: YoloCatchTarget,
) -> tuple[SequenceStep, ...]:
    ready = ready_pose(args)
    pick_above = target_pose(args, target, args.transfer_z_mm)
    pick_grasp = target_pose(args, target, target.grasp_z_mm)
    place_above = fixed_pose(args, args.place_x_mm, args.place_y_mm, args.transfer_z_mm)
    pitch_x_mm = pitch_source_x_mm(target)
    use_joint_pitch = use_joint_pitch_adjustment(args, pitch_x_mm)
    use_ik = use_planar_ik(args, pitch_x_mm)
    use_ready_planar_return = far_planar_transfer_via_ready(args, pitch_x_mm)

    steps: list[SequenceStep] = [
        SequenceStep("ready_open", ready, args.transfer_mode, "open"),
        SequenceStep(
            "pick_above",
            pick_above,
            "L",
            joint_pitch_before_pose=use_joint_pitch,
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
            require_reached=True,
        ),
        SequenceStep(
            label="pick_grasp",
            pose=pick_grasp,
            mode=args.vertical_mode,
            gripper_after="close",
            delay_before_s=args.pick_delay_s,
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
        ),
        SequenceStep(
            "pick_lift",
            pick_above,
            args.lift_mode,
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
        ),
    ]
    if args.grasp_only:
        if not args.no_return_ready:
            steps.append(
                SequenceStep(
                    "return_ready",
                    ready,
                    args.transfer_mode,
                    pitch_source_x_mm=fixed_pitch_source_x_mm(args),
                    planar_ik=use_ready_planar_return,
                )
            )
        return tuple(steps)

    if use_ready_planar_return:
        steps.append(
            SequenceStep(
                "return_ready_above",
                ready,
                args.transfer_mode,
                pitch_source_x_mm=fixed_pitch_source_x_mm(args),
                planar_ik=True,
            )
        )
    steps.extend(
        [
            SequenceStep("place_above", place_above, args.transfer_mode),
            SequenceStep("place_release", place_above, args.transfer_mode, "open"),
        ]
    )
    if not args.no_return_ready:
        steps.append(SequenceStep("return_ready", ready, args.transfer_mode))
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
    pick_above = target_pose(args, target, args.transfer_z_mm)
    pick_grasp = target_pose(args, target, target.grasp_z_mm)
    pitch_x_mm = pitch_source_x_mm(target)
    use_joint_pitch = use_joint_pitch_adjustment(args, pitch_x_mm)
    use_ik = use_planar_ik(args, pitch_x_mm)
    return (
        SequenceStep("ready_open", ready, args.transfer_mode, "open"),
        SequenceStep(
            label="return_pick_above",
            pose=pick_above,
            mode="L",
            delay_before_s=args.place_pick_delay_s,
            joint_pitch_before_pose=use_joint_pitch,
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
            require_reached=True,
        ),
        SequenceStep(
            label="return_pick_grasp",
            pose=pick_grasp,
            mode=args.vertical_mode,
            gripper_after="close",
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
        ),
        SequenceStep(
            "return_pick_lift",
            pick_above,
            args.lift_mode,
            joint_pitch_after_pose=use_joint_pitch,
            pitch_source_x_mm=pitch_x_mm,
            planar_ik=use_ik,
        ),
        SequenceStep("return_ready_above", ready, args.transfer_mode),
        SequenceStep("final_release", ready, args.transfer_mode, "open"),
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
    t_robot_s, t_robot_meta = dynamic_t_robot_s_for_x(args, source_x_mm)
    if not math.isfinite(t_arrive_s) or t_arrive_s < 0.0:
        raise ValueError("dynamic t_arrive_s must be finite and >= 0")
    if not math.isfinite(t_robot_s) or t_robot_s < 0.0:
        raise ValueError("dynamic t_robot_s must be finite and >= 0")
    wait_placement = str(getattr(args, "dynamic_wait_placement", "wait"))
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
        requested_z_mm = target.grasp_z_mm + float(args.guarded_descent_zone_mm)
        source = "grasp_z_plus_guarded_zone"

    min_descent_z_mm = min(
        float(args.transfer_z_mm),
        target.grasp_z_mm + float(args.guarded_descent_step_mm),
    )
    z_mm = min(float(args.transfer_z_mm), max(requested_z_mm, min_descent_z_mm))
    if z_mm != requested_z_mm:
        source = f"{source}_clamped"
    return target_pose(args, target, z_mm), source


def guarded_start_pose_for_grasp(
    args: argparse.Namespace,
    grasp_pose: EndPose,
) -> EndPose:
    z_mm = min(
        float(args.transfer_z_mm),
        grasp_pose.z_mm + float(args.guarded_descent_zone_mm),
    )
    return replace(grasp_pose, z_mm=z_mm)


def should_guard_step(step: SequenceStep, args: argparse.Namespace) -> bool:
    return bool(
        args.guarded_descent
        and step.gripper_after == "close"
        and step.pose.z_mm < args.transfer_z_mm
    )


def guarded_abort_return_step(
    args: argparse.Namespace,
    steps: tuple[SequenceStep, ...],
    step_index: int,
    fallback_pose: EndPose,
) -> SequenceStep | None:
    if args.guarded_abort_return == "none":
        return None
    if args.guarded_abort_return == "preposition":
        return SequenceStep(
            "protected_abort_return_preposition",
            fallback_pose,
            args.vertical_mode,
        )
    for prior_step in reversed(steps[:step_index]):
        same_xy = (
            abs(prior_step.pose.x_mm - steps[step_index].pose.x_mm) <= 1e-6
            and abs(prior_step.pose.y_mm - steps[step_index].pose.y_mm) <= 1e-6
        )
        above_target = prior_step.pose.z_mm > steps[step_index].pose.z_mm
        if same_xy and above_target:
            return SequenceStep(
                f"protected_abort_return_{prior_step.label}",
                prior_step.pose,
                args.vertical_mode,
            )
    return SequenceStep(
        "protected_abort_return_pick_above",
        replace(steps[step_index].pose, z_mm=args.transfer_z_mm),
        args.vertical_mode,
    )


def dynamic_protected_abort_return_step(
    args: argparse.Namespace,
    *,
    pick_above_pose: EndPose,
    preposition_pose: EndPose,
) -> tuple[SequenceStep | None, str | None]:
    if args.guarded_abort_return == "none":
        return None, None
    if args.guarded_abort_return == "preposition":
        return (
            SequenceStep(
                "protected_abort_return_preposition",
                preposition_pose,
                args.vertical_mode,
            ),
            "preposition",
        )
    return (
        SequenceStep(
            "protected_abort_return_pick_above",
            pick_above_pose,
            args.vertical_mode,
        ),
        "pick_above",
    )


def execute_protected_abort_return(
    controller: PiperController,
    args: argparse.Namespace,
    counts: dict[str, int],
    gripper_results: list[SafeGripperResult],
    joint_pitch_adjustments: list[JointPitchAdjustmentResult],
    planar_ik_results: list[PlanarIkExecutionResult],
    abort_step: SequenceStep | None,
    planar_ik_reuse_state: PlanarIkReuseState | None = None,
) -> tuple[str | None, str | None]:
    if abort_step is None:
        return None, None
    counts[abort_step.label] = execute_and_record_step(
        controller,
        args,
        abort_step,
        gripper_results,
        joint_pitch_adjustments,
        planar_ik_results,
        planar_ik_reuse_state,
    )
    abort_return = None
    if abort_step.label.startswith("protected_abort_return_"):
        abort_return = abort_step.label.removeprefix("protected_abort_return_")
    return abort_return, abort_step.label


def safe_gripper_abort_reason(result: SafeGripperResult) -> str:
    effort = (
        "unknown" if result.last_effort_nm is None else f"{result.last_effort_nm:.3f}Nm"
    )
    threshold = (
        "unknown" if result.max_effort_nm is None else f"{result.max_effort_nm:.3f}Nm"
    )
    return (
        "safe gripper effort threshold exceeded: "
        f"effort={effort} threshold={threshold}"
    )


def wait_until_guarded_start_reached(
    controller: PiperController,
    args: argparse.Namespace,
    start_pose: EndPose,
) -> tuple[bool, str | None, float | None]:
    timeout_s = float(args.guarded_start_timeout_s)
    deadline = time.monotonic() + timeout_s
    last_z_mm: float | None = None
    tolerance_mm = float(args.guarded_start_tolerance_mm)
    while timeout_s == 0.0 or time.monotonic() < deadline:
        feedback_pose = feedback_to_pose(controller.get_end_pose())
        if feedback_pose is not None:
            last_z_mm = float(feedback_pose.z_mm)
            if abs(last_z_mm - float(start_pose.z_mm)) <= tolerance_mm:
                return True, None, last_z_mm
        if timeout_s == 0.0:
            break
        time.sleep(min(max(float(args.command_period_s), 0.005), 0.05))
    return (
        False,
        "guarded descent start not reached: "
        f"target_z={start_pose.z_mm:.3f}mm "
        f"feedback_z={last_z_mm}mm "
        f"tolerance={tolerance_mm:.3f}mm "
        f"timeout={timeout_s:.3f}s",
        last_z_mm,
    )


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


def execute_guarded_descent(
    controller: PiperController,
    args: argparse.Namespace,
    *,
    start_pose: EndPose,
    target_pose: EndPose,
) -> GuardedDescentResult:
    start_z_mm = float(start_pose.z_mm)
    target_z_mm = float(target_pose.z_mm)
    if start_z_mm <= target_z_mm:
        return GuardedDescentResult(
            send_count=0,
            start_z_mm=start_z_mm,
            target_z_mm=target_z_mm,
            aborted=True,
            reason=(
                "guarded descent start is not above target: "
                f"start_z={start_z_mm:.3f} target_z={target_z_mm:.3f}"
            ),
        )
    start_reached, start_reason, start_feedback_z_mm = wait_until_guarded_start_reached(
        controller,
        args,
        start_pose,
    )
    if not start_reached:
        return GuardedDescentResult(
            send_count=0,
            start_z_mm=start_z_mm,
            target_z_mm=target_z_mm,
            aborted=True,
            reason=start_reason,
            last_feedback_z_mm=start_feedback_z_mm,
        )

    print(
        "Running guarded_descent: "
        f"z {start_z_mm:.3f}->{target_z_mm:.3f} mm "
        f"step={args.guarded_descent_step_mm:.3f}mm "
        f"speed={args.guarded_descent_speed}",
        flush=True,
    )
    send_count = 0
    commanded_z_mm = start_z_mm
    last_feedback_z_mm: float | None = start_feedback_z_mm
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None
    while commanded_z_mm > target_z_mm:
        previous_feedback_z_mm = last_feedback_z_mm
        if previous_feedback_z_mm is None:
            previous_pose = feedback_to_pose(controller.get_end_pose())
            if previous_pose is not None:
                previous_feedback_z_mm = float(previous_pose.z_mm)
        next_commanded_z_mm = max(
            target_z_mm,
            commanded_z_mm - float(args.guarded_descent_step_mm),
        )
        commanded_drop_mm = commanded_z_mm - next_commanded_z_mm
        commanded_z_mm = next_commanded_z_mm
        command_pose = replace(target_pose, z_mm=commanded_z_mm)
        controller.move_to_pose(
            command_pose,
            speed_percent=args.guarded_descent_speed,
            move_mode=args.vertical_mode,
        )
        send_count += 1
        (
            aborted,
            reason,
            last_feedback_z_mm,
            last_follow_error_mm,
            last_progress_mm,
        ) = (
            guarded_step_abort_status(
                controller,
                args,
                command_pose,
                previous_feedback_z_mm=previous_feedback_z_mm,
                commanded_drop_mm=commanded_drop_mm,
            )
        )
        if args.print_feedback:
            print(
                "Guarded step feedback: "
                f"command_z={command_pose.z_mm:.3f}mm "
                f"feedback_z={last_feedback_z_mm}mm "
                f"progress={last_progress_mm}mm "
                f"follow_error={last_follow_error_mm}mm",
                flush=True,
            )
        if aborted:
            controller.emergency_stop()
            return GuardedDescentResult(
                send_count=send_count,
                start_z_mm=start_z_mm,
                target_z_mm=target_z_mm,
                aborted=True,
                reason=reason,
                last_feedback_z_mm=last_feedback_z_mm,
                last_follow_error_mm=last_follow_error_mm,
                last_progress_mm=last_progress_mm,
            )
    return GuardedDescentResult(
        send_count=send_count,
        start_z_mm=start_z_mm,
        target_z_mm=target_z_mm,
        last_feedback_z_mm=last_feedback_z_mm,
        last_follow_error_mm=last_follow_error_mm,
        last_progress_mm=last_progress_mm,
    )


def guarded_step_abort_status(
    controller: PiperController,
    args: argparse.Namespace,
    command_pose: EndPose,
    *,
    previous_feedback_z_mm: float | None = None,
    commanded_drop_mm: float | None = None,
) -> tuple[bool, str | None, float | None, float | None, float | None]:
    timeout_s = (
        float(args.guarded_step_timeout_s)
        if float(args.guarded_step_timeout_s) > 0.0
        else float(args.guarded_stall_s)
    )
    deadline = time.monotonic() + timeout_s
    last_feedback_z_mm: float | None = None
    last_follow_error_mm: float | None = None
    last_progress_mm: float | None = None
    progress_min_mm = float(args.guarded_progress_min_mm)
    required_progress_mm = progress_min_mm
    if commanded_drop_mm is not None and commanded_drop_mm > 0.0 and progress_min_mm > 0.0:
        required_progress_mm = min(progress_min_mm, commanded_drop_mm)
    while True:
        feedback_pose = feedback_to_pose(controller.get_end_pose())
        if feedback_pose is not None:
            last_feedback_z_mm = float(feedback_pose.z_mm)
            last_follow_error_mm = last_feedback_z_mm - float(command_pose.z_mm)
            if previous_feedback_z_mm is not None:
                last_progress_mm = previous_feedback_z_mm - last_feedback_z_mm
            progress_ok = (
                last_progress_mm is not None
                and (
                    last_progress_mm > 0.0
                    if required_progress_mm <= 0.0
                    else last_progress_mm >= required_progress_mm
                )
            )
            follow_error_ok = last_follow_error_mm <= float(args.guarded_follow_error_mm)
            if progress_ok and follow_error_ok:
                return (
                    False,
                    None,
                    last_feedback_z_mm,
                    last_follow_error_mm,
                    last_progress_mm,
                )
        if time.monotonic() >= deadline:
            if last_feedback_z_mm is None:
                return (
                    True,
                    "no valid J6 feedback during guarded descent",
                    None,
                    None,
                    None,
                )
            progress_ok = (
                last_progress_mm is not None
                and (
                    last_progress_mm > 0.0
                    if required_progress_mm <= 0.0
                    else last_progress_mm >= required_progress_mm
                )
            )
            if progress_ok:
                return (
                    True,
                    "guarded descent feedback did not reach commanded height "
                    f"within {timeout_s:.3f}s: "
                    f"command_z={command_pose.z_mm:.3f}mm "
                    f"previous_feedback_z={previous_feedback_z_mm}mm "
                    f"feedback_z={last_feedback_z_mm:.3f}mm "
                    f"progress={last_progress_mm}mm "
                    f"follow_error={last_follow_error_mm}mm "
                    f"limit={args.guarded_follow_error_mm:.3f}mm",
                    last_feedback_z_mm,
                    last_follow_error_mm,
                    last_progress_mm,
                )
            return (
                True,
                "guarded descent height did not decrease "
                f"{required_progress_mm:.3f} mm within {timeout_s:.3f}s: "
                f"command_z={command_pose.z_mm:.3f}mm "
                f"previous_feedback_z={previous_feedback_z_mm}mm "
                f"feedback_z={last_feedback_z_mm:.3f}mm "
                f"progress={last_progress_mm}mm "
                f"follow_error={last_follow_error_mm}mm",
                last_feedback_z_mm,
                last_follow_error_mm,
                last_progress_mm,
            )
        time.sleep(min(max(float(args.command_period_s), 0.005), 0.05))


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
    if args.gripper_settle_s > 0:
        time.sleep(args.gripper_settle_s)
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
    if step.delay_before_s > 0:
        print(f"Waiting {step.delay_before_s:g}s before {step.label}", flush=True)
        time.sleep(step.delay_before_s)
    if step.planar_ik:
        pitch_source_x = (
            float(step.pitch_source_x_mm)
            if step.pitch_source_x_mm is not None
            else float(step.pose.x_mm)
        )
        cache_key = planar_ik_cache_key(args, step, pitch_source_x)
        cached = (
            None
            if planar_ik_reuse_state is None
            else planar_ik_reuse_state.targets.get(cache_key)
        )
        if cached is not None and can_reuse_planar_ik_target(
            args,
            step,
            planar_ik_reuse_state,
        ):
            send_count, planar_result = execute_reused_planar_ik_step(
                controller,
                args,
                step,
                cached,
            )
            if planar_ik_results is not None:
                planar_ik_results.append(planar_result)
            gripper_result = send_gripper(controller, args, step.gripper_after)
            return send_count, gripper_result
        send_count, planar_result = execute_planar_ik_step(controller, args, step)
        if planar_ik_reuse_state is not None:
            planar_ik_reuse_state.targets[cache_key] = PlanarIkCachedTarget(
                step.label,
                tuple(math.radians(value) for value in planar_result.target_joints_deg),
                planar_result,
            )
        if planar_ik_results is not None:
            planar_ik_results.append(planar_result)
        gripper_result = send_gripper(controller, args, step.gripper_after)
        return send_count, gripper_result
    if step.joint_pitch_before_pose:
        pitch_adjustment = apply_joint_pitch_adjustment(
            controller,
            args,
            step,
            phase="before_pose",
        )
        if pitch_adjustment is not None and joint_pitch_adjustments is not None:
            joint_pitch_adjustments.append(pitch_adjustment)
    print(f"Running {step.label}: {format_pose(step.pose)}", flush=True)
    send_count = execute_pose_for_duration(
        controller,
        step.pose,
        args.speed if speed is None else speed,
        step.mode,
        args.command_duration_s,
        args.command_period_s,
    )
    should_wait = step.require_reached or not args.no_wait_waypoints
    if should_wait:
        wait_until_pose_reached(
            controller,
            step.label,
            step.pose,
            timeout_s=args.motion_timeout_s,
            position_tolerance_mm=args.position_tolerance_mm,
            orientation_tolerance_deg=args.orientation_tolerance_deg,
        )
    if step.joint_pitch_after_pose:
        pitch_adjustment = apply_joint_pitch_adjustment(
            controller,
            args,
            step,
            phase="after_pose",
        )
        if pitch_adjustment is not None and joint_pitch_adjustments is not None:
            joint_pitch_adjustments.append(pitch_adjustment)
    gripper_result = send_gripper(controller, args, step.gripper_after)
    return send_count, gripper_result


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
            int(args.grasp_yaw_speed),
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
    guarded_result: GuardedDescentResult | None = None
    gripper_results: list[SafeGripperResult] = []
    joint_pitch_adjustments: list[JointPitchAdjustmentResult] = []
    grasp_yaw_adjustments: list[GraspYawAdjustmentResult] = []
    planar_ik_results: list[PlanarIkExecutionResult] = []
    planar_ik_reuse_state = PlanarIkReuseState()
    post_yaw_orientation: EndPose | None = None
    for index, step in enumerate(plan.steps):
        active_step = step
        if active_step.label in ("pick_grasp", "return_pick_grasp"):
            integrated_planar_ik_yaw = bool(
                use_planar_ik_yaw(args) and active_step.planar_ik
            )
            if integrated_planar_ik_yaw:
                planned_orientation = plan_grasp_yaw_orientation_for_planar_ik(
                    controller,
                    args,
                    plan.target,
                    grasp_yaw_adjustments,
                    phase="before descent",
                )
                if planned_orientation is not None:
                    active_step = replace(
                        active_step,
                        pose=with_current_orientation(active_step.pose, planned_orientation),
                        planar_ik_include_j6_yaw=True,
                    )
            else:
                active_step = apply_grasp_yaw_before_descent(
                    controller,
                    args,
                    plan.target,
                    active_step,
                    grasp_yaw_adjustments,
                )
            if (
                grasp_yaw_adjustments
                and should_inherit_post_yaw_orientation(grasp_yaw_adjustments[-1])
            ):
                post_yaw_orientation = active_step.pose
        elif (
            post_yaw_orientation is not None
            and active_step.label in ("pick_lift", "return_pick_lift")
        ):
            active_step = replace(
                active_step,
                pose=with_current_orientation(active_step.pose, post_yaw_orientation),
                planar_ik_include_j6_yaw=bool(
                    use_planar_ik_yaw(args) and active_step.planar_ik
                ),
            )
        if should_guard_step(active_step, args):
            guarded_start_pose = guarded_start_pose_for_grasp(args, active_step.pose)
            guarded_start_step = SequenceStep(
                f"{active_step.label}_guard_start",
                guarded_start_pose,
                args.vertical_mode,
                None,
                active_step.delay_before_s,
            )
            counts[guarded_start_step.label] = execute_and_record_step(
                controller,
                args,
                guarded_start_step,
                gripper_results,
                joint_pitch_adjustments,
                planar_ik_results,
                planar_ik_reuse_state,
            )
            guarded_result = execute_guarded_descent(
                controller,
                args,
                start_pose=guarded_start_pose,
                target_pose=active_step.pose,
            )
            counts[active_step.label] = guarded_result.send_count
            if guarded_result.aborted:
                reason = guarded_result.reason or "guarded descent aborted"
                print(f"Protected abort: {reason}", flush=True)
                return DynamicExecutionResult(
                    send_counts=counts,
                    protected_abort=True,
                    protected_abort_reason=reason,
                    guarded_descent=guarded_result,
                    safe_gripper_results=tuple(gripper_results),
                    joint_pitch_adjustments=tuple(joint_pitch_adjustments),
                    grasp_yaw_adjustments=tuple(grasp_yaw_adjustments),
                    planar_ik_results=tuple(planar_ik_results),
                )
            pitch_adjustment = apply_joint_pitch_adjustment(
                controller,
                args,
                active_step,
                phase="after_guarded_descent",
            )
            if pitch_adjustment is not None:
                joint_pitch_adjustments.append(pitch_adjustment)
            gripper_result = send_gripper(controller, args, active_step.gripper_after)
            record_gripper_result(gripper_results, gripper_result)
            continue
        send_count, gripper_result = execute_step(
            controller,
            args,
            active_step,
            joint_pitch_adjustments=joint_pitch_adjustments,
            grasp_yaw_adjustments=grasp_yaw_adjustments,
            planar_ik_results=planar_ik_results,
            planar_ik_reuse_state=planar_ik_reuse_state,
        )
        counts[active_step.label] = send_count
        record_gripper_result(gripper_results, gripper_result)

    if args.print_feedback:
        time.sleep(0.1)
        after_pose = feedback_to_pose(controller.get_end_pose())
        print(f"After catch J6: {format_pose(after_pose)}", flush=True)
        print(f"Feedback delta: {pose_delta(before_pose, after_pose)}", flush=True)
        print(f"Catch send counts: {counts}", flush=True)
    return DynamicExecutionResult(
        send_counts=counts,
        guarded_descent=guarded_result,
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
    ready_step = plan.steps[0]
    pick_above_step = plan.steps[1]
    grasp_step = replace(plan.steps[2], delay_before_s=0.0)
    lift_step = plan.steps[3]
    preposition_pitch_x_mm = pitch_source_x_mm(plan.target)
    preposition_pose, preposition_source = dynamic_preposition_pose(args, plan.target)
    use_preposition_pitch = use_joint_pitch_adjustment(args, preposition_pitch_x_mm)
    use_preposition_ik = use_planar_ik(args, preposition_pitch_x_mm)
    preposition_step = SequenceStep(
        "dynamic_preposition",
        preposition_pose,
        args.vertical_mode,
        joint_pitch_before_pose=use_preposition_pitch,
        joint_pitch_after_pose=use_preposition_pitch,
        pitch_source_x_mm=preposition_pitch_x_mm,
        planar_ik=use_preposition_ik,
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
    integrated_planar_ik_yaw = bool(use_planar_ik_yaw(args) and use_preposition_ik)
    if integrated_planar_ik_yaw:
        post_yaw_orientation = plan_grasp_yaw_orientation_for_planar_ik(
            controller,
            args,
            plan.target,
            grasp_yaw_adjustments,
            phase="after pick_above",
        )
    else:
        post_yaw_orientation = apply_grasp_yaw_and_get_orientation(
            controller,
            args,
            plan.target,
            grasp_yaw_adjustments,
            phase="after pick_above",
        )
    if post_yaw_orientation is not None:
        include_j6_yaw = integrated_planar_ik_yaw
        preposition_step = replace(
            preposition_step,
            pose=with_current_orientation(preposition_step.pose, post_yaw_orientation),
            planar_ik_include_j6_yaw=include_j6_yaw,
        )
        grasp_step = replace(
            grasp_step,
            pose=with_current_orientation(grasp_step.pose, post_yaw_orientation),
            planar_ik_include_j6_yaw=include_j6_yaw,
        )
        lift_step = replace(
            lift_step,
            pose=with_current_orientation(lift_step.pose, post_yaw_orientation),
            planar_ik_include_j6_yaw=include_j6_yaw,
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
    preposition_pose_for_descent = preposition_step.pose
    preposition_elapsed_s = time.monotonic() - started_s
    final_wait_s = wait_for_dynamic_final_action(
        timing,
        elapsed_s=preposition_elapsed_s,
        final_action_time_s=float(args.final_action_time_s),
        placement="wait" if not nowait else "preposition_after_nowait",
    )
    final_action_late_s = max(0.0, preposition_elapsed_s - final_action_due_s)

    guarded_result: GuardedDescentResult | None = None
    if args.guarded_descent:
        guarded_result = execute_guarded_descent(
            controller,
            args,
            start_pose=preposition_pose_for_descent,
            target_pose=grasp_step.pose,
        )
        counts[grasp_step.label] = guarded_result.send_count
        if guarded_result.aborted:
            reason = guarded_result.reason or "guarded descent aborted"
            print(f"Protected abort: {reason}", flush=True)
            return DynamicExecutionResult(
                send_counts=counts,
                protected_abort=True,
                protected_abort_reason=reason,
                preposition_z_mm=preposition_pose.z_mm,
                preposition_source=preposition_source,
                preposition_elapsed_s=preposition_elapsed_s,
                final_action_due_s=final_action_due_s,
                final_wait_s=final_wait_s,
                final_action_late_s=final_action_late_s,
                guarded_descent=guarded_result,
                safe_gripper_results=tuple(gripper_results),
                joint_pitch_adjustments=tuple(joint_pitch_adjustments),
                grasp_yaw_adjustments=tuple(grasp_yaw_adjustments),
                planar_ik_results=tuple(planar_ik_results),
            )
        pitch_adjustment = apply_joint_pitch_adjustment(
            controller,
            args,
            grasp_step,
            phase="after_guarded_descent",
        )
        if pitch_adjustment is not None:
            joint_pitch_adjustments.append(pitch_adjustment)
        gripper_result = send_gripper(controller, args, grasp_step.gripper_after)
        record_gripper_result(gripper_results, gripper_result)
    else:
        send_count, gripper_result = execute_step(
            controller,
            args,
            grasp_step,
            joint_pitch_adjustments=joint_pitch_adjustments,
            grasp_yaw_adjustments=grasp_yaw_adjustments,
            planar_ik_results=planar_ik_results,
            planar_ik_reuse_state=planar_ik_reuse_state,
        )
        counts[grasp_step.label] = send_count
        record_gripper_result(gripper_results, gripper_result)

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
        guarded_descent=guarded_result,
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
    step = SequenceStep("startup_ready", ready_pose(args), args.transfer_mode)
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
    target = target_from_command(
        command,
        camera_to_base,
        args,
    )
    plan = YoloCatchPlan(
        task=command.task,
        target=target,
        steps=build_steps(args, target),
    )
    validate_steps(controller or validator, plan.steps)
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
                    args.vertical_mode,
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
            "grasp_z_mm": target.grasp_z_mm,
            "raw_grasp_z_mm": target.raw_grasp_z_mm,
            "grasp_z_source": target.grasp_z_source,
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
            "protected_abort": execution.protected_abort,
            "protected_abort_reason": execution.protected_abort_reason,
        }
    )
    guarded_execution: dict[str, Any] = {
        "guarded_descent_enabled": bool(args.guarded_descent),
        "protected_abort": execution.protected_abort,
        "protected_abort_reason": execution.protected_abort_reason,
        "protected_abort_return": execution.protected_abort_return,
        "protected_abort_return_label": execution.protected_abort_return_label,
    }
    if execution.guarded_descent is not None:
        guarded_execution["guarded_descent"] = {
            "send_count": execution.guarded_descent.send_count,
            "start_z_mm": execution.guarded_descent.start_z_mm,
            "target_z_mm": execution.guarded_descent.target_z_mm,
            "aborted": execution.guarded_descent.aborted,
            "reason": execution.guarded_descent.reason,
            "last_feedback_z_mm": execution.guarded_descent.last_feedback_z_mm,
            "last_follow_error_mm": execution.guarded_descent.last_follow_error_mm,
            "last_progress_mm": execution.guarded_descent.last_progress_mm,
            "zone_mm": args.guarded_descent_zone_mm,
            "step_mm": args.guarded_descent_step_mm,
            "speed": args.guarded_descent_speed,
            "follow_error_mm": args.guarded_follow_error_mm,
            "stall_s": args.guarded_stall_s,
            "start_tolerance_mm": args.guarded_start_tolerance_mm,
            "start_timeout_s": args.guarded_start_timeout_s,
            "progress_min_mm": args.guarded_progress_min_mm,
            "step_timeout_s": (
                args.guarded_step_timeout_s
                if args.guarded_step_timeout_s > 0.0
                else args.guarded_stall_s
            ),
        }
    response["payload"]["guarded_execution"] = guarded_execution
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
            "guarded_descent_enabled": bool(args.guarded_descent),
            "preposition_z_mm": execution.preposition_z_mm,
            "preposition_source": execution.preposition_source,
            "preposition_clearance_mm": args.dynamic_preposition_clearance_mm,
            "preposition_elapsed_s": execution.preposition_elapsed_s,
            "final_action_due_s": execution.final_action_due_s,
            "final_wait_s": execution.final_wait_s,
            "final_action_late_s": execution.final_action_late_s,
            "protected_abort": execution.protected_abort,
            "protected_abort_reason": execution.protected_abort_reason,
            "protected_abort_return": execution.protected_abort_return,
            "protected_abort_return_label": execution.protected_abort_return_label,
        }
        if "guarded_descent" in guarded_execution:
            dynamic_execution["guarded_descent"] = guarded_execution["guarded_descent"]
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

    config = replace(ROBOT_CONFIG, can_name=args.can)
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
                f"speed={args.speed}, startup_speed={args.startup_speed}, "
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

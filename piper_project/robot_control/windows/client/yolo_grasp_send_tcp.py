"""YOLO visual grasp client: ranked detections -> camera XYZ -> TCP send.

Run on Windows from ``piper_project`` after starting the Linux catch server:

    python -m robot_control.windows.client.yolo_grasp_send_tcp --host <linux-vm-ip>

The script loads ``models/yolo26n.pt`` by default, detects objects in the
aligned RealSense RGB image, ranks valid targets by depth from near to far, and
lets the operator type a displayed rank in the terminal to send that target.
Press ``Esc`` in the image window or type ``q`` in the terminal to exit.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np

from robot_control.linux.catch.hand_eye_mapping import (
    load_camera_to_base_transform,
    map_camera_point_to_base_xyz_m,
)
from robot_control.shared.protocol import (
    CameraPointCommand,
    CameraPointTask,
    send_camera_point,
)
from robot_control.shared.paths import PROJECT_DIR
from robot_control.windows.vision.d435i_intrinsics import (
    DEFAULT_D435I_PARA,
    PinholeIntrinsics,
    find_d435i_intrinsics,
    format_intrinsics,
)
from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    deproject_pixel_to_camera_xyz,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)
from robot_control.windows.vision.detector import Detection
from robot_control.windows.vision.yolo_detector import DEFAULT_MODEL_PATH, YOLOObjectDetector


RGB_WINDOW_NAME = "YOLO ranked grasp -> catch_server"
DEPTH_WINDOW_NAME = "Aligned depth valid map"
DEFAULT_HAND_EYE_RESULT = (
    PROJECT_DIR / "robot_control" / "linux" / "hand_eye" / "hand_eye_result.json"
)
SendCameraPointFunc = Callable[[str, int, CameraPointCommand, float], dict[str, Any]]
GraspRunMode = Literal["static", "dynamic", "probe"]


@dataclass(frozen=True)
class RankedTarget:
    rank: int
    detection: Detection
    pixel: tuple[int, int]
    raw_point: CameraPoint
    send_point: CameraPoint

    @property
    def depth_mm(self) -> float:
        return self.raw_point.depth_m * 1000.0


@dataclass(frozen=True)
class DynamicGraspCandidate:
    target: RankedTarget
    center_point: CameraPoint
    base_xyz_m: tuple[float, float, float]
    t_arrive_s: float
    remaining_y_mm: float
    distance_to_pick_y_mm: float

    @property
    def base_x_mm(self) -> float:
        return self.base_xyz_m[0] * 1000.0

    @property
    def base_y_mm(self) -> float:
        return self.base_xyz_m[1] * 1000.0

    @property
    def base_z_mm(self) -> float:
        return self.base_xyz_m[2] * 1000.0


@dataclass(frozen=True)
class SuppressedDynamicTarget:
    x_mm: float
    y_mm: float
    expires_at_s: float


class TerminalInput:
    """Read terminal commands without blocking the camera/display loop."""

    def __init__(self) -> None:
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get_nowait(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _read_loop(self) -> None:
        while True:
            try:
                line = input()
            except EOFError:
                self._commands.put("q")
                return
            self._commands.put(line.strip())


class YoloGraspSender:
    def __init__(
        self,
        host: str | None,
        port: int,
        timeout_s: float,
        dry_run: bool = False,
        task: CameraPointTask = "out",
        send_func: SendCameraPointFunc = send_camera_point,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.dry_run = bool(dry_run)
        self.task = task
        self._send_func = send_func

    def send_target(
        self,
        target: RankedTarget,
        source: str,
        task: CameraPointTask | None = None,
        grasp_mode: GraspRunMode = "static",
        t_arrive_s: float | None = None,
        conveyor_speed_mm_s: float | None = None,
        pick_x_mm: float | None = None,
        pick_y_mm: float | None = None,
        detected_xyz_mm: tuple[float, float, float] | None = None,
    ) -> dict[str, Any]:
        task = self.task if task is None else task
        command = camera_point_command_from_target(
            target,
            source=source,
            task=task,
            grasp_mode=grasp_mode,
            t_arrive_s=t_arrive_s,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            pick_x_mm=pick_x_mm,
            pick_y_mm=pick_y_mm,
            detected_xyz_mm=detected_xyz_mm,
        )
        if self.dry_run:
            return {"type": "dry_run", "payload": command_payload(command)}
        if not self.host:
            raise ValueError("--host is required unless --dry-run is used")
        return self._send_func(self.host, self.port, command, self.timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect objects with models/yolo26n.pt, rank valid bboxes by depth, "
            "and send the selected camera XYZ to the Linux catch server."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "static keeps the original manual rank-send workflow; dynamic "
            "automatically sends one target when it reaches the configured "
            "conveyor timing window."
        ),
    )
    parser.add_argument("--host", help="Linux VM IP address running catch_server")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="print target XYZ without sending")
    parser.add_argument(
        "--task",
        choices=("out", "in"),
        dest="task_option",
        help=(
            "out: grasp detected bbox target and place it at the fixed place point; "
            "in: grasp detected bbox target and release at the initial ready point. "
            "This is only the default for bare rank commands; terminal commands "
            "can override it with 'out 1' or 'in 2'."
        ),
    )
    parser.set_defaults(task="out")

    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=5)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=DEFAULT_D435I_PARA,
        help="D435i parameter JSON used to select rectified fx/fy/ppx/ppy by width/height",
    )
    parser.add_argument(
        "--use-realsense-intrinsics",
        action="store_true",
        help="ignore --intrinsics-json and use intrinsics reported by the RealSense frame",
    )

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional COCO class ids to keep, e.g. --classes 39 for bottle.",
    )
    parser.add_argument(
        "--bbox-up-ratio",
        type=float,
        default=0.15,
        help=(
            "Move the grasp pixel upward from bbox center by this fraction of bbox "
            "height. 0 uses the center; 0.5 reaches the top edge."
        ),
    )
    parser.add_argument(
        "--bbox-up-px",
        type=float,
        default=0.0,
        help="Additional image-space upward pixel offset from the bbox center.",
    )
    parser.add_argument(
        "--camera-up-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Additional 3D offset in the RealSense image-up direction after "
            "deprojection. RealSense image-up is negative camera Y."
        ),
    )
    parser.add_argument(
        "--point-z-offset-mm",
        type=float,
        default=0.0,
        help=(
            "Optional camera-frame Z offset added before sending. This is depth "
            "axis, not robot-base vertical Z."
        ),
    )
    parser.add_argument(
        "--hand-eye-result",
        nargs="?",
        const=DEFAULT_HAND_EYE_RESULT,
        default=DEFAULT_HAND_EYE_RESULT,
        type=Path,
        help=(
            "Optional eye_to_hand hand_eye_result.json camera->base transform. "
            f"Default: {DEFAULT_HAND_EYE_RESULT}. Supplying the flag without a "
            "value uses the same default."
        ),
    )
    parser.add_argument(
        "--no-hand-eye-result",
        dest="hand_eye_result",
        action="store_const",
        const=None,
        help="Do not load a hand-eye transform. Requires --base-z-offset-mm 0.",
    )
    parser.add_argument(
        "--base-z-offset-mm",
        type=float,
        default=20.0,
        help=(
            "Optional robot base-frame Z offset added to the selected target "
            "before converting back to camera XYZ for TCP send. Requires "
            "--hand-eye-result."
        ),
    )
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--conveyor-speed-mm-s",
        type=float,
        help="dynamic mode conveyor speed magnitude in mm/s; required for dynamic mode",
    )
    parser.add_argument(
        "--conveyor-direction",
        choices=("negative-y", "positive-y"),
        default="negative-y",
        help=(
            "dynamic mode object travel direction in robot base coordinates. "
            "Default assumes objects move toward smaller base Y values."
        ),
    )
    parser.add_argument(
        "--dynamic-min-x-mm",
        type=float,
        default=300.0,
        help="dynamic mode lower base-X bound for accepting a target center",
    )
    parser.add_argument(
        "--dynamic-max-x-mm",
        type=float,
        default=400.0,
        help="dynamic mode upper base-X bound for accepting a target center",
    )
    parser.add_argument(
        "--dynamic-min-y-mm",
        type=float,
        default=-30.0,
        help="dynamic mode lower base-Y bound for accepting a target center",
    )
    parser.add_argument(
        "--dynamic-max-y-mm",
        type=float,
        default=30.0,
        help="dynamic mode upper base-Y bound for accepting a target center",
    )
    parser.add_argument(
        "--dynamic-pick-x-mm",
        type=float,
        default=350.0,
        help="fixed robot base X for dynamic grasp execution",
    )
    parser.add_argument(
        "--dynamic-pick-x-from-bbox-center",
        action="store_true",
        help=(
            "use the detected bbox center base-X as dynamic grasp X instead of "
            "--dynamic-pick-x-mm; dynamic Y still comes from --dynamic-pick-y-mm"
        ),
    )
    parser.add_argument(
        "--dynamic-pick-y-mm",
        type=float,
        default=-30.0,
        help="fixed robot base Y line used to compute dynamic T_arrive",
    )
    parser.add_argument(
        "--dynamic-keep-running",
        action="store_true",
        help="continue after a dynamic send instead of exiting after the first target",
    )
    parser.add_argument(
        "--dynamic-skip-memory-s",
        type=float,
        default=5.0,
        help=(
            "after a dynamic target is skipped, ignore nearby repeated detections "
            "for this many seconds"
        ),
    )
    parser.add_argument(
        "--dynamic-skip-radius-mm",
        type=float,
        default=80.0,
        help="base-XY radius used to match repeated skipped dynamic detections",
    )
    parser.add_argument(
        "--dynamic-probe-interval-s",
        type=float,
        default=1.0,
        help=(
            "dynamic mode interval for sending rank-1 bbox center probe points "
            "to Linux for camera->base coordinate logging; 0 disables probes"
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    task_option = getattr(args, "task_option", None)
    if task_option is not None:
        args.task = task_option
    if not args.dry_run and not args.host:
        raise ValueError("--host is required unless --dry-run is used")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.sample_radius < 0:
        raise ValueError("--sample-radius must be >= 0")
    if args.max_results < 1:
        raise ValueError("--max-results must be >= 1")
    if args.bbox_up_ratio < 0.0 or args.bbox_up_ratio > 0.5:
        raise ValueError("--bbox-up-ratio must be in [0, 0.5]")
    if args.bbox_up_px < 0.0:
        raise ValueError("--bbox-up-px must be >= 0")
    if args.print_interval < 0.0:
        raise ValueError("--print-interval must be >= 0")
    if args.base_z_offset_mm != 0.0 and args.hand_eye_result is None:
        raise ValueError("--base-z-offset-mm requires --hand-eye-result")
    if args.mode == "dynamic":
        if args.hand_eye_result is None:
            raise ValueError("dynamic mode requires --hand-eye-result")
        if args.conveyor_speed_mm_s is None:
            raise ValueError("--conveyor-speed-mm-s is required in dynamic mode")
        if args.conveyor_speed_mm_s <= 0.0:
            raise ValueError("--conveyor-speed-mm-s must be > 0 in dynamic mode")
        if args.dynamic_min_x_mm > args.dynamic_max_x_mm:
            raise ValueError("--dynamic-min-x-mm must be <= --dynamic-max-x-mm")
        if args.dynamic_min_y_mm > args.dynamic_max_y_mm:
            raise ValueError("--dynamic-min-y-mm must be <= --dynamic-max-y-mm")
        if args.dynamic_skip_memory_s < 0.0:
            raise ValueError("--dynamic-skip-memory-s must be >= 0")
        if args.dynamic_skip_radius_mm <= 0.0:
            raise ValueError("--dynamic-skip-radius-mm must be > 0")
        if args.dynamic_probe_interval_s < 0.0:
            raise ValueError("--dynamic-probe-interval-s must be >= 0")


def make_depth_colormap(depth_frame: Any, max_depth_m: float) -> Any:
    depth_units = float(depth_frame.get_units())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * depth_units

    valid_mask = depth_m > 0.0
    normalized = np.zeros_like(depth_m, dtype=np.uint8)
    if max_depth_m > 0.0:
        clipped = np.clip(depth_m, 0.0, max_depth_m)
        normalized = (255.0 * (1.0 - clipped / max_depth_m)).astype(np.uint8)

    colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colormap[~valid_mask] = (0, 0, 0)
    return colormap


def grasp_pixel_from_detection(
    detection: Detection,
    *,
    bbox_up_ratio: float,
    bbox_up_px: float,
) -> tuple[int, int]:
    u = int(round(detection.x + detection.width / 2.0))
    v_float = detection.y + detection.height / 2.0
    v_float -= detection.height * float(bbox_up_ratio)
    v_float -= float(bbox_up_px)
    top = int(detection.y)
    bottom = int(detection.y + max(0, detection.height - 1))
    v = int(round(max(top, min(bottom, v_float))))
    return u, v


def offset_camera_point(
    point: CameraPoint,
    *,
    camera_up_offset_mm: float,
    point_z_offset_mm: float,
) -> CameraPoint:
    return CameraPoint(
        u=point.u,
        v=point.v,
        depth_m=point.depth_m,
        x_m=point.x_m,
        y_m=point.y_m - float(camera_up_offset_mm) / 1000.0,
        z_m=point.z_m + float(point_z_offset_mm) / 1000.0,
    )


def apply_base_z_offset(
    point: CameraPoint,
    *,
    camera_to_base: np.ndarray | None,
    base_z_offset_mm: float,
) -> CameraPoint:
    if base_z_offset_mm == 0.0:
        return point
    if camera_to_base is None:
        raise ValueError("--base-z-offset-mm requires --hand-eye-result")
    if camera_to_base.shape != (4, 4):
        raise ValueError("camera_to_base transform must be 4x4")

    camera_point = np.array([point.x_m, point.y_m, point.z_m, 1.0], dtype=float)
    base_point = camera_to_base @ camera_point
    base_point[2] += float(base_z_offset_mm) / 1000.0
    adjusted_camera = np.linalg.inv(camera_to_base) @ base_point
    if not np.all(np.isfinite(adjusted_camera[:3])):
        raise ValueError("base-Z adjusted camera point is not finite")
    return CameraPoint(
        u=point.u,
        v=point.v,
        depth_m=point.depth_m,
        x_m=float(adjusted_camera[0]),
        y_m=float(adjusted_camera[1]),
        z_m=float(adjusted_camera[2]),
    )


def deproject_detection_center(
    detection: Detection,
    depth_frame: Any,
    *,
    sample_radius: int,
    intrinsics: PinholeIntrinsics | None,
) -> CameraPoint:
    return deproject_pixel_to_camera_xyz(
        depth_frame,
        detection.u,
        detection.v,
        sample_radius=sample_radius,
        intrinsics=intrinsics,
    )


def dynamic_arrival_time_s(
    center_y_mm: float,
    *,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> tuple[float, float]:
    if conveyor_speed_mm_s <= 0.0:
        raise ValueError("conveyor speed must be > 0")
    if conveyor_direction == "negative-y":
        remaining_y_mm = center_y_mm - pick_y_mm
    elif conveyor_direction == "positive-y":
        remaining_y_mm = pick_y_mm - center_y_mm
    else:
        raise ValueError("conveyor_direction must be 'negative-y' or 'positive-y'")
    return remaining_y_mm / conveyor_speed_mm_s, remaining_y_mm


def dynamic_candidate_from_target(
    target: RankedTarget,
    center_point: CameraPoint,
    *,
    camera_to_base: np.ndarray,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> DynamicGraspCandidate | None:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        center_point.x_m,
        center_point.y_m,
        center_point.z_m,
        camera_to_base,
    )
    center_x_mm = base_xyz_m[0] * 1000.0
    center_y_mm = base_xyz_m[1] * 1000.0
    if center_x_mm < min_x_mm or center_x_mm > max_x_mm:
        return None
    if center_y_mm < min_y_mm or center_y_mm > max_y_mm:
        return None

    t_arrive_s, remaining_y_mm = dynamic_arrival_time_s(
        center_y_mm,
        pick_y_mm=pick_y_mm,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        conveyor_direction=conveyor_direction,
    )
    if t_arrive_s < 0.0:
        return None

    return DynamicGraspCandidate(
        target=target,
        center_point=center_point,
        base_xyz_m=base_xyz_m,
        t_arrive_s=float(t_arrive_s),
        remaining_y_mm=float(remaining_y_mm),
        distance_to_pick_y_mm=abs(center_y_mm - pick_y_mm),
    )


def find_dynamic_candidates(
    targets: list[RankedTarget],
    depth_frame: Any,
    *,
    sample_radius: int,
    intrinsics: PinholeIntrinsics | None,
    camera_to_base: np.ndarray,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> list[DynamicGraspCandidate]:
    candidates: list[DynamicGraspCandidate] = []
    for target in targets:
        try:
            center_point = deproject_detection_center(
                target.detection,
                depth_frame,
                sample_radius=sample_radius,
                intrinsics=intrinsics,
            )
        except ValueError:
            continue
        candidate = dynamic_candidate_from_target(
            target,
            center_point,
            camera_to_base=camera_to_base,
            min_x_mm=min_x_mm,
            max_x_mm=max_x_mm,
            min_y_mm=min_y_mm,
            max_y_mm=max_y_mm,
            pick_y_mm=pick_y_mm,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            conveyor_direction=conveyor_direction,
        )
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (item.t_arrive_s, item.target.rank))


def find_dynamic_candidate(
    targets: list[RankedTarget],
    depth_frame: Any,
    *,
    sample_radius: int,
    intrinsics: PinholeIntrinsics | None,
    camera_to_base: np.ndarray,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> DynamicGraspCandidate | None:
    candidates = find_dynamic_candidates(
        targets,
        depth_frame,
        sample_radius=sample_radius,
        intrinsics=intrinsics,
        camera_to_base=camera_to_base,
        min_x_mm=min_x_mm,
        max_x_mm=max_x_mm,
        min_y_mm=min_y_mm,
        max_y_mm=max_y_mm,
        pick_y_mm=pick_y_mm,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        conveyor_direction=conveyor_direction,
    )
    return candidates[0] if candidates else None


def detections_to_ranked_targets(
    detections: list[Detection],
    depth_frame: Any,
    *,
    sample_radius: int,
    intrinsics: PinholeIntrinsics | None,
    bbox_up_ratio: float,
    bbox_up_px: float,
    camera_up_offset_mm: float,
    point_z_offset_mm: float,
    camera_to_base: np.ndarray | None = None,
    base_z_offset_mm: float = 0.0,
) -> tuple[list[RankedTarget], dict[int, str]]:
    unranked: list[RankedTarget] = []
    errors: dict[int, str] = {}
    for index, detection in enumerate(detections):
        u, v = grasp_pixel_from_detection(
            detection,
            bbox_up_ratio=bbox_up_ratio,
            bbox_up_px=bbox_up_px,
        )
        try:
            raw_point = deproject_pixel_to_camera_xyz(
                depth_frame,
                u,
                v,
                sample_radius=sample_radius,
                intrinsics=intrinsics,
            )
        except ValueError as exc:
            errors[index] = str(exc)
            continue
        send_point = offset_camera_point(
            raw_point,
            camera_up_offset_mm=camera_up_offset_mm,
            point_z_offset_mm=point_z_offset_mm,
        )
        try:
            send_point = apply_base_z_offset(
                send_point,
                camera_to_base=camera_to_base,
                base_z_offset_mm=base_z_offset_mm,
            )
        except ValueError as exc:
            errors[index] = str(exc)
            continue
        unranked.append(
            RankedTarget(
                rank=0,
                detection=detection,
                pixel=(u, v),
                raw_point=raw_point,
                send_point=send_point,
            )
        )

    ranked = sorted(
        unranked,
        key=lambda target: (
            target.raw_point.depth_m,
            target.detection.u,
            target.detection.v,
        ),
    )
    return [
        RankedTarget(
            rank=index,
            detection=target.detection,
            pixel=target.pixel,
            raw_point=target.raw_point,
            send_point=target.send_point,
        )
        for index, target in enumerate(ranked, start=1)
    ], errors


def camera_point_command_from_target(
    target: RankedTarget,
    source: str,
    task: CameraPointTask = "out",
    grasp_mode: GraspRunMode = "static",
    t_arrive_s: float | None = None,
    conveyor_speed_mm_s: float | None = None,
    pick_x_mm: float | None = None,
    pick_y_mm: float | None = None,
    detected_xyz_mm: tuple[float, float, float] | None = None,
) -> CameraPointCommand:
    detection = target.detection
    detected_x_mm = None
    detected_y_mm = None
    detected_z_mm = None
    if detected_xyz_mm is not None:
        detected_x_mm, detected_y_mm, detected_z_mm = detected_xyz_mm
    return CameraPointCommand(
        x_m=target.send_point.x_m,
        y_m=target.send_point.y_m,
        z_m=target.send_point.z_m,
        u=target.send_point.u,
        v=target.send_point.v,
        depth_m=target.raw_point.depth_m,
        source=source,
        task=task,
        rank=target.rank,
        label=detection.label,
        confidence=detection.confidence,
        class_id=detection.class_id,
        bbox_xywh=(
            float(detection.x),
            float(detection.y),
            float(detection.width),
            float(detection.height),
        ),
        grasp_mode=grasp_mode,
        t_arrive_s=t_arrive_s,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        pick_x_mm=pick_x_mm,
        pick_y_mm=pick_y_mm,
        detected_x_mm=detected_x_mm,
        detected_y_mm=detected_y_mm,
        detected_z_mm=detected_z_mm,
    )


def command_payload(command: CameraPointCommand) -> dict[str, Any]:
    return asdict(command)


def find_ranked_target(
    targets: list[RankedTarget],
    rank: int,
) -> RankedTarget | None:
    for target in targets:
        if target.rank == rank:
            return target
    return None


def format_target_line(target: RankedTarget) -> str:
    detection = target.detection
    confidence = detection.confidence if detection.confidence is not None else 0.0
    return (
        f"{target.rank}: {detection.label} conf={confidence:.2f} "
        f"depth={target.depth_mm:.1f}mm "
        f"bbox=({detection.x},{detection.y},{detection.width},{detection.height}) "
        f"pixel=({target.send_point.u},{target.send_point.v}) "
        f"send=({target.send_point.x_m:.4f},"
        f"{target.send_point.y_m:.4f},{target.send_point.z_m:.4f})m"
    )


def print_target_table(
    targets: list[RankedTarget],
    errors: dict[int, str],
    *,
    show_commands: bool = True,
) -> None:
    if targets:
        print("\nCurrent YOLO targets ranked by depth:", flush=True)
        for target in targets:
            print("  " + format_target_line(target), flush=True)
        if show_commands:
            print(
                "Type 'out N' or 'in N' to send a target, p to print again, q to quit.",
                flush=True,
            )
    elif errors:
        print(f"\nNo valid depth target: {next(iter(errors.values()))}", flush=True)
    else:
        print("\nNo YOLO target found.", flush=True)


def draw_ranked_targets(
    image: Any,
    detections: list[Detection],
    targets: list[RankedTarget],
    errors: dict[int, str],
    status: str,
) -> None:
    target_by_detection = {target.detection: target for target in targets}
    error_indexes = set(errors)

    for index, detection in enumerate(detections):
        target = target_by_detection.get(detection)
        has_error = index in error_indexes and target is None
        color = (0, 255, 0) if target is not None else (0, 0, 255) if has_error else (120, 120, 120)
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
            2,
        )

        marker_pixel = target.pixel if target is not None else (detection.u, detection.v)
        cv2.drawMarker(
            image,
            marker_pixel,
            (0, 255, 255) if target is not None else color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

        confidence = detection.confidence if detection.confidence is not None else 0.0
        if target is not None:
            text = f"{target.rank} {detection.label} {confidence:.2f} {target.depth_mm:.0f}mm"
        elif has_error:
            text = f"{detection.label} no depth"
        else:
            text = f"{detection.label} {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (detection.x, max(18, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (detection.x, max(18, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if targets:
        text = f"targets={len(targets)} nearest: " + format_camera_point(targets[0].send_point)
        text_color = (0, 255, 255)
    elif errors:
        text = next(iter(errors.values()))
        text_color = (0, 0, 255)
    else:
        text = status
        text_color = (255, 255, 255)

    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        text_color,
        1,
        cv2.LINE_AA,
    )


def send_and_print(
    sender: YoloGraspSender,
    target: RankedTarget,
    source: str,
    task: CameraPointTask,
) -> dict[str, Any]:
    response = sender.send_target(target, source=source, task=task)
    print(
        f"\nSent {task} target rank {target.rank}: {format_target_line(target)}",
        flush=True,
    )
    print(response, flush=True)
    return response


def format_dynamic_candidate(
    candidate: DynamicGraspCandidate,
    *,
    pick_y_mm: float | None = None,
) -> str:
    pick_y_text = "" if pick_y_mm is None else f"pick_y={pick_y_mm:.1f}mm "
    return (
        f"rank={candidate.target.rank} "
        f"base=({candidate.base_x_mm:.1f},{candidate.base_y_mm:.1f},"
        f"{candidate.base_z_mm:.1f})mm "
        f"{pick_y_text}"
        f"remain_y={candidate.remaining_y_mm:.1f}mm "
        f"t_arrive={candidate.t_arrive_s:.3f}s"
    )


def send_dynamic_candidate(
    sender: YoloGraspSender,
    candidate: DynamicGraspCandidate,
    *,
    source: str,
    task: CameraPointTask,
    conveyor_speed_mm_s: float,
    pick_x_mm: float,
    pick_y_mm: float,
) -> dict[str, Any]:
    response = sender.send_target(
        candidate.target,
        source=source,
        task=task,
        grasp_mode="dynamic",
        t_arrive_s=candidate.t_arrive_s,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        pick_x_mm=pick_x_mm,
        pick_y_mm=pick_y_mm,
        detected_xyz_mm=(
            candidate.base_x_mm,
            candidate.base_y_mm,
            candidate.base_z_mm,
        ),
    )
    print(
        "\nSent dynamic "
        f"{task} target: {format_dynamic_candidate(candidate, pick_y_mm=pick_y_mm)} "
        f"fixed_pick=({pick_x_mm:.1f},{pick_y_mm:.1f})mm",
        flush=True,
    )
    print(response, flush=True)
    return response


def dynamic_pick_x_mm_for_candidate(
    candidate: DynamicGraspCandidate,
    *,
    fixed_pick_x_mm: float,
    use_bbox_center_x: bool,
) -> float:
    if use_bbox_center_x:
        return candidate.base_x_mm
    return float(fixed_pick_x_mm)


def send_dynamic_probe(
    sender: YoloGraspSender,
    target: RankedTarget,
    *,
    source: str,
    task: CameraPointTask,
    center_point: CameraPoint,
) -> dict[str, Any]:
    probe_target = RankedTarget(
        rank=target.rank,
        detection=target.detection,
        pixel=(center_point.u, center_point.v),
        raw_point=center_point,
        send_point=center_point,
    )
    response = sender.send_target(
        probe_target,
        source=source,
        task=task,
        grasp_mode="probe",
    )
    print(
        "\nSent dynamic bbox-center probe "
        f"{task} rank {probe_target.rank}: {format_target_line(probe_target)}",
        flush=True,
    )
    print(response, flush=True)
    return response


def dynamic_skip_reason_from_response(response: dict[str, Any]) -> str | None:
    payload = response.get("payload")
    if isinstance(payload, dict) and payload.get("skipped") is True:
        timing = payload.get("dynamic_timing")
        if isinstance(timing, dict) and timing.get("skip_reason"):
            return str(timing["skip_reason"])
        return "dynamic target skipped by server"
    if response.get("type") == "error" and isinstance(payload, dict):
        error = str(payload.get("error", ""))
        if "too close to the pick line" in error:
            return error
    return None


def prune_suppressed_dynamic_targets(
    suppressed: list[SuppressedDynamicTarget],
    now_s: float,
) -> list[SuppressedDynamicTarget]:
    return [target for target in suppressed if target.expires_at_s > now_s]


def dynamic_candidate_is_suppressed(
    candidate: DynamicGraspCandidate,
    suppressed: list[SuppressedDynamicTarget],
    *,
    radius_mm: float,
) -> bool:
    radius_sq = float(radius_mm) * float(radius_mm)
    for target in suppressed:
        dx = candidate.base_x_mm - target.x_mm
        dy = candidate.base_y_mm - target.y_mm
        if dx * dx + dy * dy <= radius_sq:
            return True
    return False


def remember_suppressed_dynamic_target(
    suppressed: list[SuppressedDynamicTarget],
    candidate: DynamicGraspCandidate,
    *,
    now_s: float,
    memory_s: float,
) -> list[SuppressedDynamicTarget]:
    if memory_s <= 0.0:
        return suppressed
    return [
        *suppressed,
        SuppressedDynamicTarget(
            x_mm=candidate.base_x_mm,
            y_mm=candidate.base_y_mm,
            expires_at_s=now_s + memory_s,
        ),
    ]


def parse_target_command(command: str, default_task: CameraPointTask) -> tuple[CameraPointTask, int] | None:
    parts = command.strip().lower().split()
    if len(parts) == 1:
        try:
            return default_task, int(parts[0])
        except ValueError:
            return None
    if len(parts) == 2 and parts[0] in ("out", "in"):
        try:
            return parts[0], int(parts[1])
        except ValueError:
            return None
    return None


def process_terminal_command(
    command: str,
    *,
    targets: list[RankedTarget],
    errors: dict[int, str],
    sender: YoloGraspSender,
    source: str,
) -> bool:
    command = command.strip().lower()
    if command == "":
        print_target_table(targets, errors)
        return True
    if command in {"q", "quit", "exit"}:
        return False
    if command in {"p", "print", "ls", "list"}:
        print_target_table(targets, errors)
        return True

    parsed = parse_target_command(command, sender.task)
    if parsed is None:
        print(
            f"Unknown command: {command!r}. Type 'out N', 'in N', p, or q.",
            flush=True,
        )
        return True
    task, rank = parsed

    target = find_ranked_target(targets, rank)
    if target is None:
        valid = ", ".join(str(item.rank) for item in targets) or "none"
        print(f"Rank {rank} is not available in the latest frame. Valid ranks: {valid}", flush=True)
        return True
    try:
        send_and_print(sender, target, source=source, task=task)
    except Exception as exc:  # noqa: BLE001 - manual hardware UI should print full failure
        print(f"Send failed for {task} rank {rank}: {exc}", flush=True)
    return True


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    intrinsics = None
    if not args.use_realsense_intrinsics:
        intrinsics = find_d435i_intrinsics(
            args.width,
            args.height,
            args.intrinsics_json,
        )
        print(f"Using D435i rectified intrinsics: {format_intrinsics(intrinsics)}", flush=True)
    else:
        print("Using RealSense frame-reported intrinsics.", flush=True)

    camera_to_base = None
    if args.hand_eye_result is not None:
        camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
        print(f"Using hand-eye camera->base transform: {args.hand_eye_result}", flush=True)

    detector = YOLOObjectDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        max_results=args.max_results,
        imgsz=args.imgsz,
        device=args.device,
        class_ids=args.classes,
    )
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    sender = YoloGraspSender(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
        task=args.task,
    )
    terminal = TerminalInput()
    if args.mode == "static":
        terminal.start()
    source = "yolo_bbox_center_up" if args.mode == "static" else "yolo_dynamic_timed"
    last_print_time = 0.0
    latest_targets: list[RankedTarget] = []
    latest_errors: dict[int, str] = {}
    dynamic_sent = False
    suppressed_dynamic_targets: list[SuppressedDynamicTarget] = []
    last_dynamic_probe_time = 0.0
    dynamic_status = (
        "Dynamic mode: waiting for target in "
        f"X=[{args.dynamic_min_x_mm:.1f},{args.dynamic_max_x_mm:.1f}]mm, "
        f"Y=[{args.dynamic_min_y_mm:.1f},{args.dynamic_max_y_mm:.1f}]mm."
    )

    if args.mode == "static":
        print(
            "Type 'out N' or 'in N' in this terminal to send a target. "
            "Type p to print targets, q to quit.",
            flush=True,
        )
    else:
        pick_x_text = (
            "bbox-center"
            if args.dynamic_pick_x_from_bbox_center
            else f"{args.dynamic_pick_x_mm:.1f}mm"
        )
        print(
            "Dynamic mode enabled. "
            f"speed={args.conveyor_speed_mm_s:.3f}mm/s "
            f"direction={args.conveyor_direction} "
            f"pick=(x={pick_x_text}, y={args.dynamic_pick_y_mm:.1f}mm). "
            "Press Esc to exit.",
            flush=True,
        )

    if not args.no_display:
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_depth:
            cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        running = True
        while running:
            color_image, depth_frame = wait_for_aligned_rgb_depth(
                pipeline,
                align,
                timeout_ms=args.frame_timeout_ms,
            )
            if color_image is None or depth_frame is None:
                continue

            detections = detector.detect(color_image)
            latest_targets, latest_errors = detections_to_ranked_targets(
                detections,
                depth_frame,
                sample_radius=args.sample_radius,
                intrinsics=intrinsics,
                bbox_up_ratio=args.bbox_up_ratio,
                bbox_up_px=args.bbox_up_px,
                camera_up_offset_mm=args.camera_up_offset_mm,
                point_z_offset_mm=args.point_z_offset_mm,
                camera_to_base=camera_to_base,
                base_z_offset_mm=args.base_z_offset_mm,
            )

            now = time.monotonic()
            if args.print_interval == 0.0 or now - last_print_time >= args.print_interval:
                print_target_table(
                    latest_targets,
                    latest_errors,
                    show_commands=args.mode == "static",
                )
                if args.mode == "dynamic":
                    print(dynamic_status, flush=True)
                last_print_time = now

            if args.mode == "static":
                while True:
                    command = terminal.get_nowait()
                    if command is None:
                        break
                    running = process_terminal_command(
                        command,
                        targets=latest_targets,
                        errors=latest_errors,
                        sender=sender,
                        source=source,
                    )
                    if not running:
                        break
            elif not dynamic_sent or args.dynamic_keep_running:
                if camera_to_base is None:
                    raise ValueError("dynamic mode requires --hand-eye-result")
                suppressed_dynamic_targets = prune_suppressed_dynamic_targets(
                    suppressed_dynamic_targets,
                    now,
                )
                candidates = find_dynamic_candidates(
                    latest_targets,
                    depth_frame,
                    sample_radius=args.sample_radius,
                    intrinsics=intrinsics,
                    camera_to_base=camera_to_base,
                    min_x_mm=args.dynamic_min_x_mm,
                    max_x_mm=args.dynamic_max_x_mm,
                    min_y_mm=args.dynamic_min_y_mm,
                    max_y_mm=args.dynamic_max_y_mm,
                    pick_y_mm=args.dynamic_pick_y_mm,
                    conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                    conveyor_direction=args.conveyor_direction,
                )
                candidate = None
                suppressed_count = 0
                for item in candidates:
                    if dynamic_candidate_is_suppressed(
                        item,
                        suppressed_dynamic_targets,
                        radius_mm=args.dynamic_skip_radius_mm,
                    ):
                        suppressed_count += 1
                        continue
                    candidate = item
                    break
                if candidate is not None:
                    pick_x_mm = dynamic_pick_x_mm_for_candidate(
                        candidate,
                        fixed_pick_x_mm=args.dynamic_pick_x_mm,
                        use_bbox_center_x=args.dynamic_pick_x_from_bbox_center,
                    )
                    dynamic_status = (
                        "Dynamic candidate "
                        + format_dynamic_candidate(
                            candidate,
                            pick_y_mm=args.dynamic_pick_y_mm,
                        )
                        + f" pick_x={pick_x_mm:.1f}mm"
                    )
                    try:
                        response = send_dynamic_candidate(
                            sender,
                            candidate,
                            source=source,
                            task=args.task,
                            conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                            pick_x_mm=pick_x_mm,
                            pick_y_mm=args.dynamic_pick_y_mm,
                        )
                        skip_reason = dynamic_skip_reason_from_response(response)
                        if skip_reason is not None:
                            suppressed_dynamic_targets = remember_suppressed_dynamic_target(
                                suppressed_dynamic_targets,
                                candidate,
                                now_s=now,
                                memory_s=args.dynamic_skip_memory_s,
                            )
                            dynamic_status = (
                                "Dynamic candidate skipped and remembered: "
                                f"{skip_reason}"
                            )
                            print(dynamic_status, flush=True)
                        elif response.get("type") == "error":
                            dynamic_status = f"Dynamic send failed: {response}"
                            print(dynamic_status, flush=True)
                        else:
                            dynamic_sent = True
                            if args.dynamic_keep_running:
                                suppressed_dynamic_targets = remember_suppressed_dynamic_target(
                                    suppressed_dynamic_targets,
                                    candidate,
                                    now_s=now,
                                    memory_s=args.dynamic_skip_memory_s,
                                )
                        if dynamic_sent and not args.dynamic_keep_running:
                            running = False
                    except Exception as exc:  # noqa: BLE001 - hardware UI should keep context
                        dynamic_status = f"Dynamic send failed: {exc}"
                        print(dynamic_status, flush=True)
                elif suppressed_count > 0:
                    dynamic_status = (
                        "Dynamic mode: "
                        f"{suppressed_count} repeated skipped target(s) ignored."
                    )
                else:
                    dynamic_status = (
                        "Dynamic mode: no target inside configured base-coordinate "
                        "window yet."
                    )
                    if (
                        args.dynamic_probe_interval_s > 0.0
                        and latest_targets
                        and (
                            last_dynamic_probe_time == 0.0
                            or now - last_dynamic_probe_time >= args.dynamic_probe_interval_s
                        )
                    ):
                        try:
                            probe_center = deproject_detection_center(
                                latest_targets[0].detection,
                                depth_frame,
                                sample_radius=args.sample_radius,
                                intrinsics=intrinsics,
                            )
                            response = send_dynamic_probe(
                                sender,
                                latest_targets[0],
                                source=source,
                                task=args.task,
                                center_point=probe_center,
                            )
                            if response.get("type") == "ack":
                                payload = response.get("payload", {})
                                if isinstance(payload, dict):
                                    base_xyz = payload.get("base_xyz_m")
                                    if isinstance(base_xyz, dict) and all(
                                        key in base_xyz for key in ("x_m", "y_m", "z_m")
                                    ):
                                        dynamic_status = (
                                            "Dynamic probe base="
                                            f"({float(base_xyz['x_m']) * 1000.0:.1f},"
                                            f"{float(base_xyz['y_m']) * 1000.0:.1f},"
                                            f"{float(base_xyz['z_m']) * 1000.0:.1f})mm"
                                        )
                            elif response.get("type") == "error":
                                dynamic_status = f"Dynamic probe failed: {response}"
                                print(dynamic_status, flush=True)
                        except Exception as exc:  # noqa: BLE001 - hardware UI should keep context
                            dynamic_status = f"Dynamic probe failed: {exc}"
                            print(dynamic_status, flush=True)
                        finally:
                            last_dynamic_probe_time = now

            if not args.no_display:
                if args.mode == "static":
                    status = "No YOLO target found. Type q in terminal or press Esc to exit."
                else:
                    status = dynamic_status
                display_image = color_image.copy()
                draw_ranked_targets(
                    display_image,
                    detections,
                    latest_targets,
                    latest_errors,
                    status,
                )
                cv2.imshow(RGB_WINDOW_NAME, display_image)

                if not args.hide_depth:
                    depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                    draw_ranked_targets(
                        depth_debug,
                        detections,
                        latest_targets,
                        latest_errors,
                        status,
                    )
                    cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

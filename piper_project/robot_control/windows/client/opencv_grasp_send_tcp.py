"""OpenCV belt-color grasp client: mask centroid -> median depth -> TCP.

Run on Windows from ``piper_project`` after starting the Linux catch server:

    python -m robot_control.windows.client.opencv_grasp_send_tcp --host <linux-vm-ip>

The detector thresholds the conveyor belt color in HSV, inverts that mask
inside an optional ROI, converts each non-belt connected component into a mask
target, samples median depth inside that mask, and sends selected targets to
``catch_server_1`` with the same camera-point protocol used by the YOLO clients.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from robot_control.linux.catch.hand_eye_mapping import (
    load_camera_to_base_transform,
    map_camera_point_to_base_xyz_m,
)
from robot_control.shared.protocol import CameraPointTask
from robot_control.windows.client.yolo_grasp_send_tcp import (
    DEFAULT_HAND_EYE_RESULT,
    dynamic_arrival_time_s,
    dynamic_skip_reason_from_response,
    make_depth_colormap,
    parse_target_command,
)
from robot_control.windows.client.yolo_seg_grasp_send_tcp import (
    RankedSegTarget,
    SegmentationDetection,
    TerminalInput,
    YoloSegGraspSender,
    clamp_pixel_from_centroid,
    color_for_index,
    deproject_pixel_with_depth,
    mask_centroid,
    segments_to_ranked_targets,
)
from robot_control.windows.vision.d435i_intrinsics import (
    DEFAULT_D435I_PARA,
    PinholeIntrinsics,
    find_d435i_intrinsics,
    format_intrinsics,
)
from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)
from robot_control.windows.vision.detector import Detection


RGB_WINDOW_NAME = "OpenCV belt-color grasp -> catch_server_1"
MASK_WINDOW_NAME = "OpenCV belt/object masks"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


@dataclass(frozen=True)
class GreenConveyorDetectorConfig:
    lower_belt_hsv: tuple[int, int, int] = (35, 40, 40)
    upper_belt_hsv: tuple[int, int, int] = (90, 255, 255)
    reject_highlight: bool = False
    highlight_max_s: int = 70
    highlight_min_v: int = 200
    reject_shadow: bool = False
    shadow_max_v: int = 35
    min_area_px: float = 500.0
    max_area_px: float | None = None
    min_width_px: int = 1
    min_height_px: int = 1
    max_width_px: int | None = None
    max_height_px: int | None = None
    min_fill_ratio: float = 0.0
    max_results: int = 10
    morphology_kernel_size: int = 5
    morphology_iterations: int = 1
    roi_xywh: tuple[int, int, int, int] | None = None
    reject_roi_border_touching: bool = False
    label: str = "opencv_object"


@dataclass(frozen=True)
class DynamicOpenCVGraspCandidate:
    target: RankedSegTarget
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


@dataclass(frozen=True)
class ObjectShortAxisYaw:
    yaw_deg: float
    source: str
    rect_width_px: float
    rect_height_px: float
    rect_angle_deg: float


class OpenCVGraspSender(YoloSegGraspSender):
    """Camera-point sender using the mask-compatible payload fields."""


class GreenConveyorOpenCVDetector:
    """Detect non-belt-color connected components on a conveyor belt."""

    def __init__(self, config: GreenConveyorDetectorConfig | None = None) -> None:
        self.config = config or GreenConveyorDetectorConfig()
        self.last_belt_mask: np.ndarray | None = None
        self.last_green_mask: np.ndarray | None = None
        self.last_object_mask: np.ndarray | None = None
        self.last_roi_mask: np.ndarray | None = None
        self.last_highlight_mask: np.ndarray | None = None
        self.last_shadow_mask: np.ndarray | None = None

    def detect(self, frame_bgr: np.ndarray) -> list[SegmentationDetection]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a BGR image")

        height, width = frame_bgr.shape[:2]
        roi_mask = make_roi_mask((height, width), self.config.roi_xywh)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        belt_mask = hsv_in_range(hsv, self.config.lower_belt_hsv, self.config.upper_belt_hsv)
        belt_mask = cv2.bitwise_and(belt_mask, roi_mask)

        object_mask = cv2.bitwise_not(belt_mask)
        object_mask = cv2.bitwise_and(object_mask, roi_mask)
        highlight_mask = make_highlight_mask(
            hsv,
            roi_mask,
            reject=self.config.reject_highlight,
            max_s=self.config.highlight_max_s,
            min_v=self.config.highlight_min_v,
        )
        shadow_mask = make_shadow_mask(
            hsv,
            roi_mask,
            reject=self.config.reject_shadow,
            max_v=self.config.shadow_max_v,
        )
        if highlight_mask is not None:
            object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(highlight_mask))
        if shadow_mask is not None:
            object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(shadow_mask))
        object_mask = clean_binary_mask(
            object_mask,
            kernel_size=self.config.morphology_kernel_size,
            iterations=self.config.morphology_iterations,
        )

        self.last_roi_mask = roi_mask
        self.last_belt_mask = belt_mask
        self.last_green_mask = belt_mask
        self.last_object_mask = object_mask
        self.last_highlight_mask = highlight_mask
        self.last_shadow_mask = shadow_mask

        contours, _ = cv2.findContours(
            object_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        roi_bounds = roi_bounds_for_frame((height, width), self.config.roi_xywh)

        segments: list[SegmentationDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not contour_area_allowed(area, self.config):
                continue

            x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
            if not bbox_allowed(x, y, bbox_width, bbox_height, self.config):
                continue
            if self.config.reject_roi_border_touching and bbox_touches_roi_border(
                x,
                y,
                bbox_width,
                bbox_height,
                roi_bounds,
            ):
                continue

            fill_ratio = area / float(max(1, bbox_width * bbox_height))
            if fill_ratio < self.config.min_fill_ratio:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
            mask_bool = mask.astype(bool)
            try:
                centroid_u, centroid_v, mask_area_px = mask_centroid(mask_bool)
            except ValueError:
                continue
            u, v = clamp_pixel_from_centroid(
                centroid_u,
                centroid_v,
                width=width,
                height=height,
            )

            detection = Detection(
                u=u,
                v=v,
                x=int(x),
                y=int(y),
                width=int(bbox_width),
                height=int(bbox_height),
                area=area,
                label=self.config.label,
                confidence=None,
                class_id=None,
            )
            segments.append(
                SegmentationDetection(
                    detection=detection,
                    mask=mask_bool,
                    mask_area_px=mask_area_px,
                    centroid_u=centroid_u,
                    centroid_v=centroid_v,
                )
            )

        segments.sort(key=lambda item: (item.mask_area_px, item.detection.area), reverse=True)
        if self.config.max_results > 0:
            segments = segments[: self.config.max_results]
        return segments


def make_roi_mask(
    frame_hw: tuple[int, int],
    roi_xywh: tuple[int, int, int, int] | None,
) -> np.ndarray:
    height, width = int(frame_hw[0]), int(frame_hw[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, y0, x1, y1 = roi_bounds_for_frame(frame_hw, roi_xywh)
    mask[y0:y1, x0:x1] = 255
    return mask


def roi_bounds_for_frame(
    frame_hw: tuple[int, int],
    roi_xywh: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    height, width = int(frame_hw[0]), int(frame_hw[1])
    if roi_xywh is None:
        return 0, 0, width, height
    x, y, roi_width, roi_height = roi_xywh
    x0 = max(0, min(width, int(x)))
    y0 = max(0, min(height, int(y)))
    x1 = max(x0, min(width, x0 + int(roi_width)))
    y1 = max(y0, min(height, y0 + int(roi_height)))
    return x0, y0, x1, y1


def hsv_in_range(
    hsv: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
) -> np.ndarray:
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    if lower[0] <= upper[0]:
        return cv2.inRange(hsv, lower, upper)

    lower_to_max = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
    upper_to_max = np.array([179, upper[1], upper[2]], dtype=np.uint8)
    lower_from_zero = np.array([0, lower[1], lower[2]], dtype=np.uint8)
    upper_from_zero = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
    return cv2.bitwise_or(
        cv2.inRange(hsv, lower_to_max, upper_to_max),
        cv2.inRange(hsv, lower_from_zero, upper_from_zero),
    )


def make_highlight_mask(
    hsv: np.ndarray,
    roi_mask: np.ndarray,
    *,
    reject: bool,
    max_s: int,
    min_v: int,
) -> np.ndarray | None:
    if not reject:
        return None
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = ((saturation <= int(max_s)) & (value >= int(min_v))).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, roi_mask)


def make_shadow_mask(
    hsv: np.ndarray,
    roi_mask: np.ndarray,
    *,
    reject: bool,
    max_v: int,
) -> np.ndarray | None:
    if not reject:
        return None
    value = hsv[:, :, 2]
    mask = (value <= int(max_v)).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, roi_mask)


def clean_binary_mask(mask: np.ndarray, *, kernel_size: int, iterations: int) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    kernel_size = int(kernel_size)
    iterations = max(1, int(iterations))
    if kernel_size <= 1:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def contour_area_allowed(area: float, config: GreenConveyorDetectorConfig) -> bool:
    if area < config.min_area_px:
        return False
    if config.max_area_px is not None and area > config.max_area_px:
        return False
    return True


def bbox_allowed(
    x: int,
    y: int,
    width: int,
    height: int,
    config: GreenConveyorDetectorConfig,
) -> bool:
    del x, y
    if width < config.min_width_px or height < config.min_height_px:
        return False
    if config.max_width_px is not None and width > config.max_width_px:
        return False
    if config.max_height_px is not None and height > config.max_height_px:
        return False
    return True


def bbox_touches_roi_border(
    x: int,
    y: int,
    width: int,
    height: int,
    roi_bounds: tuple[int, int, int, int],
) -> bool:
    x0, y0, x1, y1 = roi_bounds
    return x <= x0 or y <= y0 or x + width >= x1 or y + height >= y1


def normalize_degrees(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _shortest_rect_edge(box_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if box_points.shape != (4, 2):
        raise ValueError("box_points must be a 4x2 array")
    edges: list[tuple[float, np.ndarray, np.ndarray]] = []
    for index in range(4):
        p1 = np.asarray(box_points[index], dtype=float)
        p2 = np.asarray(box_points[(index + 1) % 4], dtype=float)
        edges.append((float(np.linalg.norm(p2 - p1)), p1, p2))
    _, p1, p2 = min(edges, key=lambda item: item[0])
    return p1, p2


def object_short_axis_yaw_from_mask(
    segment: SegmentationDetection,
    depth_frame: Any,
    *,
    depth_m: float,
    intrinsics: PinholeIntrinsics | None,
    camera_to_base: np.ndarray | None,
) -> ObjectShortAxisYaw | None:
    if camera_to_base is None:
        return None
    mask_u8 = segment.mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None

    rect = cv2.minAreaRect(contour)
    (_, _), (rect_width_px, rect_height_px), rect_angle_deg = rect
    if rect_width_px <= 0.0 or rect_height_px <= 0.0:
        return None
    p1_px, p2_px = _shortest_rect_edge(cv2.boxPoints(rect))
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    p1_u = max(0, min(width - 1, int(round(float(p1_px[0])))))
    p1_v = max(0, min(height - 1, int(round(float(p1_px[1])))))
    p2_u = max(0, min(width - 1, int(round(float(p2_px[0])))))
    p2_v = max(0, min(height - 1, int(round(float(p2_px[1])))))

    p1_camera = deproject_pixel_with_depth(
        depth_frame,
        p1_u,
        p1_v,
        depth_m,
        intrinsics=intrinsics,
    )
    p2_camera = deproject_pixel_with_depth(
        depth_frame,
        p2_u,
        p2_v,
        depth_m,
        intrinsics=intrinsics,
    )
    p1_base = map_camera_point_to_base_xyz_m(
        p1_camera.x_m,
        p1_camera.y_m,
        p1_camera.z_m,
        camera_to_base,
    )
    p2_base = map_camera_point_to_base_xyz_m(
        p2_camera.x_m,
        p2_camera.y_m,
        p2_camera.z_m,
        camera_to_base,
    )
    dx_m = p2_base[0] - p1_base[0]
    dy_m = p2_base[1] - p1_base[1]
    if math.hypot(dx_m, dy_m) <= 1e-9:
        return None
    return ObjectShortAxisYaw(
        yaw_deg=normalize_degrees(math.degrees(math.atan2(dy_m, dx_m))),
        source="opencv_min_area_rect_short_edge",
        rect_width_px=float(rect_width_px),
        rect_height_px=float(rect_height_px),
        rect_angle_deg=float(rect_angle_deg),
    )


def attach_object_short_axis_yaw(
    targets: list[RankedSegTarget],
    depth_frame: Any,
    *,
    intrinsics: PinholeIntrinsics | None,
    camera_to_base: np.ndarray | None,
) -> list[RankedSegTarget]:
    updated: list[RankedSegTarget] = []
    for target in targets:
        yaw = object_short_axis_yaw_from_mask(
            target.segment,
            depth_frame,
            depth_m=target.raw_point.depth_m,
            intrinsics=intrinsics,
            camera_to_base=camera_to_base,
        )
        if yaw is None:
            updated.append(target)
            continue
        updated.append(
            replace(
                target,
                object_short_axis_yaw_deg=yaw.yaw_deg,
                object_yaw_source=yaw.source,
                object_rect_width_px=yaw.rect_width_px,
                object_rect_height_px=yaw.rect_height_px,
                object_rect_angle_deg=yaw.rect_angle_deg,
            )
        )
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect non-belt-color objects on a conveyor with OpenCV, use mask "
            "centroid plus mask median depth as the grasp point, and send it to "
            "Linux catch_server_1."
        )
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("static", "dynamic"),
        default="static",
        help=(
            "static keeps the manual rank-send workflow; dynamic automatically "
            "sends one mask target when it enters the configured conveyor window"
        ),
    )
    parser.add_argument("--host", help="Linux VM IP address running catch_server_1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="print target XYZ without sending")
    parser.add_argument(
        "--task",
        choices=("out", "in"),
        dest="task_option",
        help="default task for bare rank commands; terminal can override with 'out N' or 'in N'",
    )
    parser.set_defaults(task="out")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--min-depth-valid-px", type=int, default=20)
    parser.add_argument(
        "--mask-depth-erode-px",
        type=int,
        default=2,
        help="erode object mask by this pixel radius before depth median sampling; 0 disables",
    )
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

    parser.add_argument(
        "--belt-lower-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=None,
        help="lower HSV threshold for the conveyor belt/background color",
    )
    parser.add_argument(
        "--belt-upper-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=None,
        help="upper HSV threshold for the conveyor belt/background color",
    )
    parser.add_argument(
        "--green-lower-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(35, 40, 40),
        help="compatibility alias for --belt-lower-hsv",
    )
    parser.add_argument(
        "--green-upper-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(90, 255, 255),
        help="compatibility alias for --belt-upper-hsv",
    )
    parser.add_argument(
        "--reject-highlight",
        action="store_true",
        help="remove low-saturation high-value glare pixels from the object mask",
    )
    parser.add_argument(
        "--highlight-max-s",
        type=int,
        default=70,
        help="maximum HSV saturation treated as glare when --reject-highlight is used",
    )
    parser.add_argument(
        "--highlight-min-v",
        type=int,
        default=200,
        help="minimum HSV value treated as glare when --reject-highlight is used",
    )
    parser.add_argument(
        "--reject-shadow",
        action="store_true",
        help="remove very dark shadow pixels from the object mask",
    )
    parser.add_argument(
        "--shadow-max-v",
        type=int,
        default=35,
        help="maximum HSV value treated as shadow when --reject-shadow is used",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="optional rectangular conveyor ROI in image pixels",
    )
    parser.add_argument("--min-area-px", type=float, default=500.0)
    parser.add_argument("--max-area-px", type=float)
    parser.add_argument("--min-width-px", type=int, default=1)
    parser.add_argument("--min-height-px", type=int, default=1)
    parser.add_argument("--max-width-px", type=int)
    parser.add_argument("--max-height-px", type=int)
    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.0,
        help="minimum contour area divided by bounding-box area",
    )
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--morphology-kernel-size", type=int, default=5)
    parser.add_argument("--morphology-iterations", type=int, default=1)
    parser.add_argument(
        "--reject-roi-border-touching",
        action="store_true",
        help="ignore components touching the ROI/image border, useful for complete-object gating",
    )
    parser.add_argument(
        "--label",
        default="opencv_object",
        help="label stored in the TCP payload for OpenCV detections",
    )

    parser.add_argument(
        "--camera-up-offset-mm",
        type=float,
        default=0.0,
        help="additional 3D offset in the RealSense image-up direction after deprojection",
    )
    parser.add_argument(
        "--point-z-offset-mm",
        type=float,
        default=0.0,
        help="optional camera-frame Z offset added before sending",
    )
    parser.add_argument(
        "--hand-eye-result",
        nargs="?",
        const=DEFAULT_HAND_EYE_RESULT,
        default=None,
        type=Path,
        help=(
            "optional eye_to_hand hand_eye_result.json camera->base transform, "
            "required for dynamic mode and --base-z-offset-mm"
        ),
    )
    parser.add_argument(
        "--no-hand-eye-result",
        action="store_true",
        help="Do not load a hand-eye transform. This is incompatible with dynamic mode.",
    )
    parser.add_argument(
        "--base-z-offset-mm",
        type=float,
        default=0.0,
        help="optional robot base-frame Z offset before converting back to camera XYZ",
    )

    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--hide-mask", action="store_true")
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
        help="dynamic mode lower base-X bound for accepting a mask centroid",
    )
    parser.add_argument(
        "--dynamic-max-x-mm",
        type=float,
        default=400.0,
        help="dynamic mode upper base-X bound for accepting a mask centroid",
    )
    parser.add_argument(
        "--dynamic-min-y-mm",
        type=float,
        default=-30.0,
        help="dynamic mode lower base-Y bound for accepting a mask centroid",
    )
    parser.add_argument(
        "--dynamic-max-y-mm",
        type=float,
        default=30.0,
        help="dynamic mode upper base-Y bound for accepting a mask centroid",
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
            "compatibility alias: use the detected OpenCV mask centroid base-X "
            "instead of --dynamic-pick-x-mm"
        ),
    )
    parser.add_argument(
        "--dynamic-pick-x-from-mask-centroid",
        dest="dynamic_pick_x_from_bbox_center",
        action="store_true",
        help="use the detected OpenCV mask centroid base-X as dynamic grasp X",
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
            "dynamic mode interval for sending rank-1 mask-centroid probe points "
            "to Linux for camera->base coordinate logging; 0 disables probes"
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    task_option = getattr(args, "task_option", None)
    if task_option is not None:
        args.task = task_option
    normalize_belt_hsv_args(args)
    mode = getattr(args, "mode", "static")
    if getattr(args, "no_hand_eye_result", False):
        args.hand_eye_result = None
    if (
        mode == "dynamic"
        and args.hand_eye_result is None
        and not getattr(args, "no_hand_eye_result", False)
    ):
        args.hand_eye_result = DEFAULT_HAND_EYE_RESULT
    if (
        mode == "static"
        and args.hand_eye_result is None
        and not getattr(args, "no_hand_eye_result", False)
    ):
        args.hand_eye_result = DEFAULT_HAND_EYE_RESULT
    if not args.dry_run and not args.host:
        raise ValueError("--host is required unless --dry-run is used")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.max_depth_m < 0:
        raise ValueError("--max-depth-m must be >= 0")
    if args.min_depth_valid_px < 1:
        raise ValueError("--min-depth-valid-px must be >= 1")
    if args.mask_depth_erode_px < 0:
        raise ValueError("--mask-depth-erode-px must be >= 0")
    validate_hsv_bounds(args.belt_lower_hsv, "--belt-lower-hsv")
    validate_hsv_bounds(args.belt_upper_hsv, "--belt-upper-hsv")
    if args.belt_lower_hsv[1] > args.belt_upper_hsv[1]:
        raise ValueError("belt HSV lower S must be <= upper S")
    if args.belt_lower_hsv[2] > args.belt_upper_hsv[2]:
        raise ValueError("belt HSV lower V must be <= upper V")
    validate_byte_arg(args.highlight_max_s, "--highlight-max-s")
    validate_byte_arg(args.highlight_min_v, "--highlight-min-v")
    validate_byte_arg(args.shadow_max_v, "--shadow-max-v")
    if args.roi is not None:
        x, y, width, height = args.roi
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("--roi must be X Y W H with non-negative X/Y and positive W/H")
        if x + width > args.width or y + height > args.height:
            raise ValueError("--roi must be inside the configured image width/height")
    if args.min_area_px < 0:
        raise ValueError("--min-area-px must be >= 0")
    if args.max_area_px is not None and args.max_area_px < args.min_area_px:
        raise ValueError("--max-area-px must be >= --min-area-px")
    if args.min_width_px < 1 or args.min_height_px < 1:
        raise ValueError("--min-width-px and --min-height-px must be >= 1")
    if args.max_width_px is not None and args.max_width_px < args.min_width_px:
        raise ValueError("--max-width-px must be >= --min-width-px")
    if args.max_height_px is not None and args.max_height_px < args.min_height_px:
        raise ValueError("--max-height-px must be >= --min-height-px")
    if args.min_fill_ratio < 0.0 or args.min_fill_ratio > 1.0:
        raise ValueError("--min-fill-ratio must be in [0, 1]")
    if args.max_results < 1:
        raise ValueError("--max-results must be >= 1")
    if args.morphology_kernel_size < 1:
        raise ValueError("--morphology-kernel-size must be >= 1")
    if args.morphology_iterations < 1:
        raise ValueError("--morphology-iterations must be >= 1")
    if args.print_interval < 0.0:
        raise ValueError("--print-interval must be >= 0")
    if args.base_z_offset_mm != 0.0 and args.hand_eye_result is None:
        raise ValueError("--base-z-offset-mm requires --hand-eye-result")
    if mode == "dynamic":
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


def normalize_belt_hsv_args(args: argparse.Namespace) -> None:
    if getattr(args, "belt_lower_hsv", None) is None:
        args.belt_lower_hsv = args.green_lower_hsv
    if getattr(args, "belt_upper_hsv", None) is None:
        args.belt_upper_hsv = args.green_upper_hsv


def validate_hsv_bounds(values: list[int] | tuple[int, int, int], label: str) -> None:
    if len(values) != 3:
        raise ValueError(f"{label} must contain H S V")
    h, s, v = [int(item) for item in values]
    if h < 0 or h > 179:
        raise ValueError(f"{label} H must be in [0, 179]")
    if s < 0 or s > 255 or v < 0 or v > 255:
        raise ValueError(f"{label} S/V must be in [0, 255]")


def validate_byte_arg(value: int, label: str) -> None:
    if int(value) < 0 or int(value) > 255:
        raise ValueError(f"{label} must be in [0, 255]")


def detector_config_from_args(args: argparse.Namespace) -> GreenConveyorDetectorConfig:
    return GreenConveyorDetectorConfig(
        lower_belt_hsv=tuple(int(item) for item in args.belt_lower_hsv),
        upper_belt_hsv=tuple(int(item) for item in args.belt_upper_hsv),
        reject_highlight=bool(args.reject_highlight),
        highlight_max_s=int(args.highlight_max_s),
        highlight_min_v=int(args.highlight_min_v),
        reject_shadow=bool(args.reject_shadow),
        shadow_max_v=int(args.shadow_max_v),
        min_area_px=float(args.min_area_px),
        max_area_px=args.max_area_px,
        min_width_px=int(args.min_width_px),
        min_height_px=int(args.min_height_px),
        max_width_px=args.max_width_px,
        max_height_px=args.max_height_px,
        min_fill_ratio=float(args.min_fill_ratio),
        max_results=int(args.max_results),
        morphology_kernel_size=int(args.morphology_kernel_size),
        morphology_iterations=int(args.morphology_iterations),
        roi_xywh=tuple(int(item) for item in args.roi) if args.roi is not None else None,
        reject_roi_border_touching=bool(args.reject_roi_border_touching),
        label=str(args.label),
    )


def dynamic_opencv_candidate_from_target(
    target: RankedSegTarget,
    *,
    camera_to_base: np.ndarray,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> DynamicOpenCVGraspCandidate | None:
    center_point = target.raw_point
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

    return DynamicOpenCVGraspCandidate(
        target=target,
        center_point=center_point,
        base_xyz_m=base_xyz_m,
        t_arrive_s=float(t_arrive_s),
        remaining_y_mm=float(remaining_y_mm),
        distance_to_pick_y_mm=abs(center_y_mm - pick_y_mm),
    )


def find_dynamic_opencv_candidates(
    targets: list[RankedSegTarget],
    *,
    camera_to_base: np.ndarray,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> list[DynamicOpenCVGraspCandidate]:
    candidates: list[DynamicOpenCVGraspCandidate] = []
    for target in targets:
        candidate = dynamic_opencv_candidate_from_target(
            target,
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


def find_ranked_target(
    targets: list[RankedSegTarget],
    rank: int,
) -> RankedSegTarget | None:
    for target in targets:
        if target.rank == rank:
            return target
    return None


def format_target_line(target: RankedSegTarget) -> str:
    detection = target.detection
    confidence = "NA" if detection.confidence is None else f"{detection.confidence:.2f}"
    top_pixel = (
        "NA"
        if target.mask_top_u is None or target.mask_top_v is None
        else f"({target.mask_top_u:.1f},{target.mask_top_v:.1f})"
    )
    top_base_z = (
        "NA"
        if target.top_base_xyz_mm is None
        else f"{target.top_base_xyz_mm[2]:.1f}mm"
    )
    yaw_text = (
        "NA"
        if target.object_short_axis_yaw_deg is None
        else f"{target.object_short_axis_yaw_deg:.1f}deg"
    )
    return (
        f"{target.rank}: {detection.label} conf={confidence} "
        f"depth={target.depth_mm:.1f}mm "
        f"mask_area={target.segment.mask_area_px}px "
        f"valid_depth={target.mask_depth_valid_count}px "
        f"bbox=({detection.x},{detection.y},{detection.width},{detection.height}) "
        f"centroid=({target.segment.centroid_u:.1f},{target.segment.centroid_v:.1f}) "
        f"top={top_pixel} top_base_z={top_base_z} short_yaw={yaw_text} "
        f"pixel=({target.send_point.u},{target.send_point.v}) "
        f"send=({target.send_point.x_m:.4f},"
        f"{target.send_point.y_m:.4f},{target.send_point.z_m:.4f})m"
    )


def print_target_table(
    targets: list[RankedSegTarget],
    errors: dict[int, str],
    *,
    show_commands: bool = True,
) -> None:
    if targets:
        print("\nCurrent OpenCV targets ranked by mask median depth:", flush=True)
        for target in targets:
            print("  " + format_target_line(target), flush=True)
        if show_commands:
            print(
                "Type 'out N' or 'in N' to send a target, p to print again, q to quit.",
                flush=True,
            )
    elif errors:
        print(f"\nNo valid OpenCV mask-depth target: {next(iter(errors.values()))}", flush=True)
    else:
        print("\nNo OpenCV non-belt-color target found.", flush=True)


def draw_opencv_targets(
    image: Any,
    segments: list[SegmentationDetection],
    targets: list[RankedSegTarget],
    errors: dict[int, str],
    status: str,
    roi_xywh: tuple[int, int, int, int] | None,
) -> None:
    target_by_segment_id = {id(target.segment): target for target in targets}
    error_indexes = set(errors)

    overlay = image.copy()
    for index, segment in enumerate(segments):
        target = target_by_segment_id.get(id(segment))
        has_error = index in error_indexes and target is None
        color = color_for_index(index) if target is not None else (0, 0, 255) if has_error else (120, 120, 120)
        overlay[segment.mask.astype(bool)] = color
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0.0, dst=image)

    if roi_xywh is not None:
        x, y, width, height = roi_xywh
        cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 0), 1)

    for index, segment in enumerate(segments):
        detection = segment.detection
        target = target_by_segment_id.get(id(segment))
        has_error = index in error_indexes and target is None
        color = color_for_index(index) if target is not None else (0, 0, 255) if has_error else (120, 120, 120)
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
            2,
        )
        marker_pixel = (
            target.send_point.u,
            target.send_point.v,
        ) if target is not None else (
            int(round(segment.centroid_u)),
            int(round(segment.centroid_v)),
        )
        cv2.drawMarker(
            image,
            marker_pixel,
            (0, 255, 255) if target is not None else color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

        if target is not None:
            text = f"{target.rank} {detection.label} {target.depth_mm:.0f}mm"
        elif has_error:
            text = f"{detection.label} no mask depth"
        else:
            text = detection.label
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


def make_mask_debug_image(
    detector: GreenConveyorOpenCVDetector,
    segments: list[SegmentationDetection],
    targets: list[RankedSegTarget],
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if detector.last_belt_mask is not None:
        image[detector.last_belt_mask.astype(bool)] = (0, 90, 0)
    elif detector.last_green_mask is not None:
        image[detector.last_green_mask.astype(bool)] = (0, 90, 0)
    if detector.last_shadow_mask is not None:
        image[detector.last_shadow_mask.astype(bool)] = (80, 40, 0)
    if detector.last_highlight_mask is not None:
        image[detector.last_highlight_mask.astype(bool)] = (255, 255, 255)
    if detector.last_object_mask is not None:
        image[detector.last_object_mask.astype(bool)] = (180, 180, 180)

    target_by_segment_id = {id(target.segment): target for target in targets}
    for index, segment in enumerate(segments):
        color = color_for_index(index)
        image[segment.mask.astype(bool)] = color
        target = target_by_segment_id.get(id(segment))
        if target is not None:
            cv2.putText(
                image,
                str(target.rank),
                (target.send_point.u + 6, target.send_point.v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return image


def send_and_print(
    sender: OpenCVGraspSender,
    target: RankedSegTarget,
    source: str,
    task: CameraPointTask,
) -> dict[str, Any]:
    response = sender.send_target(target, source=source, task=task)
    print(
        f"\nSent {task} OpenCV target rank {target.rank}: {format_target_line(target)}",
        flush=True,
    )
    print(response, flush=True)
    return response


def format_dynamic_opencv_candidate(
    candidate: DynamicOpenCVGraspCandidate,
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


def dynamic_pick_x_mm_for_candidate(
    candidate: DynamicOpenCVGraspCandidate,
    *,
    fixed_pick_x_mm: float,
    use_mask_centroid_x: bool,
) -> float:
    if use_mask_centroid_x:
        return candidate.base_x_mm
    return float(fixed_pick_x_mm)


def send_dynamic_opencv_candidate(
    sender: OpenCVGraspSender,
    candidate: DynamicOpenCVGraspCandidate,
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
        "\nSent dynamic OpenCV "
        f"{task} target: {format_dynamic_opencv_candidate(candidate, pick_y_mm=pick_y_mm)} "
        f"fixed_pick=({pick_x_mm:.1f},{pick_y_mm:.1f})mm",
        flush=True,
    )
    print(response, flush=True)
    return response


def send_dynamic_opencv_probe(
    sender: OpenCVGraspSender,
    target: RankedSegTarget,
    *,
    source: str,
    task: CameraPointTask,
) -> dict[str, Any]:
    response = sender.send_target(
        target,
        source=source,
        task=task,
        grasp_mode="probe",
    )
    print(
        "\nSent dynamic OpenCV mask-centroid probe "
        f"{task} rank {target.rank}: {format_target_line(target)}",
        flush=True,
    )
    print(response, flush=True)
    return response


def prune_suppressed_dynamic_targets(
    suppressed: list[SuppressedDynamicTarget],
    now_s: float,
) -> list[SuppressedDynamicTarget]:
    return [target for target in suppressed if target.expires_at_s > now_s]


def dynamic_opencv_candidate_is_suppressed(
    candidate: DynamicOpenCVGraspCandidate,
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
    candidate: DynamicOpenCVGraspCandidate,
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


def process_terminal_command(
    command: str,
    *,
    targets: list[RankedSegTarget],
    errors: dict[int, str],
    sender: OpenCVGraspSender,
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
    except Exception as exc:  # noqa: BLE001 - manual hardware UI should keep context
        print(f"Send failed for {task} rank {rank}: {exc}", flush=True)
    return True


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    intrinsics: PinholeIntrinsics | None = None
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

    detector = GreenConveyorOpenCVDetector(detector_config_from_args(args))
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    sender = OpenCVGraspSender(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
        task=args.task,
    )
    terminal = TerminalInput()
    if args.mode == "static":
        terminal.start()
    source = (
        "opencv_belt_color_mask_centroid_top_median_depth"
        if args.mode == "static"
        else "opencv_belt_color_dynamic_timed"
    )
    last_print_time = 0.0
    latest_targets: list[RankedSegTarget] = []
    latest_errors: dict[int, str] = {}
    dynamic_sent = False
    suppressed_dynamic_targets: list[SuppressedDynamicTarget] = []
    last_dynamic_probe_time = 0.0
    dynamic_status = (
        "Dynamic OpenCV mode: waiting for mask centroid in "
        f"X=[{args.dynamic_min_x_mm:.1f},{args.dynamic_max_x_mm:.1f}]mm, "
        f"Y=[{args.dynamic_min_y_mm:.1f},{args.dynamic_max_y_mm:.1f}]mm."
    )

    print(
        "OpenCV belt threshold HSV "
        f"lower={tuple(args.belt_lower_hsv)} upper={tuple(args.belt_upper_hsv)} "
        f"roi={tuple(args.roi) if args.roi is not None else 'full-frame'} "
        f"reject_highlight={args.reject_highlight} "
        f"highlight=(S<={args.highlight_max_s}, V>={args.highlight_min_v}) "
        f"reject_shadow={args.reject_shadow} shadow=(V<={args.shadow_max_v})",
        flush=True,
    )
    if args.mode == "static":
        print(
            "Type 'out N' or 'in N' in this terminal to send an OpenCV mask target. "
            "Type p to print targets, q to quit.",
            flush=True,
        )
    else:
        pick_x_text = (
            "mask-centroid"
            if args.dynamic_pick_x_from_bbox_center
            else f"{args.dynamic_pick_x_mm:.1f}mm"
        )
        print(
            "Dynamic OpenCV mode enabled. "
            f"speed={args.conveyor_speed_mm_s:.3f}mm/s "
            f"direction={args.conveyor_direction} "
            f"pick=(x={pick_x_text}, y={args.dynamic_pick_y_mm:.1f}mm). "
            "Press Esc to exit.",
            flush=True,
        )

    if not args.no_display:
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_mask:
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
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

            segments = detector.detect(color_image)
            latest_targets, latest_errors = segments_to_ranked_targets(
                segments,
                depth_frame,
                intrinsics=intrinsics,
                max_depth_m=args.max_depth_m,
                min_depth_valid_px=args.min_depth_valid_px,
                mask_depth_erode_px=args.mask_depth_erode_px,
                camera_up_offset_mm=args.camera_up_offset_mm,
                point_z_offset_mm=args.point_z_offset_mm,
                camera_to_base=camera_to_base,
                base_z_offset_mm=args.base_z_offset_mm,
            )
            latest_targets = attach_object_short_axis_yaw(
                latest_targets,
                depth_frame,
                intrinsics=intrinsics,
                camera_to_base=camera_to_base,
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
                candidates = find_dynamic_opencv_candidates(
                    latest_targets,
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
                    if dynamic_opencv_candidate_is_suppressed(
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
                        use_mask_centroid_x=args.dynamic_pick_x_from_bbox_center,
                    )
                    dynamic_status = (
                        "Dynamic OpenCV candidate "
                        + format_dynamic_opencv_candidate(
                            candidate,
                            pick_y_mm=args.dynamic_pick_y_mm,
                        )
                        + f" pick_x={pick_x_mm:.1f}mm"
                    )
                    try:
                        response = send_dynamic_opencv_candidate(
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
                                "Dynamic OpenCV candidate skipped and remembered: "
                                f"{skip_reason}"
                            )
                            print(dynamic_status, flush=True)
                        elif response.get("type") == "error":
                            dynamic_status = f"Dynamic OpenCV send failed: {response}"
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
                        dynamic_status = f"Dynamic OpenCV send failed: {exc}"
                        print(dynamic_status, flush=True)
                elif suppressed_count > 0:
                    dynamic_status = (
                        "Dynamic OpenCV mode: "
                        f"{suppressed_count} repeated skipped target(s) ignored."
                    )
                else:
                    dynamic_status = (
                        "Dynamic OpenCV mode: no mask centroid inside configured "
                        "base-coordinate window yet."
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
                            response = send_dynamic_opencv_probe(
                                sender,
                                latest_targets[0],
                                source=source,
                                task=args.task,
                            )
                            if response.get("type") == "ack":
                                payload = response.get("payload", {})
                                if isinstance(payload, dict):
                                    base_xyz = payload.get("base_xyz_m")
                                    if isinstance(base_xyz, dict) and all(
                                        key in base_xyz for key in ("x_m", "y_m", "z_m")
                                    ):
                                        dynamic_status = (
                                            "Dynamic OpenCV probe base="
                                            f"({float(base_xyz['x_m']) * 1000.0:.1f},"
                                            f"{float(base_xyz['y_m']) * 1000.0:.1f},"
                                            f"{float(base_xyz['z_m']) * 1000.0:.1f})mm"
                                        )
                            elif response.get("type") == "error":
                                dynamic_status = f"Dynamic OpenCV probe failed: {response}"
                                print(dynamic_status, flush=True)
                        except Exception as exc:  # noqa: BLE001 - hardware UI should keep context
                            dynamic_status = f"Dynamic OpenCV probe failed: {exc}"
                            print(dynamic_status, flush=True)
                        finally:
                            last_dynamic_probe_time = now

            if not args.no_display:
                if args.mode == "static":
                    status = "No OpenCV target found. Type q in terminal or press Esc to exit."
                else:
                    status = dynamic_status
                display_image = color_image.copy()
                draw_opencv_targets(
                    display_image,
                    segments,
                    latest_targets,
                    latest_errors,
                    status,
                    detector.config.roi_xywh,
                )
                cv2.imshow(RGB_WINDOW_NAME, display_image)

                if not args.hide_mask:
                    mask_debug = make_mask_debug_image(
                        detector,
                        segments,
                        latest_targets,
                        color_image.shape,
                    )
                    cv2.imshow(MASK_WINDOW_NAME, mask_debug)

                if not args.hide_depth:
                    depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                    draw_opencv_targets(
                        depth_debug,
                        segments,
                        latest_targets,
                        latest_errors,
                        status,
                        detector.config.roi_xywh,
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

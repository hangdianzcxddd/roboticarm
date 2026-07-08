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
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

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
from robot_control.windows.vision.belt_depth_plane import (
    BeltDepthPlane,
    load_belt_depth_plane,
    mask_depth_percentile,
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


@dataclass
class WebPreviewWriter:
    output_dir: Path
    interval_s: float = 0.2
    jpeg_quality: int = 80
    last_write_s: float = 0.0

    @classmethod
    def from_environment(cls) -> "WebPreviewWriter | None":
        output_dir = os.environ.get("PIPER_WEB_PREVIEW_DIR")
        if not output_dir:
            return None
        try:
            interval_s = float(os.environ.get("PIPER_WEB_PREVIEW_INTERVAL_S", "0.2"))
        except ValueError:
            interval_s = 0.2
        try:
            jpeg_quality = int(os.environ.get("PIPER_WEB_PREVIEW_JPEG_QUALITY", "80"))
        except ValueError:
            jpeg_quality = 80
        return cls(
            output_dir=Path(output_dir),
            interval_s=max(0.05, interval_s),
            jpeg_quality=min(100, max(1, jpeg_quality)),
        )

    def maybe_write(
        self,
        *,
        rgb: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> None:
        now = time.monotonic()
        if now - self.last_write_s < self.interval_s:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write("opencv_rgb.jpg", rgb)
        self._write("opencv_mask.jpg", mask)
        self._write("opencv_depth.jpg", depth)
        self.last_write_s = now

    def _write(self, filename: str, image: np.ndarray | None) -> None:
        if image is None:
            return
        output_path = self.output_dir / filename
        temp_path = output_path.with_suffix(".tmp.jpg")
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        if cv2.imwrite(str(temp_path), image, params):
            temp_path.replace(output_path)


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
    morphology_close_kernel_size: int | None = None
    morphology_open_kernel_size: int | None = None
    morphology_iterations: int = 1
    contour_mask_mode: Literal["raw", "convex-hull"] = "raw"
    roi_xywh: tuple[int, int, int, int] | None = None
    reject_roi_border_touching: bool = False
    entry_edge: Literal["left", "right", "top", "bottom"] | None = None
    entry_clear_margin_px: int = 0
    entry_clear_frames: int = 1
    label: str = "opencv_object"


@dataclass(frozen=True)
class _EntryClearTrack:
    bbox_xywh: tuple[int, int, int, int]
    clear_frames: int


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
class DynamicOpenCVTargetDiagnostic:
    target: RankedSegTarget
    center_point: CameraPoint
    base_xyz_m: tuple[float, float, float]
    t_arrive_s: float
    remaining_y_mm: float
    reject_reasons: tuple[str, ...]
    suppressed: bool = False

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
class OpenCVAsyncSendRequest:
    kind: Literal["dynamic", "probe"]
    started_at_s: float
    task: CameraPointTask
    target_rank: int
    candidate: DynamicOpenCVGraspCandidate | None = None
    pick_x_mm: float | None = None
    pick_y_mm: float | None = None


@dataclass(frozen=True)
class OpenCVAsyncSendCompletion:
    request: OpenCVAsyncSendRequest
    response: dict[str, Any] | None
    error: Exception | None
    elapsed_s: float


@dataclass(frozen=True)
class ObjectShortAxisYaw:
    yaw_deg: float
    source: str
    rect_width_px: float
    rect_height_px: float
    rect_angle_deg: float


class OpenCVGraspSender(YoloSegGraspSender):
    """Camera-point sender using the mask-compatible payload fields."""


class OpenCVAsyncSendWorker:
    """Run one blocking catch_server request without stalling the vision loop."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="opencv-catch-send",
        )
        self._future: Future[dict[str, Any]] | None = None
        self._request: OpenCVAsyncSendRequest | None = None

    def busy(self) -> bool:
        return self._future is not None

    def pending_request(self) -> OpenCVAsyncSendRequest | None:
        return self._request

    def start(
        self,
        request: OpenCVAsyncSendRequest,
        send_call: Callable[[], dict[str, Any]],
    ) -> bool:
        if self._future is not None:
            return False
        self._request = request
        self._future = self._executor.submit(send_call)
        return True

    def poll(self, now_s: float) -> OpenCVAsyncSendCompletion | None:
        if self._future is None or self._request is None or not self._future.done():
            return None

        future = self._future
        request = self._request
        self._future = None
        self._request = None

        try:
            response = future.result()
            error = None
        except Exception as exc:  # noqa: BLE001 - returned to the hardware UI
            response = None
            error = exc
        return OpenCVAsyncSendCompletion(
            request=request,
            response=response,
            error=error,
            elapsed_s=max(0.0, now_s - request.started_at_s),
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


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
        self._entry_clear_tracks: list[_EntryClearTrack] = []

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
            close_kernel_size=self.config.morphology_close_kernel_size,
            open_kernel_size=self.config.morphology_open_kernel_size,
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
        previous_entry_tracks = self._entry_clear_tracks
        next_entry_tracks: list[_EntryClearTrack] = []
        used_entry_track_indexes: set[int] = set()

        segments: list[SegmentationDetection] = []
        for contour in contours:
            shape_contour = contour_for_mask_mode(contour, self.config.contour_mask_mode)
            area = float(cv2.contourArea(shape_contour))
            if not contour_area_allowed(area, self.config):
                continue

            x, y, bbox_width, bbox_height = cv2.boundingRect(shape_contour)
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

            if self.config.entry_edge is not None:
                track_index, previous_track = find_matching_entry_clear_track(
                    (int(x), int(y), int(bbox_width), int(bbox_height)),
                    previous_entry_tracks,
                    used_entry_track_indexes,
                )
                if track_index is not None:
                    used_entry_track_indexes.add(track_index)
                touches_entry = bbox_touches_entry_border(
                    x,
                    y,
                    bbox_width,
                    bbox_height,
                    roi_bounds,
                    edge=self.config.entry_edge,
                    margin_px=self.config.entry_clear_margin_px,
                )
                previous_clear_frames = 0 if previous_track is None else previous_track.clear_frames
                clear_frames = 0 if touches_entry else previous_clear_frames + 1
                next_entry_tracks.append(
                    _EntryClearTrack(
                        bbox_xywh=(int(x), int(y), int(bbox_width), int(bbox_height)),
                        clear_frames=clear_frames,
                    )
                )
                if clear_frames < self.config.entry_clear_frames:
                    continue

            fill_ratio = area / float(max(1, bbox_width * bbox_height))
            if fill_ratio < self.config.min_fill_ratio:
                continue

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [shape_contour], -1, 1, thickness=cv2.FILLED)
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

        if self.config.entry_edge is not None:
            self._entry_clear_tracks = next_entry_tracks
        else:
            self._entry_clear_tracks = []

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


def clean_binary_mask(
    mask: np.ndarray,
    *,
    kernel_size: int,
    iterations: int,
    close_kernel_size: int | None = None,
    open_kernel_size: int | None = None,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    iterations = max(1, int(iterations))
    close_size = int(kernel_size if close_kernel_size is None else close_kernel_size)
    open_size = int(kernel_size if open_kernel_size is None else open_kernel_size)

    cleaned = binary
    if close_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=iterations,
        )
    if open_size > 1:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        )
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=iterations,
        )
    return cleaned


def contour_for_mask_mode(
    contour: np.ndarray,
    mode: Literal["raw", "convex-hull"],
) -> np.ndarray:
    if mode == "raw":
        return contour
    if mode == "convex-hull":
        return cv2.convexHull(contour)
    raise ValueError(f"unsupported contour mask mode: {mode!r}")


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


def bbox_touches_entry_border(
    x: int,
    y: int,
    width: int,
    height: int,
    roi_bounds: tuple[int, int, int, int],
    *,
    edge: Literal["left", "right", "top", "bottom"],
    margin_px: int,
) -> bool:
    x0, y0, x1, y1 = roi_bounds
    margin = max(0, int(margin_px))
    if edge == "left":
        return x <= x0 + margin
    if edge == "right":
        return x + width >= x1 - margin
    if edge == "top":
        return y <= y0 + margin
    if edge == "bottom":
        return y + height >= y1 - margin
    raise ValueError(f"unsupported entry edge: {edge!r}")


def bbox_center_distance_px(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    acx = float(ax) + float(aw) / 2.0
    acy = float(ay) + float(ah) / 2.0
    bcx = float(bx) + float(bw) / 2.0
    bcy = float(by) + float(bh) / 2.0
    return float(math.hypot(acx - bcx, acy - bcy))


def find_matching_entry_clear_track(
    bbox_xywh: tuple[int, int, int, int],
    tracks: list[_EntryClearTrack],
    used_track_indexes: set[int],
) -> tuple[int | None, _EntryClearTrack | None]:
    if not tracks:
        return None, None
    _, _, width, height = bbox_xywh
    max_distance_px = max(20.0, float(max(width, height)))
    best_index: int | None = None
    best_track: _EntryClearTrack | None = None
    best_distance = float("inf")
    for index, track in enumerate(tracks):
        if index in used_track_indexes:
            continue
        distance = bbox_center_distance_px(bbox_xywh, track.bbox_xywh)
        if distance < best_distance:
            best_index = index
            best_track = track
            best_distance = distance
    if best_distance > max_distance_px:
        return None, None
    return best_index, best_track


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


def apply_belt_depth_plane_compensation(
    targets: list[RankedSegTarget],
    depth_frame: Any,
    *,
    plane: BeltDepthPlane | None,
    top_percentile: float,
    max_depth_m: float,
    intrinsics: PinholeIntrinsics | None,
    camera_to_base: np.ndarray | None,
) -> list[RankedSegTarget]:
    if plane is None:
        return targets
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    plane.validate_resolution(width, height)

    updated: list[RankedSegTarget] = []
    for target in targets:
        try:
            top_depth_m, _ = mask_depth_percentile(
                depth_frame,
                target.segment.mask,
                percentile=top_percentile,
                max_depth_m=max_depth_m,
            )
            u, v = clamp_pixel_from_centroid(
                target.segment.centroid_u,
                target.segment.centroid_v,
                width=width,
                height=height,
            )
            belt_depth_m = plane.depth_at(u, v)
            ref_u, ref_v = belt_depth_reference_pixel(plane)
            reference_depth_m = plane.depth_at(ref_u, ref_v)
            compensated_top_depth_m = max(
                0.001,
                top_depth_m - (belt_depth_m - reference_depth_m),
            )
            top_point = deproject_pixel_with_depth(
                depth_frame,
                u,
                v,
                compensated_top_depth_m,
                intrinsics=intrinsics,
            )
            top_base_xyz_mm = None
            if camera_to_base is not None:
                top_base_xyz_m = map_camera_point_to_base_xyz_m(
                    top_point.x_m,
                    top_point.y_m,
                    top_point.z_m,
                    camera_to_base,
                )
                top_base_xyz_mm = tuple(float(value) * 1000.0 for value in top_base_xyz_m)
        except (RuntimeError, ValueError) as exc:
            print(
                "Belt depth plane compensation skipped for "
                f"rank {target.rank}: {exc}",
                flush=True,
            )
            updated.append(target)
            continue

        updated.append(
            replace(
                target,
                top_point=top_point,
                top_base_xyz_mm=top_base_xyz_mm,
                mask_top_u=float(u),
                mask_top_v=float(v),
            )
        )
    return updated


def belt_depth_reference_pixel(plane: BeltDepthPlane) -> tuple[float, float]:
    if plane.roi_xywh is not None:
        x, y, width, height = plane.roi_xywh
        return float(x) + float(width) / 2.0, float(y) + float(height) / 2.0
    return float(plane.width) / 2.0, float(plane.height) / 2.0


def intrinsics_from_depth_frame(depth_frame: Any) -> PinholeIntrinsics:
    rs_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    return PinholeIntrinsics(
        width=int(rs_intrinsics.width),
        height=int(rs_intrinsics.height),
        fx=float(rs_intrinsics.fx),
        fy=float(rs_intrinsics.fy),
        ppx=float(rs_intrinsics.ppx),
        ppy=float(rs_intrinsics.ppy),
        source="realsense_frame",
    )


def project_base_point_to_pixel(
    camera_to_base: np.ndarray,
    base_xyz_m: tuple[float, float, float],
    intrinsics: PinholeIntrinsics,
) -> tuple[int, int, float] | None:
    if camera_to_base.shape != (4, 4):
        raise ValueError("camera_to_base transform must be 4x4")

    base_to_camera = np.linalg.inv(camera_to_base)
    point_camera = base_to_camera @ np.array(
        [base_xyz_m[0], base_xyz_m[1], base_xyz_m[2], 1.0],
        dtype=float,
    )
    x_m, y_m, z_m = (float(value) for value in point_camera[:3])
    if z_m <= 0.0 or not np.isfinite(z_m):
        return None

    u = intrinsics.fx * x_m / z_m + intrinsics.ppx
    v = intrinsics.fy * y_m / z_m + intrinsics.ppy
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    return int(round(u)), int(round(v)), z_m


def project_dynamic_base_roi_to_pixels(
    *,
    camera_to_base: np.ndarray,
    intrinsics: PinholeIntrinsics,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    z_mm: float,
) -> list[tuple[int, int]] | None:
    corners_m = [
        (float(min_x_mm) / 1000.0, float(min_y_mm) / 1000.0, float(z_mm) / 1000.0),
        (float(max_x_mm) / 1000.0, float(min_y_mm) / 1000.0, float(z_mm) / 1000.0),
        (float(max_x_mm) / 1000.0, float(max_y_mm) / 1000.0, float(z_mm) / 1000.0),
        (float(min_x_mm) / 1000.0, float(max_y_mm) / 1000.0, float(z_mm) / 1000.0),
    ]
    pixels: list[tuple[int, int]] = []
    for corner in corners_m:
        projected = project_base_point_to_pixel(camera_to_base, corner, intrinsics)
        if projected is None:
            return None
        u, v, _ = projected
        if abs(u) > 1_000_000 or abs(v) > 1_000_000:
            return None
        pixels.append((u, v))
    return pixels


def dynamic_roi_overlay_z_mm(
    targets: list[RankedSegTarget],
    *,
    camera_to_base: np.ndarray,
    configured_z_mm: float | None,
) -> float:
    if configured_z_mm is not None:
        return float(configured_z_mm)

    z_values_mm: list[float] = []
    for target in targets:
        base_xyz_m = map_camera_point_to_base_xyz_m(
            target.raw_point.x_m,
            target.raw_point.y_m,
            target.raw_point.z_m,
            camera_to_base,
        )
        z_mm = float(base_xyz_m[2]) * 1000.0
        if np.isfinite(z_mm):
            z_values_mm.append(z_mm)
    if z_values_mm:
        return float(np.median(np.asarray(z_values_mm, dtype=float)))
    return 0.0


def draw_dynamic_base_roi(
    image: np.ndarray,
    *,
    camera_to_base: np.ndarray,
    intrinsics: PinholeIntrinsics,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    z_mm: float,
) -> bool:
    pixels = project_dynamic_base_roi_to_pixels(
        camera_to_base=camera_to_base,
        intrinsics=intrinsics,
        min_x_mm=min_x_mm,
        max_x_mm=max_x_mm,
        min_y_mm=min_y_mm,
        max_y_mm=max_y_mm,
        z_mm=z_mm,
    )
    if pixels is None:
        return False

    return draw_dynamic_base_roi_pixels(image, pixels)


def draw_dynamic_base_roi_pixels(
    image: np.ndarray,
    pixels: list[tuple[int, int]],
) -> bool:
    if len(pixels) != 4:
        return False

    points = np.asarray(pixels, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], (255, 0, 255), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.10, image, 0.90, 0.0, dst=image)
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=(255, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    for point in pixels:
        cv2.circle(image, point, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, point, 4, (255, 0, 255), 1, lineType=cv2.LINE_AA)
    return True


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
        "--belt-depth-plane-json",
        type=Path,
        help=(
            "optional JSON from calibrate_belt_depth_plane; flattens tilted belt "
            "depth before reporting detected_top_z"
        ),
    )
    parser.add_argument(
        "--belt-depth-top-percentile",
        type=float,
        default=20.0,
        help=(
            "depth percentile inside the object mask used as the object top "
            "when --belt-depth-plane-json is enabled"
        ),
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
    parser.add_argument(
        "--morphology-close-kernel-size",
        type=int,
        help=(
            "kernel size for closing the object mask before opening; larger "
            "values fill label/texture gaps and join split object regions. "
            "Defaults to --morphology-kernel-size"
        ),
    )
    parser.add_argument(
        "--morphology-open-kernel-size",
        type=int,
        help=(
            "kernel size for opening the object mask after closing; smaller "
            "values remove isolated noise without breaking the object. Defaults "
            "to --morphology-kernel-size"
        ),
    )
    parser.add_argument("--morphology-iterations", type=int, default=1)
    parser.add_argument(
        "--contour-mask-mode",
        choices=("raw", "convex-hull"),
        default="raw",
        help=(
            "raw uses the cleaned external contour; convex-hull fills contour "
            "concavities for more stable size and minAreaRect yaw"
        ),
    )
    parser.add_argument(
        "--reject-roi-border-touching",
        action="store_true",
        help="ignore components touching the ROI/image border, useful for complete-object gating",
    )
    parser.add_argument(
        "--entry-edge",
        choices=("left", "right", "top", "bottom"),
        help=(
            "only gate objects touching this ROI/image entry edge; use this when "
            "objects should be detected after fully entering from one side"
        ),
    )
    parser.add_argument(
        "--entry-clear-margin-px",
        type=int,
        default=0,
        help="pixel margin inside --entry-edge that must be clear before accepting an object",
    )
    parser.add_argument(
        "--entry-clear-frames",
        type=int,
        default=1,
        help="consecutive frames an object must stay clear of --entry-edge before detection",
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
        "--dynamic-roi-z-mm",
        type=float,
        help=(
            "base Z height used to project the dynamic base-X/Y ROI into the "
            "image; default auto-estimates from detected targets and falls back "
            "to 0mm"
        ),
    )
    parser.add_argument(
        "--hide-dynamic-roi",
        action="store_true",
        help="hide the projected dynamic base-coordinate ROI overlay",
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
    parser.add_argument(
        "--dynamic-debug-candidates",
        action="store_true",
        help=(
            "print per-target dynamic candidate diagnostics, including base XYZ, "
            "window checks, arrival time, and skip-memory suppression"
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
    if args.belt_depth_top_percentile < 0.0 or args.belt_depth_top_percentile > 100.0:
        raise ValueError("--belt-depth-top-percentile must be in [0, 100]")
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
    if (
        args.morphology_close_kernel_size is not None
        and args.morphology_close_kernel_size < 1
    ):
        raise ValueError("--morphology-close-kernel-size must be >= 1")
    if (
        args.morphology_open_kernel_size is not None
        and args.morphology_open_kernel_size < 1
    ):
        raise ValueError("--morphology-open-kernel-size must be >= 1")
    if args.morphology_iterations < 1:
        raise ValueError("--morphology-iterations must be >= 1")
    if args.entry_clear_margin_px < 0:
        raise ValueError("--entry-clear-margin-px must be >= 0")
    if args.entry_clear_frames < 1:
        raise ValueError("--entry-clear-frames must be >= 1")
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
        if args.dynamic_roi_z_mm is not None and not math.isfinite(args.dynamic_roi_z_mm):
            raise ValueError("--dynamic-roi-z-mm must be finite")


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
        morphology_close_kernel_size=(
            int(args.morphology_close_kernel_size)
            if args.morphology_close_kernel_size is not None
            else None
        ),
        morphology_open_kernel_size=(
            int(args.morphology_open_kernel_size)
            if args.morphology_open_kernel_size is not None
            else None
        ),
        morphology_iterations=int(args.morphology_iterations),
        contour_mask_mode=args.contour_mask_mode,
        roi_xywh=tuple(int(item) for item in args.roi) if args.roi is not None else None,
        reject_roi_border_touching=bool(args.reject_roi_border_touching),
        entry_edge=args.entry_edge,
        entry_clear_margin_px=int(args.entry_clear_margin_px),
        entry_clear_frames=int(args.entry_clear_frames),
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


def dynamic_opencv_target_diagnostic_from_target(
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
) -> DynamicOpenCVTargetDiagnostic:
    center_point = target.raw_point
    base_xyz_m = map_camera_point_to_base_xyz_m(
        center_point.x_m,
        center_point.y_m,
        center_point.z_m,
        camera_to_base,
    )
    center_x_mm = base_xyz_m[0] * 1000.0
    center_y_mm = base_xyz_m[1] * 1000.0
    reject_reasons: list[str] = []
    if center_x_mm < min_x_mm:
        reject_reasons.append(f"x<{min_x_mm:.1f}")
    elif center_x_mm > max_x_mm:
        reject_reasons.append(f"x>{max_x_mm:.1f}")
    if center_y_mm < min_y_mm:
        reject_reasons.append(f"y<{min_y_mm:.1f}")
    elif center_y_mm > max_y_mm:
        reject_reasons.append(f"y>{max_y_mm:.1f}")

    t_arrive_s, remaining_y_mm = dynamic_arrival_time_s(
        center_y_mm,
        pick_y_mm=pick_y_mm,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        conveyor_direction=conveyor_direction,
    )
    if t_arrive_s < 0.0:
        reject_reasons.append("past-pick-line")

    return DynamicOpenCVTargetDiagnostic(
        target=target,
        center_point=center_point,
        base_xyz_m=base_xyz_m,
        t_arrive_s=float(t_arrive_s),
        remaining_y_mm=float(remaining_y_mm),
        reject_reasons=tuple(reject_reasons),
    )


def dynamic_opencv_diagnostic_is_suppressed(
    diagnostic: DynamicOpenCVTargetDiagnostic,
    suppressed: list[SuppressedDynamicTarget],
    *,
    radius_mm: float,
) -> bool:
    if diagnostic.reject_reasons:
        return False
    radius_sq = float(radius_mm) * float(radius_mm)
    for target in suppressed:
        dx = diagnostic.base_x_mm - target.x_mm
        dy = diagnostic.base_y_mm - target.y_mm
        if dx * dx + dy * dy <= radius_sq:
            return True
    return False


def dynamic_opencv_target_diagnostics(
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
    suppressed: list[SuppressedDynamicTarget],
    skip_radius_mm: float,
) -> list[DynamicOpenCVTargetDiagnostic]:
    diagnostics: list[DynamicOpenCVTargetDiagnostic] = []
    for target in targets:
        diagnostic = dynamic_opencv_target_diagnostic_from_target(
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
        if dynamic_opencv_diagnostic_is_suppressed(
            diagnostic,
            suppressed,
            radius_mm=skip_radius_mm,
        ):
            diagnostic = replace(diagnostic, suppressed=True)
        diagnostics.append(diagnostic)
    return diagnostics


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
    top_camera_z = "NA" if target.top_point is None else f"{target.top_point.z_m * 1000.0:.1f}mm"
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
        f"top={top_pixel} top_camera_z={top_camera_z} top_base_z={top_base_z} short_yaw={yaw_text} "
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


def format_dynamic_opencv_target_diagnostic(
    diagnostic: DynamicOpenCVTargetDiagnostic,
) -> str:
    target = diagnostic.target
    detection = target.detection
    status = (
        "candidate"
        if not diagnostic.reject_reasons
        else "reject=" + ",".join(diagnostic.reject_reasons)
    )
    if diagnostic.suppressed:
        status = f"{status},suppressed"
    return (
        f"rank={target.rank} "
        f"pixel=({diagnostic.center_point.u},{diagnostic.center_point.v}) "
        f"bbox=({detection.x},{detection.y},{detection.width},{detection.height}) "
        "camera="
        f"({diagnostic.center_point.x_m:.4f},"
        f"{diagnostic.center_point.y_m:.4f},"
        f"{diagnostic.center_point.z_m:.4f})m "
        "base="
        f"({diagnostic.base_x_mm:.1f},"
        f"{diagnostic.base_y_mm:.1f},"
        f"{diagnostic.base_z_mm:.1f})mm "
        f"remain_y={diagnostic.remaining_y_mm:.1f}mm "
        f"t_arrive={diagnostic.t_arrive_s:.3f}s "
        f"{status}"
    )


def print_dynamic_opencv_candidate_debug(
    diagnostics: list[DynamicOpenCVTargetDiagnostic],
    *,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: Literal["negative-y", "positive-y"],
) -> None:
    print(
        "\nDynamic OpenCV candidate debug: "
        f"X=[{min_x_mm:.1f},{max_x_mm:.1f}]mm "
        f"Y=[{min_y_mm:.1f},{max_y_mm:.1f}]mm "
        f"pick_y={pick_y_mm:.1f}mm "
        f"direction={conveyor_direction} "
        f"speed={conveyor_speed_mm_s:.1f}mm/s",
        flush=True,
    )
    if not diagnostics:
        print("  no ranked mask-depth targets", flush=True)
        return
    for diagnostic in diagnostics:
        print("  " + format_dynamic_opencv_target_diagnostic(diagnostic), flush=True)


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


def format_pending_async_send(
    request: OpenCVAsyncSendRequest,
    *,
    now_s: float,
) -> str:
    elapsed_s = max(0.0, now_s - request.started_at_s)
    if request.kind == "dynamic" and request.candidate is not None:
        pick_y_mm = request.pick_y_mm
        candidate = request.candidate
        pick_x_text = (
            "NA" if request.pick_x_mm is None else f"{request.pick_x_mm:.1f}mm"
        )
        return (
            "Dynamic OpenCV send in progress "
            f"elapsed={elapsed_s:.3f}s "
            f"{format_dynamic_opencv_candidate(candidate, pick_y_mm=pick_y_mm)} "
            f"pick_x={pick_x_text}"
        )
    return (
        "Dynamic OpenCV probe in progress "
        f"rank={request.target_rank} elapsed={elapsed_s:.3f}s"
    )


def start_async_dynamic_opencv_candidate(
    worker: OpenCVAsyncSendWorker,
    sender: OpenCVGraspSender,
    candidate: DynamicOpenCVGraspCandidate,
    *,
    source: str,
    task: CameraPointTask,
    conveyor_speed_mm_s: float,
    pick_x_mm: float,
    pick_y_mm: float,
    now_s: float,
) -> bool:
    request = OpenCVAsyncSendRequest(
        kind="dynamic",
        started_at_s=now_s,
        task=task,
        target_rank=candidate.target.rank,
        candidate=candidate,
        pick_x_mm=pick_x_mm,
        pick_y_mm=pick_y_mm,
    )
    return worker.start(
        request,
        partial(
            send_dynamic_opencv_candidate,
            sender,
            candidate,
            source=source,
            task=task,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            pick_x_mm=pick_x_mm,
            pick_y_mm=pick_y_mm,
        ),
    )


def start_async_dynamic_opencv_probe(
    worker: OpenCVAsyncSendWorker,
    sender: OpenCVGraspSender,
    target: RankedSegTarget,
    *,
    source: str,
    task: CameraPointTask,
    now_s: float,
) -> bool:
    request = OpenCVAsyncSendRequest(
        kind="probe",
        started_at_s=now_s,
        task=task,
        target_rank=target.rank,
    )
    return worker.start(
        request,
        partial(
            send_dynamic_opencv_probe,
            sender,
            target,
            source=source,
            task=task,
        ),
    )


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

    belt_depth_plane = None
    if args.belt_depth_plane_json is not None:
        belt_depth_plane = load_belt_depth_plane(args.belt_depth_plane_json)
        belt_depth_plane.validate_resolution(args.width, args.height)
        residual = belt_depth_plane.residuals
        residual_text = (
            "NA"
            if residual is None
            else f"rms={residual.rms_mm:.2f}mm max={residual.max_abs_mm:.2f}mm"
        )
        print(
            "Using belt depth plane "
            f"{args.belt_depth_plane_json}: "
            f"z={belt_depth_plane.a:.9f}*u+"
            f"{belt_depth_plane.b:.9f}*v+{belt_depth_plane.c:.6f} "
            f"samples={belt_depth_plane.sample_count} residuals({residual_text}) "
            f"top_percentile={args.belt_depth_top_percentile:.1f}",
            flush=True,
        )

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
    async_sender = OpenCVAsyncSendWorker()
    suppressed_dynamic_targets: list[SuppressedDynamicTarget] = []
    last_dynamic_probe_time = 0.0
    last_dynamic_debug_time = 0.0
    dynamic_status = (
        "Dynamic OpenCV mode: waiting for mask centroid in "
        f"X=[{args.dynamic_min_x_mm:.1f},{args.dynamic_max_x_mm:.1f}]mm, "
        f"Y=[{args.dynamic_min_y_mm:.1f},{args.dynamic_max_y_mm:.1f}]mm."
    )

    print(
        "OpenCV belt threshold HSV "
        f"lower={tuple(args.belt_lower_hsv)} upper={tuple(args.belt_upper_hsv)} "
        f"roi={tuple(args.roi) if args.roi is not None else 'full-frame'} "
        f"morphology=(close={args.morphology_close_kernel_size or args.morphology_kernel_size}, "
        f"open={args.morphology_open_kernel_size or args.morphology_kernel_size}, "
        f"iterations={args.morphology_iterations}) "
        f"contour_mask_mode={args.contour_mask_mode} "
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
            f"physical_roi=X=[{args.dynamic_min_x_mm:.1f},{args.dynamic_max_x_mm:.1f}]mm "
            f"Y=[{args.dynamic_min_y_mm:.1f},{args.dynamic_max_y_mm:.1f}]mm "
            "Press Esc to exit.",
            flush=True,
        )

    if not args.no_display:
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_mask:
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_depth:
            cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    web_preview = WebPreviewWriter.from_environment()
    if web_preview is not None:
        print(f"Web preview enabled: {web_preview.output_dir}", flush=True)

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
            latest_targets = apply_belt_depth_plane_compensation(
                latest_targets,
                depth_frame,
                plane=belt_depth_plane,
                top_percentile=args.belt_depth_top_percentile,
                max_depth_m=args.max_depth_m,
                intrinsics=intrinsics,
                camera_to_base=camera_to_base,
            )
            latest_targets = attach_object_short_axis_yaw(
                latest_targets,
                depth_frame,
                intrinsics=intrinsics,
                camera_to_base=camera_to_base,
            )

            now = time.monotonic()
            async_completion = async_sender.poll(now)
            if async_completion is not None:
                request = async_completion.request
                if async_completion.error is not None:
                    dynamic_status = (
                        "Dynamic OpenCV "
                        f"{request.kind} failed after {async_completion.elapsed_s:.3f}s: "
                        f"{async_completion.error}"
                    )
                    print(dynamic_status, flush=True)
                elif request.kind == "dynamic":
                    response = async_completion.response or {}
                    skip_reason = dynamic_skip_reason_from_response(response)
                    if skip_reason is not None:
                        if request.candidate is not None:
                            suppressed_dynamic_targets = remember_suppressed_dynamic_target(
                                suppressed_dynamic_targets,
                                request.candidate,
                                now_s=now,
                                memory_s=args.dynamic_skip_memory_s,
                            )
                        dynamic_status = (
                            "Dynamic OpenCV candidate skipped and remembered "
                            f"after {async_completion.elapsed_s:.3f}s: {skip_reason}"
                        )
                        print(dynamic_status, flush=True)
                    elif response.get("type") == "error":
                        dynamic_status = (
                            "Dynamic OpenCV send failed "
                            f"after {async_completion.elapsed_s:.3f}s: {response}"
                        )
                        print(dynamic_status, flush=True)
                    else:
                        dynamic_sent = True
                        if args.dynamic_keep_running and request.candidate is not None:
                            suppressed_dynamic_targets = remember_suppressed_dynamic_target(
                                suppressed_dynamic_targets,
                                request.candidate,
                                now_s=now,
                                memory_s=args.dynamic_skip_memory_s,
                            )
                        dynamic_status = (
                            "Dynamic OpenCV send completed "
                            f"after {async_completion.elapsed_s:.3f}s"
                        )
                        print(dynamic_status, flush=True)
                        if not args.dynamic_keep_running:
                            running = False
                else:
                    response = async_completion.response or {}
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
                                    f"{float(base_xyz['z_m']) * 1000.0:.1f})mm "
                                    f"after {async_completion.elapsed_s:.3f}s"
                                )
                    elif response.get("type") == "error":
                        dynamic_status = (
                            "Dynamic OpenCV probe failed "
                            f"after {async_completion.elapsed_s:.3f}s: {response}"
                        )
                        print(dynamic_status, flush=True)

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
                if async_sender.busy():
                    pending_request = async_sender.pending_request()
                    if pending_request is not None:
                        dynamic_status = format_pending_async_send(
                            pending_request,
                            now_s=now,
                        )
                else:
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
                    if (
                        args.dynamic_debug_candidates
                        and (
                            args.print_interval == 0.0
                            or now - last_dynamic_debug_time >= args.print_interval
                        )
                    ):
                        diagnostics = dynamic_opencv_target_diagnostics(
                            latest_targets,
                            camera_to_base=camera_to_base,
                            min_x_mm=args.dynamic_min_x_mm,
                            max_x_mm=args.dynamic_max_x_mm,
                            min_y_mm=args.dynamic_min_y_mm,
                            max_y_mm=args.dynamic_max_y_mm,
                            pick_y_mm=args.dynamic_pick_y_mm,
                            conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                            conveyor_direction=args.conveyor_direction,
                            suppressed=suppressed_dynamic_targets,
                            skip_radius_mm=args.dynamic_skip_radius_mm,
                        )
                        print_dynamic_opencv_candidate_debug(
                            diagnostics,
                            min_x_mm=args.dynamic_min_x_mm,
                            max_x_mm=args.dynamic_max_x_mm,
                            min_y_mm=args.dynamic_min_y_mm,
                            max_y_mm=args.dynamic_max_y_mm,
                            pick_y_mm=args.dynamic_pick_y_mm,
                            conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                            conveyor_direction=args.conveyor_direction,
                        )
                        last_dynamic_debug_time = now
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
                        started = start_async_dynamic_opencv_candidate(
                            async_sender,
                            sender,
                            candidate,
                            source=source,
                            task=args.task,
                            conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                            pick_x_mm=pick_x_mm,
                            pick_y_mm=args.dynamic_pick_y_mm,
                            now_s=now,
                        )
                        if started:
                            dynamic_status = format_pending_async_send(
                                OpenCVAsyncSendRequest(
                                    kind="dynamic",
                                    started_at_s=now,
                                    task=args.task,
                                    target_rank=candidate.target.rank,
                                    candidate=candidate,
                                    pick_x_mm=pick_x_mm,
                                    pick_y_mm=args.dynamic_pick_y_mm,
                                ),
                                now_s=now,
                            )
                        else:
                            pending_request = async_sender.pending_request()
                            dynamic_status = (
                                format_pending_async_send(pending_request, now_s=now)
                                if pending_request is not None
                                else "Dynamic OpenCV send already in progress."
                            )
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
                                or now - last_dynamic_probe_time
                                >= args.dynamic_probe_interval_s
                            )
                        ):
                            started = start_async_dynamic_opencv_probe(
                                async_sender,
                                sender,
                                latest_targets[0],
                                source=source,
                                task=args.task,
                                now_s=now,
                            )
                            if started:
                                last_dynamic_probe_time = now
                                dynamic_status = format_pending_async_send(
                                    OpenCVAsyncSendRequest(
                                        kind="probe",
                                        started_at_s=now,
                                        task=args.task,
                                        target_rank=latest_targets[0].rank,
                                    ),
                                    now_s=now,
                                )
                            else:
                                pending_request = async_sender.pending_request()
                                if pending_request is not None:
                                    dynamic_status = format_pending_async_send(
                                        pending_request,
                                        now_s=now,
                                    )

            needs_preview_images = not args.no_display or web_preview is not None
            display_image: np.ndarray | None = None
            mask_debug: np.ndarray | None = None
            depth_debug: np.ndarray | None = None
            if needs_preview_images:
                if args.mode == "static":
                    status = "No OpenCV target found. Type q in terminal or press Esc to exit."
                else:
                    status = dynamic_status
                display_image = color_image.copy()
                dynamic_roi_pixels: list[tuple[int, int]] | None = None
                if (
                    args.mode == "dynamic"
                    and not args.hide_dynamic_roi
                    and camera_to_base is not None
                ):
                    current_intrinsics = intrinsics or intrinsics_from_depth_frame(depth_frame)
                    dynamic_roi_z_mm = dynamic_roi_overlay_z_mm(
                        latest_targets,
                        camera_to_base=camera_to_base,
                        configured_z_mm=args.dynamic_roi_z_mm,
                    )
                    dynamic_roi_pixels = project_dynamic_base_roi_to_pixels(
                        camera_to_base=camera_to_base,
                        intrinsics=current_intrinsics,
                        min_x_mm=args.dynamic_min_x_mm,
                        max_x_mm=args.dynamic_max_x_mm,
                        min_y_mm=args.dynamic_min_y_mm,
                        max_y_mm=args.dynamic_max_y_mm,
                        z_mm=dynamic_roi_z_mm,
                    )
                    status = f"{status} | " + (
                        f"physical ROI z={dynamic_roi_z_mm:.1f}mm"
                        if dynamic_roi_pixels is not None
                        else "physical ROI outside camera view"
                    )
                draw_opencv_targets(
                    display_image,
                    segments,
                    latest_targets,
                    latest_errors,
                    status,
                    detector.config.roi_xywh,
                )
                if dynamic_roi_pixels is not None:
                    draw_dynamic_base_roi_pixels(display_image, dynamic_roi_pixels)

                if not args.hide_mask or web_preview is not None:
                    mask_debug = make_mask_debug_image(
                        detector,
                        segments,
                        latest_targets,
                        color_image.shape,
                    )

                if not args.hide_depth or web_preview is not None:
                    depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                    draw_opencv_targets(
                        depth_debug,
                        segments,
                        latest_targets,
                        latest_errors,
                        status,
                        detector.config.roi_xywh,
                    )
                    if dynamic_roi_pixels is not None:
                        draw_dynamic_base_roi_pixels(depth_debug, dynamic_roi_pixels)

            if web_preview is not None:
                web_preview.maybe_write(
                    rgb=display_image,
                    mask=mask_debug,
                    depth=depth_debug,
                )

            if not args.no_display:
                if display_image is not None:
                    cv2.imshow(RGB_WINDOW_NAME, display_image)
                if not args.hide_mask and mask_debug is not None:
                    cv2.imshow(MASK_WINDOW_NAME, mask_debug)
                if not args.hide_depth and depth_debug is not None:
                    cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False
    finally:
        async_sender.shutdown()
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

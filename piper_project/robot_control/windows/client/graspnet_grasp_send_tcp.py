"""Point-cloud + GraspNet visual grasp client.

This client keeps the ROI and entry-edge gating style from
``opencv_grasp_send_tcp`` but replaces color-threshold object detection with
depth/point-cloud foreground extraction and sends 6D grasp candidates to
``catch_server_graspnet``.
"""

from __future__ import annotations

import argparse
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from robot_control.linux.catch.hand_eye_mapping import (
    load_camera_to_base_transform,
    map_camera_point_to_base_xyz_m,
)
from robot_control.shared.paths import PROJECT_DIR
from robot_control.shared.protocol import (
    CameraPointCommand,
    CameraPointTask,
    GraspCandidate,
    send_camera_point,
)
from robot_control.windows.client.opencv_grasp_send_tcp import (
    GreenConveyorDetectorConfig,
    bbox_allowed,
    bbox_touches_entry_border,
    bbox_touches_roi_border,
    contour_area_allowed,
    contour_for_mask_mode,
    find_matching_entry_clear_track,
    make_roi_mask,
    roi_bounds_for_frame,
)
from robot_control.windows.client.yolo_grasp_send_tcp import (
    dynamic_arrival_time_s,
    dynamic_skip_reason_from_response,
    make_depth_colormap,
    parse_target_command,
)
from robot_control.windows.client.yolo_seg_grasp_send_tcp import (
    SegmentationDetection,
    TerminalInput,
    clamp_pixel_from_centroid,
    deproject_pixel_with_depth,
    depth_array_meters,
    mask_centroid,
)
from robot_control.windows.vision.belt_depth_plane import (
    BeltDepthPlane,
    load_belt_depth_plane,
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
    wait_for_aligned_rgb_depth,
)
from robot_control.windows.vision.detector import Detection


DEFAULT_HAND_EYE_RESULT = (
    PROJECT_DIR
    / "robot_control"
    / "linux"
    / "hand_eye"
    / "eye_to_hand_result_auto.json"
)
RGB_WINDOW_NAME = "GraspNet point-cloud grasp -> catch_server_graspnet"
MASK_WINDOW_NAME = "GraspNet depth foreground masks"
DEPTH_WINDOW_NAME = "Aligned depth valid map"
POINT_CLOUD_WINDOW_NAME = "GraspNet object point cloud"
POINT_CLOUD_TARGET_COLORS = np.asarray(
    [
        (1.0, 0.82, 0.10),
        (0.10, 0.72, 1.0),
        (0.15, 0.95, 0.45),
        (1.0, 0.35, 0.25),
        (0.72, 0.45, 1.0),
        (0.95, 0.95, 0.95),
    ],
    dtype=np.float64,
)
COORDINATE_AXIS_NAME = "camera XYZ axes"
EntryEdge = Literal["left", "right", "top", "bottom"]


@dataclass(frozen=True)
class DepthForegroundDetectorConfig:
    min_height_above_belt_mm: float = 8.0
    max_height_above_belt_mm: float = 200.0
    min_area_px: float = 500.0
    max_area_px: float | None = None
    min_width_px: int = 1
    min_height_px: int = 1
    max_width_px: int | None = None
    max_height_px: int | None = None
    min_fill_ratio: float = 0.0
    max_results: int = 10
    morphology_kernel_size: int = 3
    contour_mask_mode: Literal["raw", "convex-hull"] = "raw"
    roi_xywh: tuple[int, int, int, int] | None = None
    reject_roi_border_touching: bool = False
    entry_edge: EntryEdge | None = None
    entry_clear_margin_px: int = 0
    entry_clear_frames: int = 1
    label: str = "graspnet_object"


@dataclass(frozen=True)
class _EntryClearTrack:
    bbox_xywh: tuple[int, int, int, int]
    clear_frames: int


@dataclass(frozen=True)
class GraspNetTarget:
    rank: int
    segment: SegmentationDetection
    raw_point: CameraPoint
    send_point: CameraPoint
    base_xyz_m: tuple[float, float, float] | None
    point_cloud_camera_m: np.ndarray
    grasp_candidates: tuple[GraspCandidate, ...]
    top_point: CameraPoint | None = None
    top_base_xyz_m: tuple[float, float, float] | None = None

    @property
    def detection(self) -> Detection:
        return self.segment.detection

    @property
    def depth_mm(self) -> float:
        return self.raw_point.depth_m * 1000.0


@dataclass(frozen=True)
class DynamicGraspNetCandidate:
    target: GraspNetTarget
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


class DepthForegroundDetector:
    """Detect raised foreground objects using a calibrated belt depth plane."""

    def __init__(
        self,
        belt_plane: BeltDepthPlane,
        config: DepthForegroundDetectorConfig | None = None,
    ) -> None:
        self.belt_plane = belt_plane
        self.config = config or DepthForegroundDetectorConfig()
        self.last_foreground_mask: np.ndarray | None = None
        self.last_roi_mask: np.ndarray | None = None
        self._entry_clear_tracks: list[_EntryClearTrack] = []

    def detect(self, depth_frame: Any) -> list[SegmentationDetection]:
        depth_m = depth_array_meters(depth_frame)
        height, width = depth_m.shape[:2]
        self.belt_plane.validate_resolution(width, height)
        roi_mask = make_roi_mask((height, width), self.config.roi_xywh)

        u_grid, v_grid = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
        )
        belt_depth_m = (
            self.belt_plane.a * u_grid
            + self.belt_plane.b * v_grid
            + self.belt_plane.c
        )
        height_above_m = belt_depth_m - depth_m
        foreground = (
            (roi_mask > 0)
            & np.isfinite(depth_m)
            & (depth_m > 0.0)
            & (height_above_m >= self.config.min_height_above_belt_mm / 1000.0)
            & (height_above_m <= self.config.max_height_above_belt_mm / 1000.0)
        )

        mask = foreground.astype(np.uint8) * 255
        kernel_size = max(1, int(self.config.morphology_kernel_size))
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.last_foreground_mask = mask
        self.last_roi_mask = roi_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_bounds = roi_bounds_for_frame((height, width), self.config.roi_xywh)
        previous_tracks = self._entry_clear_tracks
        next_tracks: list[_EntryClearTrack] = []
        used_indexes: set[int] = set()
        accepted_contours: list[np.ndarray] = []
        for contour in contours:
            shape_contour = contour_for_mask_mode(contour, self.config.contour_mask_mode)
            x, y, bbox_width, bbox_height = cv2.boundingRect(shape_contour)
            if self.config.entry_edge is None:
                accepted_contours.append(contour)
                continue
            track_index, previous_track = find_matching_entry_clear_track(
                (int(x), int(y), int(bbox_width), int(bbox_height)),
                previous_tracks,
                used_indexes,
            )
            if track_index is not None:
                used_indexes.add(track_index)
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
            next_tracks.append(
                _EntryClearTrack(
                    bbox_xywh=(int(x), int(y), int(bbox_width), int(bbox_height)),
                    clear_frames=clear_frames,
                )
            )
            if clear_frames >= self.config.entry_clear_frames:
                accepted_contours.append(contour)
        self._entry_clear_tracks = next_tracks if self.config.entry_edge is not None else []
        return _segments_from_contours(mask, accepted_contours, self.config, roi_bounds)


def _segments_from_contours(
    mask: np.ndarray,
    contours: list[np.ndarray],
    config: DepthForegroundDetectorConfig,
    roi_bounds: tuple[int, int, int, int],
) -> list[SegmentationDetection]:
    height, width = mask.shape[:2]
    segments: list[SegmentationDetection] = []
    compatible_config = GreenConveyorDetectorConfig(
        min_area_px=config.min_area_px,
        max_area_px=config.max_area_px,
        min_width_px=config.min_width_px,
        min_height_px=config.min_height_px,
        max_width_px=config.max_width_px,
        max_height_px=config.max_height_px,
        min_fill_ratio=config.min_fill_ratio,
        contour_mask_mode=config.contour_mask_mode,
        roi_xywh=config.roi_xywh,
        reject_roi_border_touching=config.reject_roi_border_touching,
        entry_edge=config.entry_edge,
        entry_clear_margin_px=config.entry_clear_margin_px,
        entry_clear_frames=config.entry_clear_frames,
        label=config.label,
    )
    for contour in contours:
        shape_contour = contour_for_mask_mode(contour, config.contour_mask_mode)
        area = float(cv2.contourArea(shape_contour))
        if not contour_area_allowed(area, compatible_config):
            continue
        x, y, bbox_width, bbox_height = cv2.boundingRect(shape_contour)
        if not bbox_allowed(x, y, bbox_width, bbox_height, compatible_config):
            continue
        if config.reject_roi_border_touching and bbox_touches_roi_border(
            x,
            y,
            bbox_width,
            bbox_height,
            roi_bounds,
        ):
            continue
        fill_ratio = area / float(max(1, bbox_width * bbox_height))
        if fill_ratio < config.min_fill_ratio:
            continue

        segment_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(segment_mask, [shape_contour], -1, 1, thickness=cv2.FILLED)
        mask_bool = segment_mask.astype(bool)
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
            label=config.label,
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
    if config.max_results > 0:
        segments = segments[: config.max_results]
    return segments


class GraspNetBackend:
    """Small adapter around an optional local GraspNet/AnyGrasp backend."""

    def __init__(
        self,
        module_name: str,
        class_name: str,
        checkpoint: Path | None,
        device: str | None,
        max_candidates: int,
    ) -> None:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import GraspNet backend module {module_name!r}. "
                "Install the GraspNet/AnyGrasp package and weights first."
            ) from exc
        try:
            backend_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"GraspNet backend module {module_name!r} has no class {class_name!r}"
            ) from exc
        kwargs: dict[str, Any] = {}
        if checkpoint is not None:
            kwargs["checkpoint"] = str(checkpoint)
        if device is not None:
            kwargs["device"] = device
        try:
            self.backend = backend_cls(**kwargs)
        except TypeError:
            self.backend = backend_cls()
        self.max_candidates = int(max_candidates)

    def predict(self, point_cloud_camera_m: np.ndarray) -> tuple[GraspCandidate, ...]:
        if point_cloud_camera_m.size == 0:
            return ()
        if not hasattr(self.backend, "predict"):
            raise RuntimeError("GraspNet backend must provide predict(points_xyz_m)")
        raw = self.backend.predict(point_cloud_camera_m)
        if raw is None:
            return ()
        try:
            limited = raw[: self.max_candidates]
        except TypeError:
            limited = list(raw)[: self.max_candidates]
        return tuple(
            raw_candidate_to_protocol(candidate, index)
            for index, candidate in enumerate(limited)
        )


def raw_candidate_to_protocol(candidate: Any, index: int) -> GraspCandidate:
    if isinstance(candidate, GraspCandidate):
        return candidate
    if isinstance(candidate, dict):
        position = candidate.get("position_camera_m", candidate.get("translation"))
        rotation = candidate.get("rotation_camera", candidate.get("rotation_matrix"))
        width_m = candidate.get("width_m", candidate.get("width"))
        score = candidate.get("score")
        approach_depth_m = candidate.get("approach_depth_m", candidate.get("depth"))
        source = str(candidate.get("source", "graspnet"))
    else:
        position = getattr(candidate, "position_camera_m", None)
        if position is None:
            position = getattr(candidate, "translation", None)
        rotation = getattr(candidate, "rotation_camera", None)
        if rotation is None:
            rotation = getattr(candidate, "rotation_matrix", None)
        width_m = getattr(candidate, "width_m", getattr(candidate, "width", None))
        score = getattr(candidate, "score", None)
        approach_depth_m = getattr(
            candidate,
            "approach_depth_m",
            getattr(candidate, "depth", None),
        )
        source = str(getattr(candidate, "source", "graspnet"))
    if position is None or rotation is None:
        raise ValueError("GraspNet candidate must provide position and rotation")
    return GraspCandidate(
        position_camera_m=tuple(float(value) for value in np.asarray(position).reshape(3)),
        rotation_camera=tuple(
            tuple(float(item) for item in row)
            for row in np.asarray(rotation, dtype=float).reshape(3, 3)
        ),
        width_m=None if width_m is None else float(width_m),
        score=None if score is None else float(score),
        approach_depth_m=None if approach_depth_m is None else float(approach_depth_m),
        source=source,
        index=index,
    )


def point_cloud_from_mask(
    depth_frame: Any,
    mask: np.ndarray,
    *,
    intrinsics: PinholeIntrinsics,
    voxel_size_m: float,
    max_points: int,
) -> np.ndarray:
    depth_m = depth_array_meters(depth_frame)
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError("mask and depth frame resolutions do not match")
    rows, cols = np.nonzero(mask.astype(bool))
    valid_depth = depth_m[rows, cols]
    valid = np.isfinite(valid_depth) & (valid_depth > 0.0)
    rows = rows[valid]
    cols = cols[valid]
    z = valid_depth[valid].astype(np.float32)
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    x = (cols.astype(np.float32) - intrinsics.ppx) / intrinsics.fx * z
    y = (rows.astype(np.float32) - intrinsics.ppy) / intrinsics.fy * z
    points = np.column_stack((x, y, z)).astype(np.float32)
    if voxel_size_m > 0.0 and points.shape[0] > 0:
        points = voxel_downsample(points, voxel_size_m)
    if max_points > 0 and points.shape[0] > max_points:
        indexes = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int32)
        points = points[indexes]
    return points


def top_point_from_mask(
    depth_frame: Any,
    mask: np.ndarray,
    *,
    intrinsics: PinholeIntrinsics,
    camera_to_base: np.ndarray | None,
) -> tuple[CameraPoint, tuple[float, float, float] | None]:
    depth_m = depth_array_meters(depth_frame)
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError("mask and depth frame resolutions do not match")
    rows, cols = np.nonzero(mask.astype(bool))
    valid_depth = depth_m[rows, cols]
    valid = np.isfinite(valid_depth) & (valid_depth > 0.0)
    rows = rows[valid]
    cols = cols[valid]
    z = valid_depth[valid].astype(np.float64)
    if z.size == 0:
        raise ValueError("target mask contains no valid depth pixels")

    x = (cols.astype(np.float64) - intrinsics.ppx) / intrinsics.fx * z
    y = (rows.astype(np.float64) - intrinsics.ppy) / intrinsics.fy * z
    points = np.column_stack((x, y, z))
    finite_points = np.isfinite(points).all(axis=1)
    if not np.any(finite_points):
        raise ValueError("target mask contains no finite camera points")
    rows = rows[finite_points]
    cols = cols[finite_points]
    points = points[finite_points]

    top_base_xyz_m = None
    if camera_to_base is None:
        selected_index = int(np.argmin(points[:, 2]))
    else:
        if camera_to_base.shape != (4, 4):
            raise ValueError("camera_to_base transform must be 4x4")
        base_points = points @ camera_to_base[:3, :3].T + camera_to_base[:3, 3]
        finite_base = np.isfinite(base_points).all(axis=1)
        if not np.any(finite_base):
            raise ValueError("target mask maps to no finite base points")
        candidate_indexes = np.nonzero(finite_base)[0]
        selected_index = int(
            candidate_indexes[np.argmax(base_points[finite_base, 2])]
        )
        top_base_xyz_m = tuple(float(value) for value in base_points[selected_index])

    selected = points[selected_index]
    top_point = CameraPoint(
        u=int(cols[selected_index]),
        v=int(rows[selected_index]),
        depth_m=float(selected[2]),
        x_m=float(selected[0]),
        y_m=float(selected[1]),
        z_m=float(selected[2]),
    )
    return top_point, top_base_xyz_m


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    _, unique_indexes = np.unique(keys, axis=0, return_index=True)
    unique_indexes.sort()
    return points[unique_indexes]


def colored_point_cloud_from_targets(
    targets: list[GraspNetTarget],
) -> tuple[np.ndarray, np.ndarray]:
    point_blocks: list[np.ndarray] = []
    color_blocks: list[np.ndarray] = []
    for index, target in enumerate(targets):
        points = np.asarray(target.point_cloud_camera_m, dtype=np.float64)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        color = POINT_CLOUD_TARGET_COLORS[index % len(POINT_CLOUD_TARGET_COLORS)]
        colors = np.repeat(color.reshape(1, 3), points.shape[0], axis=0)
        point_blocks.append(points)
        color_blocks.append(colors)
    if not point_blocks:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return np.vstack(point_blocks), np.vstack(color_blocks)


def point_cloud_stats_text(
    points: np.ndarray,
    *,
    label: str,
) -> str:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return f"{label}: count=0"
    if points.ndim != 2 or points.shape[1] != 3:
        return f"{label}: invalid_shape={points.shape}"
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        return f"{label}: count=0 finite=0"
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = points.mean(axis=0)
    span_mm = (maximum - minimum) * 1000.0
    return (
        f"{label}: count={points.shape[0]} "
        f"center=({center[0]:+.4f},{center[1]:+.4f},{center[2]:+.4f})m "
        f"span=({span_mm[0]:.1f},{span_mm[1]:.1f},{span_mm[2]:.1f})mm "
        f"x=[{minimum[0]:+.4f},{maximum[0]:+.4f}]m "
        f"y=[{minimum[1]:+.4f},{maximum[1]:+.4f}]m "
        f"z=[{minimum[2]:+.4f},{maximum[2]:+.4f}]m"
    )


def print_point_cloud_stats(targets: list[GraspNetTarget]) -> None:
    all_points, _ = colored_point_cloud_from_targets(targets)
    print(point_cloud_stats_text(all_points, label="Point cloud all"), flush=True)
    for target in targets:
        print(
            point_cloud_stats_text(
                target.point_cloud_camera_m,
                label=f"  target #{target.rank}",
            ),
            flush=True,
        )


class Open3DPointCloudViewer:
    def __init__(self, *, window_name: str, point_size: float, axis_size_m: float) -> None:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "Open3D is required for --show-point-cloud. Install it with "
                "python -m pip install open3d."
            ) from exc
        self.o3d = o3d
        self.visualizer = o3d.visualization.Visualizer()
        if not self.visualizer.create_window(
            window_name=window_name,
            width=960,
            height=720,
                visible=True,
        ):
            raise RuntimeError("failed to create Open3D point cloud window")
        self.cloud = o3d.geometry.PointCloud()
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(axis_size_m),
            origin=(0.0, 0.0, 0.0),
        )
        self.visualizer.add_geometry(self.cloud)
        self.visualizer.add_geometry(self.axes)
        render_options = self.visualizer.get_render_option()
        render_options.point_size = float(point_size)
        render_options.background_color = np.asarray([0.02, 0.02, 0.02], dtype=float)
        view = self.visualizer.get_view_control()
        view.set_front([0.0, -0.35, -1.0])
        view.set_up([0.0, -1.0, 0.0])
        view.set_zoom(0.7)
        self._has_points = False

    def update(self, targets: list[GraspNetTarget]) -> bool:
        points, colors = colored_point_cloud_from_targets(targets)
        self.cloud.points = self.o3d.utility.Vector3dVector(points)
        self.cloud.colors = self.o3d.utility.Vector3dVector(colors)
        if points.size > 0 and not self._has_points:
            self.visualizer.reset_view_point(True)
            self._has_points = True
        self.visualizer.update_geometry(self.cloud)
        return bool(self.visualizer.poll_events())

    def destroy(self) -> None:
        self.visualizer.destroy_window()


def target_from_segment(
    rank: int,
    segment: SegmentationDetection,
    depth_frame: Any,
    *,
    intrinsics: PinholeIntrinsics,
    camera_to_base: np.ndarray | None,
    backend: GraspNetBackend | None,
    voxel_size_m: float,
    max_points: int,
    max_candidates: int,
) -> GraspNetTarget:
    depth_m = depth_array_meters(depth_frame)
    mask = segment.mask.astype(bool)
    values = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0.0)]
    if values.size == 0:
        raise ValueError("target mask contains no valid depth pixels")
    median_depth_m = float(np.median(values))
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    u, v = clamp_pixel_from_centroid(
        segment.centroid_u,
        segment.centroid_v,
        width=width,
        height=height,
    )
    raw_point = deproject_pixel_with_depth(
        depth_frame,
        u,
        v,
        median_depth_m,
        intrinsics=intrinsics,
    )
    base_xyz_m = None
    if camera_to_base is not None:
        base_xyz_m = map_camera_point_to_base_xyz_m(
            raw_point.x_m,
            raw_point.y_m,
            raw_point.z_m,
            camera_to_base,
        )
    points = point_cloud_from_mask(
        depth_frame,
        mask,
        intrinsics=intrinsics,
        voxel_size_m=voxel_size_m,
        max_points=max_points,
    )
    top_point, top_base_xyz_m = top_point_from_mask(
        depth_frame,
        mask,
        intrinsics=intrinsics,
        camera_to_base=camera_to_base,
    )
    grasp_candidates = (
        ()
        if backend is None
        else backend.predict(points)[:max_candidates]
    )
    return GraspNetTarget(
        rank=rank,
        segment=segment,
        raw_point=raw_point,
        send_point=raw_point,
        base_xyz_m=base_xyz_m,
        point_cloud_camera_m=points,
        grasp_candidates=tuple(grasp_candidates),
        top_point=top_point,
        top_base_xyz_m=top_base_xyz_m,
    )


def camera_point_command_from_target(
    target: GraspNetTarget,
    *,
    source: str,
    task: CameraPointTask,
    grasp_mode: Literal["static", "dynamic", "probe"],
    t_arrive_s: float | None = None,
    conveyor_speed_mm_s: float | None = None,
    pick_x_mm: float | None = None,
    pick_y_mm: float | None = None,
) -> CameraPointCommand:
    detection = target.detection
    detected_top_x_mm = None
    detected_top_y_mm = None
    detected_top_z_mm = None
    if target.top_base_xyz_m is not None:
        detected_top_x_mm = target.top_base_xyz_m[0] * 1000.0
        detected_top_y_mm = target.top_base_xyz_m[1] * 1000.0
        detected_top_z_mm = target.top_base_xyz_m[2] * 1000.0
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
        mask_area_px=target.segment.mask_area_px,
        mask_centroid_u=target.segment.centroid_u,
        mask_centroid_v=target.segment.centroid_v,
        mask_top_u=None if target.top_point is None else float(target.top_point.u),
        mask_top_v=None if target.top_point is None else float(target.top_point.v),
        grasp_mode=grasp_mode,
        t_arrive_s=t_arrive_s,
        conveyor_speed_mm_s=conveyor_speed_mm_s,
        pick_x_mm=pick_x_mm,
        pick_y_mm=pick_y_mm,
        detected_x_mm=(
            None if target.base_xyz_m is None else target.base_xyz_m[0] * 1000.0
        ),
        detected_y_mm=(
            None if target.base_xyz_m is None else target.base_xyz_m[1] * 1000.0
        ),
        detected_z_mm=(
            None if target.base_xyz_m is None else target.base_xyz_m[2] * 1000.0
        ),
        detected_top_x_mm=detected_top_x_mm,
        detected_top_y_mm=detected_top_y_mm,
        detected_top_z_mm=detected_top_z_mm,
        detected_top_camera_x_m=(
            None if target.top_point is None else target.top_point.x_m
        ),
        detected_top_camera_y_m=(
            None if target.top_point is None else target.top_point.y_m
        ),
        detected_top_camera_z_m=(
            None if target.top_point is None else target.top_point.z_m
        ),
        vision_source="graspnet",
        selected_grasp_index=0 if target.grasp_candidates else None,
        grasp_candidates=target.grasp_candidates,
    )


class GraspNetGraspSender:
    def __init__(self, host: str | None, port: int, timeout_s: float, dry_run: bool) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.dry_run = bool(dry_run)

    def send_target(
        self,
        target: GraspNetTarget,
        *,
        source: str,
        task: CameraPointTask,
        grasp_mode: Literal["static", "dynamic", "probe"],
        t_arrive_s: float | None = None,
        conveyor_speed_mm_s: float | None = None,
        pick_x_mm: float | None = None,
        pick_y_mm: float | None = None,
    ) -> dict[str, Any]:
        command = camera_point_command_from_target(
            target,
            source=source,
            task=task,
            grasp_mode=grasp_mode,
            t_arrive_s=t_arrive_s,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            pick_x_mm=pick_x_mm,
            pick_y_mm=pick_y_mm,
        )
        if self.dry_run:
            from dataclasses import asdict

            return {"type": "dry_run", "payload": asdict(command)}
        if not self.host:
            raise ValueError("--host is required unless --dry-run is used")
        return send_camera_point(self.host, self.port, command, self.timeout_s)


def dynamic_candidate_from_target(
    target: GraspNetTarget,
    *,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: str,
) -> DynamicGraspNetCandidate | None:
    if target.base_xyz_m is None:
        return None
    x_mm = target.base_xyz_m[0] * 1000.0
    y_mm = target.base_xyz_m[1] * 1000.0
    if x_mm < min_x_mm or x_mm > max_x_mm or y_mm < min_y_mm or y_mm > max_y_mm:
        return None
    try:
        t_arrive_s, remaining_y_mm = dynamic_arrival_time_s(
            y_mm,
            pick_y_mm=pick_y_mm,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            conveyor_direction=conveyor_direction,
        )
    except ValueError:
        return None
    return DynamicGraspNetCandidate(
        target=target,
        center_point=target.raw_point,
        base_xyz_m=target.base_xyz_m,
        t_arrive_s=t_arrive_s,
        remaining_y_mm=remaining_y_mm,
        distance_to_pick_y_mm=abs(remaining_y_mm),
    )


def dynamic_candidate_reject_reasons(
    target: GraspNetTarget,
    *,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: str,
) -> tuple[str, ...]:
    if target.base_xyz_m is None:
        return ("no-base-xyz",)
    x_mm = target.base_xyz_m[0] * 1000.0
    y_mm = target.base_xyz_m[1] * 1000.0
    reasons: list[str] = []
    if x_mm < min_x_mm:
        reasons.append(f"x<{min_x_mm:.1f}")
    elif x_mm > max_x_mm:
        reasons.append(f"x>{max_x_mm:.1f}")
    if y_mm < min_y_mm:
        reasons.append(f"y<{min_y_mm:.1f}")
    elif y_mm > max_y_mm:
        reasons.append(f"y>{max_y_mm:.1f}")
    try:
        t_arrive_s, _ = dynamic_arrival_time_s(
            y_mm,
            pick_y_mm=pick_y_mm,
            conveyor_speed_mm_s=conveyor_speed_mm_s,
            conveyor_direction=conveyor_direction,
        )
    except ValueError as exc:
        reasons.append(str(exc))
    else:
        if t_arrive_s < 0.0:
            reasons.append("past-pick-line")
    return tuple(reasons)


def format_dynamic_graspnet_candidate_line(
    target: GraspNetTarget,
    candidate: DynamicGraspNetCandidate | None,
    *,
    reject_reasons: tuple[str, ...],
) -> str:
    if target.base_xyz_m is None:
        base_text = "base=NA"
    else:
        base_text = (
            "base="
            f"({target.base_xyz_m[0] * 1000.0:.1f},"
            f"{target.base_xyz_m[1] * 1000.0:.1f},"
            f"{target.base_xyz_m[2] * 1000.0:.1f})mm"
        )
    top_z_text = (
        "top_z=NA"
        if target.top_base_xyz_m is None
        else f"top_z={target.top_base_xyz_m[2] * 1000.0:.1f}mm"
    )
    if candidate is None:
        status = (
            "reject=" + ",".join(reject_reasons)
            if reject_reasons
            else "reject=unknown"
        )
        timing = "remain_y=NA t_arrive=NA"
    else:
        status = "candidate"
        timing = (
            f"remain_y={candidate.remaining_y_mm:.1f}mm "
            f"t_arrive={candidate.t_arrive_s:.3f}s"
        )
    detection = target.detection
    return (
        f"rank={target.rank} "
        f"pixel=({target.raw_point.u},{target.raw_point.v}) "
        f"bbox=({detection.x},{detection.y},{detection.width},{detection.height}) "
        "camera="
        f"({target.raw_point.x_m:.4f},"
        f"{target.raw_point.y_m:.4f},"
        f"{target.raw_point.z_m:.4f})m "
        f"{base_text} {top_z_text} "
        f"grasps={len(target.grasp_candidates)} "
        f"{timing} {status}"
    )


def print_dynamic_graspnet_candidate_debug(
    targets: list[GraspNetTarget],
    candidates_by_rank: dict[int, DynamicGraspNetCandidate],
    *,
    min_x_mm: float,
    max_x_mm: float,
    min_y_mm: float,
    max_y_mm: float,
    pick_y_mm: float,
    conveyor_speed_mm_s: float,
    conveyor_direction: str,
) -> None:
    print(
        "\nDynamic GraspNet candidate debug: "
        f"X=[{min_x_mm:.1f},{max_x_mm:.1f}]mm "
        f"Y=[{min_y_mm:.1f},{max_y_mm:.1f}]mm "
        f"pick_y={pick_y_mm:.1f}mm "
        f"direction={conveyor_direction} "
        f"speed={conveyor_speed_mm_s:.1f}mm/s",
        flush=True,
    )
    if not targets:
        print("  no point-cloud targets", flush=True)
        return
    for target in targets:
        candidate = candidates_by_rank.get(target.rank)
        reject_reasons = ()
        if candidate is None:
            reject_reasons = dynamic_candidate_reject_reasons(
                target,
                min_x_mm=min_x_mm,
                max_x_mm=max_x_mm,
                min_y_mm=min_y_mm,
                max_y_mm=max_y_mm,
                pick_y_mm=pick_y_mm,
                conveyor_speed_mm_s=conveyor_speed_mm_s,
                conveyor_direction=conveyor_direction,
            )
        print(
            "  "
            + format_dynamic_graspnet_candidate_line(
                target,
                candidate,
                reject_reasons=reject_reasons,
            ),
            flush=True,
        )


def dynamic_pick_x_mm_for_candidate(
    candidate: DynamicGraspNetCandidate,
    *,
    fixed_pick_x_mm: float,
    use_detected_x: bool,
) -> float:
    if use_detected_x:
        return candidate.base_x_mm
    return float(fixed_pick_x_mm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect depth foreground objects, run a local GraspNet/AnyGrasp "
            "backend on object point clouds, and send 6D grasp candidates to "
            "catch_server_graspnet."
        )
    )
    parser.add_argument("mode", nargs="?", choices=("static", "dynamic"), default="static")
    parser.add_argument(
        "--host", help="Linux VM IP address running catch_server_graspnet"
    )
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="TCP response timeout; full server-side IK planning can take several seconds",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--task", choices=("out", "in"), default="out")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--intrinsics-json", type=Path, default=DEFAULT_D435I_PARA)
    parser.add_argument("--use-realsense-intrinsics", action="store_true")
    parser.add_argument(
        "--belt-depth-plane-json",
        type=Path,
        required=True,
        help="JSON from calibrate_belt_depth_plane used for depth foreground extraction",
    )
    parser.add_argument("--min-height-above-belt-mm", type=float, default=8.0)
    parser.add_argument("--max-height-above-belt-mm", type=float, default=200.0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--min-area-px", type=float, default=500.0)
    parser.add_argument("--max-area-px", type=float)
    parser.add_argument("--min-width-px", type=int, default=1)
    parser.add_argument("--min-height-px", type=int, default=1)
    parser.add_argument("--max-width-px", type=int)
    parser.add_argument("--max-height-px", type=int)
    parser.add_argument("--min-fill-ratio", type=float, default=0.0)
    parser.add_argument("--morphology-kernel-size", type=int, default=3)
    parser.add_argument("--contour-mask-mode", choices=("raw", "convex-hull"), default="raw")
    parser.add_argument("--reject-roi-border-touching", action="store_true")
    parser.add_argument("--entry-edge", choices=("left", "right", "top", "bottom"))
    parser.add_argument("--entry-clear-margin-px", type=int, default=0)
    parser.add_argument("--entry-clear-frames", type=int, default=1)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--voxel-size-m", type=float, default=0.004)
    parser.add_argument("--max-point-count", type=int, default=20000)
    parser.add_argument("--max-grasp-candidates", type=int, default=20)
    parser.add_argument("--graspnet-module", default="graspnet_backend")
    parser.add_argument("--graspnet-class", default="GraspNetPredictor")
    parser.add_argument("--graspnet-checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--no-graspnet", action="store_true", help="debug only: send no candidates")
    parser.add_argument("--hand-eye-result", type=Path, default=DEFAULT_HAND_EYE_RESULT)
    parser.add_argument("--no-hand-eye-result", action="store_true")
    parser.add_argument("--conveyor-speed-mm-s", type=float)
    parser.add_argument(
        "--conveyor-direction",
        choices=("negative-y", "positive-y"),
        default="negative-y",
    )
    parser.add_argument("--dynamic-min-x-mm", type=float, default=300.0)
    parser.add_argument("--dynamic-max-x-mm", type=float, default=400.0)
    parser.add_argument("--dynamic-min-y-mm", type=float, default=-30.0)
    parser.add_argument("--dynamic-max-y-mm", type=float, default=30.0)
    parser.add_argument("--dynamic-pick-x-mm", type=float, default=350.0)
    parser.add_argument(
        "--dynamic-pick-x-from-detected",
        action="store_true",
        help=(
            "use the detected target center base-X as dynamic grasp X instead of "
            "--dynamic-pick-x-mm; dynamic Y still comes from --dynamic-pick-y-mm"
        ),
    )
    parser.add_argument("--dynamic-pick-y-mm", type=float, default=-30.0)
    parser.add_argument("--dynamic-keep-running", action="store_true")
    parser.add_argument(
        "--dynamic-debug-candidates",
        action="store_true",
        help=(
            "print per-target dynamic candidate diagnostics, including base XYZ, "
            "top Z, window checks, and arrival time"
        ),
    )
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--hide-mask", action="store_true")
    parser.add_argument(
        "--show-point-cloud",
        action="store_true",
        help="open an Open3D window showing foreground object point clouds",
    )
    parser.add_argument(
        "--point-cloud-point-size",
        type=float,
        default=3.0,
        help="Open3D point size used with --show-point-cloud",
    )
    parser.add_argument(
        "--point-cloud-axis-size-m",
        type=float,
        default=0.10,
        help="camera-frame coordinate axis length in meters for --show-point-cloud",
    )
    parser.add_argument("--no-display", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.host:
        raise ValueError("--host is required unless --dry-run is used")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.max_results < 1:
        raise ValueError("--max-results must be >= 1")
    if args.max_grasp_candidates < 1:
        raise ValueError("--max-grasp-candidates must be >= 1")
    if args.max_point_count < 1:
        raise ValueError("--max-point-count must be >= 1")
    if args.voxel_size_m < 0:
        raise ValueError("--voxel-size-m must be >= 0")
    if args.point_cloud_point_size <= 0:
        raise ValueError("--point-cloud-point-size must be > 0")
    if args.point_cloud_axis_size_m <= 0:
        raise ValueError("--point-cloud-axis-size-m must be > 0")
    if args.min_height_above_belt_mm < 0:
        raise ValueError("--min-height-above-belt-mm must be >= 0")
    if args.max_height_above_belt_mm < args.min_height_above_belt_mm:
        raise ValueError("--max-height-above-belt-mm must be >= --min-height-above-belt-mm")
    if args.roi is not None:
        x, y, width, height = args.roi
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("--roi must be X Y W H with non-negative X/Y and positive W/H")
        if x + width > args.width or y + height > args.height:
            raise ValueError("--roi must be inside the configured image width/height")
    if args.mode == "dynamic":
        if args.no_hand_eye_result:
            raise ValueError("dynamic mode requires --hand-eye-result")
        if args.conveyor_speed_mm_s is None or args.conveyor_speed_mm_s <= 0.0:
            raise ValueError("--conveyor-speed-mm-s must be > 0 in dynamic mode")
        if args.dynamic_min_x_mm > args.dynamic_max_x_mm:
            raise ValueError("--dynamic-min-x-mm must be <= --dynamic-max-x-mm")
        if args.dynamic_min_y_mm > args.dynamic_max_y_mm:
            raise ValueError("--dynamic-min-y-mm must be <= --dynamic-max-y-mm")


def detector_config_from_args(args: argparse.Namespace) -> DepthForegroundDetectorConfig:
    return DepthForegroundDetectorConfig(
        min_height_above_belt_mm=float(args.min_height_above_belt_mm),
        max_height_above_belt_mm=float(args.max_height_above_belt_mm),
        min_area_px=float(args.min_area_px),
        max_area_px=args.max_area_px,
        min_width_px=int(args.min_width_px),
        min_height_px=int(args.min_height_px),
        max_width_px=args.max_width_px,
        max_height_px=args.max_height_px,
        min_fill_ratio=float(args.min_fill_ratio),
        max_results=int(args.max_results),
        morphology_kernel_size=int(args.morphology_kernel_size),
        contour_mask_mode=args.contour_mask_mode,
        roi_xywh=tuple(int(item) for item in args.roi) if args.roi is not None else None,
        reject_roi_border_touching=bool(args.reject_roi_border_touching),
        entry_edge=args.entry_edge,
        entry_clear_margin_px=int(args.entry_clear_margin_px),
        entry_clear_frames=int(args.entry_clear_frames),
    )


def load_intrinsics(args: argparse.Namespace, depth_frame: Any) -> PinholeIntrinsics:
    if args.use_realsense_intrinsics:
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
    return find_d435i_intrinsics(args.width, args.height, args.intrinsics_json)


def draw_targets(image: np.ndarray, targets: list[GraspNetTarget]) -> None:
    for target in targets:
        detection = target.detection
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (0, 255, 255),
            2,
        )
        cv2.drawMarker(
            image,
            (target.raw_point.u, target.raw_point.v),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=16,
            thickness=2,
        )
        label = f"#{target.rank} grasps={len(target.grasp_candidates)}"
        cv2.putText(
            image,
            label,
            (detection.x, max(16, detection.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (detection.x, max(16, detection.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    belt_plane = load_belt_depth_plane(args.belt_depth_plane_json)
    detector = DepthForegroundDetector(belt_plane, detector_config_from_args(args))
    camera_to_base = None if args.no_hand_eye_result else load_camera_to_base_transform(args.hand_eye_result)
    backend = None
    if not args.no_graspnet:
        backend = GraspNetBackend(
            args.graspnet_module,
            args.graspnet_class,
            args.graspnet_checkpoint,
            args.device,
            args.max_grasp_candidates,
        )

    pipeline, align = create_aligned_rgb_depth_stream(args.width, args.height, args.fps)
    sender = GraspNetGraspSender(args.host, args.port, args.timeout_s, args.dry_run)
    terminal = TerminalInput()
    terminal.start()
    source = "graspnet_point_cloud_static" if args.mode == "static" else "graspnet_point_cloud_dynamic"
    dynamic_sent = False
    intrinsics: PinholeIntrinsics | None = None
    point_cloud_viewer = None
    last_point_cloud_print_s = 0.0
    last_dynamic_debug_time_s = 0.0

    if args.show_point_cloud:
        point_cloud_viewer = Open3DPointCloudViewer(
            window_name=POINT_CLOUD_WINDOW_NAME,
            point_size=args.point_cloud_point_size,
            axis_size_m=args.point_cloud_axis_size_m,
        )

    if not args.no_display:
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_mask:
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_depth:
            cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            color_image, depth_frame = wait_for_aligned_rgb_depth(
                pipeline,
                align,
                timeout_ms=args.frame_timeout_ms,
            )
            if color_image is None or depth_frame is None:
                continue
            if intrinsics is None:
                intrinsics = load_intrinsics(args, depth_frame)
                print(format_intrinsics(intrinsics), flush=True)

            segments = detector.detect(depth_frame)
            targets: list[GraspNetTarget] = []
            for rank, segment in enumerate(segments, start=1):
                try:
                    targets.append(
                        target_from_segment(
                            rank,
                            segment,
                            depth_frame,
                            intrinsics=intrinsics,
                            camera_to_base=camera_to_base,
                            backend=backend,
                            voxel_size_m=args.voxel_size_m,
                            max_points=args.max_point_count,
                            max_candidates=args.max_grasp_candidates,
                        )
                    )
                except (RuntimeError, ValueError) as exc:
                    print(f"Skipping target {rank}: {exc}", flush=True)

            if point_cloud_viewer is not None:
                total_points = sum(
                    int(target.point_cloud_camera_m.shape[0])
                    for target in targets
                    if target.point_cloud_camera_m.ndim == 2
                )
                now_s = time.monotonic()
                if now_s - last_point_cloud_print_s >= args.print_interval:
                    print(f"Point cloud targets={len(targets)} points={total_points}", flush=True)
                    print_point_cloud_stats(targets)
                    last_point_cloud_print_s = now_s
                if not point_cloud_viewer.update(targets):
                    break

            command = terminal.get_nowait()
            if command:
                if command.lower() in ("q", "quit", "exit"):
                    break
                parsed = parse_target_command(command, default_task=args.task)
                if parsed is None:
                    print(f"Unknown command: {command!r}", flush=True)
                else:
                    task, rank = parsed
                    target = next((item for item in targets if item.rank == rank), None)
                    if target is None:
                        print(f"No target rank {rank}", flush=True)
                    else:
                        response = sender.send_target(
                            target,
                            source=source,
                            task=task,
                            grasp_mode="static",
                        )
                        print(f"Sent rank {rank}: {response}", flush=True)

            if args.mode == "dynamic" and (not dynamic_sent or args.dynamic_keep_running):
                if camera_to_base is None:
                    raise ValueError("dynamic mode requires --hand-eye-result")
                candidates = [
                    dynamic_candidate_from_target(
                        target,
                        min_x_mm=args.dynamic_min_x_mm,
                        max_x_mm=args.dynamic_max_x_mm,
                        min_y_mm=args.dynamic_min_y_mm,
                        max_y_mm=args.dynamic_max_y_mm,
                        pick_y_mm=args.dynamic_pick_y_mm,
                        conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                        conveyor_direction=args.conveyor_direction,
                    )
                    for target in targets
                ]
                candidates = [item for item in candidates if item is not None]
                candidates.sort(key=lambda item: (item.t_arrive_s, item.target.rank))
                now_s = time.monotonic()
                if (
                    args.dynamic_debug_candidates
                    and (
                        args.print_interval == 0.0
                        or now_s - last_dynamic_debug_time_s >= args.print_interval
                    )
                ):
                    print_dynamic_graspnet_candidate_debug(
                        targets,
                        {item.target.rank: item for item in candidates},
                        min_x_mm=args.dynamic_min_x_mm,
                        max_x_mm=args.dynamic_max_x_mm,
                        min_y_mm=args.dynamic_min_y_mm,
                        max_y_mm=args.dynamic_max_y_mm,
                        pick_y_mm=args.dynamic_pick_y_mm,
                        conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                        conveyor_direction=args.conveyor_direction,
                    )
                    last_dynamic_debug_time_s = now_s
                if candidates:
                    candidate = candidates[0]
                    pick_x_mm = dynamic_pick_x_mm_for_candidate(
                        candidate,
                        fixed_pick_x_mm=args.dynamic_pick_x_mm,
                        use_detected_x=args.dynamic_pick_x_from_detected,
                    )
                    response = sender.send_target(
                        candidate.target,
                        source=source,
                        task=args.task,
                        grasp_mode="dynamic",
                        t_arrive_s=candidate.t_arrive_s,
                        conveyor_speed_mm_s=args.conveyor_speed_mm_s,
                        pick_x_mm=pick_x_mm,
                        pick_y_mm=args.dynamic_pick_y_mm,
                    )
                    skip_reason = dynamic_skip_reason_from_response(response)
                    if skip_reason:
                        print(f"Dynamic target skipped: {skip_reason}", flush=True)
                    elif response.get("type") == "error":
                        print(f"Dynamic send failed: {response}", flush=True)
                    else:
                        dynamic_sent = True
                        print(
                            f"Dynamic target sent: pick_x={pick_x_mm:.1f}mm {response}",
                            flush=True,
                        )

            if not args.no_display:
                display = color_image.copy()
                draw_targets(display, targets)
                cv2.imshow(RGB_WINDOW_NAME, display)
                if not args.hide_mask and detector.last_foreground_mask is not None:
                    cv2.imshow(MASK_WINDOW_NAME, detector.last_foreground_mask)
                if not args.hide_depth:
                    cv2.imshow(DEPTH_WINDOW_NAME, make_depth_colormap(depth_frame, args.max_depth_m))
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
    finally:
        pipeline.stop()
        if point_cloud_viewer is not None:
            point_cloud_viewer.destroy()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

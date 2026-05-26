
"""Depth utilities for converting RealSense pixels to camera-frame XYZ."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraPoint:
    """A clicked image pixel and its 3D position in the camera frame."""

    u: int
    v: int
    depth_m: float
    x_m: float
    y_m: float
    z_m: float


def _require_realsense() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required for RealSense depth conversion. "
            "Install project requirements first."
        ) from exc
    return rs


def create_aligned_rgb_depth_stream(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> Tuple[Any, Any]:
    """Start RGB and depth streams and create an aligner to color coordinates."""

    rs = _require_realsense()
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align


def wait_for_aligned_rgb_depth(
    pipeline: Any,
    align: Any,
    timeout_ms: int = 10000,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """Return an RGB image and depth frame after aligning depth to the RGB stream."""

    frames = pipeline.wait_for_frames(timeout_ms)
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None

    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_frame


def read_depth_meters(
    depth_frame: Any,
    u: int,
    v: int,
    sample_radius: int = 0,
) -> float:
    """Read depth in meters at a pixel, optionally using median valid depth nearby."""

    u = int(u)
    v = int(v)
    sample_radius = int(sample_radius)

    if sample_radius <= 0:
        return float(depth_frame.get_distance(u, v))

    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    x0 = max(0, u - sample_radius)
    x1 = min(width - 1, u + sample_radius)
    y0 = max(0, v - sample_radius)
    y1 = min(height - 1, v + sample_radius)

    valid_depths = []
    for py in range(y0, y1 + 1):
        for px in range(x0, x1 + 1):
            depth_m = float(depth_frame.get_distance(px, py))
            if np.isfinite(depth_m) and depth_m > 0.0:
                valid_depths.append(depth_m)

    if not valid_depths:
        return float(depth_frame.get_distance(u, v))
    return float(np.median(valid_depths))


def deproject_pixel_to_camera_xyz(
    depth_frame: Any,
    u: int,
    v: int,
    sample_radius: int = 0,
) -> CameraPoint:
    """Convert an aligned RGB pixel to RealSense camera-frame XYZ in meters."""

    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    u = int(u)
    v = int(v)
    if u < 0 or u >= width or v < 0 or v >= height:
        raise ValueError(f"pixel ({u}, {v}) is outside depth frame {width}x{height}")

    depth_m = read_depth_meters(depth_frame, u, v, sample_radius=sample_radius)
    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError(f"invalid depth at pixel ({u}, {v}): {depth_m:.6f} m")

    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    rs = _require_realsense()
    x_m, y_m, z_m = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [float(u), float(v)],
        float(depth_m),
    )
    return CameraPoint(
        u=u,
        v=v,
        depth_m=float(depth_m),
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=float(z_m),
    )


def format_camera_point(point: CameraPoint) -> str:
    return (
        f"u={point.u}, v={point.v}, depth={point.depth_m:.4f} m, "
        f"X={point.x_m:.4f} m, Y={point.y_m:.4f} m, Z={point.z_m:.4f} m"
    )

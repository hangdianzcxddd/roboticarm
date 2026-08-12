"""Chessboard detection and pose estimation for hand-eye calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from robot_control.shared.hand_eye_protocol import PoseVector


DEFAULT_INTRINSICS: dict[str, dict[str, float | int]] = {
    "640x480": {
        "width": 640,
        "height": 480,
        "fx": 391.185,
        "fy": 391.185,
        "cx": 319.123,
        "cy": 244.756,
    },
    "848x480": {
        "width": 848,
        "height": 480,
        "fx": 431.933,
        "fy": 431.933,
        "cx": 423.031,
        "cy": 245.199,
    },
    "1280x720": {
        "width": 1280,
        "height": 720,
        "fx": 651.975,
        "fy": 651.975,
        "cx": 638.538,
        "cy": 367.927,
    },
    "1920x1080": {
        "width": 1920,
        "height": 1080,
        "fx": 977.962,
        "fy": 977.962,
        "cx": 957.807,
        "cy": 551.891,
    },
}


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray


@dataclass(frozen=True)
class ChessboardConfig:
    corners_cols: int = 11
    corners_rows: int = 8
    checker_size_m: float = 0.015


@dataclass(frozen=True)
class ChessboardDetection:
    pose: PoseVector
    corners: np.ndarray
    reprojection_error_px: float
    object_points: np.ndarray


def intrinsics_from_profile(
    profile: str,
    *,
    dist_coeffs: tuple[float, ...] | None = None,
) -> CameraIntrinsics:
    try:
        values = DEFAULT_INTRINSICS[profile]
    except KeyError:
        raise ValueError(
            f"unknown intrinsic profile {profile!r}; choose one of "
            f"{', '.join(sorted(DEFAULT_INTRINSICS))}"
        ) from None
    return make_intrinsics(
        width=int(values["width"]),
        height=int(values["height"]),
        fx=float(values["fx"]),
        fy=float(values["fy"]),
        cx=float(values["cx"]),
        cy=float(values["cy"]),
        dist_coeffs=dist_coeffs,
    )


def intrinsics_from_realsense_profile(
    video_profile: Any,
    *,
    dist_coeffs: tuple[float, ...] | None = None,
    use_live_distortion: bool = False,
) -> CameraIntrinsics:
    """Build OpenCV intrinsics from the active RealSense video profile."""

    rs_intrinsics = getattr(video_profile, "intrinsics", None)
    if rs_intrinsics is None and hasattr(video_profile, "get_intrinsics"):
        rs_intrinsics = video_profile.get_intrinsics()
    if rs_intrinsics is None:
        raise ValueError("RealSense video profile does not expose intrinsics")

    live_dist_coeffs = dist_coeffs
    if live_dist_coeffs is None and use_live_distortion:
        coeffs = getattr(rs_intrinsics, "coeffs", None)
        if coeffs is not None:
            live_dist_coeffs = tuple(float(value) for value in coeffs)

    return make_intrinsics(
        width=int(rs_intrinsics.width),
        height=int(rs_intrinsics.height),
        fx=float(rs_intrinsics.fx),
        fy=float(rs_intrinsics.fy),
        cx=float(rs_intrinsics.ppx),
        cy=float(rs_intrinsics.ppy),
        dist_coeffs=live_dist_coeffs,
    )


def make_intrinsics(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: tuple[float, ...] | None = None,
) -> CameraIntrinsics:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    coeffs = np.zeros((5, 1), dtype=np.float64)
    if dist_coeffs is not None:
        if len(dist_coeffs) not in (4, 5, 8, 12, 14):
            raise ValueError("dist_coeffs length must match an OpenCV distortion model")
        coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    return CameraIntrinsics(
        width=width,
        height=height,
        camera_matrix=camera_matrix,
        dist_coeffs=coeffs,
    )


def make_object_points(config: ChessboardConfig) -> np.ndarray:
    if config.corners_cols < 2 or config.corners_rows < 2:
        raise ValueError("chessboard must have at least 2x2 inner corners")
    if config.checker_size_m <= 0:
        raise ValueError("checker_size_m must be > 0")
    points = np.zeros(
        (config.corners_rows * config.corners_cols, 3),
        dtype=np.float32,
    )
    grid = np.mgrid[0 : config.corners_cols, 0 : config.corners_rows].T.reshape(-1, 2)
    points[:, :2] = grid.astype(np.float32) * float(config.checker_size_m)
    return points


def estimate_chessboard_pose(
    frame_bgr: np.ndarray,
    config: ChessboardConfig,
    intrinsics: CameraIntrinsics,
) -> ChessboardDetection | None:
    if frame_bgr.ndim not in (2, 3):
        raise ValueError("frame must be a grayscale or BGR image")
    gray = (
        frame_bgr
        if frame_bgr.ndim == 2
        else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    )
    pattern_size = (config.corners_cols, config.corners_rows)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    object_points = make_object_points(config)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        refined,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    error = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(projected.reshape(-1, 2) - refined.reshape(-1, 2)),
                    axis=1,
                )
            )
        )
    )
    return ChessboardDetection(
        pose=PoseVector(
            rvec=tuple(float(value) for value in rvec.reshape(3)),
            tvec_m=tuple(float(value) for value in tvec.reshape(3)),
        ),
        corners=refined,
        reprojection_error_px=error,
        object_points=object_points,
    )


def draw_detection(
    frame_bgr: np.ndarray,
    config: ChessboardConfig,
    intrinsics: CameraIntrinsics,
    detection: ChessboardDetection | None,
) -> np.ndarray:
    display = frame_bgr.copy()
    if detection is None:
        return display
    cv2.drawChessboardCorners(
        display,
        (config.corners_cols, config.corners_rows),
        detection.corners,
        True,
    )
    axis_length = config.checker_size_m * 3.0
    cv2.drawFrameAxes(
        display,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        np.asarray(detection.pose.rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(detection.pose.tvec_m, dtype=np.float64).reshape(3, 1),
        axis_length,
    )
    return display


def put_status_text(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    display = frame_bgr.copy()
    y = 24
    for line in lines:
        cv2.putText(
            display,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return display

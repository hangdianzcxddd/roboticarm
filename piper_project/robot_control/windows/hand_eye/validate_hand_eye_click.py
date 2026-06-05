"""Validate an eye-to-hand camera->base transform with clicked RealSense points.

Run on Windows from the project root:

    python -m robot_control.windows.hand_eye.validate_hand_eye_click \
      --expected-base-mm 0 0 0

Left-click a visible point whose robot-base coordinates are known. The script
prints the clicked camera XYZ, the transformed base XYZ, and the error against
``--expected-base-mm`` when supplied. Press Esc to exit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from robot_control.linux.catch.hand_eye_mapping import (
    hand_eye_quality_summary,
    load_camera_to_base_transform,
    load_hand_eye_result_document,
    map_camera_point_to_base_xyz_m,
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


DEFAULT_HAND_EYE_RESULT = (
    PROJECT_DIR / "robot_control" / "linux" / "hand_eye" / "hand_eye_result.json"
)
WINDOW_NAME = "Hand-eye click validation"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


@dataclass(frozen=True)
class ClickValidationResult:
    camera_point: CameraPoint
    base_xyz_m: tuple[float, float, float]
    expected_base_xyz_m: tuple[float, float, float] | None = None
    error_xyz_m: tuple[float, float, float] | None = None
    error_norm_m: float | None = None


class ClickHandEyeValidator:
    def __init__(
        self,
        camera_to_base: np.ndarray,
        sample_radius: int,
        intrinsics: PinholeIntrinsics | None,
        expected_base_xyz_m: tuple[float, float, float] | None,
        log_jsonl: Path | None,
    ) -> None:
        self.camera_to_base = camera_to_base
        self.sample_radius = int(sample_radius)
        self.intrinsics = intrinsics
        self.expected_base_xyz_m = expected_base_xyz_m
        self.log_jsonl = log_jsonl
        self.depth_frame: Optional[Any] = None
        self.last_result: Optional[ClickValidationResult] = None
        self.last_error: Optional[str] = None
        self.last_click: Optional[Tuple[int, int]] = None

    def update_depth_frame(self, depth_frame: Any) -> None:
        self.depth_frame = depth_frame

    def on_mouse(self, event: int, u: int, v: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.last_click = (u, v)
        if self.depth_frame is None:
            self._set_error("No depth frame is available yet.")
            return

        try:
            point = deproject_pixel_to_camera_xyz(
                self.depth_frame,
                u,
                v,
                sample_radius=self.sample_radius,
                intrinsics=self.intrinsics,
            )
            result = make_click_validation_result(
                point,
                self.camera_to_base,
                expected_base_xyz_m=self.expected_base_xyz_m,
            )
            if self.log_jsonl is not None:
                append_result_jsonl(self.log_jsonl, result)
        except Exception as exc:  # noqa: BLE001 - displayed in manual test UI
            self.last_result = None
            self._set_error(str(exc))
            return

        self.last_result = result
        self.last_error = None
        print(format_click_validation_result(result), flush=True)

    def _set_error(self, error: str) -> None:
        self.last_error = error
        print(error, flush=True)

    def last_click_pixel(self) -> Optional[Tuple[int, int]]:
        if self.last_result is None:
            return self.last_click
        return self.last_result.camera_point.u, self.last_result.camera_point.v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Click a RealSense RGB point, transform camera XYZ through an "
            "eye-to-hand camera->base matrix, and optionally compare to a known "
            "base-frame point."
        )
    )
    parser.add_argument(
        "--hand-eye-result",
        type=Path,
        default=DEFAULT_HAND_EYE_RESULT,
        help="path to eye_to_hand hand_eye_result.json containing camera -> base",
    )
    parser.add_argument(
        "--expected-base-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "known coordinates of the clicked physical point in robot base frame. "
            "Use 0 0 0 only if the clicked mark is the true base origin."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=3)
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=DEFAULT_D435I_PARA,
        help="D435i parameter JSON used to select rectified fx/fy/ppx/ppy",
    )
    parser.add_argument(
        "--use-realsense-intrinsics",
        action="store_true",
        help="ignore --intrinsics-json and use intrinsics reported by the frame",
    )
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        help="optional JSONL file for clicked validation records",
    )
    return parser


def make_click_validation_result(
    point: CameraPoint,
    camera_to_base: np.ndarray,
    *,
    expected_base_xyz_m: tuple[float, float, float] | None = None,
) -> ClickValidationResult:
    base_xyz_m = map_camera_point_to_base_xyz_m(
        point.x_m,
        point.y_m,
        point.z_m,
        camera_to_base,
    )
    if expected_base_xyz_m is None:
        return ClickValidationResult(point, base_xyz_m)

    error = tuple(
        float(base_value - expected_value)
        for base_value, expected_value in zip(base_xyz_m, expected_base_xyz_m)
    )
    return ClickValidationResult(
        camera_point=point,
        base_xyz_m=base_xyz_m,
        expected_base_xyz_m=expected_base_xyz_m,
        error_xyz_m=error,
        error_norm_m=float(np.linalg.norm(np.asarray(error, dtype=float))),
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


def format_click_validation_result(result: ClickValidationResult) -> str:
    parts = [
        format_camera_point(result.camera_point),
        f"base={format_xyz_mm(result.base_xyz_m)}",
    ]
    if result.expected_base_xyz_m is not None:
        parts.append(f"expected={format_xyz_mm(result.expected_base_xyz_m)}")
    if result.error_xyz_m is not None and result.error_norm_m is not None:
        parts.append(
            "error="
            f"{format_xyz_mm(result.error_xyz_m)} "
            f"norm={result.error_norm_m * 1000.0:.2f} mm"
        )
    return " | ".join(parts)


def format_xyz_mm(xyz_m: tuple[float, float, float]) -> str:
    return (
        f"X={xyz_m[0] * 1000.0:.2f} mm, "
        f"Y={xyz_m[1] * 1000.0:.2f} mm, "
        f"Z={xyz_m[2] * 1000.0:.2f} mm"
    )


def append_result_jsonl(path: Path, result: ClickValidationResult) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "camera": {
            "u": result.camera_point.u,
            "v": result.camera_point.v,
            "depth_m": result.camera_point.depth_m,
            "xyz_m": [
                result.camera_point.x_m,
                result.camera_point.y_m,
                result.camera_point.z_m,
            ],
        },
        "base_xyz_m": list(result.base_xyz_m),
        "expected_base_xyz_m": (
            list(result.expected_base_xyz_m)
            if result.expected_base_xyz_m is not None
            else None
        ),
        "error_xyz_m": list(result.error_xyz_m) if result.error_xyz_m is not None else None,
        "error_norm_m": result.error_norm_m,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, separators=(",", ":")) + "\n")


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


def draw_status(
    image: Any,
    validator: ClickHandEyeValidator,
    projected_expected: tuple[int, int, float] | None,
) -> None:
    if projected_expected is not None:
        u, v, _ = projected_expected
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
            cv2.drawMarker(
                image,
                (u, v),
                (0, 255, 0),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=22,
                thickness=2,
            )
            draw_text_lines(image, ["expected"], (u + 8, v - 8), (0, 255, 0))

    click_pixel = validator.last_click_pixel()
    if click_pixel is not None:
        cv2.drawMarker(
            image,
            click_pixel,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if validator.last_result is not None:
        lines = status_lines_from_result(validator.last_result)
        color = (0, 255, 255)
    elif validator.last_error:
        lines = [validator.last_error]
        color = (0, 0, 255)
    else:
        lines = ["Left-click a known base-frame point. Press Esc to exit."]
        color = (255, 255, 255)

    draw_text_lines(image, lines, (12, 26), color)


def status_lines_from_result(result: ClickValidationResult) -> list[str]:
    lines = [
        (
            f"camera u={result.camera_point.u} v={result.camera_point.v} "
            f"depth={result.camera_point.depth_m:.4f} m"
        ),
        f"base {format_xyz_mm(result.base_xyz_m)}",
    ]
    if result.error_xyz_m is not None and result.error_norm_m is not None:
        lines.append(
            f"error {format_xyz_mm(result.error_xyz_m)} "
            f"norm={result.error_norm_m * 1000.0:.2f} mm"
        )
    return lines


def draw_text_lines(
    image: Any,
    lines: list[str],
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        baseline = y + index * 22
        cv2.putText(
            image,
            line,
            (x, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def expected_base_xyz_m_from_args(args: argparse.Namespace) -> tuple[float, float, float] | None:
    if args.expected_base_mm is None:
        return None
    values = tuple(float(value) / 1000.0 for value in args.expected_base_mm)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("--expected-base-mm values must be finite")
    return values


def intrinsics_from_depth_frame(depth_frame: Any) -> PinholeIntrinsics:
    video_profile = depth_frame.profile.as_video_stream_profile()
    rs_intrinsics = video_profile.intrinsics
    return PinholeIntrinsics(
        width=int(rs_intrinsics.width),
        height=int(rs_intrinsics.height),
        fx=float(rs_intrinsics.fx),
        fy=float(rs_intrinsics.fy),
        ppx=float(rs_intrinsics.ppx),
        ppy=float(rs_intrinsics.ppy),
        source="realsense_frame",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.frame_timeout_ms <= 0:
        raise ValueError("--frame-timeout-ms must be positive")
    if args.sample_radius < 0:
        raise ValueError("--sample-radius must be >= 0")
    if args.max_depth_m <= 0:
        raise ValueError("--max-depth-m must be > 0")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
    document = load_hand_eye_result_document(args.hand_eye_result)
    quality = hand_eye_quality_summary(document)
    print(f"Using hand-eye camera->base: {args.hand_eye_result}", flush=True)
    if quality:
        print(f"Calibration quality: {quality}", flush=True)

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

    expected_base_xyz_m = expected_base_xyz_m_from_args(args)
    if expected_base_xyz_m is None:
        print(
            "No --expected-base-mm supplied; clicks will only report mapped base XYZ.",
            flush=True,
        )
    else:
        print(f"Expected clicked point: {format_xyz_mm(expected_base_xyz_m)}", flush=True)

    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    validator = ClickHandEyeValidator(
        camera_to_base=camera_to_base,
        sample_radius=args.sample_radius,
        intrinsics=intrinsics,
        expected_base_xyz_m=expected_base_xyz_m,
        log_jsonl=args.log_jsonl,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, validator.on_mouse)
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

            validator.update_depth_frame(depth_frame)
            current_intrinsics = intrinsics or intrinsics_from_depth_frame(depth_frame)
            projected_expected = None
            if expected_base_xyz_m is not None:
                projected_expected = project_base_point_to_pixel(
                    camera_to_base,
                    expected_base_xyz_m,
                    current_intrinsics,
                )

            display_image = color_image.copy()
            draw_status(display_image, validator, projected_expected)
            cv2.imshow(WINDOW_NAME, display_image)

            if not args.hide_depth:
                depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                draw_status(depth_debug, validator, projected_expected)
                cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

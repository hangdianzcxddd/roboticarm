"""Calibrate an image-space depth plane for an empty conveyor belt.

Run from ``piper_project`` on the Windows/RealSense host:

    python -m robot_control.windows.client.calibrate_belt_depth_plane \
        --roi 80 60 500 340 --output belt_depth_plane.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot_control.windows.vision.belt_depth_plane import (
    BeltDepthPlane,
    fit_belt_depth_plane,
    patch_depth_median,
    save_belt_depth_plane,
)
from robot_control.windows.vision.depth import (
    create_aligned_rgb_depth_stream,
    wait_for_aligned_rgb_depth,
)


WINDOW_NAME = "Belt depth plane calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit z_belt = a*u + b*v + c from empty conveyor depth samples."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="optional conveyor ROI in image pixels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("belt_depth_plane.json"),
        help="output JSON path",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "manual"),
        default="auto",
        help="auto samples a grid inside ROI; manual samples clicked points",
    )
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-rows", type=int, default=5)
    parser.add_argument(
        "--grid-margin-px",
        type=int,
        default=25,
        help="inset margin for automatic grid samples inside ROI/frame",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="odd patch size used for each depth sample median",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=1.5,
        help="ignore depth samples farther than this; 0 disables",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=6,
        help="minimum valid samples required before saving",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="do not open preview windows; useful for auto mode",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.frame_timeout_ms <= 0:
        raise ValueError("--frame-timeout-ms must be > 0")
    if args.roi is not None:
        x, y, width, height = args.roi
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("--roi must be X Y W H with non-negative X/Y and positive W/H")
        if x + width > args.width or y + height > args.height:
            raise ValueError("--roi must be inside the configured image width/height")
    if args.grid_cols < 2 or args.grid_rows < 2:
        raise ValueError("--grid-cols and --grid-rows must be >= 2")
    if args.grid_margin_px < 0:
        raise ValueError("--grid-margin-px must be >= 0")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be a positive odd number")
    if args.max_depth_m < 0.0:
        raise ValueError("--max-depth-m must be >= 0")
    if args.min_samples < 3:
        raise ValueError("--min-samples must be >= 3")
    if args.mode == "manual" and args.no_display:
        raise ValueError("--mode manual requires display")


def roi_bounds_for_frame(
    width: int,
    height: int,
    roi_xywh: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    if roi_xywh is None:
        return 0, 0, int(width), int(height)
    x, y, roi_width, roi_height = roi_xywh
    x0 = max(0, min(int(width), int(x)))
    y0 = max(0, min(int(height), int(y)))
    x1 = max(x0, min(int(width), x0 + int(roi_width)))
    y1 = max(y0, min(int(height), y0 + int(roi_height)))
    return x0, y0, x1, y1


def grid_sample_pixels(
    width: int,
    height: int,
    roi_xywh: tuple[int, int, int, int] | None,
    *,
    cols: int,
    rows: int,
    margin_px: int,
) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = roi_bounds_for_frame(width, height, roi_xywh)
    margin = int(margin_px)
    sx0 = min(x1 - 1, x0 + margin)
    sx1 = max(sx0, x1 - 1 - margin)
    sy0 = min(y1 - 1, y0 + margin)
    sy1 = max(sy0, y1 - 1 - margin)
    us = np.linspace(sx0, sx1, int(cols))
    vs = np.linspace(sy0, sy1, int(rows))
    return [(int(round(u)), int(round(v))) for v in vs for u in us]


def sample_depth_points(
    depth_frame: Any,
    pixels: list[tuple[int, int]],
    *,
    patch_size: int,
    max_depth_m: float,
) -> tuple[list[tuple[float, float, float]], list[str]]:
    samples: list[tuple[float, float, float]] = []
    errors: list[str] = []
    for u, v in pixels:
        try:
            depth_m = patch_depth_median(
                depth_frame,
                u,
                v,
                patch_size=patch_size,
                max_depth_m=max_depth_m,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        samples.append((float(u), float(v), depth_m))
    return samples, errors


def draw_calibration_preview(
    image: np.ndarray,
    *,
    roi_xywh: tuple[int, int, int, int] | None,
    samples: list[tuple[float, float, float]],
    plane: BeltDepthPlane | None,
    status: str,
) -> np.ndarray:
    display = image.copy()
    if roi_xywh is not None:
        x, y, width, height = roi_xywh
        cv2.rectangle(display, (x, y), (x + width, y + height), (255, 255, 0), 1)
    for u, v, _ in samples:
        cv2.circle(display, (int(round(u)), int(round(v))), 4, (0, 0, 255), -1)
    lines = [status]
    if plane is not None:
        residual = plane.residuals
        residual_text = "NA" if residual is None else f"rms={residual.rms_mm:.2f} max={residual.max_abs_mm:.2f}mm"
        lines.append(
            f"z={plane.a:.8f}u+{plane.b:.8f}v+{plane.c:.5f} samples={plane.sample_count} {residual_text}"
        )
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return display


def print_plane_summary(plane: BeltDepthPlane, output: Path) -> None:
    residual = plane.residuals
    residual_text = (
        "NA"
        if residual is None
        else (
            f"mean={residual.mean_mm:.3f}mm "
            f"rms={residual.rms_mm:.3f}mm "
            f"max_abs={residual.max_abs_mm:.3f}mm"
        )
    )
    print(
        "Saved belt depth plane "
        f"to {output}: z={plane.a:.9f}*u + {plane.b:.9f}*v + {plane.c:.6f} "
        f"samples={plane.sample_count} residuals({residual_text})",
        flush=True,
    )


def run_auto(args: argparse.Namespace) -> None:
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    try:
        color_image, depth_frame = wait_for_aligned_rgb_depth(
            pipeline,
            align,
            timeout_ms=args.frame_timeout_ms,
        )
        if color_image is None or depth_frame is None:
            raise RuntimeError("failed to capture aligned RGB/depth frame")
        pixels = grid_sample_pixels(
            args.width,
            args.height,
            tuple(args.roi) if args.roi is not None else None,
            cols=args.grid_cols,
            rows=args.grid_rows,
            margin_px=args.grid_margin_px,
        )
        samples, errors = sample_depth_points(
            depth_frame,
            pixels,
            patch_size=args.patch_size,
            max_depth_m=args.max_depth_m,
        )
        if len(samples) < args.min_samples:
            raise RuntimeError(
                f"only {len(samples)} valid depth samples, need at least {args.min_samples}; "
                f"first errors: {errors[:3]}"
            )
        plane = fit_belt_depth_plane(
            samples,
            width=args.width,
            height=args.height,
            roi_xywh=tuple(args.roi) if args.roi is not None else None,
            patch_size=args.patch_size,
        )
        save_belt_depth_plane(args.output, plane)
        print_plane_summary(plane, args.output)
        if not args.no_display:
            preview = draw_calibration_preview(
                color_image,
                roi_xywh=tuple(args.roi) if args.roi is not None else None,
                samples=samples,
                plane=plane,
                status="Auto calibration saved. Press any key to exit.",
            )
            cv2.imshow(WINDOW_NAME, preview)
            cv2.waitKey(0)
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


def run_manual(args: argparse.Namespace) -> None:
    samples: list[tuple[float, float, float]] = []
    latest_depth_frame: Any | None = None
    latest_color_image: np.ndarray | None = None
    plane: BeltDepthPlane | None = None

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        nonlocal plane
        if event != cv2.EVENT_LBUTTONDOWN or latest_depth_frame is None:
            return
        try:
            depth_m = patch_depth_median(
                latest_depth_frame,
                x,
                y,
                patch_size=args.patch_size,
                max_depth_m=args.max_depth_m,
            )
        except ValueError as exc:
            print(f"Sample rejected at ({x},{y}): {exc}", flush=True)
            return
        samples.append((float(x), float(y), depth_m))
        print(f"Sample {len(samples)}: u={x} v={y} depth={depth_m:.5f}m", flush=True)
        if len(samples) >= 3:
            plane = fit_belt_depth_plane(
                samples,
                width=args.width,
                height=args.height,
                roi_xywh=tuple(args.roi) if args.roi is not None else None,
                patch_size=args.patch_size,
            )

    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(
        "Manual calibration: click empty belt points, s=save, u=undo, q/Esc=quit.",
        flush=True,
    )
    try:
        while True:
            color_image, depth_frame = wait_for_aligned_rgb_depth(
                pipeline,
                align,
                timeout_ms=args.frame_timeout_ms,
            )
            if color_image is None or depth_frame is None:
                continue
            latest_color_image = color_image
            latest_depth_frame = depth_frame
            status = f"Manual: samples={len(samples)} click empty belt, s=save, u=undo, q=quit"
            preview = draw_calibration_preview(
                latest_color_image,
                roi_xywh=tuple(args.roi) if args.roi is not None else None,
                samples=samples,
                plane=plane,
                status=status,
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("u") and samples:
                removed = samples.pop()
                print(f"Removed sample u={removed[0]:.0f} v={removed[1]:.0f}", flush=True)
                plane = None
                if len(samples) >= 3:
                    plane = fit_belt_depth_plane(
                        samples,
                        width=args.width,
                        height=args.height,
                        roi_xywh=tuple(args.roi) if args.roi is not None else None,
                        patch_size=args.patch_size,
                    )
            if key == ord("s"):
                if len(samples) < args.min_samples:
                    print(
                        f"Need at least {args.min_samples} valid samples before saving; "
                        f"currently {len(samples)}.",
                        flush=True,
                    )
                    continue
                plane = fit_belt_depth_plane(
                    samples,
                    width=args.width,
                    height=args.height,
                    roi_xywh=tuple(args.roi) if args.roi is not None else None,
                    patch_size=args.patch_size,
                )
                save_belt_depth_plane(args.output, plane)
                print_plane_summary(plane, args.output)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.mode == "auto":
        run_auto(args)
    else:
        run_manual(args)


if __name__ == "__main__":
    main()

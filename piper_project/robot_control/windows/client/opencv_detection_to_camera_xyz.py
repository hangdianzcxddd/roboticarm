"""Manual RealSense test: OpenCV detection center -> aligned depth -> camera XYZ.

Run from ``piper_project``:

    python -m robot_control.windows.client.opencv_detection_to_camera_xyz

Default mode uses OpenCV MOG2 background subtraction. Keep the target out of the
view during warmup, then place it into the scene. Press ``r`` to relearn the
background and ``Esc`` to exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    deproject_pixel_to_camera_xyz,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)
from robot_control.windows.vision.detector import (
    BackgroundSubtractorDetector,
    ColorThresholdDetector,
    ContourDetectionConfig,
    Detection,
)


RGB_WINDOW_NAME = "OpenCV detection -> camera XYZ"
MASK_WINDOW_NAME = "OpenCV target mask"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a target with OpenCV and print RealSense camera XYZ."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=800.0)
    parser.add_argument("--max-area", type=float, default=0.0)
    parser.add_argument("--max-results", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument(
        "--algorithm",
        choices=("background", "color"),
        default="background",
        help="OpenCV detection algorithm.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=45,
        help="Background frames to learn before reporting detections.",
    )
    parser.add_argument("--history", type=int, default=120)
    parser.add_argument("--var-threshold", type=float, default=25.0)
    parser.add_argument(
        "--detect-learning-rate",
        type=float,
        default=0.0,
        help="MOG2 learning rate after warmup. 0 freezes the background.",
    )
    parser.add_argument(
        "--lower-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(35, 60, 60),
        help="Lower HSV threshold for --algorithm color.",
    )
    parser.add_argument(
        "--upper-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(85, 255, 255),
        help="Upper HSV threshold for --algorithm color.",
    )
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--hide-mask", action="store_true")
    return parser


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


def create_detector(args: argparse.Namespace):
    config = ContourDetectionConfig(
        min_area=args.min_area,
        max_area=args.max_area if args.max_area > 0.0 else None,
        max_results=args.max_results,
        morphology_kernel_size=args.kernel_size,
    )
    if args.algorithm == "background":
        return BackgroundSubtractorDetector(
            config=config,
            history=args.history,
            var_threshold=args.var_threshold,
            detect_shadows=True,
        )
    return ColorThresholdDetector(
        lower_hsv=tuple(args.lower_hsv),
        upper_hsv=tuple(args.upper_hsv),
        config=config,
    )


def draw_detections(
    image: Any,
    detections: list[Detection],
    point: Optional[CameraPoint],
    error: Optional[str],
    status: str,
) -> None:
    for detection in detections:
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            (0, 255, 0),
            2,
        )
        cv2.drawMarker(
            image,
            (detection.u, detection.v),
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            image,
            f"u={detection.u}, v={detection.v}, area={detection.area:.0f}",
            (detection.x, max(18, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    if point is not None:
        text = format_camera_point(point)
        color = (0, 255, 255)
    elif error:
        text = error
        color = (0, 0, 255)
    else:
        text = status
        color = (255, 255, 255)

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
        color,
        1,
        cv2.LINE_AA,
    )


def detection_to_camera_point(
    detection: Detection,
    depth_frame: Any,
    sample_radius: int,
) -> CameraPoint:
    return deproject_pixel_to_camera_xyz(
        depth_frame,
        detection.u,
        detection.v,
        sample_radius=sample_radius,
    )


def main() -> None:
    args = build_parser().parse_args()
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    detector = create_detector(args)
    warmup_remaining = args.warmup_frames if args.algorithm == "background" else 0
    last_print_time = 0.0
    last_point: Optional[CameraPoint] = None
    last_error: Optional[str] = None

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
                timeout_ms=args.timeout_ms,
            )
            if color_image is None or depth_frame is None:
                continue

            if args.algorithm == "background" and warmup_remaining > 0:
                detector.apply(color_image, learning_rate=-1.0)
                warmup_remaining -= 1
                detections: list[Detection] = []
                status = (
                    f"Learning background: {warmup_remaining} frames left. "
                    "Keep target out of view. Press r to relearn."
                )
                last_point = None
                last_error = None
            else:
                if args.algorithm == "background":
                    detections = detector.detect(
                        color_image,
                        learning_rate=args.detect_learning_rate,
                    )
                else:
                    detections = detector.detect(color_image)

                status = "No target found. Press r to relearn background."
                last_point = None
                last_error = None
                if detections:
                    try:
                        last_point = detection_to_camera_point(
                            detections[0],
                            depth_frame,
                            sample_radius=args.sample_radius,
                        )
                    except ValueError as exc:
                        last_error = str(exc)

                    now = time.monotonic()
                    if now - last_print_time >= args.print_interval:
                        if last_point is not None:
                            print(format_camera_point(last_point), flush=True)
                        elif last_error:
                            print(last_error, flush=True)
                        last_print_time = now

            display_image = color_image.copy()
            draw_detections(display_image, detections, last_point, last_error, status)
            cv2.imshow(RGB_WINDOW_NAME, display_image)

            if not args.hide_mask and getattr(detector, "last_mask", None) is not None:
                cv2.imshow(MASK_WINDOW_NAME, detector.last_mask)

            if not args.hide_depth:
                depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                draw_detections(depth_debug, detections, last_point, last_error, status)
                cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r") and args.algorithm == "background":
                detector.reset()
                warmup_remaining = args.warmup_frames
                last_point = None
                last_error = None
                print("Background reset. Keep target out of view.", flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

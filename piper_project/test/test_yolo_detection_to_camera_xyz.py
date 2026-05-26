"""Manual RealSense test: YOLO bbox center -> aligned depth -> camera XYZ.

Run from ``piper_project``:

    python test/test_yolo_detection_to_camera_xyz.py

The script loads ``models/yolo26n.pt`` by default. It detects objects in the
aligned RGB image, reads depth at each bbox center, and prints camera-frame XYZ.
Press ``Esc`` to exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    deproject_pixel_to_camera_xyz,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)
from vision.detector import Detection
from vision.yolo_detector import DEFAULT_MODEL_PATH, YOLOObjectDetector


RGB_WINDOW_NAME = "YOLO detection -> camera XYZ"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect objects with YOLO and print RealSense camera XYZ."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-results", type=int, default=3)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional COCO class ids to keep, e.g. --classes 39 for bottle.",
    )
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--hide-depth", action="store_true")
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


def draw_detections(
    image: Any,
    detections: list[Detection],
    points: dict[int, CameraPoint],
    errors: dict[int, str],
    status: str,
) -> None:
    for index, detection in enumerate(detections):
        color = (0, 255, 0) if index in points else (0, 0, 255)
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
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

        confidence = detection.confidence if detection.confidence is not None else 0.0
        text = (
            f"{detection.label} {confidence:.2f} "
            f"u={detection.u}, v={detection.v}"
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

    if points:
        first_point = next(iter(points.values()))
        text = format_camera_point(first_point)
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


def main() -> None:
    args = build_parser().parse_args()
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
    last_print_time = 0.0

    cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
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

            detections = detector.detect(color_image)
            points: dict[int, CameraPoint] = {}
            errors: dict[int, str] = {}
            for index, detection in enumerate(detections):
                try:
                    points[index] = detection_to_camera_point(
                        detection,
                        depth_frame,
                        sample_radius=args.sample_radius,
                    )
                except ValueError as exc:
                    errors[index] = str(exc)

            now = time.monotonic()
            if now - last_print_time >= args.print_interval:
                if points:
                    for index, point in points.items():
                        detection = detections[index]
                        print(
                            f"{detection.label}: {format_camera_point(point)}",
                            flush=True,
                        )
                elif errors:
                    print(next(iter(errors.values())), flush=True)
                last_print_time = now

            status = "No YOLO target found."
            display_image = color_image.copy()
            draw_detections(display_image, detections, points, errors, status)
            cv2.imshow(RGB_WINDOW_NAME, display_image)

            if not args.hide_depth:
                depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                draw_detections(depth_debug, detections, points, errors, status)
                cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

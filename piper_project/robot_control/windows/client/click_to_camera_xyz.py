"""Manual RealSense click test: RGB pixel -> aligned depth -> camera XYZ.

Run from ``piper_project``:

    python -m robot_control.windows.client.click_to_camera_xyz

Left-click the RGB image to print camera-frame X/Y/Z in meters. Press Esc to
exit. This is a manual hardware test, not an automated unit test.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    deproject_pixel_to_camera_xyz,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)


WINDOW_NAME = "RealSense aligned RGB click -> camera XYZ"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


class ClickToXYZPrinter:
    def __init__(self, sample_radius: int) -> None:
        self.sample_radius = sample_radius
        self.depth_frame: Optional[Any] = None
        self.last_point: Optional[CameraPoint] = None
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
            self.last_error = "No depth frame is available yet."
            print(self.last_error, flush=True)
            return

        try:
            point = deproject_pixel_to_camera_xyz(
                self.depth_frame,
                u,
                v,
                sample_radius=self.sample_radius,
            )
        except ValueError as exc:
            self.last_error = str(exc)
            self.last_point = None
            print(self.last_error, flush=True)
            return

        self.last_point = point
        self.last_error = None
        print(format_camera_point(point), flush=True)

    def last_click_pixel(self) -> Optional[Tuple[int, int]]:
        if self.last_point is None:
            return self.last_click
        return self.last_point.u, self.last_point.v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Click an aligned RealSense RGB image and print camera XYZ."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--sample-radius",
        type=int,
        default=3,
        help=(
            "Use median valid depth in a square neighborhood around the click. "
            "Default 3 is more stable than reading one pixel. Use 0 for exact pixel depth."
        ),
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=1.5,
        help="Maximum depth shown in the debug depth colormap.",
    )
    parser.add_argument(
        "--hide-depth",
        action="store_true",
        help="Do not show the aligned depth debug window.",
    )
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


def draw_status(image: Any, printer: ClickToXYZPrinter) -> None:
    click_pixel = printer.last_click_pixel()
    if click_pixel is not None:
        cv2.drawMarker(
            image,
            click_pixel,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if printer.last_point is not None:
        text = format_camera_point(printer.last_point)
        color = (0, 255, 255)
    elif printer.last_error:
        text = printer.last_error
        color = (0, 0, 255)
    else:
        text = "Left-click a point. Press Esc to exit."
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


def main() -> None:
    args = build_parser().parse_args()
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    printer = ClickToXYZPrinter(sample_radius=args.sample_radius)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, printer.on_mouse)
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

            printer.update_depth_frame(depth_frame)
            display_image = color_image.copy()
            draw_status(display_image, printer)
            cv2.imshow(WINDOW_NAME, display_image)
            if not args.hide_depth:
                depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                draw_status(depth_debug, printer)
                cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

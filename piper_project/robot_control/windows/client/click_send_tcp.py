"""Windows-side RealSense click test: camera XYZ -> TCP motion server.

Run on Windows from ``piper_project``:

    python -m robot_control.windows.client.click_send_tcp --host <linux-vm-ip> --port 5005

Left-click the aligned RGB image to send the camera XYZ point to the Linux-side
``python -m robot_control.linux.server.tcp_server``. Press ``Esc`` to exit.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from robot_control.shared.protocol import CameraPointCommand, send_camera_point
from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    deproject_pixel_to_camera_xyz,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)


WINDOW_NAME = "Click camera XYZ -> TCP motion server"
DEPTH_WINDOW_NAME = "Aligned depth valid map"


class ClickTcpSender:
    def __init__(
        self,
        host: str,
        port: int,
        sample_radius: int,
        timeout_s: float,
    ) -> None:
        self.host = host
        self.port = port
        self.sample_radius = sample_radius
        self.timeout_s = timeout_s
        self.depth_frame: Optional[Any] = None
        self.last_point: Optional[CameraPoint] = None
        self.last_error: Optional[str] = None
        self.last_response: Optional[str] = None
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
            )
            response = send_camera_point(
                self.host,
                self.port,
                CameraPointCommand(
                    x_m=point.x_m,
                    y_m=point.y_m,
                    z_m=point.z_m,
                    u=point.u,
                    v=point.v,
                    depth_m=point.depth_m,
                    source="manual_click",
                ),
                timeout_s=self.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - displayed in manual test UI
            self.last_point = None
            self._set_error(str(exc))
            return

        self.last_point = point
        self.last_error = None
        self.last_response = str(response)
        print(format_camera_point(point), flush=True)
        print(response, flush=True)

    def _set_error(self, error: str) -> None:
        self.last_error = error
        self.last_response = None
        print(error, flush=True)

    def last_click_pixel(self) -> Optional[Tuple[int, int]]:
        if self.last_point is None:
            return self.last_click
        return self.last_point.u, self.last_point.v


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Click a RealSense image point and send camera XYZ over TCP."
    )
    parser.add_argument("--host", required=True, help="Linux VM IP address")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=3)
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


def draw_status(image: Any, sender: ClickTcpSender) -> None:
    click_pixel = sender.last_click_pixel()
    if click_pixel is not None:
        cv2.drawMarker(
            image,
            click_pixel,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

    if sender.last_point is not None:
        text = format_camera_point(sender.last_point)
        color = (0, 255, 255)
    elif sender.last_error:
        text = sender.last_error
        color = (0, 0, 255)
    else:
        text = "Left-click to send camera XYZ. Press Esc to exit."
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
    sender = ClickTcpSender(
        host=args.host,
        port=args.port,
        sample_radius=args.sample_radius,
        timeout_s=args.timeout_s,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, sender.on_mouse)
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

            sender.update_depth_frame(depth_frame)
            display_image = color_image.copy()
            draw_status(display_image, sender)
            cv2.imshow(WINDOW_NAME, display_image)
            if not args.hide_depth:
                depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                draw_status(depth_debug, sender)
                cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Manual RealSense RGB preview.

Run on Windows from the project root:

    python -m robot_control.windows.client.realsense_rgb_preview

This is the replacement for the old ``test/test_camera.py`` script. It opens
the RealSense RGB stream and displays a live OpenCV window. Press Esc to exit.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


WINDOW_NAME = "RealSense RGB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview RealSense RGB stream.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for RealSense preview.") from exc

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )

    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames(10000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow(WINDOW_NAME, color_image)
            if cv2.waitKey(1) == 27:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

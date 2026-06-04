"""OpenCV visual grasp client: detect target center and send camera XYZ.

Run on Windows from ``piper_project`` after starting the Linux catch server:

    python -m robot_control.windows.client.opencv_grasp_send_tcp --host <linux-vm-ip>

The Linux side should run ``robot_control.linux.catch.catch_server`` with
``--hand-eye-result`` so the camera XYZ sent here is transformed by the current
camera->base hand-eye matrix before motion planning.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from robot_control.shared.protocol import CameraPointCommand, send_camera_point
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
from robot_control.windows.vision.detector import (
    BackgroundSubtractorDetector,
    ColorThresholdDetector,
    ContourDetectionConfig,
    Detection,
)


RGB_WINDOW_NAME = "OpenCV visual grasp -> catch_server"
MASK_WINDOW_NAME = "OpenCV grasp mask"
DEPTH_WINDOW_NAME = "Aligned depth valid map"
SendCameraPointFunc = Callable[[str, int, CameraPointCommand, float], dict[str, Any]]


@dataclass(frozen=True)
class SelectedTarget:
    detection: Detection
    point: CameraPoint


class DetectionStability:
    """Require a detection center to stay near one pixel location for N frames."""

    def __init__(self, required_frames: int = 5, max_center_shift_px: float = 20.0) -> None:
        self.required_frames = max(1, int(required_frames))
        self.max_center_shift_px = float(max_center_shift_px)
        self._last_center: tuple[int, int] | None = None
        self._stable_frames = 0

    @property
    def stable_frames(self) -> int:
        return self._stable_frames

    def reset(self) -> None:
        self._last_center = None
        self._stable_frames = 0

    def update(self, detection: Detection | None) -> bool:
        if detection is None:
            self.reset()
            return False

        center = (int(detection.u), int(detection.v))
        if self._last_center is None:
            self._last_center = center
            self._stable_frames = 1
            return self.is_stable

        shift = float(np.hypot(center[0] - self._last_center[0], center[1] - self._last_center[1]))
        if shift <= self.max_center_shift_px:
            self._stable_frames += 1
        else:
            self._stable_frames = 1
        self._last_center = center
        return self.is_stable

    @property
    def is_stable(self) -> bool:
        return self._stable_frames >= self.required_frames


class OpenCvGraspSender:
    def __init__(
        self,
        host: str | None,
        port: int,
        timeout_s: float,
        dry_run: bool = False,
        send_func: SendCameraPointFunc = send_camera_point,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.dry_run = bool(dry_run)
        self._send_func = send_func

    def send_target(
        self,
        target: SelectedTarget,
        source: str,
    ) -> dict[str, Any]:
        command = camera_point_command_from_target(target, source=source)
        if self.dry_run:
            return {
                "type": "dry_run",
                "payload": {
                    "source": command.source,
                    "u": command.u,
                    "v": command.v,
                    "x_m": command.x_m,
                    "y_m": command.y_m,
                    "z_m": command.z_m,
                    "depth_m": command.depth_m,
                },
            }
        if not self.host:
            raise ValueError("--host is required unless --dry-run is used")
        return self._send_func(self.host, self.port, command, self.timeout_s)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect one or more objects with OpenCV, choose a target center, "
            "convert it to RealSense camera XYZ, and send it to Linux catch_server."
        )
    )
    parser.add_argument("--host", help="Linux VM IP address running catch_server")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="print target XYZ without sending")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--sample-radius", type=int, default=5)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument(
        "--intrinsics-json",
        type=Path,
        default=DEFAULT_D435I_PARA,
        help="D435i parameter JSON used to select rectified fx/fy/ppx/ppy by width/height",
    )
    parser.add_argument(
        "--use-realsense-intrinsics",
        action="store_true",
        help="ignore --intrinsics-json and use intrinsics reported by the RealSense frame",
    )

    parser.add_argument(
        "--algorithm",
        choices=("background", "color"),
        default="background",
        help="OpenCV detection algorithm",
    )
    parser.add_argument("--min-area", type=float, default=800.0)
    parser.add_argument("--max-area", type=float, default=0.0)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--history", type=int, default=120)
    parser.add_argument("--var-threshold", type=float, default=25.0)
    parser.add_argument(
        "--detect-learning-rate",
        type=float,
        default=0.0,
        help="MOG2 learning rate after warmup; 0 freezes the learned background",
    )
    parser.add_argument(
        "--lower-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(35, 60, 60),
        help="Lower HSV threshold for --algorithm color",
    )
    parser.add_argument(
        "--upper-hsv",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(85, 255, 255),
        help="Upper HSV threshold for --algorithm color",
    )

    parser.add_argument(
        "--target-policy",
        choices=("largest", "center", "left-to-right", "right-to-left"),
        default="largest",
        help="simple target selection strategy when multiple boxes are detected",
    )
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--max-center-shift-px", type=float, default=20.0)
    parser.add_argument("--auto-send", action="store_true")
    parser.add_argument("--send-cooldown-s", type=float, default=12.0)
    parser.add_argument(
        "--max-sends",
        type=int,
        default=1,
        help="maximum automatic sends; 0 means unlimited",
    )
    parser.add_argument("--exit-after-send", action="store_true")
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--hide-mask", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.host:
        raise ValueError("--host is required unless --dry-run is used")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.sample_radius < 0:
        raise ValueError("--sample-radius must be >= 0")
    if args.min_area < 0:
        raise ValueError("--min-area must be >= 0")
    if args.max_area < 0:
        raise ValueError("--max-area must be >= 0")
    if args.max_results < 1:
        raise ValueError("--max-results must be >= 1")
    if args.stable_frames < 1:
        raise ValueError("--stable-frames must be >= 1")
    if args.max_center_shift_px < 0:
        raise ValueError("--max-center-shift-px must be >= 0")
    if args.send_cooldown_s < 0:
        raise ValueError("--send-cooldown-s must be >= 0")
    if args.max_sends < 0:
        raise ValueError("--max-sends must be >= 0")
    if args.no_display and not (args.auto_send and args.exit_after_send):
        raise ValueError(
            "--no-display requires --auto-send --exit-after-send; otherwise "
            "there is no keyboard UI to stop the loop"
        )


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


def select_detection(
    detections: list[Detection],
    frame_shape: tuple[int, ...],
    policy: str,
) -> Detection | None:
    if not detections:
        return None
    if policy == "largest":
        return max(detections, key=lambda item: item.area)
    if policy == "left-to-right":
        return min(detections, key=lambda item: (item.u, -item.area))
    if policy == "right-to-left":
        return max(detections, key=lambda item: (item.u, item.area))
    if policy == "center":
        height, width = int(frame_shape[0]), int(frame_shape[1])
        center_u = width / 2.0
        center_v = height / 2.0
        return min(
            detections,
            key=lambda item: (
                (item.u - center_u) ** 2 + (item.v - center_v) ** 2,
                -item.area,
            ),
        )
    raise ValueError(f"unknown target policy: {policy}")


def detection_to_camera_point(
    detection: Detection,
    depth_frame: Any,
    sample_radius: int,
    intrinsics: PinholeIntrinsics | None,
) -> CameraPoint:
    return deproject_pixel_to_camera_xyz(
        depth_frame,
        detection.u,
        detection.v,
        sample_radius=sample_radius,
        intrinsics=intrinsics,
    )


def camera_point_command_from_target(
    target: SelectedTarget,
    source: str,
) -> CameraPointCommand:
    return CameraPointCommand(
        x_m=target.point.x_m,
        y_m=target.point.y_m,
        z_m=target.point.z_m,
        u=target.point.u,
        v=target.point.v,
        depth_m=target.point.depth_m,
        source=source,
    )


def can_auto_send(
    *,
    auto_send: bool,
    is_stable: bool,
    sent_count: int,
    max_sends: int,
    last_send_time: float | None,
    cooldown_s: float,
    now: float,
) -> bool:
    if not auto_send or not is_stable:
        return False
    if max_sends > 0 and sent_count >= max_sends:
        return False
    if last_send_time is None:
        return True
    return now - last_send_time >= cooldown_s


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


def draw_detections(
    image: Any,
    detections: list[Detection],
    selected: Detection | None,
    target: SelectedTarget | None,
    error: str | None,
    status: str,
) -> None:
    for detection in detections:
        is_selected = selected is not None and detection == selected
        color = (0, 255, 255) if is_selected else (0, 255, 0)
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
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.putText(
            image,
            f"{detection.label} u={detection.u}, v={detection.v}, area={detection.area:.0f}",
            (detection.x, max(18, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if target is not None:
        text = format_camera_point(target.point)
        text_color = (0, 255, 255)
    elif error:
        text = error
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


def send_and_print(
    sender: OpenCvGraspSender,
    target: SelectedTarget,
    source: str,
) -> dict[str, Any]:
    response = sender.send_target(target, source=source)
    print(format_camera_point(target.point), flush=True)
    print(response, flush=True)
    return response


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

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

    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    detector = create_detector(args)
    sender = OpenCvGraspSender(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    stability = DetectionStability(
        required_frames=args.stable_frames,
        max_center_shift_px=args.max_center_shift_px,
    )
    warmup_remaining = args.warmup_frames if args.algorithm == "background" else 0
    last_send_time: float | None = None
    sent_count = 0
    source = f"opencv_{args.algorithm}"

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

            detections: list[Detection] = []
            selected: Detection | None = None
            target: SelectedTarget | None = None
            error: str | None = None

            if args.algorithm == "background" and warmup_remaining > 0:
                detector.apply(color_image, learning_rate=-1.0)
                warmup_remaining -= 1
                stability.reset()
                status = (
                    f"Learning background: {warmup_remaining} frames left. "
                    "Keep target out of view."
                )
            else:
                if args.algorithm == "background":
                    detections = detector.detect(
                        color_image,
                        learning_rate=args.detect_learning_rate,
                    )
                else:
                    detections = detector.detect(color_image)

                selected = select_detection(
                    detections,
                    color_image.shape,
                    policy=args.target_policy,
                )
                is_stable = stability.update(selected)
                status = (
                    f"targets={len(detections)} stable={stability.stable_frames}/"
                    f"{args.stable_frames}. Press g to grasp, r to relearn."
                )
                if selected is not None:
                    try:
                        point = detection_to_camera_point(
                            selected,
                            depth_frame,
                            sample_radius=args.sample_radius,
                            intrinsics=intrinsics,
                        )
                        target = SelectedTarget(selected, point)
                    except ValueError as exc:
                        error = str(exc)

                now = time.monotonic()
                if target is not None and can_auto_send(
                    auto_send=args.auto_send,
                    is_stable=is_stable,
                    sent_count=sent_count,
                    max_sends=args.max_sends,
                    last_send_time=last_send_time,
                    cooldown_s=args.send_cooldown_s,
                    now=now,
                ):
                    send_and_print(sender, target, source=source)
                    last_send_time = now
                    sent_count += 1
                    if args.exit_after_send:
                        break

            if not args.no_display:
                display_image = color_image.copy()
                draw_detections(display_image, detections, selected, target, error, status)
                cv2.imshow(RGB_WINDOW_NAME, display_image)

                if not args.hide_mask and getattr(detector, "last_mask", None) is not None:
                    cv2.imshow(MASK_WINDOW_NAME, detector.last_mask)

                if not args.hide_depth:
                    depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                    draw_detections(depth_debug, detections, selected, target, error, status)
                    cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                if key == ord("r") and args.algorithm == "background":
                    detector.reset()
                    stability.reset()
                    warmup_remaining = args.warmup_frames
                    print("Background reset. Keep target out of view.", flush=True)
                if key == ord("g") and target is not None:
                    send_and_print(sender, target, source=source)
                    last_send_time = time.monotonic()
                    sent_count += 1
                    if args.exit_after_send:
                        break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

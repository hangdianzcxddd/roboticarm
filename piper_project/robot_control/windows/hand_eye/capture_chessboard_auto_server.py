"""Serve live chessboard detections for automatic hand-eye sample collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from robot_control.shared.hand_eye_auto_protocol import (
    AutoHandEyeDetection,
    AutoHandEyeRequest,
    auto_hand_eye_request_from_message,
    make_auto_hand_eye_error_message,
    make_auto_hand_eye_response_message,
    recv_json_line,
    send_json_line,
)
from robot_control.windows.hand_eye.capture_chessboard import (
    build_intrinsics,
    format_intrinsics,
)
from robot_control.windows.hand_eye.chessboard import (
    ChessboardConfig,
    ChessboardDetection,
    DEFAULT_INTRINSICS,
    CameraIntrinsics,
    draw_detection,
    estimate_chessboard_pose,
    put_status_text,
)


WINDOW_NAME = "Auto hand-eye chessboard vision server"
DEFAULT_IMAGE_DIR = Path(__file__).with_name("captures_auto")


@dataclass(frozen=True)
class DetectionMetrics:
    detection: ChessboardDetection
    margin_px: float
    center_u: float
    center_v: float
    width_px: float
    height_px: float


@dataclass(frozen=True)
class StableDetectionState:
    metrics: DetectionMetrics
    stable_count: int


class LatestFrameState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.detection: ChessboardDetection | None = None
        self.stable_state: StableDetectionState | None = None
        self.message = "waiting for frames"

    def update(
        self,
        frame: np.ndarray,
        detection: ChessboardDetection | None,
        stable_state: StableDetectionState | None,
        message: str,
    ) -> None:
        with self._lock:
            self.frame = frame.copy()
            self.detection = detection
            self.stable_state = stable_state
            self.message = message

    def snapshot(
        self,
    ) -> tuple[
        np.ndarray | None,
        ChessboardDetection | None,
        StableDetectionState | None,
        str,
    ]:
        with self._lock:
            return (
                None if self.frame is None else self.frame.copy(),
                self.detection,
                self.stable_state,
                self.message,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Windows RealSense chessboard detector as a request/response "
            "service for Linux automatic hand-eye collection."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--intrinsic-profile",
        choices=tuple(sorted(DEFAULT_INTRINSICS)),
        default=None,
        help="preloaded RealSense rectified color intrinsics profile",
    )
    parser.add_argument("--fx", type=float, help="override camera fx")
    parser.add_argument("--fy", type=float, help="override camera fy")
    parser.add_argument("--cx", type=float, help="override camera principal point x")
    parser.add_argument("--cy", type=float, help="override camera principal point y")
    parser.add_argument(
        "--dist-coeffs",
        type=float,
        nargs="+",
        help="OpenCV distortion coefficients; omitted means zero distortion",
    )
    parser.add_argument(
        "--use-live-distortion",
        action="store_true",
        help="use distortion coefficients reported by the active RealSense color profile",
    )
    parser.add_argument("--corners-cols", type=int, default=11)
    parser.add_argument("--corners-rows", type=int, default=8)
    parser.add_argument("--checker-size-mm", type=float, default=15.0)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--no-save-image",
        action="store_true",
        help="ignore capture requests that ask to save images",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live chessboard preview window",
    )
    parser.add_argument(
        "--stability-max-center-shift-px",
        type=float,
        default=3.0,
        help="maximum frame-to-frame chessboard center shift counted as stable",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("pyrealsense2 is required for RealSense capture.") from exc

    board_config = ChessboardConfig(
        corners_cols=args.corners_cols,
        corners_rows=args.corners_rows,
        checker_size_m=args.checker_size_mm / 1000.0,
    )
    if not args.no_save_image:
        args.image_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    profile = pipeline.start(config)
    intrinsics = build_intrinsics(args, profile, rs.stream.color)
    print(format_intrinsics(intrinsics), flush=True)

    state = LatestFrameState()
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=serve,
        args=(args, state, board_config, stop_event),
        daemon=True,
    )
    server_thread.start()

    if args.display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    previous_metrics: DetectionMetrics | None = None
    stable_count = 0
    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames(10000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            detection = estimate_chessboard_pose(frame, board_config, intrinsics)
            stable_state = None
            if detection is None:
                previous_metrics = None
                stable_count = 0
                message = "chessboard: not detected"
            else:
                metrics = detection_metrics(detection, intrinsics)
                if (
                    previous_metrics is not None
                    and center_shift_px(metrics, previous_metrics)
                    <= args.stability_max_center_shift_px
                ):
                    stable_count += 1
                else:
                    stable_count = 1
                previous_metrics = metrics
                stable_state = StableDetectionState(metrics, stable_count)
                message = (
                    "chessboard: detected "
                    f"reproj={detection.reprojection_error_px:.3f}px "
                    f"margin={metrics.margin_px:.1f}px "
                    f"stable={stable_count}"
                )
            state.update(frame, detection, stable_state, message)

            if args.display:
                display = draw_detection(frame, board_config, intrinsics, detection)
                display = put_status_text(
                    display,
                    [
                        f"server {args.host}:{args.port}",
                        message,
                        "Esc closes the vision server.",
                    ],
                )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    stop_event.set()
                    break
    finally:
        stop_event.set()
        pipeline.stop()
        cv2.destroyAllWindows()


def validate_args(args: argparse.Namespace) -> None:
    if args.port <= 0:
        raise SystemExit("--port must be > 0")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.corners_cols < 2 or args.corners_rows < 2:
        raise SystemExit("--corners-cols and --corners-rows must be >= 2")
    if args.checker_size_mm <= 0:
        raise SystemExit("--checker-size-mm must be > 0")
    if args.stability_max_center_shift_px < 0:
        raise SystemExit("--stability-max-center-shift-px must be >= 0")


def serve(
    args: argparse.Namespace,
    state: LatestFrameState,
    board_config: ChessboardConfig,
    stop_event: threading.Event,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        server.settimeout(0.5)
        print(
            f"Auto hand-eye vision server listening on {args.host}:{args.port}",
            flush=True,
        )
        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(args.timeout_s)
                try:
                    with conn.makefile("rb") as sock_file:
                        message = recv_json_line(sock_file)
                    if message is None:
                        continue
                    request = auto_hand_eye_request_from_message(message)
                    detection = handle_request(args, state, board_config, request)
                    response = make_auto_hand_eye_response_message(detection)
                except Exception as exc:  # noqa: BLE001 - return error to Linux
                    print(f"Vision request failed from {addr}: {exc}", file=sys.stderr)
                    response = make_auto_hand_eye_error_message(str(exc))
                send_json_line(conn, response)


def handle_request(
    args: argparse.Namespace,
    state: LatestFrameState,
    board_config: ChessboardConfig,
    request: AutoHandEyeRequest,
) -> AutoHandEyeDetection:
    frame, detection, stable_state, message = state.snapshot()
    del message
    if frame is None or detection is None or stable_state is None:
        return AutoHandEyeDetection(
            detected=False,
            qualified=False,
            reason="chessboard not detected",
            required_corners=board_config.corners_cols * board_config.corners_rows,
        )
    quality = qualify_detection(stable_state, request, board_config)
    image_path = None
    if (
        request.action == "capture"
        and quality.qualified
        and request.save_image
        and not args.no_save_image
    ):
        image_path = save_image(args.image_dir, frame, request.sample_id)
    return AutoHandEyeDetection(
        detected=True,
        qualified=quality.qualified,
        reason=quality.reason,
        target_to_camera=detection.pose,
        reprojection_error_px=detection.reprojection_error_px,
        detected_corners=board_config.corners_cols * board_config.corners_rows,
        required_corners=board_config.corners_cols * board_config.corners_rows,
        margin_px=stable_state.metrics.margin_px,
        center_u=stable_state.metrics.center_u,
        center_v=stable_state.metrics.center_v,
        width_px=stable_state.metrics.width_px,
        height_px=stable_state.metrics.height_px,
        stable_count=stable_state.stable_count,
        image_path=str(image_path) if image_path is not None else None,
    )


@dataclass(frozen=True)
class QualityResult:
    qualified: bool
    reason: str | None


def qualify_detection(
    stable_state: StableDetectionState,
    request: AutoHandEyeRequest,
    board_config: ChessboardConfig,
) -> QualityResult:
    detection = stable_state.metrics.detection
    required = board_config.corners_cols * board_config.corners_rows
    if detection.corners.shape[0] != required:
        return QualityResult(False, "not all chessboard corners detected")
    if stable_state.metrics.margin_px < request.min_margin_px:
        return QualityResult(
            False,
            (
                f"corner margin {stable_state.metrics.margin_px:.1f}px below "
                f"{request.min_margin_px:.1f}px"
            ),
        )
    if detection.reprojection_error_px > request.max_reprojection_error_px:
        return QualityResult(
            False,
            (
                f"reprojection {detection.reprojection_error_px:.3f}px exceeds "
                f"{request.max_reprojection_error_px:.3f}px"
            ),
        )
    if stable_state.stable_count < request.stable_frames:
        return QualityResult(
            False,
            f"stable frames {stable_state.stable_count} below {request.stable_frames}",
        )
    return QualityResult(True, None)


def detection_metrics(
    detection: ChessboardDetection,
    intrinsics: CameraIntrinsics,
) -> DetectionMetrics:
    points = detection.corners.reshape(-1, 2)
    min_u = float(np.min(points[:, 0]))
    max_u = float(np.max(points[:, 0]))
    min_v = float(np.min(points[:, 1]))
    max_v = float(np.max(points[:, 1]))
    margin = min(
        min_u,
        min_v,
        float(intrinsics.width - 1) - max_u,
        float(intrinsics.height - 1) - max_v,
    )
    return DetectionMetrics(
        detection=detection,
        margin_px=float(margin),
        center_u=float(np.mean(points[:, 0])),
        center_v=float(np.mean(points[:, 1])),
        width_px=float(max_u - min_u),
        height_px=float(max_v - min_v),
    )


def center_shift_px(current: DetectionMetrics, previous: DetectionMetrics) -> float:
    return float(
        np.hypot(
            current.center_u - previous.center_u,
            current.center_v - previous.center_v,
        )
    )


def save_image(image_dir: Path, frame: np.ndarray, sample_id: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_id = "sample" if sample_id is None else sanitize_filename(sample_id)
    path = image_dir / f"{safe_id}_{timestamp}.png"
    cv2.imwrite(str(path), frame)
    return path


def sanitize_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


if __name__ == "__main__":
    main()

"""Capture chessboard hand-eye samples on the Windows camera host."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from robot_control.shared.hand_eye_protocol import (
    HandEyeSampleCommand,
    send_hand_eye_sample,
)
from robot_control.windows.hand_eye.chessboard import (
    ChessboardConfig,
    DEFAULT_INTRINSICS,
    draw_detection,
    estimate_chessboard_pose,
    intrinsics_from_profile,
    intrinsics_from_realsense_profile,
    make_intrinsics,
    put_status_text,
)


WINDOW_NAME = "Hand-eye chessboard capture"
DEFAULT_IMAGE_DIR = Path(__file__).with_name("captures")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect an 8x11 inner-corner chessboard, estimate target->camera "
            "pose, and send samples to the Linux hand-eye server."
        )
    )
    parser.add_argument("--host", required=True, help="Linux host/IP")
    parser.add_argument("--port", type=int, default=5010)
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
        help="do not write captured images to disk",
    )
    parser.add_argument(
        "--reprojection-warning-px",
        type=float,
        default=1.5,
        help="print a warning above this reprojection RMS error",
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

    sample_count = 0
    last_message = "Press c to capture, Esc to exit."
    profile = pipeline.start(config)
    intrinsics = build_intrinsics(args, profile, rs.stream.color)
    print(format_intrinsics(intrinsics), flush=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frames = pipeline.wait_for_frames(10000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            detection = estimate_chessboard_pose(frame, board_config, intrinsics)
            display = draw_detection(frame, board_config, intrinsics, detection)
            status_lines = [last_message]
            if detection is None:
                status_lines.append("chessboard: not detected")
            else:
                status_lines.append(
                    "chessboard: detected "
                    f"reproj={detection.reprojection_error_px:.3f}px "
                    f"t={format_tvec(detection.pose.tvec_m)} m"
                )
            display = put_status_text(display, status_lines)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("c"), ord("C")):
                if detection is None:
                    last_message = "Capture skipped: chessboard not detected."
                    print(last_message, flush=True)
                    continue
                sample_count += 1
                image_path = save_image(args, frame, sample_count)
                command = HandEyeSampleCommand(
                    target_to_camera=detection.pose,
                    sample_id=f"sample-{sample_count:03d}",
                    image_path=str(image_path) if image_path is not None else None,
                    reprojection_error_px=detection.reprojection_error_px,
                    detected_corners=args.corners_cols * args.corners_rows,
                )
                if detection.reprojection_error_px > args.reprojection_warning_px:
                    print(
                        "warning: high chessboard reprojection error "
                        f"{detection.reprojection_error_px:.3f}px",
                        flush=True,
                    )
                response = send_hand_eye_sample(
                    args.host,
                    args.port,
                    command,
                    timeout_s=args.timeout_s,
                )
                last_message = format_response(response, sample_count)
                print(last_message, flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.corners_cols < 2 or args.corners_rows < 2:
        raise SystemExit("--corners-cols and --corners-rows must be >= 2")
    if args.checker_size_mm <= 0:
        raise SystemExit("--checker-size-mm must be > 0")


def build_intrinsics(args: argparse.Namespace, pipeline_profile=None, color_stream=None):
    dist_coeffs = tuple(args.dist_coeffs) if args.dist_coeffs is not None else None
    if any(value is not None for value in (args.fx, args.fy, args.cx, args.cy)):
        if any(value is None for value in (args.fx, args.fy, args.cx, args.cy)):
            raise SystemExit("--fx, --fy, --cx, and --cy must be supplied together")
        return make_intrinsics(
            width=args.width,
            height=args.height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            dist_coeffs=dist_coeffs,
        )
    if args.intrinsic_profile is not None:
        intrinsics = intrinsics_from_profile(args.intrinsic_profile, dist_coeffs=dist_coeffs)
        if intrinsics.width != args.width or intrinsics.height != args.height:
            raise SystemExit(
                f"--intrinsic-profile {args.intrinsic_profile} is "
                f"{intrinsics.width}x{intrinsics.height}; match --width/--height "
                "or provide --fx --fy --cx --cy."
            )
        return intrinsics

    if pipeline_profile is None:
        raise SystemExit("live RealSense intrinsics require an active pipeline profile")
    if color_stream is None:
        raise SystemExit("live RealSense intrinsics require rs.stream.color")
    color_profile = pipeline_profile.get_stream(color_stream).as_video_stream_profile()
    intrinsics = intrinsics_from_realsense_profile(
        color_profile,
        dist_coeffs=dist_coeffs,
        use_live_distortion=args.use_live_distortion,
    )
    if intrinsics.width != args.width or intrinsics.height != args.height:
        raise SystemExit(
            f"live RealSense color intrinsics are {intrinsics.width}x{intrinsics.height}; "
            "match --width/--height or provide --fx --fy --cx --cy."
        )
    return intrinsics


def format_intrinsics(intrinsics) -> str:
    matrix = intrinsics.camera_matrix
    coeffs = ", ".join(f"{float(value):.6g}" for value in intrinsics.dist_coeffs.reshape(-1))
    return (
        "Using camera intrinsics: "
        f"{intrinsics.width}x{intrinsics.height} "
        f"fx={matrix[0, 0]:.6f} fy={matrix[1, 1]:.6f} "
        f"cx={matrix[0, 2]:.6f} cy={matrix[1, 2]:.6f} "
        f"dist=[{coeffs}]"
    )


def save_image(args: argparse.Namespace, frame: np.ndarray, sample_count: int) -> Path | None:
    if args.no_save_image:
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = args.image_dir / f"hand_eye_{sample_count:03d}_{timestamp}.png"
    cv2.imwrite(str(path), frame)
    return path


def format_tvec(tvec_m: tuple[float, float, float]) -> str:
    return f"({tvec_m[0]:.4f}, {tvec_m[1]:.4f}, {tvec_m[2]:.4f})"


def format_response(response: dict, local_sample_count: int) -> str:
    if response.get("type") == "hand_eye_error":
        payload = response.get("payload", {})
        return f"Linux rejected sample {local_sample_count}: {payload.get('error')}"
    payload = response.get("payload", {})
    return (
        f"Saved sample {payload.get('sample_index')} on Linux "
        f"(local capture {local_sample_count})."
    )


if __name__ == "__main__":
    main()

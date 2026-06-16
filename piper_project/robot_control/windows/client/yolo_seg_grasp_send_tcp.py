"""YOLO-seg visual grasp client: mask -> centroid -> median depth -> TCP send.

Run on Windows from ``piper_project`` after starting the Linux catch server:

    python -m robot_control.windows.client.yolo_seg_grasp_send_tcp --host <linux-vm-ip>

The script loads ``models/yolo26n-seg.pt`` by default. For each segmentation
instance it computes the binary mask centroid, samples the median valid depth
inside the mask, deprojects the centroid ray with that median depth, ranks
targets by depth from near to far, and sends the selected point to
``robot_control.linux.catch.catch_server_1`` using the existing JSON-lines
camera point protocol.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from robot_control.linux.catch.hand_eye_mapping import load_camera_to_base_transform
from robot_control.shared.paths import MODELS_DIR, PROJECT_DIR
from robot_control.shared.protocol import (
    CameraPointCommand,
    CameraPointTask,
    send_camera_point,
)
from robot_control.windows.client.yolo_grasp_send_tcp import (
    apply_base_z_offset,
    make_depth_colormap,
    offset_camera_point,
    parse_target_command,
)
from robot_control.windows.vision.d435i_intrinsics import (
    DEFAULT_D435I_PARA,
    PinholeIntrinsics,
    find_d435i_intrinsics,
    format_intrinsics,
)
from robot_control.windows.vision.depth import (
    CameraPoint,
    create_aligned_rgb_depth_stream,
    format_camera_point,
    wait_for_aligned_rgb_depth,
)
from robot_control.windows.vision.detector import Detection
from robot_control.windows.vision.yolo_detector import (
    configure_ultralytics_config_dir,
    require_yolo_model,
)


DEFAULT_SEG_MODEL_PATH = MODELS_DIR / "yolo26n-seg.pt"
DEFAULT_HAND_EYE_RESULT = (
    PROJECT_DIR / "robot_control" / "linux" / "hand_eye" / "hand_eye_result.json"
)
RGB_WINDOW_NAME = "YOLO-seg mask grasp -> catch_server_1"
MASK_WINDOW_NAME = "YOLO-seg masks"
DEPTH_WINDOW_NAME = "Aligned depth valid map"
SendCameraPointFunc = Callable[[str, int, CameraPointCommand, float], dict[str, Any]]


@dataclass(frozen=True)
class SegmentationDetection:
    detection: Detection
    mask: np.ndarray
    mask_area_px: int
    centroid_u: float
    centroid_v: float


@dataclass(frozen=True)
class MaskDepthStats:
    median_m: float
    valid_count: int


@dataclass(frozen=True)
class RankedSegTarget:
    rank: int
    segment: SegmentationDetection
    raw_point: CameraPoint
    send_point: CameraPoint
    mask_depth_valid_count: int

    @property
    def detection(self) -> Detection:
        return self.segment.detection

    @property
    def depth_mm(self) -> float:
        return self.raw_point.depth_m * 1000.0


class TerminalInput:
    """Read terminal commands without blocking the camera/display loop."""

    def __init__(self) -> None:
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def get_nowait(self) -> str | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def _read_loop(self) -> None:
        while True:
            try:
                line = input()
            except EOFError:
                self._commands.put("q")
                return
            self._commands.put(line.strip())


class YOLOSegmentationDetector:
    """Run Ultralytics YOLO segmentation and return mask-aware detections."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_SEG_MODEL_PATH,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_results: int = 10,
        imgsz: int = 640,
        device: str | None = None,
        class_ids: list[int] | None = None,
        min_mask_area_px: int = 100,
    ) -> None:
        YOLO = load_ultralytics_yolo()
        self.model_path = require_yolo_model(model_path)
        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_results = int(max_results)
        self.imgsz = int(imgsz)
        self.device = device
        self.class_ids = class_ids
        self.min_mask_area_px = int(min_mask_area_px)
        self.names = getattr(self.model, "names", {}) or {}

    def detect(self, frame_bgr: np.ndarray) -> list[SegmentationDetection]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            classes=self.class_ids,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None or len(boxes) == 0:
            return []

        binary_masks = result_masks_to_binary(masks, frame_bgr.shape)
        if not binary_masks:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        segments: list[SegmentationDetection] = []
        for index, (bbox, confidence, class_id) in enumerate(
            zip(xyxy, confidences, classes)
        ):
            if index >= len(binary_masks):
                break
            mask = binary_masks[index]
            try:
                centroid_u, centroid_v, mask_area_px = mask_centroid(mask)
            except ValueError:
                continue
            if mask_area_px < self.min_mask_area_px:
                continue

            x1, y1, x2, y2 = [float(value) for value in bbox]
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            label = str(self.names.get(int(class_id), f"class_{int(class_id)}"))
            detection = Detection(
                u=int(round(centroid_u)),
                v=int(round(centroid_v)),
                x=int(round(x1)),
                y=int(round(y1)),
                width=int(round(width)),
                height=int(round(height)),
                area=float(mask_area_px),
                label=label,
                confidence=float(confidence),
                class_id=int(class_id),
            )
            segments.append(
                SegmentationDetection(
                    detection=detection,
                    mask=mask,
                    mask_area_px=mask_area_px,
                    centroid_u=float(centroid_u),
                    centroid_v=float(centroid_v),
                )
            )

        segments.sort(
            key=lambda item: (
                item.detection.confidence
                if item.detection.confidence is not None
                else 0.0,
                item.mask_area_px,
            ),
            reverse=True,
        )
        if self.max_results > 0:
            segments = segments[: self.max_results]
        return segments


class YoloSegGraspSender:
    def __init__(
        self,
        host: str | None,
        port: int,
        timeout_s: float,
        dry_run: bool = False,
        task: CameraPointTask = "out",
        send_func: SendCameraPointFunc = send_camera_point,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.dry_run = bool(dry_run)
        self.task = task
        self._send_func = send_func

    def send_target(
        self,
        target: RankedSegTarget,
        source: str,
        task: CameraPointTask | None = None,
    ) -> dict[str, Any]:
        command = camera_point_command_from_target(
            target,
            source=source,
            task=self.task if task is None else task,
        )
        if self.dry_run:
            return {"type": "dry_run", "payload": asdict(command)}
        if not self.host:
            raise ValueError("--host is required unless --dry-run is used")
        return self._send_func(self.host, self.port, command, self.timeout_s)


def load_ultralytics_yolo() -> Any:
    configure_ultralytics_config_dir()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO segmentation. "
            "Install project requirements first."
        ) from exc
    return YOLO


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def result_masks_to_binary(masks: Any, frame_shape: tuple[int, ...]) -> list[np.ndarray]:
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    data = getattr(masks, "data", None)
    if data is not None:
        values = _tensor_to_numpy(data)
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        binary_masks: list[np.ndarray] = []
        for item in values:
            mask = np.asarray(item) > 0.5
            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            binary_masks.append(mask)
        return binary_masks

    polygons = getattr(masks, "xy", None)
    if polygons is None:
        return []

    binary_masks = []
    for polygon in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        points = np.asarray(polygon, dtype=np.float32)
        if points.size == 0:
            binary_masks.append(mask.astype(bool))
            continue
        points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [points], 1)
        binary_masks.append(mask.astype(bool))
    return binary_masks


def mask_centroid(mask: np.ndarray) -> tuple[float, float, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel array")
    mask_bool = mask.astype(bool)
    mask_area_px = int(np.count_nonzero(mask_bool))
    if mask_area_px <= 0:
        raise ValueError("mask is empty")

    moments = cv2.moments(mask_bool.astype(np.uint8), binaryImage=True)
    if moments["m00"] != 0:
        centroid_u = float(moments["m10"] / moments["m00"])
        centroid_v = float(moments["m01"] / moments["m00"])
    else:
        ys, xs = np.nonzero(mask_bool)
        centroid_u = float(np.mean(xs))
        centroid_v = float(np.mean(ys))
    return centroid_u, centroid_v, mask_area_px


def depth_array_meters(depth_frame: Any) -> np.ndarray:
    if hasattr(depth_frame, "get_data") and hasattr(depth_frame, "get_units"):
        raw = np.asanyarray(depth_frame.get_data())
        return raw.astype(np.float32) * float(depth_frame.get_units())

    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    depth_m = np.zeros((height, width), dtype=np.float32)
    for v in range(height):
        for u in range(width):
            depth_m[v, u] = float(depth_frame.get_distance(u, v))
    return depth_m


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    erode_px = int(erode_px)
    if erode_px <= 0:
        return mask.astype(bool)
    kernel_size = 2 * erode_px + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if np.count_nonzero(eroded) == 0:
        return mask.astype(bool)
    return eroded


def mask_depth_median(
    depth_frame: Any,
    mask: np.ndarray,
    *,
    max_depth_m: float,
    erode_px: int = 0,
) -> MaskDepthStats:
    depth_m = depth_array_meters(depth_frame)
    if depth_m.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape[:2]} does not match depth frame {depth_m.shape[:2]}"
        )

    sample_mask = erode_mask(mask, erode_px)
    valid = sample_mask & np.isfinite(depth_m) & (depth_m > 0.0)
    if max_depth_m > 0.0:
        valid &= depth_m <= float(max_depth_m)
    values = depth_m[valid]
    if values.size == 0:
        raise ValueError("mask has no valid depth pixels")
    return MaskDepthStats(
        median_m=float(np.median(values)),
        valid_count=int(values.size),
    )


def deproject_pixel_with_depth(
    depth_frame: Any,
    u: int,
    v: int,
    depth_m: float,
    *,
    intrinsics: PinholeIntrinsics | None = None,
) -> CameraPoint:
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())
    u = int(u)
    v = int(v)
    depth_m = float(depth_m)
    if u < 0 or u >= width or v < 0 or v >= height:
        raise ValueError(f"pixel ({u}, {v}) is outside depth frame {width}x{height}")
    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError(f"invalid mask median depth at pixel ({u}, {v}): {depth_m:.6f} m")

    if intrinsics is None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required when using RealSense frame intrinsics."
            ) from exc
        rs_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        x_m, y_m, z_m = rs.rs2_deproject_pixel_to_point(
            rs_intrinsics,
            [float(u), float(v)],
            depth_m,
        )
    else:
        if intrinsics.width != width or intrinsics.height != height:
            raise ValueError(
                "intrinsics resolution "
                f"{intrinsics.width}x{intrinsics.height} does not match depth frame "
                f"{width}x{height}"
            )
        x_m = (float(u) - intrinsics.ppx) / intrinsics.fx * depth_m
        y_m = (float(v) - intrinsics.ppy) / intrinsics.fy * depth_m
        z_m = depth_m
    return CameraPoint(
        u=u,
        v=v,
        depth_m=depth_m,
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=float(z_m),
    )


def clamp_pixel_from_centroid(
    centroid_u: float,
    centroid_v: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    u = int(round(float(centroid_u)))
    v = int(round(float(centroid_v)))
    return max(0, min(width - 1, u)), max(0, min(height - 1, v))


def segments_to_ranked_targets(
    segments: list[SegmentationDetection],
    depth_frame: Any,
    *,
    intrinsics: PinholeIntrinsics | None,
    max_depth_m: float,
    min_depth_valid_px: int,
    mask_depth_erode_px: int,
    camera_up_offset_mm: float,
    point_z_offset_mm: float,
    camera_to_base: np.ndarray | None = None,
    base_z_offset_mm: float = 0.0,
) -> tuple[list[RankedSegTarget], dict[int, str]]:
    unranked: list[RankedSegTarget] = []
    errors: dict[int, str] = {}
    width = int(depth_frame.get_width())
    height = int(depth_frame.get_height())

    for index, segment in enumerate(segments):
        try:
            depth_stats = mask_depth_median(
                depth_frame,
                segment.mask,
                max_depth_m=max_depth_m,
                erode_px=mask_depth_erode_px,
            )
            if depth_stats.valid_count < min_depth_valid_px:
                raise ValueError(
                    "mask has too few valid depth pixels: "
                    f"{depth_stats.valid_count} < {min_depth_valid_px}"
                )
            u, v = clamp_pixel_from_centroid(
                segment.centroid_u,
                segment.centroid_v,
                width=width,
                height=height,
            )
            raw_point = deproject_pixel_with_depth(
                depth_frame,
                u,
                v,
                depth_stats.median_m,
                intrinsics=intrinsics,
            )
            send_point = offset_camera_point(
                raw_point,
                camera_up_offset_mm=camera_up_offset_mm,
                point_z_offset_mm=point_z_offset_mm,
            )
            send_point = apply_base_z_offset(
                send_point,
                camera_to_base=camera_to_base,
                base_z_offset_mm=base_z_offset_mm,
            )
        except (RuntimeError, ValueError) as exc:
            errors[index] = str(exc)
            continue

        unranked.append(
            RankedSegTarget(
                rank=0,
                segment=segment,
                raw_point=raw_point,
                send_point=send_point,
                mask_depth_valid_count=depth_stats.valid_count,
            )
        )

    ranked = sorted(
        unranked,
        key=lambda target: (
            target.raw_point.depth_m,
            target.detection.u,
            target.detection.v,
        ),
    )
    return [
        RankedSegTarget(
            rank=index,
            segment=target.segment,
            raw_point=target.raw_point,
            send_point=target.send_point,
            mask_depth_valid_count=target.mask_depth_valid_count,
        )
        for index, target in enumerate(ranked, start=1)
    ], errors


def camera_point_command_from_target(
    target: RankedSegTarget,
    source: str,
    task: CameraPointTask = "out",
) -> CameraPointCommand:
    detection = target.detection
    return CameraPointCommand(
        x_m=target.send_point.x_m,
        y_m=target.send_point.y_m,
        z_m=target.send_point.z_m,
        u=target.send_point.u,
        v=target.send_point.v,
        depth_m=target.raw_point.depth_m,
        source=source,
        task=task,
        rank=target.rank,
        label=detection.label,
        confidence=detection.confidence,
        class_id=detection.class_id,
        bbox_xywh=(
            float(detection.x),
            float(detection.y),
            float(detection.width),
            float(detection.height),
        ),
        mask_area_px=target.segment.mask_area_px,
        mask_centroid_u=target.segment.centroid_u,
        mask_centroid_v=target.segment.centroid_v,
        mask_depth_median_m=target.raw_point.depth_m,
        mask_depth_valid_count=target.mask_depth_valid_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect YOLO segmentation masks, use mask centroid plus mask median "
            "depth as the grasp point, and send it to Linux catch_server_1."
        )
    )
    parser.add_argument("--host", help="Linux VM IP address running catch_server_1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true", help="print target XYZ without sending")
    parser.add_argument(
        "--task",
        choices=("out", "in"),
        dest="task_option",
        help="default task for bare rank commands; terminal can override with 'out N' or 'in N'",
    )
    parser.set_defaults(task="out")

    parser.add_argument("--model", type=Path, default=DEFAULT_SEG_MODEL_PATH)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-timeout-ms", type=int, default=10000)
    parser.add_argument("--max-depth-m", type=float, default=1.5)
    parser.add_argument("--min-depth-valid-px", type=int, default=20)
    parser.add_argument(
        "--mask-depth-erode-px",
        type=int,
        default=2,
        help="erode mask by this pixel radius before depth median sampling; 0 disables",
    )
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

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--min-mask-area-px", type=int, default=100)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class ids to keep, e.g. --classes 39 for bottle.",
    )

    parser.add_argument(
        "--camera-up-offset-mm",
        type=float,
        default=0.0,
        help="additional 3D offset in the RealSense image-up direction after deprojection",
    )
    parser.add_argument(
        "--point-z-offset-mm",
        type=float,
        default=0.0,
        help="optional camera-frame Z offset added before sending",
    )
    parser.add_argument(
        "--hand-eye-result",
        nargs="?",
        const=DEFAULT_HAND_EYE_RESULT,
        default=None,
        type=Path,
        help=(
            "optional eye_to_hand hand_eye_result.json camera->base transform, "
            "only needed for --base-z-offset-mm"
        ),
    )
    parser.add_argument(
        "--base-z-offset-mm",
        type=float,
        default=0.0,
        help="optional robot base-frame Z offset before converting back to camera XYZ",
    )

    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument("--hide-depth", action="store_true")
    parser.add_argument("--hide-mask", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    task_option = getattr(args, "task_option", None)
    if task_option is not None:
        args.task = task_option
    if not args.dry_run and not args.host:
        raise ValueError("--host is required unless --dry-run is used")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.max_depth_m < 0:
        raise ValueError("--max-depth-m must be >= 0")
    if args.min_depth_valid_px < 1:
        raise ValueError("--min-depth-valid-px must be >= 1")
    if args.mask_depth_erode_px < 0:
        raise ValueError("--mask-depth-erode-px must be >= 0")
    if args.max_results < 1:
        raise ValueError("--max-results must be >= 1")
    if args.min_mask_area_px < 1:
        raise ValueError("--min-mask-area-px must be >= 1")
    if args.print_interval < 0.0:
        raise ValueError("--print-interval must be >= 0")
    if args.base_z_offset_mm != 0.0 and args.hand_eye_result is None:
        raise ValueError("--base-z-offset-mm requires --hand-eye-result")


def find_ranked_target(
    targets: list[RankedSegTarget],
    rank: int,
) -> RankedSegTarget | None:
    for target in targets:
        if target.rank == rank:
            return target
    return None


def format_target_line(target: RankedSegTarget) -> str:
    detection = target.detection
    confidence = detection.confidence if detection.confidence is not None else 0.0
    return (
        f"{target.rank}: {detection.label} conf={confidence:.2f} "
        f"depth={target.depth_mm:.1f}mm "
        f"mask_area={target.segment.mask_area_px}px "
        f"valid_depth={target.mask_depth_valid_count}px "
        f"centroid=({target.segment.centroid_u:.1f},{target.segment.centroid_v:.1f}) "
        f"pixel=({target.send_point.u},{target.send_point.v}) "
        f"send=({target.send_point.x_m:.4f},"
        f"{target.send_point.y_m:.4f},{target.send_point.z_m:.4f})m"
    )


def print_target_table(
    targets: list[RankedSegTarget],
    errors: dict[int, str],
) -> None:
    if targets:
        print("\nCurrent YOLO-seg targets ranked by mask median depth:", flush=True)
        for target in targets:
            print("  " + format_target_line(target), flush=True)
        print(
            "Type 'out N' or 'in N' to send a target, p to print again, q to quit.",
            flush=True,
        )
    elif errors:
        print(f"\nNo valid mask-depth target: {next(iter(errors.values()))}", flush=True)
    else:
        print("\nNo YOLO-seg target found.", flush=True)


def color_for_index(index: int) -> tuple[int, int, int]:
    colors = (
        (0, 255, 0),
        (0, 200, 255),
        (255, 180, 0),
        (255, 0, 180),
        (180, 255, 0),
        (0, 120, 255),
    )
    return colors[index % len(colors)]


def draw_segmentation_targets(
    image: Any,
    segments: list[SegmentationDetection],
    targets: list[RankedSegTarget],
    errors: dict[int, str],
    status: str,
) -> None:
    target_by_segment_id = {id(target.segment): target for target in targets}
    error_indexes = set(errors)

    overlay = image.copy()
    for index, segment in enumerate(segments):
        target = target_by_segment_id.get(id(segment))
        has_error = index in error_indexes and target is None
        color = color_for_index(index) if target is not None else (0, 0, 255) if has_error else (120, 120, 120)
        overlay[segment.mask.astype(bool)] = color
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0.0, dst=image)

    for index, segment in enumerate(segments):
        detection = segment.detection
        target = target_by_segment_id.get(id(segment))
        has_error = index in error_indexes and target is None
        color = color_for_index(index) if target is not None else (0, 0, 255) if has_error else (120, 120, 120)
        cv2.rectangle(
            image,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            color,
            2,
        )
        marker_pixel = (
            target.send_point.u,
            target.send_point.v,
        ) if target is not None else (
            int(round(segment.centroid_u)),
            int(round(segment.centroid_v)),
        )
        cv2.drawMarker(
            image,
            marker_pixel,
            (0, 255, 255) if target is not None else color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )

        confidence = detection.confidence if detection.confidence is not None else 0.0
        if target is not None:
            text = f"{target.rank} {detection.label} {confidence:.2f} {target.depth_mm:.0f}mm"
        elif has_error:
            text = f"{detection.label} no mask depth"
        else:
            text = f"{detection.label} {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (detection.x, max(18, detection.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
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

    if targets:
        text = f"targets={len(targets)} nearest: " + format_camera_point(targets[0].send_point)
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


def make_mask_debug_image(
    segments: list[SegmentationDetection],
    targets: list[RankedSegTarget],
    frame_shape: tuple[int, ...],
) -> np.ndarray:
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    image = np.zeros((height, width, 3), dtype=np.uint8)
    target_by_segment_id = {id(target.segment): target for target in targets}
    for index, segment in enumerate(segments):
        color = color_for_index(index)
        image[segment.mask.astype(bool)] = color
        target = target_by_segment_id.get(id(segment))
        if target is not None:
            cv2.putText(
                image,
                str(target.rank),
                (target.send_point.u + 6, target.send_point.v - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return image


def send_and_print(
    sender: YoloSegGraspSender,
    target: RankedSegTarget,
    source: str,
    task: CameraPointTask,
) -> dict[str, Any]:
    response = sender.send_target(target, source=source, task=task)
    print(
        f"\nSent {task} YOLO-seg target rank {target.rank}: {format_target_line(target)}",
        flush=True,
    )
    print(response, flush=True)
    return response


def process_terminal_command(
    command: str,
    *,
    targets: list[RankedSegTarget],
    errors: dict[int, str],
    sender: YoloSegGraspSender,
    source: str,
) -> bool:
    command = command.strip().lower()
    if command == "":
        print_target_table(targets, errors)
        return True
    if command in {"q", "quit", "exit"}:
        return False
    if command in {"p", "print", "ls", "list"}:
        print_target_table(targets, errors)
        return True

    parsed = parse_target_command(command, sender.task)
    if parsed is None:
        print(
            f"Unknown command: {command!r}. Type 'out N', 'in N', p, or q.",
            flush=True,
        )
        return True
    task, rank = parsed
    target = find_ranked_target(targets, rank)
    if target is None:
        valid = ", ".join(str(item.rank) for item in targets) or "none"
        print(f"Rank {rank} is not available in the latest frame. Valid ranks: {valid}", flush=True)
        return True
    try:
        send_and_print(sender, target, source=source, task=task)
    except Exception as exc:  # noqa: BLE001 - manual hardware UI should keep context
        print(f"Send failed for {task} rank {rank}: {exc}", flush=True)
    return True


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

    camera_to_base = None
    if args.hand_eye_result is not None:
        camera_to_base = load_camera_to_base_transform(args.hand_eye_result)
        print(f"Using hand-eye camera->base transform: {args.hand_eye_result}", flush=True)

    detector = YOLOSegmentationDetector(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        max_results=args.max_results,
        imgsz=args.imgsz,
        device=args.device,
        class_ids=args.classes,
        min_mask_area_px=args.min_mask_area_px,
    )
    pipeline, align = create_aligned_rgb_depth_stream(
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    sender = YoloSegGraspSender(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
        task=args.task,
    )
    terminal = TerminalInput()
    terminal.start()
    source = "yolo_seg_mask_centroid_median_depth"
    last_print_time = 0.0
    latest_targets: list[RankedSegTarget] = []
    latest_errors: dict[int, str] = {}

    print(
        "Type 'out N' or 'in N' in this terminal to send a mask target. "
        "Type p to print targets, q to quit.",
        flush=True,
    )

    if not args.no_display:
        cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_mask:
            cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        if not args.hide_depth:
            cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    try:
        running = True
        while running:
            color_image, depth_frame = wait_for_aligned_rgb_depth(
                pipeline,
                align,
                timeout_ms=args.frame_timeout_ms,
            )
            if color_image is None or depth_frame is None:
                continue

            segments = detector.detect(color_image)
            latest_targets, latest_errors = segments_to_ranked_targets(
                segments,
                depth_frame,
                intrinsics=intrinsics,
                max_depth_m=args.max_depth_m,
                min_depth_valid_px=args.min_depth_valid_px,
                mask_depth_erode_px=args.mask_depth_erode_px,
                camera_up_offset_mm=args.camera_up_offset_mm,
                point_z_offset_mm=args.point_z_offset_mm,
                camera_to_base=camera_to_base,
                base_z_offset_mm=args.base_z_offset_mm,
            )

            now = time.monotonic()
            if args.print_interval == 0.0 or now - last_print_time >= args.print_interval:
                print_target_table(latest_targets, latest_errors)
                last_print_time = now

            while True:
                command = terminal.get_nowait()
                if command is None:
                    break
                running = process_terminal_command(
                    command,
                    targets=latest_targets,
                    errors=latest_errors,
                    sender=sender,
                    source=source,
                )
                if not running:
                    break

            if not args.no_display:
                status = "No YOLO-seg target found. Type q in terminal or press Esc to exit."
                display_image = color_image.copy()
                draw_segmentation_targets(
                    display_image,
                    segments,
                    latest_targets,
                    latest_errors,
                    status,
                )
                cv2.imshow(RGB_WINDOW_NAME, display_image)

                if not args.hide_mask:
                    mask_debug = make_mask_debug_image(
                        segments,
                        latest_targets,
                        color_image.shape,
                    )
                    cv2.imshow(MASK_WINDOW_NAME, mask_debug)

                if not args.hide_depth:
                    depth_debug = make_depth_colormap(depth_frame, args.max_depth_m)
                    draw_segmentation_targets(
                        depth_debug,
                        segments,
                        latest_targets,
                        latest_errors,
                        status,
                    )
                    cv2.imshow(DEPTH_WINDOW_NAME, depth_debug)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

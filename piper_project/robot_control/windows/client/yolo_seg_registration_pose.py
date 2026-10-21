"""YOLO instance segmentation + Open3D registration 6D pose client.

This is an intentionally small starting point for a stationary (or briefly
stopped) conveyor.  YOLO selects the object pixels, aligned RealSense depth is
converted to a camera-frame point cloud, and a class-specific CAD/template
cloud is registered to it.  The resulting object pose is composed with a
configured object->grasp transform and sent through the existing
``GraspCandidate`` JSON-lines protocol.

Run from ``piper_project``::

    python -m robot_control.windows.client.yolo_seg_registration_pose \
        --seg-model models/best-seg.pt \
        --templates models/registration/templates.example.json \
        --host 192.168.1.10

Press ``s`` to send the best valid pose, or ``q`` to quit.  If ``--host`` is
omitted, pressing ``s`` only prints the payload.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from robot_control.shared.protocol import (
    CameraPointCommand,
    GraspCandidate,
    send_camera_point,
)
from robot_control.windows.client.graspnet_grasp_send_tcp import point_cloud_from_mask
from robot_control.windows.client.yolo_seg_grasp_send_tcp import (
    SegmentationDetection,
    YOLOSegmentationDetector,
)
from robot_control.windows.vision.d435i_intrinsics import find_d435i_intrinsics
from robot_control.windows.vision.depth import (
    create_aligned_rgb_depth_stream,
    wait_for_aligned_rgb_depth,
)


@dataclass
class Template:
    label: str
    model_path: Path
    model_scale: float
    voxel_size_m: float
    min_fitness: float
    max_rmse_m: float
    object_from_grasp: np.ndarray
    source_down: Any
    source_feature: Any


@dataclass(frozen=True)
class PoseEstimate:
    segment: SegmentationDetection
    camera_from_object: np.ndarray
    camera_from_grasp: np.ndarray
    fitness: float
    rmse_m: float
    point_count: int

    @property
    def score(self) -> float:
        confidence = float(self.segment.detection.confidence or 0.0)
        return confidence * self.fitness / (1.0 + 50.0 * self.rmse_m)


def require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required: python -m pip install open3d") from exc
    return o3d


def matrix4(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    return result


def read_model_cloud(o3d: Any, path: Path, scale: float, sample_points: int) -> Any:
    cloud = o3d.io.read_point_cloud(str(path))
    if not cloud.has_points():
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            raise ValueError(f"cannot read point cloud or triangle mesh: {path}")
        cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
    cloud.scale(float(scale), center=np.zeros(3))
    return cloud


def preprocess_cloud(o3d: Any, cloud: Any, voxel_size_m: float) -> tuple[Any, Any]:
    down = cloud.voxel_down_sample(float(voxel_size_m))
    if len(down.points) < 30:
        raise ValueError(f"too few downsampled points: {len(down.points)}")
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size_m * 2.5,
            max_nn=40,
        )
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size_m * 5.0,
            max_nn=100,
        ),
    )
    return down, feature


def load_templates(config_path: Path, o3d: Any) -> dict[str, Template]:
    document = json.loads(config_path.read_text(encoding="utf-8"))
    objects = document.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError("template config must contain a non-empty 'objects' map")

    result: dict[str, Template] = {}
    for label, raw in objects.items():
        if not isinstance(raw, dict):
            raise ValueError(f"objects.{label} must be an object")
        model_path = Path(str(raw["model_path"]))
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
        scale = float(raw.get("model_scale", 1.0))
        voxel = float(raw.get("voxel_size_m", 0.004))
        cloud = read_model_cloud(
            o3d,
            model_path,
            scale=scale,
            sample_points=int(raw.get("sample_points", 50000)),
        )
        source_down, source_feature = preprocess_cloud(o3d, cloud, voxel)
        result[str(label)] = Template(
            label=str(label),
            model_path=model_path,
            model_scale=scale,
            voxel_size_m=voxel,
            min_fitness=float(raw.get("min_fitness", 0.30)),
            max_rmse_m=float(raw.get("max_rmse_m", 0.012)),
            object_from_grasp=matrix4(
                raw.get("object_from_grasp", np.eye(4).tolist()),
                f"objects.{label}.object_from_grasp",
            ),
            source_down=source_down,
            source_feature=source_feature,
        )
    return result


def numpy_cloud(o3d: Any, points: np.ndarray) -> Any:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(cloud.points) >= 30:
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cloud


def register_template(o3d: Any, template: Template, target_points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return camera_from_object, fitness and inlier RMSE.

    The global FPFH/RANSAC stage avoids needing an initial pose.  Two
    point-to-plane ICP passes then refine the result.  Production systems
    should add class-specific workspace, upright and symmetry constraints.
    """

    target_full = numpy_cloud(o3d, target_points)
    target_down, target_feature = preprocess_cloud(
        o3d,
        target_full,
        template.voxel_size_m,
    )
    distance = template.voxel_size_m * 1.8
    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        template.source_down,
        target_down,
        template.source_feature,
        target_feature,
        True,
        distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.90),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 0.999),
    )

    transform = np.asarray(global_result.transformation, dtype=np.float64)
    for multiplier in (1.5, 0.75):
        threshold = template.voxel_size_m * multiplier
        refined = o3d.pipelines.registration.registration_icp(
            template.source_down,
            target_down,
            threshold,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        transform = np.asarray(refined.transformation, dtype=np.float64)
    return transform, float(refined.fitness), float(refined.inlier_rmse)


def estimate_segment_pose(
    o3d: Any,
    segment: SegmentationDetection,
    depth_frame: Any,
    intrinsics: Any,
    template: Template,
    *,
    erode_px: int,
    max_points: int,
) -> PoseEstimate:
    mask = segment.mask.astype(np.uint8)
    if erode_px > 0:
        size = erode_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(mask, kernel).astype(bool)
        if np.count_nonzero(eroded) >= 100:
            mask = eroded
    points = point_cloud_from_mask(
        depth_frame,
        mask.astype(bool),
        intrinsics=intrinsics,
        voxel_size_m=template.voxel_size_m * 0.5,
        max_points=max_points,
    )
    if len(points) < 100:
        raise ValueError(f"too few object depth points: {len(points)}")

    camera_from_object, fitness, rmse_m = register_template(o3d, template, points)
    # T_camera_grasp = T_camera_object @ T_object_grasp.  In other words,
    # object_from_grasp describes the desired gripper frame in model axes.
    camera_from_grasp = camera_from_object @ template.object_from_grasp
    return PoseEstimate(
        segment=segment,
        camera_from_object=camera_from_object,
        camera_from_grasp=camera_from_grasp,
        fitness=fitness,
        rmse_m=rmse_m,
        point_count=len(points),
    )


def valid_estimate(estimate: PoseEstimate, template: Template) -> bool:
    transform = estimate.camera_from_object
    rotation = transform[:3, :3]
    return bool(
        estimate.fitness >= template.min_fitness
        and estimate.rmse_m <= template.max_rmse_m
        and np.all(np.isfinite(transform))
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2)
        and math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=2e-2)
        and transform[2, 3] > 0.0
    )


def command_from_estimate(estimate: PoseEstimate, rank: int) -> CameraPointCommand:
    detection = estimate.segment.detection
    transform = estimate.camera_from_grasp
    rotation = tuple(tuple(float(v) for v in row) for row in transform[:3, :3])
    position = tuple(float(v) for v in transform[:3, 3])
    candidate = GraspCandidate(
        position_camera_m=position,
        rotation_camera=rotation,
        score=float(estimate.score),
        source="yolo_seg_pointcloud_registration",
        index=0,
    )
    return CameraPointCommand(
        x_m=position[0],
        y_m=position[1],
        z_m=position[2],
        u=int(round(estimate.segment.centroid_u)),
        v=int(round(estimate.segment.centroid_v)),
        depth_m=position[2],
        source="yolo_seg_pointcloud_registration",
        task="out",
        rank=int(rank),
        label=detection.label,
        confidence=detection.confidence,
        class_id=detection.class_id,
        bbox_xywh=(detection.x, detection.y, detection.width, detection.height),
        mask_area_px=estimate.segment.mask_area_px,
        mask_centroid_u=estimate.segment.centroid_u,
        mask_centroid_v=estimate.segment.centroid_v,
        grasp_mode="static",
        vision_source="pointcloud_registration",
        selected_grasp_index=0,
        grasp_candidates=(candidate,),
    )


def draw_estimate(frame: np.ndarray, estimate: PoseEstimate, valid: bool) -> None:
    detection = estimate.segment.detection
    color = (30, 220, 30) if valid else (30, 30, 230)
    cv2.rectangle(
        frame,
        (detection.x, detection.y),
        (detection.x + detection.width, detection.y + detection.height),
        color,
        2,
    )
    text = (
        f"{detection.label} fit={estimate.fitness:.2f} "
        f"rmse={estimate.rmse_m * 1000.0:.1f}mm"
    )
    cv2.putText(
        frame,
        text,
        (detection.x, max(20, detection.y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seg-model", type=Path, required=True)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--erode-mask-px", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--registration-interval-s", type=float, default=0.4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    o3d = require_open3d()
    templates = load_templates(args.templates.resolve(), o3d)
    detector = YOLOSegmentationDetector(
        model_path=args.seg_model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    intrinsics = find_d435i_intrinsics(args.width, args.height)
    pipeline, align = create_aligned_rgb_depth_stream(args.width, args.height, args.fps)
    latest: list[PoseEstimate] = []
    last_registration_s = -math.inf
    try:
        while True:
            frame, depth_frame = wait_for_aligned_rgb_depth(pipeline, align)
            if frame is None or depth_frame is None:
                continue
            now = time.monotonic()
            if now - last_registration_s >= args.registration_interval_s:
                estimates: list[PoseEstimate] = []
                for segment in detector.detect(frame):
                    template = templates.get(segment.detection.label)
                    if template is None:
                        continue
                    try:
                        estimate = estimate_segment_pose(
                            o3d,
                            segment,
                            depth_frame,
                            intrinsics,
                            template,
                            erode_px=args.erode_mask_px,
                            max_points=args.max_points,
                        )
                    except (RuntimeError, ValueError) as exc:
                        print(f"skip {segment.detection.label}: {exc}", flush=True)
                        continue
                    if valid_estimate(estimate, template):
                        estimates.append(estimate)
                latest = sorted(estimates, key=lambda item: item.score, reverse=True)
                last_registration_s = now

            display = frame.copy()
            for estimate in latest:
                draw_estimate(display, estimate, True)
            cv2.putText(
                display,
                "s: send best pose   q: quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO-seg + point-cloud registration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                if not latest:
                    print("No valid registered pose to send", flush=True)
                    continue
                command = command_from_estimate(latest[0], rank=1)
                if args.host:
                    response = send_camera_point(args.host, args.port, command, args.timeout_s)
                    print(json.dumps(response, ensure_ascii=False), flush=True)
                else:
                    print(json.dumps(asdict(command), ensure_ascii=False), flush=True)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

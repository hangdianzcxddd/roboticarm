from __future__ import annotations

import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - environment guard
    cv2 = None

if cv2 is not None:
    from robot_control.shared.protocol import GraspCandidate
    from robot_control.windows.client.graspnet_grasp_send_tcp import (
        DepthForegroundDetector,
        DepthForegroundDetectorConfig,
        GraspNetTarget,
        build_parser,
        camera_point_command_from_target,
        colored_point_cloud_from_targets,
        dynamic_candidate_from_target,
        dynamic_pick_x_mm_for_candidate,
        load_intrinsics,
        point_cloud_stats_text,
        point_cloud_from_mask,
        target_from_segment,
        validate_args,
    )
    from robot_control.windows.client.yolo_seg_grasp_send_tcp import SegmentationDetection
    from robot_control.windows.vision.belt_depth_plane import BeltDepthPlane
    from robot_control.windows.vision.d435i_intrinsics import (
        DEFAULT_D435I_PARA,
        PinholeIntrinsics,
    )
    from robot_control.windows.vision.depth import CameraPoint
    from robot_control.windows.vision.detector import Detection


class FakeDepthFrame:
    def __init__(self, depth_m: np.ndarray) -> None:
        self.depth_m = np.asarray(depth_m, dtype=np.float32)

    def get_width(self) -> int:
        return int(self.depth_m.shape[1])

    def get_height(self) -> int:
        return int(self.depth_m.shape[0])

    def get_data(self):
        return np.round(self.depth_m / 0.001).astype(np.uint16)

    def get_units(self) -> float:
        return 0.001

    def get_distance(self, u: int, v: int) -> float:
        return float(self.depth_m[v, u])


def flat_plane(width: int, height: int, depth_m: float) -> BeltDepthPlane:
    return BeltDepthPlane(width=width, height=height, a=0.0, b=0.0, c=depth_m)


@unittest.skipIf(cv2 is None, "opencv-python is required for GraspNet client tests")
class TestGraspNetGraspSendTcp(unittest.TestCase):
    def test_load_intrinsics_uses_width_height_path_order(self):
        class Args:
            use_realsense_intrinsics = False
            intrinsics_json = DEFAULT_D435I_PARA
            width = 1280
            height = 720

        intrinsics = load_intrinsics(Args(), depth_frame=None)

        self.assertEqual(intrinsics.width, 1280)
        self.assertEqual(intrinsics.height, 720)

    def test_dynamic_debug_candidates_argument_is_stored(self):
        args = build_parser().parse_args(
            [
                "dynamic",
                "--dry-run",
                "--belt-depth-plane-json",
                "belt.json",
                "--conveyor-speed-mm-s",
                "80",
                "--dynamic-debug-candidates",
            ]
        )

        validate_args(args)

        self.assertTrue(args.dynamic_debug_candidates)

    def test_colored_point_cloud_from_targets_combines_target_points(self):
        detection = Detection(
            u=2,
            v=2,
            x=1,
            y=1,
            width=3,
            height=3,
            area=9.0,
            label="graspnet_object",
        )
        segment = SegmentationDetection(
            detection=detection,
            mask=np.ones((4, 4), dtype=bool),
            mask_area_px=16,
            centroid_u=2.0,
            centroid_v=2.0,
        )
        point = CameraPoint(u=2, v=2, depth_m=0.5, x_m=0.0, y_m=0.0, z_m=0.5)
        first = GraspNetTarget(
            rank=1,
            segment=segment,
            raw_point=point,
            send_point=point,
            base_xyz_m=None,
            point_cloud_camera_m=np.asarray(
                [[0.0, 0.0, 0.5], [0.01, 0.0, 0.5]],
                dtype=np.float32,
            ),
            grasp_candidates=(),
        )
        second = GraspNetTarget(
            rank=2,
            segment=segment,
            raw_point=point,
            send_point=point,
            base_xyz_m=None,
            point_cloud_camera_m=np.asarray([[0.0, 0.02, 0.6]], dtype=np.float32),
            grasp_candidates=(),
        )

        points, colors = colored_point_cloud_from_targets([first, second])

        self.assertEqual(points.shape, (3, 3))
        self.assertEqual(colors.shape, (3, 3))
        self.assertTrue(np.allclose(points[2], [0.0, 0.02, 0.6]))
        self.assertFalse(np.allclose(colors[0], colors[2]))

    def test_point_cloud_stats_text_reports_xyz_ranges(self):
        text = point_cloud_stats_text(
            np.asarray(
                [
                    [0.0, -0.01, 0.50],
                    [0.02, 0.03, 0.56],
                ],
                dtype=np.float32,
            ),
            label="sample",
        )

        self.assertIn("sample: count=2", text)
        self.assertIn("span=(20.0,40.0,60.0)mm", text)
        self.assertIn("z=[+0.5000,+0.5600]m", text)

    def test_depth_foreground_detector_uses_plane_roi_and_entry_edge(self):
        depth = np.full((20, 30), 0.50, dtype=np.float32)
        depth[6:12, 20:28] = 0.46
        detector = DepthForegroundDetector(
            flat_plane(30, 20, 0.50),
            DepthForegroundDetectorConfig(
                min_height_above_belt_mm=20.0,
                max_height_above_belt_mm=80.0,
                min_area_px=10,
                morphology_kernel_size=1,
                roi_xywh=(5, 4, 23, 12),
                entry_edge="right",
                entry_clear_frames=2,
            ),
        )

        self.assertEqual(detector.detect(FakeDepthFrame(depth)), [])
        depth[6:12, 20:28] = 0.50
        depth[6:12, 16:24] = 0.46
        first_clear = detector.detect(FakeDepthFrame(depth))
        self.assertEqual(first_clear, [])
        second_clear = detector.detect(FakeDepthFrame(depth))

        self.assertEqual(len(second_clear), 1)
        self.assertEqual(second_clear[0].detection.label, "graspnet_object")
        self.assertGreater(second_clear[0].mask_area_px, 10)

    def test_point_cloud_from_mask_deprojects_depth_pixels(self):
        depth = np.full((4, 4), 0.50, dtype=np.float32)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        points = point_cloud_from_mask(
            FakeDepthFrame(depth),
            mask,
            intrinsics=PinholeIntrinsics(
                width=4,
                height=4,
                fx=100.0,
                fy=100.0,
                ppx=2.0,
                ppy=2.0,
            ),
            voxel_size_m=0.0,
            max_points=100,
        )

        self.assertEqual(points.shape, (4, 3))
        self.assertTrue(np.allclose(points[:, 2], 0.50))

    def test_command_payload_carries_graspnet_candidates(self):
        detection = Detection(
            u=2,
            v=2,
            x=1,
            y=1,
            width=3,
            height=3,
            area=9.0,
            label="graspnet_object",
        )
        segment = SegmentationDetection(
            detection=detection,
            mask=np.ones((4, 4), dtype=bool),
            mask_area_px=16,
            centroid_u=2.0,
            centroid_v=2.0,
        )
        point = CameraPoint(u=2, v=2, depth_m=0.5, x_m=0.0, y_m=0.0, z_m=0.5)
        target = GraspNetTarget(
            rank=1,
            segment=segment,
            raw_point=point,
            send_point=point,
            base_xyz_m=(0.2, 0.0, 0.12),
            point_cloud_camera_m=np.zeros((4, 3), dtype=np.float32),
            grasp_candidates=(
                GraspCandidate(
                    position_camera_m=(0.0, 0.0, 0.5),
                    rotation_camera=(
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    score=0.9,
                ),
            ),
        )

        command = camera_point_command_from_target(
            target,
            source="graspnet_test",
            task="out",
            grasp_mode="static",
        )

        self.assertEqual(command.vision_source, "graspnet")
        self.assertEqual(command.selected_grasp_index, 0)
        self.assertEqual(len(command.grasp_candidates), 1)
        self.assertAlmostEqual(command.detected_x_mm, 200.0)

    def test_target_from_segment_sends_detected_top_z_payload(self):
        depth = np.full((4, 4), 0.50, dtype=np.float32)
        depth[1, 1] = 0.44
        depth[1, 2] = 0.48
        depth[2, 1] = 0.46
        depth[2, 2] = 0.49
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        detection = Detection(
            u=2,
            v=2,
            x=1,
            y=1,
            width=2,
            height=2,
            area=4.0,
            label="graspnet_object",
        )
        segment = SegmentationDetection(
            detection=detection,
            mask=mask,
            mask_area_px=4,
            centroid_u=1.5,
            centroid_v=1.5,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[2, 2] = -1.0
        camera_to_base[2, 3] = 1.0

        target = target_from_segment(
            1,
            segment,
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=4,
                height=4,
                fx=100.0,
                fy=100.0,
                ppx=2.0,
                ppy=2.0,
            ),
            camera_to_base=camera_to_base,
            backend=None,
            voxel_size_m=0.0,
            max_points=100,
            max_candidates=20,
        )
        command = camera_point_command_from_target(
            target,
            source="graspnet_test",
            task="out",
            grasp_mode="dynamic",
            t_arrive_s=1.0,
            conveyor_speed_mm_s=140.0,
            pick_x_mm=350.0,
            pick_y_mm=0.0,
        )

        self.assertEqual((command.mask_top_u, command.mask_top_v), (1.0, 1.0))
        self.assertAlmostEqual(command.detected_top_camera_z_m, 0.44, places=5)
        self.assertAlmostEqual(command.detected_top_z_mm, 560.0, places=4)

    def test_dynamic_pick_x_can_use_detected_base_x(self):
        detection = Detection(
            u=2,
            v=2,
            x=1,
            y=1,
            width=3,
            height=3,
            area=9.0,
            label="graspnet_object",
        )
        segment = SegmentationDetection(
            detection=detection,
            mask=np.ones((4, 4), dtype=bool),
            mask_area_px=16,
            centroid_u=2.0,
            centroid_v=2.0,
        )
        point = CameraPoint(u=2, v=2, depth_m=0.5, x_m=0.0, y_m=0.0, z_m=0.5)
        target = GraspNetTarget(
            rank=1,
            segment=segment,
            raw_point=point,
            send_point=point,
            base_xyz_m=(0.405, -0.50, 0.12),
            point_cloud_camera_m=np.zeros((4, 3), dtype=np.float32),
            grasp_candidates=(),
        )
        candidate = dynamic_candidate_from_target(
            target,
            min_x_mm=300.0,
            max_x_mm=500.0,
            min_y_mm=-600.0,
            max_y_mm=-400.0,
            pick_y_mm=0.0,
            conveyor_speed_mm_s=140.0,
            conveyor_direction="positive-y",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None

        self.assertAlmostEqual(
            dynamic_pick_x_mm_for_candidate(
                candidate,
                fixed_pick_x_mm=350.0,
                use_detected_x=False,
            ),
            350.0,
        )
        self.assertAlmostEqual(
            dynamic_pick_x_mm_for_candidate(
                candidate,
                fixed_pick_x_mm=350.0,
                use_detected_x=True,
            ),
            405.0,
        )


if __name__ == "__main__":
    unittest.main()

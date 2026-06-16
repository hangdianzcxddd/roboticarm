from __future__ import annotations

from argparse import Namespace
import unittest

import numpy as np

from robot_control.shared.protocol import camera_point_from_message, make_camera_point_message
from robot_control.windows.client.yolo_seg_grasp_send_tcp import (
    RankedSegTarget,
    SegmentationDetection,
    YoloSegGraspSender,
    camera_point_command_from_target,
    clamp_pixel_from_centroid,
    deproject_pixel_with_depth,
    dynamic_pick_x_mm_for_candidate,
    dynamic_seg_candidate_from_target,
    find_dynamic_seg_candidates,
    mask_centroid,
    mask_depth_median,
    segments_to_ranked_targets,
    validate_args,
)
from robot_control.windows.vision.d435i_intrinsics import PinholeIntrinsics
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


def detection(label: str = "cup") -> Detection:
    return Detection(
        u=0,
        v=0,
        x=1,
        y=2,
        width=3,
        height=4,
        area=12.0,
        label=label,
        confidence=0.9,
        class_id=41,
    )


def segment(mask: np.ndarray, label: str = "cup") -> SegmentationDetection:
    u, v, area = mask_centroid(mask)
    return SegmentationDetection(
        detection=detection(label),
        mask=mask.astype(bool),
        mask_area_px=area,
        centroid_u=u,
        centroid_v=v,
    )


class TestYoloSegGraspSendTcp(unittest.TestCase):
    def test_mask_centroid_uses_mask_geometry(self):
        mask = np.zeros((10, 12), dtype=bool)
        mask[2:6, 4:10] = True

        u, v, area = mask_centroid(mask)

        self.assertAlmostEqual(u, 6.5)
        self.assertAlmostEqual(v, 3.5)
        self.assertEqual(area, 24)

    def test_mask_depth_median_uses_only_valid_mask_pixels(self):
        depth = np.zeros((6, 6), dtype=np.float32)
        depth[1:5, 1:5] = np.array(
            [
                [0.20, 0.40, 0.60, 0.80],
                [0.30, 0.50, 0.70, 0.90],
                [0.00, 0.00, 1.60, 0.20],
                [0.30, 0.40, 0.50, 0.60],
            ],
            dtype=np.float32,
        )
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:5, 1:5] = True

        stats = mask_depth_median(
            FakeDepthFrame(depth),
            mask,
            max_depth_m=1.5,
            erode_px=0,
        )

        self.assertEqual(stats.valid_count, 13)
        self.assertAlmostEqual(stats.median_m, 0.50, places=3)

    def test_deproject_pixel_with_supplied_median_depth(self):
        point = deproject_pixel_with_depth(
            FakeDepthFrame(np.ones((100, 160), dtype=np.float32)),
            100,
            70,
            0.50,
            intrinsics=PinholeIntrinsics(
                width=160,
                height=100,
                fx=400.0,
                fy=500.0,
                ppx=80.0,
                ppy=50.0,
                source="test",
            ),
        )

        self.assertEqual((point.u, point.v), (100, 70))
        self.assertAlmostEqual(point.depth_m, 0.50)
        self.assertAlmostEqual(point.x_m, 0.025)
        self.assertAlmostEqual(point.y_m, 0.020)
        self.assertAlmostEqual(point.z_m, 0.50)

    def test_segments_rank_by_mask_median_depth(self):
        near_mask = np.zeros((8, 8), dtype=bool)
        near_mask[1:4, 1:4] = True
        far_mask = np.zeros((8, 8), dtype=bool)
        far_mask[4:7, 4:7] = True
        depth = np.ones((8, 8), dtype=np.float32)
        depth[near_mask] = 0.25
        depth[far_mask] = 0.55

        targets, errors = segments_to_ranked_targets(
            [segment(far_mask, "far"), segment(near_mask, "near")],
            FakeDepthFrame(depth),
            intrinsics=PinholeIntrinsics(
                width=8,
                height=8,
                fx=100.0,
                fy=100.0,
                ppx=4.0,
                ppy=4.0,
            ),
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            camera_up_offset_mm=0.0,
            point_z_offset_mm=0.0,
            base_z_offset_mm=0.0,
        )

        self.assertEqual(errors, {})
        self.assertEqual([target.rank for target in targets], [1, 2])
        self.assertEqual([target.detection.label for target in targets], ["near", "far"])
        self.assertAlmostEqual(targets[0].raw_point.depth_m, 0.25, places=3)
        self.assertAlmostEqual(targets[1].raw_point.depth_m, 0.55, places=3)

    def test_command_carries_mask_metadata_through_protocol(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:4, 2:5] = True
        seg = segment(mask)
        target = RankedSegTarget(
            rank=2,
            segment=seg,
            raw_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((6, 6), 0.45, dtype=np.float32)),
                3,
                2,
                0.45,
                intrinsics=PinholeIntrinsics(
                    width=6,
                    height=6,
                    fx=100.0,
                    fy=100.0,
                    ppx=3.0,
                    ppy=3.0,
                ),
            ),
            send_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((6, 6), 0.45, dtype=np.float32)),
                3,
                2,
                0.45,
                intrinsics=PinholeIntrinsics(
                    width=6,
                    height=6,
                    fx=100.0,
                    fy=100.0,
                    ppx=3.0,
                    ppy=3.0,
                ),
            ),
            mask_depth_valid_count=9,
        )

        command = camera_point_command_from_target(
            target,
            source="yolo_seg_test",
            task="in",
        )
        parsed = camera_point_from_message(make_camera_point_message(command))

        self.assertEqual(parsed.source, "yolo_seg_test")
        self.assertEqual(parsed.task, "in")
        self.assertEqual(parsed.rank, 2)
        self.assertEqual(parsed.label, "cup")
        self.assertEqual(parsed.mask_area_px, 9)
        self.assertAlmostEqual(parsed.mask_centroid_u, seg.centroid_u)
        self.assertAlmostEqual(parsed.mask_centroid_v, seg.centroid_v)
        self.assertAlmostEqual(parsed.mask_depth_median_m, 0.45)
        self.assertEqual(parsed.mask_depth_valid_count, 9)

    def test_sender_dry_run_returns_payload_without_network(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        target = RankedSegTarget(
            rank=1,
            segment=segment(mask),
            raw_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            send_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            mask_depth_valid_count=4,
        )
        sender = YoloSegGraspSender(
            host=None,
            port=5005,
            timeout_s=1.0,
            dry_run=True,
            task="out",
        )

        response = sender.send_target(target, source="yolo_seg_test")

        self.assertEqual(response["type"], "dry_run")
        self.assertEqual(response["payload"]["rank"], 1)
        self.assertEqual(response["payload"]["source"], "yolo_seg_test")
        self.assertEqual(response["payload"]["mask_area_px"], 4)

    def test_clamps_centroid_to_depth_frame(self):
        self.assertEqual(
            clamp_pixel_from_centroid(-2.1, 20.8, width=10, height=8),
            (0, 7),
        )

    def test_base_z_offset_requires_hand_eye_transform(self):
        args = Namespace(
            mode="static",
            dry_run=True,
            host=None,
            timeout_s=1.0,
            width=640,
            height=480,
            fps=30,
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            max_results=1,
            min_mask_area_px=1,
            print_interval=1.0,
            base_z_offset_mm=20.0,
            hand_eye_result=None,
            task_option=None,
            no_hand_eye_result=False,
        )

        with self.assertRaisesRegex(ValueError, "--base-z-offset-mm requires"):
            validate_args(args)

    def test_dynamic_candidate_uses_mask_centroid_base_coordinates(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        target = RankedSegTarget(
            rank=1,
            segment=segment(mask),
            raw_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            send_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            mask_depth_valid_count=4,
        )
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.35, -0.45, -0.10])

        candidate = dynamic_seg_candidate_from_target(
            target,
            camera_to_base=camera_to_base,
            min_x_mm=300.0,
            max_x_mm=500.0,
            min_y_mm=-600.0,
            max_y_mm=-400.0,
            pick_y_mm=0.0,
            conveyor_speed_mm_s=225.0,
            conveyor_direction="positive-y",
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.base_x_mm, 350.0)
        self.assertAlmostEqual(candidate.base_y_mm, -450.0)
        self.assertAlmostEqual(candidate.base_z_mm, 300.0)
        self.assertAlmostEqual(candidate.remaining_y_mm, 450.0)
        self.assertAlmostEqual(candidate.t_arrive_s, 2.0)
        self.assertAlmostEqual(
            dynamic_pick_x_mm_for_candidate(
                candidate,
                fixed_pick_x_mm=360.0,
                use_mask_centroid_x=True,
            ),
            350.0,
        )

    def test_dynamic_command_carries_timing_pick_and_mask_metadata(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        target = RankedSegTarget(
            rank=1,
            segment=segment(mask),
            raw_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            send_point=deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            ),
            mask_depth_valid_count=4,
        )

        command = camera_point_command_from_target(
            target,
            source="yolo_seg_dynamic_timed",
            task="out",
            grasp_mode="dynamic",
            t_arrive_s=1.25,
            conveyor_speed_mm_s=223.0,
            pick_x_mm=350.0,
            pick_y_mm=0.0,
            detected_xyz_mm=(340.0, -420.0, 120.0),
        )
        parsed = camera_point_from_message(make_camera_point_message(command))

        self.assertEqual(parsed.grasp_mode, "dynamic")
        self.assertAlmostEqual(parsed.t_arrive_s, 1.25)
        self.assertAlmostEqual(parsed.conveyor_speed_mm_s, 223.0)
        self.assertAlmostEqual(parsed.pick_x_mm, 350.0)
        self.assertAlmostEqual(parsed.pick_y_mm, 0.0)
        self.assertAlmostEqual(parsed.detected_x_mm, 340.0)
        self.assertAlmostEqual(parsed.detected_y_mm, -420.0)
        self.assertAlmostEqual(parsed.detected_z_mm, 120.0)
        self.assertEqual(parsed.mask_area_px, 4)
        self.assertEqual(parsed.mask_depth_valid_count, 4)

    def test_dynamic_mode_defaults_hand_eye_result_and_requires_speed(self):
        args = Namespace(
            mode="dynamic",
            dry_run=True,
            host=None,
            timeout_s=1.0,
            width=640,
            height=480,
            fps=30,
            max_depth_m=1.5,
            min_depth_valid_px=1,
            mask_depth_erode_px=0,
            max_results=1,
            min_mask_area_px=1,
            print_interval=1.0,
            base_z_offset_mm=0.0,
            hand_eye_result=None,
            no_hand_eye_result=False,
            conveyor_speed_mm_s=None,
            dynamic_min_x_mm=300.0,
            dynamic_max_x_mm=500.0,
            dynamic_min_y_mm=-600.0,
            dynamic_max_y_mm=-400.0,
            dynamic_skip_memory_s=5.0,
            dynamic_skip_radius_mm=80.0,
            dynamic_probe_interval_s=1.0,
            task_option=None,
        )

        with self.assertRaisesRegex(ValueError, "--conveyor-speed-mm-s"):
            validate_args(args)

        self.assertIsNotNone(args.hand_eye_result)

    def test_dynamic_candidates_sort_by_arrival_then_rank(self):
        def make_target(rank: int, y_m: float) -> RankedSegTarget:
            mask = np.zeros((4, 4), dtype=bool)
            mask[1:3, 1:3] = True
            point = deproject_pixel_with_depth(
                FakeDepthFrame(np.full((4, 4), 0.40, dtype=np.float32)),
                2,
                2,
                0.40,
                intrinsics=PinholeIntrinsics(
                    width=4,
                    height=4,
                    fx=100.0,
                    fy=100.0,
                    ppx=2.0,
                    ppy=2.0,
                ),
            )
            point = type(point)(
                u=point.u,
                v=point.v,
                depth_m=point.depth_m,
                x_m=0.35,
                y_m=y_m,
                z_m=point.z_m,
            )
            return RankedSegTarget(
                rank=rank,
                segment=segment(mask),
                raw_point=point,
                send_point=point,
                mask_depth_valid_count=4,
            )

        candidates = find_dynamic_seg_candidates(
            [make_target(2, -0.40), make_target(1, -0.50)],
            camera_to_base=np.eye(4, dtype=float),
            min_x_mm=300.0,
            max_x_mm=500.0,
            min_y_mm=-600.0,
            max_y_mm=-300.0,
            pick_y_mm=0.0,
            conveyor_speed_mm_s=100.0,
            conveyor_direction="positive-y",
        )

        self.assertEqual([candidate.target.rank for candidate in candidates], [2, 1])


if __name__ == "__main__":
    unittest.main()

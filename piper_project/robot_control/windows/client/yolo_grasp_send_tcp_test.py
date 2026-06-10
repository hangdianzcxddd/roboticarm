from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from robot_control.windows.client.yolo_grasp_send_tcp import (
    RankedTarget,
    YoloGraspSender,
    apply_base_z_offset,
    build_parser,
    camera_point_command_from_target,
    detections_to_ranked_targets,
    dynamic_arrival_time_s,
    dynamic_candidate_from_target,
    dynamic_candidate_is_suppressed,
    dynamic_pick_x_mm_for_candidate,
    dynamic_skip_reason_from_response,
    remember_suppressed_dynamic_target,
    grasp_pixel_from_detection,
    offset_camera_point,
    parse_target_command,
    process_terminal_command,
    send_dynamic_probe,
    validate_args,
)
from robot_control.windows.vision.depth import CameraPoint
from robot_control.windows.vision.detector import Detection


def detection(
    *,
    u: int = 50,
    v: int = 60,
    x: int = 20,
    y: int = 30,
    width: int = 60,
    height: int = 80,
    label: str = "target",
    confidence: float = 0.5,
) -> Detection:
    return Detection(
        u=u,
        v=v,
        x=x,
        y=y,
        width=width,
        height=height,
        area=float(width * height),
        label=label,
        confidence=confidence,
    )


def camera_point(u: int, v: int, depth_m: float) -> CameraPoint:
    return CameraPoint(
        u=u,
        v=v,
        depth_m=depth_m,
        x_m=0.01,
        y_m=0.02,
        z_m=depth_m,
    )


class TestYoloGraspSendTcp(unittest.TestCase):
    def test_grasp_pixel_moves_up_from_bbox_center(self):
        selected = grasp_pixel_from_detection(
            detection(x=10, y=20, width=100, height=80),
            bbox_up_ratio=0.25,
            bbox_up_px=5.0,
        )

        self.assertEqual(selected, (60, 35))

    def test_offset_camera_point_moves_in_image_up_direction(self):
        point = camera_point(10, 20, 0.50)

        shifted = offset_camera_point(
            point,
            camera_up_offset_mm=20.0,
            point_z_offset_mm=5.0,
        )

        self.assertEqual((shifted.u, shifted.v), (10, 20))
        self.assertAlmostEqual(shifted.depth_m, 0.50)
        self.assertAlmostEqual(shifted.x_m, 0.01)
        self.assertAlmostEqual(shifted.y_m, 0.00)
        self.assertAlmostEqual(shifted.z_m, 0.505)

    def test_apply_base_z_offset_adjusts_camera_xyz_through_hand_eye_transform(self):
        point = camera_point(10, 20, 0.50)
        camera_to_base = np.eye(4, dtype=float)
        camera_to_base[:3, 3] = np.array([0.10, -0.20, 0.30])

        shifted = apply_base_z_offset(
            point,
            camera_to_base=camera_to_base,
            base_z_offset_mm=20.0,
        )

        self.assertAlmostEqual(shifted.x_m, point.x_m)
        self.assertAlmostEqual(shifted.y_m, point.y_m)
        self.assertAlmostEqual(shifted.z_m, point.z_m + 0.02)

    def test_detections_are_ranked_nearest_depth_first(self):
        detections = [
            detection(x=-20, width=60, label="far"),
            detection(x=-10, width=60, label="near"),
            detection(x=0, width=60, label="middle"),
        ]
        depths_by_u = {10: 0.30, 20: 0.10, 30: 0.20}

        def fake_deproject(depth_frame, u, v, sample_radius, intrinsics):
            del depth_frame, v, sample_radius, intrinsics
            return camera_point(u, 99, depths_by_u[u])

        with patch(
            "robot_control.windows.client.yolo_grasp_send_tcp.deproject_pixel_to_camera_xyz",
            side_effect=fake_deproject,
        ):
            targets, errors = detections_to_ranked_targets(
                detections,
                depth_frame=object(),
                sample_radius=3,
                intrinsics=None,
                bbox_up_ratio=0.0,
                bbox_up_px=0.0,
                camera_up_offset_mm=0.0,
                point_z_offset_mm=0.0,
                base_z_offset_mm=0.0,
            )

        self.assertEqual(errors, {})
        self.assertEqual([target.rank for target in targets], [1, 2, 3])
        self.assertEqual([target.detection.label for target in targets], ["near", "middle", "far"])
        self.assertEqual([target.depth_mm for target in targets], [100.0, 200.0, 300.0])

    def test_base_z_offset_requires_hand_eye_transform(self):
        class Args:
            dry_run = True
            host = None
            timeout_s = 1.0
            sample_radius = 0
            max_results = 1
            bbox_up_ratio = 0.0
            bbox_up_px = 0.0
            print_interval = 1.0
            base_z_offset_mm = 20.0
            hand_eye_result = None

        with self.assertRaisesRegex(ValueError, "--base-z-offset-mm requires"):
            validate_args(Args())

    def test_dynamic_mode_requires_conveyor_speed_input(self):
        args = build_parser().parse_args(["dynamic", "--dry-run"])

        with self.assertRaisesRegex(ValueError, "--conveyor-speed-mm-s is required"):
            validate_args(args)

    def test_target_converts_to_click_compatible_camera_point_command(self):
        target = RankedTarget(
            rank=2,
            detection=detection(label="bottle"),
            pixel=(11, 22),
            raw_point=camera_point(11, 22, 0.55),
            send_point=CameraPoint(
                u=11,
                v=22,
                depth_m=0.55,
                x_m=0.10,
                y_m=-0.02,
                z_m=0.55,
            ),
        )

        command = camera_point_command_from_target(
            target,
            source="yolo_test",
            task="in",
        )

        self.assertEqual(command.u, 11)
        self.assertEqual(command.v, 22)
        self.assertEqual(command.depth_m, 0.55)
        self.assertEqual(command.x_m, 0.10)
        self.assertEqual(command.y_m, -0.02)
        self.assertEqual(command.z_m, 0.55)
        self.assertEqual(command.source, "yolo_test")
        self.assertEqual(command.task, "in")
        self.assertEqual(command.rank, 2)
        self.assertEqual(command.label, "bottle")
        self.assertEqual(command.bbox_xywh, (20.0, 30.0, 60.0, 80.0))

    def test_dynamic_arrival_time_uses_y_distance_to_pick_line(self):
        t_arrive, remaining = dynamic_arrival_time_s(
            20.0,
            pick_y_mm=-30.0,
            conveyor_speed_mm_s=100.0,
            conveyor_direction="negative-y",
        )

        self.assertAlmostEqual(remaining, 50.0)
        self.assertAlmostEqual(t_arrive, 0.5)

    def test_dynamic_candidate_accepts_configured_base_window(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        center_point = CameraPoint(
            u=target.detection.u,
            v=target.detection.v,
            depth_m=0.40,
            x_m=0.35,
            y_m=0.02,
            z_m=0.12,
        )

        candidate = dynamic_candidate_from_target(
            target,
            center_point,
            camera_to_base=np.eye(4, dtype=float),
            min_x_mm=300.0,
            max_x_mm=400.0,
            min_y_mm=-30.0,
            max_y_mm=30.0,
            pick_y_mm=-30.0,
            conveyor_speed_mm_s=100.0,
            conveyor_direction="negative-y",
        )

        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertAlmostEqual(candidate.base_x_mm, 350.0)
        self.assertAlmostEqual(candidate.base_y_mm, 20.0)
        self.assertAlmostEqual(candidate.remaining_y_mm, 50.0)
        self.assertAlmostEqual(candidate.t_arrive_s, 0.5)

    def test_dynamic_pick_x_can_use_bbox_center_base_x(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=camera_point(10, 20, 0.40),
        )
        candidate = dynamic_candidate_from_target(
            target,
            CameraPoint(
                u=50,
                v=60,
                depth_m=0.40,
                x_m=0.405,
                y_m=-0.50,
                z_m=0.12,
            ),
            camera_to_base=np.eye(4, dtype=float),
            min_x_mm=300.0,
            max_x_mm=500.0,
            min_y_mm=-600.0,
            max_y_mm=-400.0,
            pick_y_mm=-30.0,
            conveyor_speed_mm_s=223.0,
            conveyor_direction="positive-y",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None

        self.assertAlmostEqual(
            dynamic_pick_x_mm_for_candidate(
                candidate,
                fixed_pick_x_mm=350.0,
                use_bbox_center_x=False,
            ),
            350.0,
        )
        self.assertAlmostEqual(
            dynamic_pick_x_mm_for_candidate(
                candidate,
                fixed_pick_x_mm=350.0,
                use_bbox_center_x=True,
            ),
            405.0,
        )

    def test_dynamic_candidate_rejects_outside_x_range(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=camera_point(10, 20, 0.40),
        )

        candidate = dynamic_candidate_from_target(
            target,
            CameraPoint(
                u=50,
                v=60,
                depth_m=0.40,
                x_m=0.25,
                y_m=0.02,
                z_m=0.12,
            ),
            camera_to_base=np.eye(4, dtype=float),
            min_x_mm=300.0,
            max_x_mm=400.0,
            min_y_mm=-30.0,
            max_y_mm=30.0,
            pick_y_mm=-30.0,
            conveyor_speed_mm_s=100.0,
            conveyor_direction="negative-y",
        )

        self.assertIsNone(candidate)

    def test_dynamic_skip_response_can_suppress_repeated_candidate(self):
        response = {
            "type": "ack",
            "payload": {
                "skipped": True,
                "dynamic_timing": {"skip_reason": "too close"},
            },
        }

        reason = dynamic_skip_reason_from_response(response)

        self.assertEqual(reason, "too close")

    def test_dynamic_suppression_matches_nearby_base_xy(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=camera_point(10, 20, 0.40),
        )
        candidate = dynamic_candidate_from_target(
            target,
            CameraPoint(
                u=50,
                v=60,
                depth_m=0.40,
                x_m=0.35,
                y_m=0.02,
                z_m=0.12,
            ),
            camera_to_base=np.eye(4, dtype=float),
            min_x_mm=300.0,
            max_x_mm=400.0,
            min_y_mm=-30.0,
            max_y_mm=30.0,
            pick_y_mm=-30.0,
            conveyor_speed_mm_s=100.0,
            conveyor_direction="negative-y",
        )
        self.assertIsNotNone(candidate)
        assert candidate is not None

        suppressed = remember_suppressed_dynamic_target(
            [],
            candidate,
            now_s=10.0,
            memory_s=5.0,
        )

        self.assertTrue(
            dynamic_candidate_is_suppressed(
                candidate,
                suppressed,
                radius_mm=80.0,
            )
        )

    def test_parse_target_command_accepts_task_and_rank(self):
        self.assertEqual(parse_target_command("out 1", "in"), ("out", 1))
        self.assertEqual(parse_target_command("in 2", "out"), ("in", 2))
        self.assertEqual(parse_target_command("3", "out"), ("out", 3))
        self.assertIsNone(parse_target_command("sideways 1", "out"))
        self.assertIsNone(parse_target_command("in two", "out"))

    def test_sender_dry_run_returns_payload_without_network(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        sender = YoloGraspSender(
            host=None,
            port=5005,
            timeout_s=1.0,
            dry_run=True,
            task="in",
        )

        response = sender.send_target(target, source="yolo_test")

        self.assertEqual(response["type"], "dry_run")
        self.assertEqual(response["payload"]["rank"], 1)
        self.assertEqual(response["payload"]["task"], "in")
        self.assertEqual(response["payload"]["label"], "cup")
        self.assertEqual(response["payload"]["u"], 10)
        self.assertEqual(response["payload"]["x_m"], 0.1)

    def test_dynamic_probe_sends_probe_grasp_mode(self):
        target = RankedTarget(
            rank=1,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        sent = []

        def fake_send(host, port, command, timeout_s):
            del host, port, timeout_s
            sent.append(command)
            return {"type": "ack", "payload": {"base_xyz_m": {}}}

        sender = YoloGraspSender(
            host="127.0.0.1",
            port=5005,
            timeout_s=1.0,
            dry_run=False,
            task="out",
            send_func=fake_send,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            response = send_dynamic_probe(
                sender,
                target,
                source="yolo_dynamic_timed",
                task="out",
                center_point=CameraPoint(
                    u=50,
                    v=60,
                    depth_m=0.42,
                    x_m=0.3,
                    y_m=-0.1,
                    z_m=0.42,
                ),
            )

        self.assertEqual(response["type"], "ack")
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0].grasp_mode, "probe")
        self.assertEqual((sent[0].u, sent[0].v), (50, 60))
        self.assertAlmostEqual(sent[0].x_m, 0.3)
        self.assertAlmostEqual(sent[0].y_m, -0.1)
        self.assertIsNone(sent[0].pick_x_mm)
        self.assertIsNone(sent[0].pick_y_mm)

    def test_terminal_command_sends_task_from_command_prefix(self):
        target = RankedTarget(
            rank=2,
            detection=detection(label="cup"),
            pixel=(10, 20),
            raw_point=camera_point(10, 20, 0.40),
            send_point=CameraPoint(
                u=10,
                v=20,
                depth_m=0.40,
                x_m=0.1,
                y_m=0.2,
                z_m=0.4,
            ),
        )
        sent: list[tuple[int, str]] = []

        def fake_send(host, port, command, timeout_s):
            del host, port, timeout_s
            sent.append((command.rank, command.task))
            return {"type": "ack", "payload": {}}

        sender = YoloGraspSender(
            host="127.0.0.1",
            port=5005,
            timeout_s=1.0,
            dry_run=False,
            task="out",
            send_func=fake_send,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            keep_running = process_terminal_command(
                "in 2",
                targets=[target],
                errors={},
                sender=sender,
                source="yolo_test",
            )

        self.assertTrue(keep_running)
        self.assertEqual(sent, [(2, "in")])


if __name__ == "__main__":
    unittest.main()

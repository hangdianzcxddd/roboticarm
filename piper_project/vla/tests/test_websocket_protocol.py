import unittest

import numpy as np

from vla.cloud_protocol import WebSocketCloudClient, make_lingbot_observation, parse_action_response
from vla.deploy.msgpack_numpy import Packer, unpackb


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.timeout = None

    def settimeout(self, value):
        self.timeout = value

    def send(self, value, **kwargs):
        self.sent.append(value)

    def send_binary(self, value):
        self.sent.append(value)

    def recv(self):
        return '{"ok":true,"action_chunk":[[0,1,2,3,4,5,6]]}'

    def close(self):
        pass


class FakeLingbotSocket(FakeSocket):
    def __init__(self):
        super().__init__()
        self.replies = [
            Packer().pack({}),
            Packer().pack({"action": np.zeros((2, 7), dtype=np.float32)}),
        ]
        self.binary_sent = []

    def send_binary(self, value):
        self.binary_sent.append(value)

    def recv(self):
        return self.replies.pop(0)


class WebSocketProtocolTest(unittest.TestCase):
    def test_official_observation_schema(self):
        obs = make_lingbot_observation(
            np.zeros((10, 12, 3), dtype=np.uint8),
            np.zeros(7, dtype=np.float32),
            "pick",
        )
        self.assertEqual(obs["observation.images.cam_high"].shape, (10, 12, 3))
        self.assertEqual(obs["observation.images.cam_high"].dtype, np.uint8)
        self.assertEqual(obs["observation.state"].shape, (7,))
        self.assertEqual(obs["task"], "pick")

    def test_mirror_single_camera_to_three_keys(self):
        obs = make_lingbot_observation(
            np.zeros((4, 5, 3), dtype=np.uint8),
            np.zeros(7, dtype=np.float32),
            "pick",
            mirror_wrist_images=True,
        )
        self.assertEqual(
            {key for key in obs if key.startswith("observation.images.")},
            {
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
            },
        )
        np.testing.assert_array_equal(
            obs["observation.images.cam_high"],
            obs["observation.images.cam_left_wrist"],
        )

    def test_response_variants(self):
        for raw in (
            '{"ok":true,"action_chunk":[[0,1,2,3,4,5,6]]}',
            '{"action":[0,1,2,3,4,5]}',
            '[[0,1,2,3,4,5,6]]',
        ):
            chunk, _ = parse_action_response(raw)
            self.assertEqual(chunk.shape[1], 6 if 'action"' in raw else 7)

    def test_send_keeps_observation_contract(self):
        socket = FakeSocket()
        client = WebSocketCloudClient(
            "ws://127.0.0.1:9330",
            protocol="json",
            websocket_factory=lambda *args, **kwargs: socket,
        )
        client.connect()
        chunk, _ = client.send_observation(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros(7),
            "pick",
        )
        self.assertEqual(chunk.shape, (1, 7))
        self.assertEqual(set(__import__("json").loads(socket.sent[0])), {"image_b64", "state", "instruction"})

    def test_msgpack_send_uses_binary_robotwin_contract(self):
        socket = FakeLingbotSocket()
        client = WebSocketCloudClient(
            "ws://127.0.0.1:9330",
            protocol="lingbot_msgpack",
            mirror_wrist_images=True,
            websocket_factory=lambda *args, **kwargs: socket,
        )
        client.connect()
        chunk, _ = client.send_observation(
            np.zeros((2, 3, 3), dtype=np.uint8),
            np.zeros(7, dtype=np.float32),
            "pick",
        )

        self.assertEqual(chunk.shape, (2, 7))
        self.assertEqual(len(socket.binary_sent), 1)
        payload = unpackb(socket.binary_sent[0], raw=False)
        self.assertEqual(
            set(payload),
            {
                "observation.images.cam_high",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
                "observation.state",
                "task",
            },
        )
        self.assertEqual(payload["observation.images.cam_high"].shape, (2, 3, 3))
        self.assertEqual(payload["observation.state"].shape, (7,))
        self.assertEqual(payload["task"], "pick")


if __name__ == "__main__":
    unittest.main()

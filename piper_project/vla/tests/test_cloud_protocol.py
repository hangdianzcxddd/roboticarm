import unittest
from unittest.mock import Mock, patch

import numpy as np

from vla.cloud_protocol import make_inference_payload, request_cloud_action


class CloudProtocolTest(unittest.TestCase):
    def test_request_keeps_existing_three_fields_and_7d_state(self):
        payload = make_inference_payload(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros(7), "pick")
        self.assertEqual(set(payload), {"image_b64", "state", "instruction"})
        self.assertEqual(len(payload["state"]), 7)

    @patch("vla.cloud_protocol.requests.post")
    def test_parse_action_chunk(self, post):
        response = Mock()
        response.json.return_value = {"ok": True, "action_chunk": [[0.0] * 7]}
        response.raise_for_status.return_value = None
        post.return_value = response
        chunk, _ = request_cloud_action("http://cloud/infer", np.zeros((2, 2, 3), dtype=np.uint8), np.zeros(7), "pick", 1)
        self.assertEqual(chunk.shape, (1, 7))


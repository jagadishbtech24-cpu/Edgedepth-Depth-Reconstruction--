import unittest

import numpy as np

from src.models.guided_depth_net import EdgeGuideNet
from src.models.weights_loader import run_dl_inference


class ModelInferenceTests(unittest.TestCase):
    def test_inference_preserves_odd_input_dimensions(self):
        model = EdgeGuideNet(base_channels=4)
        depth = np.full((35, 37), 2.0, dtype=np.float32)
        rgb = np.zeros((35, 37, 3), dtype=np.float32)
        mask = np.ones_like(depth, dtype=np.float32)

        output, confidence = run_dl_inference(model, depth, rgb, mask=mask, device="cpu")

        self.assertEqual(output.shape, depth.shape)
        self.assertEqual(confidence.shape, depth.shape)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.isfinite(confidence).all())


if __name__ == "__main__":
    unittest.main()

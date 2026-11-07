import os
import unittest

import numpy as np
import torch

from src.models.depth_unet import (
    DEFAULT_DEPTH_UNET_WEIGHTS,
    DEPTH_UNET_INPUT_SIZE,
    DepthUNet,
    load_depth_unet,
    prepare_depth_unet_input,
    run_depth_unet_inference,
    scatter_sparse_depth,
)
from src.utils.dataset import load_depth_map, load_rgb_image
from src.utils.degradations import downsample_depth


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PREFIX = os.path.join(ROOT, "data", "samples", "nyuv2_nyu_0000")


class DepthUNetTests(unittest.TestCase):
    def test_double_conv_uses_bias_free_convolutions_and_checkpoint_names(self):
        model = DepthUNet()
        self.assertIn("enc1.block.0.weight", model.state_dict())
        self.assertNotIn("enc1.block.0.bias", model.state_dict())
        self.assertIn("enc1.block.1.running_mean", model.state_dict())
        self.assertIn("up4.weight", model.state_dict())

    def test_preprocessing_builds_rgb_depth_mask_channels(self):
        rgb = np.full((240, 320, 3), 0.25, dtype=np.float32)
        depth = np.zeros((240, 320), dtype=np.float32)
        depth[4, 8] = 5.0
        mask = np.zeros_like(depth)
        mask[4, 8] = 1.0

        model_input = prepare_depth_unet_input(rgb, depth, mask)

        self.assertEqual(tuple(model_input.shape), (1, 5, 240, 320))
        self.assertTrue(torch.allclose(model_input[0, :3, 0, 0], torch.full((3,), 0.25)))
        self.assertEqual(float(model_input[0, 3, 4, 8]), 0.5)
        self.assertEqual(float(model_input[0, 4, 4, 8]), 1.0)
        self.assertEqual(float(model_input[0, 3, 0, 0]), 0.0)
        self.assertEqual(float(model_input[0, 4, 0, 0]), 0.0)

    def test_sparse_scatter_supports_training_scales(self):
        for scale in (2, 4, 8):
            with self.subTest(scale=scale):
                source = np.full((16 // scale, 24 // scale), 2.0, dtype=np.float32)
                sparse, mask = scatter_sparse_depth(source, (16, 24), scale)
                self.assertEqual(int(mask.sum()), source.size)
                self.assertEqual(float(sparse[0, 0]), 2.0)
                self.assertEqual(float(mask[0, 0]), 1.0)
                self.assertEqual(float(mask[1, 1]), 0.0)

    def test_scale_grid_can_be_applied_to_full_resolution_reference(self):
        source = np.full((16, 24), 2.0, dtype=np.float32)
        sparse, mask = scatter_sparse_depth(
            source, (16, 24), 4, force_sample_grid=True
        )
        self.assertEqual(int(mask.sum()), 24)
        self.assertEqual(float(mask[0, 0]), 1.0)
        self.assertEqual(float(mask[0, 1]), 0.0)
        self.assertEqual(float(mask[0, 4]), 1.0)

    @unittest.skipUnless(
        os.path.isfile(DEFAULT_DEPTH_UNET_WEIGHTS)
        and os.path.isfile(SAMPLE_PREFIX + "_rgb.png")
        and os.path.isfile(SAMPLE_PREFIX + "_depth.npy"),
        "Supplied DepthUNet checkpoint and NYU sample are required.",
    )
    def test_checkpoint_strict_load_and_real_sample_inference(self):
        model = load_depth_unet(device="cpu")
        self.assertIsInstance(model, torch.nn.DataParallel)

        rgb = load_rgb_image(SAMPLE_PREFIX + "_rgb.png")
        reference = load_depth_map(SAMPLE_PREFIX + "_depth.npy")
        self.assertEqual(reference.shape, rgb.shape[:2])
        low_depth = downsample_depth(reference, scale_factor=4)
        sparse, mask = scatter_sparse_depth(low_depth, DEPTH_UNET_INPUT_SIZE, 4)

        prediction = run_depth_unet_inference(model, rgb, sparse, mask=mask, device="cpu")

        self.assertEqual(prediction.shape, reference.shape)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 10.0)


if __name__ == "__main__":
    unittest.main()

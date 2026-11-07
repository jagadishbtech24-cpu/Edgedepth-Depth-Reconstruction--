import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.models.weights_loader import DEFAULT_WEIGHTS_PATH
from src.services.reconstruction import (
    PROJECT_ROOT,
    reconstruct,
    resolve_edgeguide_checkpoint,
)


class ReconstructionCheckpointTests(unittest.TestCase):
    def test_default_edgeguide_checkpoint_is_project_relative_and_present(self):
        checkpoint = resolve_edgeguide_checkpoint()

        self.assertEqual(checkpoint, Path(DEFAULT_WEIGHTS_PATH).resolve())
        self.assertEqual(checkpoint, (PROJECT_ROOT / "models" / "edge_guide_net.pt").resolve())
        self.assertTrue(checkpoint.is_file())

    def test_relative_override_resolves_from_project_root(self):
        with patch.dict(os.environ, {"EDGEDEPTH_CHECKPOINT": r"models\edge_guide_net_nyu_v2.pt"}):
            checkpoint = resolve_edgeguide_checkpoint()

        self.assertEqual(
            checkpoint,
            (PROJECT_ROOT / "models" / "edge_guide_net_nyu_v2.pt").resolve(),
        )

    def test_missing_edgeguide_keeps_classical_and_depthunet_methods_available(self):
        rgb = np.full((48, 64, 3), 0.4, dtype=np.float32)
        sparse_depth = np.zeros((48, 64), dtype=np.float32)
        sparse_depth[::8, ::8] = 2.0
        depthunet_prediction = np.full((48, 64), 1.75, dtype=np.float32)
        missing_checkpoint = r"models\edge_guide_net_nyu_v2.pt"

        with (
            patch.dict(os.environ, {"EDGEDEPTH_CHECKPOINT": missing_checkpoint}),
            patch("src.services.reconstruction._load_depth_unet", return_value=object()),
            patch(
                "src.services.reconstruction.run_depth_unet_inference",
                return_value=depthunet_prediction,
            ),
        ):
            result = reconstruct(
                rgb,
                sparse_depth,
                settings={"scale": 4},
                sensor_is_low_resolution=True,
            )

        self.assertIn("Nearest", result["predictions"])
        self.assertIn("Guided filter", result["predictions"])
        self.assertIn("DepthUNet", result["predictions"])
        self.assertNotIn("EdgeGuideNet", result["predictions"])
        self.assertNotIn("EdgeDepth hybrid", result["predictions"])
        self.assertIsNone(result["affinity"])
        self.assertTrue(result["warnings"])
        self.assertIn("models\\edge_guide_net_nyu_v2.pt", result["warnings"][0])
        self.assertIn(DEFAULT_WEIGHTS_PATH, result["warnings"][0])


if __name__ == "__main__":
    unittest.main()

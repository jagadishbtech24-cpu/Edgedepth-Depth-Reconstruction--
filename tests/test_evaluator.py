import unittest

import numpy as np

from src.metrics.evaluator import (
    compute_boundary_f1,
    compute_rmse,
    evaluate_all_metrics,
)


class EvaluatorTests(unittest.TestCase):
    def test_metrics_ignore_invalid_ground_truth_and_explicit_mask(self):
        gt = np.full((16, 16), 2.0, dtype=np.float32)
        pred = gt.copy()
        pred[2, 2] = 50.0
        gt[2, 2] = 0.0
        mask = np.ones_like(gt, dtype=bool)
        mask[3, 3] = False
        pred[3, 3] = 100.0

        metrics = evaluate_all_metrics(pred, gt, mask)

        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["Valid_Pixels"], gt.size - 2)

    def test_flat_depth_has_no_false_boundaries(self):
        depth = np.full((20, 20), 1.5, dtype=np.float32)
        result = compute_boundary_f1(depth, depth)

        self.assertEqual(result["f1_score"], 1.0)
        self.assertTrue(np.isnan(evaluate_all_metrics(depth, depth)["Edge_RMSE"]))

    def test_boundary_metrics_detect_a_depth_step(self):
        gt = np.ones((32, 32), dtype=np.float32)
        gt[:, 16:] = 3.0

        result = compute_boundary_f1(gt.copy(), gt)

        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1_score"], 1.0)
        self.assertEqual(compute_rmse(gt, gt), 0.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_rmse(np.ones((4, 4)), np.ones((4, 5)))


if __name__ == "__main__":
    unittest.main()

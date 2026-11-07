import os
import tempfile
import unittest

import cv2
import numpy as np

from src.utils.dataset import get_full_nyu_v2_scenes
from train import _find_pairs


class DatasetRootTests(unittest.TestCase):
    def test_explicit_dataset_root_supports_direct_and_nested_layouts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = os.path.join(temp_dir, "nyu-depth-png")
            train_image_dir = os.path.join(dataset_root, "images", "train")
            train_depth_dir = os.path.join(dataset_root, "depth", "train")
            val_image_dir = os.path.join(dataset_root, "images", "val")
            val_depth_dir = os.path.join(dataset_root, "depth", "val")
            for path in (train_image_dir, train_depth_dir, val_image_dir, val_depth_dir):
                os.makedirs(path)
            for index in range(10):
                scene = f"scene_{index:04d}"
                cv2.imwrite(
                    os.path.join(train_image_dir, f"{scene}.jpg"),
                    np.zeros((8, 8, 3), dtype=np.uint8),
                )
                cv2.imwrite(
                    os.path.join(train_depth_dir, f"{scene}.png"),
                    np.full((8, 8), 1000, dtype=np.uint16),
                )
            cv2.imwrite(
                os.path.join(val_image_dir, "scene_1000.jpg"),
                np.zeros((8, 8, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                os.path.join(val_depth_dir, "scene_1000.png"),
                np.full((8, 8), 1000, dtype=np.uint16),
            )

            samples_dir = os.path.join(temp_dir, "project", "data", "samples")
            os.makedirs(samples_dir)

            self.assertEqual(len(_find_pairs(samples_dir, "train", dataset_root)), 9)
            self.assertEqual(len(_find_pairs(samples_dir, "val", dataset_root)), 1)
            self.assertEqual(
                len(get_full_nyu_v2_scenes("val", 10, dataset_root=dataset_root)), 1
            )
            self.assertEqual(
                len(get_full_nyu_v2_scenes("val", 10, dataset_root=temp_dir)), 1
            )

    def test_explicit_invalid_dataset_root_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                _find_pairs(temp_dir, "train", dataset_root=temp_dir)


if __name__ == "__main__":
    unittest.main()

"""
Dataset and File I/O utilities for RGB-D pairs.
Loads NYUv2 (full 1449 train/val pairs), Middlebury, KITTI, and custom uploaded formats (.png, .npy, .jpg).
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image

def load_rgb_image(path: str) -> np.ndarray:
    """Loads RGB image as float32 in [0, 1]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"RGB image not found at: {path}")
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image from {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def load_depth_map(path: str) -> np.ndarray:
    """
    Loads depth map:
    - If .npy: loaded as float32 array directly (meters).
    - If .png/.tiff: checked for 16-bit (mm -> meters) or 8-bit.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Depth map not found at: {path}")
        
    if path.endswith('.npy'):
        depth = np.load(path).astype(np.float32)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read depth map from {path}")
            
        if img.ndim == 3:
            img = img[:, :, 0]
            
        if img.dtype == np.uint16:
            # Official NYU Depth V2 standard: 16-bit millimeter depth (1000 = 1.0 meter)
            depth = img.astype(np.float32) / 1000.0
        elif img.dtype == np.uint8:
            depth = img.astype(np.float32) / 255.0 * 5.0
        else:
            depth = img.astype(np.float32)
            
    return depth


def get_available_presets(samples_dir: str = None) -> dict:
    """Discovers available preset sample datasets in data/samples/."""
    if samples_dir is None:
        samples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'samples')
        
    presets = {}
    if not os.path.exists(samples_dir):
        return presets
        
    rgb_files = sorted(glob.glob(os.path.join(samples_dir, "*_rgb.png")))
    for rgb_path in rgb_files:
        base = os.path.basename(rgb_path).replace("_rgb.png", "")
        depth_npy = os.path.join(samples_dir, f"{base}_depth.npy")
        depth_png = os.path.join(samples_dir, f"{base}_depth.png")
        
        depth_path = depth_npy if os.path.exists(depth_npy) else (depth_png if os.path.exists(depth_png) else None)
        if depth_path:
            presets[base] = {
                'name': base.replace("_", " ").title(),
                'rgb_path': rgb_path,
                'depth_path': depth_path
            }
            
    return presets


def get_full_nyu_v2_scenes(split: str = 'val', max_count: int = 100,
                           dataset_root: str = None) -> dict:
    """
    Returns available scenes from the official NYU Depth V2 dataset folder.
    split: 'val' (654 scenes) or 'train' (795 scenes).
    """
    if dataset_root is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'nyu_depth_v2', 'nyu-depth-png'
        )
    else:
        dataset_root = os.path.abspath(dataset_root)
        nested_root = os.path.join(dataset_root, 'nyu-depth-png')
        if os.path.isdir(nested_root):
            base_dir = nested_root
        elif os.path.isdir(os.path.join(dataset_root, 'images')) and os.path.isdir(
            os.path.join(dataset_root, 'depth')
        ):
            base_dir = dataset_root
        else:
            raise FileNotFoundError(
                f"Expected images/ and depth/ folders, or a nyu-depth-png subfolder, "
                f"under dataset_root: {dataset_root}"
            )
    img_dir = os.path.join(base_dir, 'images', split)
    depth_dir = os.path.join(base_dir, 'depth', split)
    
    scenes = {}
    if not (os.path.exists(img_dir) and os.path.exists(depth_dir)):
        return scenes
        
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:max_count]
    for img_p in img_files:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        depth_p = os.path.join(depth_dir, f"{stem}.png")
        if os.path.exists(depth_p):
            scenes[f"nyu_official_{split}_{stem}"] = {
                'name': f"NYUv2 Official ({split}): {stem}",
                'rgb_path': img_p,
                'depth_path': depth_p
            }
            
    return scenes

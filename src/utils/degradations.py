"""
Degradation Simulator for Depth Maps.
Simulates real-world sensor imperfections:
- Downsampling (2x, 4x, 8x, 16x)
- Sparse LiDAR sampling (uniform, random, scanline)
- Missing depth values / sensor dropout (holes, object shadows)
- Sensor noise (Gaussian, speckle, salt-and-pepper)
"""

import numpy as np
import cv2

def downsample_depth(depth: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Subsamples depth map by an integer scale factor (e.g. 2, 4, 8).
    Uses nearest-neighbor subsampling to simulate low-resolution depth sensor grid.
    """
    H, W = depth.shape[:2]
    new_h = max(1, H // scale_factor)
    new_w = max(1, W // scale_factor)
    lr_depth = cv2.resize(depth.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return lr_depth


def simulate_sparse_depth(depth: np.ndarray, sparsity_ratio: float = 0.05,
                          pattern: str = 'random', rng=None) -> tuple:
    """
    Simulates sparse depth sensor (such as dToF or LiDAR).
    
    Args:
        depth: High-resolution ground truth depth
        sparsity_ratio: fraction of pixels observed (e.g. 0.05 = 5%)
        pattern: 'random', 'grid', or 'scanline'
        
    Returns:
        (sparse_depth, valid_mask)
    """
    H, W = depth.shape[:2]
    rng = rng or np.random.default_rng()
    sparse_d = np.zeros_like(depth, dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=bool)
    
    if pattern == 'random':
        num_samples = int(H * W * sparsity_ratio)
        flat_indices = rng.choice(H * W, size=num_samples, replace=False)
        ys = flat_indices // W
        xs = flat_indices % W
        valid_mask[ys, xs] = True
        sparse_d[valid_mask] = depth[valid_mask]
        
    elif pattern == 'scanline':
        # Simulates spinning LiDAR (e.g. 16 or 32 beams)
        num_lines = min(H, max(1, int(H * sparsity_ratio * 4)))
        step = max(1, H // num_lines)
        for y in range(0, H, step):
            valid_mask[y, :] = True
        sparse_d[valid_mask] = depth[valid_mask]
        
    elif pattern == 'grid':
        stride = max(2, int(1.0 / np.sqrt(max(sparsity_ratio, 1e-8))))
        valid_mask[::stride, ::stride] = True
        sparse_d[valid_mask] = depth[valid_mask]
        
    return sparse_d, valid_mask


def add_sensor_noise(depth: np.ndarray, 
                     gaussian_sigma: float = 0.03, 
                     speckle_prob: float = 0.01,
                     rng=None) -> np.ndarray:
    """
    Simulates physical sensor noise:
    - Multiplicative / additive Gaussian depth jitter
    - Impulsive speckle noise
    """
    noisy_d = depth.copy().astype(np.float32)
    rng = rng or np.random.default_rng()
    valid = noisy_d > 1e-4
    
    if gaussian_sigma > 0:
        # Depth-dependent noise (uncertainty increases with distance)
        noise = rng.normal(0.0, gaussian_sigma, size=depth.shape).astype(np.float32)
        noisy_d[valid] += noisy_d[valid] * noise[valid]
        
    if speckle_prob > 0:
        speckle_mask = (rng.random(depth.shape) < speckle_prob) & valid
        noisy_d[speckle_mask] *= rng.uniform(0.5, 1.5, size=np.sum(speckle_mask)).astype(np.float32)
        
    noisy_d = np.maximum(noisy_d, 0.0)
    return noisy_d


def simulate_missing_holes(depth: np.ndarray, 
                           num_holes: int = 4, 
                           max_hole_radius: int = 25,
                           rng=None) -> tuple:
    """
    Simulates sensor dropouts (e.g., absorbing dark surfaces, specular reflection, or occlusion boundaries).
    """
    H, W = depth.shape[:2]
    rng = rng or np.random.default_rng()
    corrupted_d = depth.copy()
    hole_mask = np.zeros((H, W), dtype=bool)
    
    for _ in range(num_holes):
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))
        max_radius = max(6, int(max_hole_radius))
        ry = int(rng.integers(5, max_radius))
        rx = int(rng.integers(5, max_radius))
        
        y_min, y_max = max(0, cy - ry), min(H, cy + ry)
        x_min, x_max = max(0, cx - rx), min(W, cx + rx)
        
        corrupted_d[y_min:y_max, x_min:x_max] = 0.0
        hole_mask[y_min:y_max, x_min:x_max] = True
        
    return corrupted_d, hole_mask

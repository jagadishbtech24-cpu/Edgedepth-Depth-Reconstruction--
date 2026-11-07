"""
Depth Discontinuity Detection and Surface Normal Variation Module.
Detects geometric boundaries, occlusion silhouettes, and distinguishes physical depth steps
from flat planar regions.
"""

import numpy as np
import cv2

def compute_depth_gradients(depth: np.ndarray) -> tuple:
    """
    Computes horizontal and vertical spatial depth gradients and magnitude using Sobel kernels.
    """
    d_float = depth.astype(np.float32)
    gx = cv2.Sobel(d_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, grad_mag


def compute_relative_depth_discontinuity(depth: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """
    Detects depth discontinuities using relative depth differences:
    rel_diff = |D(p) - D(q)| / min(D(p), D(q))
    This is scale-invariant (e.g., 10cm jump at 1m is 10%, same as 1m jump at 10m).
    
    Returns:
        binary_discontinuity_mask: uint8 array (255 for boundary, 0 otherwise)
    """
    d_safe = np.maximum(depth.astype(np.float32), 1e-4)
    
    # Kernel for local maximum and minimum
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    d_max = cv2.dilate(d_safe, kernel)
    d_min = cv2.erode(d_safe, kernel)
    
    rel_discontinuity = (d_max - d_min) / np.maximum(d_min, 1e-4)
    discontinuity_mask = (rel_discontinuity > threshold).astype(np.uint8) * 255
    return discontinuity_mask


def compute_surface_normals(depth: np.ndarray, focal_length: float = 500.0) -> np.ndarray:
    """
    Approximates 3D surface normals from depth map.
    
    Args:
        depth: 2D array (H, W) in metric units
        focal_length: camera focal length in pixels
        
    Returns:
        normals: (H, W, 3) normalized unit vectors in [-1, 1]
    """
    H, W = depth.shape
    d_safe = np.maximum(depth.astype(np.float32), 1e-4)
    
    # Pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    cx, cy = W / 2.0, H / 2.0
    
    # 3D points
    X = (u - cx) * d_safe / focal_length
    Y = (v - cy) * d_safe / focal_length
    Z = d_safe
    
    # Gradients in 3D
    dX_dx = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=3)
    dY_dx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
    dZ_dx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3)
    
    dX_dy = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=3)
    dY_dy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
    dZ_dy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3)
    
    # Cross product: N = dP/dx x dP/dy
    nx = dY_dx * dZ_dy - dZ_dx * dY_dy
    ny = dZ_dx * dX_dy - dX_dx * dZ_dy
    nz = dX_dx * dY_dy - dY_dx * dX_dy
    
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.maximum(norm, 1e-8)
    
    nx /= norm
    ny /= norm
    nz /= norm
    
    normals = np.stack([nx, ny, nz], axis=-1)
    return normals.astype(np.float32)


def detect_discontinuities(depth: np.ndarray, 
                           grad_thresh_percentile: float = 90.0,
                           rel_step_thresh: float = 0.08) -> dict:
    """
    Comprehensive discontinuity analysis:
    - Gradient magnitude
    - Relative step mask
    - Normal variation mask
    - Combined geometric boundary map
    """
    gx, gy, grad_mag = compute_depth_gradients(depth)
    
    # Adaptive threshold based on percentile of non-zero gradients
    valid_grads = grad_mag[grad_mag > 1e-5]
    if len(valid_grads) > 0:
        thresh = np.percentile(valid_grads, grad_thresh_percentile)
    else:
        thresh = 0.1
        
    grad_mask = (grad_mag > thresh).astype(np.uint8) * 255
    rel_mask = compute_relative_depth_discontinuity(depth, threshold=rel_step_thresh)
    
    # Combined boundary
    combined = cv2.bitwise_or(grad_mask, rel_mask)
    
    return {
        'grad_mag': grad_mag,
        'grad_mask': grad_mask,
        'rel_mask': rel_mask,
        'boundary_mask': combined
    }

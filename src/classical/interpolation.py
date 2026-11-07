"""
Classical interpolation baselines for depth map upsampling:
- Nearest-neighbor
- Bilinear
- Bicubic
With support for mask-aware interpolation to prevent zero/invalid pixel bleeding.
"""

import numpy as np
import cv2

def interpolate_depth(depth: np.ndarray, target_shape: tuple, method: str = 'bicubic', mask: np.ndarray = None) -> np.ndarray:
    """
    Upsample depth map to target_shape (height, width).
    
    Args:
        depth: 2D numpy array of shape (H, W) or (H, W, 1)
        target_shape: tuple (target_H, target_W)
        method: 'nearest', 'bilinear', or 'bicubic'
        mask: optional boolean mask of valid depth pixels (True = valid)
        
    Returns:
        upsampled_depth: 2D numpy array of shape (target_H, target_W), float32
    """
    if depth.ndim == 3:
        depth = depth.squeeze()
        
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    d_float = depth.astype(np.float32)
    
    cv_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp_flag = cv_methods.get(method.lower(), cv2.INTER_CUBIC)
    
    if mask is None or np.all(mask):
        # Direct interpolation
        upsampled = cv2.resize(d_float, (target_w, target_h), interpolation=interp_flag)
    else:
        # Mask-aware normalized interpolation: avoids spreading 0/NaN into valid regions
        mask_float = mask.astype(np.float32)
        valid_depth = np.where(mask, d_float, 0.0).astype(np.float32)
        
        upsampled_val = cv2.resize(valid_depth, (target_w, target_h), interpolation=interp_flag)
        upsampled_weight = cv2.resize(mask_float, (target_w, target_h), interpolation=interp_flag)
        
        # Avoid division by zero
        safe_weight = np.maximum(upsampled_weight, 1e-6)
        upsampled = upsampled_val / safe_weight
        
        # For regions with zero weight, fallback to nearest neighbor
        if np.any(upsampled_weight < 1e-4):
            nearest_fill = cv2.resize(d_float, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            upsampled[upsampled_weight < 1e-4] = nearest_fill[upsampled_weight < 1e-4]

    # Clean any numerical artifacts
    upsampled = np.nan_to_num(upsampled, nan=0.0, posinf=0.0, neginf=0.0)
    return upsampled.astype(np.float32)

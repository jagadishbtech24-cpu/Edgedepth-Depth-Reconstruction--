"""
Guided Image Filter (GIF) implementation (He et al., ECCV 2010 / TPAMI 2013).
Supports both grayscale and full 3-channel RGB guidance images with O(1) box-filter speed.
"""

import numpy as np
import cv2

def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Computes box filter (mean filter) with window radius r in O(1) using cv2.boxFilter."""
    ksize = (2 * r + 1, 2 * r + 1)
    return cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)


def guided_filter_mono(guide: np.ndarray, depth: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    Monochrome Guided Filter.
    
    Args:
        guide: 2D array (H, W) in [0, 1]
        depth: 2D array (H, W) depth map
        radius: local window radius
        eps: regularization parameter (controls smoothness vs edge preservation)
        
    Returns:
        output_depth: 2D array (H, W)
    """
    I = guide.astype(np.float32)
    p = depth.astype(np.float32)
    
    mean_I = box_filter(I, radius)
    mean_p = box_filter(p, radius)
    mean_Ip = box_filter(I * p, radius)
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box_filter(I * I, radius)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    q = mean_a * I + mean_b
    return q.astype(np.float32)


def guided_filter_color(guide_rgb: np.ndarray, depth: np.ndarray, radius: int = 8, eps: float = 1e-3) -> np.ndarray:
    """
    Color Guided Filter using 3-channel RGB guidance.
    Solves 3x3 linear system per pixel analytically using matrix inversion formula.
    
    Args:
        guide_rgb: 3D array (H, W, 3) in [0, 1]
        depth: 2D array (H, W)
        radius: window radius
        eps: regularization term
        
    Returns:
        output_depth: 2D array (H, W)
    """
    H, W = depth.shape
    I = guide_rgb.astype(np.float32)
    if I.max() > 1.0:
        I = I / 255.0
        
    p = depth.astype(np.float32)
    
    Ir = I[:, :, 0]
    Ig = I[:, :, 1]
    Ib = I[:, :, 2]
    
    # Means of RGB channels
    mean_Ir = box_filter(Ir, radius)
    mean_Ig = box_filter(Ig, radius)
    mean_Ib = box_filter(Ib, radius)
    mean_p  = box_filter(p, radius)
    
    # Cross products
    mean_Ip_r = box_filter(Ir * p, radius)
    mean_Ip_g = box_filter(Ig * p, radius)
    mean_Ip_b = box_filter(Ib * p, radius)
    
    # Covariances between I and p
    cov_Ip_r = mean_Ip_r - mean_Ir * mean_p
    cov_Ip_g = mean_Ip_g - mean_Ig * mean_p
    cov_Ip_b = mean_Ip_b - mean_Ib * mean_p
    
    # Variances and covariances of I
    var_I_rr = box_filter(Ir * Ir, radius) - mean_Ir * mean_Ir + eps
    var_I_rg = box_filter(Ir * Ig, radius) - mean_Ir * mean_Ig
    var_I_rb = box_filter(Ir * Ib, radius) - mean_Ir * mean_Ib
    var_I_gg = box_filter(Ig * Ig, radius) - mean_Ig * mean_Ig + eps
    var_I_gb = box_filter(Ig * Ib, radius) - mean_Ig * mean_Ib
    var_I_bb = box_filter(Ib * Ib, radius) - mean_Ib * mean_Ib + eps
    
    # 3x3 symmetric matrix inversion for:
    # [ var_rr  var_rg  var_rb ]
    # [ var_rg  var_gg  var_gb ]
    # [ var_rb  var_gb  var_bb ]
    
    # Determinant of Sigma + eps*I
    inv_00 = var_I_gg * var_I_bb - var_I_gb * var_I_gb
    inv_01 = var_I_gb * var_I_rb - var_I_rg * var_I_bb
    inv_02 = var_I_rg * var_I_gb - var_I_gg * var_I_rb
    
    inv_11 = var_I_rr * var_I_bb - var_I_rb * var_I_rb
    inv_12 = var_I_rb * var_I_rg - var_I_rr * var_I_gb
    inv_22 = var_I_rr * var_I_gg - var_I_rg * var_I_rg
    
    det = var_I_rr * inv_00 + var_I_rg * inv_01 + var_I_rb * inv_02
    det = np.maximum(det, 1e-12)
    
    inv_00 /= det
    inv_01 /= det
    inv_02 /= det
    inv_11 /= det
    inv_12 /= det
    inv_22 /= det
    
    # Linear coefficients a_r, a_g, a_b
    a_r = inv_00 * cov_Ip_r + inv_01 * cov_Ip_g + inv_02 * cov_Ip_b
    a_g = inv_01 * cov_Ip_r + inv_11 * cov_Ip_g + inv_12 * cov_Ip_b
    a_b = inv_02 * cov_Ip_r + inv_12 * cov_Ip_g + inv_22 * cov_Ip_b
    
    b = mean_p - a_r * mean_Ir - a_g * mean_Ig - a_b * mean_Ib
    
    mean_a_r = box_filter(a_r, radius)
    mean_a_g = box_filter(a_g, radius)
    mean_a_b = box_filter(a_b, radius)
    mean_b   = box_filter(b, radius)
    
    q = mean_a_r * Ir + mean_a_g * Ig + mean_a_b * Ib + mean_b
    return q.astype(np.float32)


def guided_filter(depth: np.ndarray, 
                  guidance: np.ndarray, 
                  radius: int = 8, 
                  eps: float = 1e-3, 
                  color: bool = True) -> np.ndarray:
    """
    Main entry point for Guided Image Filter.
    Automatically handles grayscale or color guidance.
    """
    if guidance.ndim == 3 and color:
        return guided_filter_color(guidance, depth, radius, eps)
    elif guidance.ndim == 3 and not color:
        gray = cv2.cvtColor(guidance.astype(np.uint8) if guidance.max() > 1.0 else (guidance * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return guided_filter_mono(gray, depth, radius, eps)
    else:
        g = guidance.astype(np.float32)
        if g.max() > 1.0:
            g = g / 255.0
        return guided_filter_mono(g, depth, radius, eps)

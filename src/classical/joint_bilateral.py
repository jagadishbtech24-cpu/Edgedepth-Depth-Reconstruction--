"""
Joint Bilateral Filter (JBF) for RGB-guided depth map upsampling and refinement.
Preserves depth edges by using spatial proximity in depth and color similarity in high-resolution RGB guidance.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F

def joint_bilateral_filter(depth: np.ndarray, 
                           guidance_rgb: np.ndarray, 
                           radius: int = 5, 
                           sigma_spatial: float = 3.0, 
                           sigma_range: float = 0.1, 
                           use_torch: bool = True) -> np.ndarray:
    """
    Applies Joint (Cross) Bilateral Filtering to upsampled depth guided by RGB image.
    
    Args:
        depth: 2D array (H, W) of upsampled depth values (float32)
        guidance_rgb: 3D array (H, W, 3) RGB image in [0, 1] or [0, 255]
        radius: Half-window size for bilateral kernel (kernel size = 2*radius + 1)
        sigma_spatial: Gaussian sigma for spatial distance
        sigma_range: Gaussian sigma for RGB color difference (normalized to [0, 1])
        use_torch: Whether to use PyTorch tensor vectorized acceleration
        
    Returns:
        refined_depth: 2D array (H, W) float32
    """
    H, W = depth.shape[:2]
    
    # Normalize guidance to [0, 1]
    guide = guidance_rgb.astype(np.float32)
    if guide.max() > 1.0:
        guide = guide / 255.0
    if guide.ndim == 2:
        guide = np.stack([guide]*3, axis=-1)
        
    depth_in = depth.astype(np.float32)
    
    if use_torch:
        try:
            return _joint_bilateral_torch(depth_in, guide, radius, sigma_spatial, sigma_range)
        except Exception:
            pass  # fallback to numpy implementation
            
    return _joint_bilateral_numpy(depth_in, guide, radius, sigma_spatial, sigma_range)


def _joint_bilateral_torch(depth: np.ndarray, guide: np.ndarray, radius: int, sigma_s: float, sigma_r: float) -> np.ndarray:
    """Vectorized PyTorch implementation using unfold for fast sliding-window cross-filtering."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    H, W = depth.shape
    d_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    g_tensor = torch.from_numpy(guide).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    ksize = 2 * radius + 1
    
    # Precompute spatial Gaussian weights
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
                          torch.arange(-radius, radius + 1, device=device, dtype=torch.float32),
                          indexing='ij')
    spatial_dist_sq = x**2 + y**2
    spatial_weight = torch.exp(-spatial_dist_sq / (2.0 * (sigma_s ** 2))).view(1, ksize * ksize, 1, 1)
    
    # Unfold depth and guidance into local patches
    pad = radius
    d_pad = F.pad(d_tensor, (pad, pad, pad, pad), mode='replicate')
    g_pad = F.pad(g_tensor, (pad, pad, pad, pad), mode='replicate')
    
    # Unfold gives shape (1, C * ksize * ksize, L) where L = H * W
    d_patches = F.unfold(d_pad, kernel_size=ksize).view(1, ksize * ksize, H, W)
    g_patches = F.unfold(g_pad, kernel_size=ksize).view(1, 3, ksize * ksize, H, W)
    
    # Central pixel values for guidance
    g_center = g_tensor.unsqueeze(2)  # (1, 3, 1, H, W)
    
    # Color difference: sum of squared differences across 3 RGB channels
    color_diff_sq = torch.sum((g_patches - g_center) ** 2, dim=1)  # (1, ksize*ksize, H, W)
    range_weight = torch.exp(-color_diff_sq / (2.0 * (sigma_r ** 2)))
    
    # Combined bilateral weights
    weights = spatial_weight * range_weight  # (1, ksize*ksize, H, W)
    
    # Weight normalization
    weight_sum = torch.sum(weights, dim=1, keepdim=True) + 1e-8
    filtered = torch.sum(d_patches * weights, dim=1, keepdim=True) / weight_sum
    
    out = filtered.squeeze().cpu().numpy().astype(np.float32)
    return out


def _joint_bilateral_numpy(depth: np.ndarray, guide: np.ndarray, radius: int, sigma_s: float, sigma_r: float) -> np.ndarray:
    """NumPy spatial shift implementation for CPU compatibility."""
    H, W = depth.shape
    accum_depth = np.zeros((H, W), dtype=np.float32)
    accum_weights = np.zeros((H, W), dtype=np.float32)
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_dist_sq = dx * dx + dy * dy
            if spatial_dist_sq > radius * radius:
                continue
            spatial_w = np.exp(-spatial_dist_sq / (2.0 * sigma_s * sigma_s))
            
            # Shifted arrays
            d_shift = np.roll(np.roll(depth, dy, axis=0), dx, axis=1)
            g_shift = np.roll(np.roll(guide, dy, axis=0), dx, axis=1)
            
            # Mask out boundary wrap-around
            valid_mask = np.ones((H, W), dtype=bool)
            if dy > 0:
                valid_mask[:dy, :] = False
            elif dy < 0:
                valid_mask[dy:, :] = False
            if dx > 0:
                valid_mask[:, :dx] = False
            elif dx < 0:
                valid_mask[:, dx:] = False
                
            color_diff_sq = np.sum((guide - g_shift) ** 2, axis=-1)
            range_w = np.exp(-color_diff_sq / (2.0 * sigma_r * sigma_r))
            
            total_w = spatial_w * range_w * valid_mask.astype(np.float32)
            accum_depth += d_shift * total_w
            accum_weights += total_w
            
    filtered = accum_depth / np.maximum(accum_weights, 1e-8)
    return filtered.astype(np.float32)

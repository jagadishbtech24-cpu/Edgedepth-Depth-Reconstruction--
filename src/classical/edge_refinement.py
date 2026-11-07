"""
Edge Detection and Boundary-Aware Refinement Module.
Extracts multi-scale edges from RGB guidance, detects texture-copying artifacts,
and sharpens depth boundaries aligned with physical object contours.
"""

import numpy as np
import cv2

def extract_rgb_edges(rgb: np.ndarray, 
                      canny_low: int = 50, 
                      canny_high: int = 150) -> dict:
    """
    Extracts photometric edges and gradient magnitudes from RGB guidance image.
    """
    if rgb.max() <= 1.0:
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
    else:
        rgb_u8 = rgb.astype(np.uint8)
        
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    
    # Canny edges
    canny = cv2.Canny(gray, canny_low, canny_high)
    
    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = mag / (mag.max() + 1e-8)
    
    return {
        'gray': gray,
        'canny': canny,
        'grad_mag': mag_norm,
        'gx': gx,
        'gy': gy
    }


def compute_texture_vs_boundary_affinity(rgb: np.ndarray, 
                                         depth_lr_upsampled: np.ndarray,
                                         search_radius: int = 4) -> np.ndarray:
    """
    Computes an Anti-Texture-Copying Affinity Weight Map [0, 1].
    
    Texture-Copying Problem:
    Flat surfaces with high texture contrast (e.g. brick walls, book covers, patterned rugs)
    produce strong RGB gradients. If naive guided filters use these gradients, they create
    false depth ridges (texture copying).
    
    Solution:
    A true depth discontinuity manifests both as an RGB edge AND has significant local depth variation
    within a small spatial window. If RGB has strong edges but local depth is virtually planar/constant,
    it is pure texture -> suppress guidance weight.
    """
    rgb_edges = extract_rgb_edges(rgb)
    rgb_grad = rgb_edges['grad_mag']
    
    # Local depth range in window
    ksize = 2 * search_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    d_max = cv2.dilate(depth_lr_upsampled, kernel)
    d_min = cv2.erode(depth_lr_upsampled, kernel)
    depth_variation = (d_max - d_min) / (depth_lr_upsampled + 1e-4)
    
    # Normalize depth variation to [0, 1] with soft sigmoid-like threshold
    depth_var_norm = np.clip(depth_variation / 0.05, 0.0, 1.0)
    
    # Geometric edge confidence: high RGB edge supported by local depth variation
    # High confidence = true geometric boundary
    # Low confidence = flat surface texture or smooth region
    geometric_affinity = rgb_grad * depth_var_norm
    
    # Dilation to cover boundary transition band
    affinity_band = cv2.dilate(geometric_affinity.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return np.clip(affinity_band, 0.0, 1.0)


def boundary_aware_sharpening(depth: np.ndarray, 
                              guidance_rgb: np.ndarray,
                              affinity_map: np.ndarray,
                              strength: float = 0.5) -> np.ndarray:
    """
    Selectively sharpens depth boundaries that coincide with verified geometric edges,
    avoiding overshoot and preventing texture-copying on flat surfaces.
    """
    # High-pass filter via unsharp masking
    blur = cv2.GaussianBlur(depth, (5, 5), sigmaX=1.5)
    high_freq = depth - blur
    
    # Modulate high-frequency enhancement with anti-texture affinity map
    sharpened = depth + strength * high_freq * affinity_map
    
    # Ensure no negative depths
    sharpened = np.maximum(sharpened, 0.0)
    return sharpened.astype(np.float32)

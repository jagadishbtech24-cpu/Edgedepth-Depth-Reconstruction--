"""
Noise Removal and Hole Filling Module.
Handles sensor dropout, missing depth values (holes), salt-and-pepper noise,
and sparse depth maps (e.g., LiDAR-like patterns).
"""

import numpy as np
import cv2

def remove_depth_noise(depth: np.ndarray, 
                       method: str = 'median', 
                       ksize: int = 5) -> np.ndarray:
    """
    Removes depth sensor noise (salt-and-pepper, speckle, Gaussian noise)
    while attempting to preserve depth edges.
    """
    d_float = depth.astype(np.float32)
    
    if method == 'median':
        # Convert float to scaled int or use scipy/cv2
        # cv2.medianBlur works on CV_8U, CV_16U, or CV_32F (ksize 3 or 5)
        if ksize in [3, 5]:
            denoised = cv2.medianBlur(d_float, ksize)
        else:
            # Fallback to bilateral or 5x5
            denoised = cv2.medianBlur(d_float, 5)
    elif method == 'bilateral':
        denoised = cv2.bilateralFilter(d_float, d=ksize, sigmaColor=0.1, sigmaSpace=ksize/2.0)
    elif method == 'gaussian':
        denoised = cv2.GaussianBlur(d_float, (ksize, ksize), sigmaX=ksize/3.0)
    else:
        denoised = d_float
        
    return denoised.astype(np.float32)


def fill_depth_holes_morphological(depth: np.ndarray, 
                                   mask_invalid: np.ndarray, 
                                   max_radius: int = 7) -> np.ndarray:
    """
    Fills small missing depth holes using morphological closing and dilation.
    """
    filled = depth.copy()
    for r in range(3, max_radius + 1, 2):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        dilated = cv2.dilate(filled, kernel)
        still_invalid = (filled <= 1e-4) | mask_invalid
        filled[still_invalid] = dilated[still_invalid]
        if not np.any(filled <= 1e-4):
            break
    return filled


def fill_depth_holes_inpaint(depth: np.ndarray, 
                             mask_invalid: np.ndarray = None, 
                             method: str = 'telea', 
                             inpaint_radius: int = 5) -> np.ndarray:
    """
    Fills depth holes using PDE-based inpainting (Fast Marching Telea or Navier-Stokes).
    """
    d_float = depth.astype(np.float32)
    
    if mask_invalid is None:
        mask_invalid = (d_float <= 1e-4) | np.isnan(d_float) | np.isinf(d_float)
        
    inpaint_mask = mask_invalid.astype(np.uint8) * 255
    
    if not np.any(inpaint_mask):
        return d_float
        
    # Scale depth to [0, 255] or [0, 65535] for OpenCV inpaint if needed
    val_min = np.min(d_float[~mask_invalid]) if np.any(~mask_invalid) else 0.0
    val_max = np.max(d_float[~mask_invalid]) if np.any(~mask_invalid) else 1.0
    val_range = max(val_max - val_min, 1e-6)
    
    d_norm = ((d_float - val_min) / val_range * 255.0).astype(np.uint8)
    
    flag = cv2.INPAINT_TELEA if method.lower() == 'telea' else cv2.INPAINT_NS
    inpainted_u8 = cv2.inpaint(d_norm, inpaint_mask, inpaintRadius=inpaint_radius, flags=flag)
    
    # Restore original metric scale
    reconstructed = (inpainted_u8.astype(np.float32) / 255.0) * val_range + val_min
    
    # Keep original valid pixels exact
    reconstructed[~mask_invalid] = d_float[~mask_invalid]
    return reconstructed.astype(np.float32)


def fill_depth_holes_normalized_convolution(depth: np.ndarray, 
                                            mask_valid: np.ndarray, 
                                            sigma: float = 4.0, 
                                            iterations: int = 3) -> np.ndarray:
    """
    Normalized convolution for sparse depth inpainting (Eldar et al.).
    Propagates valid depth values according to Gaussian spatial weighting.
    """
    d = depth.astype(np.float32).copy()
    m = mask_valid.astype(np.float32).copy()
    
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
        
    for _ in range(iterations):
        if np.all(m > 0.99):
            break
        blurred_val = cv2.GaussianBlur(d * m, (ksize, ksize), sigmaX=sigma)
        blurred_mask = cv2.GaussianBlur(m, (ksize, ksize), sigmaX=sigma)
        
        safe_mask = np.maximum(blurred_mask, 1e-6)
        interpolated = blurred_val / safe_mask
        
        # Update missing locations
        fill_mask = (m < 0.5) & (blurred_mask > 1e-3)
        d[fill_mask] = interpolated[fill_mask]
        m[fill_mask] = 1.0
        
    return d.astype(np.float32)


def preprocess_depth(depth: np.ndarray, 
                     denoise: bool = True, 
                     fill_holes: bool = True,
                     denoise_ksize: int = 3) -> tuple:
    """
    Comprehensive preprocessing pipeline:
    1. Detects invalid / missing pixels.
    2. Performs edge-preserving median denoising on valid pixels.
    3. Inpaints holes using PDE fast marching and normalized convolution.
    
    Returns:
        (clean_depth, valid_mask)
    """
    d = depth.astype(np.float32)
    invalid_mask = (d <= 1e-4) | np.isnan(d) | np.isinf(d)
    valid_mask = ~invalid_mask
    
    clean_d = d.copy()
    clean_d[invalid_mask] = 0.0
    
    if denoise and np.any(valid_mask):
        clean_d = remove_depth_noise(clean_d, method='median', ksize=denoise_ksize)
        # Restore zero on invalid so they remain explicitly marked
        clean_d[invalid_mask] = 0.0
        
    if fill_holes and np.any(invalid_mask):
        if np.sum(valid_mask) > 10:  # has sufficient valid pixels
            clean_d = fill_depth_holes_inpaint(clean_d, mask_invalid=invalid_mask, method='telea', inpaint_radius=5)
            # Second pass normalized convolution for remaining holes if any
            rem_invalid = clean_d <= 1e-4
            if np.any(rem_invalid):
                clean_d = fill_depth_holes_normalized_convolution(clean_d, ~rem_invalid, sigma=3.0)
                
    return clean_d.astype(np.float32), valid_mask

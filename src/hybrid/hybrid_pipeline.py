"""
Hybrid Depth Reconstruction Pipeline:
Seamlessly fuses classical edge-aware filtering (Joint Bilateral / Guided Filter)
with Deep Learning (EdgeGuideNet) and Anti-Texture-Copying regularization.
"""

import numpy as np
import cv2
import torch
import time
from ..classical.interpolation import interpolate_depth
from ..classical.hole_filling import preprocess_depth
from ..classical.guided_filter import guided_filter
from ..classical.joint_bilateral import joint_bilateral_filter
from ..classical.edge_refinement import compute_texture_vs_boundary_affinity, boundary_aware_sharpening
from ..classical.discontinuity import detect_discontinuities
from ..models.weights_loader import load_edge_guide_model, run_dl_inference

class HybridDepthReconstructor:
    """
    State-of-the-art Hybrid Depth Reconstruction Engine.
    Combines:
    1. Preprocessing: Hole filling and noise removal
    2. Deep Learning: EdgeGuideNet multi-scale semantic super-resolution
    3. Classical Filtering: Guided / Joint Bilateral refinement
    4. Anti-Texture Mitigation: Geometric vs. Photometric affinity weighting
    5. Measurement Fidelity enforcement
    """
    def __init__(self, model_weights_path: str = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_edge_guide_model(model_weights_path, device=self.device)
        
    def reconstruct(self, 
                    depth_lr: np.ndarray, 
                    rgb_hr: np.ndarray, 
                    scale_factor: int = 4, 
                    target_shape: tuple = None,
                    denoise: bool = True,
                    fill_holes: bool = True,
                    filter_type: str = 'guided', # 'guided' or 'bilateral'
                    anti_texture: bool = True,
                    sharpen_strength: float = 0.35) -> dict:
        """
        Executes the full hybrid reconstruction pipeline.
        
        Args:
            depth_lr: Low-resolution or sparse depth map
            rgb_hr: High-resolution RGB guidance image (H, W, 3)
            scale_factor: Upsampling factor (2, 4, 8, etc.)
            target_shape: Explicit (H, W) or inferred from rgb_hr
            denoise: Whether to apply sensor noise removal
            fill_holes: Whether to fill invalid/missing depth values
            filter_type: Classical refinement filter ('guided' or 'bilateral')
            anti_texture: Whether to enforce anti-texture-copying suppression
            sharpen_strength: Boundary sharpening factor [0.0, 1.0]
            
        Returns:
            dict containing:
                'depth_reconstructed': Final high-resolution edge-preserved depth
                'depth_dl': Pure deep learning output before classical refinement
                'depth_bicubic': Standard bicubic interpolation baseline
                'depth_classical': Classical guided filter on bicubic
                'affinity_map': Anti-texture-copying geometric affinity map
                'discontinuities': Detected depth boundary mask
                'valid_mask': Preprocessed validity mask
        """
        started = time.perf_counter()
        if target_shape is None:
            target_h, target_w = rgb_hr.shape[0], rgb_hr.shape[1]
        else:
            target_h, target_w = target_shape
        if rgb_hr.ndim != 3 or rgb_hr.shape[2] != 3 or rgb_hr.shape[:2] != (target_h, target_w):
            raise ValueError(
                f"rgb_hr must have shape ({target_h}, {target_w}, 3), got {rgb_hr.shape}"
            )
        if filter_type.lower() not in {"guided", "bilateral"}:
            raise ValueError("filter_type must be 'guided' or 'bilateral'")

        timings_ms = {}
        # 1. Classical Preprocessing: Noise removal and hole filling on LR input
        stage_started = time.perf_counter()
        clean_lr, mask_lr = preprocess_depth(depth_lr, denoise=denoise, fill_holes=fill_holes)
        timings_ms["preprocessing"] = (time.perf_counter() - stage_started) * 1000.0
        
        # 2. Baseline Interpolation to target resolution
        stage_started = time.perf_counter()
        depth_bicubic = interpolate_depth(clean_lr, (target_h, target_w), method='bicubic', mask=mask_lr)
        timings_ms["base_interpolation"] = (time.perf_counter() - stage_started) * 1000.0
        
        # 3. Deep Learning Super-Resolution & Refinement
        stage_started = time.perf_counter()
        mask_hr = cv2.resize(
            mask_lr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST
        )
        depth_dl, dl_edge_conf = run_dl_inference(
            self.model, depth_bicubic, rgb_hr, mask=mask_hr, device=self.device
        )
        timings_ms["deep_inference"] = (time.perf_counter() - stage_started) * 1000.0
        
        # 4. Anti-Texture-Copying Affinity Analysis
        stage_started = time.perf_counter()
        if anti_texture:
            affinity_map = compute_texture_vs_boundary_affinity(rgb_hr, depth_dl, search_radius=4)
        else:
            affinity_map = np.ones((target_h, target_w), dtype=np.float32)
        timings_ms["affinity"] = (time.perf_counter() - stage_started) * 1000.0
            
        # 5. Boundary-Preserving Classical Filtering on Deep Feature Map
        stage_started = time.perf_counter()
        # We blend the guidance image: where affinity is high, use full RGB; where affinity is low (texture on flat), use smoothed RGB
        if anti_texture:
            rgb_blur = cv2.GaussianBlur(rgb_hr, (7, 7), sigmaX=2.0)
            rgb_guidance_conditioned = (rgb_hr * affinity_map[:, :, None] + rgb_blur * (1.0 - affinity_map[:, :, None])).astype(np.float32)
        else:
            rgb_guidance_conditioned = rgb_hr.astype(np.float32)
            
        if filter_type.lower() == 'bilateral':
            depth_filtered = joint_bilateral_filter(depth_dl, rgb_guidance_conditioned, radius=4, sigma_spatial=3.0, sigma_range=0.08)
        else:
            depth_filtered = guided_filter(depth_dl, rgb_guidance_conditioned, radius=6, eps=1e-3, color=True)
            
        # 6. Edge Sharpening & Boundary Snapping
        if sharpen_strength > 0:
            depth_final = boundary_aware_sharpening(depth_filtered, rgb_hr, affinity_map, strength=sharpen_strength)
        else:
            depth_final = depth_filtered
        timings_ms["hybrid_refinement"] = (time.perf_counter() - stage_started) * 1000.0
        
        # 7. Classical-only comparison (for ablation & benchmarking)
        stage_started = time.perf_counter()
        depth_classical = guided_filter(depth_bicubic, rgb_hr, radius=6, eps=1e-3, color=True)
        timings_ms["guided_baseline"] = (time.perf_counter() - stage_started) * 1000.0
        
        # 8. Boundary discontinuity analysis on final reconstruction
        stage_started = time.perf_counter()
        discontinuities = detect_discontinuities(depth_final)
        timings_ms["discontinuity_analysis"] = (time.perf_counter() - stage_started) * 1000.0
        timings_ms["total"] = (time.perf_counter() - started) * 1000.0
        
        return {
            'depth_reconstructed': depth_final,
            'depth_dl': depth_dl,
            'depth_bicubic': depth_bicubic,
            'depth_classical': depth_classical,
            'affinity_map': affinity_map,
            'discontinuities': discontinuities['boundary_mask'],
            'boundary_conf': dl_edge_conf,
            'valid_mask': mask_lr,
            'timings_ms': timings_ms,
        }

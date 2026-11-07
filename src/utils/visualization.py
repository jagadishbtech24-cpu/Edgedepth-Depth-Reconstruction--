"""
Depth Visualization and 3D Interactive Plotting Utilities.
Includes:
- Colormapping (turbo, magma, viridis, plasma)
- Error difference heatmaps
- Boundary overlay visualization
- 3D interactive mesh generation via Plotly
"""

import numpy as np
import cv2
import plotly.graph_objects as go

def colorize_depth(depth: np.ndarray, 
                   colormap: str = 'turbo', 
                   vmin: float = None, 
                   vmax: float = None) -> np.ndarray:
    """
    Converts 2D depth map into a vibrant 3-channel RGB colorized representation.
    """
    d = depth.astype(np.float32)
    valid = d > 1e-4
    
    if not np.any(valid):
        return np.zeros((*d.shape, 3), dtype=np.uint8)
        
    if vmin is None:
        vmin = float(np.percentile(d[valid], 1.0))
    if vmax is None:
        vmax = float(np.percentile(d[valid], 99.0))
        
    vmax = max(vmax, vmin + 1e-6)
    
    norm = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    
    cv_cmaps = {
        'turbo': cv2.COLORMAP_TURBO,
        'magma': cv2.COLORMAP_MAGMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'jet': cv2.COLORMAP_JET
    }
    
    cmap_id = cv_cmaps.get(colormap.lower(), cv2.COLORMAP_TURBO)
    colorized_bgr = cv2.applyColorMap(norm_u8, cmap_id)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    
    # Set invalid pixels to black
    colorized_rgb[~valid] = [0, 0, 0]
    return colorized_rgb


def compute_error_heatmap(pred: np.ndarray, 
                          gt: np.ndarray, 
                          colormap: str = 'magma',
                          vmax: float = None,
                          mask: np.ndarray = None) -> np.ndarray:
    """
    Computes visual error heatmap |pred - gt|.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 1e-4)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {gt.shape}")
        valid &= mask
    error = np.zeros_like(gt, dtype=np.float32)
    error[valid] = np.abs(pred[valid] - gt[valid])
    
    if vmax is None:
        vmax = float(np.percentile(error[valid], 98.0)) if np.any(valid) else 1.0
        vmax = max(vmax, 0.05)
        
    norm_error = np.clip(error / vmax, 0.0, 1.0)
    norm_u8 = (norm_error * 255.0).astype(np.uint8)
    
    cv_cmaps = {
        'magma': cv2.COLORMAP_MAGMA,
        'jet': cv2.COLORMAP_JET,
        'turbo': cv2.COLORMAP_TURBO,
        'hot': cv2.COLORMAP_HOT
    }
    cmap_id = cv_cmaps.get(colormap.lower(), cv2.COLORMAP_MAGMA)
    colorized_bgr = cv2.applyColorMap(norm_u8, cmap_id)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)
    colorized_rgb[~valid] = 0
    return colorized_rgb


def overlay_edges_on_image(image: np.ndarray, 
                           edge_mask: np.ndarray, 
                           color: tuple = (0, 255, 128), 
                           alpha: float = 0.7) -> np.ndarray:
    """
    Overlays detected edge boundaries onto RGB image or depth visualization.
    """
    img = image.copy()
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    mask = edge_mask > 0
    overlay = img.copy()
    overlay[mask] = color
    
    blended = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
    return blended


def create_3d_point_cloud_plotly(depth: np.ndarray, 
                                 rgb: np.ndarray = None, 
                                 downsample_factor: int = 4, 
                                 focal_length: float = 500.0) -> go.Figure:
    """
    Generates an interactive 3D surface/point cloud figure using Plotly.
    Allows inspection of depth step discontinuities and boundary sharpness in 3D.
    """
    H, W = depth.shape[:2]
    step = max(1, downsample_factor)
    
    sub_d = depth[::step, ::step].astype(np.float32)
    sub_h, sub_w = sub_d.shape
    
    # 2D Grid
    u, v = np.meshgrid(np.arange(0, W, step), np.arange(0, H, step))
    cx, cy = W / 2.0, H / 2.0
    
    # Valid filter
    valid = sub_d > 1e-4
    
    X = (u - cx) * sub_d / focal_length
    Y = (v - cy) * sub_d / focal_length
    Z = sub_d
    
    # Colors
    if rgb is not None:
        sub_rgb = rgb[::step, ::step]
        if sub_rgb.max() <= 1.0:
            sub_rgb = (sub_rgb * 255.0).astype(np.uint8)
        colors = [f"rgb({sub_rgb[y, x, 0]},{sub_rgb[y, x, 1]},{sub_rgb[y, x, 2]})" 
                  for y in range(sub_h) for x in range(sub_w) if valid[y, x]]
    else:
        colors = Z[valid]
        
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X[valid],
            y=Y[valid],
            z=Z[valid],
            mode='markers',
            marker=dict(
                size=2,
                color=colors if rgb is not None else Z[valid],
                colorscale='Turbo' if rgb is None else None,
                opacity=0.85
            )
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Depth Z (meters)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=0.0, y=-0.5, z=-1.8),
                up=dict(x=0, y=-1, z=0)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title='Interactive 3D Depth Point Cloud'
    )
    return fig

"""
Model weights manager and high-level inference interface for EdgeGuideNet.
"""

from pathlib import Path
import torch
import numpy as np
from .guided_depth_net import EdgeGuideNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = str(PROJECT_ROOT / "models" / "edge_guide_net.pt")

def load_edge_guide_model(weights_path: str = None, device: str = None) -> EdgeGuideNet:
    """
    Instantiates EdgeGuideNet and loads a checkpoint.

    Missing or incompatible checkpoints are reported instead of silently running
    random weights, which could otherwise look like a successful reconstruction.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    model = EdgeGuideNet(base_channels=32)
    
    checkpoint_path = Path(weights_path or DEFAULT_WEIGHTS_PATH).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"EdgeGuideNet checkpoint not found at '{checkpoint_path}'. "
            f"The project checkpoint is '{DEFAULT_WEIGHTS_PATH}'. "
            "Set EDGEDEPTH_CHECKPOINT to an existing EdgeGuideNet checkpoint, "
            "or remove the override to use the project checkpoint."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def run_dl_inference(model: EdgeGuideNet, 
                     depth_interp: np.ndarray, 
                     rgb: np.ndarray, 
                     mask: np.ndarray = None,
                     device: str = None) -> tuple:
    """
    Runs EdgeGuideNet deep learning inference on a single (depth, rgb) pair.
    
    Args:
        model: EdgeGuideNet instance
        depth_interp: (H, W) float32 upsampled depth map
        rgb: (H, W, 3) float32 or uint8 RGB guidance
        mask: (H, W) optional boolean or float32 validity mask
        
    Returns:
        (pred_depth, boundary_confidence): both (H, W) float32 numpy arrays
    """
    if device is None:
        device = next(model.parameters()).device
        
    depth_interp = np.asarray(depth_interp, dtype=np.float32)
    if depth_interp.ndim != 2:
        raise ValueError(f"depth_interp must be a 2D array, got {depth_interp.shape}")
    H, W = depth_interp.shape
    
    # Preprocess RGB
    g = np.asarray(rgb, dtype=np.float32)
    if g.ndim == 2:
        g = np.repeat(g[:, :, None], 3, axis=2)
    if g.ndim != 3 or g.shape[2] != 3 or g.shape[:2] != (H, W):
        raise ValueError(f"rgb must have shape ({H}, {W}, 3), got {g.shape}")
    if g.max() > 1.0:
        g = g / 255.0
    g = np.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0)
    depth_interp = np.nan_to_num(depth_interp, nan=0.0, posinf=0.0, neginf=0.0)
        
    # Tensors
    d_tensor = torch.from_numpy(depth_interp.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    rgb_tensor = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).to(device)
    
    if mask is not None:
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape != (H, W):
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {(H, W)}")
        m_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    else:
        m_tensor = (d_tensor > 1e-4).float()

    pad_h, pad_w = (-H) % 4, (-W) % 4
    if pad_h or pad_w:
        d_tensor = torch.nn.functional.pad(d_tensor, (0, pad_w, 0, pad_h), mode="replicate")
        rgb_tensor = torch.nn.functional.pad(rgb_tensor, (0, pad_w, 0, pad_h), mode="replicate")
        m_tensor = torch.nn.functional.pad(m_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    model.eval()
    with torch.inference_mode():
        pred_d, pred_edge = model(d_tensor, rgb_tensor, m_tensor)
        
    out_d = pred_d[0, 0, :H, :W].cpu().numpy().astype(np.float32)
    out_edge = pred_edge[0, 0, :H, :W].cpu().numpy().astype(np.float32)
    
    return out_d, out_edge

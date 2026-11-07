"""Checkpoint-compatible DepthUNet reconstructed from the supplied state dict."""

import os
from importlib import import_module

import cv2
import numpy as np
import torch
import torch.nn as nn

DEPTH_UNET_INPUT_SIZE = (240, 320)
SUPPORTED_SCALES = (2, 4, 8)
DEFAULT_DEPTH_UNET_WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "best_depth_model.pth",
)


class DoubleConv(nn.Module):
    """Two bias-free convolutions with batch normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DepthUNet(nn.Module):
    """Four-level U-Net matching the supplied checkpoint tensor names."""

    def __init__(self, in_channels: int = 5, out_channels: int = 1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 384)
        self.up4 = nn.ConvTranspose2d(384, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1)) * 10.0


def _safe_checkpoint_load(path: str, device: str) -> dict:
    """Load tensor weights while allowing only NumPy scalar/dtype checkpoint metadata."""
    safe_globals_context = getattr(torch.serialization, "safe_globals", None)
    if safe_globals_context is None:
        raise RuntimeError(
            "Safely loading this DepthUNet checkpoint requires PyTorch 2.6 or newer."
        )
    numpy_multiarray = import_module("numpy.core.multiarray")
    safe_globals = [
        numpy_multiarray.scalar,
        (numpy_multiarray.scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
        type(np.dtype(np.int32)),
    ]
    with safe_globals_context(safe_globals):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"DepthUNet checkpoint must be a mapping: {path}")
    return checkpoint


def load_depth_unet(weights_path: str = None, device: str = None) -> nn.Module:
    """Strictly load the trained checkpoint, retaining its DataParallel key prefix."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if weights_path is None:
        weights_path = DEFAULT_DEPTH_UNET_WEIGHTS
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DepthUNet checkpoint not found: {weights_path}")

    checkpoint = _safe_checkpoint_load(weights_path, device="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("DepthUNet checkpoint has no model_state_dict.")

    keys = tuple(state_dict)
    has_module_prefix = all(key.startswith("module.") for key in keys)
    if has_module_prefix:
        model = nn.DataParallel(DepthUNet(in_channels=5, out_channels=1))
    elif any(key.startswith("module.") for key in keys):
        raise ValueError("DepthUNet checkpoint mixes DataParallel and unwrapped parameter keys.")
    else:
        model = DepthUNet(in_channels=5, out_channels=1)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def scatter_sparse_depth(
    depth: np.ndarray,
    target_shape: tuple,
    scale_factor: int,
    force_sample_grid: bool = False,
) -> tuple:
    """Place low-resolution depth samples on their registered high-resolution grid."""
    if scale_factor not in SUPPORTED_SCALES:
        raise ValueError(f"scale_factor must be one of {SUPPORTED_SCALES}, got {scale_factor}")

    source = np.asarray(depth, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"depth must be a 2D array, got {source.shape}")
    target_h, target_w = target_shape
    if target_h < 1 or target_w < 1:
        raise ValueError(f"target_shape must be positive, got {target_shape}")

    valid = np.isfinite(source) & (source > 1e-4)
    sparse = np.zeros((target_h, target_w), dtype=np.float32)
    mask = np.zeros((target_h, target_w), dtype=np.float32)
    source_h, source_w = source.shape
    source_is_target_size = (source_h, source_w) == (target_h, target_w)
    if source_is_target_size and not force_sample_grid:
        sparse[valid] = source[valid]
        mask[valid] = 1.0
        return sparse, mask

    if not source_is_target_size or force_sample_grid:
        sample_h = max(1, target_h // scale_factor)
        sample_w = max(1, target_w // scale_factor)
        if source.shape != (sample_h, sample_w):
            source = cv2.resize(
                source, (sample_w, sample_h), interpolation=cv2.INTER_NEAREST
            )
            valid = np.isfinite(source) & (source > 1e-4)
        source_h, source_w = source.shape
        ys = np.arange(source_h) * scale_factor
        xs = np.arange(source_w) * scale_factor
    else:
        ys = np.arange(source_h)
        xs = np.arange(source_w)
    ys = np.clip(ys, 0, target_h - 1)
    xs = np.clip(xs, 0, target_w - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sparse[yy[valid], xx[valid]] = source[valid]
    mask[yy[valid], xx[valid]] = 1.0
    return sparse, mask


def prepare_depth_unet_input(
    rgb: np.ndarray,
    sparse_depth: np.ndarray,
    mask: np.ndarray = None,
) -> torch.Tensor:
    """Create RGB [0,1] + normalized sparse depth + binary mask in NCHW order."""
    rgb_array = np.asarray(rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb_array.shape}")
    depth = np.asarray(sparse_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"sparse_depth must be a 2D array, got {depth.shape}")

    rgb_array = rgb_array.astype(np.float32)
    rgb_array = np.nan_to_num(rgb_array, nan=0.0, posinf=1.0, neginf=0.0)
    if np.issubdtype(np.asarray(rgb).dtype, np.integer) or (
        rgb_array.size and np.max(rgb_array) > 1.0
    ):
        rgb_array /= 255.0
    rgb_array = np.clip(rgb_array, 0.0, 1.0)

    valid = np.isfinite(depth) & (depth > 1e-4)
    if mask is not None:
        supplied_mask = np.asarray(mask)
        if supplied_mask.shape != depth.shape:
            raise ValueError(f"mask shape {supplied_mask.shape} must match depth shape {depth.shape}")
        valid &= np.isfinite(supplied_mask) & (supplied_mask > 0)
    normalized_depth = np.zeros_like(depth, dtype=np.float32)
    normalized_depth[valid] = np.clip(depth[valid], 0.0, 10.0) / 10.0
    binary_mask = valid.astype(np.float32)

    input_h, input_w = DEPTH_UNET_INPUT_SIZE
    rgb_small = cv2.resize(rgb_array, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    if depth.shape == DEPTH_UNET_INPUT_SIZE:
        depth_small = normalized_depth
        mask_small = binary_mask
    else:
        depth_small = cv2.resize(
            normalized_depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST
        )
        mask_small = cv2.resize(
            binary_mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST
        )
    mask_small = (mask_small > 0.5).astype(np.float32)
    depth_small[mask_small == 0] = 0.0
    five_channel = np.concatenate(
        [rgb_small, depth_small[:, :, None], mask_small[:, :, None]], axis=2
    )
    return torch.from_numpy(five_channel.transpose(2, 0, 1).copy()).unsqueeze(0)


def run_depth_unet_inference(
    model: nn.Module,
    rgb: np.ndarray,
    sparse_depth: np.ndarray,
    mask: np.ndarray = None,
    device: str = None,
) -> np.ndarray:
    """Run DepthUNet and return a metric-depth map registered to the source RGB."""
    if device is None:
        device = str(next(model.parameters()).device)
    rgb_array = np.asarray(rgb)
    model_input = prepare_depth_unet_input(rgb_array, sparse_depth, mask).to(device)
    model.eval()
    with torch.inference_mode():
        prediction = model(model_input)
    if prediction.shape != (1, 1, *DEPTH_UNET_INPUT_SIZE):
        raise ValueError(f"Unexpected DepthUNet output shape: {tuple(prediction.shape)}")

    depth_small = prediction[0, 0].cpu().numpy().astype(np.float32)
    target_h, target_w = rgb_array.shape[:2]
    if (target_h, target_w) != DEPTH_UNET_INPUT_SIZE:
        depth_small = cv2.resize(
            depth_small, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )
    return np.clip(np.nan_to_num(depth_small, nan=0.0, posinf=10.0, neginf=0.0), 0.0, 10.0)

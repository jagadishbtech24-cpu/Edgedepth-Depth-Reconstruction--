"""Reusable EdgeDepth reconstruction service, independent of any UI framework."""

import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..classical.discontinuity import detect_discontinuities
from ..classical.guided_filter import guided_filter
from ..classical.interpolation import interpolate_depth
from ..classical.joint_bilateral import joint_bilateral_filter
from ..hybrid.hybrid_pipeline import HybridDepthReconstructor
from ..metrics.evaluator import evaluate_all_metrics
from ..models.depth_unet import (
    DEFAULT_DEPTH_UNET_WEIGHTS,
    DEPTH_UNET_INPUT_SIZE,
    load_depth_unet,
    run_depth_unet_inference,
    scatter_sparse_depth,
)
from ..models.weights_loader import DEFAULT_WEIGHTS_PATH
from ..utils.degradations import (
    add_sensor_noise,
    downsample_depth,
    simulate_missing_holes,
    simulate_sparse_depth,
)

METHODS = (
    "Nearest",
    "Bilinear",
    "Bicubic",
    "Joint bilateral",
    "Guided filter",
    "EdgeGuideNet",
    "DepthUNet",
    "EdgeDepth hybrid",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EDGEGUIDE_CHECKPOINT = Path(DEFAULT_WEIGHTS_PATH)


def resolve_edgeguide_checkpoint() -> Path:
    """Resolve explicit relative overrides against the project root."""
    configured = os.environ.get("EDGEDEPTH_CHECKPOINT")
    checkpoint = Path(configured) if configured else DEFAULT_EDGEGUIDE_CHECKPOINT
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    return checkpoint.resolve()


@lru_cache(maxsize=2)
def _load_reconstructor(checkpoint_path: str) -> HybridDepthReconstructor:
    return HybridDepthReconstructor(model_weights_path=checkpoint_path)


@lru_cache(maxsize=2)
def _load_depth_unet(checkpoint_path: str):
    return load_depth_unet(checkpoint_path)


def _timed(function):
    started = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - started) * 1000.0


def reconstruct(
    rgb: np.ndarray,
    input_depth: np.ndarray,
    reference: np.ndarray = None,
    settings: dict = None,
    sensor_is_low_resolution: bool = False,
) -> dict:
    """Run all existing baselines, learned models, metrics, and hybrid refinement."""
    settings = settings or {}
    scale = int(settings.get("scale", 4))
    if scale not in (2, 4, 8):
        raise ValueError(f"scale must be one of 2, 4, or 8, got {scale}")

    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    height, width = rgb.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("RGB dimensions must be non-zero.")

    if reference is not None and not sensor_is_low_resolution:
        if reference.shape != (height, width):
            raise ValueError(
                f"Reference depth shape {reference.shape} must match RGB shape {(height, width)}"
            )
        low_depth = downsample_depth(reference, scale_factor=scale)
    else:
        low_depth = np.asarray(input_depth, dtype=np.float32).copy()
    if low_depth.ndim != 2:
        raise ValueError(f"input_depth must be a 2D array, got {low_depth.shape}")

    rng = np.random.default_rng(int(settings.get("seed", 42)))
    noise_sigma = float(settings.get("noise_sigma", 0.0))
    sparsity = float(settings.get("sparsity", 0.0))
    if noise_sigma > 0:
        low_depth = add_sensor_noise(low_depth, gaussian_sigma=noise_sigma, rng=rng)
    if sparsity > 0:
        low_depth, _ = simulate_sparse_depth(
            low_depth,
            sparsity_ratio=sparsity,
            pattern=settings.get("lidar_pattern", "random"),
            rng=rng,
        )
    if bool(settings.get("simulate_holes", False)):
        low_depth, _ = simulate_missing_holes(
            low_depth,
            num_holes=3,
            max_hole_radius=max(6, min(low_depth.shape) // 12),
            rng=rng,
        )
    if not np.any(np.isfinite(low_depth) & (low_depth > 1e-4)):
        raise ValueError("The degraded sensor depth contains no valid measurements.")

    valid_low = np.isfinite(low_depth) & (low_depth > 1e-4)
    predictions = {}
    timings = {}
    for method, interpolation in (
        ("Nearest", "nearest"),
        ("Bilinear", "bilinear"),
        ("Bicubic", "bicubic"),
    ):
        predictions[method], timings[method] = _timed(
            lambda interpolation=interpolation: interpolate_depth(
                low_depth,
                (height, width),
                method=interpolation,
                mask=valid_low,
            )
        )

    predictions["Joint bilateral"], timings["Joint bilateral"] = _timed(
        lambda: joint_bilateral_filter(
            predictions["Bicubic"], rgb, radius=4, sigma_spatial=3.0, sigma_range=0.1
        )
    )

    warnings = []
    affinity_map = None
    boundary_confidence = None
    discontinuities = None
    edgeguide_checkpoint = resolve_edgeguide_checkpoint()
    if edgeguide_checkpoint.is_file():
        reconstructor = _load_reconstructor(str(edgeguide_checkpoint))
        filter_type = "bilateral" if settings.get("filter", "guided") == "bilateral" else "guided"
        hybrid_result = reconstructor.reconstruct(
            low_depth,
            rgb,
            scale_factor=scale,
            target_shape=(height, width),
            filter_type=filter_type,
            anti_texture=bool(settings.get("anti_texture", True)),
            sharpen_strength=float(settings.get("sharpen", 0.25)),
        )
        predictions["Guided filter"] = hybrid_result["depth_classical"]
        predictions["EdgeGuideNet"] = hybrid_result["depth_dl"]
        predictions["EdgeDepth hybrid"] = hybrid_result["depth_reconstructed"]
        timings["Guided filter"] = hybrid_result["timings_ms"]["guided_baseline"]
        timings["EdgeGuideNet"] = hybrid_result["timings_ms"]["deep_inference"]
        timings["EdgeDepth hybrid"] = hybrid_result["timings_ms"]["total"]
        affinity_map = hybrid_result["affinity_map"]
        boundary_confidence = hybrid_result["boundary_conf"]
        discontinuities = hybrid_result["discontinuities"]
    else:
        warnings.append(
            f"EdgeGuideNet checkpoint not found at '{edgeguide_checkpoint}'. "
            f"Expected the project checkpoint '{DEFAULT_WEIGHTS_PATH}'. "
            "EdgeGuideNet and EdgeDepth hybrid were skipped; classical methods and "
            "DepthUNet are still available. Remove an invalid EDGEDEPTH_CHECKPOINT "
            "override or set it to an existing EdgeGuideNet checkpoint."
        )
        predictions["Guided filter"], timings["Guided filter"] = _timed(
            lambda: guided_filter(
                predictions["Bicubic"], rgb, radius=6, eps=1e-3, color=True
            )
        )

    depth_unet_checkpoint = os.environ.get(
        "DEPTHUNET_CHECKPOINT", DEFAULT_DEPTH_UNET_WEIGHTS
    )
    if not os.path.isabs(depth_unet_checkpoint):
        depth_unet_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            depth_unet_checkpoint,
        )
    depth_unet = _load_depth_unet(os.path.abspath(depth_unet_checkpoint))
    sparse_depth_hr, sparse_mask_hr = scatter_sparse_depth(
        low_depth,
        DEPTH_UNET_INPUT_SIZE,
        scale_factor=scale,
        force_sample_grid=not sensor_is_low_resolution,
    )
    predictions["DepthUNet"], timings["DepthUNet"] = _timed(
        lambda: run_depth_unet_inference(
            depth_unet, rgb, sparse_depth_hr, mask=sparse_mask_hr
        )
    )

    metrics = {}
    valid_reference = None
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float32)
        if reference.shape != (height, width):
            raise ValueError(
                f"Reference depth shape {reference.shape} must match RGB shape {(height, width)}"
            )
        valid_reference = np.isfinite(reference) & (reference > 1e-4)
        metrics = {
            method: evaluate_all_metrics(prediction, reference, mask=valid_reference)
            for method, prediction in predictions.items()
        }

    boundaries = {
        method: detect_discontinuities(prediction)["boundary_mask"]
        for method, prediction in predictions.items()
    }
    return {
        "rgb": rgb,
        "low_depth": low_depth,
        "reference": reference,
        "predictions": predictions,
        "metrics": metrics,
        "timings_ms": timings,
        "affinity": affinity_map,
        "boundary_confidence": boundary_confidence,
        "discontinuities": discontinuities,
        "boundaries": boundaries,
        "scale": scale,
        "valid_reference": valid_reference,
        "sensor_is_low_resolution": bool(sensor_is_low_resolution),
        "warnings": warnings,
    }

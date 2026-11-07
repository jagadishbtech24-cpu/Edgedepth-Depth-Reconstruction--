"""Ground-truth-aware metrics and lightweight execution profiling."""

import time
import tracemalloc

import cv2
import numpy as np


def _validated_arrays(pred, gt, mask=None):
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError(f"pred and gt must be same-shaped 2D arrays; got {pred.shape} and {gt.shape}")

    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 1e-4)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != gt.shape:
            raise ValueError(f"mask shape {mask.shape} does not match depth shape {gt.shape}")
        valid &= mask
    return pred, gt, valid


def _valid_errors(pred, gt, mask=None):
    pred, gt, valid = _validated_arrays(pred, gt, mask)
    return pred[valid], gt[valid], valid


def compute_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """RMSE in the input depth unit, over finite, positive ground-truth pixels."""
    p, g, _ = _valid_errors(pred, gt, mask)
    return float(np.sqrt(np.mean((p - g) ** 2))) if p.size else float("nan")


def compute_mae(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """MAE in the input depth unit, over finite, positive ground-truth pixels."""
    p, g, _ = _valid_errors(pred, gt, mask)
    return float(np.mean(np.abs(p - g))) if p.size else float("nan")


def compute_abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean absolute relative depth error."""
    p, g, _ = _valid_errors(pred, gt, mask)
    return float(np.mean(np.abs(p - g) / g)) if p.size else float("nan")


def compute_delta_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 1.25,
                           mask: np.ndarray = None) -> float:
    """Fraction for which max(pred/gt, gt/pred) is below ``threshold``."""
    p, g, _ = _valid_errors(pred, gt, mask)
    if not p.size:
        return float("nan")
    positive = p > 1e-4
    ratio = np.full_like(p, np.inf)
    ratio[positive] = np.maximum(p[positive] / g[positive], g[positive] / p[positive])
    return float(np.mean(ratio <= threshold))


def compute_psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = None,
                 mask: np.ndarray = None) -> float:
    """Depth PSNR (dB), using the valid ground-truth range unless explicitly supplied."""
    p, g, _ = _valid_errors(pred, gt, mask)
    if not p.size:
        return float("nan")
    mse = float(np.mean((p - g) ** 2))
    if mse == 0:
        return float("inf")
    data_range = float(max_val) if max_val is not None else float(np.max(g) - np.min(g))
    if data_range <= 0:
        data_range = 1.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray, max_val: float = None,
                 mask: np.ndarray = None, k1: float = 0.01, k2: float = 0.03) -> float:
    """SSIM over valid pixels with normalized local statistics around missing samples."""
    pred, gt, valid = _validated_arrays(pred, gt, mask)
    if not np.any(valid):
        return float("nan")

    height, width = gt.shape
    window_size = min(11, min(height, width))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        return float("nan")

    data_range = float(max_val) if max_val is not None else float(np.ptp(gt[valid]))
    if data_range <= 0:
        data_range = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    kernel = cv2.getGaussianKernel(window_size, 1.5)
    kernel = kernel @ kernel.T
    valid_f = valid.astype(np.float32)
    support = cv2.filter2D(valid_f, -1, kernel, borderType=cv2.BORDER_REFLECT)
    safe_support = np.maximum(support, 1e-8)

    p = np.where(valid, pred, 0).astype(np.float32)
    g = np.where(valid, gt, 0).astype(np.float32)
    mu_p = cv2.filter2D(p, -1, kernel, borderType=cv2.BORDER_REFLECT) / safe_support
    mu_g = cv2.filter2D(g, -1, kernel, borderType=cv2.BORDER_REFLECT) / safe_support
    var_p = cv2.filter2D(p * p, -1, kernel, borderType=cv2.BORDER_REFLECT) / safe_support - mu_p**2
    var_g = cv2.filter2D(g * g, -1, kernel, borderType=cv2.BORDER_REFLECT) / safe_support - mu_g**2
    cov_pg = cv2.filter2D(p * g, -1, kernel, borderType=cv2.BORDER_REFLECT) / safe_support - mu_p * mu_g

    numerator = (2 * mu_p * mu_g + c1) * (2 * cov_pg + c2)
    denominator = (mu_p**2 + mu_g**2 + c1) * (var_p + var_g + c2)
    ssim = numerator / np.maximum(denominator, 1e-12)
    centers = valid & (support >= 0.5)
    return float(np.mean(ssim[centers])) if np.any(centers) else float("nan")


def _edge_map(depth, valid, edge_percentile):
    depth_f = np.where(valid, depth, 0).astype(np.float32)
    gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    support = cv2.erode(
        valid.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if not np.any(support):
        support = valid
    values = magnitude[support]
    if not values.size or float(np.max(values)) <= 1e-8:
        return np.zeros(depth.shape, dtype=bool)
    threshold = float(np.percentile(values, edge_percentile))
    return support & (magnitude >= threshold) & (magnitude > 1e-8)


def compute_edge_rmse(pred: np.ndarray, gt: np.ndarray, edge_percentile: float = 85.0,
                      mask: np.ndarray = None) -> float:
    """RMSE on high-gradient ground-truth pixels with valid local support."""
    pred, gt, valid = _validated_arrays(pred, gt, mask)
    edges = _edge_map(gt, valid, edge_percentile)
    if not np.any(edges):
        return float("nan")
    return compute_rmse(pred, gt, mask=edges)


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, edge_percentile: float = 85.0,
                        mask: np.ndarray = None, tolerance: int = 1) -> dict:
    """Boundary precision, recall and F1 with an explicit pixel tolerance."""
    pred, gt, valid = _validated_arrays(pred, gt, mask)
    edge_gt = _edge_map(gt, valid, edge_percentile)
    edge_pred = _edge_map(pred, valid, edge_percentile)
    n_gt, n_pred = int(edge_gt.sum()), int(edge_pred.sum())
    if n_gt == 0 and n_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}
    if n_gt == 0 or n_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    kernel_size = 2 * max(0, int(tolerance)) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    gt_neighborhood = cv2.dilate(edge_gt.astype(np.uint8), kernel).astype(bool)
    pred_neighborhood = cv2.dilate(edge_pred.astype(np.uint8), kernel).astype(bool)
    precision = float(np.count_nonzero(edge_pred & gt_neighborhood) / n_pred)
    recall = float(np.count_nonzero(edge_gt & pred_neighborhood) / n_gt)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def compute_gradient_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean absolute error between signed horizontal/vertical depth gradients."""
    pred, gt, valid = _validated_arrays(pred, gt, mask)
    pred_f = np.where(valid, pred, 0).astype(np.float32)
    gt_f = np.where(valid, gt, 0).astype(np.float32)
    pred_x = cv2.Sobel(pred_f, cv2.CV_32F, 1, 0, ksize=3)
    pred_y = cv2.Sobel(pred_f, cv2.CV_32F, 0, 1, ksize=3)
    gt_x = cv2.Sobel(gt_f, cv2.CV_32F, 1, 0, ksize=3)
    gt_y = cv2.Sobel(gt_f, cv2.CV_32F, 0, 1, ksize=3)
    support = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                        borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not np.any(support):
        support = valid
    if not np.any(support):
        return float("nan")
    return float(np.mean(np.abs(pred_x[support] - gt_x[support]) +
                         np.abs(pred_y[support] - gt_y[support])))


def profile_execution(func, *args, repeats: int = 3, warmup: int = 1, **kwargs) -> tuple:
    """Return result, median latency (ms), and Python-traced peak allocation (MiB).

    The memory figure excludes native allocations made by NumPy, OpenCV, and PyTorch.
    """
    repeats = max(1, int(repeats))
    for _ in range(max(0, int(warmup))):
        func(*args, **kwargs)

    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    latencies = []
    result = None
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            latencies.append((time.perf_counter() - start) * 1000.0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()
    return result, float(np.median(latencies)), float(peak / (1024.0 * 1024.0))


def evaluate_all_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> dict:
    """Compute accuracy, relative-depth, structural and boundary metrics."""
    p, g, valid = _valid_errors(pred, gt, mask)
    if not p.size:
        return {
            "RMSE": float("nan"), "MAE": float("nan"), "AbsRel": float("nan"),
            "Delta1": float("nan"), "Delta2": float("nan"), "Delta3": float("nan"),
            "PSNR": float("nan"), "SSIM": float("nan"), "Edge_RMSE": float("nan"),
            "Gradient_MAE": float("nan"), "Boundary_Precision": float("nan"),
            "Boundary_Recall": float("nan"), "Boundary_F1": float("nan"),
            "Valid_Pixels": 0,
        }

    boundary = compute_boundary_f1(pred, gt, mask=valid)
    return {
        "RMSE": float(np.sqrt(np.mean((p - g) ** 2))),
        "MAE": float(np.mean(np.abs(p - g))),
        "AbsRel": compute_abs_rel(pred, gt, mask=valid),
        "Delta1": compute_delta_accuracy(pred, gt, 1.25, mask=valid),
        "Delta2": compute_delta_accuracy(pred, gt, 1.25**2, mask=valid),
        "Delta3": compute_delta_accuracy(pred, gt, 1.25**3, mask=valid),
        "PSNR": compute_psnr(pred, gt, mask=valid),
        "SSIM": compute_ssim(pred, gt, mask=valid),
        "Edge_RMSE": compute_edge_rmse(pred, gt, mask=valid),
        "Gradient_MAE": compute_gradient_error(pred, gt, mask=valid),
        "Boundary_Precision": boundary["precision"],
        "Boundary_Recall": boundary["recall"],
        "Boundary_F1": boundary["f1_score"],
        "Valid_Pixels": int(valid.sum()),
    }

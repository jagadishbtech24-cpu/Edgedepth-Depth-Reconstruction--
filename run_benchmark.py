"""Reproducible, held-out NYU Depth V2 comparison for EdgeDepth."""

import argparse
import os
import platform
import time

import cv2
import numpy as np
import pandas as pd
import torch

from src.classical.guided_filter import guided_filter
from src.classical.interpolation import interpolate_depth
from src.classical.joint_bilateral import joint_bilateral_filter
from src.metrics.evaluator import evaluate_all_metrics, profile_execution
from src.models.weights_loader import run_dl_inference
from src.hybrid.hybrid_pipeline import HybridDepthReconstructor
from src.utils.dataset import get_full_nyu_v2_scenes, load_depth_map, load_rgb_image
from src.utils.degradations import downsample_depth


ROOT = os.path.dirname(os.path.abspath(__file__))
METHODS = (
    "Nearest",
    "Bilinear",
    "Bicubic",
    "Joint bilateral",
    "Guided filter",
    "EdgeGuideNet",
    "EdgeDepth hybrid",
)


def _scene_subset(scenes: dict, max_scenes: int, seed: int) -> dict:
    keys = sorted(scenes)
    if max_scenes <= 0 or len(keys) <= max_scenes:
        return {key: scenes[key] for key in keys}
    selected = np.random.default_rng(seed).choice(len(keys), max_scenes, replace=False)
    return {keys[int(index)]: scenes[keys[int(index)]] for index in sorted(selected)}


def write_benchmark_report(frame: pd.DataFrame, output_md_path: str = None,
                           output_csv_path: str = None, scales=(2, 4, 8),
                           device: str = "unknown", elapsed_seconds: float = 0.0,
                           repeats: int = 1, seed: int = 42,
                           checkpoint_path: str = ""):
    """Write reusable CSV and Markdown outputs from measured per-scene rows."""
    output_md_path = output_md_path or os.path.join(ROOT, "eval_P4_T01.md")
    output_csv_path = output_csv_path or os.path.join(ROOT, "eval_P4_T01.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_md_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    frame.to_csv(output_csv_path, index=False)

    metric_columns = [
        "RMSE", "MAE", "AbsRel", "Delta1", "Delta2", "Delta3", "PSNR",
        "SSIM", "Edge_RMSE", "Gradient_MAE", "Boundary_Precision",
        "Boundary_Recall", "Boundary_F1", "Latency_ms", "Python_peak_alloc_MB",
    ]
    compact_summary = frame.groupby("Method")[metric_columns].mean().reset_index()
    per_scene = frame.groupby(["Method", "Scene_ID"], as_index=False)[metric_columns].mean()
    summary = per_scene.groupby("Method")[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "Method" if column[0] == "Method" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    best_rmse = compact_summary.loc[compact_summary["RMSE"].idxmin(), "Method"]
    best_boundary = compact_summary.loc[compact_summary["Boundary_F1"].idxmax(), "Method"]
    lines = [
        "1MS22CS001, 1MS22CS002, 1MS22CS003, 1MS22CS004",
        "",
        "# EdgeDepth held-out evaluation",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset: official NYU Depth V2 validation split; {frame['Scene_ID'].nunique()} scenes",
        f"- EdgeGuideNet checkpoint: `{checkpoint_path}`",
        f"- Scales: {', '.join(str(scale) + 'x' for scale in scales)}",
        f"- Device: {device} ({platform.platform()})",
        f"- Repeats per method: {repeats}; latency is the median of timed calls after one warm-up",
        f"- Runtime: {elapsed_seconds / 60.0:.1f} minutes; seed: {seed}",
        "- Metrics are computed only where ground-truth depth is finite and positive.",
        "- Aggregate metrics average scene-scale runs equally; mean and standard deviation below first average scales within each scene.",
        "- Python peak allocation excludes memory allocated internally by NumPy, OpenCV, and PyTorch; it is not process peak RSS.",
        "- Checkpoint provenance depends on the selected weights; inspect its training metadata before interpreting results.",
        "- This is a finite validation subset, not a claim of universal or real-time performance.",
        "",
        "## Aggregate results",
        "",
        compact_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Mean and standard deviation across scenes",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Results by scale",
        "",
    ]
    for scale in scales:
        scale_frame = frame[frame["Scale"] == f"{scale}x"].groupby("Method")[metric_columns].mean().reset_index()
        lines.extend([f"### {scale}x", "", scale_frame.to_markdown(index=False, floatfmt=".4f"), ""])
    lines.extend([
        "## Observed leaders (this run only)",
        "",
        f"- Lowest mean RMSE: **{best_rmse}**.",
        f"- Highest mean boundary F1: **{best_boundary}**.",
        "- These rankings are descriptive and may change with the scene subset, sensor degradation, and hardware.",
        "",
        "## Metric definitions",
        "",
        "- **RMSE / MAE:** absolute depth error in meters.",
        "- **AbsRel:** mean absolute error divided by ground-truth depth.",
        "- **Delta1/2/3:** fraction within multiplicative thresholds 1.25, 1.25², and 1.25³.",
        "- **PSNR / SSIM:** image-structure measures over valid depth support; PSNR uses the valid ground-truth depth range.",
        "- **Edge RMSE:** depth error on the strongest 15% of supported ground-truth gradients.",
        "- **Boundary precision / recall / F1:** depth-edge overlap with one-pixel tolerance.",
        "- **Gradient MAE:** error in signed spatial depth gradients.",
        "- **Latency:** median wall-clock milliseconds for one method call.",
        "",
        f"Per-scene results: `{os.path.basename(output_csv_path)}`.",
        "",
    ])
    with open(output_md_path, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))
    print(f"Saved report: {output_md_path}")
    print(f"Saved per-scene CSV: {output_csv_path}")


def run_evaluation_benchmark(output_md_path: str = None, output_csv_path: str = None,
                             max_scenes: int = 12, repeats: int = 3,
                             seed: int = 42, scales=(2, 4, 8),
                             checkpoint_path: str = None, dataset_root: str = None):
    scenes = get_full_nyu_v2_scenes(
        split="val", max_count=10000, dataset_root=dataset_root
    )
    if not scenes:
        raise RuntimeError(
            "The official NYU Depth V2 validation split was not found. "
            "The benchmark will not substitute training or curated scenes."
        )
    selected = _scene_subset(scenes, max_scenes, seed)
    if not selected:
        raise ValueError("max_scenes must be a positive number")

    checkpoint_path = checkpoint_path or os.path.join(ROOT, "models", "edge_guide_net.pt")
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"EdgeGuideNet checkpoint not found: {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reconstructor = HybridDepthReconstructor(model_weights_path=checkpoint_path, device=device)
    results = []
    started = time.perf_counter()
    print(
        f"Evaluating {len(selected)} held-out NYU validation scenes, "
        f"scales {list(scales)}, device={device}, repeats={repeats}"
    )

    for scene_index, (scene_key, info) in enumerate(selected.items(), start=1):
        rgb = load_rgb_image(info["rgb_path"])
        target = load_depth_map(info["depth_path"])
        if rgb.shape[:2] != target.shape:
            rgb = cv2.resize(rgb, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(target) & (target > 1e-4)
        if not np.any(valid):
            print(f"Skipping {scene_key}: no valid ground-truth depth")
            continue
        height, width = target.shape
        print(f"[{scene_index}/{len(selected)}] {info['name']} ({height}x{width})")

        for scale in scales:
            low_depth = downsample_depth(target, scale_factor=int(scale))
            base = interpolate_depth(low_depth, (height, width), method="bicubic")

            method_functions = {
                "Nearest": lambda d=low_depth: interpolate_depth(d, (height, width), method="nearest"),
                "Bilinear": lambda d=low_depth: interpolate_depth(d, (height, width), method="bilinear"),
                "Bicubic": lambda d=low_depth: interpolate_depth(d, (height, width), method="bicubic"),
                "Joint bilateral": lambda b=base, i=rgb: joint_bilateral_filter(
                    b, i, radius=4, sigma_spatial=3.0, sigma_range=0.1
                ),
                "Guided filter": lambda b=base, i=rgb: guided_filter(
                    b, i, radius=6, eps=1e-3, color=True
                ),
                "EdgeGuideNet": lambda b=base, i=rgb, m=reconstructor.model, dev=device:
                    run_dl_inference(m, b, i, device=dev)[0],
                "EdgeDepth hybrid": lambda d=low_depth, i=rgb, s=int(scale), shape=(height, width), r=reconstructor:
                    r.reconstruct(
                        d, i, scale_factor=s, target_shape=shape,
                        filter_type="guided", anti_texture=True, sharpen_strength=0.35,
                    )["depth_reconstructed"],
            }

            for method_name in METHODS:
                fn = method_functions[method_name]
                prediction, latency_ms, python_peak_mb = profile_execution(
                    fn, repeats=repeats, warmup=1
                )
                metrics = evaluate_all_metrics(prediction, target, mask=valid)
                metrics.update({
                    "Scene": info["name"],
                    "Scene_ID": scene_key,
                    "Scale": f"{scale}x",
                    "Method": method_name,
                    "Latency_ms": latency_ms,
                    "Python_peak_alloc_MB": python_peak_mb,
                })
                results.append(metrics)
            print(f"  scale {scale}x complete")

    if not results:
        raise RuntimeError("No scenes with valid ground truth were evaluated")
    frame = pd.DataFrame(results)
    elapsed = time.perf_counter() - started
    write_benchmark_report(
        frame,
        output_md_path=output_md_path,
        output_csv_path=output_csv_path,
        scales=scales,
        device=device,
        elapsed_seconds=elapsed,
        repeats=repeats,
        seed=seed,
        checkpoint_path=os.path.relpath(checkpoint_path, ROOT),
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("models", "edge_guide_net.pt"),
        help="Trained EdgeGuideNet checkpoint to evaluate.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Folder containing images/ and depth/, or its parent containing nyu-depth-png/.",
    )
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    run_evaluation_benchmark(
        output_md_path=args.output_md,
        output_csv_path=args.output_csv,
        max_scenes=args.max_scenes,
        repeats=args.repeats,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()

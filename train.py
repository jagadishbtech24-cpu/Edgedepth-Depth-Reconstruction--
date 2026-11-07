"""Train EdgeGuideNet from the official NYU train split with a fixed holdout."""

import argparse
import glob
import os
import random
import tempfile
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.classical.interpolation import interpolate_depth
from src.models.guided_depth_net import EdgeGuideNet
from src.models.losses import EdgePreservingLoss
from src.utils.dataset import load_depth_map, load_rgb_image


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(ROOT, "models", "edge_guide_net.pt")


def _find_pairs(samples_dir: str, split: str, dataset_root: str = None):
    nyu_root = os.path.join(os.path.dirname(samples_dir), "nyu_depth_v2", "nyu-depth-png")
    if dataset_root:
        dataset_root = os.path.abspath(dataset_root)
        nested_root = os.path.join(dataset_root, "nyu-depth-png")
        if os.path.isdir(nested_root):
            nyu_root = nested_root
        elif os.path.isdir(os.path.join(dataset_root, "images")) and os.path.isdir(
            os.path.join(dataset_root, "depth")
        ):
            nyu_root = dataset_root
        else:
            raise FileNotFoundError(
                f"Expected images/ and depth/ folders, or a nyu-depth-png subfolder, "
                f"under --dataset-root: {dataset_root}"
            )
    image_dir = os.path.join(nyu_root, "images", "train")
    depth_dir = os.path.join(nyu_root, "depth", "train")
    pairs = []
    if os.path.isdir(image_dir) and os.path.isdir(depth_dir):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        for image_path in image_paths:
            depth_path = os.path.join(depth_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.png")
            if os.path.isfile(depth_path):
                pairs.append((image_path, depth_path))
    if not pairs:
        return []
    rng = np.random.default_rng(42)
    order = rng.permutation(len(pairs))
    cut = max(1, int(round(len(pairs) * 0.9)))
    selected = order[:cut] if split == "train" else order[cut:]
    return [pairs[int(index)] for index in selected]


class DepthUpsamplingDataset(Dataset):
    """Loads paired scenes lazily and creates aligned, randomized training crops."""

    def __init__(self, samples_dir: str, split: str = "train", patch_size: int = 96,
                 samples_per_epoch: int = None, validation_samples: int = 24,
                 seed: int = 42, dataset_root: str = None):
        self.samples_dir = samples_dir
        self.split = split
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch) if samples_per_epoch is not None else None
        self.seed = int(seed)
        self.epoch = 0

        pairs = _find_pairs(samples_dir, split, dataset_root=dataset_root)
        if not pairs and dataset_root is None:
            sample_pairs = []
            for rgb_path in sorted(glob.glob(os.path.join(samples_dir, "*_rgb.png"))):
                base = rgb_path[:-len("_rgb.png")]
                depth_path = f"{base}_depth.npy"
                if not os.path.isfile(depth_path):
                    depth_path = f"{base}_depth.png"
                if os.path.isfile(depth_path):
                    sample_pairs.append((rgb_path, depth_path))
            if len(sample_pairs) < 2:
                raise RuntimeError(
                    f"Need at least two RGB-D pairs to build train/validation sets; found {len(sample_pairs)}"
                )
            cut = max(1, int(round(len(sample_pairs) * 0.8)))
            pairs = sample_pairs[:cut] if split == "train" else sample_pairs[cut:]
            if not pairs:
                raise RuntimeError(f"No {split} RGB-D pairs are available in {samples_dir}")
        elif not pairs:
            raise RuntimeError(
                f"No paired NYU {split} images and depths found under {dataset_root}. "
                f"Expected images\\train and depth\\train (or corresponding val folders)."
            )

        if split != "train" and len(pairs) > validation_samples:
            rng = np.random.default_rng(seed)
            selected = np.sort(rng.choice(len(pairs), size=validation_samples, replace=False))
            pairs = [pairs[int(index)] for index in selected]
        self.pairs = pairs

    def __len__(self):
        if self.split != "train" or self.samples_per_epoch is None:
            return len(self.pairs)
        return min(self.samples_per_epoch, len(self.pairs))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * max(len(self), 1) + index)
        pair_index = index % len(self.pairs)
        rgb_path, depth_path = self.pairs[pair_index]
        rgb = load_rgb_image(rgb_path)
        depth = load_depth_map(depth_path)
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid_target = np.isfinite(depth) & (depth > 1e-4)
        depth = np.where(valid_target, depth, 0).astype(np.float32)
        height, width = depth.shape

        if height < self.patch_size or width < self.patch_size:
            rgb = cv2.resize(rgb, (max(width, self.patch_size), max(height, self.patch_size)))
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            valid_target = depth > 1e-4
            height, width = depth.shape
        if self.split == "train":
            y0 = int(rng.integers(height - self.patch_size + 1))
            x0 = int(rng.integers(width - self.patch_size + 1))
        else:
            y0 = (height - self.patch_size) // 2
            x0 = (width - self.patch_size) // 2
        y1, x1 = y0 + self.patch_size, x0 + self.patch_size
        rgb = rgb[y0:y1, x0:x1].copy()
        depth = depth[y0:y1, x0:x1].copy()
        valid_target = valid_target[y0:y1, x0:x1].copy()

        if self.split == "train":
            if rng.random() < 0.5:
                rgb, depth, valid_target = np.fliplr(rgb), np.fliplr(depth), np.fliplr(valid_target)
            if rng.random() < 0.5:
                rgb, depth, valid_target = np.flipud(rgb), np.flipud(depth), np.flipud(valid_target)

        scales = (2, 4, 8)
        scale = int(rng.choice(scales)) if self.split == "train" else scales[index % len(scales)]
        low_h, low_w = max(1, self.patch_size // scale), max(1, self.patch_size // scale)
        low_depth = cv2.resize(depth, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
        low_valid = low_depth > 1e-4

        if self.split == "train":
            if rng.random() < 0.5:
                noise = rng.normal(0.0, 0.02, low_depth.shape).astype(np.float32)
                low_depth[low_valid] *= 1.0 + noise[low_valid]
            if rng.random() < 0.35:
                keep = rng.random(low_valid.shape) >= 0.12
                low_valid &= keep
                low_depth[~low_valid] = 0.0
            if rng.random() < 0.2:
                cy = int(rng.integers(low_h))
                cx = int(rng.integers(low_w))
                ry = max(1, low_h // 10)
                rx = max(1, low_w // 10)
                low_valid[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx] = False
                low_depth[~low_valid] = 0.0

        interp = interpolate_depth(low_depth, (self.patch_size, self.patch_size),
                                   method="bicubic", mask=low_valid)
        input_mask = cv2.resize(low_valid.astype(np.uint8),
                                (self.patch_size, self.patch_size),
                                interpolation=cv2.INTER_NEAREST).astype(np.float32)

        depth_f = depth.astype(np.float32)
        gx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)
        valid_grad = grad[valid_target]
        threshold = float(np.percentile(valid_grad, 85)) if valid_grad.size else float("inf")
        boundary = ((grad >= threshold) & (grad > 1e-8) & valid_target).astype(np.float32)

        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        return (
            torch.from_numpy(interp[None].copy()).float(),
            rgb_t,
            torch.from_numpy(input_mask[None].copy()).float(),
            torch.from_numpy(depth[None].copy()).float(),
            torch.from_numpy(boundary[None].copy()).float(),
            torch.from_numpy(valid_target[None].astype(np.float32)),
        )


def _load_checkpoint(model, checkpoint_path, device):
    if not os.path.isfile(checkpoint_path):
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return True


def _validate(model, loader, criterion, device):
    model.eval()
    total_squared_error = 0.0
    total_pixels = 0
    total_loss = 0.0
    batches = 0
    with torch.inference_mode():
        for interp, rgb, input_mask, target, boundary, target_mask in loader:
            interp, rgb = interp.to(device), rgb.to(device)
            input_mask, target = input_mask.to(device), target.to(device)
            boundary, target_mask = boundary.to(device), target_mask.to(device)
            pred, pred_boundary = model(interp, rgb, input_mask)
            loss = criterion(pred, pred_boundary, target, boundary, target_mask)
            total_loss += float(loss["total_loss"].item())
            valid = target_mask > 0
            total_squared_error += float(((pred - target).square() * target_mask).sum().item())
            total_pixels += int(valid.sum().item())
            batches += 1
    rmse = float(np.sqrt(total_squared_error / max(total_pixels, 1)))
    return rmse, total_loss / max(batches, 1)


def train_edge_guide_net(epochs: int = 5, batch_size: int = 8, lr: float = 2e-4,
                         patch_size: int = 64, samples_per_epoch: int = None,
                         validation_samples: int = 24, seed: int = 42,
                         save_path: str = None, from_scratch: bool = False,
                         dataset_root: str = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, os.cpu_count() or 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples_dir = os.path.join(ROOT, "data", "samples")
    train_data = DepthUpsamplingDataset(
        samples_dir, "train", patch_size, samples_per_epoch, validation_samples, seed,
        dataset_root=dataset_root,
    )
    val_data = DepthUpsamplingDataset(
        samples_dir, "val", patch_size, samples_per_epoch, validation_samples, seed,
        dataset_root=dataset_root,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0, generator=generator,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    model = EdgeGuideNet(base_channels=32).to(device)
    checkpoint_path = save_path or DEFAULT_CHECKPOINT
    loaded = _load_checkpoint(model, checkpoint_path, device) if not from_scratch else False
    criterion = EdgePreservingLoss(w_depth=1.0, w_grad=0.4, w_bce=0.15).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05
    )

    baseline_rmse, baseline_loss = _validate(model, val_loader, criterion, device)
    best_rmse = float("inf") if from_scratch else baseline_rmse
    best_epoch = 0
    print(
        f"Device: {device}; train pairs: {len(train_data.pairs)}; "
        f"validation pairs: {len(val_data.pairs)}; checkpoint loaded: {loaded}; "
        f"training samples per epoch: {len(train_data)}"
    )
    print(f"Initial validation RMSE: {baseline_rmse:.6f} m; loss: {baseline_loss:.6f}")

    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        batch_count = 0
        started = time.perf_counter()
        for interp, rgb, input_mask, target, boundary, target_mask in train_loader:
            interp, rgb = interp.to(device), rgb.to(device)
            input_mask, target = input_mask.to(device), target.to(device)
            boundary, target_mask = boundary.to(device), target_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred, pred_boundary = model(interp, rgb, input_mask)
            losses = criterion(pred, pred_boundary, target, boundary, target_mask)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += float(losses["total_loss"].item())
            batch_count += 1

        scheduler.step()
        val_rmse, val_loss = _validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{epochs:02d} | train loss {loss_sum / max(batch_count, 1):.6f} "
            f"| val loss {val_loss:.6f} | val RMSE {val_rmse:.6f} m "
            f"| {time.perf_counter() - started:.1f}s"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            checkpoint = {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "validation_rmse": val_rmse,
                "seed": seed,
                "training_dataset": "NYU Depth V2 official training split",
                "training_pairs": len(train_data.pairs),
                "training_samples_per_epoch": len(train_data),
                "validation_dataset": "fixed holdout from NYU Depth V2 training split",
                "validation_pairs": len(val_data.pairs),
                "from_scratch": from_scratch,
            }
            fd, temp_path = tempfile.mkstemp(
                prefix="edge-guide-", suffix=".pt",
                dir=os.path.dirname(os.path.abspath(checkpoint_path)),
            )
            os.close(fd)
            try:
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, checkpoint_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    improved = best_epoch > 0
    print(
        f"Training complete. Validation RMSE: {baseline_rmse:.6f} -> {best_rmse:.6f} m. "
        f"{'Saved epoch ' + str(best_epoch) + ' to ' + checkpoint_path if improved else 'No checkpoint replaced; validation did not improve.'}"
    )
    return {
        "baseline_rmse": baseline_rmse,
        "best_rmse": best_rmse,
        "best_epoch": best_epoch,
        "checkpoint_path": checkpoint_path if improved else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=None,
        help="Default visits every training pair once per epoch.",
    )
    parser.add_argument("--validation-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Folder containing images/ and depth/, or its parent containing nyu-depth-png/.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Initialize new model weights instead of loading the existing checkpoint.",
    )
    args = parser.parse_args()
    train_edge_guide_net(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        validation_samples=args.validation_samples,
        seed=args.seed,
        save_path=args.checkpoint,
        from_scratch=args.from_scratch,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()

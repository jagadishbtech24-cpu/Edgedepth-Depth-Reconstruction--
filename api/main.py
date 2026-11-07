"""FastAPI endpoints for EdgeDepth scene selection and reconstruction."""

import base64
import io
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError, validator

from src.metrics.evaluator import compute_boundary_f1
from src.services.reconstruction import (
    reconstruct,
    resolve_edgeguide_checkpoint,
)
from src.utils.dataset import (
    get_available_presets,
    get_full_nyu_v2_scenes,
    load_depth_map,
    load_rgb_image,
)
from src.utils.visualization import (
    colorize_depth,
    compute_error_heatmap,
    overlay_edges_on_image,
)

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT / "frontend" / "dist"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
RESULT_TTL_SECONDS = 20 * 60
MAX_STORED_RESULTS = 3


class ReconstructionSettings(BaseModel):
    scale: int = 4
    noise_sigma: float = Field(default=0.0, ge=0.0, le=0.08)
    sparsity: float = Field(default=0.0, ge=0.0, le=0.2)
    lidar_pattern: str = "grid"
    simulate_holes: bool = False
    filter: str = "guided"
    anti_texture: bool = True
    sharpen: float = Field(default=0.25, ge=0.0, le=1.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)

    @validator("scale")
    def validate_scale(cls, value):
        if value not in (2, 4, 8):
            raise ValueError("scale must be 2, 4, or 8")
        return value

    @validator("lidar_pattern")
    def validate_lidar_pattern(cls, value):
        if value not in {"random", "scanline", "grid"}:
            raise ValueError("lidar_pattern must be random, scanline, or grid")
        return value

    @validator("filter")
    def validate_filter(cls, value):
        if value not in {"guided", "bilateral"}:
            raise ValueError("filter must be guided or bilateral")
        return value


app = FastAPI(
    title="EdgeDepth API",
    version="1.0.0",
    description="RGB-guided depth reconstruction and scientific evaluation.",
)
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("FRONTEND_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_results = OrderedDict()
_results_lock = threading.Lock()


def _scene_catalog(source: str) -> dict:
    if source == "presets":
        return get_available_presets()
    if source == "nyu":
        return get_full_nyu_v2_scenes(split="val", max_count=10000)
    raise ValueError("source must be 'nyu' or 'presets'")


def _read_rgb_upload(data: bytes, filename: str) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("RGB image exceeds the 16-megapixel upload limit.")
            return np.asarray(image, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode RGB image '{filename}'.") from exc


def _read_depth_upload(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy" or data.startswith(b"\x93NUMPY"):
        try:
            depth = np.load(io.BytesIO(data), allow_pickle=False)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Could not decode depth array '{filename}'.") from exc
    else:
        encoded = np.frombuffer(data, dtype=np.uint8)
        depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Could not decode depth image '{filename}'.")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0 * 5.0
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth data must be 2D; got shape {depth.shape}.")
    if depth.size > MAX_IMAGE_PIXELS:
        raise ValueError("Depth map exceeds the 16-megapixel upload limit.")
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def _png_data_url(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG", optimize=True
    )
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _depth_preview(depth: np.ndarray, low: float, high: float) -> str:
    return _png_data_url(colorize_depth(depth, vmin=low, vmax=high, colormap="turbo"))


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    return value


def _store_result(result: dict) -> str:
    result_id = uuid.uuid4().hex
    now = time.time()
    with _results_lock:
        expired = [
            key
            for key, stored in _results.items()
            if now - stored["created_at"] > RESULT_TTL_SECONDS
        ]
        for key in expired:
            _results.pop(key, None)
        _results[result_id] = {"created_at": now, "result": result}
        while len(_results) > MAX_STORED_RESULTS:
            _results.popitem(last=False)
    return result_id


def _get_result(result_id: str) -> dict:
    with _results_lock:
        stored = _results.get(result_id)
        if stored is None or time.time() - stored["created_at"] > RESULT_TTL_SECONDS:
            _results.pop(result_id, None)
            raise HTTPException(
                status_code=404,
                detail="Result expired or was evicted. Run reconstruction again.",
            )
        _results.move_to_end(result_id)
        return stored["result"]


def _prepare_response(result: dict, result_id: str, scene_name: str) -> dict:
    reference = result["reference"]
    display_source = reference if reference is not None else result["low_depth"]
    valid_source = display_source[np.isfinite(display_source) & (display_source > 1e-4)]
    if valid_source.size:
        low, high = (float(value) for value in np.percentile(valid_source, [1, 99]))
        high = max(high, low + 1e-4)
    else:
        low, high = 0.0, 1.0

    previews = {
        method: _depth_preview(depth, low, high)
        for method, depth in result["predictions"].items()
    }
    boundaries = {
        method: _png_data_url(
            overlay_edges_on_image(result["rgb"], edge, color=(73, 200, 192))
        )
        for method, edge in result["boundaries"].items()
    }
    errors = {}
    if reference is not None:
        errors = {
            method: _png_data_url(
                compute_error_heatmap(
                    prediction,
                    reference,
                    mask=result["valid_reference"],
                )
            )
            for method, prediction in result["predictions"].items()
        }
    reference_edges = None
    boundary_metrics = {}
    if reference is not None:
        from src.classical.discontinuity import detect_discontinuities

        reference_boundary = detect_discontinuities(reference)["boundary_mask"]
        reference_edges = _png_data_url(
            overlay_edges_on_image(result["rgb"], reference_boundary, color=(255, 184, 77))
        )
        for method, prediction in result["predictions"].items():
            boundary_metrics[method] = compute_boundary_f1(
                prediction,
                reference,
                mask=result["valid_reference"],
            )

    valid_depth = result["low_depth"][
        np.isfinite(result["low_depth"]) & (result["low_depth"] > 1e-4)
    ]
    return _finite_json(
        {
            "result_id": result_id,
            "scene_name": scene_name,
            "width": int(result["rgb"].shape[1]),
            "height": int(result["rgb"].shape[0]),
            "scale": int(result["scale"]),
            "sensor_is_low_resolution": result["sensor_is_low_resolution"],
            "valid_input_samples": int(valid_depth.size),
            "rgb_preview": _png_data_url(
                np.clip(result["rgb"] * 255.0, 0, 255).astype(np.uint8)
            ),
            "input_depth_preview": _depth_preview(result["low_depth"], low, high),
            "reference_preview": (
                _depth_preview(reference, low, high) if reference is not None else None
            ),
            "prediction_previews": previews,
            "boundary_previews": boundaries,
            "reference_boundary_preview": reference_edges,
            "error_previews": errors,
            "affinity_preview": (
                _depth_preview(result["affinity"], 0.0, 1.0)
                if result["affinity"] is not None
                else None
            ),
            "predictions": list(result["predictions"]),
            "metrics": result["metrics"],
            "boundary_metrics": boundary_metrics,
            "timings_ms": result["timings_ms"],
            "depth_range_m": [low, high],
            "methods": list(result["predictions"]),
            "warnings": result["warnings"],
        }
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        "edgeguide_checkpoint": resolve_edgeguide_checkpoint().is_file(),
        "depthunet_checkpoint": os.path.isfile(
            os.environ.get(
                "DEPTHUNET_CHECKPOINT",
                str(ROOT / "models" / "best_depth_model.pth"),
            )
        ),
    }


@app.get("/api/scenes")
def list_scenes(source: str = "presets"):
    try:
        scenes = _scene_catalog(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "source": source,
        "scenes": [
            {"key": key, "name": scene["name"]}
            for key, scene in scenes.items()
        ],
    }


@app.post("/api/reconstruct")
async def run_reconstruction(
    source: str = Form(...),
    scene_key: Optional[str] = Form(default=None),
    sensor_is_low_resolution: bool = Form(default=False),
    settings_json: str = Form(default="{}"),
    rgb_file: Optional[UploadFile] = File(default=None),
    depth_file: Optional[UploadFile] = File(default=None),
    reference_file: Optional[UploadFile] = File(default=None),
):
    try:
        settings = ReconstructionSettings.parse_obj(json.loads(settings_json))
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid reconstruction settings: {exc}") from exc

    try:
        if source in {"presets", "nyu"}:
            if sensor_is_low_resolution:
                raise ValueError("Preset RGB-D scenes are reference depth inputs.")
            if not scene_key:
                raise ValueError("Choose a scene before running reconstruction.")
            scenes = _scene_catalog(source)
            if scene_key not in scenes:
                raise ValueError("The selected scene is unavailable.")
            selected = scenes[scene_key]
            rgb = load_rgb_image(selected["rgb_path"])
            reference = load_depth_map(selected["depth_path"])
            input_depth = reference
            scene_name = selected["name"]
        elif source == "upload":
            if rgb_file is None or depth_file is None:
                raise ValueError("Upload both an RGB image and a depth map.")
            rgb_bytes = await rgb_file.read()
            depth_bytes = await depth_file.read()
            if len(rgb_bytes) > MAX_UPLOAD_BYTES or len(depth_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError("Each uploaded image must be smaller than 64 MB.")
            rgb = _read_rgb_upload(rgb_bytes, rgb_file.filename or "rgb")
            input_depth = _read_depth_upload(
                depth_bytes, depth_file.filename or "depth"
            )
            if reference_file is not None:
                reference_bytes = await reference_file.read()
                if len(reference_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("Reference depth must be smaller than 64 MB.")
                reference = _read_depth_upload(
                    reference_bytes, reference_file.filename or "reference"
                )
            elif sensor_is_low_resolution:
                reference = None
            else:
                reference = input_depth
            scene_name = "Custom RGB-D upload"
        else:
            raise ValueError("source must be 'nyu', 'presets', or 'upload'.")

        if not sensor_is_low_resolution and reference is not None and reference.shape != rgb.shape[:2]:
            raise ValueError(
                f"Reference depth shape {reference.shape} must match RGB shape {rgb.shape[:2]}."
            )

        result = reconstruct(
            rgb,
            input_depth,
            reference,
            settings=settings.dict(),
            sensor_is_low_resolution=sensor_is_low_resolution,
        )
    except (ValueError, FileNotFoundError, OSError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result_id = _store_result(result)
    return _prepare_response(result, result_id, scene_name)


@app.get("/api/results/{result_id}/depth/{method}")
def download_depth(result_id: str, method: str):
    result = _get_result(result_id)
    if method not in result["predictions"]:
        raise HTTPException(status_code=404, detail=f"Unknown reconstruction method: {method}")
    output = io.BytesIO()
    np.save(output, result["predictions"][method].astype(np.float32), allow_pickle=False)
    output.seek(0)
    filename = method.lower().replace(" ", "_").replace("-", "_") + "_depth.npy"
    return StreamingResponse(
        output,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/results/{result_id}/surface/{method}")
def get_surface(result_id: str, method: str):
    result = _get_result(result_id)
    if method not in result["predictions"]:
        raise HTTPException(status_code=404, detail=f"Unknown reconstruction method: {method}")
    depth = result["predictions"][method]
    rgb = result["rgb"]
    step = max(1, max(depth.shape) // 100)
    sampled_depth = depth[::step, ::step]
    sampled_rgb = rgb[::step, ::step]
    height, width = sampled_depth.shape
    source_height, source_width = depth.shape
    x, y = np.meshgrid(
        np.arange(0, source_width, step),
        np.arange(0, source_height, step),
    )
    return _finite_json(
        {
            "x": x.tolist(),
            "y": y.tolist(),
            "z": sampled_depth.tolist(),
            "rgb": np.clip(sampled_rgb * 255.0, 0, 255).astype(np.uint8).tolist(),
            "shape": [height, width],
            "step": step,
        }
    )


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:

    @app.get("/")
    def api_root():
        return {
            "name": "EdgeDepth API",
            "docs": "/docs",
            "frontend": "Run the React app with `npm run dev` in frontend/.",
        }

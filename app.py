"""Streamlit dashboard for EdgeDepth reconstruction and evaluation."""

import io
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.classical.discontinuity import detect_discontinuities
from src.metrics.evaluator import compute_boundary_f1
from src.services.reconstruction import reconstruct, resolve_edgeguide_checkpoint
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

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 640 * 480
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

st.set_page_config(
    page_title="EdgeDepth | Scientific workspace",
    page_icon=":material/center_focus_strong:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300, max_entries=4)
def scene_catalog(source: str) -> dict:
    if source == "presets":
        return get_available_presets()
    if source == "nyu":
        return get_full_nyu_v2_scenes(split="val", max_count=10000)
    raise ValueError(f"Unknown scene source: {source}")


def read_rgb_upload(data: bytes, filename: str) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Could not decode RGB image '{filename}'.") from exc

    with image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("RGB image must be PNG or JPEG.")
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError(
                "RGB image exceeds the 640×480 (307,200-pixel) processing limit. "
                "Resize the registered RGB and depth images to the same dimensions."
            )
        try:
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Could not decode RGB image '{filename}'.") from exc


def read_depth_upload(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy" or data.startswith(b"\x93NUMPY"):
        encoded = io.BytesIO(data)
        try:
            version = np.lib.format.read_magic(encoded)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(encoded)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = np.lib.format.read_array_header_2_0(encoded)
            else:
                raise ValueError(f"Unsupported NPY version {version}.")
        except (ValueError, OSError, EOFError) as exc:
            raise ValueError(f"Could not decode depth array '{filename}'.") from exc
        if dtype.kind not in "iuf":
            raise ValueError("Depth data must contain real numeric values.")
        if math.prod(shape) > MAX_IMAGE_PIXELS:
            raise ValueError(
                "Depth map exceeds the 640×480 (307,200-pixel) processing limit. "
                "Resize the registered RGB and depth images consistently."
            )
        try:
            encoded.seek(0)
            depth = np.load(encoded, allow_pickle=False)
        except (ValueError, OSError, EOFError) as exc:
            raise ValueError(f"Could not decode depth array '{filename}'.") from exc
    else:
        try:
            depth = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
        except cv2.error as exc:
            raise ValueError(f"Could not decode depth image '{filename}'.") from exc
        if depth is None:
            raise ValueError(f"Could not decode depth image '{filename}'.")
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0 * 5.0

    depth = np.asarray(depth)
    if depth.ndim > 2:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Depth data must be 2D; got shape {depth.shape}.")
    if depth.size > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Depth map exceeds the 640×480 (307,200-pixel) processing limit. "
            "Resize the registered RGB and depth images consistently."
        )
    if depth.dtype.kind not in "iuf":
        raise ValueError("Depth data must contain real numeric values.")
    try:
        depth = depth.astype(np.float32, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Could not convert depth data in '{filename}' to float32.") from exc
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)


def uploaded_bytes(uploaded_file, label: str) -> bytes:
    if uploaded_file is None:
        raise ValueError(f"Upload a {label}.")
    if uploaded_file.size > MAX_UPLOAD_BYTES:
        raise ValueError(f"{label} must be smaller than 64 MB.")
    return uploaded_file.getvalue()


def metric_rows(result: dict) -> pd.DataFrame:
    rows = []
    for method in result["predictions"]:
        metrics = result["metrics"].get(method, {})
        boundary = result["boundary_metrics"].get(method, {})
        rows.append(
            {
                "Method": method,
                "RMSE (m)": metrics.get("RMSE"),
                "MAE (m)": metrics.get("MAE"),
                "AbsRel": metrics.get("AbsRel"),
                "SSIM": metrics.get("SSIM"),
                "Edge RMSE (m)": metrics.get("Edge_RMSE"),
                "Boundary F1": boundary.get("f1_score"),
                "Latency (ms)": result["timings_ms"].get(method),
            }
        )
    return pd.DataFrame(rows)


def csv_bytes(result: dict) -> bytes:
    return metric_rows(result).to_csv(index=False).encode("utf-8")


def depth_preview(depth: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return colorize_depth(depth, vmin=limits[0], vmax=limits[1], colormap="turbo")


def display_image_pair(left_title: str, left_image, right_title: str, right_image):
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader(left_title)
            st.image(left_image, alt=left_title, width="stretch")
    with right:
        with st.container(border=True):
            st.subheader(right_title)
            st.image(right_image, alt=right_title, width="stretch")


def create_surface(depth: np.ndarray) -> go.Figure:
    step = max(1, max(depth.shape) // 110)
    sampled = depth[::step, ::step].astype(np.float32)
    sampled[sampled <= 1e-4] = np.nan
    height, width = sampled.shape
    x, y = np.meshgrid(np.arange(width) * step, np.arange(height) * step)
    finite = sampled[np.isfinite(sampled)]
    lower, upper = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)

    figure = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=sampled,
                colorscale="Turbo",
                cmin=float(lower),
                cmax=float(max(upper, lower + 1e-4)),
                colorbar={"title": "Depth (m)"},
                connectgaps=False,
                hovertemplate=(
                    "x %{x}<br>y %{y}<br>depth %{z:.3f} m<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c4ced2"},
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        scene={
            "xaxis_title": "Image x (px)",
            "yaxis_title": "Image y (px)",
            "zaxis_title": "Depth (m)",
            "aspectmode": "data",
            "bgcolor": "rgba(0,0,0,0)",
            "camera": {"eye": {"x": 1.65, "y": -1.65, "z": 1.1}},
        },
    )
    return figure


def run_input(source, scene_key, files, sensor_is_low_resolution, settings):
    if source in {"presets", "nyu"}:
        if sensor_is_low_resolution:
            raise ValueError("Curated RGB-D scenes are reference depth inputs.")
        scenes = scene_catalog(source)
        selected = scenes.get(scene_key)
        if selected is None:
            raise ValueError("The selected scene is unavailable.")
        rgb = load_rgb_image(selected["rgb_path"])
        reference = load_depth_map(selected["depth_path"])
        input_depth = reference
        scene_name = selected["name"]
    else:
        rgb_file, depth_file, reference_file = files
        rgb = read_rgb_upload(uploaded_bytes(rgb_file, "RGB image"), rgb_file.name)
        input_depth = read_depth_upload(
            uploaded_bytes(depth_file, "depth map"), depth_file.name
        )
        if reference_file is not None:
            reference = read_depth_upload(
                uploaded_bytes(reference_file, "reference depth map"),
                reference_file.name,
            )
        elif sensor_is_low_resolution:
            reference = None
        else:
            reference = input_depth
        scene_name = "Custom RGB-D upload"

    if input_depth.size > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Depth input exceeds the 640×480 (307,200-pixel) processing limit."
        )
    if (
        source == "upload"
        and not sensor_is_low_resolution
        and input_depth.shape != rgb.shape[:2]
    ):
        raise ValueError(
            "RGB and depth dimensions must match unless the depth input is marked "
            "sparse or low-resolution."
        )
    if reference is not None:
        if reference.shape != rgb.shape[:2]:
            raise ValueError(
                f"Reference depth shape {reference.shape} must match RGB shape "
                f"{rgb.shape[:2]}."
            )

    result = reconstruct(
        rgb,
        input_depth,
        reference,
        settings=settings,
        sensor_is_low_resolution=sensor_is_low_resolution,
    )
    result["boundary_metrics"] = {}
    if reference is not None:
        result["boundary_metrics"] = {
            method: compute_boundary_f1(
                prediction,
                reference,
                mask=result["valid_reference"],
            )
            for method, prediction in result["predictions"].items()
        }
    return {"data": result, "scene_name": scene_name}


def render_sidebar():
    with st.sidebar:
        st.title("EdgeDepth")
        st.caption("SCIENTIFIC DEPTH RECONSTRUCTION")
        page = st.radio(
            "Workspace",
            [
                "Overview",
                "Evaluation",
                "Boundary analysis",
                "3D inspection",
                "Model notes",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.subheader("Experiment")
        source_label = st.selectbox(
            "Input source",
            ["Curated RGB-D scenes", "NYU validation", "Upload custom RGB-D"],
        )
        source = {
            "Curated RGB-D scenes": "presets",
            "NYU validation": "nyu",
            "Upload custom RGB-D": "upload",
        }[source_label]

        scene_key = None
        if source != "upload":
            scenes = scene_catalog(source)
            if scenes:
                scene_keys = list(scenes)
                scene_key = st.selectbox(
                    "Scene",
                    scene_keys,
                    format_func=lambda key: scenes[key]["name"],
                )
            else:
                st.warning(
                    "No scenes were found. Curated examples are included; add the "
                    "NYU validation files to enable the full NYU list."
                )

        with st.form("reconstruction-form"):
            rgb_file = depth_file = reference_file = None
            sensor_is_low_resolution = False
            if source == "upload":
                rgb_file = st.file_uploader(
                    "Registered RGB image",
                    type=["png", "jpg", "jpeg"],
                    help=(
                        "Use an RGB image aligned with the supplied depth map. "
                        "Maximum 640×480 (307,200 pixels) and 64 MB."
                    ),
                )
                depth_file = st.file_uploader(
                    "Depth map",
                    type=["png", "tif", "tiff", "npy"],
                    help="16-bit PNG/TIFF values are interpreted as millimetres. "
                    "NumPy arrays are interpreted in metres. Maximum 640×480 "
                    "(307,200 pixels) and 64 MB.",
                )
                sensor_is_low_resolution = st.checkbox(
                    "Depth is sparse or low-resolution sensor input",
                    value=False,
                )
                reference_file = st.file_uploader(
                    "Optional registered reference depth",
                    type=["png", "tif", "tiff", "npy"],
                )

            scale = st.select_slider(
                "Sampling factor",
                options=[2, 4, 8],
                value=4,
                format_func=lambda value: f"{value}×",
            )
            with st.expander("Advanced settings"):
                filter_name = st.selectbox(
                    "Hybrid refinement",
                    ["guided", "bilateral"],
                    format_func=lambda value: (
                        "Guided filter" if value == "guided" else "Joint bilateral"
                    ),
                )
                noise_sigma = st.slider(
                    "Relative sensor noise", 0.0, 0.08, 0.0, 0.01
                )
                sparsity = st.slider("Additional sparsity", 0.0, 0.2, 0.0, 0.01)
                lidar_pattern = st.selectbox(
                    "Sampling pattern", ["grid", "random", "scanline"]
                )
                simulate_holes = st.checkbox("Simulate sensor holes", value=False)
                anti_texture = st.checkbox(
                    "Suppress texture copying", value=True
                )
                sharpen = st.slider(
                    "Boundary refinement", 0.0, 1.0, 0.25, 0.05
                )
                seed = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=42,
                    step=1,
                )

            submitted = st.form_submit_button(
                "Run reconstruction",
                type="primary",
                width="stretch",
                icon=":material/play_arrow:",
            )

        if submitted:
            settings = {
                "scale": scale,
                "noise_sigma": noise_sigma,
                "sparsity": sparsity,
                "lidar_pattern": lidar_pattern,
                "simulate_holes": simulate_holes,
                "filter": filter_name,
                "anti_texture": anti_texture,
                "sharpen": sharpen,
                "seed": int(seed),
            }
            try:
                with st.spinner("Running reconstruction models and evaluation..."):
                    st.session_state["result_bundle"] = run_input(
                        source,
                        scene_key,
                        (rgb_file, depth_file, reference_file),
                        sensor_is_low_resolution,
                        settings,
                    )
                st.session_state.pop("run_error", None)
            except (
                ValueError,
                TypeError,
                OverflowError,
                EOFError,
                FileNotFoundError,
                OSError,
                cv2.error,
                RuntimeError,
            ) as exc:
                st.session_state["run_error"] = str(exc)
                st.session_state.pop("result_bundle", None)
            except MemoryError:
                st.session_state["run_error"] = (
                    "The app ran out of memory while processing these inputs. "
                    "Resize the registered RGB and depth images to 640×480 or smaller."
                )
                st.session_state.pop("result_bundle", None)

        st.divider()
        device_name = "CUDA" if torch.cuda.is_available() else "CPU"
        st.caption(f"Inference device · {device_name}")
        if resolve_edgeguide_checkpoint().is_file():
            st.caption("EdgeGuideNet checkpoint · available")
        else:
            st.caption("EdgeGuideNet checkpoint · unavailable")

    return page


def render_overview(bundle, method, result, limits):
    scene_name = bundle["scene_name"]
    st.caption(f"SCIENTIFIC IMAGING / DEPTH RECONSTRUCTION / {scene_name.upper()}")
    st.title("Reconstruction overview")
    st.write(
        "Compare registered RGB-D inputs, actual model predictions, and "
        "reference-based measurements."
    )

    metric_data = result["metrics"]
    best_method = min(
        (
            (name, values.get("RMSE"))
            for name, values in metric_data.items()
            if np.isfinite(values.get("RMSE", np.nan))
        ),
        key=lambda item: item[1],
        default=(None, None),
    )[0]
    primary_method = "EdgeDepth hybrid" if "EdgeDepth hybrid" in result["predictions"] else method
    primary_boundary = result["boundary_metrics"].get(primary_method, {}).get("f1_score")
    with st.container(horizontal=True):
        st.metric(
            "Best RMSE",
            f"{metric_data[best_method]['RMSE']:.4f} m" if best_method else "—",
            border=True,
        )
        st.metric("Best method", best_method or "Reference required", border=True)
        st.metric(
            "Boundary F1",
            f"{primary_boundary:.3f}" if primary_boundary is not None else "—",
            border=True,
        )
        st.metric(
            "Input dimensions",
            f"{result['rgb'].shape[1]} × {result['rgb'].shape[0]}",
            border=True,
        )

    rgb_preview = np.clip(result["rgb"] * 255.0, 0, 255).astype(np.uint8)
    prediction = depth_preview(result["predictions"][method], limits)
    display_image_pair("RGB guidance", rgb_preview, f"{method} prediction", prediction)
    lower, upper = st.columns(2)
    with lower:
        with st.container(border=True):
            st.subheader("Sensor depth input")
            st.image(
                depth_preview(result["low_depth"], limits),
                alt="Colorized sensor depth input",
                width="stretch",
            )
            st.caption(
                f"{int(result['valid_reference'].sum()) if result['valid_reference'] is not None else int(np.count_nonzero(result['low_depth'] > 1e-4)):,} "
                "valid reference pixels"
                if result["valid_reference"] is not None
                else f"{int(np.count_nonzero(result['low_depth'] > 1e-4)):,} valid sensor samples"
            )
    with upper:
        with st.container(border=True):
            st.subheader("Reference depth")
            if result["reference"] is None:
                st.info(
                    "No registered reference was supplied. Quantitative metrics "
                    "and error maps are intentionally omitted."
                )
            else:
                st.image(
                    depth_preview(result["reference"], limits),
                    alt="Colorized reference depth map",
                    width="stretch",
                )
    if result["reference"] is not None:
        st.subheader("Absolute depth error")
        error_map = compute_error_heatmap(
            result["predictions"][method],
            result["reference"],
            mask=result["valid_reference"],
        )
        st.image(error_map, alt=f"Absolute error map for {method}", width="stretch")

    st.subheader("Method outputs")
    with st.container(horizontal=True, horizontal_alignment="left"):
        for output_method in result["predictions"]:
            st.download_button(
                f"Export {output_method}",
                data=npy_bytes(result["predictions"][output_method]),
                file_name=f"{output_method.lower().replace(' ', '_')}_depth.npy",
                mime="application/octet-stream",
                key=f"export-{output_method}",
            )


def npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, np.asarray(array, dtype=np.float32), allow_pickle=False)
    return output.getvalue()


def render_evaluation(result):
    st.caption("QUANTITATIVE EVALUATION")
    st.title("Method performance")
    if not result["metrics"]:
        st.info(
            "Metrics require a registered reference depth map. Upload one or choose "
            "a curated RGB-D scene to enable evaluation."
        )
        return
    rows = metric_rows(result)
    chart_data = rows.set_index("Method")[["RMSE (m)", "MAE (m)", "Edge RMSE (m)"]]
    st.bar_chart(
        chart_data,
        x_label="Reconstruction method",
        y_label="Error (m)",
        color=["#5bc8c2", "#7798a2", "#c89d68"],
        sort="RMSE (m)",
        alt="RMSE, MAE, and edge RMSE by reconstruction method",
    )
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "RMSE (m)": st.column_config.NumberColumn(format="%.4f"),
            "MAE (m)": st.column_config.NumberColumn(format="%.4f"),
            "AbsRel": st.column_config.NumberColumn(format="%.4f"),
            "SSIM": st.column_config.NumberColumn(format="%.4f"),
            "Edge RMSE (m)": st.column_config.NumberColumn(format="%.4f"),
            "Boundary F1": st.column_config.NumberColumn(format="%.3f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
        alt="Depth reconstruction metrics by method",
    )
    st.download_button(
        "Download evaluation CSV",
        data=csv_bytes(result),
        file_name="edgedepth_metrics.csv",
        mime="text/csv",
    )
    st.caption(
        "Metrics use finite, positive reference-depth pixels. Interpret results for "
        "this scene and sampling configuration; rankings are not universal."
    )


def render_boundaries(result, method, limits):
    st.caption("GEOMETRY REVIEW")
    st.title("Boundary analysis")
    prediction = result["predictions"][method]
    prediction_edges = detect_discontinuities(prediction)["boundary_mask"]
    edge_overlay = overlay_edges_on_image(
        result["rgb"], prediction_edges, color=(73, 200, 192)
    )
    image_pairs = [
        ("RGB with predicted boundaries", edge_overlay),
        ("Colorized prediction", depth_preview(prediction, limits)),
    ]
    if result["affinity"] is not None:
        image_pairs.append(
            ("Geometric affinity", depth_preview(result["affinity"], (0.0, 1.0)))
        )
    if result["reference"] is not None:
        reference_edges = detect_discontinuities(result["reference"])["boundary_mask"]
        image_pairs.append(
            (
                "Reference boundaries",
                overlay_edges_on_image(
                    result["rgb"], reference_edges, color=(255, 184, 77)
                ),
            )
        )
    for index in range(0, len(image_pairs), 2):
        left, right = st.columns(2)
        for column, (title, image) in zip(
            (left, right), image_pairs[index : index + 2]
        ):
            with column:
                with st.container(border=True):
                    st.subheader(title)
                    st.image(image, alt=title, width="stretch")
    score = result["boundary_metrics"].get(method)
    if score is not None:
        st.subheader(f"{method} boundary score")
        cols = st.columns(3)
        cols[0].metric("Precision", f"{score['precision']:.3f}", border=True)
        cols[1].metric("Recall", f"{score['recall']:.3f}", border=True)
        cols[2].metric("F1 score", f"{score['f1_score']:.3f}", border=True)
    elif result["reference"] is None:
        st.info("Boundary scores require registered reference depth.")


def render_surface(result, method):
    st.caption("SPATIAL INSPECTION")
    st.title("Interactive depth surface")
    st.write("Drag to rotate, scroll to zoom, and hover for pixel-level depth values.")
    st.plotly_chart(
        create_surface(result["predictions"][method]),
        width="stretch",
        config={"displaylogo": False, "responsive": True},
    )


def render_model_notes(result):
    st.caption("MODEL AND RECONSTRUCTION REFERENCE")
    st.title("Model notes")
    st.write(
        "EdgeDepth compares interpolation baselines, RGB-guided filtering, two "
        "separate learned models, and a hybrid refinement. Model outputs are not "
        "assumed to be accurate without reference-based evaluation."
    )
    notes = [
        (
            "Classical baselines",
            "Nearest, bilinear, and bicubic interpolation, plus joint bilateral and "
            "guided filtering.",
        ),
        (
            "EdgeGuideNet",
            "Residual RGB-guided CNN loaded from models/edge_guide_net.pt. It is "
            "separate from DepthUNet.",
        ),
        (
            "DepthUNet",
            "Five-channel RGB, sparse depth, and validity-mask input, loaded strictly "
            "from models/best_depth_model.pth.",
        ),
        (
            "EdgeDepth hybrid",
            "Combines EdgeGuideNet predictions, guided or joint bilateral filtering, "
            "and optional anti-texture refinement.",
        ),
        (
            "Evaluation",
            "RMSE, MAE, AbsRel, delta accuracy, PSNR, SSIM, edge metrics, boundary "
            "precision/recall/F1, and per-method latency.",
        ),
    ]
    for heading, description in notes:
        with st.container(border=True):
            st.subheader(heading)
            st.write(description)
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)


def main():
    page = render_sidebar()
    if error := st.session_state.get("run_error"):
        st.error(error, title="Reconstruction failed")

    bundle = st.session_state.get("result_bundle")
    if bundle is None:
        st.title("EdgeDepth")
        st.subheader("Edge-preserving depth reconstruction and evaluation")
        st.write(
            "Select a curated RGB-D scene or upload registered RGB and depth inputs "
            "in the sidebar, then run reconstruction to compare model outputs."
        )
        with st.container(horizontal=True):
            st.metric("Reconstruction methods", "8", border=True)
            st.metric("Learned models", "2 independent branches", border=True)
            st.metric("Sampling factors", "2× · 4× · 8×", border=True)
        st.info(
            "RGB-only input cannot be evaluated by this depth upsampling workflow. "
            "Provide an aligned sparse or low-resolution depth map; a reference depth "
            "map is needed for quantitative metrics."
        )
        return

    result = bundle["data"]
    available_methods = list(result["predictions"])
    default_method = (
        "EdgeDepth hybrid"
        if "EdgeDepth hybrid" in available_methods
        else available_methods[0]
    )
    method = st.selectbox(
        "Method to inspect",
        available_methods,
        index=available_methods.index(default_method),
    )
    display_source = result["reference"]
    if display_source is None:
        display_source = result["low_depth"]
    valid_source = display_source[
        np.isfinite(display_source) & (display_source > 1e-4)
    ]
    if valid_source.size:
        low, high = (float(value) for value in np.percentile(valid_source, [1, 99]))
        high = max(high, low + 1e-4)
    else:
        low, high = 0.0, 1.0
    limits = (low, high)

    if page == "Overview":
        render_overview(bundle, method, result, limits)
    elif page == "Evaluation":
        render_evaluation(result)
    elif page == "Boundary analysis":
        render_boundaries(result, method, limits)
    elif page == "3D inspection":
        render_surface(result, method)
    else:
        render_model_notes(result)


if __name__ == "__main__":
    main()

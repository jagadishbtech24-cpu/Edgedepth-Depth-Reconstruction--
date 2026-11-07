import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app import (
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    read_depth_upload,
    read_rgb_upload,
    run_input,
    uploaded_bytes,
)


def encode_npy(array):
    encoded = io.BytesIO()
    np.save(encoded, array, allow_pickle=False)
    return encoded.getvalue()


def encode_npy_header(shape, dtype=np.float32):
    encoded = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        encoded,
        {
            "descr": np.dtype(dtype).str,
            "fortran_order": False,
            "shape": shape,
        },
    )
    return encoded.getvalue()


def encode_rgb(size=(64, 48)):
    encoded = io.BytesIO()
    Image.new("RGB", size, color=(30, 80, 120)).save(encoded, format="PNG")
    return encoded.getvalue()


def test_depth_upload_detects_numpy_content_without_filename_extension():
    expected = np.array([[0.0, 1.25], [2.5, 3.75]], dtype=np.float32)

    actual = read_depth_upload(encode_npy(expected), "blob")

    np.testing.assert_array_equal(actual, expected)


def test_depth_upload_preserves_single_row_dimensions():
    actual = read_depth_upload(encode_npy(np.array([[1.0, 2.0]])), "row.npy")

    assert actual.shape == (1, 2)


def test_rgb_upload_decodes_png_to_rgb_float32():
    actual = read_rgb_upload(encode_rgb(), "scene.png")

    assert actual.shape == (48, 64, 3)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual[0, 0], [30 / 255, 80 / 255, 120 / 255])


def test_rgb_upload_rejects_corrupt_image_with_useful_error():
    with pytest.raises(ValueError, match="Could not decode RGB image 'broken.png'"):
        read_rgb_upload(b"not an image", "broken.png")


def test_rgb_upload_rejects_image_above_cloud_processing_limit():
    data = encode_rgb(size=(MAX_IMAGE_PIXELS + 1, 1))

    with pytest.raises(ValueError, match="640×480"):
        read_rgb_upload(data, "oversized.png")


def test_depth_upload_rejects_non_2d_arrays():
    with pytest.raises(ValueError, match="Depth data must be 2D"):
        read_depth_upload(encode_npy(np.ones((2, 2, 2), dtype=np.float32)), "bad.npy")


def test_depth_upload_rejects_corrupt_image():
    with pytest.raises(ValueError, match="Could not decode depth image 'bad.png'"):
        read_depth_upload(b"not an image", "bad.png")


def test_depth_upload_rejects_non_numeric_arrays():
    with pytest.raises(ValueError, match="real numeric values"):
        read_depth_upload(
            encode_npy(np.full((2, 2), "near", dtype="<U4")), "bad.npy"
        )


def test_depth_upload_rejects_array_above_cloud_processing_limit():
    data = encode_npy(np.ones((1, MAX_IMAGE_PIXELS + 1), dtype=np.float32))

    with pytest.raises(ValueError, match="640×480"):
        read_depth_upload(data, "oversized.npy")


def test_depth_upload_rejects_oversized_npy_header_before_array_allocation():
    data = encode_npy_header((MAX_IMAGE_PIXELS + 1, 1))

    with pytest.raises(ValueError, match="640×480"):
        read_depth_upload(data, "oversized.npy")


def test_custom_upload_rejects_mismatched_registered_depth_dimensions():
    rgb_data = encode_rgb()
    depth_data = encode_npy(np.ones((24, 32), dtype=np.float32))
    rgb_file = SimpleNamespace(name="scene.png", size=len(rgb_data), getvalue=lambda: rgb_data)
    depth_file = SimpleNamespace(name="depth.npy", size=len(depth_data), getvalue=lambda: depth_data)

    with patch("app.reconstruct") as reconstruction:
        with pytest.raises(ValueError, match="RGB and depth dimensions must match"):
            run_input(
                "upload",
                None,
                (rgb_file, depth_file, None),
                sensor_is_low_resolution=False,
                settings={"scale": 4},
            )

    reconstruction.assert_not_called()


def test_uploaded_bytes_rejects_files_above_upload_limit():
    too_large = SimpleNamespace(size=MAX_UPLOAD_BYTES + 1)

    with pytest.raises(ValueError, match="smaller than 64 MB"):
        uploaded_bytes(too_large, "RGB image")

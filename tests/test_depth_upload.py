import io

import numpy as np

from api.main import _read_depth_upload


def test_depth_upload_detects_numpy_content_without_filename_extension():
    expected = np.array([[0.0, 1.25], [2.5, 3.75]], dtype=np.float32)
    encoded = io.BytesIO()
    np.save(encoded, expected, allow_pickle=False)

    actual = _read_depth_upload(encoded.getvalue(), "blob")

    np.testing.assert_array_equal(actual, expected)

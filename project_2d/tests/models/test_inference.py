import numpy as np
import pytest
from unittest.mock import Mock
from PIL import Image
from io import BytesIO
from src.npecage.models.inference import predict_mask, remove_padding, extend_mask_to_original_size, make_overlay


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture
def dummy_model():
    # Simulate 4x4 = 16 patches, each producing a 16x16 prediction
    return Mock(predict=Mock(return_value=np.random.rand(16, 16, 16, 1)))


def test_predict_mask_shape(dummy_image, dummy_model):
    patch_size = 16
    mask = predict_mask(dummy_model, dummy_image, patch_size)
    assert mask.shape == dummy_image.shape


def test_remove_padding():
    mask = np.ones((100, 100))
    padding_info = {
        "pad_top": 10, "pad_bottom": 10,
        "pad_left": 20, "pad_right": 20
    }
    cropped = remove_padding(mask, padding_info)
    assert cropped.shape == (80, 60), "Padding not removed correctly"


def test_extend_mask_to_original_size():
    cropped = np.ones((80, 60))
    cropping_info = {
        "crop_top": 10, "crop_bottom": 10,
        "crop_left": 20, "crop_right": 20
    }
    extended = extend_mask_to_original_size(cropped, cropping_info)
    assert extended.shape == (100, 100), "Extended size doesn't match original dimensions"
    assert np.all(extended[10:90, 20:80] == 1), "Cropped mask not placed correctly"


def _png_bytes_from_array(arr):
    """Helper: convert a 2-D uint8 array to PNG bytes (mode='L')."""
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_make_overlay_basic():
    # Arrange
    h, w = 100, 100
    original = Image.new("RGB", (w, h), color="white")

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 30:50] = 255                    # a solid rectangle mask
    mask_png = _png_bytes_from_array(mask)

    tip_coords = [(10, 10), (80, 70), "No Roots Detected"]

    # Act
    overlay = make_overlay(original, mask_png, tip_coords)

    # Assert
    # basic properties
    assert overlay.size == original.size               # same (width, height)
    assert overlay.mode == "RGBA"

    arr = np.array(overlay)                            # shape (H, W, 4)

    # mask really tinted the intended area (greenish) and left others white
    outside_pixel = arr[0, 0, :3]                      # pure white expected
    inside_pixel = arr[25, 35, :3]                     # must differ from white
    assert np.array_equal(outside_pixel, [255, 255, 255]), "Pixel outside mask changed unexpectedly"
    assert not np.array_equal(inside_pixel, [255, 255, 255]), "Mask overlay had no visible effect"
    assert inside_pixel[1] >= inside_pixel[0] and inside_pixel[1] >= inside_pixel[2], \
        "Masked pixel is not predominantly green"

    # red tip markers were drawn (pure red pixels exist)
    red_idxs = np.where(
        (arr[:, :, 0] == 255) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
    )
    assert red_idxs[0].size > 0, "No red tip markers found in the overlay"

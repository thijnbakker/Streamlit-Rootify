import pytest
import numpy as np
import cv2
from src.npecage.data.data_preprocessing import crop_image, padder


@pytest.fixture
def sample_image():
    """
    Create a synthetic image with a white square in the center on black background.
    """
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
    return image


def test_crop_image_finds_largest_contour(sample_image):
    cropped, info = crop_image(sample_image)

    # Check that output is grayscale
    assert len(cropped.shape) == 2

    # Check cropping margins are correct
    assert isinstance(info, dict)
    assert all(key in info for key in ["crop_top", "crop_bottom", "crop_left", "crop_right"])

    # The crop should be centered and square
    assert cropped.shape[0] == cropped.shape[1]


def test_crop_image_handles_no_contours():
    # Completely black image (no contours)
    blank_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cropped, info = crop_image(blank_image)

    # Expecting full image back in grayscale and no cropping
    assert cropped.shape == (100, 100)
    assert info == {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}


def test_padder_adds_correct_padding():
    image = np.ones((103, 205), dtype=np.uint8) * 255  # grayscale image
    patch_size = 32
    padded, info = padder(image, patch_size)

    # New dimensions must be divisible by patch_size
    assert padded.shape[0] % patch_size == 0
    assert padded.shape[1] % patch_size == 0

    # Padding info should match the difference
    total_pad_h = padded.shape[0] - image.shape[0]
    total_pad_w = padded.shape[1] - image.shape[1]

    assert total_pad_h == info["pad_top"] + info["pad_bottom"]
    assert total_pad_w == info["pad_left"] + info["pad_right"]


def test_padder_no_padding_needed():
    image = np.ones((64, 64), dtype=np.uint8)
    patch_size = 32
    padded, info = padder(image, patch_size)

    # Should remain unchanged
    assert padded.shape == image.shape
    assert info == {"pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}

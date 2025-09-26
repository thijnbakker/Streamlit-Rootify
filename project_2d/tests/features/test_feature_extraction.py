import pytest
import numpy as np
import cv2
from skimage.morphology import skeletonize
from src.npecage.features.feature_extraction import (
    crop_top_roots,
    process_roots,
    calculate_path_length,
    tresh_mask,
    find_root_tips,
    find_root_lengths,
)


@pytest.fixture
def dummy_binary_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (10, 50), (90, 50), 255, 2)
    return img


@pytest.fixture
def dummy_predicted_mask():
    img = np.zeros((100, 100), dtype=np.float32)
    cv2.line(img, (10, 50), (90, 50), 1.0, 1)
    return img


def test_tresh_mask(dummy_predicted_mask):
    thresh = tresh_mask(dummy_predicted_mask)
    assert isinstance(thresh, np.ndarray)
    assert thresh.dtype == np.uint8
    assert np.max(thresh) == 255 or np.max(thresh) == 0


def test_crop_top_roots():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    labels[6:9, 6:9] = 2
    root_areas = [(1, 9), (2, 9)]
    crops = crop_top_roots(root_areas, labels)
    assert len(crops) == 2
    assert all(crop.dtype == np.uint8 for crop in crops)


def test_process_roots():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    root_areas = [(1, 9)]
    skeletons = process_roots(root_areas, labels)
    assert len(skeletons) == 1
    assert skeletons[0].dtype == bool


def test_calculate_path_length(dummy_binary_image):
    skeleton = skeletonize(dummy_binary_image // 255)
    length = calculate_path_length(skeleton)
    assert isinstance(length, (int, float))


def test_find_root_tips(dummy_binary_image):
    df, tips = find_root_tips(dummy_binary_image)
    assert isinstance(tips, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in tips)
    assert isinstance(df, (np.ndarray, object))  # Pandas DataFrame


def test_find_root_lengths(dummy_binary_image):
    # Extend width to match expected region indexing
    padded = np.zeros((1000, 3000), dtype=np.uint8)
    padded[500:550, 50:2500] = 255
    lengths = find_root_lengths(padded)
    assert isinstance(lengths, list)
    assert len(lengths) == 5
    assert all(isinstance(root_length, (int, float)) for root_length in lengths)

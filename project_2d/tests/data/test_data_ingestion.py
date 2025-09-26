import pytest
import numpy as np
import cv2
from src.npecage.data.data_ingestion import read_image


@pytest.fixture
def test_image_path(tmp_path):
    # Create a temporary black image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    file_path = tmp_path / "../../test_image.png"
    cv2.imwrite(str(file_path), image)
    return str(file_path), image


def test_read_valid_image(test_image_path):
    image_path, original_image = test_image_path
    loaded_image = read_image(image_path)
    assert loaded_image is not None
    assert loaded_image.shape == original_image.shape


def test_read_invalid_image_path():
    with pytest.raises(ValueError, match="could not be loaded"):
        read_image("non_existent_image.jpg")

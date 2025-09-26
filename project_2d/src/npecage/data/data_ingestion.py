import cv2
import logging

logger = logging.getLogger(__name__)


def read_image(image_path):
    """
    Reads an image from the specified file path using OpenCV.

    Args:
        image_path (str): The path to the image file to be loaded.

    Raises:
        ValueError: If the image cannot be loaded from the given path.

    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    logger.info(f"Reading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Image at {image_path} could not be loaded.")
        raise ValueError(f"Image at {image_path} could not be loaded.")
    logger.info(f"Image loaded successfully from {image_path}.")
    return image

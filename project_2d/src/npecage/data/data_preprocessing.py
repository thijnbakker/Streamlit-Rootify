import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def crop_image(image):
    """
    Crop the input image around the largest contour detected after edge detection.

    Steps:
    1. Convert the input image to grayscale and apply Gaussian blur.
    2. Detect edges using Canny edge detection with adaptive thresholds based on mean intensity.
    3. Use morphological closing to close gaps in the edges.
    4. Find contours and select the largest one.
    5. Compute a square bounding box centered on the largest contour.
    6. Crop the grayscale image using this bounding box.
    7. Return the cropped image and cropping margins.

    Args:
        image (numpy.ndarray): Input BGR image.

    Returns:
        cropped_image (numpy.ndarray): Grayscale cropped image centered on largest contour.
        cropping_info (dict): Dictionary with cropping margins relative to original image:
            {
                "crop_top": int,
                "crop_bottom": int,
                "crop_left": int,
                "crop_right": int
            }
    """

    # Step 1: Read and preprocess the image
    logger.info("Preprocessing the image...")
    logger.debug(f"Original image shape: {image.shape}")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.info("Converted image to grayscale.")
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    logger.info("Applied Gaussian blur to the image.")

    # Step 2: Detect edges using Canny edge detection
    mean_intensity = np.mean(blurred_image)
    threshold1 = mean_intensity * 0.15
    threshold2 = mean_intensity * 0.7
    edges = cv2.Canny(blurred_image, threshold1, threshold2)
    logger.info("Detected edges using Canny edge detection.")

    # Step 3: Close gaps in the edges using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    logger.info("Applied morphological closing to the edges.")

    # Step 4: Find the largest contour and compute its bounding box
    contours, _ = cv2.findContours(closed_edges, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found in the image.")
        return gray_image, {"crop_top": 0, "crop_bottom": 0, "crop_left": 0, "crop_right": 0}
    logger.debug(f"Found {len(contours)} contours in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    logger.debug(f"Size of the largest contour: {cv2.contourArea(largest_contour)}")
    x, y, w, h = cv2.boundingRect(largest_contour)
    side_length = max(w, h)
    x_center, y_center = x + w // 2, y + h // 2
    logger.debug(f"Bounding box of the largest contour: x={x}, y={y}, w={w}, h={h}")

    # Step 5: Crop the image around the largest contour
    x_start = max(x_center - side_length // 2, 0)
    y_start = max(y_center - side_length // 2, 0)
    x_end = min(x_start + side_length, gray_image.shape[1])
    y_end = min(y_start + side_length, gray_image.shape[0])

    cropped_image = gray_image[y_start:y_end, x_start:x_end]
    logger.info("Cropped the image around the largest contour.")
    logger.debug(f"Cropped image shape: {cropped_image.shape}")

    # Step 6: Save the cropping info
    cropping_info = {
        "crop_top": y_start,
        "crop_bottom": gray_image.shape[0] - y_end,
        "crop_left": x_start,
        "crop_right": gray_image.shape[1] - x_end
    }
    logger.debug(f"Cropping info: {cropping_info}")

    return cropped_image, cropping_info


def padder(image, patch_size):
    """
    Pad the input image so that its height and width are divisible by patch_size.

    Padding is distributed evenly on all sides and applied with black pixels.

    Args:
        image (numpy.ndarray): Input image (grayscale or color).
        patch_size (int): The size that image dimensions should be divisible by.

    Returns:
        padded_image (numpy.ndarray): The padded image.
        padding_info (dict): Dictionary with padding amounts applied on each side:
            {
                "pad_top": int,
                "pad_bottom": int,
                "pad_left": int,
                "pad_right": int
            }
    """

    logger.info("Padding the image...")
    logger.debug(f"Original image shape: {image.shape}")

    # Extract the image dimensions
    h, w = image.shape[:2]
    logger.debug(f"Image dimensions: height={h}, width={w}")

    # Calculate padding to make dimensions divisible by patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    logger.debug(f"Padding required: height={pad_h}, width={pad_w}")

    # Divide padding between top/bottom and left/right
    top_padding = pad_h // 2
    bottom_padding = pad_h - top_padding
    left_padding = pad_w // 2
    right_padding = pad_w - left_padding
    logger.debug(f"Padding distribution: top={top_padding}, bottom={bottom_padding}, left={left_padding}, right={right_padding}")

    # Apply padding using OpenCV
    padded_image = cv2.copyMakeBorder(
        image,
        top_padding, bottom_padding,
        left_padding, right_padding,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]  # Black padding
    )
    logger.info("Applied padding to the image.")

    # Store padding information in a dictionary
    padding_info = {
        "pad_top": top_padding,
        "pad_bottom": bottom_padding,
        "pad_left": left_padding,
        "pad_right": right_padding,
    }

    return padded_image, padding_info

import numpy as np
import cv2
import logging
from patchify import patchify, unpatchify
from PIL import Image, ImageDraw
from io import BytesIO


def predict_mask(model, petri_dish, patch_size):
    """
    Predict a mask for the given petri dish image by dividing it into patches,
    converting patches to grayscale, and running model inference patch-wise.

    Args:
        model (tensorflow.keras.Model): Trained Keras model for mask prediction.
        petri_dish (numpy.ndarray): Grayscale input image of the petri dish.
        patch_size (int): Size of square patches to split the image into.

    Returns:
        numpy.ndarray: Reconstructed predicted mask from patch predictions,
                       with shape corresponding to the padded petri dish image.
    """

    logging.info("Starting mask prediction...")
    logging.debug(f"Input image shape: {petri_dish.shape}")
    # Convert grayscale to 3-channel image
    petri_dish_padded_3d = np.expand_dims(petri_dish, axis=-1)
    logging.debug(f"3D image shape after expanding dimensions: {petri_dish_padded_3d.shape}")
    petri_dish_padded_3d = np.repeat(petri_dish_padded_3d, 3, axis=-1)
    logging.debug(f"3D image shape after repeating channels: {petri_dish_padded_3d.shape}")
    logging.info("Converted grayscale image to 3-channel format.")

    # Create patches
    patches = patchify(petri_dish_padded_3d, (patch_size, patch_size, 3), step=patch_size)
    i, j = patches.shape[0], patches.shape[1]
    logging.info("Created patches from the image.")
    logging.debug(f"Number of patches: {i * j}, Patch shape: {patches.shape}")

    # Convert each RGB patch to grayscale
    patches_gray = np.array([
        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)  # Use BGR unless confirmed RGB
        for patch in patches.reshape(-1, patch_size, patch_size, 3)
    ])
    logging.info("Converted patches to grayscale.")
    patches_gray = patches_gray.reshape(-1, patch_size, patch_size, 1)
    logging.debug(f"Grayscale patches shape: {patches_gray.shape}")

    # Normalize and predict
    prediction = model.predict(patches_gray / 255, batch_size=16)
    logging.info("Model prediction completed.")

    # Reshape and reconstruct full image
    prediction = prediction.reshape(i, j, patch_size, patch_size)
    predicted_mask = unpatchify(prediction, (petri_dish_padded_3d.shape[0], petri_dish_padded_3d.shape[1]))
    logging.info("Reconstructed the predicted mask from patches.")

    return predicted_mask


def remove_padding(predicted_mask, padding_info):
    """
    Remove padding from a predicted mask using provided padding information.

    Args:
        predicted_mask (numpy.ndarray): Mask predicted on padded image.
        padding_info (dict): Dictionary containing padding sizes with keys:
            "pad_top", "pad_bottom", "pad_left", "pad_right".

    Returns:
        numpy.ndarray: Mask with padding removed, restoring original dimensions.
    """

    logging.info("Removing padding from the predicted mask...")
    # Remove padding
    cropped_mask = predicted_mask[
        padding_info["pad_top"] : -padding_info["pad_bottom"],
        padding_info["pad_left"] : -padding_info["pad_right"],
    ]
    logging.info("Padding removed successfully.")
    logging.debug(f"Cropped mask shape: {cropped_mask.shape}")

    return cropped_mask


def extend_mask_to_original_size(cropped_mask, cropping_info):
    """
    Extend a cropped mask back to the original image size by placing it inside a
    zero-padded array according to cropping margins.

    Args:
        cropped_mask (numpy.ndarray): Cropped mask image.
        cropping_info (dict): Dictionary with cropping margins with keys:
            "crop_top", "crop_bottom", "crop_left", "crop_right".

    Returns:
        numpy.ndarray: Extended mask with the same size as the original image,
                       where cropped_mask is placed at the proper location.
    """
    logging.info("Extending the mask to original size...")

    original_h = cropping_info["crop_top"] + cropped_mask.shape[0] + cropping_info["crop_bottom"]
    original_w = cropping_info["crop_left"] + cropped_mask.shape[1] + cropping_info["crop_right"]
    logging.debug(f"Original mask size: height={original_h}, width={original_w}")

    extended_mask = np.zeros((original_h, original_w), dtype=cropped_mask.dtype)
    logging.debug(f"Extended mask shape: {extended_mask.shape}")
    extended_mask[
        cropping_info["crop_top"] : cropping_info["crop_top"] + cropped_mask.shape[0],
        cropping_info["crop_left"] : cropping_info["crop_left"] + cropped_mask.shape[1],
    ] = cropped_mask
    logging.info("Mask extended to original size successfully.")
    logging.debug(f"Final extended mask shape: {extended_mask.shape}")

    return extended_mask


def make_overlay(
    original: Image.Image,
    mask_png: bytes,
    tip_coords,
    mask_alpha: int = 100,
    tip_radius: int = 8,
):
    """
    Returns a Pillow RGBA image showing:
    • the original photo
    • a translucent green mask
    • red circles on every (y, x) coordinate in `tip_coords`
    """
    base = original.convert("RGBA")

    # 1) translucent green version of mask
    mask_img = Image.open(BytesIO(mask_png)).convert("L")
    mask_arr = np.array(mask_img)
    # green overlay (R,G,B,A)
    green = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    green[..., 1] = 255                      # G channel
    green[..., 3] = (mask_arr > 0) * mask_alpha
    green_rgba = Image.fromarray(green, mode="RGBA")
    base = Image.alpha_composite(base, green_rgba)

    # 2) red tip markers
    draw = ImageDraw.Draw(base)
    for tip in tip_coords:
        if tip == "No Roots Detected":
            continue
        y, x = tip  # (row, col)
        draw.ellipse(
            [(x - tip_radius, y - tip_radius),
             (x + tip_radius, y + tip_radius)],
            outline=(255, 0, 0, 255),
            width=3,
        )
    return base

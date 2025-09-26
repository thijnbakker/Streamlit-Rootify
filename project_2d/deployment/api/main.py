import sys
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from src.npecage.data.data_preprocessing import crop_image, padder
from src.npecage.models.inference import predict_mask, remove_padding, extend_mask_to_original_size, make_overlay
from src.npecage.utils.helpers import f1
from src.npecage.features.feature_extraction import tresh_mask, find_root_tips, find_root_lengths
import cv2
import numpy as np
import io
from tensorflow.keras.models import load_model

from PIL import Image
from io import BytesIO

app = FastAPI()

# Load model once during app startup
# For local training use the path "../models/model-best-2.h5"


MODEL_PATH = "models/model-best-thijn.h5"
model = load_model(MODEL_PATH, custom_objects={"f1": f1})


@app.post("/process-image/")
async def process_image(file: UploadFile = File(...), patch_size: int = 256):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        cropped_image, crop_info = crop_image(image)
        padded_image, pad_info = padder(cropped_image, patch_size)

        _, buffer = cv2.imencode(".png", padded_image)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/predict-mask/")
async def predict_mask_api(file: UploadFile = File(...), patch_size: int = 256):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    try:
        # Step 1: Crop and pad image
        cropped_image, crop_info = crop_image(image)
        padded_image, pad_info = padder(cropped_image, patch_size)

        # Step 2: Predict mask
        predicted = predict_mask(model, padded_image, patch_size)

        # Step 3: Remove padding & extend back to original size
        unpadded_mask = remove_padding(predicted, pad_info)
        full_size_mask = extend_mask_to_original_size(unpadded_mask, crop_info)

        # Step 4: Encode and return mask as image
        mask_uint8 = (full_size_mask * 255).astype(np.uint8)
        _, buffer = cv2.imencode(".png", mask_uint8)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/extract-root-tips/")
async def extract_root_tips(
    file: UploadFile = File(...),
    patch_size: int = 256
):
    try:
        # Read input image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        # Step 1: Crop and pad image
        cropped_image, crop_info = crop_image(image)
        padded_image, pad_info = padder(cropped_image, patch_size)

        # Step 2: Predict mask
        predicted = predict_mask(model, padded_image, patch_size)

        # Step 3: Remove padding & extend back to original size
        unpadded_mask = remove_padding(predicted, pad_info)
        full_size_mask = extend_mask_to_original_size(unpadded_mask, crop_info)

        # Step 4: Convert mask to uint8 for thresholding
        mask_uint8 = (full_size_mask * 255).astype(np.uint8)

        # Step 5: Threshold and extract tips
        thresholded = tresh_mask(mask_uint8)
        _, tip_coordinates = find_root_tips(thresholded)

        # Return the coordinates as JSON
        return JSONResponse(content={"tip_coordinates": tip_coordinates})

    except Exception as e:
        print(f"Error in /extract-root-tips/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract root tips: {str(e)}")


@app.post("/calculate-root-lengths/")
async def calculate_root_lengths(file: UploadFile = File(...), patch_size: int = 256):
    try:
        # Step 1: Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        # Step 2: Crop and pad
        cropped_image, crop_info = crop_image(image)
        padded_image, pad_info = padder(cropped_image, patch_size)

        # Step 3: Predict mask
        predicted = predict_mask(model, padded_image, patch_size)

        # Step 4: Remove padding and restore original image size
        unpadded_mask = remove_padding(predicted, pad_info)

        # Step 5: Threshold the mask
        mask_uint8 = (unpadded_mask * 255).astype(np.uint8)
        thresholded = tresh_mask(mask_uint8)

        # Step 6: Calculate root lengths
        root_lengths = find_root_lengths(thresholded)

        # Return the results
        return JSONResponse(content={"root_lengths": root_lengths})

    except Exception as e:
        print(f"Error in /calculate-root-lengths/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate root lengths: {str(e)}")


@app.post("/mask-overlay/")
async def mask_overlay(file: UploadFile = File(...), patch_size: int = 256):
    """
    Returns a PNG with the predicted root mask (semi-transparent green)
    and red circles on the detected root tips, over the original photograph.
    """
    try:
        # ── read & decode ────────────────────────────────────────────────────
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        # ── preprocessing & inference (identical to other endpoints) ─────────
        cropped_image, crop_info = crop_image(image_bgr)
        padded_image, pad_info = padder(cropped_image, patch_size)

        predicted = predict_mask(model, padded_image, patch_size)
        unpadded_mask = remove_padding(predicted, pad_info)
        full_size_mask = extend_mask_to_original_size(unpadded_mask, crop_info)

        mask_uint8 = (full_size_mask * 255).astype(np.uint8)

        # ── tips ────────────────────────────────────────────────────────────
        thresholded = tresh_mask(mask_uint8)
        _, tip_coords = find_root_tips(thresholded)

        # ── build overlay ───────────────────────────────────────────────────
        # convert OpenCV BGR → RGB → Pillow image
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)

        # mask_uint8 → bytes PNG
        _, mask_buffer = cv2.imencode(".png", mask_uint8)
        overlay = make_overlay(original_pil, mask_buffer.tobytes(), tip_coords)

        # encode overlay to PNG
        out_buffer = BytesIO()
        overlay.save(out_buffer, format="PNG")
        out_buffer.seek(0)

        return StreamingResponse(out_buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build overlay: {str(e)}")

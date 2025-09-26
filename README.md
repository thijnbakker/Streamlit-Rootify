# **🌿 Rootify User Guide**

### Welcome to Rootify, an AI-powered Streamlit app designed to support **plant scientists and researchers** in analyzing root structures through image processing. This guide explains how to use the app for:

- 🧠 Root mask generation

- 📍 Root tip detection

- 📏 Root length measurement

## **🚀 Getting Started**

### **Step 1: Open the App**
#### Launch the Streamlit app in your browser. You'll be greeted with the Rootify welcome screen and feature overview.

## **🗂️ Choose Input Mode**

### You have two input options:

- **Single Image** – Analyze one image at a time.

- **Folder of Images** – Analyze multiple images in a batch.

### Select your preferred mode using the radio buttons.

## **🔧 Set Conversion Factor**

#### Set the **Pixels per Millimeter (PPM)** value using the number input field.
#### This is used to convert pixel measurements into real-world lengths (mm).
#### *Default:* 1.0

## **📤 Upload Your Image(s)**

- If in **Single Image** mode: click to upload one .jpg, .jpeg, or .png file.

- If in **Folder of Images** mode: upload multiple image files at once.

### Once uploaded:

- A **preview** of each image will appear.

- The app will automatically start processing each image.

### Accepted Image Requirements

Please ensure that the uploaded images meet the following criteria:

- The image contains a **Petri dish with clearly visible edges**, allowing clear separation from the background.

- The Petri dish includes **no more than five plants**.

- All plants must be **Arabidopsis thaliana**.

- The **roots must not be overlapping or crossing** each other.

## **🧠 What Happens During Processing?**

### For each uploaded image, Rootify:

1. **Generates a root mask** using a deep learning model.

2. **Extracts root tip coordinates.**

3. **Calculates root lengths** (based on the given pixel-to-mm conversion).

### You'll see:

- 🔬 Loading spinner during processing.

- ✅ Success messages when steps complete.

## **📊 View Results**

### For each image:

- 🖼️ **Mask:** See and download the predicted root mask.

- 📍 **Tip Coordinates:** Listed with labels (e.g., Tip 1: [x, y]).

- 📏 **Root Lengths:** Displayed in millimeters (e.g., Root 1: 25 mm).

## **📥 Export Results**

### At the bottom of the app, you can download:

1. **All Predicted Masks (ZIP)**

    - Includes individual *_mask.png files for each uploaded image.

2. **CSV Summary File**

    - Contains:

        - image name

        - root_lengths_mm: Comma-separated lengths

        - tip_coordinates: Semicolon-separated coordinate lists

## **🔄 Reset Application**

### Click the **🔄 Reset Application** button at any time to:

- Clear uploaded images

- Reset session state

- Restart from the input selection

## **⚠️ Error Handling**

- ❌ Invalid image files are rejected with a clear message.

- 🛠️ Unexpected model response formats trigger a warning.

- 📉 No roots detected will be noted per root index.

## **🧪 Technical Requirements**

- Accepted file types: .jpg, .jpeg, .png

- Backend dependencies: Custom APIs exposed in deployment.api.main:

    - predict_mask_api

    - extract_root_tips

    - calculate_root_lengths

## **💬 Support**

### For technical issues or questions about integrating your own models, reach out to the development team.
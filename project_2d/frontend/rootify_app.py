import streamlit as st
from PIL import Image, UnidentifiedImageError
import asyncio
import os
import sys
import json
import zipfile
import pandas as pd
import tempfile
from io import BytesIO
from fastapi import UploadFile

# Import API
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import deployment.api.main as api  # Must expose: predict_mask_api, extract_root_tips, calculate_root_lengths, mask_overlay

# Page setup
st.set_page_config(page_title="Rootify", page_icon="🌱", layout="centered")

# Custom CSS to match portfolio theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Root variables matching portfolio */
    :root {
        --primary: #d4a574;
        --primary-dark: #8b6f47;
        --secondary: #f4a460;
        --dark: #1a1410;
        --light: #faf6f0;
        --gray: #8b7355;
    }
    
    /* Main app background */
    .stApp {
        background-color: #1a1410;
        color: #faf6f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, rgba(212, 165, 116, 0.1), rgba(244, 164, 96, 0.1));
        border: 1px solid rgba(212, 165, 116, 0.2);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-title {
        background: linear-gradient(135deg, #d4a574 0%, #8b6f47 50%, #f4a460 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4em;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Streamlit component overrides */
    .stSelectbox > label, .stRadio > label, .stNumberInput > label, 
    .stFileUploader > label {
        color: #d4a574 !important;
        font-weight: 600;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #d4a574 0%, #8b6f47 50%, #f4a460 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(212, 165, 116, 0.4);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #d4a574 0%, #8b6f47 50%, #f4a460 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(212, 165, 116, 0.4);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(212, 165, 116, 0.1);
        border: 2px dashed rgba(212, 165, 116, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(212, 165, 116, 0.2);
        border: 1px solid #d4a574;
        color: #faf6f0;
    }
    
    .stError {
        background-color: rgba(220, 53, 69, 0.2);
        border: 1px solid #dc3545;
        color: #faf6f0;
    }
    
    .stWarning {
        background-color: rgba(244, 164, 96, 0.2);
        border: 1px solid #f4a460;
        color: #faf6f0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #d4a574 !important;
    }
    
    /* Input fields */
    .stNumberInput input {
        background-color: rgba(212, 165, 116, 0.1);
        border: 1px solid rgba(212, 165, 116, 0.3);
        border-radius: 8px;
        color: #faf6f0;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: rgba(212, 165, 116, 0.05);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Section headers */
    h3 {
        color: #f4a460;
        border-bottom: 2px solid #d4a574;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #d4a574, #f4a460);
        margin: 2rem 0;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        border: 2px solid rgba(212, 165, 116, 0.3);
    }
    
    /* Image captions styling */
    .stImage > div > div > div {
        color: #8b7355 !important;
        font-style: italic;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Text styling */
    p, li {
        color: #faf6f0;
    }
    
    /* Custom result sections */
    .result-section {
        background: linear-gradient(135deg, rgba(212, 165, 116, 0.05), rgba(244, 164, 96, 0.05));
        border: 1px solid rgba(212, 165, 116, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- UI HEADER ---
st.markdown(
    """
    <div class='main-header'>
        <h1 class='main-title'>🌿 Rootify</h1>
        <p style='font-size: 1.2em; color: #8b7355; margin-bottom: 1rem;'><strong>Welcome to Rootify</strong>, a smart tool for plant scientists and researchers.</p>
        <p style='font-size: 1em; color: #8b7355; margin-bottom: 1rem;'>This application allows you to:</p>
        <ul style='display: inline-block; text-align: middle; font-size: 1em; color: #8b7355;'>
            <p><strong style='color: #d4a574;'> Generate root masks</strong></p>
            <p><strong style='color: #d4a574;'> Identify root tip coordinates</strong></p>
            <p><strong style='color: #d4a574;'> Estimate root lengths</strong></p>
        </ul>
        <p style='font-size: 0.9em; color: #8b7355; margin-top: 1rem;'>Powered by deep learning and designed for usability in plant phenotyping and root morphology studies.</p>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("### Select Input Type")
input_mode = st.radio("Choose input mode:", ["Single Image", "Folder of Images"])

# --- Reset button ---
if st.button("Reset Application"):
    st.session_state.clear()
    st._rerun()

accepted_types = ["png", "jpg", "jpeg"]
patch_size = 256
conversion_factor = st.number_input("Pixels per millimeter (optional)", min_value=0.01, value=1.0, step=0.01)

if input_mode == "Single Image":
    uploaded_files = st.file_uploader("📷 Upload a single image", type=accepted_types, accept_multiple_files=False)
    uploaded_files = [uploaded_files] if uploaded_files else []
else:
    uploaded_files = st.file_uploader("📁 Upload multiple images", type=accepted_types, accept_multiple_files=True)

# Prepare result stores
if "all_masks" not in st.session_state:
    st.session_state["all_masks"] = {}

if "results_table" not in st.session_state:
    st.session_state["results_table"] = []

if "all_overlays" not in st.session_state:
    st.session_state["all_overlays"] = {}

# --- Process each image ---
for uploaded_file in uploaded_files:
    if uploaded_file is None:
        continue

    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        preview_image = Image.open(BytesIO(image_bytes))
        st.image(preview_image, caption=f"📸 Uploaded: {uploaded_file.name}")

        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        valid_image = True
    except (UnidentifiedImageError, OSError):
        st.error(f"❌ {uploaded_file.name} is not a valid image.")
        valid_image = False

    if valid_image:
        with st.spinner(f"☀️ Processing {uploaded_file.name}..."):

            async def full_analysis():
                upload_file = UploadFile(filename=uploaded_file.name, file=BytesIO(image_bytes))
                mask_result = await api.predict_mask_api(upload_file, patch_size=patch_size)

                mask_bytes = BytesIO()
                async for chunk in mask_result.body_iterator:
                    mask_bytes.write(chunk)
                mask_bytes.seek(0)

                upload_file_tips = UploadFile(filename=uploaded_file.name, file=BytesIO(image_bytes))
                tips_result = await api.extract_root_tips(upload_file_tips, patch_size=patch_size)

                upload_file_lengths = UploadFile(filename=uploaded_file.name, file=BytesIO(image_bytes))
                lengths_result = await api.calculate_root_lengths(upload_file_lengths, patch_size=patch_size)

                upload_file_overlay = UploadFile(filename=uploaded_file.name, file=BytesIO(image_bytes))
                overlay_response = await api.mask_overlay(upload_file_overlay, patch_size=patch_size)

                overlay_bytes = BytesIO()
                async for chunk in overlay_response.body_iterator:
                    overlay_bytes.write(chunk)
                overlay_bytes.seek(0)

                return mask_bytes, tips_result, lengths_result, overlay_bytes

            mask_bytes, tips_response, lengths_response, overlay_bytes = asyncio.run(full_analysis())
            st.session_state[f"mask_{uploaded_file.name}"] = mask_bytes.getvalue()

            # Store for download
            st.session_state["all_masks"][uploaded_file.name] = st.session_state[f"mask_{uploaded_file.name}"]

            # Display results - simplified approach to avoid layout issues
            
            # Mask section - pure HTML with styled header
            st.markdown("""
            <div class='result-section' style='text-align: center;'>
                <h4 style='color: #f4a460; border-bottom: 2px solid #d4a574; padding-bottom: 0.5rem; margin-bottom: 1rem;'>💡 Predicted Root Mask</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(BytesIO(st.session_state[f"mask_{uploaded_file.name}"]), caption="Generated root mask", use_container_width=True)

            # Overlay section - pure HTML with styled header  
            st.markdown("""
            <div class='result-section' style='text-align: center;'>
                <h4 style='color: #f4a460; border-bottom: 2px solid #d4a574; padding-bottom: 0.5rem; margin-bottom: 1rem;'>🌟 Mask Overlay</h4>
            </div>
            """, unsafe_allow_html=True)
            st.image(overlay_bytes, caption="Root mask overlaid on original image", use_container_width=True)

            # 🔹 Save overlay for later bulk download
            st.session_state["all_overlays"][uploaded_file.name] = overlay_bytes.getvalue()

            st.download_button(
                label=f"💾 Download Mask for {uploaded_file.name}",
                data=st.session_state[f"mask_{uploaded_file.name}"],
                file_name=f"{uploaded_file.name}_mask.png",
                mime="image/png"
            )

            # Parse responses
            if hasattr(lengths_response, "body"):
                length_data = json.loads(lengths_response.body.decode("utf-8"))
            else:
                length_data = lengths_response

            if isinstance(tips_response, dict):
                tip_coordinates = tips_response.get("tip_coordinates", [])
            else:
                tip_coordinates = json.loads(tips_response.body.decode("utf-8"))["tip_coordinates"]

            raw_lengths = length_data.get("root_lengths", [])
            processed_lengths = [int(round(length / conversion_factor)) for length in raw_lengths]

            # Display tip coordinates in styled container
            display_tip_coordinates = []
            tip_coordinates_html = "<div class='result-section'><h4 style='color: #f4a460; border-bottom: 2px solid #d4a574; padding-bottom: 0.5rem; margin-bottom: 1rem;'>💫 Root Tips Analysis</h4><p style='color: #d4a574; font-weight: 600; margin-bottom: 1rem;'>👍 Root tips extracted successfully!</p><p style='color: #faf6f0; font-weight: 600; margin-bottom: 0.5rem;'>Tip Coordinates:</p>"
            
            for idx, length in enumerate(processed_lengths):
                try:
                    coord = tip_coordinates[idx]
                except IndexError:
                    coord = None

                if length == 0 or coord is None:
                    tip_coordinates_html += f"<p style='color: #faf6f0;'><strong>Tip {idx + 1}:</strong> No Roots Detected</p>"
                    display_tip_coordinates.append("No Roots Detected")
                else:
                    tip_coordinates_html += f"<p style='color: #faf6f0;'><strong>Tip {idx + 1}:</strong> {coord}</p>"
                    display_tip_coordinates.append(coord)
            
            tip_coordinates_html += "</div>"
            st.markdown(tip_coordinates_html, unsafe_allow_html=True)

            # Display root lengths in styled container
            root_lengths_html = "<div class='result-section'><h4 style='color: #f4a460; border-bottom: 2px solid #d4a574; padding-bottom: 0.5rem; margin-bottom: 1rem;'>📏 Root Length Analysis</h4>"
            
            if "root_lengths" in length_data:
                root_lengths_html += "<p style='color: #d4a574; font-weight: 600; margin-bottom: 1rem;'>👍 Root lengths calculated successfully!</p>"
                root_lengths_html += "<p style='color: #faf6f0; font-weight: 600; margin-bottom: 0.5rem;'>Root Lengths:</p>"
                for idx, length in enumerate(processed_lengths):
                    root_lengths_html += f"<p style='color: #faf6f0;'><strong>Root {idx + 1}:</strong> {length} mm</p>"
            elif "detail" in length_data:
                root_lengths_html += f"<p style='color: #dc3545; font-weight: 600; margin-bottom: 1rem;'>❌ {length_data['detail']}</p>"
            else:
                root_lengths_html += "<p style='color: #f4a460; font-weight: 600; margin-bottom: 1rem;'>⚠️ Unexpected format in root length response.</p>"
            
            root_lengths_html += "</div>"
            st.markdown(root_lengths_html, unsafe_allow_html=True)

            # Save results for CSV
            row = {
                "image": uploaded_file.name,
                "root_lengths_mm": processed_lengths,
                "tip_coordinates": display_tip_coordinates
            }
            st.session_state["results_table"].append(row)

# --- Download All Results ---
if st.session_state["all_masks"]:
    st.markdown("### ✨ Download All Results")
    st.markdown("<br>", unsafe_allow_html=True)

    # ZIP file with all masks
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        with zipfile.ZipFile(tmp_zip.name, "w") as zipf:
            for filename, mask_data in st.session_state["all_masks"].items():
                zipf.writestr(f"{filename}_mask.png", mask_data)

        with open(tmp_zip.name, "rb") as f:
            st.download_button(
                label="📁 Download All Masks (ZIP)",
                data=f,
                file_name="all_predicted_masks.zip",
                mime="application/zip"
            )
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    
    # ZIP file with all overlays
    if st.session_state.get("all_overlays"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, "w") as zipf:
                for filename, overlay_data in st.session_state["all_overlays"].items():
                    zipf.writestr(f"{filename}_overlay.png", overlay_data)

            with open(tmp_zip.name, "rb") as f:
                st.download_button(
                    label="📁 Download All Overlays (ZIP)",
                    data=f,
                    file_name="all_overlays.zip",
                    mime="application/zip",
                )

    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing

    # CSV file with lengths and tips
    df = pd.DataFrame(st.session_state["results_table"])
    df["root_lengths_mm"] = df["root_lengths_mm"].apply(lambda x: ", ".join(map(str, x)))
    df["tip_coordinates"] = df["tip_coordinates"].apply(lambda x: "; ".join([str(tip) for tip in x]))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Download CSV (Lengths & Tip Coordinates)",
        data=csv,
        file_name="root_analysis_results.csv",
        mime="text/csv"
    )

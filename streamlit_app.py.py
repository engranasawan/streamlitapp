# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import re
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image

# === Load credentials ===
creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)
client = vision.ImageAnnotatorClient(credentials=creds)

# === Streamlit UI ===
st.title("⚡ Energy Meter OCR Reader")
uploaded_file = st.file_uploader("Upload an image of the meter", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Read and preprocess
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    def crop_roi(img, frac=(0.10, 0.40, 0.15, 0.85)):
        h, w = img.shape[:2]
        t, b, l, r = (int(frac[0]*h), int(frac[1]*h), int(frac[2]*w), int(frac[3]*w))
        return img[t:b, l:r]

    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        return cv2.bilateralFilter(img, 9, 75, 75)

    roi = crop_roi(img)
    proc_img = preprocess(roi)
    _, buf = cv2.imencode('.png', proc_img)
    vision_img = vision.Image(content=buf.tobytes())
    response = client.text_detection(image=vision_img)
    texts = response.text_annotations

    digits, status = '', 'not_found'
    if texts and len(texts) > 1:
        for t in texts[1:]:
            clean = re.sub(r'\D', '', t.description.strip())
            if len(clean) == 6:
                digits, status = clean, 'valid'
                break

    if digits:
        st.success(f"🔢 Meter Reading: {digits} ({status})")
    else:
        st.error("❌ No valid 6-digit meter reading found.")

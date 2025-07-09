import streamlit as st
import numpy as np
import cv2
import io
import re
from google.cloud import vision
from google.oauth2 import service_account

# === CONFIGURATION ===
ROI_FRAC = (0.10, 0.40, 0.15, 0.85)
DISPLAY_SIZE = (300, 300)

# === GCP Setup ===
creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)
client = vision.ImageAnnotatorClient(credentials=creds)

# === Helper Functions ===
def preprocess_for_ocr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img_inpaint = cv2.inpaint(img_clahe, mask, 5, cv2.INPAINT_TELEA)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_inpaint, table)

    return cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

def crop_roi(img, frac):
    h, w = img.shape[:2]
    t, b, l, r = (int(frac[0]*h), int(frac[1]*h), int(frac[2]*w), int(frac[3]*w))
    return img[t:b, l:r]

# === Streamlit App UI ===
st.title("📷 Energy Meter OCR")
uploaded_file = st.file_uploader("Upload meter image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    orig = cv2.imdecode(file_bytes, 1)

    roi = crop_roi(orig, ROI_FRAC)
    proc_roi = preprocess_for_ocr(roi)

    _, buf = cv2.imencode('.png', proc_roi)
    vision_image = vision.Image(content=buf.tobytes())
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    if texts and len(texts) > 1:
        vis = roi.copy()
        for text in texts[1:]:
            pts = [(v.x, v.y) for v in text.bounding_poly.vertices]
            for j in range(4):
                cv2.line(vis, pts[j], pts[(j+1)%4], (0, 255, 0), 2)

        vis_small = cv2.resize(vis, DISPLAY_SIZE)
        st.image(cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB), caption="OCR Region Detection")

        # === Digit Filtering Logic ===
        candidates = []
        for item in texts[1:]:
            raw = item.description.strip()
            cleaned = re.sub(r'\D', '', raw)

            if len(cleaned) == 6:
                candidates.append((cleaned, "valid"))
            elif len(cleaned) == 5:
                candidates.append((cleaned + "x", "partial"))
            elif len(cleaned) == 7:
                candidates.append((cleaned[:6], "valid"))

        if candidates:
            digits, status = candidates[0]
            st.success(f"📌 Meter Reading: {digits} ({status})")
        else:
            st.warning("⚠️ No valid 5–7 digit numeric pattern found.")
    else:
        st.error("❌ No text regions detected.")

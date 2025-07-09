import streamlit as st
import numpy as np
import pandas as pd
import io
import re
import cv2
from google.cloud import vision
from google.oauth2 import service_account

# === CONFIGURATION ===
ROI_FRAC = (0.10, 0.40, 0.15, 0.85)
DISPLAY_SIZE = (300, 300)

# === Preprocessing Function ===
def preprocess_for_ocr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    img_inpaint = cv2.inpaint(img_clahe, mask, 5, cv2.INPAINT_TELEA)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_inpaint, table)

    return cv2.bilateralFilter(img_gamma, d=9, sigmaColor=75, sigmaSpace=75)

# === ROI Crop with Zoom ===
def crop_roi(img, frac, zoom_percent=20):
    h, w = img.shape[:2]
    t, b = int(frac[0] * h), int(frac[1] * h)
    l, r = int(frac[2] * w), int(frac[3] * w)

    roi = img[t:b, l:r]
    roi_h, roi_w = roi.shape[:2]

    pad_h = int(roi_h * (zoom_percent / 100) / 2)
    pad_w = int(roi_w * (zoom_percent / 100) / 2)

    new_t = max(t - pad_h, 0)
    new_b = min(b + pad_h, h)
    new_l = max(l - pad_w, 0)
    new_r = min(r + pad_w, w)

    return img[new_t:new_b, new_l:new_r]

# === Load Google Credentials from secrets.toml ===
creds = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = vision.ImageAnnotatorClient(credentials=creds)

# === Streamlit UI ===
st.set_page_config(page_title="Meter OCR", layout="wide")
st.title("🔢 AI-Powered Energy Meter Reading (OCR)")

uploaded_images = st.file_uploader("📸 Upload Meter Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    results = []
    for uploaded_file in uploaded_images:
        st.subheader(f"📷 Image: `{uploaded_file.name}`")
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        orig = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if orig is None:
            st.error("❌ Could not read image.")
            continue

        roi = crop_roi(orig, ROI_FRAC)
        proc_roi = preprocess_for_ocr(roi)

        _, buf = cv2.imencode('.png', proc_roi)
        vision_image = vision.Image(content=buf.tobytes())
        resp = client.text_detection(image=vision_image)
        texts = resp.text_annotations

        digits, status = "", ""
        candidates = []

        if texts and len(texts) > 1:
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
                st.success(f"📌 Meter Reading: **{digits}** ({status})")
                results.append((uploaded_file.name, digits))

                vis = roi.copy()
                for text in texts[1:]:
                    pts = [(v.x, v.y) for v in text.bounding_poly.vertices]
                    for j in range(4):
                        cv2.line(vis, pts[j], pts[(j+1)%4], (0,255,0), 2)
                vis_resized = cv2.resize(vis, DISPLAY_SIZE)
                st.image(vis_resized, caption="Detected Region", channels="BGR")
            else:
                st.warning("⚠️ No valid 5–7 digit numeric pattern found.")
        else:
            st.warning("❌ No text detected.")

    if results:
        df = pd.DataFrame(results, columns=["image_name", "meter_reading"])
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv_bytes, "meter_readings.csv", "text/csv")

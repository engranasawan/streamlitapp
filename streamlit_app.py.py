import streamlit as st
import numpy as np
import cv2
import re
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account

# === CONFIG ===
ROI_FRAC = (0.10, 0.40, 0.15, 0.85)
DISPLAY_SIZE = (300, 300)
ZOOM_FACTOR = 1.2

# === Streamlit UI ===
st.title("🔍 Energy Meter Reading (OCR)")
uploaded_img = st.file_uploader("Upload a meter image", type=["png", "jpg", "jpeg"])
st.markdown("---")

# === Setup GCP Vision Client from secrets ===
creds_dict = st.secrets["gcp_service_account"]
creds = service_account.Credentials.from_service_account_info(creds_dict)
client = vision.ImageAnnotatorClient(credentials=creds)

# === Helper Functions ===
def crop_roi(img, frac, zoom_factor=1.2):
    h, w = img.shape[:2]
    t, b, l, r = (int(frac[0]*h), int(frac[1]*h), int(frac[2]*w), int(frac[3]*w))
    cropped = img[t:b, l:r]
    zoomed = cv2.resize(cropped, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

    # Center crop to original size
    zh, zw = zoomed.shape[:2]
    ch, cw = b - t, r - l
    start_y = max((zh - ch) // 2, 0)
    start_x = max((zw - cw) // 2, 0)
    zoom_cropped = zoomed[start_y:start_y+ch, start_x:start_x+cw]

    return zoom_cropped

def preprocess_for_ocr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_clahe, mask, 5, cv2.INPAINT_TELEA)

    gamma = 1.2
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    gamma_corrected = cv2.LUT(inpainted, table)

    return cv2.bilateralFilter(gamma_corrected, 9, 75, 75)

def extract_digits(texts):
    candidates = []
    for item in texts[1:]:
        raw = item.description.strip()
        cleaned = re.sub(r"\D", "", raw)

        if len(cleaned) == 6:
            candidates.append((cleaned, "valid"))
        elif len(cleaned) == 5:
            candidates.append((cleaned + "x", "partial"))
        elif len(cleaned) == 7:
            candidates.append((cleaned[:6], "valid"))

    return candidates[0] if candidates else ("", "")

# === Main Logic ===
if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    roi = crop_roi(img_np, ROI_FRAC, zoom_factor=ZOOM_FACTOR)
    proc_roi = preprocess_for_ocr(roi)

    _, img_encoded = cv2.imencode('.png', proc_roi)
    vision_image = vision.Image(content=img_encoded.tobytes())
    response = client.text_detection(image=vision_image)
    texts = response.text_annotations

    if texts:
        digits, status = extract_digits(texts)
        st.success(f"📟 Meter Reading: {digits} ({status})" if digits else "⚠️ No valid reading found.")
        st.image(cv2.cvtColor(proc_roi, cv2.COLOR_BGR2RGB), caption="Processed ROI", width=300)
    else:
        st.warning("No text detected by Google OCR.")

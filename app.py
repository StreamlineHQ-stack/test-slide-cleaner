import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Slide Text Remover Test", layout="centered")

st.title("✨ Fast Slide Text Remover — Test App")
st.markdown("Testing EasyOCR + OpenCV Inpainting on your reference slides.")

@st.cache_resource
def load_reader():
    import easyocr
    with st.spinner("Loading EasyOCR model..."):
        return easyocr.Reader(["en"], gpu=False, verbose=False)

uploaded_file = st.file_uploader("Upload an inspiration slide", type=["png", "jpg", "jpeg"])

conf_thresh = st.slider("Detection Confidence", 0.1, 0.7, 0.25, 0.05)
expand = st.slider("Mask Expand", 1.0, 2.0, 1.35, 0.05)
dilate = st.slider("Mask Dilate", 1, 25, 13, 2)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Slide")
        st.image(original_image, use_container_width=True)
        
    if st.button("🚀 Erase Text Instantly", type="primary"):
        reader = load_reader()
        
        with st.spinner("Detecting and removing text..."):
            t0 = time.time()
            img_rgb = np.array(original_image)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]
            
            max_side = 1200
            scale = 1.0
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = img_bgr

            results = reader.readtext(small)
            mask = np.zeros((h, w), dtype=np.uint8)
            detected = []

            for bbox, text, conf in results:
                if conf < conf_thresh or not text.strip():
                    continue
                pts = np.array(bbox, dtype=np.float32) / scale
                center = pts.mean(axis=0)
                pts = (pts - center) * expand + center
                pts = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
                detected.append(f"{text} ({conf:.2f})")

            if dilate > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
                mask = cv2.dilate(mask, kernel, iterations=1)

            if mask.sum() > 0:
                # Fast OpenCV Inpainting
                cleaned_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(cleaned_rgb)
            else:
                result_pil = original_image.copy()

            elapsed = time.time() - t0

        with col2:
            st.subheader(f"Cleaned ({elapsed:.1f}s)")
            st.image(result_pil, use_container_width=True)
            
        st.success(f"Done! Detected text: {', '.join(detected) if detected else 'None'}")
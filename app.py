#!/usr/bin/env python3
"""
Slide Text Remover - Commercial Quality Version
================================================
EasyOCR (text detection) + LaMa (neural large-mask inpainting)

This version is designed to be sold as a product.
LaMa produces significantly better results than classical inpainting
on real design / inspirational / marketing slides.

Author: Built for commercial use
License note: LaMa is Apache-2.0 → fully commercial friendly
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ---------- Model loading (lazy) ----------
_READER = None
_LAMA = None

def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        print("Loading EasyOCR...")
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        print("EasyOCR ready.")
    return _READER

def get_lama():
    global _LAMA
    if _LAMA is None:
        from simple_lama_inpainting import SimpleLama
        print("Loading LaMa model (first time downloads ~200MB)...")
        _LAMA = SimpleLama()
        print(f"LaMa ready on device: {_LAMA.device}")
    return _LAMA


def create_text_mask(
    img_bgr: np.ndarray,
    conf_thresh: float = 0.25,
    expand: float = 1.35,
    dilate_size: int = 13,
    max_side: int = 1200,
) -> Tuple[np.ndarray, List[str]]:
    """Detect text with EasyOCR and build a refined mask."""
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr

    reader = get_reader()
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

    if dilate_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, detected


def remove_text_lama(
    img_bgr: np.ndarray,
    conf_thresh: float = 0.25,
    expand: float = 1.35,
    dilate_size: int = 13,
    max_side: int = 1200,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Full pipeline: EasyOCR → mask → LaMa inpainting
    Returns cleaned BGR image, list of detected texts, processing time
    """
    t0 = time.time()

    mask, detected = create_text_mask(
        img_bgr,
        conf_thresh=conf_thresh,
        expand=expand,
        dilate_size=dilate_size,
        max_side=max_side,
    )

    if mask.sum() == 0:
        return img_bgr.copy(), ["(no text detected)"], time.time() - t0

    # Prepare for LaMa (expects RGB PIL + grayscale mask)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    mask_pil = Image.fromarray(mask)

    lama = get_lama()
    result_pil = lama(img_pil, mask_pil)

    # Back to BGR numpy
    result_rgb = np.array(result_pil)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    elapsed = time.time() - t0
    return result_bgr, detected, elapsed


def process_image(
    image: Image.Image,
    conf_thresh: float = 0.25,
    expand: float = 1.35,
    dilate: int = 13,
) -> Tuple[Optional[Image.Image], str]:
    """Gradio-compatible processing function."""
    if image is None:
        return None, "Please upload an image."

    try:
        # Convert PIL → BGR
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cleaned_bgr, detected, elapsed = remove_text_lama(
            img_bgr,
            conf_thresh=conf_thresh,
            expand=expand,
            dilate_size=dilate,
        )

        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        cleaned_pil = Image.fromarray(cleaned_rgb)

        info = f"✅ Done in {elapsed:.1f}s\n\nDetected text regions:\n"
        if detected:
            info += "\n".join(f"• {d}" for d in detected[:15])
            if len(detected) > 15:
                info += f"\n... and {len(detected)-15} more"
        else:
            info += "(none above confidence threshold)"

        # Cleanup
        del img_bgr, cleaned_bgr
        gc.collect()

        return cleaned_pil, info

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def build_ui():
    with gr.Blocks(
        title="Slide Text Remover — Pro",
        theme=gr.themes.Soft(primary_hue="indigo"),
        css="""
        .gradio-container { max-width: 1100px !important; }
        footer { display: none !important; }
        """
    ) as demo:
        gr.Markdown(
            """
            # ✨ Slide Text Remover — Pro
            **EasyOCR + LaMa Neural Inpainting**

            Upload any inspirational / marketing / poster slide and remove the text cleanly.  
            Powered by **LaMa** (Large Mask Inpainting) for commercial-grade results.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    type="pil",
                    label="Upload Slide",
                    height=420,
                )
                with gr.Accordion("Advanced Settings", open=False):
                    conf = gr.Slider(0.1, 0.7, value=0.25, step=0.05, label="Detection Confidence (lower = more aggressive)")
                    expand = gr.Slider(1.1, 1.8, value=1.35, step=0.05, label="Mask Expand")
                    dilate = gr.Slider(5, 25, value=13, step=2, label="Mask Dilate (pixels)")
                btn = gr.Button("Remove Text", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_img = gr.Image(type="pil", label="Cleaned Result", height=420)
                info_box = gr.Textbox(label="Status / Detected Text", lines=8, interactive=False)

        btn.click(
            fn=process_image,
            inputs=[input_img, conf, expand, dilate],
            outputs=[output_img, info_box],
        )

        gr.Markdown(
            """
            ---
            **Tips for best results**
            - Lower confidence (0.15–0.20) if text is being missed
            - Increase Expand / Dilate if leftover letter edges remain
            - Works best on solid / soft / photo backgrounds
            - First run downloads the LaMa model (~200 MB) — only once

            *Built for commercial use. LaMa model is Apache-2.0 licensed.*
            """
        )

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  Slide Text Remover — Pro (EasyOCR + LaMa)")
    print("=" * 60)
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

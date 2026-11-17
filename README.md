# Slide Text Remover — Pro

**Commercial-grade text removal for inspirational, marketing & design slides.**

Uses:
- **EasyOCR** → robust text detection (many fonts)
- **LaMa (Big-LaMa)** → state-of-the-art neural large-mask inpainting

This combination produces significantly cleaner results than classical OpenCV inpainting, especially on photo and textured backgrounds.

---

## Why this is sellable

| Feature                    | Free tools (TELEA etc.) | This App (LaMa)      |
|---------------------------|-------------------------|----------------------|
| Quality on real slides    | Often blurry / artifacts | Natural reconstruction |
| Large text areas          | Weak                    | Strong               |
| License of core model     | N/A                     | Apache 2.0 (commercial OK) |
| Ready for customers       | No                      | Yes (Gradio UI)      |

You can legitimately charge for the polished product, UI, support, and updates while using the open-source LaMa model under Apache 2.0.

---

## Quick Start

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open **http://localhost:7860** in your browser.

First launch will download the LaMa model (~196 MB). This happens only once.

---

## Usage

1. Upload any slide / poster / quote image
2. (Optional) Adjust confidence / expand / dilate in Advanced Settings
3. Click **Remove Text**
4. Download the cleaned result

### Recommended settings

- **Most slides**: default values work well
- **Missed text**: lower Confidence to 0.15–0.20
- **Leftover edges**: increase Expand to 1.45–1.6 and Dilate to 17–21

---

## System Requirements

- Python 3.9+
- 4 GB+ RAM recommended (8 GB better)
- GPU optional but much faster (LaMa auto-detects CUDA)
- Works on CPU (slower but usable)

---

## Project Structure

```
SlideTextRemover/
├── app.py                # Main application (Gradio UI + pipeline)
├── requirements.txt
└── README.md
```

---

## Commercial Notes

- LaMa model: Apache-2.0 → free for commercial use
- EasyOCR: Apache-2.0
- You own the application code, UI, packaging, and branding
- Suggested positioning: “Professional AI text remover for designers & marketers”

---

## Troubleshooting

**Out of memory**  
→ Close other apps. Or lower `max_side` inside `create_text_mask` (default 1200).

**Text still partially visible**  
→ Lower confidence + increase expand/dilate.

**First run is slow**  
→ Normal — model is downloading and loading.

---

Built to be product-ready.

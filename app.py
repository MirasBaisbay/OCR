"""
Document Parsing Playground
A Streamlit application replicating a professional document parsing interface,
powered by PaddleOCR-VL-1.5 / PP-DocLayout for layout analysis.
"""

import streamlit as st
import json
import io
import random
import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Conditional imports – gracefully degrade when heavy libs are missing
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from paddleocr import PaddleOCRVL
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Parsing Playground",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – styled cards, split-screen, polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Google Fonts ---------- */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---------- Root variables ---------- */
:root {
    --clr-bg: #0e1117;
    --clr-surface: #161b22;
    --clr-surface-alt: #1c2333;
    --clr-border: #2a3142;
    --clr-text: #e6edf3;
    --clr-text-muted: #8b949e;
    --clr-accent: #58a6ff;
    --clr-header: #f7768e;
    --clr-text-block: #7aa2f7;
    --clr-figure: #bb9af7;
    --clr-table: #73daca;
    --clr-title: #ff9e64;
    --clr-footer: #9ece6a;
    --radius: 10px;
}

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--clr-surface);
    border-right: 1px solid var(--clr-border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--clr-text) !important;
}

/* ---------- Block card ---------- */
.block-card {
    background: var(--clr-surface);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.block-card:hover {
    border-color: var(--clr-accent);
}

.block-label {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
}

.block-text {
    color: var(--clr-text);
    font-size: 14px;
    line-height: 1.65;
    margin: 0;
    white-space: pre-wrap;
}

.block-conf {
    margin-top: 8px;
    font-size: 11px;
    color: var(--clr-text-muted);
    font-family: 'JetBrains Mono', monospace;
}

/* ---------- Type-specific colors ---------- */
.label-header, .label-pageheader   { background: rgba(247,118,142,0.15); color: #f7768e; }
.label-sectionheader               { background: rgba(247,118,142,0.15); color: #f7768e; }
.label-text                        { background: rgba(122,162,247,0.15); color: #7aa2f7; }
.label-figure                      { background: rgba(187,154,247,0.15); color: #bb9af7; }
.label-table                       { background: rgba(115,218,202,0.15); color: #73daca; }
.label-title                       { background: rgba(255,158,100,0.15); color: #ff9e64; }
.label-footer, .label-pagefooter   { background: rgba(158,206,106,0.15); color: #9ece6a; }
.label-reference                   { background: rgba(86,156,214,0.15); color: #569cd6; }
.label-equation                    { background: rgba(206,145,120,0.15); color: #ce9178; }
.label-list                        { background: rgba(78,201,176,0.15); color: #4ec9b0; }

/* ---------- Stat pills ---------- */
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 12px 0 20px;
}
.stat-pill {
    background: var(--clr-surface-alt);
    border: 1px solid var(--clr-border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: var(--clr-text);
    font-family: 'JetBrains Mono', monospace;
}
.stat-pill b {
    color: var(--clr-accent);
}

/* ---------- Hero header ---------- */
.hero {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.hero-icon {
    font-size: 28px;
}
.hero h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    color: var(--clr-text);
}
.hero-sub {
    font-size: 13px;
    color: var(--clr-text-muted);
    margin: 0 0 16px 0;
}

/* ---------- Misc polish ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 13px;
}
div[data-testid="stImage"] img {
    border-radius: var(--radius);
    border: 1px solid var(--clr-border);
}

/* ---------- JSON container ---------- */
.json-wrap {
    background: var(--clr-surface);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius);
    padding: 16px;
    max-height: 700px;
    overflow-y: auto;
}

/* ---------- Markdown result ---------- */
.md-result {
    background: var(--clr-surface);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius);
    padding: 20px;
    color: var(--clr-text);
    font-size: 14px;
    line-height: 1.7;
    max-height: 700px;
    overflow-y: auto;
}

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    padding: 80px 20px;
    color: var(--clr-text-muted);
}
.empty-state .icon {
    font-size: 56px;
    margin-bottom: 16px;
    opacity: 0.5;
}
.empty-state h3 {
    color: var(--clr-text);
    margin-bottom: 8px;
}
.empty-state p {
    font-size: 14px;
    max-width: 380px;
    margin: 0 auto;
}

/* ---------- Legend ---------- */
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 10px 0 4px;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    color: var(--clr-text-muted);
    font-family: 'JetBrains Mono', monospace;
}
.legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 3px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TYPE_COLORS = {
    "header":        ("#f7768e", "label-header"),
    "pageheader":    ("#f7768e", "label-pageheader"),
    "sectionheader": ("#f7768e", "label-sectionheader"),
    "text":          ("#7aa2f7", "label-text"),
    "figure":        ("#bb9af7", "label-figure"),
    "table":         ("#73daca", "label-table"),
    "title":         ("#ff9e64", "label-title"),
    "footer":        ("#9ece6a", "label-footer"),
    "pagefooter":    ("#9ece6a", "label-pagefooter"),
    "reference":     ("#569cd6", "label-reference"),
    "equation":      ("#ce9178", "label-equation"),
    "list":          ("#4ec9b0", "label-list"),
}

DEFAULT_COLOR = ("#58a6ff", "label-text")


def get_type_info(t: str):
    key = t.lower().replace("_", "").replace("-", "").replace(" ", "")
    return TYPE_COLORS.get(key, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# PDF → Image conversion
# ---------------------------------------------------------------------------
def pdf_page_to_image(pdf_bytes: bytes, page_num: int = 0, dpi: int = 200) -> "Image.Image":
    """Convert a specific PDF page to a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(page_num)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


# ---------------------------------------------------------------------------
# PaddleOCR inference
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_paddleocr_vl():
    """Load PaddleOCR-VL-1.5 pipeline (cached)."""
    import os, paddle
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    paddle.disable_static()  # Force dynamic mode
    return PaddleOCRVL()


def _create_fresh_pipeline():
    """Create a new pipeline instance (bypasses cache)."""
    import os, paddle
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    paddle.disable_static()
    return PaddleOCRVL()


def run_paddleocr_vl(pipeline, img: "Image.Image"):
    """Run PaddleOCR-VL and normalize results into block dicts."""
    import tempfile, os, paddle

    paddle.disable_static()  # Ensure dynamic mode before every call

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp, format="PNG")
        tmp_path = tmp.name

    try:
        try:
            output = list(pipeline.predict(tmp_path))
        except RuntimeError as e:
            if "static graph mode" in str(e) or "int(Tensor)" in str(e):
                # Known PaddlePaddle bug: recreate pipeline and retry
                st.warning("Reinitializing OCR engine (PaddlePaddle static graph workaround)...")
                load_paddleocr_vl.clear()  # Clear the cached resource
                pipeline = _create_fresh_pipeline()
                # Store the new pipeline so the caller can use it next time
                st.session_state["_paddle_pipeline"] = pipeline
                output = list(pipeline.predict(tmp_path))
            else:
                raise
    finally:
        os.unlink(tmp_path)

    output_list = output
    blocks = []
    idx = 0

    # ---- Label mapping: PaddleOCR layout labels → our UI type names ----
    LABEL_MAP = {
        "doc_title": "title",
        "title": "title",
        "text": "text",
        "paragraph": "text",
        "image": "figure",
        "figure": "figure",
        "figure_caption": "text",
        "table": "table",
        "table_caption": "text",
        "header": "pageheader",
        "page_header": "pageheader",
        "page-header": "pageheader",
        "footer": "pagefooter",
        "page_footer": "pagefooter",
        "page-footer": "pagefooter",
        "reference": "reference",
        "equation": "equation",
        "formula": "equation",
        "list": "list",
        "section_header": "sectionheader",
        "section-header": "sectionheader",
        "abstract": "text",
        "seal": "figure",
    }

    def _normalize_label(raw_label: str) -> str:
        key = raw_label.lower().strip().replace(" ", "_")
        return LABEL_MAP.get(key, "text")

    for res in output_list:
        # ------------------------------------------------------------------
        # Build a score lookup from layout_det_res.boxes (coordinate → score)
        # ------------------------------------------------------------------
        score_lookup = {}  # (x0, y0, x1, y1) → score
        layout_det = res.get("layout_det_res")
        if isinstance(layout_det, dict):
            for box_info in (layout_det.get("boxes") or []):
                if isinstance(box_info, dict):
                    coord = box_info.get("coordinate")
                    score = box_info.get("score", 0.0)
                    if coord and len(coord) == 4:
                        score_lookup[tuple(int(c) for c in coord)] = float(score)

        # ------------------------------------------------------------------
        # Primary: iterate parsing_res_list (PaddleOCRVLBlock objects)
        # Each has: .bbox, .content, .label, .image, .polygon_points
        # ------------------------------------------------------------------
        parsing_list = res.get("parsing_res_list") or []

        for pr in parsing_list:
            # --- bbox ---
            bbox_raw = getattr(pr, "bbox", None) or [0, 0, img.width, img.height]
            bbox = [int(v) for v in bbox_raw]

            # --- label ---
            raw_label = getattr(pr, "label", "text") or "text"
            block_type = _normalize_label(raw_label)

            # --- text content ---
            text = getattr(pr, "content", "") or ""

            # --- confidence: match against layout_det score_lookup ---
            conf = score_lookup.get(tuple(bbox), 0.90)

            blocks.append({
                "id": idx,
                "type": block_type,
                "bbox": bbox,
                "text": str(text).strip(),
                "confidence": conf,
            })
            idx += 1

        # ------------------------------------------------------------------
        # Fallback: if parsing_res_list was empty, try layout_det_res.boxes
        # ------------------------------------------------------------------
        if not blocks and layout_det and isinstance(layout_det, dict):
            for box_info in (layout_det.get("boxes") or []):
                if not isinstance(box_info, dict):
                    continue
                coord = box_info.get("coordinate", [0, 0, img.width, img.height])
                bbox = [int(c) for c in coord] if len(coord) == 4 else [0, 0, img.width, img.height]
                raw_label = box_info.get("label", "text") or "text"
                block_type = _normalize_label(raw_label)
                conf = float(box_info.get("score", 0.0))

                blocks.append({
                    "id": idx,
                    "type": block_type,
                    "bbox": bbox,
                    "text": "",
                    "confidence": conf,
                })
                idx += 1

    if not blocks:
        print("WARNING: No blocks extracted from document! Check pipeline output.")

    return blocks


# ---------------------------------------------------------------------------
# Mock data fallback
# ---------------------------------------------------------------------------
def generate_mock_blocks(img_w: int, img_h: int) -> list:
    """Generate plausible mock layout blocks for UI testing."""
    blocks = []
    idx = 0
    margin = int(img_w * 0.07)

    # Page header
    blocks.append({
        "id": idx, "type": "pageheader",
        "bbox": [margin, int(img_h * 0.02), img_w - margin, int(img_h * 0.06)],
        "text": "RESEARCH ARTICLE  |  Open Access  |  DOI: 10.1234/example.2026",
        "confidence": 0.97,
    }); idx += 1

    # Title
    blocks.append({
        "id": idx, "type": "title",
        "bbox": [margin, int(img_h * 0.07), img_w - margin, int(img_h * 0.14)],
        "text": "PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing",
        "confidence": 0.98,
    }); idx += 1

    # Section header
    blocks.append({
        "id": idx, "type": "sectionheader",
        "bbox": [margin, int(img_h * 0.15), int(img_w * 0.4), int(img_h * 0.18)],
        "text": "1. Introduction",
        "confidence": 0.96,
    }); idx += 1

    # Two text blocks
    blocks.append({
        "id": idx, "type": "text",
        "bbox": [margin, int(img_h * 0.19), img_w - margin, int(img_h * 0.35)],
        "text": (
            "Document understanding has emerged as a core research topic in the field of computer vision and "
            "natural language processing. Real-world documents exhibit significant diversity in layout, typography, "
            "and visual complexity, posing substantial challenges for automated parsing systems. Recent advances in "
            "vision-language models (VLMs) have shown promising results on benchmark datasets, yet their robustness "
            "under physical distortions commonly encountered in scanning, photography, and mobile capture remains "
            "an open question."
        ),
        "confidence": 0.95,
    }); idx += 1

    blocks.append({
        "id": idx, "type": "text",
        "bbox": [margin, int(img_h * 0.36), img_w - margin, int(img_h * 0.50)],
        "text": (
            "In this paper, we present PaddleOCR-VL-1.5, a next-generation model that achieves state-of-the-art "
            "accuracy of 94.5% on OmniDocBench v1.5 while maintaining an ultra-compact 0.9B parameter footprint. "
            "Our model introduces polygonal detection for irregular document regions and extends capabilities to "
            "include seal recognition and text spotting, enabling comprehensive document understanding across "
            "challenging real-world conditions."
        ),
        "confidence": 0.94,
    }); idx += 1

    # Figure
    blocks.append({
        "id": idx, "type": "figure",
        "bbox": [int(img_w * 0.15), int(img_h * 0.52), int(img_w * 0.85), int(img_h * 0.75)],
        "text": "Figure 1: Overview of the PaddleOCR-VL-1.5 architecture showing the vision encoder, language decoder, and multi-task prediction heads.",
        "confidence": 0.93,
    }); idx += 1

    # Section header 2
    blocks.append({
        "id": idx, "type": "sectionheader",
        "bbox": [margin, int(img_h * 0.76), int(img_w * 0.5), int(img_h * 0.79)],
        "text": "2. Related Work",
        "confidence": 0.96,
    }); idx += 1

    # Another text block
    blocks.append({
        "id": idx, "type": "text",
        "bbox": [margin, int(img_h * 0.80), img_w - margin, int(img_h * 0.92)],
        "text": (
            "Prior work on document layout analysis has progressed from rule-based methods to deep learning "
            "approaches. Early CNN-based models such as Mask R-CNN were adapted for document regions, while "
            "transformer architectures like LayoutLMv3 and DocFormer incorporated spatial and textual features. "
            "More recently, generative VLMs have shown promising results on end-to-end document parsing tasks."
        ),
        "confidence": 0.93,
    }); idx += 1

    # Footer
    blocks.append({
        "id": idx, "type": "pagefooter",
        "bbox": [margin, int(img_h * 0.95), img_w - margin, int(img_h * 0.98)],
        "text": "Page 1 of 12  |  Preprint – Under Review",
        "confidence": 0.91,
    }); idx += 1

    return blocks


# ---------------------------------------------------------------------------
# Drawing overlays on image
# ---------------------------------------------------------------------------
def draw_overlays(img: "Image.Image", blocks: list) -> "Image.Image":
    """Draw bounding boxes and labels on the image."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    # Try to get a reasonable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

    for block in blocks:
        color_hex, _ = get_type_info(block["type"])
        # Parse hex to RGB
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)

        x0, y0, x1, y1 = block["bbox"]

        # Filled rectangle with transparency
        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 30), outline=(r, g, b, 200), width=2)

        # Label background
        label = block["type"].upper()
        text_bbox = font.getbbox(label)
        tw = text_bbox[2] - text_bbox[0] + 10
        th = text_bbox[3] - text_bbox[1] + 6
        label_y = max(y0 - th - 2, 0)
        draw.rectangle([x0, label_y, x0 + tw, label_y + th], fill=(r, g, b, 210))
        draw.text((x0 + 5, label_y + 2), label, fill=(255, 255, 255, 255), font=font)

    return overlay


# ---------------------------------------------------------------------------
# Block card renderer
# ---------------------------------------------------------------------------
def render_block_card(block: dict, original_img: "Image.Image | None"):
    """Render a styled card for a single detected block."""
    btype = block["type"]
    _, css_class = get_type_info(btype)
    label = btype.upper()
    conf = block.get("confidence", 0)
    text = block.get("text", "")

    card_html = f"""
    <div class="block-card">
        <span class="block-label {css_class}">{label}</span>
        <span class="block-conf" style="float:right;">id:{block['id']}  conf:{conf:.0%}</span>
    """

    if btype in ("figure", "image") and original_img is not None:
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)
        # Crop and display the figure region
        x0, y0, x1, y1 = block["bbox"]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(original_img.width, x1); y1 = min(original_img.height, y1)
        if x1 > x0 and y1 > y0:
            cropped = original_img.crop((x0, y0, x1, y1))
            st.image(cropped, use_container_width=True, caption=text[:100] if text else "Figure region")
    else:
        display_text = text if text else "(no text extracted)"
        card_html += f'<p class="block-text">{display_text}</p>'
        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Blocks → Markdown converter
# ---------------------------------------------------------------------------
def blocks_to_markdown(blocks: list) -> str:
    lines = []
    for b in blocks:
        btype = b["type"].lower()
        text = b.get("text", "").strip()
        if not text:
            continue
        if btype in ("title",):
            lines.append(f"# {text}\n")
        elif btype in ("sectionheader", "header"):
            lines.append(f"## {text}\n")
        elif btype in ("pageheader",):
            lines.append(f"*{text}*\n")
        elif btype in ("pagefooter", "footer"):
            lines.append(f"---\n*{text}*\n")
        elif btype == "figure":
            lines.append(f"![{text}](figure)\n")
        elif btype == "table":
            lines.append(f"```\n{text}\n```\n")
        elif btype == "equation":
            lines.append(f"$$\n{text}\n$$\n")
        else:
            lines.append(f"{text}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="hero">
        <span class="hero-icon">📄</span>
        <h1>DocParse Playground</h1>
    </div>
    <p class="hero-sub">Powered by PaddleOCR-VL-1.5 &nbsp;·&nbsp; Layout Analysis</p>
    """, unsafe_allow_html=True)

    st.divider()
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Upload a single-page or multi-page PDF. The first page will be analyzed.",
    )

    st.divider()
    st.markdown("##### ⚙️ Settings")
    dpi = st.slider("Render DPI", 100, 400, 200, step=50, help="Higher DPI = sharper image but slower")
    lang = st.selectbox("OCR Language", ["en", "ch"], index=0)
    show_overlays = st.checkbox("Show bounding boxes", value=True)

    page_num = 0
    if uploaded_file is not None and HAS_FITZ:
        pdf_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = doc.page_count
        doc.close()
        if total_pages > 1:
            page_num = st.number_input("Page number", 0, total_pages - 1, 0, help="0-indexed page")

    st.divider()
    mode_label = "🟢 PaddleOCR-VL-1.5" if HAS_PADDLE else "🟡 Mock Mode"
    st.markdown(f"**Engine:** {mode_label}")
    if not HAS_PADDLE:
        st.caption("PaddleOCR not installed — using mock data for UI preview. Install `paddlepaddle-gpu` and `paddleocr` for real inference.")

    st.divider()
    st.markdown(
        '<p style="font-size:11px;color:#8b949e;text-align:center;">'
        'Built with Streamlit · PaddleOCR-VL-1.5<br>'
        '<a href="https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5" target="_blank" '
        'style="color:#58a6ff;">Model Card ↗</a></p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📄</div>
        <h3>No document uploaded</h3>
        <p>Upload a PDF in the sidebar to get started. The layout analyzer will detect headers, text blocks, figures, tables, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # Show capability legend
    st.markdown("---")
    st.markdown("##### Detected Region Types")
    legend_html = '<div class="legend">'
    for name, (color, _) in TYPE_COLORS.items():
        legend_html += f'<div class="legend-item"><span class="legend-dot" style="background:{color};"></span>{name.upper()}</div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

    st.stop()

# ---------- Process the PDF ----------
if not HAS_FITZ or not HAS_PIL:
    st.error("Missing required libraries: `pymupdf` and/or `Pillow`. Please install them.")
    st.stop()

pdf_bytes = uploaded_file.read()
uploaded_file.seek(0)

with st.spinner("Converting PDF page to image..."):
    page_img = pdf_page_to_image(pdf_bytes, page_num=page_num, dpi=dpi)

# ---------- Run inference / mock ----------
cache_key = hashlib.md5(pdf_bytes).hexdigest() + f"_p{page_num}_d{dpi}_l{lang}"

if cache_key not in st.session_state:
    if HAS_PADDLE:
        with st.spinner("Running PaddleOCR-VL-1.5..."):
            pipeline = st.session_state.get("_paddle_pipeline") or load_paddleocr_vl()
            blocks = run_paddleocr_vl(pipeline, page_img)
    else:
        blocks = generate_mock_blocks(page_img.width, page_img.height)
    st.session_state[cache_key] = blocks
else:
    blocks = st.session_state[cache_key]

# ---------- Summary stats ----------
type_counts: dict[str, int] = {}
for b in blocks:
    t = b["type"]
    type_counts[t] = type_counts.get(t, 0) + 1

stats_html = '<div class="stat-row">'
stats_html += f'<span class="stat-pill"><b>{len(blocks)}</b>&nbsp;blocks</span>'
for t, count in sorted(type_counts.items()):
    color, _ = get_type_info(t)
    stats_html += f'<span class="stat-pill" style="border-color:{color}40;"><b style="color:{color};">{count}</b>&nbsp;{t}</span>'
stats_html += '</div>'
st.markdown(stats_html, unsafe_allow_html=True)

# ---------- Two-column layout ----------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("##### 🖼️ Document View")
    if show_overlays and blocks:
        annotated = draw_overlays(page_img, blocks)
        st.image(annotated, use_container_width=True)
    else:
        st.image(page_img, use_container_width=True)

    # Legend
    legend_html = '<div class="legend">'
    for t in type_counts:
        color, _ = get_type_info(t)
        legend_html += f'<div class="legend-item"><span class="legend-dot" style="background:{color};"></span>{t.upper()}</div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)

with col_right:
    st.markdown("##### 📋 Extraction Results")
    tab_blocks, tab_json, tab_md = st.tabs(["Blocks", "JSON", "Markdown"])

    with tab_blocks:
        for block in blocks:
            render_block_card(block, page_img)

    with tab_json:
        json_str = json.dumps(blocks, indent=2, ensure_ascii=False)
        st.markdown(f'<div class="json-wrap"><pre><code>{json_str}</code></pre></div>', unsafe_allow_html=True)

    with tab_md:
        md_content = blocks_to_markdown(blocks)
        st.markdown(f'<div class="md-result">', unsafe_allow_html=True)
        st.markdown(md_content)
        st.markdown('</div>', unsafe_allow_html=True)
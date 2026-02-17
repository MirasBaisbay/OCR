# 📄 DocParse Playground

A Streamlit-based document parsing interface powered by [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) and [PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_list.md). Upload any PDF and get instant layout analysis with bounding-box overlays, extracted text blocks, structured JSON output, and Markdown conversion.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Application Components](#application-components)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Mock Mode](#mock-mode)
- [Known Issues & Workarounds](#known-issues--workarounds)
- [Project Structure](#project-structure)
- [Links & References](#links--references)
- [License](#license)

---

## Features

- **PDF upload & multi-page navigation** — Upload any PDF; navigate pages with a page selector for multi-page documents.
- **GPU-accelerated OCR** — Uses PaddleOCR-VL-1.5 (0.9B parameter VLM) for text extraction and PP-DocLayoutV3 for layout detection.
- **Visual bounding-box overlays** — Color-coded rectangles drawn over detected regions (titles, text, figures, tables, equations, etc.) with labeled tags.
- **Three output views** — Switch between styled Block cards, raw JSON, and generated Markdown.
- **Mock mode** — Runs without PaddlePaddle installed using synthetic data, so you can develop and preview the UI anywhere.
- **Configurable settings** — Adjust render DPI (100–400), OCR language (English/Chinese), and toggle bounding-box visibility.
- **Session-state caching** — Inference results are cached by content hash + settings, so re-rendering the same page is instant.

---

## Architecture Overview

```
┌──────────────┐      ┌───────────────┐      ┌──────────────────┐
│  PDF Upload  │─────▶│  PyMuPDF      │─────▶│  PIL Image       │
│  (Streamlit) │      │  (fitz)       │      │  (page render)   │
└──────────────┘      └───────────────┘      └────────┬─────────┘
                                                      │
                                                      ▼
                                            ┌──────────────────┐
                                            │  PaddleOCR-VL    │
                                            │  pipeline        │
                                            │                  │
                                            │  ┌────────────┐  │
                                            │  │PP-DocLayout│  │ ← Layout detection
                                            │  │    V3      │  │   (boxes, labels, scores)
                                            │  └────────────┘  │
                                            │  ┌────────────┐  │
                                            │  │PaddleOCR-VL│  │ ← VLM text extraction
                                            │  │   1.5      │  │   (content per region)
                                            │  └────────────┘  │
                                            └────────┬─────────┘
                                                     │
                                                     ▼
                                           ┌───────────────────┐
                                           │  Normalized Blocks │
                                           │  [{id, type, bbox, │
                                           │    text, confidence}]│
                                           └────────┬──────────┘
                                                    │
                              ┌──────────────┬──────┴───────┬──────────────┐
                              ▼              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │ Overlay  │  │  Block   │  │   JSON   │  │ Markdown │
                        │ Drawing  │  │  Cards   │  │  Export  │  │ Convert  │
                        └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## Prerequisites

- **Python 3.10+** (tested on 3.10 and 3.11)
- **NVIDIA GPU** with CUDA 11.8+ (for GPU inference; CPU fallback is possible but slow)
- **CUDA Toolkit 11.8** and **cuDNN 8.9**

---

## Installation

### 1. Create a Conda Environment

```bash
conda create -n paddleocr python=3.10 -y
conda activate paddleocr
```

### 2. Install PaddlePaddle (GPU)

Install the CUDA 11.8 GPU build of PaddlePaddle:

```bash
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

> **CPU-only alternative:** `pip install paddlepaddle` (inference will be significantly slower).

Verify the installation:

```bash
python -c "import paddle; paddle.utils.run_check()"
```

### 3. Install PaddleOCR

```bash
pip install paddleocr>=2.10.0
```

This pulls in `paddlex` and the `PaddleOCRVL` pipeline automatically.

### 4. Install Application Dependencies

```bash
pip install streamlit pymupdf Pillow numpy
```

### 5. (First Run) Model Downloads

On first launch, PaddleOCR will automatically download:
- **PP-DocLayoutV3** (~50 MB) — layout detection model
- **PaddleOCR-VL-1.5-0.9B** (~1.8 GB) — vision-language model for text extraction

Models are cached at `~/.paddlex/official_models/` and reused on subsequent runs.

---

## Usage

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**Quick start:**

1. Click **Browse Files** in the sidebar and upload a PDF
2. Wait for inference (first run loads models; subsequent pages are faster)
3. Explore the three output tabs: **Blocks**, **JSON**, **Markdown**
4. Toggle **Show bounding boxes** to see/hide the visual overlay
5. Use the **Page number** selector for multi-page PDFs

---

## Application Components

### PDF-to-Image Conversion (`pdf_page_to_image`)

Converts a specific PDF page to a PIL Image using PyMuPDF. Render resolution is controlled by the DPI slider (default 200). Higher DPI produces sharper images but increases processing time and VRAM usage.

### PaddleOCR-VL Pipeline (`load_paddleocr_vl` / `run_paddleocr_vl`)

The core inference engine. The pipeline runs in two stages:

1. **PP-DocLayoutV3** performs layout detection, producing bounding boxes with class labels (e.g., `doc_title`, `text`, `image`, `table`) and confidence scores.
2. **PaddleOCR-VL-1.5** runs on each detected region as a vision-language model, extracting text content.

The `run_paddleocr_vl` function normalizes the raw output into a flat list of block dicts:

```python
{
    "id": 0,                    # Sequential index
    "type": "title",            # Normalized type (see label mapping below)
    "bbox": [130, 35, 1384, 98], # [x0, y0, x1, y1] pixel coordinates
    "text": "Extracted text...", # Content from VLM
    "confidence": 0.93          # Detection confidence score
}
```

**Label mapping** — PaddleOCR's raw labels are mapped to the UI's type system:

| PaddleOCR Label | UI Type | Color |
|---|---|---|
| `doc_title` | `title` | Orange |
| `text`, `paragraph`, `abstract` | `text` | Blue |
| `image`, `figure`, `seal` | `figure` | Purple |
| `table` | `table` | Teal |
| `header`, `page_header` | `pageheader` | Pink |
| `footer`, `page_footer` | `pagefooter` | Green |
| `section_header` | `sectionheader` | Pink |
| `equation`, `formula` | `equation` | Brown |
| `reference` | `reference` | Steel blue |
| `list` | `list` | Cyan |

### Bounding-Box Overlay (`draw_overlays`)

Draws semi-transparent color-coded rectangles over the original page image using PIL's `ImageDraw` with RGBA mode. Each box gets a small label tag positioned above its top edge.

### Block Card Renderer (`render_block_card`)

Renders each detected block as a styled HTML card with:
- A colored type badge (e.g., `TITLE`, `TEXT`, `FIGURE`)
- The block ID and confidence score
- Extracted text content (or a cropped image region for figure/image types)

### Markdown Converter (`blocks_to_markdown`)

Converts the block list into a Markdown string using type-aware formatting:
- **Titles** → `# heading`
- **Section headers** → `## heading`
- **Page headers** → `*italic*`
- **Tables** → fenced code blocks
- **Equations** → `$$` math blocks
- **Figures** → `![alt](figure)`
- **Text** → plain paragraphs

### Mock Data Generator (`generate_mock_blocks`)

When PaddlePaddle is not installed, the app generates realistic synthetic blocks that mimic a research paper layout. This allows full UI development and testing without a GPU.

---

## How It Works

### Data Flow

1. **Upload** — User uploads a PDF via the Streamlit sidebar file uploader.
2. **Render** — PyMuPDF converts the selected page to a PIL Image at the configured DPI.
3. **Cache check** — An MD5 hash of the PDF bytes + page number + DPI + language forms a cache key. If results exist in `st.session_state`, inference is skipped.
4. **Inference** — The image is saved to a temporary PNG file and passed to `pipeline.predict()`. The raw output contains:
   - `layout_det_res` — A dict with a `boxes` list. Each box is a dict: `{cls_id, label, score, coordinate, order, polygon_points}`.
   - `parsing_res_list` — A list of `PaddleOCRVLBlock` objects. Each has `.bbox`, `.content`, `.label`, `.image`, `.polygon_points`.
5. **Normalization** — The function iterates `parsing_res_list` (primary source of text + bboxes) and cross-references `layout_det_res.boxes` for confidence scores.
6. **Display** — The two-column layout shows the annotated image on the left and tabbed results (Blocks / JSON / Markdown) on the right.

### PaddleOCR-VL Output Structure

```
res (dict)
├── input_path, page_index, page_count, width, height
├── layout_det_res (dict)
│   ├── input_img (ndarray)
│   └── boxes (list of dict)
│       └── {cls_id, label, score, coordinate: [x0,y0,x1,y1], order, polygon_points}
├── parsing_res_list (list of PaddleOCRVLBlock)
│   └── .bbox [x0,y0,x1,y1]
│       .content "extracted text"
│       .label "doc_title" | "text" | "image" | ...
│       .image {path, img} | None
│       .polygon_points (ndarray)
├── table_res_list (list)
├── spotting_res (dict)
├── imgs_in_doc (list)
└── model_settings (dict)
```

---

## Configuration

All settings are available in the sidebar:

| Setting | Default | Range | Description |
|---|---|---|---|
| **Render DPI** | 200 | 100–400 | Resolution for PDF-to-image conversion. Higher = sharper but slower. |
| **OCR Language** | `en` | `en`, `ch` | Language hint passed to the pipeline. |
| **Show bounding boxes** | ✅ On | On/Off | Toggle the visual overlay on the document image. |
| **Page number** | 0 | 0 to N-1 | 0-indexed page selector (appears for multi-page PDFs). |

---

## Mock Mode

When PaddlePaddle is not installed, the app automatically falls back to **Mock Mode** (indicated by 🟡 in the sidebar). Mock mode generates a synthetic layout mimicking a research paper with:

- Page header, title, section headers
- Text paragraphs with realistic content
- A figure region
- A page footer

This is useful for UI development, theming, and testing on machines without a GPU.

---

## Known Issues & Workarounds

### PaddlePaddle Static Graph Mode Error

**Symptom:** `RuntimeError: int(Tensor) is not supported in static graph mode` when switching pages.

**Cause:** PaddlePaddle's internal state can switch to static graph mode after the first inference, breaking `int(Tensor)` calls on subsequent predictions.

**Workaround (implemented):** The app calls `paddle.disable_static()` before every prediction. If the error still occurs, the pipeline is automatically destroyed and recreated with a fresh instance. The new pipeline is stored in `st.session_state` for subsequent calls.

### Model Source Connectivity Check

**Symptom:** Slow startup while "Checking connectivity to model hosters."

**Workaround (implemented):** The environment variable `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` is set to `True` to bypass this check.

### CUDNN Version Warning

**Symptom:** `WARNING: The installed Paddle is compiled with CUDNN 8.9, but CUDNN version in your machine is 8.9`

**Impact:** This is a cosmetic false-positive warning. It does not affect functionality.

---

## Project Structure

```
paddle/
├── app.py              # Main Streamlit application (single-file)
├── README.md           # This file
└── debug_paddle.py     # Diagnostic script for inspecting PaddleOCR-VL output structure
```

### `app.py` Internal Structure

| Section | Lines | Description |
|---|---|---|
| Imports & conditional imports | 1–39 | Graceful degradation when heavy libs are missing |
| Custom CSS | 54–270 | Dark theme with Google Fonts (DM Sans, JetBrains Mono) |
| Constants & helpers | 272–296 | `TYPE_COLORS` mapping and `get_type_info()` |
| `pdf_page_to_image()` | 301–309 | PDF page → PIL Image via PyMuPDF |
| `load_paddleocr_vl()` | 315–321 | Cached pipeline loader with `@st.cache_resource` |
| `run_paddleocr_vl()` | 332–430 | Inference + output normalization |
| `generate_mock_blocks()` | 435–470 | Synthetic block generator for mock mode |
| `draw_overlays()` | 476–605 | Bounding-box visualization with PIL |
| `render_block_card()` | 611–640 | HTML card renderer for individual blocks |
| `blocks_to_markdown()` | 645–668 | Block list → Markdown converter |
| Sidebar UI | 670–719 | File upload, settings, engine status |
| Main content | 722–819 | Two-column layout, tabs, rendering logic |

---

## Links & References

### Models

| Model | Description | Link |
|---|---|---|
| **PaddleOCR-VL-1.5** | 0.9B vision-language model for document text extraction | [Hugging Face](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) |
| **PP-DocLayoutV3** | Layout detection model (document regions) | [GitHub](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_list.md) |

### Libraries

| Library | Purpose | Link |
|---|---|---|
| **PaddlePaddle** | Deep learning framework | [paddlepaddle.org.cn](https://www.paddlepaddle.org.cn/en) |
| **PaddleOCR** | OCR toolkit with VL pipeline | [GitHub](https://github.com/PaddlePaddle/PaddleOCR) · [PyPI](https://pypi.org/project/paddleocr/) |
| **Streamlit** | Web UI framework | [streamlit.io](https://streamlit.io/) · [Docs](https://docs.streamlit.io/) |
| **PyMuPDF (fitz)** | PDF rendering and manipulation | [PyPI](https://pypi.org/project/PyMuPDF/) · [Docs](https://pymupdf.readthedocs.io/) |
| **Pillow (PIL)** | Image processing and drawing | [PyPI](https://pypi.org/project/Pillow/) |
| **NumPy** | Array operations | [numpy.org](https://numpy.org/) |

### Documentation

- [PaddleOCR-VL Quick Start](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/paddleocr_vl.html)
- [PaddleX Pipeline API Reference](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/pipeline_develop_guide.html)
- [PP-DocLayout Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/ppocr/model_list.md)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)

---

## License

This project is provided as-is for research and educational purposes.

- **PaddleOCR** is licensed under the [Apache 2.0 License](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).
- **PaddlePaddle** is licensed under the [Apache 2.0 License](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE).
- **Streamlit** is licensed under the [Apache 2.0 License](https://github.com/streamlit/streamlit/blob/develop/LICENSE).
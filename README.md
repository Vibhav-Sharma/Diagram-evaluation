# Flowchart Text Understanding Engine

Clean rebuild of **text-only** understanding for messy student flowchart images.

This pipeline does **not** extract graphs, arrows, topology, or grading — only semantic text nodes.

## Architecture

```
Image
  → Global PaddleOCR (full image, polygons)
  → OCR normalization
  → Role pre-classification (NODE_TEXT | CONNECTOR_LABEL | NOISE)
  → Barrier mask (structural ink walls)
  → Local semantic fusion (multiline phrases, LoS-validated)
  → Conservative LoS consolidation
  → Final semantic nodes
```

### Design principles

- **Global OCR-first** — not contour-first crops
- **Local fusion before global separation** — fuse `BUY` / `CHEAP` / `FOOD` before splitting on separators
- **Line-of-Sight (LoS)** — two groups merge only if no heavy ink blocks the segment between centers
- **Under-merge over over-merge** — LLMs can join split nodes later; bad merges destroy meaning
- **Strict typed objects** — no raw numpy or Paddle tuples in semantic stages

## Project layout

```
text_engine/
  models.py           # OCRBlock, TextGroup, SemanticNode, PipelineResult
  ocr.py              # Global PaddleOCR + normalization
  role_classifier.py  # NODE_TEXT / CONNECTOR_LABEL / NOISE
  barrier_mask.py     # Ink-wall mask (OpenCV)
  los.py              # Line-of-sight validation
  grouping.py         # Local fusion + final nodes
  pipeline.py         # Orchestration
  visualize.py        # Debug overlays
cli.py                # Command-line runner
requirements.txt
```

## Setup

### 1. Python environment

Python **3.9–3.11** recommended (PaddlePaddle wheel support varies by platform).

```powershell
cd "c:\Users\vibha\internship\Diagram evaluation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows note:** If `paddlepaddle` fails to install, use the official CPU wheel from [PaddlePaddle install docs](https://www.paddlepaddle.org.cn/install/quick) for your Python version, then:

```powershell
pip install paddleocr opencv-python numpy
```

First run downloads OCR models (~100MB) into `~/.paddleocr/`.

### 2. Run on an image

```powershell
python cli.py path\to\flowchart.png -o output.json
```

Debug overlays are written by default to `path\to\debug\<image_stem>/`:

| File | Description |
|------|-------------|
| `01_ocr_polygons.png` | Raw OCR detections |
| `04_local_fusion.png` | After local semantic fusion |
| `05_barrier_mask.png` | Binary ink barrier mask |
| `05_barrier_overlay.png` | Barrier heatmap on image |
| `06_los_checks.png` | Green = merge allowed, red = blocked |
| `07_final_nodes.png` | Final semantic nodes |
| `result.json` | Full structured debug dump |

Skip debug output:

```powershell
python cli.py flowchart.png --no-debug
```

### 3. Python API

```python
from text_engine import run_text_pipeline

result = run_text_pipeline("flowchart.png")
for node in result.final_nodes:
    print(node.text, node.confidence)
```

Example output shape:

```json
[
  {
    "id": "node_0001",
    "text": "BUY CHEAP FOOD",
    "bbox": [120.0, 45.0, 310.0, 98.0],
    "center": {"x": 215.0, "y": 71.5},
    "confidence": 0.94,
    "role": "node_text",
    "source_group_ids": ["lf_0001"]
  }
]
```

## Tuning (iterative debugging)

Edit thresholds in:

- `text_engine/grouping.py` — spatial fusion gaps, overlap ratios, max chain length
- `text_engine/los.py` — `max_ink_ratio`, sample density
- `text_engine/barrier_mask.py` — morphology kernel sizes
- `text_engine/ocr.py` — `min_confidence`
- `text_engine/role_classifier.py` — connector token list

## What this does NOT do

- Graph / edge extraction
- Arrow direction
- Flow semantics or grading
- Topology parsing

Those belong in a separate downstream stage that consumes `final_nodes` JSON.

# Diagram-evaluation

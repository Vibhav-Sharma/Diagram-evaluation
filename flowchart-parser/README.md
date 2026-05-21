# Flowchart Parser — Visual Debug Workstation

Full-stack tool for tuning the **text-understanding engine** on messy student flowcharts.

- **Backend:** FastAPI + PaddleOCR pipeline (`backend/text_engine/`)
- **Frontend:** React + Vite + Tailwind + HTML Canvas overlays

## Folder structure

```
flowchart-parser/
├── backend/
│   ├── main.py              # FastAPI app (POST /parse, GET /health)
│   ├── requirements.txt
│   ├── api/schemas.py
│   └── text_engine/         # OCR → fusion → LoS → nodes
├── frontend/
│   ├── src/
│   │   ├── components/      # UploadPanel, DebugViewer, JsonViewer, …
│   │   ├── canvas/          # Canvas overlay renderers
│   │   └── config/stages.ts # Dynamic debug stages
│   └── package.json
└── README.md
```

## Prerequisites

- Python 3.9–3.11
- Node.js 18+
- PaddlePaddle + PaddleOCR (first run downloads models)

## Backend setup

```powershell
cd flowchart-parser\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `paddlepaddle` fails on Windows, install the CPU wheel from [PaddlePaddle docs](https://www.paddlepaddle.org.cn/install/quick), then `pip install paddleocr opencv-python fastapi uvicorn`.

### Run API

```powershell
cd flowchart-parser\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://127.0.0.1:8000/docs

### Windows OneDNN / PIR error

If parse fails with:

`ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<...>]`

This is a known PaddlePaddle 3.3+ CPU bug. The backend disables MKLDNN (`enable_mkldnn=False`). **Stop and restart uvicorn** so the OCR engine reloads.

Optional downgrade:

```powershell
pip install "paddlepaddle>=3.2.0,<3.3.0"
```

### POST /parse

Multipart form:

| Field | Type | Description |
|-------|------|-------------|
| `file` | image | PNG / JPEG flowchart |
| `config` | JSON string | Tuning sliders (optional) |

```json
{
  "min_ocr_confidence": 0.25,
  "ocr_padding": 3,
  "los_threshold": 0.12,
  "merge_threshold": 1.0,
  "barrier_sensitivity": 1.0
}
```

Response includes: `ocr_blocks`, `local_fusion_groups`, `merge_edges`, `barrier_mask_png_base64`, `final_nodes`, `logs`, `stats`.

## Frontend setup

```powershell
cd flowchart-parser\frontend
npm install
npm run dev
```

Open http://localhost:5173

Vite proxies `/api/*` → `http://127.0.0.1:8000` (see `vite.config.ts`).

### Production build

```powershell
npm run build
npm run preview
```

Set `VITE_API_URL=http://127.0.0.1:8000` if not using the dev proxy.

## Using the debug workstation

1. **Upload** — drag/drop or pick a flowchart (PNG/JPG).
2. **Tune** — sliders for OCR padding, LoS threshold, merge threshold, barrier sensitivity.
3. **Parse Flowchart** — runs `POST /parse`.
4. **Stage tabs** — inspect each pipeline stage on canvas:
   - Raw OCR / polygons / confidence heatmap
   - Local fusion groups (color-coded)
   - Barrier mask overlay
   - LoS checks (green = allowed, red = blocked)
   - Final semantic nodes
5. **JSON panel** — copy or download `flowchart_output.json`.

## Adding a new debug stage

Edit `frontend/src/config/stages.ts` only — add a stage with `layers: LayerId[]`. Implement new layer renderers in `frontend/src/canvas/renderers.ts` if needed.

## CLI (optional)

From repo root with `text_engine` on `PYTHONPATH`:

```powershell
cd flowchart-parser\backend
python -c "from text_engine.pipeline import run_text_pipeline; ..."
```

Or use the original `cli.py` in the parent `Diagram evaluation` folder.

## Design notes

- **Canvas overlays** — scales to container, devicePixelRatio-aware, no DOM box per detection.
- **Under-merge bias** — LoS + conservative fusion; tune via sliders and inspect blocked merges.
- **No graph/topology** — text understanding only; downstream systems consume `final_nodes`.

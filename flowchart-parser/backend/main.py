"""
FastAPI server for flowchart text-understanding debug workstation.
"""

from __future__ import annotations

# Configure Paddle/OneDNN BEFORE any paddleocr import (Windows CPU fix)
import text_engine.paddle_env  # noqa: F401

import json
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend root is on path
BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from api.schemas import HealthResponse, ParseConfig
from text_engine.config import PipelineConfig
from text_engine.pipeline import run_text_pipeline_from_bytes

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg"}
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="Flowchart Text Engine",
    description="Visual debugging API for messy student flowchart OCR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/parse")
async def parse_flowchart(
    file: UploadFile = File(...),
    config: str = Form(default="{}"),
):
    """
    Parse a flowchart image and return full debug payload.

    Multipart: `file` (image), optional `config` (JSON string).
    """
    content_type = (file.content_type or "").lower()
    suffix = Path(file.filename or "upload.png").suffix.lower()

    if content_type not in ALLOWED_TYPES and suffix not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use PNG or JPEG.",
        )

    try:
        cfg_dict = json.loads(config) if config else {}
        api_cfg = ParseConfig(**cfg_dict)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid config JSON: {exc}") from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    pipeline_cfg = PipelineConfig(
        min_ocr_confidence=api_cfg.min_ocr_confidence,
        ocr_padding=api_cfg.ocr_padding,
        los_threshold=api_cfg.los_threshold,
        merge_threshold=api_cfg.merge_threshold,
        barrier_sensitivity=api_cfg.barrier_sensitivity,
    )

    try:
        payload, logs = run_text_pipeline_from_bytes(
            raw,
            filename=file.filename or "upload.png",
            config=pipeline_cfg,
        )
        payload["logs"] = logs
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""Stage 1 — Global PaddleOCR on the entire image."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union

import numpy as np

import text_engine.paddle_env  # noqa: F401 — must run before paddle import
from text_engine.models import BBox, OCRBlock, Point

_OCR_ENGINE = None


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x — disable MKLDNN/OneDNN (Windows PIR crash workaround)
        _OCR_ENGINE = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            device="cpu",
        )
    return _OCR_ENGINE


def normalize_ocr_text(text: str) -> str:
    """Stage 2 — OCR normalization (whitespace / artifacts / punctuation)."""
    if not text:
        return ""
    t = text.replace("\n", " ").replace("\t", " ")
    t = re.sub(r"[^\S ]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = t.strip().upper()
    
    # Preserve specific connector labels
    connectors = {"YES", "NO", "Y", "N"}
    if t in connectors:
        return t
        
    # Strip leading/trailing punctuation (like periods, commas, colons)
    t = re.sub(r"^[\W_]+", "", t)
    t = re.sub(r"[\W_]+$", "", t)
    
    replacements = {
        " l ": " I ",
        " 0 ": " O ",
    }
    padded = f" {t} "
    for old, new in replacements.items():
        padded = padded.replace(old, new)
    return padded.strip()


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    import cv2

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return img
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def _polygon_from_box(box: Sequence[Sequence[float]]) -> List[Point]:
    return [Point(float(x), float(y)) for x, y in box]


def _center_from_polygon(polygon: List[Point]) -> Point:
    xs = [p.x for p in polygon]
    ys = [p.y for p in polygon]
    return Point(sum(xs) / len(xs), sum(ys) / len(ys))


def _poly_to_point_list(poly: Any) -> List[Point]:
    arr = np.asarray(poly)
    if arr.ndim == 1 and arr.size >= 4:
        # flat bbox fallback
        return []
    pts = arr.reshape(-1, 2)
    return [Point(float(x), float(y)) for x, y in pts]


def _parse_paddleocr_v3(result: Any) -> List[Tuple[List[Point], str, float]]:
    """PaddleOCR 3.x / PaddleX: OCRResult with rec_polys, rec_texts, rec_scores."""
    lines: List[Tuple[List[Point], str, float]] = []

    if not hasattr(result, "get"):
        return lines
    data = result

    polys = data.get("rec_polys") or data.get("dt_polys") or []
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or []

    for i, text in enumerate(texts):
        if i >= len(polys):
            break
        polygon = _poly_to_point_list(polys[i])
        if len(polygon) < 3:
            continue
        conf = float(scores[i]) if i < len(scores) else 1.0
        lines.append((polygon, str(text), conf))

    return lines


def _parse_paddleocr_v2(page: Any) -> List[Tuple[List[Point], str, float]]:
    """PaddleOCR 2.x: list of [box, (text, confidence)]."""
    lines: List[Tuple[List[Point], str, float]] = []
    if not page:
        return lines
    for line in page:
        if not line or len(line) < 2:
            continue
        box, rec = line[0], line[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            text, conf = rec[0], rec[1]
        else:
            text, conf = str(rec), 1.0
        polygon = _polygon_from_box(box)
        if len(polygon) < 3:
            continue
        lines.append((polygon, str(text), float(conf)))
    return lines


def _run_ocr_inference(engine: Any, img: np.ndarray) -> List[Tuple[List[Point], str, float]]:
    """Call predict/ocr and normalize output across PaddleOCR versions."""
    if hasattr(engine, "predict"):
        raw = engine.predict(img)
    else:
        raw = engine.ocr(img, cls=True)

    if not raw:
        return []

    first = raw[0]

    # v3: single OCRResult dict-like per image
    if isinstance(first, dict) or hasattr(first, "get") or hasattr(first, "json"):
        return _parse_paddleocr_v3(first)

    # v2: list of detections
    if isinstance(first, list):
        return _parse_paddleocr_v2(first)

    # v3 wrapped in extra list layer
    if isinstance(raw, list) and len(raw) == 1:
        return _parse_paddleocr_v3(raw[0])

    return []


def run_global_ocr(
    image: Union[str, Path, np.ndarray],
    min_confidence: float = 0.25,
) -> List[OCRBlock]:
    """
    Run PaddleOCR on the full image and return strictly typed OCRBlock list.

    Uses OCR polygons (quadrilateral), not axis-aligned boxes alone.
    """
    img = _load_image(image)
    engine = _get_ocr_engine()
    detections = _run_ocr_inference(engine, img)

    blocks: List[OCRBlock] = []
    idx = 0
    for polygon, text, conf_f in detections:
        if conf_f < min_confidence:
            continue
        normalized = normalize_ocr_text(text)
        if not normalized:
            continue
        bbox = BBox.from_polygon(polygon)
        center = _center_from_polygon(polygon)
        idx += 1
        blocks.append(
            OCRBlock(
                id=f"ocr_{idx:04d}",
                text=normalized,
                polygon=polygon,
                bbox=bbox,
                center=center,
                confidence=conf_f,
            )
        )
    return blocks

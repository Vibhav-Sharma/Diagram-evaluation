"""Stage 3 — Role pre-classification (NODE_TEXT, CONNECTOR_LABEL, NOISE)."""

from __future__ import annotations

import re
from typing import List, Set

from text_engine.models import OCRBlock, TextRole

# Preserved short tokens — never treated as noise.
CONNECTOR_TOKENS: Set[str] = {
    "YES",
    "NO",
    "Y",
    "N",
    "IF",
    "GO",
    "TRUE",
    "FALSE",
    "OK",
}

NOISE_PATTERNS = re.compile(r"^[^A-Z0-9?]+$")


def _is_connector_text(text: str) -> bool:
    t = text.strip().upper()
    if t in CONNECTOR_TOKENS:
        return True
    # Single-letter decision labels
    if len(t) == 1 and t in {"Y", "N"}:
        return True
    return False


def _median_block_height(blocks: List[OCRBlock]) -> float:
    heights = [b.bbox.height for b in blocks if b.bbox.height > 0]
    if not heights:
        return 20.0
    heights.sort()
    return heights[len(heights) // 2]


def classify_roles(blocks: List[OCRBlock]) -> List[OCRBlock]:
    """
    Tag each OCR block with TextRole.

    Connector labels: short, decision-like tokens (preserved, not deleted).
    Noise: empty symbols, extreme low confidence gibberish.
    """
    if not blocks:
        return blocks

    median_h = _median_block_height(blocks)
    areas = sorted(b.bbox.area for b in blocks if b.bbox.area > 0)
    median_area = areas[len(areas) // 2] if areas else 1.0

    for block in blocks:
        t = block.text.strip().upper()
        words = t.split()

        if _is_connector_text(t):
            block.role = TextRole.CONNECTOR_LABEL
            continue

        if NOISE_PATTERNS.match(t) or len(t) <= 1 and t not in CONNECTOR_TOKENS:
            if block.confidence < 0.4:
                block.role = TextRole.NOISE
                continue

        # Very small, low-confidence, non-word fragments
        if (
            len(words) == 1
            and len(t) <= 2
            and t not in CONNECTOR_TOKENS
            and block.confidence < 0.5
            and block.bbox.height < median_h * 0.45
        ):
            block.role = TextRole.NOISE
            continue

        # Tiny speck detections far below typical node area
        if block.bbox.area < median_area * 0.02 and block.confidence < 0.35:
            block.role = TextRole.NOISE
            continue

        block.role = TextRole.NODE_TEXT

    return blocks

"""Stage 5 — Binary barrier mask (ink walls: borders, arrows, separator lines)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from text_engine.models import OCRBlock


def generate_barrier_mask(
    image: Union[str, Path, np.ndarray],
    ocr_blocks: Optional[list[OCRBlock]] = None,
    text_clearance_px: int = 3,
) -> np.ndarray:
    """
    Build a uint8 mask where 255 = structural ink barrier, 0 = clear corridor.

    Pipeline:
    grayscale → adaptive threshold → morphological closing → dilation
    → optional text-hole punching so OCR regions are not treated as barriers.
    """
    import cv2

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        bgr = image.copy()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Invert: dark ink becomes bright for morphology
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Connect strokes (boxes, arrows, separator lines)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    barrier = cv2.dilate(closed, dilate_kernel, iterations=1)

    # Punch holes over OCR polygons so text itself is not a wall between lines
    if ocr_blocks:
        for block in ocr_blocks:
            pts = np.array([[p.x, p.y] for p in block.polygon], dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(barrier, [pts], 0)
                x0, y0, x1, y1 = [int(v) for v in block.bbox.as_list()]
                pad = text_clearance_px
                cv2.rectangle(
                    barrier,
                    (max(0, x0 - pad), max(0, y0 - pad)),
                    (x1 + pad, y1 + pad),
                    0,
                    thickness=-1,
                )

    return barrier


def save_barrier_mask(mask: np.ndarray, path: Union[str, Path]) -> str:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return str(path)


def mask_shape(mask: np.ndarray) -> Tuple[int, int]:
    h, w = mask.shape[:2]
    return h, w

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ComplexityMetrics:
    ocr_density: float
    ocr_overlap_ratio: float
    fragmentation_ratio: float
    barrier_density: float
    confidence_variance: float
    complexity_score: float
    merge_mode: str

def analyze_image_complexity(blocks: List[dict], image_shape: tuple, barrier_mask: Optional[np.ndarray]) -> ComplexityMetrics:
    if not blocks:
        return ComplexityMetrics(0, 0, 0, 0, 0, 0, "AGGRESSIVE")
        
    img_area = image_shape[0] * image_shape[1]
    
    # 1. OCR Density
    total_ocr_area = 0.0
    for b in blocks:
        if isinstance(b, dict) and "bbox" in b:
            total_ocr_area += (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
        elif hasattr(b, "bbox"):
            total_ocr_area += (b.bbox.x_max - b.bbox.x_min) * (b.bbox.y_max - b.bbox.y_min)

    ocr_density = min(1.0, total_ocr_area / (img_area + 1))
    
    # 2. Fragmentation
    fragmentation_ratio = min(1.0, len(blocks) / 50.0) # Heuristic: >50 blocks is highly fragmented
    
    # 3. Barrier Density
    barrier_density = 0.0
    if barrier_mask is not None:
        barrier_pixels = cv2.countNonZero(barrier_mask)
        barrier_density = min(1.0, barrier_pixels / (img_area + 1))
        
    # 4. Confidence Variance
    confs = [b.get("confidence", 0.0) if isinstance(b, dict) else b.confidence for b in blocks]
    confidence_variance = float(np.var(confs)) if len(confs) > 1 else 0.0
    
    # Compute overall complexity score [0, 1]
    # High fragmentation, high barrier density, and high confidence variance -> more messy
    score = (ocr_density * 0.2) + (fragmentation_ratio * 0.4) + (barrier_density * 0.3) + (confidence_variance * 0.1)
    complexity_score = min(1.0, max(0.0, score))
    
    mode = "AGGRESSIVE"
    if complexity_score > 0.6:
        mode = "CONSERVATIVE"
    elif complexity_score > 0.3:
        mode = "BALANCED"
        
    return ComplexityMetrics(
        ocr_density=ocr_density,
        ocr_overlap_ratio=0.0, # Placeholder
        fragmentation_ratio=fragmentation_ratio,
        barrier_density=barrier_density,
        confidence_variance=confidence_variance,
        complexity_score=complexity_score,
        merge_mode=mode
    )

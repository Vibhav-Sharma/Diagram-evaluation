"""End-to-end text understanding pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from text_engine.barrier_mask import generate_barrier_mask, save_barrier_mask
from text_engine.grouping import (
    _connector_groups,
    consolidate_with_los,
    groups_to_semantic_nodes,
    local_semantic_fusion,
)
from text_engine.models import PipelineResult
from text_engine.ocr import run_global_ocr
from text_engine.role_classifier import classify_roles
from text_engine import visualize


def run_text_pipeline(
    image_path: Union[str, Path],
    debug_dir: Optional[Union[str, Path]] = None,
    min_ocr_confidence: float = 0.25,
    save_debug: bool = True,
) -> PipelineResult:
    """
    Full rebuild pipeline:

    1. Global OCR
    2. OCR normalization (inside OCR stage)
    3. Role pre-classification
    4. Local semantic fusion
    5. Barrier mask generation
    6. LoS-validated consolidation
    7. Final semantic nodes
    """
    image_path = Path(image_path)
    if debug_dir is None and save_debug:
        debug_dir = image_path.parent / "debug" / image_path.stem
    debug_path = Path(debug_dir) if debug_dir else None

    # Stage 1 + 2
    ocr_blocks = run_global_ocr(image_path, min_confidence=min_ocr_confidence)

    # Stage 3
    ocr_blocks = classify_roles(ocr_blocks)

    # Stage 5 (before fusion so LoS + separator checks can use it)
    import cv2

    bgr = cv2.imread(str(image_path))
    barrier_mask = generate_barrier_mask(bgr, ocr_blocks=ocr_blocks)

    barrier_file: Optional[str] = None
    if debug_path and save_debug:
        debug_path.mkdir(parents=True, exist_ok=True)
        barrier_file = save_barrier_mask(
            barrier_mask, debug_path / "05_barrier_mask.png"
        )

    # Stage 4 — local fusion with LoS
    local_groups, fusion_edges = local_semantic_fusion(
        ocr_blocks,
        barrier_mask=barrier_mask,
        los_validate=True,
    )

    # Connector labels kept as separate groups
    connector_groups = _connector_groups(ocr_blocks)
    all_groups = local_groups + connector_groups

    # Stage 6 — conservative LoS consolidation on fused groups
    merged_groups, los_edges = consolidate_with_los(
        all_groups, barrier_mask, max_passes=1
    )
    all_merge_edges = fusion_edges + los_edges

    # Stage 7
    final_nodes = groups_to_semantic_nodes(merged_groups)

    result = PipelineResult(
        ocr_blocks=ocr_blocks,
        local_fusion_groups=local_groups,
        final_nodes=final_nodes,
        merge_edges=all_merge_edges,
        barrier_mask_path=barrier_file,
        debug_dir=str(debug_path) if debug_path else None,
    )

    if debug_path and save_debug:
        visualize.draw_ocr_blocks(
            image_path, ocr_blocks, debug_path / "01_ocr_polygons.png"
        )
        visualize.draw_barrier_overlay(
            image_path, barrier_mask, debug_path / "05_barrier_overlay.png"
        )
        visualize.draw_fusion_groups(
            image_path, local_groups, debug_path / "04_local_fusion.png"
        )
        visualize.draw_los_checks(
            image_path,
            merged_groups,
            all_merge_edges,
            barrier_mask,
            debug_path / "06_los_checks.png",
        )
        visualize.draw_final_nodes(
            image_path, final_nodes, debug_path / "07_final_nodes.png"
        )

        with open(debug_path / "result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ocr_blocks": [b.to_dict() for b in ocr_blocks],
                    "local_fusion_groups": [g.to_dict() for g in local_groups],
                    "final_nodes": result.nodes_as_dicts(),
                    "merge_edges": [
                        {
                            "group_a_id": e.group_a_id,
                            "group_b_id": e.group_b_id,
                            "reason": e.reason,
                            "allowed": e.allowed,
                            "ink_hits": e.ink_hits,
                            "ink_ratio": e.ink_ratio,
                        }
                        for e in all_merge_edges
                    ],
                },
                f,
                indent=2,
            )

    return result

"""End-to-end text understanding pipeline (API + CLI)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from text_engine.barrier_mask import generate_barrier_mask, save_barrier_mask
from text_engine.config import PipelineConfig
from text_engine.grouping import (
    _connector_groups,
    consolidate_with_los,
    groups_to_semantic_nodes,
    local_semantic_fusion,
)
from text_engine.models import PipelineResult, TextGroup
from text_engine.ocr import run_global_ocr
from text_engine.role_classifier import classify_roles
from text_engine.serialize import build_api_response
from text_engine import visualize


def run_text_pipeline(
    image_path: Union[str, Path],
    debug_dir: Optional[Union[str, Path]] = None,
    config: Optional[PipelineConfig] = None,
    save_debug: bool = True,
) -> PipelineResult:
    """Run pipeline; returns PipelineResult (use run_text_pipeline_api for JSON)."""
    cfg = config or PipelineConfig()
    image_path = Path(image_path)
    if debug_dir is None and save_debug:
        debug_dir = image_path.parent / "debug" / image_path.stem
    debug_path = Path(debug_dir) if debug_dir else None

    ocr_blocks = run_global_ocr(image_path, min_confidence=cfg.min_ocr_confidence)
    ocr_blocks = classify_roles(ocr_blocks)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    barrier_mask = generate_barrier_mask(
        bgr,
        ocr_blocks=ocr_blocks,
        text_clearance_px=cfg.ocr_padding,
        barrier_sensitivity=cfg.barrier_sensitivity,
    )

    barrier_file: Optional[str] = None
    if debug_path and save_debug:
        debug_path.mkdir(parents=True, exist_ok=True)
        barrier_file = save_barrier_mask(
            barrier_mask, debug_path / "05_barrier_mask.png"
        )

    local_groups, fusion_edges = local_semantic_fusion(
        ocr_blocks,
        barrier_mask=barrier_mask,
        los_validate=True,
        config=cfg,
    )

    connector_groups = _connector_groups(ocr_blocks)
    all_groups = local_groups + connector_groups

    merged_groups, los_edges = consolidate_with_los(
        all_groups, barrier_mask, max_passes=1, config=cfg
    )
    all_merge_edges = fusion_edges + los_edges
    final_nodes = groups_to_semantic_nodes(merged_groups)

    result = PipelineResult(
        ocr_blocks=ocr_blocks,
        local_fusion_groups=local_groups,
        final_nodes=final_nodes,
        merge_edges=all_merge_edges,
        barrier_mask_path=barrier_file,
        debug_dir=str(debug_path) if debug_path else None,
    )
    result._merged_groups = merged_groups  # type: ignore[attr-defined]
    result._barrier_mask = barrier_mask  # type: ignore[attr-defined]
    result._bgr_shape = (bgr.shape[1], bgr.shape[0])  # type: ignore[attr-defined]

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
                build_api_response(
                    result,
                    bgr.shape[1],
                    bgr.shape[0],
                    barrier_mask,
                    merged_groups,
                    cfg,
                    logs=[],
                ),
                f,
                indent=2,
            )

    return result


def run_text_pipeline_from_bytes(
    image_bytes: bytes,
    filename: str = "upload.png",
    config: Optional[PipelineConfig] = None,
) -> Tuple[dict, List[str]]:
    """Run pipeline from uploaded bytes; return API dict + logs."""
    cfg = config or PipelineConfig()
    logs: List[str] = []
    suffix = Path(filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    logs.append(f"Received image ({len(image_bytes)} bytes)")
    logs.append(f"Config: {cfg.to_dict()}")

    try:
        result = run_text_pipeline(tmp_path, config=cfg, save_debug=False)
        merged_groups: List[TextGroup] = getattr(result, "_merged_groups", [])
        barrier_mask: np.ndarray = getattr(result, "_barrier_mask")
        w, h = getattr(result, "_bgr_shape", (0, 0))

        logs.append(f"OCR detections: {len(result.ocr_blocks)}")
        logs.append(f"Local fusion groups: {len(result.local_fusion_groups)}")
        logs.append(f"Final semantic nodes: {len(result.final_nodes)}")
        logs.append(
            f"Merge attempts: {len(result.merge_edges)} "
            f"({sum(1 for e in result.merge_edges if e.allowed)} allowed, "
            f"{sum(1 for e in result.merge_edges if not e.allowed)} blocked)"
        )

        payload = build_api_response(
            result, w, h, barrier_mask, merged_groups, cfg, logs
        )
        return payload, logs
    finally:
        Path(tmp_path).unlink(missing_ok=True)

"""Serialize pipeline results for the FastAPI frontend."""

from __future__ import annotations

import base64
from typing import Any, List

import cv2
import numpy as np

from text_engine.config import PipelineConfig
from text_engine.models import MergeEdge, OCRBlock, PipelineResult, SemanticNode, TextGroup


def _encode_png_b64(array: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", array)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _merge_edge_dict(e: MergeEdge) -> dict[str, Any]:
    return {
        "group_a_id": e.group_a_id,
        "group_b_id": e.group_b_id,
        "reason": e.reason,
        "allowed": e.allowed,
        "ink_hits": e.ink_hits,
        "ink_ratio": e.ink_ratio,
    }


def _group_with_block_polygons(g: TextGroup) -> dict[str, Any]:
    d = g.to_dict()
    d["blocks"] = [b.to_dict() for b in g.blocks]
    return d


def build_api_response(
    result: PipelineResult,
    image_width: int,
    image_height: int,
    barrier_mask: np.ndarray,
    merged_groups: List[TextGroup],
    config: PipelineConfig,
    logs: List[str],
) -> dict[str, Any]:
    allowed = [e for e in result.merge_edges if e.allowed]
    blocked = [e for e in result.merge_edges if not e.allowed]

    id_to_center: dict[str, dict] = {}
    for g in merged_groups:
        id_to_center[g.id] = {"x": g.center.x, "y": g.center.y}
    for b in result.ocr_blocks:
        id_to_center[b.id] = {"x": b.center.x, "y": b.center.y}

    merge_edges_enriched = []
    for e in result.merge_edges:
        ed = _merge_edge_dict(e)
        ca = id_to_center.get(e.group_a_id)
        cb = id_to_center.get(e.group_b_id)
        if ca and cb:
            ed["center_a"] = ca
            ed["center_b"] = cb
        merge_edges_enriched.append(ed)

    return {
        "image": {"width": image_width, "height": image_height},
        "config": config.to_dict(),
        "logs": logs,
        "ocr_blocks": [b.to_dict() for b in result.ocr_blocks],
        "local_fusion_groups": [_group_with_block_polygons(g) for g in result.local_fusion_groups],
        "merged_groups": [_group_with_block_polygons(g) for g in merged_groups],
        "merge_edges": merge_edges_enriched,
        "allowed_merge_edges": [e for e in merge_edges_enriched if e["allowed"]],
        "blocked_merge_edges": [e for e in merge_edges_enriched if not e["allowed"]],
        "final_nodes": result.nodes_as_dicts(),
        "barrier_mask_png_base64": _encode_png_b64(barrier_mask),
        "stats": {
            "ocr_count": len(result.ocr_blocks),
            "fusion_group_count": len(result.local_fusion_groups),
            "final_node_count": len(result.final_nodes),
            "merge_attempts": len(result.merge_edges),
            "blocked_merges": len(blocked),
            "allowed_merges": len(allowed),
        },
    }

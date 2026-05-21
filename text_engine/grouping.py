"""Stage 4 & 7 — Local semantic fusion and final node generation."""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from text_engine.los import can_merge_groups
from text_engine.models import (
    BBox,
    MergeEdge,
    OCRBlock,
    Point,
    SemanticNode,
    TextGroup,
    TextRole,
)
from text_engine.role_classifier import CONNECTOR_TOKENS

# ---------------------------------------------------------------------------
# Spatial / alignment helpers
# ---------------------------------------------------------------------------


def _median(values: List[float], default: float = 1.0) -> float:
    if not values:
        return default
    s = sorted(values)
    return s[len(s) // 2]


def _blocks_to_groups(blocks: List[OCRBlock]) -> List[TextGroup]:
    groups: List[TextGroup] = []
    for i, b in enumerate(blocks):
        groups.append(
            TextGroup(
                id=f"g_{b.id}",
                blocks=[b],
                text=b.text,
                bbox=b.bbox,
                center=b.center,
                confidence=b.confidence,
                role=b.role,
            )
        )
    return groups


def _rebuild_group(group: TextGroup, reading_order: str = "top_left") -> TextGroup:
    """Merge block text and geometry into one group."""
    blocks = sorted(
        group.blocks,
        key=lambda b: (b.center.y, b.center.x)
        if reading_order == "top_left"
        else (b.center.x, b.center.y),
    )
    text = " ".join(b.text for b in blocks)
    text = re.sub(r"\s{2,}", " ", text).strip()

    all_points = [p for b in blocks for p in b.polygon]
    bbox = BBox.from_polygon(all_points) if all_points else blocks[0].bbox

    cx = sum(b.center.x for b in blocks) / len(blocks)
    cy = sum(b.center.y for b in blocks) / len(blocks)
    conf = sum(b.confidence for b in blocks) / len(blocks)

    return TextGroup(
        id=group.id,
        blocks=blocks,
        text=text,
        bbox=bbox,
        center=Point(cx, cy),
        confidence=conf,
        role=group.role,
    )


# ---------------------------------------------------------------------------
# Local fusion candidacy (geometry only — no separator aggression)
# ---------------------------------------------------------------------------


def _horizontal_gap(a: BBox, b: BBox) -> float:
    if a.x_max < b.x_min:
        return b.x_min - a.x_max
    if b.x_max < a.x_min:
        return a.x_min - b.x_max
    return 0.0


def _vertical_gap(a: BBox, b: BBox) -> float:
    if a.y_max < b.y_min:
        return b.y_min - a.y_max
    if b.y_max < a.y_min:
        return a.y_min - b.y_max
    return 0.0


def _x_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    denom = max(1.0, min(a.width, b.width))
    return overlap / denom


def _y_overlap_ratio(a: BBox, b: BBox) -> float:
    overlap = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    denom = max(1.0, min(a.height, b.height))
    return overlap / denom


def _has_immediate_separator_between(
    a: OCRBlock, b: OCRBlock, barrier_mask: Optional[np.ndarray]
) -> bool:
    """
    Detect a strong ink wall directly between two blocks (conservative).

    Only used to BLOCK fusion — never to split words early.
    """
    if barrier_mask is None:
        return False

    # Sample midpoint region between bboxes
    mid_x = (a.center.x + b.center.x) / 2
    mid_y = (a.center.y + b.center.y) / 2
    h_gap = _horizontal_gap(a.bbox, b.bbox)
    v_gap = _vertical_gap(a.bbox, b.bbox)

    import cv2

    h, w = barrier_mask.shape[:2]
    # Narrow probe strip perpendicular to stacking direction
    if v_gap >= h_gap:
        # vertically separated — check horizontal band between
        y0 = int(min(a.bbox.y_max, b.bbox.y_max))
        y1 = int(max(a.bbox.y_min, b.bbox.y_min))
        x0 = int(min(a.bbox.x_min, b.bbox.x_min) - 2)
        x1 = int(max(a.bbox.x_max, b.bbox.x_max) + 2)
    else:
        x0 = int(min(a.bbox.x_max, b.bbox.x_max))
        x1 = int(max(a.bbox.x_min, b.bbox.x_min))
        y0 = int(min(a.bbox.y_min, b.bbox.y_min) - 2)
        y1 = int(max(a.bbox.y_max, b.bbox.y_max) + 2)

    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return False

    roi = barrier_mask[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    ink_ratio = float(np.mean(roi >= 200))
    return ink_ratio > 0.35


def _semantically_plausible_merge(text_a: str, text_b: str) -> bool:
    """Lightweight plausibility — prefer under-merging."""
    a = text_a.strip().upper()
    b = text_b.strip().upper()
    if a in CONNECTOR_TOKENS or b in CONNECTOR_TOKENS:
        return False
    # Don't fuse two full questions
    if "?" in a and "?" in b:
        return False
    # Avoid merging very long unrelated phrases
    if len(a.split()) > 8 and len(b.split()) > 8:
        return False
    return True


def _local_fusion_candidate(
    a: OCRBlock,
    b: OCRBlock,
    median_h: float,
    median_w: float,
    barrier_mask: Optional[np.ndarray],
) -> Tuple[bool, str]:
    """
    Decide if two OCR blocks belong to the same visual node (local only).

    Rules: close, aligned, plausible, no immediate separator.
    """
    if a.role != TextRole.NODE_TEXT or b.role != TextRole.NODE_TEXT:
        return False, "role_mismatch"

    if not _semantically_plausible_merge(a.text, b.text):
        return False, "semantic_block"

    if _has_immediate_separator_between(a, b, barrier_mask):
        return False, "separator_between"

    h_gap = _horizontal_gap(a.bbox, b.bbox)
    v_gap = _vertical_gap(a.bbox, b.bbox)
    max_h_gap = median_w * 0.55
    max_v_gap = median_h * 1.35

    x_ov = _x_overlap_ratio(a.bbox, b.bbox)
    y_ov = _y_overlap_ratio(a.bbox, b.bbox)

    # Multiline stack (e.g. BUY / CHEAP / FOOD)
    if v_gap <= max_v_gap and x_ov >= 0.35:
        return True, "vertical_stack"

    # Horizontal continuation (e.g. WAIT FOR KETTLE / TO BOIL)
    if h_gap <= max_h_gap and y_ov >= 0.4:
        return True, "horizontal_continue"

    # Tight diagonal wrap in handwritten boxes
    dist = math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
    if dist <= median_h * 1.8 and (x_ov >= 0.2 or y_ov >= 0.2):
        if h_gap <= max_h_gap * 1.2 and v_gap <= max_v_gap * 1.2:
            return True, "tight_neighbor"

    return False, "spatial_reject"


# ---------------------------------------------------------------------------
# Union-find grouping
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _apply_union_find(
    items: List[OCRBlock],
    edges: List[Tuple[int, int]],
    max_component_size: int = 6,
) -> List[List[OCRBlock]]:
    uf = _UnionFind(len(items))
    sizes: Dict[int, int] = {i: 1 for i in range(len(items))}

    for i, j in edges:
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            continue
        if sizes[ri] + sizes[rj] > max_component_size:
            continue
        uf.union(i, j)
        root = uf.find(i)
        sizes[root] = sizes.get(ri, 1) + sizes.get(rj, 1)

    clusters: Dict[int, List[OCRBlock]] = {}
    for idx, block in enumerate(items):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(block)

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def local_semantic_fusion(
    blocks: List[OCRBlock],
    barrier_mask: Optional[np.ndarray] = None,
    los_validate: bool = True,
    max_chain: int = 6,
) -> Tuple[List[TextGroup], List[MergeEdge]]:
    """
    Stage 4 — Fuse multiline node text BEFORE global separation.

    Uses local neighborhoods + LoS when barrier mask is available.
    """
    node_blocks = [b for b in blocks if b.role == TextRole.NODE_TEXT]
    if not node_blocks:
        return [], []

    heights = [b.bbox.height for b in node_blocks]
    widths = [b.bbox.width for b in node_blocks]
    median_h = _median(heights, 20.0)
    median_w = _median(widths, 40.0)

    merge_edges: List[MergeEdge] = []
    uf_edges: List[Tuple[int, int]] = []

    n = len(node_blocks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = node_blocks[i], node_blocks[j]
            ok, reason = _local_fusion_candidate(
                a, b, median_h, median_w, barrier_mask=None
            )
            if not ok:
                continue

            if los_validate and barrier_mask is not None:
                ga = TextGroup(
                    id=a.id,
                    blocks=[a],
                    text=a.text,
                    bbox=a.bbox,
                    center=a.center,
                    confidence=a.confidence,
                )
                gb = TextGroup(
                    id=b.id,
                    blocks=[b],
                    text=b.text,
                    bbox=b.bbox,
                    center=b.center,
                    confidence=b.confidence,
                )
                edge = can_merge_groups(ga, gb, barrier_mask, edge_reason=reason)
                merge_edges.append(edge)
                if not edge.allowed:
                    continue
                if _has_immediate_separator_between(a, b, barrier_mask):
                    continue

            uf_edges.append((i, j))

    clusters = _apply_union_find(node_blocks, uf_edges, max_component_size=max_chain)
    groups: List[TextGroup] = []
    for idx, cluster in enumerate(clusters, start=1):
        g = TextGroup(
            id=f"lf_{idx:04d}",
            blocks=cluster,
            text="",
            bbox=cluster[0].bbox,
            center=cluster[0].center,
            confidence=cluster[0].confidence,
            role=TextRole.NODE_TEXT,
        )
        groups.append(_rebuild_group(g))

    return groups, merge_edges


def _connector_groups(blocks: List[OCRBlock]) -> List[TextGroup]:
    groups: List[TextGroup] = []
    for i, b in enumerate(blocks):
        if b.role == TextRole.CONNECTOR_LABEL:
            groups.append(
                TextGroup(
                    id=f"conn_{i:04d}",
                    blocks=[b],
                    text=b.text,
                    bbox=b.bbox,
                    center=b.center,
                    confidence=b.confidence,
                    role=TextRole.CONNECTOR_LABEL,
                )
            )
    return groups


def _group_center_distance(a: TextGroup, b: TextGroup) -> float:
    return math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)


def _should_attempt_global_merge(
    a: TextGroup, b: TextGroup, median_h: float
) -> Tuple[bool, str]:
    """Conservative second pass — only very tight remnants."""
    if a.role != TextRole.NODE_TEXT or b.role != TextRole.NODE_TEXT:
        return False, "role"
    if not _semantically_plausible_merge(a.text, b.text):
        return False, "semantic"
    dist = _group_center_distance(a, b)
    if dist > median_h * 3.0:
        return False, "distance"
    x_ov = _x_overlap_ratio(a.bbox, b.bbox)
    y_ov = _y_overlap_ratio(a.bbox, b.bbox)
    h_gap = _horizontal_gap(a.bbox, b.bbox)
    v_gap = _vertical_gap(a.bbox, b.bbox)
    if v_gap <= median_h * 1.2 and x_ov >= 0.4:
        return True, "global_vertical"
    if h_gap <= median_h * 0.8 and y_ov >= 0.45:
        return True, "global_horizontal"
    return False, "reject"


def consolidate_with_los(
    groups: List[TextGroup],
    barrier_mask: np.ndarray,
    max_passes: int = 1,
) -> Tuple[List[TextGroup], List[MergeEdge]]:
    """
    Optional conservative merge pass on already-fused groups using LoS.
    """
    if len(groups) < 2:
        return groups, []

    node_groups = [g for g in groups if g.role == TextRole.NODE_TEXT]
    others = [g for g in groups if g.role != TextRole.NODE_TEXT]
    if len(node_groups) < 2:
        return groups, []

    heights = [g.bbox.height for g in node_groups]
    median_h = _median(heights, 20.0)
    merge_edges: List[MergeEdge] = []
    changed = True
    passes = 0

    while changed and passes < max_passes:
        changed = False
        passes += 1
        n = len(node_groups)
        uf = _UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                ok, reason = _should_attempt_global_merge(
                    node_groups[i], node_groups[j], median_h
                )
                if not ok:
                    continue
                edge = can_merge_groups(
                    node_groups[i], node_groups[j], barrier_mask, edge_reason=reason
                )
                merge_edges.append(edge)
                if edge.allowed:
                    uf.union(i, j)

        clusters: Dict[int, List[TextGroup]] = {}
        for idx, g in enumerate(node_groups):
            root = uf.find(idx)
            clusters.setdefault(root, []).append(g)

        new_groups: List[TextGroup] = []
        for idx, cluster in enumerate(clusters.values(), start=1):
            if len(cluster) == 1:
                new_groups.append(cluster[0])
                continue
            blocks: List[OCRBlock] = []
            for g in cluster:
                blocks.extend(g.blocks)
            merged = TextGroup(
                id=f"mg_{idx:04d}",
                blocks=blocks,
                text="",
                bbox=cluster[0].bbox,
                center=cluster[0].center,
                confidence=cluster[0].confidence,
                role=TextRole.NODE_TEXT,
            )
            new_groups.append(_rebuild_group(merged))
            changed = True

        node_groups = new_groups

    return node_groups + others, merge_edges


def groups_to_semantic_nodes(groups: List[TextGroup]) -> List[SemanticNode]:
    """Stage 7 — Final semantic nodes."""
    nodes: List[SemanticNode] = []
    for i, g in enumerate(groups, start=1):
        nodes.append(
            SemanticNode(
                id=f"node_{i:04d}",
                text=g.text,
                bbox=g.bbox,
                center=g.center,
                confidence=g.confidence,
                role=g.role,
                source_group_ids=[g.id],
            )
        )
    return nodes

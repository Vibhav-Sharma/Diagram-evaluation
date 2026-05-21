"""Stage 6 — Line-of-Sight validation against barrier mask."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from text_engine.config import PipelineConfig
from text_engine.models import MergeEdge, Point, TextGroup


def _sample_line(
    p0: Point, p1: Point, num_samples: int
) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for i in range(num_samples):
        t = i / max(1, num_samples - 1)
        x = int(round(p0.x + (p1.x - p0.x) * t))
        y = int(round(p0.y + (p1.y - p0.y) * t))
        points.append((x, y))
    return points


def line_of_sight_clear(
    center_a: Point,
    center_b: Point,
    barrier_mask: np.ndarray,
    ink_threshold: int = 200,
    max_ink_ratio: float = 0.12,
    min_samples: int = 24,
) -> Tuple[bool, int, float]:
    """
    Return (clear, ink_hits, ink_ratio) for the segment between two centers.

    A merge is blocked when too many sampled pixels along the line hit heavy ink.
    """
    h, w = barrier_mask.shape[:2]
    dist = math.hypot(center_b.x - center_a.x, center_b.y - center_a.y)
    samples = max(min_samples, int(dist / 4))

    hits = 0
    total = 0
    for x, y in _sample_line(center_a, center_b, samples):
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        total += 1
        if int(barrier_mask[y, x]) >= ink_threshold:
            hits += 1

    if total == 0:
        return True, 0, 0.0

    ratio = hits / total
    clear = ratio <= max_ink_ratio
    return clear, hits, ratio


def can_merge_groups(
    group_a: TextGroup,
    group_b: TextGroup,
    barrier_mask: np.ndarray,
    edge_reason: str = "los_check",
    config: PipelineConfig | None = None,
) -> MergeEdge:
    cfg = config or PipelineConfig()
    clear, hits, ratio = line_of_sight_clear(
        group_a.center,
        group_b.center,
        barrier_mask,
        max_ink_ratio=cfg.los_threshold,
    )
    return MergeEdge(
        group_a_id=group_a.id,
        group_b_id=group_b.id,
        reason=edge_reason,
        allowed=clear,
        ink_hits=hits,
        ink_ratio=round(ratio, 4),
    )


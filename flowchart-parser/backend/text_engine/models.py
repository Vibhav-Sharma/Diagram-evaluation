"""Strict typed structures for the text-understanding pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box [x_min, y_min, x_max, y_max]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_list(self) -> List[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    @classmethod
    def from_polygon(cls, polygon: Sequence[Point]) -> "BBox":
        xs = [p.x for p in polygon]
        ys = [p.y for p in polygon]
        return cls(min(xs), min(ys), max(xs), max(ys))


class TextRole(str, Enum):
    NODE_TEXT = "node_text"
    CONNECTOR_LABEL = "connector_label"
    NOISE = "noise"


@dataclass
class OCRBlock:
    """Single PaddleOCR detection — never pass raw numpy into downstream stages."""

    id: str
    text: str
    polygon: List[Point]
    bbox: BBox
    center: Point
    confidence: float
    role: TextRole = TextRole.NODE_TEXT

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "polygon": [[p.x, p.y] for p in self.polygon],
            "bbox": self.bbox.as_list(),
            "center": {"x": self.center.x, "y": self.center.y},
            "confidence": round(self.confidence, 4),
            "role": self.role.value,
        }


@dataclass
class TextGroup:
    """Fused OCR blocks sharing one visual text node."""

    id: str
    blocks: List[OCRBlock]
    text: str
    bbox: BBox
    center: Point
    confidence: float
    semantic_confidence: float = 1.0
    geometry_confidence: float = 1.0
    overall_confidence: float = 1.0
    role: TextRole = TextRole.NODE_TEXT

    @property
    def block_ids(self) -> List[str]:
        return [b.id for b in self.blocks]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox": self.bbox.as_list(),
            "center": {"x": self.center.x, "y": self.center.y},
            "confidence": round(self.confidence, 4),
            "semantic_confidence": round(self.semantic_confidence, 4),
            "geometry_confidence": round(self.geometry_confidence, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "role": self.role.value,
            "block_ids": self.block_ids,
        }


@dataclass
class SemanticNode:
    """Final clean semantic text node."""

    id: str
    text: str
    bbox: BBox
    center: Point
    confidence: float
    semantic_confidence: float = 1.0
    geometry_confidence: float = 1.0
    overall_confidence: float = 1.0
    role: TextRole = TextRole.NODE_TEXT
    source_group_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "bbox": self.bbox.as_list(),
            "center": {"x": self.center.x, "y": self.center.y},
            "confidence": round(self.confidence, 4),
            "semantic_confidence": round(self.semantic_confidence, 4),
            "geometry_confidence": round(self.geometry_confidence, 4),
            "overall_confidence": round(self.overall_confidence, 4),
            "role": self.role.value,
            "source_group_ids": self.source_group_ids,
        }


@dataclass
class MergeEdge:
    """Candidate merge between two groups (for debugging)."""

    group_a_id: str
    group_b_id: str
    reason: str
    allowed: bool
    ink_hits: int = 0
    ink_ratio: float = 0.0
    semantic_confidence: float = 0.0
    contextual_similarity: float = 0.0
    semantic_reasoning: str = ""


@dataclass
class PipelineResult:
    """Full pipeline output with intermediate artifacts for debugging."""

    ocr_blocks: List[OCRBlock]
    local_fusion_groups: List[TextGroup]
    final_nodes: List[SemanticNode]
    merge_edges: List[MergeEdge] = field(default_factory=list)
    barrier_mask_path: Optional[str] = None
    debug_dir: Optional[str] = None

    def nodes_as_dicts(self) -> List[dict[str, Any]]:
        return [n.to_dict() for n in self.final_nodes]

"""Debug overlays — OCR polygons, fusion groups, LoS checks, final nodes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from text_engine.models import MergeEdge, OCRBlock, SemanticNode, TextGroup


def _load_bgr(image: Union[str, Path, np.ndarray]):
    import cv2

    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(str(image))
        return img.copy()
    return image.copy()


def _color(idx: int) -> Tuple[int, int, int]:
    palette = [
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (241, 196, 15),
        (231, 76, 60),
        (26, 188, 156),
        (230, 126, 34),
    ]
    return palette[idx % len(palette)]


def draw_ocr_blocks(
    image: Union[str, Path, np.ndarray],
    blocks: List[OCRBlock],
    out_path: Union[str, Path],
) -> str:
    import cv2

    canvas = _load_bgr(image)
    for i, b in enumerate(blocks):
        pts = np.array([[int(p.x), int(p.y)] for p in b.polygon], np.int32)
        color = _color(i)
        cv2.polylines(canvas, [pts], True, color, 2)
        cv2.putText(
            canvas,
            f"{b.id} {b.text[:24]}",
            (int(b.bbox.x_min), max(12, int(b.bbox.y_min) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return str(path)


def draw_fusion_groups(
    image: Union[str, Path, np.ndarray],
    groups: List[TextGroup],
    out_path: Union[str, Path],
) -> str:
    import cv2

    canvas = _load_bgr(image)
    for i, g in enumerate(groups):
        color = _color(i)
        for b in g.blocks:
            pts = np.array([[int(p.x), int(p.y)] for p in b.polygon], np.int32)
            cv2.polylines(canvas, [pts], True, color, 2)
        bb = g.bbox.as_list()
        cv2.rectangle(
            canvas,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            color,
            2,
        )
        cv2.putText(
            canvas,
            g.text[:40],
            (int(bb[0]), int(bb[3]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return str(path)


def draw_los_checks(
    image: Union[str, Path, np.ndarray],
    groups: List[TextGroup],
    merge_edges: List[MergeEdge],
    barrier_mask: Optional[np.ndarray],
    out_path: Union[str, Path],
) -> str:
    import cv2

    canvas = _load_bgr(image)
    edge_map = {(e.group_a_id, e.group_b_id): e for e in merge_edges}
    edge_map.update({(e.group_b_id, e.group_a_id): e for e in merge_edges})

    id_to_group = {g.id: g for g in groups}
    drawn: set = set()

    for (a_id, b_id), edge in edge_map.items():
        key = tuple(sorted((a_id, b_id)))
        if key in drawn:
            continue
        drawn.add(key)
        ga = id_to_group.get(a_id)
        gb = id_to_group.get(b_id)
        if not ga or not gb:
            continue
        p0 = (int(ga.center.x), int(ga.center.y))
        p1 = (int(gb.center.x), int(gb.center.y))
        color = (46, 204, 113) if edge.allowed else (0, 0, 255)
        cv2.line(canvas, p0, p1, color, 2)
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        label = "OK" if edge.allowed else "BLOCK"
        cv2.putText(
            canvas,
            label,
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return str(path)


def draw_final_nodes(
    image: Union[str, Path, np.ndarray],
    nodes: List[SemanticNode],
    out_path: Union[str, Path],
) -> str:
    import cv2

    canvas = _load_bgr(image)
    for i, n in enumerate(nodes):
        color = _color(i)
        bb = n.bbox.as_list()
        cv2.rectangle(
            canvas,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            color,
            3,
        )
        cv2.putText(
            canvas,
            n.text[:50],
            (int(bb[0]), max(16, int(bb[1]) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
    return str(path)


def draw_barrier_overlay(
    image: Union[str, Path, np.ndarray],
    barrier_mask: np.ndarray,
    out_path: Union[str, Path],
    alpha: float = 0.45,
) -> str:
    import cv2

    canvas = _load_bgr(image)
    heat = cv2.applyColorMap(barrier_mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(canvas, 1 - alpha, heat, alpha, 0)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)
    return str(path)

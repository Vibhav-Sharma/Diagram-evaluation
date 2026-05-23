import type { LayerId, MergeEdge, OCRBlock, ParseResponse, SemanticNode, TextGroup } from "../types";
import { colorForIndex, confidenceColor, hexToRgba } from "../utils/colors";

export interface RenderContext {
  ctx: CanvasRenderingContext2D;
  scale: number;
  data: ParseResponse;
  image: HTMLImageElement | null;
  barrierImage: HTMLImageElement | null;
  groupColorMap: Map<string, string>;
}

function drawImageLayer(rc: RenderContext) {
  if (!rc.image) return;
  const { ctx, scale, image } = rc;
  ctx.drawImage(image, 0, 0, image.naturalWidth * scale, image.naturalHeight * scale);
}

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  polygon: number[][],
  scale: number,
  stroke: string,
  fill?: string,
  lineWidth = 2
) {
  if (polygon.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(polygon[0][0] * scale, polygon[0][1] * scale);
  for (let i = 1; i < polygon.length; i++) {
    ctx.lineTo(polygon[i][0] * scale, polygon[i][1] * scale);
  }
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function drawBBox(
  ctx: CanvasRenderingContext2D,
  bbox: number[],
  scale: number,
  stroke: string,
  fill?: string,
  lineWidth = 2
) {
  const [x0, y0, x1, y1] = bbox;
  ctx.beginPath();
  ctx.rect(x0 * scale, y0 * scale, (x1 - x0) * scale, (y1 - y0) * scale);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  color: string,
  fontSize = 11
) {
  ctx.font = `600 ${fontSize}px JetBrains Mono, monospace`;
  ctx.fillStyle = "rgba(0,0,0,0.65)";
  const m = ctx.measureText(text);
  ctx.fillRect(x - 2, y - fontSize, m.width + 6, fontSize + 6);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

export function buildGroupColorMap(data: ParseResponse): Map<string, string> {
  const map = new Map<string, string>();
  data.local_fusion_groups.forEach((g, i) => map.set(g.id, colorForIndex(i)));
  data.merged_groups.forEach((g, i) => {
    if (!map.has(g.id)) map.set(g.id, colorForIndex(i + 20));
  });
  return map;
}

function renderOcrPolygons(rc: RenderContext) {
  rc.data.ocr_blocks.forEach((b: OCRBlock, i) => {
    const c = colorForIndex(i);
    drawPolygon(rc.ctx, b.polygon, rc.scale, c, hexToRgba(c, 0.12));
  });
}

function renderOcrLabels(rc: RenderContext) {
  rc.data.ocr_blocks.forEach((b: OCRBlock, i) => {
    const c = colorForIndex(i);
    drawPolygon(rc.ctx, b.polygon, rc.scale, c, hexToRgba(c, 0.08), 1.5);
    drawLabel(
      rc.ctx,
      `${b.text} (${(b.confidence * 100).toFixed(0)}%)`,
      b.bbox[0] * rc.scale,
      Math.max(14, b.bbox[1] * rc.scale - 4),
      c,
      10
    );
  });
}

function renderOcrConfidence(rc: RenderContext) {
  rc.data.ocr_blocks.forEach((b) => {
    const c = confidenceColor(b.confidence);
    drawPolygon(rc.ctx, b.polygon, rc.scale, c, hexToRgba(c, 0.25));
    drawLabel(
      rc.ctx,
      b.confidence.toFixed(2),
      b.center.x * rc.scale,
      b.center.y * rc.scale,
      "#fff",
      10
    );
  });
}

function renderFusionGroups(rc: RenderContext, groups: TextGroup[]) {
  groups.forEach((g) => {
    const c = rc.groupColorMap.get(g.id) ?? "#3b82f6";
    const blocks = g.blocks ?? rc.data.ocr_blocks.filter((b) => g.block_ids.includes(b.id));
    blocks.forEach((b) => {
      drawPolygon(rc.ctx, b.polygon, rc.scale, c, hexToRgba(c, 0.2), 2.5);
    });
    drawBBox(rc.ctx, g.bbox, rc.scale, c, undefined, 3);
    drawLabel(
      rc.ctx,
      g.text.slice(0, 48),
      g.bbox[0] * rc.scale + 4,
      g.bbox[3] * rc.scale + 14,
      c,
      12
    );
  });
}

function renderMergeEdges(rc: RenderContext, edges: MergeEdge[], allowed: boolean | null) {
  const idToCenter = new Map<string, { x: number; y: number }>();
  const register = (id: string, c?: { x: number; y: number }) => {
    if (c) idToCenter.set(id, c);
  };
  rc.data.merged_groups.forEach((g) => register(g.id, g.center));
  rc.data.local_fusion_groups.forEach((g) => register(g.id, g.center));
  rc.data.ocr_blocks.forEach((b) => register(b.id, b.center));

  edges.forEach((e) => {
    if (allowed !== null && e.allowed !== allowed) return;
    const ca = e.center_a ?? idToCenter.get(e.group_a_id);
    const cb = e.center_b ?? idToCenter.get(e.group_b_id);
    if (!ca || !cb) return;
    const color = e.allowed ? "#22c55e" : "#ef4444";
    rc.ctx.beginPath();
    rc.ctx.moveTo(ca.x * rc.scale, ca.y * rc.scale);
    rc.ctx.lineTo(cb.x * rc.scale, cb.y * rc.scale);
    rc.ctx.strokeStyle = color;
    rc.ctx.lineWidth = e.allowed ? 2.5 : 2;
    rc.ctx.setLineDash(e.allowed ? [] : [6, 4]);
    rc.ctx.stroke();
    rc.ctx.setLineDash([]);
    const mx = ((ca.x + cb.x) / 2) * rc.scale;
    const my = ((ca.y + cb.y) / 2) * rc.scale;
    const label = e.semantic_reasoning ? e.semantic_reasoning : (e.allowed ? "OK" : "BLOCK");
    drawLabel(
      rc.ctx,
      label,
      mx,
      my,
      color,
      9
    );
  });
}

function renderFinalNodes(rc: RenderContext) {
  rc.data.final_nodes.forEach((n: SemanticNode, i) => {
    const c = colorForIndex(i);
    drawBBox(rc.ctx, n.bbox, rc.scale, c, hexToRgba(c, 0.1), 3);
    drawLabel(
      rc.ctx,
      `${n.id}: ${n.text} (${(n.confidence * 100).toFixed(0)}%)`,
      n.bbox[0] * rc.scale + 4,
      Math.max(16, n.bbox[1] * rc.scale - 6),
      c,
      11
    );
  });
}

function renderBarrierMask(rc: RenderContext) {
  if (!rc.barrierImage) return;
  const { ctx, scale, barrierImage, image } = rc;
  const w = (image?.naturalWidth ?? barrierImage.naturalWidth) * scale;
  const h = (image?.naturalHeight ?? barrierImage.naturalHeight) * scale;
  if (rc.image) {
    ctx.globalAlpha = 0.35;
    ctx.drawImage(rc.image, 0, 0, w, h);
    ctx.globalAlpha = 1;
  }
  ctx.globalAlpha = 0.85;
  ctx.drawImage(barrierImage, 0, 0, w, h);
  ctx.globalAlpha = 1;
}

const LAYER_RENDERERS: Record<LayerId, (rc: RenderContext) => void> = {
  image: drawImageLayer,
  ocr_polygons: renderOcrPolygons,
  ocr_labels: renderOcrLabels,
  ocr_confidence: renderOcrConfidence,
  fusion_groups: (rc) => renderFusionGroups(rc, rc.data.local_fusion_groups),
  barrier_mask: renderBarrierMask,
  los_allowed: (rc) => renderMergeEdges(rc, rc.data.merge_edges, true),
  los_blocked: (rc) => renderMergeEdges(rc, rc.data.merge_edges, false),
  los_all: (rc) => renderMergeEdges(rc, rc.data.merge_edges, null),
  final_nodes: renderFinalNodes,
};

export function renderLayers(layers: LayerId[], rc: RenderContext) {
  const { ctx, image } = rc;

  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  if (!layers.includes("barrier_mask") || layers.length > 1) {
    // default dark backdrop when not mask-only
  }

  for (const layer of layers) {
    const fn = LAYER_RENDERERS[layer];
    if (fn) fn(rc);
  }

  if (layers.includes("image") && !image && layers.length === 1) {
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Inter, sans-serif";
    ctx.fillText("Upload an image to begin", 24, 40);
  }
}

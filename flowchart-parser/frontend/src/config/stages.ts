import type { DebugStage } from "../types";

/**
 * Dynamic stage registry — add stages here without changing render components.
 */
export const DEBUG_STAGES: DebugStage[] = [
  {
    id: "original",
    label: "Original",
    description: "Uploaded flowchart image",
    layers: ["image"],
  },
  {
    id: "raw_ocr",
    label: "Raw OCR",
    description: "Every detection with text + confidence labels",
    layers: ["image", "ocr_labels"],
  },
  {
    id: "ocr_polygons",
    label: "OCR Polygons",
    description: "Quadrilateral OCR regions color-coded",
    layers: ["image", "ocr_polygons"],
  },
  {
    id: "ocr_confidence",
    label: "OCR Confidence",
    description: "Polygon color mapped to confidence score",
    layers: ["image", "ocr_confidence"],
  },
  {
    id: "local_fusion",
    label: "Local Fusion",
    description: "OCR blocks fused into visual node groups",
    layers: ["image", "fusion_groups"],
  },
  {
    id: "barrier_mask",
    label: "Barrier Mask",
    description: "Structural ink walls (binary mask overlay)",
    layers: ["barrier_mask"],
  },
  {
    id: "los_checks",
    label: "LoS Checks",
    description: "All merge attempts: green allowed, red blocked",
    layers: ["image", "los_all"],
  },
  {
    id: "los_allowed",
    label: "Allowed Merges",
    description: "Successful line-of-sight merge paths",
    layers: ["image", "los_allowed"],
  },
  {
    id: "blocked_merges",
    label: "Blocked Merges",
    description: "Merges rejected by barrier / LoS",
    layers: ["image", "los_blocked"],
  },
  {
    id: "final_nodes",
    label: "Final Nodes",
    description: "Clean semantic text nodes",
    layers: ["image", "final_nodes"],
  },
];

export function getStageById(id: string): DebugStage | undefined {
  return DEBUG_STAGES.find((s) => s.id === id);
}

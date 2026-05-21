export interface Point {
  x: number;
  y: number;
}

export interface OCRBlock {
  id: string;
  text: string;
  polygon: number[][];
  bbox: number[];
  center: Point;
  confidence: number;
  role: string;
}

export interface TextGroup {
  id: string;
  text: string;
  bbox: number[];
  center: Point;
  confidence: number;
  role: string;
  block_ids: string[];
  blocks?: OCRBlock[];
}

export interface MergeEdge {
  group_a_id: string;
  group_b_id: string;
  reason: string;
  allowed: boolean;
  ink_hits: number;
  ink_ratio: number;
  center_a?: Point;
  center_b?: Point;
}

export interface SemanticNode {
  id: string;
  text: string;
  bbox: number[];
  center: Point;
  confidence: number;
  role: string;
  source_group_ids?: string[];
}

export interface ParseConfig {
  min_ocr_confidence: number;
  ocr_padding: number;
  los_threshold: number;
  merge_threshold: number;
  barrier_sensitivity: number;
}

export interface ParseStats {
  ocr_count: number;
  fusion_group_count: number;
  final_node_count: number;
  merge_attempts: number;
  blocked_merges: number;
  allowed_merges: number;
}

export interface ParseResponse {
  image: { width: number; height: number };
  config: ParseConfig;
  logs: string[];
  ocr_blocks: OCRBlock[];
  local_fusion_groups: TextGroup[];
  merged_groups: TextGroup[];
  merge_edges: MergeEdge[];
  allowed_merge_edges: MergeEdge[];
  blocked_merge_edges: MergeEdge[];
  final_nodes: SemanticNode[];
  barrier_mask_png_base64: string;
  stats: ParseStats;
}

export type LayerId =
  | "image"
  | "ocr_polygons"
  | "ocr_labels"
  | "ocr_confidence"
  | "fusion_groups"
  | "barrier_mask"
  | "los_allowed"
  | "los_blocked"
  | "los_all"
  | "final_nodes";

export interface DebugStage {
  id: string;
  label: string;
  description: string;
  layers: LayerId[];
}

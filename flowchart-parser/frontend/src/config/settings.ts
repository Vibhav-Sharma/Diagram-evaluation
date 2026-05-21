import type { ParseConfig } from "../types";

export const DEFAULT_CONFIG: ParseConfig = {
  min_ocr_confidence: 0.25,
  ocr_padding: 3,
  los_threshold: 0.12,
  merge_threshold: 1.0,
  barrier_sensitivity: 1.0,
};

export interface SettingDef {
  key: keyof ParseConfig;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
}

export const SETTING_DEFS: SettingDef[] = [
  {
    key: "min_ocr_confidence",
    label: "OCR min confidence",
    hint: "Drop low-confidence detections",
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    key: "ocr_padding",
    label: "OCR padding (barrier holes)",
    hint: "Clearance punched around text in barrier mask",
    min: 0,
    max: 20,
    step: 1,
  },
  {
    key: "los_threshold",
    label: "LoS ink threshold",
    hint: "Max ink ratio along line-of-sight (lower = stricter)",
    min: 0.02,
    max: 0.4,
    step: 0.01,
  },
  {
    key: "merge_threshold",
    label: "Merge threshold",
    hint: "Spatial merge scale (>1 = more merging)",
    min: 0.5,
    max: 2,
    step: 0.05,
  },
  {
    key: "barrier_sensitivity",
    label: "Barrier sensitivity",
    hint: "Ink wall strength (>1 = thicker barriers)",
    min: 0.5,
    max: 2,
    step: 0.05,
  },
];

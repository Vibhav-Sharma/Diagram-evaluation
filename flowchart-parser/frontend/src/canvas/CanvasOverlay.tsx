import { useEffect, useRef, useState } from "react";
import type { DebugStage, ParseResponse } from "../types";
import { buildGroupColorMap, renderLayers, type RenderContext } from "./renderers";

interface Props {
  imageUrl: string | null;
  data: ParseResponse | null;
  stage: DebugStage;
}

export default function CanvasOverlay({ imageUrl, data, stage }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [barrierImage, setBarrierImage] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!imageUrl) {
      setImage(null);
      return;
    }
    const img = new Image();
    img.onload = () => setImage(img);
    img.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    if (!data?.barrier_mask_png_base64) {
      setBarrierImage(null);
      return;
    }
    const img = new Image();
    img.onload = () => setBarrierImage(img);
    img.src = `data:image/png;base64,${data.barrier_mask_png_base64}`;
  }, [data?.barrier_mask_png_base64]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const draw = () => {
      const rect = container.getBoundingClientRect();
      const imgW = data?.image.width ?? image?.naturalWidth ?? 800;
      const imgH = data?.image.height ?? image?.naturalHeight ?? 600;
      const scale = Math.min(
        (rect.width - 24) / imgW,
        (rect.height - 24) / imgH,
        2
      );
      const cw = Math.floor(imgW * scale);
      const ch = Math.floor(imgH * scale);

      const dpr = window.devicePixelRatio || 1;
      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const rc: RenderContext = {
        ctx,
        scale,
        data: data!,
        image,
        barrierImage,
        groupColorMap: data ? buildGroupColorMap(data) : new Map(),
      };

      const emptyData: ParseResponse = {
        image: {
          width: image?.naturalWidth ?? 800,
          height: image?.naturalHeight ?? 600,
        },
        config: {
          min_ocr_confidence: 0.25,
          ocr_padding: 3,
          los_threshold: 0.12,
          merge_threshold: 1,
          barrier_sensitivity: 1,
        },
        logs: [],
        ocr_blocks: [],
        local_fusion_groups: [],
        merged_groups: [],
        merge_edges: [],
        allowed_merge_edges: [],
        blocked_merge_edges: [],
        final_nodes: [],
        barrier_mask_png_base64: "",
        stats: {
          ocr_count: 0,
          fusion_group_count: 0,
          final_node_count: 0,
          merge_attempts: 0,
          blocked_merges: 0,
          allowed_merges: 0,
        },
      };

      if (data) {
        renderLayers(stage.layers, rc);
      } else if (image) {
        renderLayers(stage.layers.includes("barrier_mask") ? ["image"] : stage.layers, {
          ...rc,
          data: emptyData,
        });
      }
    };

    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(container);
    return () => ro.disconnect();
  }, [image, barrierImage, data, stage, imageUrl]);

  return (
    <div
      ref={containerRef}
      className="flex h-full w-full items-center justify-center overflow-auto p-3"
    >
      <canvas
        ref={canvasRef}
        className="rounded-lg shadow-2xl ring-1 ring-slate-700/80"
      />
    </div>
  );
}

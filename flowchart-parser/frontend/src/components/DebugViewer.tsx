import CanvasOverlay from "../canvas/CanvasOverlay";
import { getStageById } from "../config/stages";
import type { ParseResponse } from "../types";
import StageTabs from "./StageTabs";

interface Props {
  imageUrl: string | null;
  data: ParseResponse | null;
  activeStageId: string;
  onStageChange: (id: string) => void;
  loading?: boolean;
}

export default function DebugViewer({
  imageUrl,
  data,
  activeStageId,
  onStageChange,
  loading,
}: Props) {
  const stage = getStageById(activeStageId) ?? getStageById("original")!;

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl bg-surface-raised ring-1 ring-slate-700/60">
      <StageTabs
        activeId={activeStageId}
        onChange={onStageChange}
        hasResult={!!data}
        disabled={loading}
      />
      <div className="relative min-h-0 flex-1">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3">
              <div className="h-10 w-10 animate-spin rounded-full border-2 border-accent border-t-transparent" />
              <p className="text-sm text-slate-300">Running OCR pipeline…</p>
            </div>
          </div>
        )}
        <div className="flex h-full justify-between gap-0">
          <div className="min-w-0 flex-1">
            <CanvasOverlay imageUrl={imageUrl} data={data} stage={stage} />
          </div>
          {data && (
            <aside className="hidden w-52 shrink-0 border-l border-slate-700/80 p-3 xl:block">
              <p className="text-[10px] uppercase text-muted">Stage</p>
              <p className="text-sm font-medium text-slate-200">{stage.label}</p>
              <p className="mt-2 text-xs text-muted">{stage.description}</p>
              <div className="mt-4 space-y-1 text-[10px] text-muted">
                <p>OCR: {data.stats.ocr_count}</p>
                <p>Fusion: {data.stats.fusion_group_count}</p>
                <p>Nodes: {data.stats.final_node_count}</p>
                <p className="text-success">Allowed: {data.stats.allowed_merges}</p>
                <p className="text-danger">Blocked: {data.stats.blocked_merges}</p>
              </div>
            </aside>
          )}
        </div>
      </div>
    </div>
  );
}

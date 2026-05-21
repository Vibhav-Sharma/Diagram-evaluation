import { useCallback, useEffect, useState } from "react";
import { checkHealth, parseFlowchart } from "./api/client";
import DebugViewer from "./components/DebugViewer";
import JsonViewer from "./components/JsonViewer";
import ProcessingLogs from "./components/ProcessingLogs";
import SettingsPanel from "./components/SettingsPanel";
import UploadPanel from "./components/UploadPanel";
import { DEFAULT_CONFIG } from "./config/settings";
import type { ParseConfig, ParseResponse } from "./types";
import { downloadJson } from "./utils/download";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [config, setConfig] = useState<ParseConfig>(DEFAULT_CONFIG);
  const [result, setResult] = useState<ParseResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [stageId, setStageId] = useState("original");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then(setApiOnline);
  }, []);

  const onFileSelect = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setLogs([]);
    setError(null);
    setStageId("original");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  }, [previewUrl]);

  const handleParse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setLogs(["Sending image to parser…"]);
    try {
      const data = await parseFlowchart(file, config);
      setResult(data);
      setLogs(data.logs ?? []);
      setStageId("final_nodes");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setLogs((prev) => [...prev, `Failed: ${msg}`]);
    } finally {
      setLoading(false);
    }
  };

  const jsonOutput = result?.final_nodes ?? null;

  return (
    <div className="flex h-screen flex-col">
      <header className="flex shrink-0 items-center justify-between border-b border-slate-700/80 bg-surface-raised px-5 py-3">
        <div>
          <h1 className="text-lg font-bold tracking-tight text-white">
            Flowchart Parser
            <span className="ml-2 text-sm font-normal text-accent-glow">
              Debug Workstation
            </span>
          </h1>
          <p className="text-xs text-muted">
            Visual semantic text grouping · OCR → fusion → LoS → nodes
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 ring-1 ${
              apiOnline
                ? "bg-success/10 text-success ring-success/30"
                : apiOnline === false
                  ? "bg-danger/10 text-danger ring-danger/30"
                  : "bg-slate-800 text-muted ring-slate-600"
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                apiOnline ? "bg-success" : apiOnline === false ? "bg-danger" : "bg-slate-500"
              }`}
            />
            API {apiOnline ? "online" : apiOnline === false ? "offline" : "…"}
          </span>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[1fr_auto] lg:grid-cols-[300px_1fr] lg:grid-rows-[1fr_220px]">
        {/* Left panel */}
        <aside className="flex flex-col overflow-y-auto border-b border-slate-700/80 bg-surface p-4 lg:border-b-0 lg:border-r">
          <UploadPanel
            previewUrl={previewUrl}
            fileName={file?.name ?? null}
            onFileSelect={onFileSelect}
            disabled={loading}
          />

          <button
            type="button"
            onClick={handleParse}
            disabled={!file || loading}
            className="mt-4 w-full rounded-lg bg-accent py-2.5 text-sm font-semibold text-white shadow-lg shadow-accent/25 transition hover:bg-accent-glow disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Parsing…" : "Parse Flowchart"}
          </button>

          <div className="mt-6">
            <SettingsPanel
              config={config}
              onChange={setConfig}
              disabled={loading}
            />
          </div>

          <ProcessingLogs logs={logs} error={error} />
        </aside>

        {/* Right — debug viewer */}
        <main className="min-h-0 p-3 lg:col-start-2 lg:row-start-1">
          <DebugViewer
            imageUrl={previewUrl}
            data={result}
            activeStageId={stageId}
            onStageChange={setStageId}
            loading={loading}
          />
        </main>

        {/* Bottom — JSON */}
        <section className="h-[220px] min-h-0 lg:col-span-2">
          <JsonViewer
            data={jsonOutput}
            onDownload={() =>
              jsonOutput && downloadJson(jsonOutput, "flowchart_output.json")
            }
          />
        </section>
      </div>
    </div>
  );
}

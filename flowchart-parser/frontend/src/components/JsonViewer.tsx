import { useState } from "react";

interface Props {
  data: unknown;
  onDownload?: () => void;
}

export default function JsonViewer({ data, onDownload }: Props) {
  const [copied, setCopied] = useState(false);
  const text = data ? JSON.stringify(data, null, 2) : "// Parse a flowchart to see JSON output";

  const copy = async () => {
    if (!data) return;
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex h-full flex-col border-t border-slate-700/80 bg-surface-raised">
      <div className="flex items-center justify-between px-4 py-2">
        <h2 className="text-sm font-semibold text-slate-200">JSON Output</h2>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={copy}
            disabled={!data}
            className="rounded-md bg-surface-overlay px-3 py-1 text-xs text-slate-300 ring-1 ring-slate-600 hover:bg-slate-700 disabled:opacity-40"
          >
            {copied ? "Copied!" : "Copy"}
          </button>
          {onDownload && (
            <button
              type="button"
              onClick={onDownload}
              disabled={!data}
              className="rounded-md bg-accent px-3 py-1 text-xs font-medium text-white hover:bg-accent-glow disabled:opacity-40"
            >
              Download JSON
            </button>
          )}
        </div>
      </div>
      <pre className="flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed text-emerald-400/90">
        {text}
      </pre>
    </div>
  );
}

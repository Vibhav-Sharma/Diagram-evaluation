interface Props {
  logs: string[];
  error?: string | null;
}

export default function ProcessingLogs({ logs, error }: Props) {
  return (
    <div className="mt-4">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
        Processing logs
      </h3>
      <div className="max-h-32 overflow-y-auto rounded-lg bg-black/40 p-2 font-mono text-[10px] leading-relaxed ring-1 ring-slate-800">
        {error && (
          <p className="text-danger">ERROR: {error}</p>
        )}
        {logs.length === 0 && !error && (
          <p className="text-muted">Waiting for parse…</p>
        )}
        {logs.map((line, i) => (
          <p key={i} className="text-slate-400">
            <span className="text-accent">›</span> {line}
          </p>
        ))}
      </div>
    </div>
  );
}

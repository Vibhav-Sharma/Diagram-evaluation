import type { ParseConfig, ParseResponse } from "../types";

const API_BASE = import.meta.env.VITE_API_URL ?? "/api";

export async function parseFlowchart(
  file: File,
  config: ParseConfig
): Promise<ParseResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("config", JSON.stringify(config));

  const res = await fetch(`${API_BASE}/parse`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof err.detail === "string" ? err.detail : JSON.stringify(err)
    );
  }

  return res.json() as Promise<ParseResponse>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

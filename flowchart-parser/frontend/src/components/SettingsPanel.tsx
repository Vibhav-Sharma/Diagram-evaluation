import { SETTING_DEFS } from "../config/settings";
import type { ParseConfig } from "../types";

interface Props {
  config: ParseConfig;
  onChange: (config: ParseConfig) => void;
  disabled?: boolean;
}

export default function SettingsPanel({ config, onChange, disabled }: Props) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
        Tuning
      </h3>
      {SETTING_DEFS.map((def) => (
        <label key={def.key} className="block">
          <div className="flex justify-between text-xs">
            <span className="text-slate-300">{def.label}</span>
            <span className="font-mono text-accent-glow">
              {config[def.key].toFixed(def.step < 1 ? 2 : 0)}
            </span>
          </div>
          <input
            type="range"
            min={def.min}
            max={def.max}
            step={def.step}
            value={config[def.key]}
            disabled={disabled}
            onChange={(e) =>
              onChange({ ...config, [def.key]: parseFloat(e.target.value) })
            }
            className="mt-1 h-1.5 w-full cursor-pointer accent-accent"
          />
          <p className="mt-0.5 text-[10px] text-muted">{def.hint}</p>
        </label>
      ))}
    </div>
  );
}

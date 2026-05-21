import { DEBUG_STAGES } from "../config/stages";

interface Props {
  activeId: string;
  onChange: (id: string) => void;
  disabled?: boolean;
  hasResult?: boolean;
}

export default function StageTabs({
  activeId,
  onChange,
  disabled,
  hasResult,
}: Props) {
  return (
    <div className="flex flex-wrap gap-1 border-b border-slate-700/80 bg-surface-raised/80 px-2 py-2">
      {DEBUG_STAGES.map((stage) => {
        const isDisabled =
          disabled || (stage.id !== "original" && !hasResult);

        return (
          <button
            key={stage.id}
            type="button"
            disabled={isDisabled}
            onClick={() => onChange(stage.id)}
            className={`rounded-md px-2.5 py-1.5 text-xs font-medium transition ${
              activeId === stage.id
                ? "bg-accent text-white shadow-lg shadow-accent/20"
                : "text-muted hover:bg-surface-overlay hover:text-slate-200"
            } ${isDisabled ? "cursor-not-allowed opacity-40" : ""}`}
            title={stage.description}
          >
            {stage.label}
          </button>
        );
      })}
    </div>
  );
}

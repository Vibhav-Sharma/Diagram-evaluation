import { useCallback, useRef, useState } from "react";

interface Props {
  previewUrl: string | null;
  fileName: string | null;
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const ACCEPT = ["image/png", "image/jpeg", "image/jpg"];

export default function UploadPanel({
  previewUrl,
  fileName,
  onFileSelect,
  disabled,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const ok =
        ACCEPT.includes(file.type) ||
        /\.(png|jpe?g)$/i.test(file.name);
      if (!ok) {
        alert("Please upload PNG or JPEG.");
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-4 text-center transition ${
          dragOver
            ? "border-accent bg-accent/10"
            : "border-slate-600 hover:border-slate-500 hover:bg-surface-overlay/50"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".png,.jpg,.jpeg,image/png,image/jpeg"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="text-sm font-medium text-slate-200">
          Drop flowchart image
        </p>
        <p className="mt-1 text-xs text-muted">PNG, JPG, JPEG</p>
      </div>

      {previewUrl && (
        <div className="overflow-hidden rounded-lg ring-1 ring-slate-700">
          <img
            src={previewUrl}
            alt="Preview"
            className="max-h-36 w-full object-contain bg-black/40"
          />
          {fileName && (
            <p className="truncate px-2 py-1 font-mono text-xs text-muted">
              {fileName}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

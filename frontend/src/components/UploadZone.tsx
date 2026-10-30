import { useCallback, useId, useRef, useState } from "react";

const IMAGE_EXT = [".png", ".jpg", ".jpeg", ".bmp"];
const VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"];
const ACCEPTED = [...IMAGE_EXT, ...VIDEO_EXT];

interface UploadZoneProps {
  onFile: (file: File, kind: "image" | "video") => void;
}

function classify(file: File): "image" | "video" | null {
  const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
  if (IMAGE_EXT.includes(ext)) return "image";
  if (VIDEO_EXT.includes(ext)) return "video";
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("video/")) return "video";
  return null;
}

export default function UploadZone({ onFile }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (!file) return;
      const kind = classify(file);
      if (!kind) {
        setError(`Unsupported file type. Use ${ACCEPTED.join(", ")}.`);
        return;
      }
      setError(null);
      onFile(file, kind);
    },
    [onFile],
  );

  return (
    <div className="w-full">
      <label
        htmlFor={inputId}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`depth-grid group relative flex min-h-72 w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-8 py-14 text-center transition-colors ${
          dragging
            ? "border-[var(--accent)] bg-[var(--accent-soft)]"
            : "border-[var(--border-strong)] bg-[var(--bg-elevated)] hover:border-[var(--text-faint)]"
        }`}
      >
        <PortholeIcon active={dragging} />
        <div className="space-y-1">
          <p className="font-display text-xl text-[var(--text)]">
            {dragging ? "Drop it in" : "Drag a photo or video here"}
          </p>
          <p className="text-sm text-[var(--text-muted)]">
            or <span className="text-[var(--accent)] underline underline-offset-4">browse your files</span>
          </p>
        </div>
        <p className="font-mono text-xs tracking-wide text-[var(--text-faint)] uppercase">
          {ACCEPTED.join(" · ")}
        </p>
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept={ACCEPTED.join(",")}
          className="sr-only"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </label>
      {error && <p className="mt-3 text-sm text-[var(--danger)]">{error}</p>}
    </div>
  );
}

function PortholeIcon({ active }: { active: boolean }) {
  return (
    <svg
      width="56"
      height="56"
      viewBox="0 0 56 56"
      fill="none"
      className={`transition-transform ${active ? "scale-110" : "animate-drift"}`}
      aria-hidden="true"
    >
      <circle cx="28" cy="28" r="24" stroke="var(--border-strong)" strokeWidth="2" />
      <circle cx="28" cy="28" r="17" stroke="var(--text-faint)" strokeWidth="1.5" />
      <path
        d="M20 30c2-6 5-9 8-9s6 3 8 9"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinecap="round"
        fill="none"
      />
      <circle cx="28" cy="24" r="2" fill="var(--accent)" />
    </svg>
  );
}

import { useRef } from "react";

interface VideoCompareProps {
  beforeSrc: string;
  afterSrc: string;
}

/** Side-by-side original/enhanced video, play state kept in sync. */
export default function VideoCompare({ beforeSrc, afterSrc }: VideoCompareProps) {
  const beforeRef = useRef<HTMLVideoElement>(null);
  const afterRef = useRef<HTMLVideoElement>(null);

  const syncTo = (source: HTMLVideoElement, target: HTMLVideoElement | null) => {
    if (!target) return;
    if (Math.abs(target.currentTime - source.currentTime) > 0.15) {
      target.currentTime = source.currentTime;
    }
  };

  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Panel label="Original" accentClass="text-[var(--text-muted)]">
        <video
          ref={beforeRef}
          src={beforeSrc}
          controls
          onPlay={() => afterRef.current?.play()}
          onPause={() => afterRef.current?.pause()}
          onSeeked={(e) => beforeRef.current && syncTo(e.currentTarget, afterRef.current)}
          onTimeUpdate={(e) => syncTo(e.currentTarget, afterRef.current)}
          className="h-full w-full rounded-xl bg-black"
        />
      </Panel>
      <Panel label="Enhanced" accentClass="text-[var(--success)]">
        <video
          ref={afterRef}
          src={afterSrc}
          controls
          onPlay={() => beforeRef.current?.play()}
          onPause={() => beforeRef.current?.pause()}
          onSeeked={(e) => afterRef.current && syncTo(e.currentTarget, beforeRef.current)}
          className="h-full w-full rounded-xl bg-black"
        />
      </Panel>
    </div>
  );
}

function Panel({
  label,
  accentClass,
  children,
}: {
  label: string;
  accentClass: string;
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
      <p className={`mb-2 font-mono text-[11px] tracking-wide uppercase ${accentClass}`}>{label}</p>
      {children}
    </div>
  );
}

import { useCallback, useRef, useState } from "react";

interface CompareSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function CompareSlider({
  beforeSrc,
  afterSrc,
  beforeLabel = "Original",
  afterLabel = "Enhanced",
}: CompareSliderProps) {
  const [pct, setPct] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const next = ((clientX - rect.left) / rect.width) * 100;
    setPct(Math.min(100, Math.max(0, next)));
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative aspect-video w-full touch-none overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--bg-elevated)] select-none"
      onPointerDown={(e) => {
        dragging.current = true;
        (e.target as Element).setPointerCapture(e.pointerId);
        updateFromClientX(e.clientX);
      }}
      onPointerMove={(e) => dragging.current && updateFromClientX(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
    >
      {/* Enhanced image fills the frame */}
      <img src={afterSrc} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" />

      {/* Original image, same size and position, clipped to the left pct */}
      <img
        src={beforeSrc}
        alt={beforeLabel}
        className="absolute inset-0 h-full w-full object-contain"
        style={{ clipPath: `inset(0 ${100 - pct}% 0 0)` }}
      />

      <div
        className="absolute inset-y-0 w-px bg-[var(--accent)] shadow-[0_0_12px_rgba(255,107,74,0.6)]"
        style={{ left: `${pct}%` }}
      >
        <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-[var(--accent)] bg-[var(--bg)] shadow-lg">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
            <path d="M5 2 1 7l4 5M9 2l4 5-4 5" stroke="var(--accent)" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      </div>

      <span className="absolute top-3 left-3 rounded-full bg-black/50 px-2.5 py-1 font-mono text-[11px] tracking-wide text-[var(--text)] uppercase backdrop-blur">
        {beforeLabel}
      </span>
      <span className="absolute top-3 right-3 rounded-full bg-black/50 px-2.5 py-1 font-mono text-[11px] tracking-wide text-[var(--success)] uppercase backdrop-blur">
        {afterLabel}
      </span>
    </div>
  );
}

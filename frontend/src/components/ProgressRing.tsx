interface ProgressRingProps {
  progress: number | null; // 0..1, or null while indeterminate
  size?: number;
}

export default function ProgressRing({ progress, size = 64 }: ProgressRingProps) {
  const stroke = 4;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = progress == null ? 0.12 : Math.min(1, Math.max(0, progress));

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg width={size} height={size} className={progress == null ? "animate-spin" : ""}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="var(--border)"
          strokeWidth={stroke}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="var(--accent)"
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - pct)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: progress == null ? undefined : "stroke-dashoffset 0.3s ease" }}
        />
      </svg>
      {progress != null && (
        <span className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-[var(--text)]">
          {Math.round(progress * 100)}%
        </span>
      )}
    </div>
  );
}

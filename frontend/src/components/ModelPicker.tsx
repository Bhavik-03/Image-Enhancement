interface ModelPickerProps {
  models: string[];
  value: string;
  onChange: (model: string) => void;
  defaultModel: string;
}

export default function ModelPicker({ models, value, onChange, defaultModel }: ModelPickerProps) {
  if (models.length <= 1) return null;

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="font-mono text-xs tracking-wide text-[var(--text-faint)] uppercase">
        Model
      </span>
      {models.map((m) => {
        const active = m === value;
        return (
          <button
            key={m}
            type="button"
            onClick={() => onChange(m)}
            className={`rounded-full border px-3 py-1.5 text-sm transition-colors ${
              active
                ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent-strong)]"
                : "border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:border-[var(--border-strong)] hover:text-[var(--text)]"
            }`}
          >
            {m === defaultModel ? "Pretrained" : m.replace(/_best\.pth$/, "")}
          </button>
        );
      })}
    </div>
  );
}

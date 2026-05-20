import { useState, useMemo } from "react";

/**
 * Lightweight JSON viewer. Prefers correctness + density over fanciness; no
 * third-party tree component is pulled in for v1.
 */
export function JsonView({
  value,
  collapsed = false,
  maxHeight = "24rem",
}: {
  value: unknown;
  collapsed?: boolean;
  maxHeight?: string;
}) {
  const [open, setOpen] = useState(!collapsed);
  const pretty = useMemo(() => safeStringify(value), [value]);
  return (
    <div className="border border-ink-800 bg-ink-900 rounded-sm">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-2 py-1 text-2xs font-mono uppercase tracking-wider text-ink-400 hover:text-ink-100 hover:bg-ink-800 transition-colors"
      >
        <span>{open ? "▾ json" : "▸ json"}</span>
        <span className="text-ink-500">{pretty.split("\n").length} lines</span>
      </button>
      {open ? (
        <pre
          className="mono text-xs leading-snug px-3 py-2 overflow-auto text-ink-200 bg-ink-950/50"
          style={{ maxHeight }}
        >
          {pretty}
        </pre>
      ) : null}
    </div>
  );
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

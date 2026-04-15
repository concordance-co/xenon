import { useQuery } from "@tanstack/react-query";
import { useState, useMemo } from "react";
import { api } from "@/lib/api";
import type { PromptExample, PromptSection } from "@/types/api";
import { Section } from "@/components/Inspector";

export function PromptTab({ runId, stepName }: { runId: string; stepName: string }) {
  const q = useQuery({
    queryKey: ["prompt", runId, stepName],
    queryFn: () => api.getPromptPreview(runId, stepName, { max_examples: 3 }),
  });
  const [active, setActive] = useState(0);

  if (q.isLoading) return <div className="p-3 text-2xs font-mono text-ink-400">Loading prompt preview…</div>;
  if (q.error)
    return (
      <div className="p-3 text-2xs font-mono text-status-fail">
        Failed to load prompt: {(q.error as Error).message}
      </div>
    );
  if (!q.data) return null;
  const data = q.data;
  if (!data.available) {
    return (
      <div className="p-3 text-2xs font-mono text-status-fail border border-status-fail/40 bg-status-fail/5">
        prompt preview unresolved · {data.reason ?? "unknown"}
      </div>
    );
  }
  const examples = data.examples;
  if (examples.length === 0)
    return <div className="p-3 text-2xs font-mono text-ink-500">No prompt examples available.</div>;

  const current = examples[Math.min(active, examples.length - 1)];

  return (
    <div className="p-3 space-y-3">
      {data.degraded ? (
        <div className="border border-status-warn/40 bg-status-warn/5 text-status-warn p-2 text-2xs font-mono">
          section-level fallback · {data.degraded_reason ?? "exact tokenization unavailable"}
        </div>
      ) : null}
      {examples.length > 1 ? (
        <nav className="flex gap-1 text-2xs font-mono">
          {examples.map((e, i) => (
            <button
              type="button"
              key={e.example_key}
              onClick={() => setActive(i)}
              className={`px-2 py-0.5 border rounded-sm ${
                i === active
                  ? "border-accent text-accent"
                  : "border-ink-700 text-ink-400 hover:text-ink-100 hover:border-ink-500"
              }`}
            >
              {e.example_key.slice(0, 14)}
            </button>
          ))}
        </nav>
      ) : null}
      {current.selection ? (
        <div className="px-2 py-1.5 bg-ink-850 border border-ink-800 text-2xs font-mono text-ink-200">
          <span className="text-ink-500 mr-1">selection</span>
          {current.selection.sentence}
        </div>
      ) : null}
      <Section title="prompt">
        <PromptText example={current} />
      </Section>
      {current.warnings.length > 0 ? (
        <Section title="warnings">
          <ul className="text-2xs font-mono text-status-warn space-y-0.5">
            {current.warnings.map((w, i) => (
              <li key={i}>· {w}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

function PromptText({ example }: { example: PromptExample }) {
  const segments = useMemo(() => splitBySections(example.text, example.sections), [example]);
  return (
    <div className="bg-ink-950 border border-ink-800 rounded-sm p-3 max-h-[32rem] overflow-auto">
      <div className="mono text-xs whitespace-pre-wrap leading-relaxed">
        {segments.map((seg, i) =>
          seg.section ? (
            <span
              key={i}
              className={[
                "relative",
                seg.section.selected
                  ? "bg-accent/20 ring-1 ring-accent/60"
                  : "bg-ink-800/60",
                "text-ink-50 rounded-sm px-0.5",
              ].join(" ")}
              title={`${seg.section.label}${
                seg.section.token_start != null ? ` · tokens ${seg.section.token_start}..${seg.section.token_end}` : ""
              }`}
            >
              {seg.text}
              {seg.section.selected ? (
                <span className="absolute -top-3 left-0 text-[0.625rem] font-mono tracking-widest text-accent">
                  {seg.section.label}
                </span>
              ) : null}
            </span>
          ) : (
            <span key={i}>{seg.text}</span>
          ),
        )}
      </div>
    </div>
  );
}

function splitBySections(
  text: string,
  sections: PromptSection[],
): Array<{ text: string; section: PromptSection | null }> {
  if (sections.length === 0) return [{ text, section: null }];
  // Assume non-overlapping, sorted by char_start. Fall back safely if not.
  const sorted = [...sections]
    .filter((s) => s.char_start >= 0 && s.char_end > s.char_start && s.char_end <= text.length)
    .sort((a, b) => a.char_start - b.char_start);
  const out: Array<{ text: string; section: PromptSection | null }> = [];
  let cursor = 0;
  for (const s of sorted) {
    if (s.char_start < cursor) continue; // overlap — skip
    if (s.char_start > cursor) out.push({ text: text.slice(cursor, s.char_start), section: null });
    out.push({ text: text.slice(s.char_start, s.char_end), section: s });
    cursor = s.char_end;
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor), section: null });
  return out;
}

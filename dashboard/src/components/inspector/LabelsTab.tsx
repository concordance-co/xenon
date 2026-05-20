import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Section } from "@/components/Inspector";
import type { LabelDistribution } from "@/types/api";
import { truncate } from "@/lib/format";

export function LabelsTab({ runId, stepName }: { runId: string; stepName: string }) {
  const q = useQuery({
    queryKey: ["labels", runId, stepName],
    queryFn: () => api.getLabelPreview(runId, stepName),
  });

  if (q.isLoading) return <div className="p-3 text-2xs font-mono text-ink-400">Loading labels…</div>;
  if (q.error)
    return (
      <div className="p-3 text-2xs font-mono text-status-fail">
        Failed to load labels: {(q.error as Error).message}
      </div>
    );
  if (!q.data) return null;
  const data = q.data;
  if (!data.available) {
    return (
      <div className="p-3 text-2xs font-mono text-status-warn border border-status-warn/40 bg-status-warn/5">
        labels unavailable · {data.reason ?? "unknown reason"}
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3">
      {data.resolved_from_step ? (
        <div className="text-2xs font-mono text-ink-400">
          resolved from upstream step <span className="text-ink-200">{data.resolved_from_step}</span>
        </div>
      ) : null}
      {data.labels.length === 0 ? (
        <div className="text-2xs font-mono text-ink-500">No labels exposed on this dataset.</div>
      ) : (
        data.labels.map((lbl) => <DistributionPanel key={lbl.label_name} dist={lbl} />)
      )}
      {data.samples.length > 0 ? (
        <Section title={`sample mapping (${data.samples.length})`}>
          <ul className="text-2xs font-mono text-ink-300 divide-y divide-ink-800">
            {data.samples.map((s) => (
              <li key={s.example_key} className="py-1">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-ink-200">{s.example_key}</span>
                  <span className="text-ink-500 truncate max-w-[12rem]" title={JSON.stringify(s.labels)}>
                    {Object.entries(s.labels)
                      .map(([k, v]) => `${k}=${stringifyLabel(v)}`)
                      .slice(0, 3)
                      .join(" ")}
                  </span>
                </div>
                <div className="text-ink-500">{truncate(s.prompt_preview, 140)}</div>
              </li>
            ))}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

function DistributionPanel({ dist }: { dist: LabelDistribution }) {
  return (
    <Section title={`label · ${dist.label_name}${dist.source_step ? ` (from ${dist.source_step})` : ""}`}>
      {dist.numeric_summary ? (
        <div className="grid grid-cols-4 gap-x-3 gap-y-1 text-2xs font-mono">
          <Stat label="min" value={dist.numeric_summary.min.toFixed(3)} />
          <Stat label="max" value={dist.numeric_summary.max.toFixed(3)} />
          <Stat label="mean" value={dist.numeric_summary.mean.toFixed(3)} />
          <Stat label="stddev" value={dist.numeric_summary.stddev.toFixed(3)} />
        </div>
      ) : null}
      <div className="mt-1 text-2xs text-ink-500 font-mono">{dist.unique_values} unique values</div>
      <div className="mt-2 space-y-1">
        {dist.buckets.map((b) => (
          <div key={b.value} className="flex items-center gap-2">
            <span className="text-2xs font-mono text-ink-200 w-32 truncate" title={b.value}>
              {b.value}
            </span>
            <div className="flex-1 bg-ink-800 h-1.5 rounded-sm overflow-hidden">
              <div className="bg-accent h-full" style={{ width: `${Math.max(2, b.fraction * 100)}%` }} />
            </div>
            <span className="text-2xs font-mono text-ink-400 w-16 text-right">
              {b.count} · {(b.fraction * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="field-label">{label}</div>
      <div className="text-ink-100">{value}</div>
    </div>
  );
}

function stringifyLabel(v: unknown): string {
  if (v == null) return "null";
  if (typeof v === "string") return v.length > 12 ? `${v.slice(0, 11)}…` : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return "…";
}

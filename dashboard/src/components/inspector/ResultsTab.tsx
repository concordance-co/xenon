import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import type { StepDetail } from "@/types/api";
import { Section } from "@/components/Inspector";
import { JsonView } from "@/components/JsonView";
import { DataTable, columnsForRows } from "@/components/DataTable";
import { api } from "@/lib/api";

export function ResultsTab({ detail, runId }: { detail: StepDetail; runId: string }) {
  const q = useQuery({
    queryKey: ["result", runId, detail.step.step_name],
    queryFn: () => api.getStepResult(runId, detail.step.step_name),
  });
  const data = q.data ?? null;

  return (
    <div className="p-3 space-y-3">
      {detail.report_artifact_id ? (
        <Section title="report">
          <Link
            to={`/runs/${runId}/reports/${detail.report_artifact_id}`}
            className="text-xs font-mono text-accent hover:underline"
          >
            open report gallery →
          </Link>
        </Section>
      ) : null}

      {q.isLoading ? (
        <div className="text-2xs font-mono text-ink-400">Loading result…</div>
      ) : q.error ? (
        <div className="text-2xs font-mono text-status-fail">
          Failed to load result: {(q.error as Error).message}
        </div>
      ) : !data?.available ? (
        <div className="border border-status-warn/40 bg-status-warn/5 text-status-warn p-2 text-2xs font-mono">
          result unavailable · {data?.reason ?? "no result recorded"}
        </div>
      ) : (
        <>
          {data.headline ? (
            <Section title="headline">
              <HeadlineGrid headline={data.headline} />
            </Section>
          ) : null}

          {(data.tables ?? []).map((t) => (
            <Section
              key={t.name}
              title={`table · ${t.name} (${t.total_rows ?? t.rows.length}${t.truncated ? ` · preview ${t.rows.length}` : ""})`}
            >
              <DataTable rows={t.rows} columns={columnsForRows(t.rows)} maxHeight="20rem" />
            </Section>
          ))}

          <Section title="raw result">
            {data.path ? (
              <div className="text-[0.58rem] font-mono text-ink-600 uppercase tracking-widest mb-1 truncate">
                {data.path}
              </div>
            ) : null}
            {data.truncated && data.truncation_reason ? (
              <div className="mb-2 border border-status-warn/40 bg-status-warn/5 text-status-warn p-2 text-2xs font-mono">
                {data.truncation_reason}
              </div>
            ) : null}
            <JsonView value={data.payload ?? {}} collapsed />
          </Section>
        </>
      )}
    </div>
  );
}

function HeadlineGrid({ headline }: { headline: Record<string, unknown> }) {
  const entries = Object.entries(headline);
  return (
    <div className="grid grid-cols-2 gap-2 text-2xs font-mono">
      {entries.map(([key, value]) => (
        <div
          key={key}
          className="border border-ink-800 bg-ink-950/40 px-2 py-1.5 min-w-0"
        >
          <div className="field-label truncate">{key}</div>
          <div className="text-ink-100 text-[0.8rem] font-semibold truncate mt-0.5 tabular-nums">
            {renderHeadline(value)}
          </div>
        </div>
      ))}
    </div>
  );
}

function renderHeadline(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    return Math.abs(value) >= 1000 ? value.toFixed(0) : value.toFixed(4);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

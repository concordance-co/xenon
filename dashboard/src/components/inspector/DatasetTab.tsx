import { useQuery } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api } from "@/lib/api";
import { Section } from "@/components/Inspector";
import { JsonView } from "@/components/JsonView";
import { truncate } from "@/lib/format";

export function DatasetTab({ runId, stepName }: { runId: string; stepName: string }) {
  const [sourceStep, setSourceStep] = useState<string | undefined>(undefined);
  const [sampleSize, setSampleSize] = useState<number>(5);
  const q = useQuery({
    queryKey: ["dataset", runId, stepName, sourceStep, sampleSize],
    queryFn: () =>
      api.getDatasetPreview(runId, stepName, {
        source_step: sourceStep,
        sample_size: sampleSize,
      }),
  });

  if (q.isLoading) return <div className="p-3 text-2xs font-mono text-ink-400">Loading dataset preview…</div>;
  if (q.error) {
    return (
      <div className="p-3 text-2xs font-mono text-status-fail">
        Failed to load dataset: {(q.error as Error).message}
      </div>
    );
  }
  if (!q.data) return null;
  const data = q.data;

  if (!data.available) {
    return (
      <div className="p-3 space-y-3">
        <div className="border border-status-warn/40 bg-status-warn/5 text-status-warn p-2 text-2xs font-mono">
          dataset unavailable · {data.reason ?? "unknown reason"}
        </div>
        {data.dataset_options && data.dataset_options.length > 0 ? (
          <Section title="dataset options">
            <div className="space-y-1">
              {data.dataset_options.map((opt) => (
                <button
                  key={opt.step_name}
                  type="button"
                  onClick={() => setSourceStep(opt.step_name)}
                  className="w-full text-left text-2xs font-mono text-ink-100 hover:bg-ink-800 px-2 py-1 border border-ink-800 rounded-sm"
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </Section>
        ) : null}
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
      <Section title="dataset">
        <ConstructionList source={data.source} totalRows={data.total_rows ?? null} />
        {data.source?.query ? (
          <div className="mt-2">
            <div className="field-label mb-1">sql</div>
            <pre className="mono text-[0.65rem] bg-ink-950 border border-ink-800 text-ink-200 whitespace-pre-wrap break-words p-2 rounded-sm max-h-40 overflow-auto">
              {data.source.query}
            </pre>
          </div>
        ) : null}
      </Section>
      <Section
        title={`sample · ${data.rows.length}${data.total_rows ? ` of ${data.total_rows}` : ""}`}
      >
        <div className="flex items-center gap-1.5 mb-2">
          <span className="field-label">rows</span>
          {[3, 5, 10, 25].map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setSampleSize(n)}
              className={[
                "px-1.5 py-0.5 text-[0.6rem] font-mono border rounded-[2px]",
                sampleSize === n
                  ? "border-accent text-accent bg-accent/10"
                  : "border-ink-700 text-ink-500 hover:text-ink-100",
              ].join(" ")}
            >
              {n}
            </button>
          ))}
        </div>
        <DatasetTable rows={data.rows} />
      </Section>
    </div>
  );
}

function DatasetTable({
  rows,
}: {
  rows: Array<{
    example_key: string;
    case_key: string | null;
    prompt_preview: string;
    labels: Record<string, unknown>;
    metadata: Record<string, unknown>;
  }>;
}) {
  const [open, setOpen] = useState<string | null>(null);
  if (rows.length === 0) {
    return <div className="text-2xs font-mono text-ink-500">No rows sampled.</div>;
  }
  return (
    <div className="border border-ink-800 rounded-sm overflow-hidden">
      <table className="w-full text-2xs font-mono">
        <thead className="bg-ink-800 text-ink-400">
          <tr>
            <th className="text-left px-2 py-1">key</th>
            <th className="text-left px-2 py-1">case</th>
            <th className="text-left px-2 py-1">prompt</th>
            <th className="text-left px-2 py-1">labels</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isOpen = open === r.example_key;
            return (
              <Fragment key={r.example_key}>
                <tr
                  className="border-t border-ink-800 hover:bg-ink-850 cursor-pointer"
                  onClick={() => setOpen(isOpen ? null : r.example_key)}
                >
                  <td className="px-2 py-1 text-ink-200 align-top">{r.example_key}</td>
                  <td className="px-2 py-1 text-ink-400 align-top">{r.case_key ?? "—"}</td>
                  <td className="px-2 py-1 text-ink-300 max-w-[16rem] align-top">
                    {truncate(r.prompt_preview, 120)}
                  </td>
                  <td className="px-2 py-1 text-ink-400 align-top truncate max-w-[10rem]">
                    {labelBrief(r.labels)}
                  </td>
                </tr>
                {isOpen ? (
                  <tr className="bg-ink-950/40">
                    <td colSpan={4} className="p-2">
                      <div className="space-y-2">
                        <details className="border border-ink-800 rounded-sm">
                          <summary className="px-2 py-1 text-2xs uppercase tracking-wider text-ink-400">
                            full prompt
                          </summary>
                          <pre className="mono text-xs whitespace-pre-wrap p-2 text-ink-200">
                            {r.prompt_preview}
                          </pre>
                        </details>
                        <JsonView value={r.labels} collapsed />
                        <JsonView value={r.metadata} collapsed />
                      </div>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function labelBrief(labels: Record<string, unknown>): string {
  const keys = Object.keys(labels);
  if (keys.length === 0) return "—";
  const first = keys.slice(0, 2).map((k) => `${k}=${stringify(labels[k])}`);
  return first.join(" ") + (keys.length > 2 ? ` +${keys.length - 2}` : "");
}

function stringify(v: unknown): string {
  if (v == null) return "null";
  if (typeof v === "string") return v.length > 12 ? `${v.slice(0, 11)}…` : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return "…";
}

function ConstructionList({
  source,
  totalRows,
}: {
  source: import("@/types/api").DatasetSourceInfo | null | undefined;
  totalRows: number | null;
}) {
  if (!source) return null;
  const items: Array<{ label: string; value: string }> = [];
  items.push({ label: "kind", value: source.kind });
  if (source.name) items.push({ label: "name", value: source.name });
  if (source.deferred !== undefined)
    items.push({ label: "deferred", value: source.deferred ? "yes" : "no" });
  if (typeof totalRows === "number")
    items.push({ label: "total_rows", value: String(totalRows) });
  if (source.total_examples != null)
    items.push({ label: "total_examples", value: String(source.total_examples) });
  for (const c of source.construction ?? []) items.push(c);
  if (source.dataset_id) items.push({ label: "dataset_id", value: source.dataset_id });
  return (
    <ul className="divide-y divide-ink-800">
      {items.map((item, i) => (
        <li key={`${item.label}-${i}`} className="py-1 first:pt-0 last:pb-0">
          <div className="field-label break-all leading-snug">{item.label}</div>
          <div className="text-[0.7rem] font-mono text-ink-200 break-words whitespace-pre-wrap leading-snug mt-0.5">
            {item.value || <span className="text-ink-500">—</span>}
          </div>
        </li>
      ))}
    </ul>
  );
}

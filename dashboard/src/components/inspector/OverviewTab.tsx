import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type {
  DatasetPreview,
  LabelDistribution,
  StepDetail,
  StepSummary,
} from "@/types/api";
import { api } from "@/lib/api";
import { formatDuration, shortHash, truncate } from "@/lib/format";

/**
 * Operator-focused overview. Reads as a quick brief of what this step is
 * doing: where we capture, what the dataset looks like, what labels flow
 * through it. Timing and hash noise is tucked under a `details` toggle.
 */
export function OverviewTab({
  step,
  runId,
  stepDetail,
}: {
  step: StepSummary;
  runId: string;
  stepDetail: StepDetail | null;
}) {
  const hasCapture = Boolean(stepDetail?.has_dataset);

  const datasetQ = useQuery({
    queryKey: ["dataset", runId, step.step_name, 3, "overview"],
    queryFn: () => api.getDatasetPreview(runId, step.step_name, { sample_size: 3 }),
    enabled: hasCapture,
  });
  const labelQ = useQuery({
    queryKey: ["labels", runId, step.step_name, 25, "overview"],
    queryFn: () => api.getLabelPreview(runId, step.step_name, {}),
    enabled: hasCapture,
  });

  const dataset = datasetQ.data ?? null;
  const labels = labelQ.data ?? null;

  const sites = extractSites(stepDetail?.spec);

  return (
    <div className="p-3 space-y-3">
      {/* Where we capture */}
      {sites.length > 0 ? (
        <Panel title="capture sites" hint={`${sites.length}`}>
          <ul className="divide-y divide-ink-800">
            {sites.map((site, i) => (
              <li key={(site.name as string) ?? i} className="py-1.5 first:pt-0 last:pb-0">
                <SiteLine site={site} />
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}

      {/* Dataset at a glance */}
      {hasCapture ? (
        <Panel
          title="dataset"
          hint={
            dataset?.source
              ? [
                  dataset.source.name,
                  dataset.source.total_examples != null
                    ? `${dataset.source.total_examples} rows`
                    : dataset.source.deferred
                      ? "deferred"
                      : null,
                ]
                  .filter(Boolean)
                  .join(" · ")
              : undefined
          }
        >
          {datasetQ.isLoading ? (
            <Note>loading…</Note>
          ) : !dataset?.available ? (
            <Note tone="warn">{dataset?.reason ?? "unavailable"}</Note>
          ) : (
            <DatasetSnapshot dataset={dataset} stepDetail={stepDetail} />
          )}
        </Panel>
      ) : null}

      {/* Labels / behaviors */}
      {hasCapture ? (
        <Panel
          title="labels"
          hint={labels?.available ? `${labels.labels.length} fields` : undefined}
        >
          {labelQ.isLoading ? (
            <Note>loading…</Note>
          ) : !labels?.available ? (
            <Note tone="warn">{labels?.reason ?? "unavailable"}</Note>
          ) : labels.labels.length === 0 ? (
            <Note>no labels on the sampled rows</Note>
          ) : (
            <ul className="divide-y divide-ink-800">
              {labels.labels.map((l) => (
                <li key={l.label_name} className="py-1.5 first:pt-0 last:pb-0">
                  <LabelLine dist={l} />
                </li>
              ))}
            </ul>
          )}
        </Panel>
      ) : null}

      {/* Upstream / downstream — compact, not a bloated section */}
      {(step.resolved_depends_on.length > 0 || stepDetail?.downstream.length) ? (
        <Panel title="flow">
          <div className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-2xs font-mono">
            <span className="field-label pt-0.5">upstream</span>
            <DepChips
              names={step.resolved_depends_on}
              linkBase={`/runs/${runId}`}
              fallback="—"
            />
            <span className="field-label pt-0.5">downstream</span>
            <DepChips
              names={(stepDetail?.downstream ?? []).map((d) => d.step_name)}
              linkBase={`/runs/${runId}`}
              fallback="—"
            />
          </div>
        </Panel>
      ) : null}

      {/* Report link — prominent if available */}
      {stepDetail?.report_artifact_id ? (
        <div className="border border-accent/40 bg-accent/5 px-3 py-2 text-xs font-mono flex items-center gap-2">
          <span className="text-accent">→</span>
          <Link
            to={`/runs/${runId}/reports/${stepDetail.report_artifact_id}`}
            className="text-accent hover:underline"
          >
            open report gallery
          </Link>
        </div>
      ) : null}

      {/* Noisy stuff — collapsed */}
      <CollapsibleDetails step={step} runId={runId} />
    </div>
  );
}

/* ------------------------------------------------------------------------ */

type Site = Record<string, unknown>;

function extractSites(spec: Record<string, unknown> | undefined): Site[] {
  if (!spec) return [];
  const sitesRaw = spec["sites"];
  return Array.isArray(sitesRaw) ? (sitesRaw as Site[]) : [];
}

function SiteLine({ site }: { site: Site }) {
  const name = (site.name as string) ?? "—";
  const layers = site.layers as unknown;
  const layersStr = Array.isArray(layers)
    ? (layers as unknown[]).map(String).join(",")
    : "—";
  const tokens = site.tokens as { kind?: string; value?: unknown } | undefined;
  const tokenLabel = tokens
    ? tokens.kind === "section"
      ? `section(${String(tokens.value)})`
      : tokens.kind === "slice"
        ? `slice`
        : tokens.kind ?? "—"
    : "—";
  const record = site.record as unknown;
  const isMoE = Array.isArray(record);
  const siteKind = (site.site as string | undefined) ?? (isMoE ? "moe" : "residual");
  return (
    <div className="flex items-center gap-2 flex-wrap text-[0.7rem] font-mono">
      <span className="chip chip-muted text-ink-200 shrink-0">
        {isMoE ? "moe" : "residual"}
      </span>
      <span className="text-ink-50 font-semibold truncate">{name}</span>
      {!isMoE ? <span className="text-ink-500 shrink-0">@ {siteKind}</span> : null}
      <span className="text-ink-500 shrink-0">
        · layers <span className="text-ink-200">{layersStr}</span>
      </span>
      <span className="text-ink-500 shrink-0">
        · tokens <span className="text-ink-200">{tokenLabel}</span>
      </span>
      {isMoE ? (
        <span className="text-ink-500 shrink-0 truncate">
          · record{" "}
          <span className="text-ink-200">
            {(record as Array<Record<string, unknown>>)
              .map((r) => String(r.kind))
              .join(", ")}
          </span>
        </span>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------------ */

function DatasetSnapshot({
  dataset,
  stepDetail,
}: {
  dataset: DatasetPreview;
  stepDetail: StepDetail | null;
}) {
  const src = dataset.source;
  const rows = dataset.rows ?? [];
  const promptColumn =
    src?.prompt_column ??
    (stepDetail?.spec?.dataset as Record<string, unknown> | undefined)?.prompt_column;
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-[6rem_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-2xs font-mono">
        <KVRow label="kind" value={src?.kind ?? "—"} />
        {src?.name ? <KVRow label="name" value={src.name} /> : null}
        {typeof promptColumn === "string" ? (
          <KVRow label="prompt_col" value={promptColumn} />
        ) : null}
        {src?.table ? <KVRow label="table" value={src.table} /> : null}
        {src?.env_var ? <KVRow label="env" value={src.env_var} /> : null}
        {dataset.resolved_from_step ? (
          <KVRow label="from_step" value={dataset.resolved_from_step} />
        ) : null}
      </div>
      {rows.length > 0 ? (
        <div>
          <div className="field-label mb-1">sample prompts</div>
          <ul className="space-y-1.5">
            {rows.map((r) => (
              <PromptTile
                key={r.example_key}
                exampleKey={r.example_key}
                caseKey={r.case_key ?? null}
                text={r.prompt_preview}
              />
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------------ */

const PROMPT_COLLAPSED_CHARS = 320;

function PromptTile({
  exampleKey,
  caseKey,
  text,
}: {
  exampleKey: string;
  caseKey: string | null;
  text: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const needsToggle = text.length > PROMPT_COLLAPSED_CHARS;
  const shown = expanded || !needsToggle ? text : truncate(text, PROMPT_COLLAPSED_CHARS);
  return (
    <li className="border border-ink-800 bg-ink-950/40">
      <button
        type="button"
        onClick={() => needsToggle && setExpanded((s) => !s)}
        disabled={!needsToggle}
        className={[
          "w-full text-left px-2 py-1.5",
          needsToggle ? "cursor-pointer hover:bg-ink-900/40" : "cursor-default",
        ].join(" ")}
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-2 mb-0.5 text-[0.58rem] font-mono uppercase tracking-widest text-ink-500">
          <span>{exampleKey}</span>
          {caseKey ? (
            <>
              <span>·</span>
              <span className="text-ink-400 normal-case">{caseKey}</span>
            </>
          ) : null}
          {needsToggle ? (
            <span className="ml-auto text-ink-600">
              {expanded ? "▾ collapse" : `▸ expand · ${text.length} chars`}
            </span>
          ) : null}
        </div>
        <div
          className={[
            "mono text-[0.7rem] text-ink-200 whitespace-pre-wrap break-words leading-snug",
            expanded ? "max-h-[32rem] overflow-auto" : "",
          ].join(" ")}
        >
          {shown}
        </div>
      </button>
    </li>
  );
}

function LabelLine({ dist }: { dist: LabelDistribution }) {
  if (dist.numeric_summary) {
    const ns = dist.numeric_summary;
    return (
      <div className="flex items-baseline gap-2 text-[0.7rem] font-mono">
        <span className="text-ink-50 font-semibold truncate min-w-0">{dist.label_name}</span>
        <span className="text-ink-500 shrink-0 ml-auto tabular-nums">
          μ={ns.mean.toFixed(3)} · σ={ns.stddev.toFixed(3)} · [{ns.min.toFixed(2)}, {ns.max.toFixed(2)}]
        </span>
      </div>
    );
  }
  const top = dist.buckets.slice(0, 4);
  const extra = dist.buckets.length - top.length;
  return (
    <div>
      <div className="flex items-baseline gap-2 text-[0.7rem] font-mono">
        <span className="text-ink-50 font-semibold truncate min-w-0">{dist.label_name}</span>
        <span className="text-ink-500 shrink-0 text-[0.625rem]">
          {dist.unique_values} unique
        </span>
      </div>
      <div className="mt-0.5 flex items-center gap-1 flex-wrap text-[0.625rem] font-mono">
        {top.map((b) => (
          <span
            key={b.value}
            className="inline-flex items-center gap-1 border border-ink-800 bg-ink-950/40 px-1.5 py-0.5 rounded-[2px]"
          >
            <span className="text-ink-200 truncate max-w-[10rem]">{b.value}</span>
            <span className="text-ink-500 tabular-nums">
              {(b.fraction * 100).toFixed(0)}%
            </span>
          </span>
        ))}
        {extra > 0 ? <span className="text-ink-600">+{extra}</span> : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------ */

function DepChips({
  names,
  linkBase,
  fallback,
}: {
  names: string[];
  linkBase: string;
  fallback: string;
}) {
  if (names.length === 0) return <span className="text-ink-600">{fallback}</span>;
  return (
    <div className="flex flex-wrap gap-1">
      {names.map((d) => (
        <Link
          key={d}
          to={linkBase}
          className="chip chip-muted text-ink-200 hover:text-accent hover:border-accent/50"
        >
          {d}
        </Link>
      ))}
    </div>
  );
}

function CollapsibleDetails({
  step,
  runId,
}: {
  step: StepSummary;
  runId: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-dashed border-ink-800 pt-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 px-1 text-[0.58rem] font-mono uppercase tracking-[0.18em] text-ink-500 hover:text-ink-100"
      >
        <span>{open ? "▾ details" : "▸ details"}</span>
        <span className="text-ink-600">timing · hashes · artifact ids</span>
      </button>
      {open ? (
        <dl className="mt-2 grid grid-cols-[9rem_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-2xs font-mono">
          <DetailRow label="started" value={step.started_at ?? "—"} />
          <DetailRow label="finished" value={step.finished_at ?? "—"} />
          <DetailRow
            label="duration"
            value={formatDuration(step.started_at, step.finished_at)}
          />
          <DetailRow label="runner" value={step.runner} />
          <DetailRow label="step_index" value={String(step.step_index)} />
          <DetailRow
            label="semantic_hash"
            value={shortHash(step.step_semantic_hash, 20)}
          />
          <DetailRow label="spec_hash" value={shortHash(step.step_spec_hash, 20)} />
          {step.runtime_app_id ? (
            <DetailRow label="runtime_app_id" value={step.runtime_app_id} />
          ) : null}
          {step.reused_from_run_id ? (
            <DetailRow label="reused_from_run_id" value={step.reused_from_run_id} />
          ) : null}
          {step.artifact_id ? (
            <DetailRow label="artifact_id" value={step.artifact_id} />
          ) : null}
          {step.artifact_kind ? (
            <DetailRow label="artifact_kind" value={step.artifact_kind} />
          ) : null}
          {step.depends_on.length > 0 ? (
            <>
              <dt className="field-label pt-0.5">declared_deps</dt>
              <dd className="flex flex-wrap gap-1">
                {step.depends_on.map((d) => (
                  <span key={d} className="chip chip-muted text-ink-200">
                    {d}
                  </span>
                ))}
              </dd>
            </>
          ) : null}
          <DetailRow label="run_id" value={runId} />
        </dl>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------------ */

function Panel({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <section>
      <div className="flex items-baseline gap-2 mb-1">
        <div className="field-label">{title}</div>
        {hint ? (
          <div className="text-[0.58rem] font-mono text-ink-600 tracking-widest">{hint}</div>
        ) : null}
      </div>
      <div className="border border-ink-800 bg-ink-900 rounded-sm p-2">{children}</div>
    </section>
  );
}

function KVRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <>
      <dt className="field-label">{label}</dt>
      <dd className="text-ink-200 break-words">{value ?? "—"}</dd>
    </>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="field-label break-all">{label}</dt>
      <dd className="text-ink-200 font-mono break-all">{value}</dd>
    </>
  );
}

function Note({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "warn" }) {
  return (
    <div
      className={`text-2xs font-mono ${tone === "warn" ? "text-status-warn" : "text-ink-500"}`}
    >
      {children}
    </div>
  );
}


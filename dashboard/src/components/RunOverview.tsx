import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import type {
  DatasetPreview,
  ResultPreview,
  RunDetail,
  StepSummary,
} from "@/types/api";
import { api } from "@/lib/api";
import { familyAccent, statusDotClass } from "@/lib/status";
import { formatDuration, formatRelative } from "@/lib/format";

/**
 * Run-level overview: a KPI strip at the top, then dense visual tiles
 * grouped by step family. No prose walls, no outline nav — it reads like
 * a hardware ops dashboard, not a reading view.
 */
export function RunOverview({ detail }: { detail: RunDetail }) {
  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-6xl mx-auto p-4 space-y-5">
        <RunKpiStrip detail={detail} />
        <CapturesSection detail={detail} />
        <DatasetsSection detail={detail} />
        <ReadoutsSection detail={detail} families={["readout"]} heading="readouts" />
        <ReadoutsSection
          detail={detail}
          families={["representation"]}
          heading="representations"
        />
        <DeriveRow detail={detail} />
        <ReportSection detail={detail} />
      </div>
    </div>
  );
}

/* =========================================================================
 * KPI STRIP
 * ========================================================================= */

function RunKpiStrip({ detail }: { detail: RunDetail }) {
  const counts = detail.run.step_counts;
  const reused = counts.reused;
  const total = Math.max(1, counts.total);
  const progress = (counts.completed + counts.reused) / total;
  const families = detail.nodes.reduce<Record<string, number>>((acc, node) => {
    if (!node.family) return acc;
    acc[node.family] = (acc[node.family] ?? 0) + 1;
    return acc;
  }, {});
  return (
    <section className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-5 gap-2">
      <KpiTile
        label="status"
        valueNode={
          <span className="flex items-center gap-2">
            <span className={`dot ${statusDotClass(detail.run.status)}`} />
            <span className="uppercase tracking-widest text-[0.72rem]">
              {detail.run.status}
            </span>
          </span>
        }
        footer={
          detail.run.error ? (
            <span className="text-status-fail truncate" title={detail.run.error}>
              {detail.run.error}
            </span>
          ) : (
            formatRelative(detail.run.started_at)
          )
        }
      />
      <KpiTile
        label="duration"
        value={formatDuration(detail.run.started_at, detail.run.finished_at)}
        footer={detail.run.finished_at ? "finished" : "in flight"}
      />
      <KpiTile
        label="steps"
        value={String(counts.total)}
        footer={
          <StepProgressBar
            completed={counts.completed}
            reused={counts.reused}
            running={counts.running}
            failed={counts.failed}
            pending={counts.pending}
            total={counts.total}
          />
        }
      />
      <KpiTile
        label="reused"
        value={`${reused}/${counts.total}`}
        footer={`${Math.round(progress * 100)}% done`}
      />
      <KpiTile
        label="families"
        valueNode={
          <span className="flex items-baseline gap-2 flex-wrap">
            {Object.entries(families)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([fam, n]) => {
                const acc = familyAccent(fam);
                return (
                  <span key={fam} className="flex items-baseline gap-1">
                    <span className="text-xl font-semibold tabular-nums leading-none">
                      {n}
                    </span>
                    <span className={`text-[0.58rem] uppercase tracking-widest ${acc.text}`}>
                      {fam}
                    </span>
                  </span>
                );
              })}
          </span>
        }
      />
    </section>
  );
}

function StepProgressBar(props: {
  completed: number;
  reused: number;
  running: number;
  failed: number;
  pending: number;
  total: number;
}) {
  const total = Math.max(1, props.total);
  const seg = (n: number) => `${(n / total) * 100}%`;
  return (
    <div
      className="mt-1 flex h-1.5 overflow-hidden rounded-sm bg-ink-800"
      title={`✓ ${props.completed} · ↻ ${props.reused} · ⟳ ${props.running} · ✗ ${props.failed} · · ${props.pending}`}
    >
      {props.completed > 0 && <span className="bg-status-ok" style={{ width: seg(props.completed) }} />}
      {props.reused > 0 && <span className="bg-status-reuse" style={{ width: seg(props.reused) }} />}
      {props.running > 0 && <span className="bg-status-run animate-pulse" style={{ width: seg(props.running) }} />}
      {props.failed > 0 && <span className="bg-status-fail" style={{ width: seg(props.failed) }} />}
      {props.pending > 0 && <span className="bg-ink-600" style={{ width: seg(props.pending) }} />}
    </div>
  );
}

/* =========================================================================
 * CAPTURES SECTION
 * ========================================================================= */

function CapturesSection({ detail }: { detail: RunDetail }) {
  const captures = detail.steps.filter((s) => s.family === "capture");
  if (captures.length === 0) return null;
  // Captures are the operationally most important part of a run — always
  // one card per row so the layer/token strips read at full width.
  return (
    <Section title={`captures · ${captures.length}`}>
      <div className="space-y-2">
        {captures.map((step) => (
          <CaptureCard key={step.step_name} step={step} runId={detail.run.run_id} />
        ))}
      </div>
    </Section>
  );
}

function CaptureCard({ step, runId }: { step: StepSummary; runId: string }) {
  const q = useQuery({
    queryKey: ["step", runId, step.step_name],
    queryFn: () => api.getStep(runId, step.step_name),
  });
  const datasetQ = useQuery({
    queryKey: ["dataset", runId, step.step_name, 1, "overview-capture"],
    queryFn: () => api.getDatasetPreview(runId, step.step_name, { sample_size: 1 }),
  });
  const detail = q.data;
  const sites = Array.isArray(detail?.spec?.sites)
    ? (detail!.spec!.sites as Array<Record<string, unknown>>)
    : [];
  const numLayers = detectNumLayers(detail?.spec ?? {}, sites);
  const examples =
    datasetQ.data?.total_rows ??
    datasetQ.data?.source?.total_examples ??
    null;
  const totalLayerHits = sites.reduce((acc, site) => {
    const layers = site.layers;
    return acc + (Array.isArray(layers) ? layers.length : 0);
  }, 0);
  const capturesCount =
    examples != null && totalLayerHits > 0 ? examples * totalLayerHits : null;
  return (
    <TileCard step={step}>
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_14rem] gap-3">
        <div className="space-y-2.5 min-w-0">
          {sites.length === 0 ? (
            <SkeletonLine />
          ) : (
            sites.map((site, i) => (
              <SiteMicroVisual
                key={(site.name as string) ?? i}
                site={site}
                numLayers={numLayers}
              />
            ))
          )}
        </div>
        <div className="grid grid-cols-3 xl:grid-cols-1 gap-1.5 text-[0.6rem] font-mono">
          <Stat
            label="examples"
            value={examples != null ? examples.toLocaleString() : "—"}
          />
          <Stat
            label="sites × layers"
            value={`${sites.length} × ${totalLayerHits}`}
          />
          <Stat
            label="tensors"
            value={capturesCount != null ? capturesCount.toLocaleString() : "—"}
            accent
          />
        </div>
      </div>
    </TileCard>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="border border-ink-800 bg-ink-950/40 rounded-sm px-1.5 py-1">
      <div className="field-label truncate">{label}</div>
      <div
        className={[
          "text-[0.85rem] font-semibold tabular-nums leading-none mt-0.5 truncate",
          accent ? "text-accent" : "text-ink-100",
        ].join(" ")}
      >
        {value}
      </div>
    </div>
  );
}

function firstMeaningfulLine(text: string): string {
  for (const line of text.split("\n")) {
    if (line.trim().length > 0) return line;
  }
  return text;
}

function SiteMicroVisual({
  site,
  numLayers,
}: {
  site: Record<string, unknown>;
  numLayers: number;
}) {
  const layers = Array.isArray(site.layers) ? (site.layers as number[]) : [];
  const captured = new Set(layers);
  const isMoE = Array.isArray(site.record);
  const name = (site.name as string) ?? "—";
  const siteKind = (site.site as string | undefined) ?? (isMoE ? "moe" : "residual");
  const tokens = site.tokens as { kind?: string; value?: unknown } | undefined;
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[0.625rem] font-mono mb-0.5">
        <span
          className={[
            "px-1 py-[1px] rounded-[2px] border uppercase tracking-widest text-[0.58rem]",
            isMoE
              ? "border-fuchsia-400/40 text-fuchsia-300"
              : "border-amber-400/40 text-amber-300",
          ].join(" ")}
        >
          {isMoE ? "moe" : "residual"}
        </span>
        <span className="text-ink-100 font-semibold truncate">{name}</span>
        {!isMoE ? <span className="text-ink-500 shrink-0">@ {siteKind}</span> : null}
        <span className="ml-auto text-ink-600 tabular-nums">
          {layers.length}/{numLayers}
        </span>
      </div>
      <LayerStrip numLayers={numLayers} captured={captured} />
      <div className="mt-1 flex items-center gap-1.5 text-[0.58rem] font-mono text-ink-500">
        <TokenStrip tokens={tokens} />
        <span className="text-ink-300 uppercase tracking-wider">{tokenLabel(tokens)}</span>
      </div>
    </div>
  );
}

function LayerStrip({
  numLayers,
  captured,
}: {
  numLayers: number;
  captured: Set<number>;
}) {
  const cells = Array.from({ length: numLayers }, (_, i) => i);
  // Give captured cells extra flex-grow so the layer-number label has room
  // to render legibly even on models with many layers.
  return (
    <div
      className="flex items-end gap-[2px] h-6"
      title={
        captured.size > 0
          ? `layers ${[...captured].sort((a, b) => a - b).join(", ")}`
          : "no layers"
      }
    >
      {cells.map((i) => {
        const hit = captured.has(i);
        return (
          <span
            key={i}
            className={[
              "rounded-sm min-w-[2px] relative",
              hit
                ? "bg-amber-400/80 h-full flex items-center justify-center flex-[3]"
                : "bg-ink-800 h-1/2 flex-1",
            ].join(" ")}
          >
            {hit ? (
              <span className="text-[0.55rem] font-mono font-semibold text-ink-950 tabular-nums leading-none px-[1px]">
                {i}
              </span>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

function TokenStrip({
  tokens,
}: {
  tokens: { kind?: string; value?: unknown } | undefined;
}) {
  const kind = tokens?.kind;
  const cells = 7;
  const selected = new Set<number>();
  if (kind === "last") selected.add(cells - 1);
  else if (kind === "first") selected.add(0);
  else if (kind === "full_sequence") for (let i = 0; i < cells; i++) selected.add(i);
  else if (kind === "section") for (let i = 2; i <= 4; i++) selected.add(i);
  else if (kind === "slice") {
    const v = tokens?.value as { start?: number; stop?: number } | undefined;
    const start = Math.min(Math.max(0, v?.start ?? 0), cells - 1);
    const stop = Math.min(Math.max(start + 1, v?.stop ?? start + 1), cells);
    for (let i = start; i < stop; i++) selected.add(i);
  }
  return (
    <span className="inline-flex items-center gap-[1px]" aria-hidden>
      {Array.from({ length: cells }, (_, i) => (
        <span
          key={i}
          className={[
            "w-1 h-2.5 rounded-[1px]",
            selected.has(i) ? "bg-accent" : "bg-ink-700",
          ].join(" ")}
        />
      ))}
    </span>
  );
}

function tokenLabel(
  tokens: { kind?: string; value?: unknown } | undefined,
): string {
  if (!tokens) return "—";
  if (tokens.kind === "section") return `sec(${String(tokens.value)})`;
  if (tokens.kind === "slice") {
    const v = tokens.value as { start?: number; stop?: number } | undefined;
    return `slice ${v?.start ?? ""}..${v?.stop ?? ""}`;
  }
  return tokens.kind ?? "—";
}

function detectNumLayers(
  spec: Record<string, unknown>,
  sites: Array<Record<string, unknown>>,
): number {
  const engine = spec.engine as Record<string, unknown> | undefined;
  const engineLayers =
    typeof engine?.num_layers === "number" ? (engine.num_layers as number) : null;
  if (engineLayers && engineLayers > 0) return engineLayers;
  let max = 0;
  for (const site of sites) {
    const layers = site.layers;
    if (Array.isArray(layers)) {
      for (const l of layers) if (typeof l === "number" && l > max) max = l;
    }
  }
  return Math.max(max + 1, 8);
}

/* =========================================================================
 * DATASETS SECTION — one card per capture-step dataset, with construction
 * info, SQL / table / selection, and sample prompts.
 * ========================================================================= */

function DatasetsSection({ detail }: { detail: RunDetail }) {
  const captures = detail.steps.filter((s) => s.family === "capture");
  if (captures.length === 0) return null;
  return (
    <Section title={`dataset · ${captures.length}`}>
      <div className="space-y-2">
        {captures.map((step) => (
          <DatasetCard
            key={step.step_name}
            step={step}
            runId={detail.run.run_id}
          />
        ))}
      </div>
    </Section>
  );
}

function DatasetCard({ step, runId }: { step: StepSummary; runId: string }) {
  const q = useQuery({
    queryKey: ["dataset", runId, step.step_name, 5, "overview-dataset"],
    queryFn: () => api.getDatasetPreview(runId, step.step_name, { sample_size: 5 }),
  });
  return (
    <div className="border border-ink-800 bg-ink-900 rounded-sm">
      <TileBanner step={step} />
      <div className="px-3 py-2.5">
        {q.isLoading ? (
          <SkeletonLine />
        ) : !q.data?.available ? (
          <MutedNote>{q.data?.reason ?? "unavailable"}</MutedNote>
        ) : (
          <DatasetBody data={q.data} />
        )}
      </div>
    </div>
  );
}

function DatasetBody({ data }: { data: DatasetPreview }) {
  const src = data.source ?? null;
  const exampleCount =
    data.total_rows ??
    src?.total_examples ??
    (data.rows?.length ?? 0);
  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,20rem)_minmax(0,1fr)] gap-4">
      {/* Left column: identity + construction + SQL */}
      <div className="space-y-2 min-w-0">
        <div className="flex items-baseline gap-3">
          <span className="text-2xl font-semibold text-ink-50 tabular-nums leading-none">
            {src?.deferred ? "deferred" : exampleCount.toLocaleString()}
          </span>
          <span className="text-[0.58rem] font-mono uppercase tracking-widest text-ink-500">
            {src?.deferred ? src?.kind : "examples"}
          </span>
        </div>
        <IdentityList source={src} />
        {src?.query ? (
          <div>
            <div className="field-label mb-1">sql</div>
            <pre className="mono text-[0.625rem] bg-ink-950 border border-ink-800 text-ink-200 whitespace-pre-wrap break-words p-2 rounded-sm max-h-48 overflow-auto">
              {src.query}
            </pre>
          </div>
        ) : null}
        {src?.selection_keys && src.selection_keys.length > 0 ? (
          <div>
            <div className="field-label mb-1">selection keys</div>
            <div className="text-[0.65rem] font-mono text-ink-300 truncate">
              {src.selection_keys.slice(0, 5).join(", ")}
              {src.selection_keys.length > 5
                ? ` · +${src.selection_keys.length - 5}`
                : ""}
            </div>
          </div>
        ) : null}
      </div>
      {/* Right column: sample prompts */}
      <div className="min-w-0">
        <div className="field-label mb-1">
          sample prompts · {data.rows.length}
        </div>
        {data.rows.length === 0 ? (
          <MutedNote>no rows sampled</MutedNote>
        ) : (
          <ul className="space-y-1.5">
            {data.rows.slice(0, 4).map((r) => (
              <PromptPreviewCard
                key={r.example_key}
                exampleKey={r.example_key}
                caseKey={r.case_key}
                text={r.prompt_preview}
                labels={r.labels}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function IdentityList({
  source,
}: {
  source: DatasetPreview["source"] | null | undefined;
}) {
  if (!source) return null;
  // Build an ordered list of KV rows from the richest fields first. Skip
  // anything the workflow didn't actually specify so we don't render em-dash
  // rows.
  const rows: Array<[string, string]> = [];
  if (source.kind) rows.push(["kind", source.kind]);
  if (source.name) rows.push(["name", source.name]);
  if (source.env_var) rows.push(["env_var", source.env_var]);
  if (source.table) rows.push(["table", source.table]);
  if (source.prompt_column) rows.push(["prompt_column", source.prompt_column]);
  if (source.example_key_column) rows.push(["example_key_column", source.example_key_column]);
  if (source.label_columns && source.label_columns.length > 0)
    rows.push(["labels", source.label_columns.join(", ")]);
  if (source.case_columns && source.case_columns.length > 0)
    rows.push(["cases", source.case_columns.join(", ")]);
  if (source.metadata_columns && source.metadata_columns.length > 0)
    rows.push(["metadata", source.metadata_columns.join(", ")]);
  if (source.limit != null) rows.push(["limit", String(source.limit)]);
  if (source.dataset_id) rows.push(["dataset_id", source.dataset_id]);
  if (rows.length === 0) return null;
  return (
    <dl className="grid grid-cols-[9rem_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[0.65rem] font-mono">
      {rows.map(([k, v]) => (
        <div className="contents" key={k}>
          <dt className="field-label">{k}</dt>
          <dd className="text-ink-200 truncate" title={v}>
            {v}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function PromptPreviewCard({
  exampleKey,
  caseKey,
  text,
  labels,
}: {
  exampleKey: string;
  caseKey: string | null;
  text: string;
  labels: Record<string, unknown>;
}) {
  const [open, setOpen] = useState(false);
  const needsToggle = text.length > 220;
  return (
    <li className="border border-ink-800 bg-ink-950/40 rounded-sm">
      <button
        type="button"
        onClick={() => needsToggle && setOpen(!open)}
        disabled={!needsToggle}
        className={[
          "w-full text-left px-2 py-1.5",
          needsToggle ? "cursor-pointer hover:bg-ink-900/60" : "cursor-default",
        ].join(" ")}
      >
        <div className="flex items-center gap-1.5 text-[0.55rem] font-mono uppercase tracking-widest text-ink-600">
          <span>{exampleKey}</span>
          {caseKey ? (
            <>
              <span>·</span>
              <span className="text-ink-400 normal-case">{caseKey}</span>
            </>
          ) : null}
          {needsToggle ? (
            <span className="ml-auto text-ink-600 normal-case tracking-normal">
              {open ? "▾" : `▸ ${text.length} chars`}
            </span>
          ) : null}
        </div>
        <div
          className={[
            "mono text-[0.7rem] text-ink-200 leading-snug mt-0.5",
            open ? "whitespace-pre-wrap break-words max-h-[24rem] overflow-auto" : "",
          ].join(" ")}
          style={
            open
              ? undefined
              : {
                  display: "-webkit-box",
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden",
                }
          }
        >
          {open ? text : firstMeaningfulLine(text)}
        </div>
        {labels && Object.keys(labels).length > 0 ? (
          <div className="mt-1 flex items-center gap-1 flex-wrap">
            {Object.entries(labels).slice(0, 3).map(([k, v]) => (
              <span
                key={k}
                className="px-1 py-[1px] text-[0.55rem] font-mono text-ink-300 border border-ink-700 rounded-[2px]"
                title={`${k}=${String(v)}`}
              >
                {formatLabelPill(v)}
              </span>
            ))}
            {Object.keys(labels).length > 3 ? (
              <span className="text-[0.55rem] font-mono text-ink-600">
                +{Object.keys(labels).length - 3}
              </span>
            ) : null}
          </div>
        ) : null}
      </button>
    </li>
  );
}

function formatLabelPill(v: unknown): string {
  if (v == null) return "null";
  if (typeof v === "string") return v.length > 18 ? `${v.slice(0, 17)}…` : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return "…";
}

/* =========================================================================
 * READOUTS / REPRESENTATIONS SECTIONS
 * ========================================================================= */

function ReadoutsSection({
  detail,
  families,
  heading,
}: {
  detail: RunDetail;
  families: string[];
  heading: string;
}) {
  const steps = detail.steps.filter(
    (s) => s.family !== null && families.includes(s.family),
  );
  if (steps.length === 0) return null;
  return (
    <Section title={heading}>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2">
        {steps.map((step) => (
          <ResultCard key={step.step_name} step={step} runId={detail.run.run_id} />
        ))}
      </div>
    </Section>
  );
}

// Preferred metric patterns, in priority order. Real result shapes rarely
// use bare `balanced_accuracy` — more often it's `best_raw_balanced_accuracy`
// (residualized probe), `within_baseline_balanced_accuracy` (transfer), or
// `mean_balanced_accuracy`. We walk these patterns and keep the first match.
const PRIMARY_METRIC_PATTERNS: RegExp[] = [
  /^(best|overall|mean|test)_balanced_accuracy$/,
  /^balanced_accuracy$/,
  /balanced_accuracy$/,              // e.g. best_raw_balanced_accuracy
  /^(best|overall|mean|test)_auroc$/,
  /^auroc$/,
  /auroc$/,
  /^(best|overall|mean|test)_auprc$/,
  /auprc$/,
  /^(best|overall|mean|test)_accuracy$/,
  /^accuracy$/,
  /accuracy$/,
  /^(best|overall|mean)_f1$/,
  /^f1$/,
  /separation/,
  /explained_variance|variance_explained/,
  /cohen_d/,
  /spearman_r|pearson_r|kendall_tau/,
  /top\d+_accuracy/,
  /mean_reciprocal_rank|mrr$/,
  /(score|loss)$/,
];
const SECONDARY_METRIC_PATTERNS: RegExp[] = [
  /auroc$/,
  /accuracy$/,
  /f1$/,
  /auprc$/,
  /separation/,
];
// Keys we NEVER want to surface as a metric — they're counts / shape info /
// indices / identity, not performance.
const BLOCKED_METRIC_PATTERNS: RegExp[] = [
  /_count$/,
  /^n_(?!egative)/,   // n_folds, n_components etc (but not "negative_*")
  /^num_/,
  /_dim$/,
  /_size$/,
  /^best_layer$/,
  /^best_component$/,
  /^(seed|random_state|schema_version)$/,
  /_id$/,
  /^example_/,
  /^cohort_/,
  /^class_/,
  /^fold_(count|number)$/,
  /^hidden_/,
];
function isBlockedKey(key: string): boolean {
  return BLOCKED_METRIC_PATTERNS.some((r) => r.test(key));
}

function ResultCard({ step, runId }: { step: StepSummary; runId: string }) {
  const q = useQuery({
    queryKey: ["result", runId, step.step_name],
    queryFn: () => api.getStepResult(runId, step.step_name),
  });
  const data = q.data ?? null;
  const picked = pickMetric(data);
  return (
    <TileCard step={step}>
      {q.isLoading ? (
        <SkeletonLine />
      ) : !data?.available ? (
        <MutedNote>{data?.reason ?? "no result"}</MutedNote>
      ) : picked.primary ? (
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-[0.58rem] font-mono uppercase tracking-widest text-ink-500 truncate">
              {picked.primary.key}
            </span>
            {picked.bestLayer != null ? (
              <span className="text-[0.58rem] font-mono uppercase tracking-widest text-ink-600 tabular-nums shrink-0">
                best @ L{picked.bestLayer}
              </span>
            ) : picked.secondary ? (
              <span className="text-[0.58rem] font-mono uppercase tracking-widest text-ink-600 truncate">
                {picked.secondary.key}
              </span>
            ) : null}
          </div>
          <div className="flex items-baseline justify-between gap-2">
            <MetricNumber value={picked.primary.value} />
            {picked.secondary && picked.bestLayer == null ? (
              <MetricNumber value={picked.secondary.value} size="sm" />
            ) : null}
          </div>
          {picked.lines.length > 0 ? (
            <LayerChart
              lines={picked.lines}
              metricKey={picked.yMetricLabel || picked.primary.key}
              bestLayer={picked.bestLayer}
              bestValue={picked.bestValue}
            />
          ) : picked.bars.length > 0 ? (
            <BarChart
              bars={picked.bars}
              metricKey={picked.yMetricLabel || picked.primary.key}
            />
          ) : picked.flatSeries.length > 1 ? (
            <Sparkline
              series={picked.flatSeries}
              label={picked.flatSeriesLabel}
            />
          ) : null}
        </div>
      ) : (
        <ResultDebug data={data} />
      )}
    </TileCard>
  );
}

function MetricNumber({
  value,
  size = "lg",
}: {
  value: number;
  size?: "lg" | "sm";
}) {
  return (
    <span
      className={[
        "font-semibold tabular-nums text-ink-50 leading-none",
        size === "lg" ? "text-2xl" : "text-base text-ink-300",
      ].join(" ")}
    >
      {formatMetric(value)}
    </span>
  );
}

function formatMetric(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Number.isInteger(v)) return String(v);
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(3);
}

const LINE_PALETTE = [
  "rgb(224,164,88)",   // accent amber
  "rgb(110,168,201)",  // sky
  "rgb(163,132,196)",  // purple
  "rgb(212,103,90)",   // hot
  "rgb(127,176,105)",  // green
];

function LayerChart({
  lines,
  metricKey,
  bestLayer,
  bestValue,
}: {
  lines: LayerLine[];
  metricKey: string;
  bestLayer: number | null;
  bestValue: number | null;
}) {
  const nonEmpty = lines.filter((l) => l.points.length > 1);
  if (nonEmpty.length === 0) return null;

  const W = 240;
  const H = 68;
  const padL = 16;
  const padR = 6;
  const padT = 8;
  const padB = 14;

  const allPoints = nonEmpty.flatMap((l) => l.points);
  const xs = allPoints.map((p) => p.layer);
  const ys = allPoints.map((p) => p.value);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yRaw = { min: Math.min(...ys), max: Math.max(...ys) };
  const bounded = yRaw.min >= 0 && yRaw.max <= 1.0001;
  const yMin = bounded ? 0 : yRaw.min - (yRaw.max - yRaw.min) * 0.1;
  const yMax = bounded ? 1 : yRaw.max + (yRaw.max - yRaw.min) * 0.1;
  const span = Math.max(0.0001, yMax - yMin);
  const xSpan = Math.max(1, xMax - xMin);
  const sx = (x: number) => padL + ((x - xMin) / xSpan) * (W - padL - padR);
  const sy = (y: number) => padT + (1 - (y - yMin) / span) * (H - padT - padB);

  const pathFor = (pts: LayerLine["points"]) =>
    pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.layer).toFixed(1)},${sy(p.value).toFixed(1)}`)
      .join(" ");

  const halfY = bounded ? sy(0.5) : null;
  const primaryColor = LINE_PALETTE[0];

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="w-full h-[68px]"
        role="img"
        aria-label={`${metricKey} per layer`}
      >
        <line
          x1={padL}
          x2={W - padR}
          y1={H - padB}
          y2={H - padB}
          stroke="rgb(50,46,39)"
          strokeWidth={0.5}
        />
        {halfY != null ? (
          <line
            x1={padL}
            x2={W - padR}
            y1={halfY}
            y2={halfY}
            stroke="rgb(68,64,58)"
            strokeWidth={0.5}
            strokeDasharray="2 3"
          />
        ) : null}

        {/* primary line also gets a faint area fill so it's visually dominant */}
        {nonEmpty[0] && nonEmpty[0].points.length > 1 ? (
          <path
            d={`${pathFor(nonEmpty[0].points)} L${sx(
              nonEmpty[0].points.at(-1)!.layer,
            ).toFixed(1)},${H - padB} L${sx(nonEmpty[0].points[0].layer).toFixed(1)},${
              H - padB
            } Z`}
            fill="rgb(224 164 88 / 0.10)"
          />
        ) : null}

        {nonEmpty.map((line, i) => (
          <path
            key={line.label}
            d={pathFor(line.points)}
            fill="none"
            stroke={LINE_PALETTE[i % LINE_PALETTE.length]}
            strokeWidth={i === 0 ? 1.4 : 1.1}
            strokeDasharray={line.style === "dashed" ? "3 2" : undefined}
            strokeLinejoin="round"
            opacity={i === 0 ? 1 : 0.9}
          />
        ))}

        {/* points on the primary line only — keeps it scannable */}
        {nonEmpty[0].points.map((p) => (
          <circle
            key={p.layer}
            cx={sx(p.layer)}
            cy={sy(p.value)}
            r={1.5}
            fill={primaryColor}
          >
            <title>{`${nonEmpty[0].label} · L${p.layer} = ${p.value.toFixed(4)}`}</title>
          </circle>
        ))}

        {bestLayer != null && bestValue != null ? (
          <>
            <line
              x1={sx(bestLayer)}
              x2={sx(bestLayer)}
              y1={padT}
              y2={H - padB}
              stroke={primaryColor}
              strokeWidth={0.5}
              strokeDasharray="1 2"
              opacity={0.6}
            />
            <circle
              cx={sx(bestLayer)}
              cy={sy(bestValue)}
              r={2.8}
              fill="none"
              stroke={primaryColor}
              strokeWidth={1.25}
            />
          </>
        ) : null}

        <text
          x={padL - 3}
          y={sy(yMax) + 3}
          fontSize={6}
          fill="rgb(97,92,84)"
          textAnchor="end"
          fontFamily="ui-monospace, monospace"
        >
          {formatTick(yMax)}
        </text>
        <text
          x={padL - 3}
          y={sy(yMin) + 1}
          fontSize={6}
          fill="rgb(97,92,84)"
          textAnchor="end"
          fontFamily="ui-monospace, monospace"
        >
          {formatTick(yMin)}
        </text>
        <text
          x={padL}
          y={H - 2}
          fontSize={6}
          fill="rgb(97,92,84)"
          fontFamily="ui-monospace, monospace"
        >
          L{xMin}
        </text>
        <text
          x={W - padR}
          y={H - 2}
          fontSize={6}
          fill="rgb(97,92,84)"
          textAnchor="end"
          fontFamily="ui-monospace, monospace"
        >
          L{xMax}
        </text>
      </svg>
      {nonEmpty.length > 1 ? (
        <div className="flex items-center gap-2 flex-wrap mt-0.5 text-[0.55rem] font-mono text-ink-500">
          {nonEmpty.map((line, i) => (
            <span key={line.label} className="flex items-center gap-1">
              <span
                className="inline-block w-2.5 h-[2px]"
                style={{
                  background: LINE_PALETTE[i % LINE_PALETTE.length],
                  borderStyle: line.style === "dashed" ? "dashed" : "solid",
                  borderTopWidth: 1,
                  borderColor: LINE_PALETTE[i % LINE_PALETTE.length],
                }}
                aria-hidden
              />
              <span className="truncate max-w-[8rem]" title={line.label}>
                {line.label}
              </span>
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function BarChart({
  bars,
  metricKey,
}: {
  bars: BarPoint[];
  metricKey: string;
}) {
  if (bars.length === 0) return null;
  const max = Math.max(...bars.map((b) => Math.abs(b.value)), 1);
  return (
    <div>
      <ul className="space-y-0.5">
        {bars.map((b) => {
          const frac = Math.max(0.04, Math.min(1, Math.abs(b.value) / max));
          const color =
            b.group === "within"
              ? "bg-amber-400/60"
              : b.group === "transfer"
                ? "bg-sky-400/60"
                : "bg-accent/60";
          return (
            <li
              key={b.label}
              className="grid grid-cols-[minmax(0,1fr)_3.5rem] gap-1.5 items-center text-[0.6rem] font-mono"
            >
              <div>
                <div className="text-ink-400 truncate" title={b.label}>
                  {b.label}
                </div>
                <div className="h-1.5 bg-ink-800 rounded-sm overflow-hidden">
                  <div
                    className={color}
                    style={{ width: `${frac * 100}%`, height: "100%" }}
                  />
                </div>
              </div>
              <div className="text-ink-100 tabular-nums text-right">
                {formatMetric(b.value)}
              </div>
            </li>
          );
        })}
      </ul>
      <div className="text-[0.55rem] font-mono text-ink-600 tracking-widest uppercase mt-1 truncate">
        {metricKey}
      </div>
    </div>
  );
}

function formatTick(v: number): string {
  if (!Number.isFinite(v)) return "";
  if (Math.abs(v) < 10) return v.toFixed(2);
  return v.toFixed(0);
}

function Sparkline({ series, label }: { series: number[]; label: string }) {
  const min = Math.min(...series);
  const max = Math.max(...series);
  const range = max - min || 1;
  return (
    <div>
      <div className="flex items-end gap-[2px] h-5">
        {series.map((v, i) => {
          const h = Math.max(8, ((v - min) / range) * 100);
          return (
            <span
              key={i}
              className="flex-1 min-w-[3px] bg-accent/70 rounded-sm"
              style={{ height: `${h}%` }}
              title={`${label}[${i}] = ${formatMetric(v)}`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[0.58rem] font-mono text-ink-600 tabular-nums mt-0.5">
        <span>{label}</span>
        <span>
          μ {formatMetric(mean(series))} · σ {formatMetric(stddev(series))}
        </span>
      </div>
    </div>
  );
}

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
}
function stddev(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / xs.length);
}

function ResultDebug({ data }: { data: ResultPreview }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="text-[0.6rem] font-mono uppercase tracking-widest text-status-warn hover:text-ink-100"
      >
        {open ? "▾ result keys" : "▸ no metric matched · inspect keys"}
      </button>
      {open ? (
        <pre className="mt-1 mono text-[0.6rem] bg-ink-950 border border-ink-800 text-ink-300 whitespace-pre-wrap break-words p-1.5 rounded-sm max-h-32 overflow-auto">
          {summarizeKeys(data.payload ?? {})}
        </pre>
      ) : null}
    </div>
  );
}

function summarizeKeys(value: unknown, prefix = "", depth = 0): string {
  if (depth > 4) return "";
  if (!isObj(value)) return "";
  const lines: string[] = [];
  for (const [k, v] of Object.entries(value)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "number" || typeof v === "string" || typeof v === "boolean" || v == null) {
      lines.push(`${path} = ${formatDebugValue(v)}`);
    } else if (Array.isArray(v)) {
      lines.push(`${path}[] (${v.length})`);
    } else if (isObj(v)) {
      const nested = summarizeKeys(v, path, depth + 1);
      if (nested) lines.push(nested);
      else lines.push(`${path} {…}`);
    }
    if (lines.length > 24) {
      lines.push("…");
      break;
    }
  }
  return lines.join("\n");
}

function formatDebugValue(v: unknown): string {
  if (v == null) return "null";
  if (typeof v === "string") return v.length > 32 ? `"${v.slice(0, 31)}…"` : `"${v}"`;
  if (typeof v === "number")
    return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}

interface LayerLine {
  label: string;
  points: Array<{ layer: number; value: number }>;
  style: "solid" | "dashed";
}

interface BarPoint {
  label: string;
  value: number;
  group?: string;
}

interface MetricPick {
  primary: { key: string; value: number } | null;
  secondary: { key: string; value: number } | null;
  lines: LayerLine[];            // multi-series layer chart
  yMetricLabel: string;          // y-axis label (shared across lines)
  bars: BarPoint[];              // categorical chart (text_baseline etc)
  bestLayer: number | null;
  bestValue: number | null;
  flatSeries: number[];
  flatSeriesLabel: string;
}

function pickMetric(result: ResultPreview | null): MetricPick {
  const out: MetricPick = {
    primary: null,
    secondary: null,
    lines: [],
    yMetricLabel: "",
    bars: [],
    bestLayer: null,
    bestValue: null,
    flatSeries: [],
    flatSeriesLabel: "",
  };
  if (!result || !result.available) return out;

  // Specialized extractors for known result shapes. They're additive —
  // whatever they can fill in is merged, and the generic walker below fills
  // in anything left over. This keeps us working on shapes that don't exactly
  // match any single reference (e.g. text_baseline in grouped_cv mode,
  // transfer_probe in split_results mode).
  const payload = result.payload ?? {};
  const kind = String(payload.kind ?? "");
  const specialized =
    kind === "probe_result"
      ? chartProbe(payload)
      : kind === "transfer_probe_result"
        ? chartTransfer(payload)
        : kind === "residualized_probe_result"
          ? chartResidualized(payload)
          : kind === "text_baseline_result"
            ? chartTextBaseline(payload)
            : null;
  if (specialized) Object.assign(out, specialized);

  // Prefer the "best" sub-object the result-producing op wrote out explicitly:
  //   summary.per_layer[best_layer] / summary.best / summary.overall / etc.
  // These carry the actual eval numbers; the flat summary often only has
  // shape/identity info like class_count.
  const bestPocket = locateBestPocket(payload);

  const bag = collectScalarMetrics(result, bestPocket);
  const candidateKeys = Object.keys(bag);

  const firstMatch = (patterns: RegExp[], excludeKey: string | null = null) => {
    for (const pattern of patterns) {
      for (const key of candidateKeys) {
        if (key === excludeKey) continue;
        if (pattern.test(key)) return { key, value: bag[key]! };
      }
    }
    return null;
  };

  if (out.primary == null) {
    out.primary = firstMatch(PRIMARY_METRIC_PATTERNS);
  }
  if (out.secondary == null) {
    out.secondary = firstMatch(SECONDARY_METRIC_PATTERNS, out.primary?.key ?? null);
  }

  // Fallback 1: any float in [-1, 1] (looks like an actual metric).
  if (out.primary == null) {
    for (const key of candidateKeys) {
      const value = bag[key];
      if (!Number.isInteger(value) && Math.abs(value) <= 1.0001) {
        out.primary = { key, value };
        break;
      }
    }
  }
  // Fallback 2: any non-blocked float at all (could be a loss, a delta, etc).
  if (out.primary == null) {
    for (const key of candidateKeys) {
      const value = bag[key];
      if (!Number.isInteger(value)) {
        out.primary = { key, value };
        break;
      }
    }
  }
  // Fallback 3: any non-blocked numeric at all. Better to show *something*
  // than to silently swallow the result.
  if (out.primary == null && candidateKeys.length > 0) {
    const key = candidateKeys[0];
    out.primary = { key, value: bag[key] };
  }

  // If specialized didn't produce a chart but we have a primary metric, try
  // the generic per-layer series extraction — covers any op kind with a
  // payload.layers / per_layer shape we don't have a specialized handler for.
  if (out.lines.length === 0 && out.bars.length === 0 && out.primary) {
    const layers = extractLayerSeries(payload, out.primary.key);
    if (layers) {
      out.lines = [
        { label: out.primary.key, points: layers.points, style: "solid" },
      ];
      out.yMetricLabel = out.primary.key;
      const explicitBest = Number(
        (isObj(payload.summary) && (payload.summary as Record<string, unknown>).best_layer) ??
          payload.best_layer,
      );
      if (Number.isFinite(explicitBest)) {
        const match = layers.points.find((p) => p.layer === explicitBest);
        if (match) {
          out.bestLayer = match.layer;
          out.bestValue = match.value;
        }
      }
      if (out.bestLayer == null) {
        let best = layers.points[0];
        for (const p of layers.points) if (p.value > best.value) best = p;
        out.bestLayer = best.layer;
        out.bestValue = best.value;
      }
      if (out.bestValue != null) {
        out.primary = { key: out.primary.key, value: out.bestValue };
      }
    }
  }

  // Generic bar-chart fallback: if the payload contains a flat mapping of
  // category → metric (e.g. text_baseline grouped_cv, per-cohort results),
  // and we couldn't build layer lines, try to synthesize bars.
  if (out.lines.length === 0 && out.bars.length === 0) {
    const bars = extractBarsFromMapping(payload);
    if (bars.length > 0) {
      out.bars = bars.slice(0, 8);
      if (!out.yMetricLabel) out.yMetricLabel = out.primary?.key ?? "value";
    }
  }

  // Fallback per-fold sparkline for results that don't have per-layer data.
  if (
    out.lines.length === 0 &&
    out.bars.length === 0 &&
    out.primary
  ) {
    const flat = locateFlatSeries(payload, out.primary.key);
    if (flat) {
      out.flatSeries = flat.values;
      out.flatSeriesLabel = flat.label;
    }
  }
  return out;
}

/**
 * Find a mapping inside the payload that looks like "label → metric" — e.g.
 * `results.grouped_cv.per_cohort[cohort] = {balanced_accuracy: 0.8}` or
 * `summary.per_cohort = {a: 0.8, b: 0.6}`. Returns a flattened bar list.
 */
function extractBarsFromMapping(payload: Record<string, unknown>): BarPoint[] {
  const bars: BarPoint[] = [];
  const METRIC_KEYS = [
    "balanced_accuracy",
    "accuracy",
    "auroc",
    "f1",
    "separation",
  ];
  const CANDIDATE_NAMES = /^(per_|by_|grouped|cohort|split|direction|class|group)/i;

  const visit = (label: string, value: unknown) => {
    if (isObj(value)) {
      // Case A: { metric: number, ... }
      for (const m of METRIC_KEYS) {
        if (typeof value[m] === "number" && Number.isFinite(value[m] as number)) {
          bars.push({ label, value: value[m] as number, group: metricGroup(label) });
          return;
        }
      }
      // Case B: nested map of sub-label -> {metric}
      for (const [k, v] of Object.entries(value)) {
        if (isObj(v)) visit(`${label} · ${k}`, v);
        else if (typeof v === "number" && Number.isFinite(v) && !isBlockedKey(k)) {
          bars.push({ label: `${label} · ${k}`, value: v, group: metricGroup(label) });
        }
      }
    } else if (typeof value === "number" && Number.isFinite(value)) {
      bars.push({ label, value, group: metricGroup(label) });
    }
  };

  const roots: Array<[string, unknown]> = [];
  const summary = isObj(payload.summary) ? (payload.summary as Record<string, unknown>) : {};
  const results = isObj(payload.results) ? (payload.results as Record<string, unknown>) : {};
  for (const [k, v] of Object.entries({ ...payload, ...summary, ...results })) {
    if (CANDIDATE_NAMES.test(k) && (isObj(v) || typeof v === "number")) roots.push([k, v]);
  }
  for (const [rootLabel, rootValue] of roots) {
    if (isObj(rootValue)) {
      for (const [sub, sv] of Object.entries(rootValue)) {
        visit(`${rootLabel} · ${sub}`, sv);
      }
    } else {
      visit(rootLabel, rootValue);
    }
    if (bars.length > 0) break; // prefer the first root that produced bars
  }
  bars.sort((a, b) => b.value - a.value);
  return bars;
}

function metricGroup(label: string): string {
  if (/within|baseline/i.test(label)) return "within";
  if (/transfer|cross/i.test(label)) return "transfer";
  return "default";
}

interface LayerSeries {
  points: Array<{ layer: number; value: number }>;
  metricKey: string;
  label: string;
}

/* -------------------------------------------------------------------------
 * Specialized chart extractors — modeled after pipelines_v2/reporting/charts/*.
 * Each returns a partial MetricPick with filled lines / bars / primary.
 * ------------------------------------------------------------------------- */

type Specialized = Partial<MetricPick>;

// probe_result: result.layers = [{layer, balanced_accuracy, auroc, accuracy, ...}]
// Render the primary metric as a line across layers, overlay up to two other
// metrics as dashed secondary lines.
function chartProbe(payload: Record<string, unknown>): Specialized | null {
  const layers = Array.isArray(payload.layers) ? (payload.layers as Array<Record<string, unknown>>) : [];
  if (layers.length === 0) return {};
  const metrics = detectLayerMetrics(layers, [
    "balanced_accuracy",
    "auroc",
    "accuracy",
    "selectivity",
  ]);
  if (metrics.length === 0) return {};
  const lines: LayerLine[] = metrics.map((m, i) => ({
    label: m,
    points: pointsForMetric(layers, m),
    style: i === 0 ? "solid" : "dashed",
  }));
  const best = bestFromLine(lines[0], payload);
  return {
    lines,
    yMetricLabel: metrics[0],
    primary: { key: metrics[0], value: best?.value ?? lines[0].points.at(-1)?.value ?? 0 },
    secondary: metrics.length > 1
      ? { key: metrics[1], value: bestFromLine(lines[1], payload)?.value ?? 0 }
      : null,
    bestLayer: best?.layer ?? null,
    bestValue: best?.value ?? null,
  };
}

// residualized_probe_result: rows have raw_* and residualized_* variants.
// Show the raw vs residualized lines on the same axes — divergence is the
// story (signal mediated by the control variable).
function chartResidualized(payload: Record<string, unknown>): Specialized | null {
  const rows = Array.isArray(payload.rows) ? (payload.rows as Array<Record<string, unknown>>) : [];
  if (rows.length === 0) return {};
  const metrics = ["balanced_accuracy", "auroc"].filter((m) =>
    rows.some((r) => typeof r[`raw_${m}`] === "number"),
  );
  if (metrics.length === 0) return {};
  const metric = metrics[0];
  const rawLine: LayerLine = {
    label: "raw",
    points: pointsForMetric(rows, `raw_${metric}`),
    style: "dashed",
  };
  const residLine: LayerLine = {
    label: "residualized",
    points: pointsForMetric(rows, `residualized_${metric}`),
    style: "solid",
  };
  if (residLine.points.length === 0 && rawLine.points.length === 0) return {};
  const bestResid = argmax(residLine.points);
  const bestRaw = argmax(rawLine.points);
  return {
    lines: [residLine, rawLine].filter((l) => l.points.length > 0),
    yMetricLabel: metric,
    primary: bestResid
      ? { key: `residualized_${metric}`, value: bestResid.value }
      : bestRaw
        ? { key: `raw_${metric}`, value: bestRaw.value }
        : null,
    secondary:
      bestResid && bestRaw
        ? {
            key: `Δ residualized−raw`,
            value: bestResid.value - bestRaw.value,
          }
        : null,
    bestLayer: bestResid?.layer ?? bestRaw?.layer ?? null,
    bestValue: bestResid?.value ?? bestRaw?.value ?? null,
  };
}

// transfer_probe_result: layers contain cross_cohort_transfer keyed by
// direction. Plot one line per direction for the primary metric, dashed
// within-baseline reference averaged across cohorts for context.
//
// Also handles split_results (layer-indexed) as a secondary shape, so runs
// that ran in split_results mode still get a chart.
function chartTransfer(payload: Record<string, unknown>): Specialized | null {
  const layers = Array.isArray(payload.layers) ? (payload.layers as Array<Record<string, unknown>>) : [];
  if (layers.length === 0) {
    // No `layers` array at all — let the generic walker handle whatever
    // scalar summary is present. Return an empty specialized so we still
    // flow through the fallback chart logic.
    return {};
  }
  const metric = "balanced_accuracy";
  const byDirection = new Map<string, Array<{ layer: number; value: number }>>();
  const withinByLayer = new Map<number, number[]>();
  // Split_results fallback — direction -> split_name -> [points]
  const bySplit = new Map<string, Array<{ layer: number; value: number }>>();

  for (const layerRow of layers) {
    const layer = Number(layerRow.layer);
    if (!Number.isFinite(layer)) continue;
    const cross = layerRow.cross_cohort_transfer;
    if (isObj(cross)) {
      for (const [direction, entry] of Object.entries(cross)) {
        if (!isObj(entry)) continue;
        const v = entry[metric];
        if (typeof v === "number" && Number.isFinite(v)) {
          if (!byDirection.has(direction)) byDirection.set(direction, []);
          byDirection.get(direction)!.push({ layer, value: v });
        }
      }
    }
    const within = layerRow.within_cohort_baseline;
    if (isObj(within)) {
      const vals: number[] = [];
      for (const [, cohortMetrics] of Object.entries(within)) {
        if (isObj(cohortMetrics) && typeof cohortMetrics[metric] === "number") {
          vals.push(cohortMetrics[metric] as number);
        }
      }
      if (vals.length > 0) withinByLayer.set(layer, vals);
    }
    const splits = layerRow.split_results;
    if (isObj(splits)) {
      for (const [split, entry] of Object.entries(splits)) {
        if (!isObj(entry)) continue;
        const v = entry[metric];
        if (typeof v === "number" && Number.isFinite(v)) {
          if (!bySplit.has(split)) bySplit.set(split, []);
          bySplit.get(split)!.push({ layer, value: v });
        }
      }
    }
  }

  const directionLines: LayerLine[] = [...byDirection.entries()]
    .slice(0, 3)
    .map(([direction, points]) => ({
      label: direction,
      points: points.sort((a, b) => a.layer - b.layer),
      style: "solid" as const,
    }));
  const splitLines: LayerLine[] = [...bySplit.entries()]
    .slice(0, 3)
    .map(([split, points]) => ({
      label: `split · ${split}`,
      points: points.sort((a, b) => a.layer - b.layer),
      style: "solid" as const,
    }));

  const primarySet = directionLines.length > 0 ? directionLines : splitLines;
  if (primarySet.length === 0) {
    // Neither cross-cohort nor split_results present. Try to at least
    // extract a single per-layer series from whatever top-level metric the
    // layers[] have (e.g. a plain `balanced_accuracy` column).
    const anyPoints = pointsForMetric(layers, metric);
    if (anyPoints.length > 1) {
      const best = argmax(anyPoints);
      return {
        lines: [{ label: metric, points: anyPoints, style: "solid" }],
        yMetricLabel: metric,
        primary: best ? { key: metric, value: best.value } : null,
        secondary: null,
        bestLayer: best?.layer ?? null,
        bestValue: best?.value ?? null,
      };
    }
    return {};
  }

  const withinPoints: Array<{ layer: number; value: number }> = [];
  for (const [layer, vals] of [...withinByLayer.entries()].sort((a, b) => a[0] - b[0])) {
    withinPoints.push({
      layer,
      value: vals.reduce((a, b) => a + b, 0) / vals.length,
    });
  }
  const lines: LayerLine[] = [...primarySet];
  if (withinPoints.length > 1) {
    lines.push({ label: "within-cohort", points: withinPoints, style: "dashed" });
  }
  let best: { layer: number; value: number; line: string } | null = null;
  for (const line of primarySet) {
    for (const p of line.points) {
      if (!best || p.value > best.value) best = { ...p, line: line.label };
    }
  }
  return {
    lines,
    yMetricLabel: metric,
    primary: best ? { key: `transfer·${metric}`, value: best.value } : null,
    secondary: null,
    bestLayer: best?.layer ?? null,
    bestValue: best?.value ?? null,
  };
}

// text_baseline_result: no layers. Cover all three text-baseline modes —
// cross_cohort_transfer / split_results / grouped_cv — by scanning for any
// {metric: number} entries inside the results envelope.
function chartTextBaseline(payload: Record<string, unknown>): Specialized | null {
  const results = isObj(payload.results)
    ? (payload.results as Record<string, unknown>)
    : payload;
  const metric = "balanced_accuracy";
  const bars: BarPoint[] = [];

  const within = results.within_cohort_baseline;
  if (isObj(within)) {
    for (const [cohort, entry] of Object.entries(within)) {
      if (isObj(entry) && typeof entry[metric] === "number") {
        bars.push({
          label: `within · ${cohort}`,
          value: entry[metric] as number,
          group: "within",
        });
      }
    }
  }
  const cross = results.cross_cohort_transfer;
  if (isObj(cross)) {
    for (const [direction, entry] of Object.entries(cross)) {
      if (isObj(entry) && typeof entry[metric] === "number") {
        bars.push({
          label: `xfer · ${direction}`,
          value: entry[metric] as number,
          group: "transfer",
        });
      }
    }
  }
  // grouped_cv mode: { grouped_cv: {balanced_accuracy, auroc, ...} } OR
  //                  { grouped_cv: {per_cohort: {a: {ba}, b: {ba}}, overall: {ba}} }
  const grouped = results.grouped_cv;
  if (isObj(grouped)) {
    if (typeof grouped[metric] === "number") {
      bars.push({
        label: "grouped_cv",
        value: grouped[metric] as number,
        group: "default",
      });
    }
    const perCohort = grouped.per_cohort ?? grouped.cohorts;
    if (isObj(perCohort)) {
      for (const [cohort, entry] of Object.entries(perCohort)) {
        if (isObj(entry) && typeof entry[metric] === "number") {
          bars.push({
            label: `cohort · ${cohort}`,
            value: entry[metric] as number,
            group: "within",
          });
        } else if (typeof entry === "number" && Number.isFinite(entry)) {
          bars.push({
            label: `cohort · ${cohort}`,
            value: entry,
            group: "within",
          });
        }
      }
    }
  }
  // split_results: per-split train/test dictionaries.
  const splits = results.split_results;
  if (isObj(splits)) {
    for (const [split, entry] of Object.entries(splits)) {
      if (isObj(entry) && typeof entry[metric] === "number") {
        bars.push({
          label: `split · ${split}`,
          value: entry[metric] as number,
          group: "transfer",
        });
      }
    }
  }
  // Last-ditch for text_baseline: scan any top-level scalar metric (loss,
  // accuracy, f1, etc.) so we at least show a number.
  if (bars.length === 0) {
    const out: Specialized = {};
    // Just let the generic walker pick a scalar — return an empty specialized
    // result so the fallback bar/scalar/chart logic downstream kicks in.
    return out;
  }

  bars.sort((a, b) => b.value - a.value);
  const best = bars[0];
  const firstTransfer = bars.find((b) => b.group === "transfer");
  return {
    bars: bars.slice(0, 8),
    lines: [],
    yMetricLabel: metric,
    primary: { key: best.label, value: best.value },
    secondary:
      firstTransfer && firstTransfer !== best
        ? { key: firstTransfer.label, value: firstTransfer.value }
        : null,
    bestLayer: null,
    bestValue: best.value,
  };
}

function detectLayerMetrics(
  rows: Array<Record<string, unknown>>,
  candidates: string[],
): string[] {
  const out: string[] = [];
  for (const m of candidates) {
    if (rows.some((r) => typeof r[m] === "number" && Number.isFinite(r[m] as number))) {
      out.push(m);
    }
  }
  return out.slice(0, 3);
}

function pointsForMetric(
  rows: Array<Record<string, unknown>>,
  metric: string,
): Array<{ layer: number; value: number }> {
  const out: Array<{ layer: number; value: number }> = [];
  for (const r of rows) {
    const layer = Number(r.layer ?? r.layer_index ?? r.layer_idx);
    const v = r[metric];
    if (Number.isFinite(layer) && typeof v === "number" && Number.isFinite(v)) {
      out.push({ layer, value: v });
    }
  }
  return out.sort((a, b) => a.layer - b.layer);
}

function argmax(
  points: Array<{ layer: number; value: number }>,
): { layer: number; value: number } | null {
  if (points.length === 0) return null;
  let best = points[0];
  for (const p of points) if (p.value > best.value) best = p;
  return best;
}

function bestFromLine(
  line: LayerLine,
  payload: Record<string, unknown>,
): { layer: number; value: number } | null {
  const explicit = Number(
    (isObj(payload.summary) && (payload.summary as Record<string, unknown>).best_layer) ??
      payload.best_layer,
  );
  if (Number.isFinite(explicit)) {
    const m = line.points.find((p) => p.layer === explicit);
    if (m) return m;
  }
  return argmax(line.points);
}

function extractLayerSeries(
  payload: Record<string, unknown>,
  metricKey: string,
): LayerSeries | null {
  const summary = isObj(payload.summary) ? (payload.summary as Record<string, unknown>) : {};
  const candidates: Array<{ label: string; value: unknown }> = [
    { label: "summary.per_layer", value: summary.per_layer },
    { label: "summary.by_layer", value: summary.by_layer },
    { label: "summary.layers", value: summary.layers },
    { label: "per_layer", value: payload.per_layer },
    { label: "by_layer", value: payload.by_layer },
    { label: "layers", value: payload.layers },
    { label: "rows", value: payload.rows },
    { label: "summary.rows", value: summary.rows },
  ];
  for (const { label, value } of candidates) {
    const points = collectLayerPoints(value, metricKey);
    if (points.length > 1) return { points, metricKey, label };
  }
  return null;
}

function collectLayerPoints(
  value: unknown,
  metricKey: string,
): Array<{ layer: number; value: number }> {
  const points: Array<{ layer: number; value: number }> = [];
  if (Array.isArray(value)) {
    for (const entry of value) {
      if (!isObj(entry)) continue;
      const layer = Number(
        entry.layer ?? entry.layer_index ?? entry.layer_idx ?? entry.index,
      );
      const metric = entry[metricKey];
      if (Number.isFinite(layer) && typeof metric === "number" && Number.isFinite(metric)) {
        points.push({ layer, value: metric });
      }
    }
  } else if (isObj(value)) {
    for (const [k, v] of Object.entries(value)) {
      const layer = Number(k);
      if (!Number.isFinite(layer)) continue;
      if (typeof v === "number" && Number.isFinite(v)) {
        points.push({ layer, value: v });
      } else if (isObj(v)) {
        const metric = v[metricKey];
        if (typeof metric === "number" && Number.isFinite(metric)) {
          points.push({ layer, value: metric });
        }
      }
    }
  }
  points.sort((a, b) => a.layer - b.layer);
  return points;
}

function locateFlatSeries(
  payload: Record<string, unknown>,
  metricKey: string,
): { values: number[]; label: string } | null {
  const summary = isObj(payload.summary) ? (payload.summary as Record<string, unknown>) : {};
  const sources: Array<[string, unknown]> = [
    ["folds", payload.folds],
    ["summary.folds", summary.folds],
    ["per_fold", payload.per_fold],
    ["per_class", payload.per_class],
  ];
  for (const [label, candidate] of sources) {
    const values = seriesFromCandidate(candidate, metricKey);
    if (values.length > 1) return { values, label };
  }
  return null;
}

interface Pocket {
  obj: Record<string, unknown>;
  pathLabel: string;
}

function locateBestPocket(payload: Record<string, unknown>): Pocket | null {
  const summary = isObj(payload.summary) ? payload.summary : null;

  // 1. summary.per_layer[best_layer] — the canonical shape for probes.
  if (summary && isObj(summary.per_layer)) {
    const bestLayer = summary.best_layer ?? payload.best_layer;
    if (bestLayer != null) {
      const perLayer = summary.per_layer as Record<string, unknown>;
      const entry =
        perLayer[String(bestLayer)] ?? perLayer[String(Number(bestLayer))];
      if (isObj(entry)) return { obj: entry, pathLabel: `per_layer[${bestLayer}]` };
    }
  }
  // 2. payload.per_layer[best_layer] — same, at the top level.
  if (isObj(payload.per_layer)) {
    const bestLayer = payload.best_layer;
    if (bestLayer != null) {
      const entry = (payload.per_layer as Record<string, unknown>)[String(bestLayer)];
      if (isObj(entry)) return { obj: entry, pathLabel: `per_layer[${bestLayer}]` };
    }
  }
  // 3. Explicit `best` / `overall` sub-objects.
  for (const key of ["best", "overall", "aggregate", "headline"]) {
    const candidate = summary?.[key] ?? payload[key];
    if (isObj(candidate)) return { obj: candidate, pathLabel: `summary.${key}` };
  }
  return null;
}

function collectScalarMetrics(
  result: ResultPreview,
  bestPocket: Pocket | null,
): Record<string, number> {
  const out: Record<string, number> = {};
  const seen = new WeakSet<object>();

  const ingest = (record: unknown) => {
    if (!isObj(record)) return;
    if (seen.has(record)) return;
    seen.add(record);
    for (const [k, v] of Object.entries(record)) {
      if (typeof v === "number" && Number.isFinite(v) && !isBlockedKey(k) && out[k] == null) {
        out[k] = v;
      }
    }
  };

  // Recursive walk: visit every nested mapping in the tree and ingest its
  // scalar leaves. Avoids missing metrics that live deep in `summary.X.Y.Z`.
  const walk = (value: unknown) => {
    if (!isObj(value)) {
      if (Array.isArray(value)) {
        for (const item of value) walk(item);
      }
      return;
    }
    if (seen.has(value)) return;
    ingest(value);
    for (const v of Object.values(value)) walk(v);
  };

  // Priority order: best pocket first (most meaningful), then headline,
  // then a full recursive walk of the payload.
  if (bestPocket) walk(bestPocket.obj);
  walk(result.headline ?? null);
  if (isObj(result.payload)) {
    // Re-walk specific high-value pockets first so they win over deeper noise.
    walk(result.payload.metrics);
    walk((result.payload.summary as unknown));
    walk(result.payload);
  }
  return out;
}

function seriesFromCandidate(candidate: unknown, metricKey: string): number[] {
  if (Array.isArray(candidate)) {
    if (candidate.every((v) => typeof v === "number")) return candidate as number[];
    const out: number[] = [];
    for (const entry of candidate) {
      if (isObj(entry)) {
        const v = entry[metricKey];
        if (typeof v === "number" && Number.isFinite(v)) out.push(v);
      }
    }
    return out;
  }
  if (isObj(candidate)) {
    // Map shape: {layer_or_class_key: {metric_key: num}} or {k: num}.
    const keys = Object.keys(candidate).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });
    const out: number[] = [];
    for (const k of keys) {
      const v = (candidate as Record<string, unknown>)[k];
      if (typeof v === "number" && Number.isFinite(v)) out.push(v);
      else if (isObj(v)) {
        const inner = v[metricKey];
        if (typeof inner === "number" && Number.isFinite(inner)) out.push(inner);
      }
    }
    return out;
  }
  return [];
}

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/* =========================================================================
 * DERIVE ROW
 * ========================================================================= */

function DeriveRow({ detail }: { detail: RunDetail }) {
  const steps = detail.steps.filter((s) => s.family === "derive");
  if (steps.length === 0) return null;
  return (
    <Section title="derivations">
      <div className="flex flex-wrap gap-2">
        {steps.map((step) => (
          <div
            key={step.step_name}
            className="border border-ink-800 bg-ink-900 px-2 py-1 flex items-center gap-2 text-[0.7rem] font-mono rounded-sm"
          >
            <span className={`dot ${statusDotClass(step.status)}`} />
            <span className="text-ink-100 font-semibold">{step.step_name}</span>
            <span className="text-ink-500">· {step.spec_kind}</span>
          </div>
        ))}
      </div>
    </Section>
  );
}

/* =========================================================================
 * REPORT SECTION — figure thumbnails
 * ========================================================================= */

function ReportSection({ detail }: { detail: RunDetail }) {
  const reportSteps = useMemo(
    () =>
      detail.steps.filter(
        (s) => s.artifact_kind === "report" && s.artifact_id != null,
      ),
    [detail],
  );
  if (reportSteps.length === 0) return null;
  return (
    <Section title="report">
      <div className="space-y-2">
        {reportSteps.map((step) => (
          <ReportStripCard
            key={step.step_name}
            step={step}
            runId={detail.run.run_id}
            artifactId={step.artifact_id!}
          />
        ))}
      </div>
    </Section>
  );
}

function ReportStripCard({
  step,
  runId,
  artifactId,
}: {
  step: StepSummary;
  runId: string;
  artifactId: string;
}) {
  const q = useQuery({
    queryKey: ["report", artifactId],
    queryFn: () => api.getReport(artifactId),
  });
  const data = q.data ?? null;
  const primaries = (data?.figures ?? []).filter((f) => f.primary).slice(0, 6);
  return (
    <div className="border border-ink-800 bg-ink-900 rounded-sm">
      <TileBanner step={step} rightAction={<ReportLink runId={runId} artifactId={artifactId} />} />
      <div className="px-3 py-2.5">
        {q.isLoading ? (
          <SkeletonLine />
        ) : !data ? (
          <MutedNote>report unavailable</MutedNote>
        ) : (
          <>
            <div className="flex items-center gap-3 text-[0.625rem] font-mono text-ink-500 mb-2">
              <span className="tabular-nums">{data.figures.length} figures</span>
              <span className="tabular-nums">{data.tables.length} tables</span>
              <span className="tabular-nums">{data.results.length} results</span>
            </div>
            {primaries.length > 0 ? (
              <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-1.5">
                {primaries.map((fig) => (
                  <ReportThumb
                    key={fig.figure_id}
                    artifactId={artifactId}
                    path={fig.path}
                    alt={fig.title ?? fig.figure_id}
                  />
                ))}
              </div>
            ) : (
              <MutedNote>no primary figures yet</MutedNote>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function ReportThumb({
  artifactId,
  path,
  alt,
}: {
  artifactId: string;
  path: string;
  alt: string;
}) {
  const url = api.reportAssetUrl(artifactId, path);
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="block border border-ink-800 bg-ink-950 aspect-[4/3] flex items-center justify-center overflow-hidden group hover:border-accent/60"
      title={alt}
    >
      <img
        src={url}
        alt={alt}
        loading="lazy"
        className="max-w-full max-h-full object-contain opacity-90 group-hover:opacity-100"
      />
    </a>
  );
}

function ReportLink({ runId, artifactId }: { runId: string; artifactId: string }) {
  return (
    <a
      href={`/runs/${runId}/reports/${artifactId}`}
      className="text-[0.65rem] font-mono uppercase tracking-widest text-accent hover:underline"
    >
      open report →
    </a>
  );
}

/* =========================================================================
 * Shared tile + section primitives
 * ========================================================================= */

function TileCard({
  step,
  children,
}: {
  step: StepSummary;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-ink-800 bg-ink-900 rounded-sm flex flex-col">
      <TileBanner step={step} />
      <div className="px-3 py-2.5 flex-1">{children}</div>
    </div>
  );
}

function TileBanner({
  step,
  rightAction,
}: {
  step: StepSummary;
  rightAction?: React.ReactNode;
}) {
  const accent = familyAccent(step.family);
  const failed = step.status === "failed";
  return (
    <div className="flex items-stretch border-b border-ink-800 bg-ink-950/40">
      <span className={`w-[3px] shrink-0 ${accent.bar}`} aria-hidden />
      <div className="flex flex-col gap-0.5 px-2.5 py-1.5 min-w-0 flex-1">
        <div className="flex items-center gap-2 min-w-0">
          <span className="mono text-[0.75rem] font-semibold text-ink-50 break-words min-w-0 flex-1">
            {step.step_name}
          </span>
          {failed ? (
            <span
              className="text-[0.55rem] font-mono uppercase tracking-widest text-status-fail shrink-0"
              title="failed"
            >
              failed
            </span>
          ) : null}
          {rightAction ? <div className="shrink-0">{rightAction}</div> : null}
        </div>
        <span
          className={`text-[0.55rem] font-mono uppercase tracking-widest ${accent.text}`}
        >
          {step.spec_kind ?? step.family}
        </span>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="flex items-center gap-2 mb-2">
        <div className="w-1 h-3.5 bg-accent" />
        <h2 className="mono text-[0.6rem] uppercase tracking-[0.22em] text-ink-300 font-semibold">
          {title}
        </h2>
        <div className="flex-1 h-px bg-ink-800" />
      </div>
      {children}
    </section>
  );
}

function KpiTile({
  label,
  value,
  valueNode,
  footer,
}: {
  label: string;
  value?: string;
  valueNode?: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <div className="border border-ink-800 bg-ink-900 rounded-sm px-3 py-2.5">
      <div className="field-label">{label}</div>
      <div className="mt-1 text-ink-50 font-semibold tabular-nums leading-none min-h-[1.5rem]">
        {valueNode ?? (
          <span className="text-xl">{value ?? "—"}</span>
        )}
      </div>
      {footer != null ? (
        <div className="mt-1.5 text-[0.625rem] font-mono text-ink-500 truncate">
          {footer}
        </div>
      ) : null}
    </div>
  );
}

function SkeletonLine() {
  return <div className="h-4 w-3/5 bg-ink-850 animate-pulse rounded-sm" />;
}

function MutedNote({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-[0.65rem] font-mono text-ink-500">{children}</span>
  );
}

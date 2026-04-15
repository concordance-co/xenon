import { useQuery } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type {
  DatasetPreview,
  LabelDistribution,
  LabelDistributionBucket,
  StepDetail,
  StepSummary,
} from "@/types/api";
import { api } from "@/lib/api";
import { formatDuration, shortHash, truncate } from "@/lib/format";

/**
 * Operator-focused overview. Layout is a tile grid rather than a paragraph
 * stack — each panel answers one question visually (where, what, how much,
 * what-do-the-prompts-look-like) instead of assaulting the reader with
 * mono-text walls.
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
  const isCapture = stepDetail?.spec?.kind === "capture";

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

  return (
    <div className="p-3 space-y-3">
      <FlowMap step={step} stepDetail={stepDetail} runId={runId} />

      {isCapture ? (
        <CaptureSitesTile spec={stepDetail!.spec} />
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {hasCapture ? (
          <DatasetTile dataset={dataset} loading={datasetQ.isLoading} />
        ) : null}
        {hasCapture ? (
          <LabelsTile
            labels={labels?.labels ?? []}
            available={labels?.available ?? false}
            reason={labels?.reason ?? null}
            loading={labelQ.isLoading}
          />
        ) : null}
      </div>

      {hasCapture ? (
        <PromptsTile
          rows={dataset?.rows ?? []}
          available={dataset?.available ?? false}
          reason={dataset?.reason ?? null}
          loading={datasetQ.isLoading}
        />
      ) : null}

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

      <CollapsibleDetails step={step} runId={runId} />
    </div>
  );
}

/* =========================================================================
 * FLOW MAP  —  [upstream] ─▶ THIS ─▶ [downstream]
 * ========================================================================= */

function FlowMap({
  step,
  stepDetail,
  runId: _runId,
}: {
  step: StepSummary;
  stepDetail: StepDetail | null;
  runId: string;
}) {
  const upstream = step.resolved_depends_on ?? [];
  const downstream = (stepDetail?.downstream ?? []).map((d) => d.step_name);
  if (upstream.length === 0 && downstream.length === 0) return null;
  return (
    <Tile>
      <TileHeader label="flow" />
      <div className="flex items-center gap-1.5 flex-wrap text-[0.7rem] font-mono">
        <FlowChips names={upstream} align="right" />
        <Arrow hidden={upstream.length === 0} />
        <span
          className="px-2 py-1 bg-accent/15 border border-accent text-accent rounded-[2px] font-semibold truncate max-w-[16rem]"
          title={step.step_name}
        >
          {step.step_name}
        </span>
        <Arrow hidden={downstream.length === 0} />
        <FlowChips names={downstream} align="left" />
      </div>
    </Tile>
  );
}

function FlowChips({ names, align }: { names: string[]; align: "left" | "right" }) {
  if (names.length === 0) return null;
  const visible = names.slice(0, 3);
  const rest = names.length - visible.length;
  const nodes = visible.map((n) => (
    <span
      key={n}
      className="px-1.5 py-0.5 border border-ink-700 text-ink-200 rounded-[2px] truncate max-w-[10rem]"
      title={n}
    >
      {n}
    </span>
  ));
  const more =
    rest > 0 ? (
      <span key="more" className="text-ink-500 text-[0.625rem]">
        +{rest}
      </span>
    ) : null;
  return (
    <div
      className={[
        "flex items-center gap-1 flex-wrap min-w-0",
        align === "right" ? "justify-end" : "justify-start",
      ].join(" ")}
    >
      {nodes}
      {more}
    </div>
  );
}

function Arrow({ hidden }: { hidden?: boolean }) {
  if (hidden) return null;
  return <span className="text-ink-600 select-none shrink-0">─▶</span>;
}

/* =========================================================================
 * CAPTURE SITES — visual layer strip + token strip
 * ========================================================================= */

type Site = Record<string, unknown>;

function CaptureSitesTile({ spec }: { spec: Record<string, unknown> }) {
  const sitesRaw = spec.sites;
  if (!Array.isArray(sitesRaw) || sitesRaw.length === 0) return null;
  const sites = sitesRaw as Site[];
  const numLayers = detectNumLayers(spec, sites);
  return (
    <Tile>
      <TileHeader
        label="capture sites"
        meta={`${sites.length} site${sites.length === 1 ? "" : "s"}`}
      />
      <ul className="space-y-2.5">
        {sites.map((site, i) => (
          <li key={(site.name as string) ?? i}>
            <SiteVisual site={site} numLayers={numLayers} />
          </li>
        ))}
      </ul>
    </Tile>
  );
}

function detectNumLayers(spec: Record<string, unknown>, sites: Site[]): number {
  const engine = spec.engine as Record<string, unknown> | undefined;
  const engineLayers =
    typeof engine?.num_layers === "number" ? (engine.num_layers as number) : null;
  if (engineLayers && engineLayers > 0) return engineLayers;
  let max = 0;
  for (const site of sites) {
    const layers = site.layers;
    if (Array.isArray(layers)) {
      for (const l of layers) {
        if (typeof l === "number" && l > max) max = l;
      }
    }
  }
  return Math.max(max + 1, 8);
}

function SiteVisual({ site, numLayers }: { site: Site; numLayers: number }) {
  const name = (site.name as string) ?? "—";
  const layers = Array.isArray(site.layers) ? (site.layers as number[]) : [];
  const tokens = site.tokens as { kind?: string; value?: unknown } | undefined;
  const record = site.record as unknown;
  const isMoE = Array.isArray(record);
  const siteKind = (site.site as string | undefined) ?? (isMoE ? "moe" : "residual");
  const captured = new Set(layers);
  return (
    <div>
      <div className="flex items-center gap-2 flex-wrap text-[0.7rem] font-mono mb-1">
        <span
          className={[
            "px-1.5 py-0.5 rounded-[2px] border text-[0.58rem] uppercase tracking-widest",
            isMoE
              ? "border-fuchsia-400/40 text-fuchsia-300"
              : "border-amber-400/40 text-amber-300",
          ].join(" ")}
        >
          {isMoE ? "moe" : "residual"}
        </span>
        <span className="text-ink-50 font-semibold truncate">{name}</span>
        {!isMoE ? <span className="text-ink-500">@ {siteKind}</span> : null}
        <span className="ml-auto text-ink-600 text-[0.625rem]">
          {layers.length}/{numLayers} layers
        </span>
      </div>
      <LayerStrip numLayers={numLayers} captured={captured} />
      <div className="mt-1.5 flex items-center gap-3 text-[0.65rem] font-mono text-ink-400">
        <span className="flex items-center gap-1.5">
          <span className="field-label">tokens</span>
          <TokenStrip tokens={tokens} />
          <span className="text-ink-300">{tokenLabel(tokens)}</span>
        </span>
        {isMoE ? (
          <span className="flex items-center gap-1 flex-wrap">
            <span className="field-label">record</span>
            {(record as Array<Record<string, unknown>>).map((r, i) => (
              <span
                key={i}
                className="px-1 py-0.5 border border-ink-700 rounded-[2px] text-[0.58rem] text-ink-300"
              >
                {String(r.kind)}
              </span>
            ))}
          </span>
        ) : null}
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
              "rounded-sm min-w-[3px] relative",
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
  // Schematic 7-cell strip showing which token positions the selector picks.
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
            "w-1.5 h-3 rounded-[1px]",
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
  if (tokens.kind === "section") return `section("${String(tokens.value)}")`;
  if (tokens.kind === "slice") {
    const v = tokens.value as { start?: number; stop?: number } | undefined;
    return `slice(${v?.start ?? ""}..${v?.stop ?? ""})`;
  }
  return tokens.kind ?? "—";
}

/* =========================================================================
 * DATASET TILE — iconic, not encyclopedic
 * ========================================================================= */

function DatasetTile({
  dataset,
  loading,
}: {
  dataset: DatasetPreview | null;
  loading: boolean;
}) {
  return (
    <Tile>
      <TileHeader label="dataset" meta={dataset?.source?.name ?? null} />
      {loading ? (
        <Note>loading…</Note>
      ) : !dataset?.available ? (
        <Note tone="warn">{dataset?.reason ?? "unavailable"}</Note>
      ) : (
        <DatasetBody dataset={dataset} />
      )}
    </Tile>
  );
}

function DatasetBody({ dataset }: { dataset: DatasetPreview }) {
  const src = dataset.source;
  const count =
    dataset.total_rows ??
    src?.total_examples ??
    (src?.deferred ? null : dataset.rows.length);
  const countLabel = src?.deferred
    ? "deferred"
    : count != null
      ? count.toLocaleString()
      : "—";
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-3">
        <span className="text-xl font-semibold text-ink-50 tabular-nums leading-none">
          {countLabel}
        </span>
        <span className="text-[0.625rem] font-mono uppercase tracking-widest text-ink-500">
          {src?.deferred ? src.kind : "examples"}
        </span>
      </div>
      <div className="text-[0.65rem] font-mono text-ink-400 space-y-0.5">
        {src?.kind ? <div>source · <span className="text-ink-200">{src.kind}</span></div> : null}
        {src?.env_var ? (
          <div>env · <span className="text-ink-200">{src.env_var}</span></div>
        ) : null}
        {src?.table ? (
          <div>table · <span className="text-ink-200">{src.table}</span></div>
        ) : null}
        {src?.prompt_column ? (
          <div>prompt_col · <span className="text-ink-200">{src.prompt_column}</span></div>
        ) : null}
        {dataset.resolved_from_step ? (
          <div>
            from_step · <span className="text-ink-200">{dataset.resolved_from_step}</span>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* =========================================================================
 * LABELS TILE — real bar charts
 * ========================================================================= */

function LabelsTile({
  labels,
  available,
  reason,
  loading,
}: {
  labels: LabelDistribution[];
  available: boolean;
  reason: string | null;
  loading: boolean;
}) {
  return (
    <Tile>
      <TileHeader
        label="labels"
        meta={available ? `${labels.length} field${labels.length === 1 ? "" : "s"}` : null}
      />
      {loading ? (
        <Note>loading…</Note>
      ) : !available ? (
        <Note tone="warn">{reason ?? "unavailable"}</Note>
      ) : labels.length === 0 ? (
        <Note>no labels on the sampled rows</Note>
      ) : (
        <ul className="space-y-2.5">
          {labels.map((l) => (
            <li key={l.label_name}>
              <LabelBar dist={l} />
            </li>
          ))}
        </ul>
      )}
    </Tile>
  );
}

const BUCKET_PALETTE = [
  "bg-amber-400/80",
  "bg-amber-400/50",
  "bg-amber-400/30",
  "bg-amber-400/20",
  "bg-amber-400/10",
];

function LabelBar({ dist }: { dist: LabelDistribution }) {
  if (dist.numeric_summary) {
    const ns = dist.numeric_summary;
    const { min, max, mean } = ns;
    const pct = max > min ? ((mean - min) / (max - min)) * 100 : 50;
    return (
      <div>
        <div className="flex items-baseline justify-between gap-2 text-[0.7rem] font-mono">
          <span className="text-ink-50 font-semibold truncate">{dist.label_name}</span>
          <span className="text-ink-500 tabular-nums text-[0.625rem]">
            μ={ns.mean.toFixed(3)} · σ={ns.stddev.toFixed(3)}
          </span>
        </div>
        <div className="relative h-1.5 mt-1 bg-ink-800 rounded-sm">
          <span
            className="absolute top-0 h-full w-[2px] bg-accent"
            style={{ left: `${Math.max(0, Math.min(100, pct))}%` }}
            aria-hidden
          />
        </div>
        <div className="flex justify-between text-[0.58rem] font-mono text-ink-600 tabular-nums mt-0.5">
          <span>{min.toFixed(2)}</span>
          <span>{max.toFixed(2)}</span>
        </div>
      </div>
    );
  }
  const top = dist.buckets.slice(0, 4);
  const extra = dist.buckets.length - top.length;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2 text-[0.7rem] font-mono">
        <span className="text-ink-50 font-semibold truncate">{dist.label_name}</span>
        <span className="text-ink-500 text-[0.625rem]">{dist.unique_values} unique</span>
      </div>
      <StackedBar buckets={dist.buckets} />
      <ul className="mt-1.5 space-y-0.5">
        {top.map((b, i) => (
          <BucketRow key={b.value} bucket={b} color={BUCKET_PALETTE[i] ?? BUCKET_PALETTE[BUCKET_PALETTE.length - 1]} />
        ))}
        {extra > 0 ? (
          <li className="text-[0.58rem] font-mono text-ink-600">+{extra} more</li>
        ) : null}
      </ul>
    </div>
  );
}

function StackedBar({ buckets }: { buckets: LabelDistributionBucket[] }) {
  if (buckets.length === 0) return null;
  return (
    <div
      className="flex h-2 mt-1 overflow-hidden rounded-sm bg-ink-800"
      role="img"
      aria-label="label distribution"
    >
      {buckets.map((b, i) => (
        <span
          key={b.value}
          className={BUCKET_PALETTE[i] ?? BUCKET_PALETTE[BUCKET_PALETTE.length - 1]}
          style={{ width: `${Math.max(1, b.fraction * 100)}%` }}
          title={`${b.value} · ${(b.fraction * 100).toFixed(1)}% (${b.count})`}
        />
      ))}
    </div>
  );
}

function BucketRow({
  bucket,
  color,
}: {
  bucket: LabelDistributionBucket;
  color: string;
}) {
  return (
    <li className="flex items-center gap-2 text-[0.625rem] font-mono">
      <span className={`w-1.5 h-1.5 rounded-sm shrink-0 ${color}`} aria-hidden />
      <span className="text-ink-200 truncate flex-1 min-w-0" title={bucket.value}>
        {bucket.value}
      </span>
      <span className="text-ink-500 tabular-nums shrink-0">
        {(bucket.fraction * 100).toFixed(0)}%
      </span>
    </li>
  );
}

/* =========================================================================
 * PROMPTS TILE — compact cards, click to expand
 * ========================================================================= */

function PromptsTile({
  rows,
  available,
  reason,
  loading,
}: {
  rows: DatasetPreview["rows"];
  available: boolean;
  reason: string | null;
  loading: boolean;
}) {
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <Tile>
      <TileHeader
        label="sample prompts"
        meta={available ? `${rows.length}` : null}
      />
      {loading ? (
        <Note>loading…</Note>
      ) : !available ? (
        <Note tone="warn">{reason ?? "unavailable"}</Note>
      ) : rows.length === 0 ? (
        <Note>no rows sampled</Note>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
          {rows.map((r) => (
            <PromptCard
              key={r.example_key}
              exampleKey={r.example_key}
              caseKey={r.case_key ?? null}
              text={r.prompt_preview}
              labels={r.labels ?? {}}
              expanded={expanded === r.example_key}
              onToggle={() =>
                setExpanded(expanded === r.example_key ? null : r.example_key)
              }
            />
          ))}
        </div>
      )}
    </Tile>
  );
}

function PromptCard({
  exampleKey,
  caseKey,
  text,
  labels,
  expanded,
  onToggle,
}: {
  exampleKey: string;
  caseKey: string | null;
  text: string;
  labels: Record<string, unknown>;
  expanded: boolean;
  onToggle: () => void;
}) {
  const preview = firstLine(text, 120);
  const labelPills = Object.entries(labels).slice(0, 3);
  return (
    <div
      className={[
        "border transition-colors rounded-[2px]",
        expanded
          ? "col-span-full border-accent/60 bg-ink-950/60"
          : "border-ink-800 bg-ink-950/40 hover:border-ink-600",
      ].join(" ")}
    >
      <button
        type="button"
        onClick={onToggle}
        className="w-full text-left px-2 py-1.5 focus:outline-none focus-visible:ring-1 focus-visible:ring-accent"
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-1.5 text-[0.58rem] font-mono uppercase tracking-widest text-ink-500">
          <span>{exampleKey}</span>
          {caseKey ? (
            <>
              <span>·</span>
              <span className="text-ink-400 normal-case">{caseKey}</span>
            </>
          ) : null}
          <span className="ml-auto text-ink-600">{expanded ? "▾" : "▸"}</span>
        </div>
        <div
          className={[
            "mono text-[0.7rem] text-ink-200 leading-snug mt-1",
            expanded ? "whitespace-pre-wrap break-words" : "truncate",
          ].join(" ")}
        >
          {expanded ? text : preview}
        </div>
        {labelPills.length > 0 ? (
          <div className="mt-1 flex items-center gap-1 flex-wrap">
            {labelPills.map(([k, v]) => (
              <span
                key={k}
                className="px-1 py-[1px] text-[0.58rem] font-mono text-ink-300 border border-ink-700 rounded-[2px]"
                title={`${k}=${stringify(v)}`}
              >
                {stringify(v)}
              </span>
            ))}
            {Object.keys(labels).length > labelPills.length ? (
              <span className="text-[0.58rem] font-mono text-ink-600">
                +{Object.keys(labels).length - labelPills.length}
              </span>
            ) : null}
          </div>
        ) : null}
      </button>
    </div>
  );
}

function firstLine(text: string, max: number): string {
  const line = text.split("\n").find((l) => l.trim().length > 0) ?? text;
  return truncate(line, max);
}

function stringify(v: unknown): string {
  if (v == null) return "null";
  if (typeof v === "string") return v.length > 16 ? `${v.slice(0, 15)}…` : v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  return "…";
}

/* =========================================================================
 * COLLAPSIBLE DETAILS — hashes, runtime ids, timing (unchanged from before)
 * ========================================================================= */

function CollapsibleDetails({ step, runId }: { step: StepSummary; runId: string }) {
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

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="field-label break-all">{label}</dt>
      <dd className="text-ink-200 font-mono break-all">{value}</dd>
    </>
  );
}

/* =========================================================================
 * Primitives
 * ========================================================================= */

function Tile({ children }: { children: ReactNode }) {
  return (
    <section className="border border-ink-800 bg-ink-900 rounded-sm p-2.5">
      {children}
    </section>
  );
}

function TileHeader({ label, meta }: { label: string; meta?: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-2 mb-1.5">
      <span className="field-label">{label}</span>
      {meta ? (
        <span className="text-[0.58rem] font-mono text-ink-600 tracking-widest truncate">
          {meta}
        </span>
      ) : null}
    </div>
  );
}

function Note({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: "muted" | "warn";
}) {
  return (
    <div
      className={`text-2xs font-mono ${tone === "warn" ? "text-status-warn" : "text-ink-500"}`}
    >
      {children}
    </div>
  );
}


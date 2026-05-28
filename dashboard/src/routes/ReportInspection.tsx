import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { DataTable, columnsForRows, formatCell } from "@/components/DataTable";
import { JsonView } from "@/components/JsonView";
import type { ReportDetail, ReportResult } from "@/types/api";

type JsonRecord = Record<string, unknown>;

type ResultCandidate = {
  path: string;
  name: string;
  stepName: string | null;
  bytes: number;
  reason: string;
};

type LoadedResult = {
  candidate: ResultCandidate;
  payload: unknown;
};

type TraceRow = {
  id: string;
  sourceStep: string;
  caseKey: string;
  labelSummary: string;
  probeScores: JsonRecord;
  scoreSummary: string;
  spanSummary: string;
  prompt: string;
  response: string;
  labels: JsonRecord;
  spans: JsonRecord[];
  raw: unknown;
};

const AUTO_LOAD_BYTES = 8_000_000;
const TRACE_ROW_LIMIT = 2_000;

export function ReportInspection({ data }: { data: ReportDetail }) {
  const [active, setActive] = useState<"traces" | "probes" | "inputs">("traces");
  const [forceLoad, setForceLoad] = useState(false);
  const [search, setSearch] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const inputRows = useMemo(() => buildInputRows(data), [data]);
  const probeRows = useMemo(() => buildProbeRows(data), [data]);
  const behaviorRows = useMemo(() => buildBehaviorRows(data), [data]);
  const candidates = useMemo(() => buildResultCandidates(data), [data]);
  const candidateBytes = candidates.reduce((sum, item) => sum + item.bytes, 0);
  const shouldAutoLoad = candidates.length > 0 && candidateBytes <= AUTO_LOAD_BYTES;
  const shouldLoad = forceLoad || shouldAutoLoad;

  const resultQuery = useQuery({
    queryKey: ["report-inspection-results", data.artifact_id, candidates.map((item) => item.path).join("|")],
    queryFn: () => fetchResultCandidates(data.artifact_id, candidates),
    enabled: candidates.length > 0 && shouldLoad,
  });

  const traceRows = useMemo(() => {
    const loadedRows = (resultQuery.data ?? []).flatMap((item) => buildTraceRowsFromResult(item));
    return [...behaviorRows, ...loadedRows].slice(0, TRACE_ROW_LIMIT);
  }, [behaviorRows, resultQuery.data]);

  const filteredTraceRows = useMemo(() => {
    const needle = search.trim().toLowerCase();
    if (!needle) return traceRows;
    return traceRows.filter((row) =>
      [
        row.id,
        row.sourceStep,
        row.caseKey,
        row.labelSummary,
        row.scoreSummary,
        row.spanSummary,
        row.prompt,
        row.response,
      ]
        .join("\n")
        .toLowerCase()
        .includes(needle),
    );
  }, [search, traceRows]);

  const selectedRow =
    filteredTraceRows.find((row) => row.id === selectedId) ?? filteredTraceRows[0] ?? null;

  if (inputRows.length === 0 && probeRows.length === 0 && candidates.length === 0 && behaviorRows.length === 0) {
    return null;
  }

  return (
    <section>
      <ResearchTitle>research workbench</ResearchTitle>
      <div className="border border-ink-800 bg-ink-900/75">
        <div className="border-b border-ink-800 bg-ink-950/35 p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="field-label">merged report inspection</div>
              <p className="mt-1 max-w-4xl text-xs font-mono leading-relaxed text-ink-300">
                Prompt rows, responses, labels, probe locations, spans, and result tables from the
                copied report bundle.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              <Counter label="inputs" value={inputRows.length} />
              <Counter label="probe rows" value={probeRows.length} />
              <Counter label="trace files" value={candidates.length + behaviorRows.length} />
              <Counter label="trace rows" value={traceRows.length} />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 border-b border-ink-800 bg-ink-900 px-3 py-2">
          {(["traces", "probes", "inputs"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => setActive(tab)}
              className={[
                "px-2.5 py-1 text-[0.65rem] font-mono uppercase tracking-[0.16em] transition-colors",
                active === tab
                  ? "bg-accent/12 text-accent border border-accent/50"
                  : "text-ink-500 hover:text-ink-100 border border-transparent hover:border-ink-700",
              ].join(" ")}
            >
              {tab}
            </button>
          ))}
          {candidates.length > 0 && !shouldLoad ? (
            <button
              type="button"
              onClick={() => setForceLoad(true)}
              className="btn-ghost ml-auto"
              title="load copied generation results into the prompt/response table"
            >
              load traces ({formatBytes(candidateBytes)})
            </button>
          ) : null}
          {resultQuery.isLoading ? (
            <span className="ml-auto text-2xs font-mono text-ink-500">loading trace rows...</span>
          ) : null}
          {resultQuery.error ? (
            <span className="ml-auto text-2xs font-mono text-status-fail">
              trace load failed: {(resultQuery.error as Error).message}
            </span>
          ) : null}
        </div>

        {active === "traces" ? (
          <TraceWorkbench
            candidates={candidates}
            rows={filteredTraceRows}
            selected={selectedRow}
            search={search}
            setSearch={setSearch}
            setSelectedId={setSelectedId}
            shouldLoad={shouldLoad}
            forceLoad={() => setForceLoad(true)}
            candidateBytes={candidateBytes}
          />
        ) : active === "probes" ? (
          <ProbeWorkbench rows={probeRows} />
        ) : (
          <InputWorkbench rows={inputRows} />
        )}
      </div>
    </section>
  );
}

function TraceWorkbench({
  candidates,
  rows,
  selected,
  search,
  setSearch,
  setSelectedId,
  shouldLoad,
  forceLoad,
  candidateBytes,
}: {
  candidates: ResultCandidate[];
  rows: TraceRow[];
  selected: TraceRow | null;
  search: string;
  setSearch: (value: string) => void;
  setSelectedId: (value: string) => void;
  shouldLoad: boolean;
  forceLoad: () => void;
  candidateBytes: number;
}) {
  if (candidates.length > 0 && !shouldLoad) {
    return (
      <div className="p-5">
        <div className="border border-ink-800 bg-ink-950/45 p-5">
          <div className="field-label">prompt / response results</div>
          <p className="mt-2 max-w-2xl text-xs font-mono leading-relaxed text-ink-300">
            This report has copied generation result files. Load them to build the trace table with
            prompts, responses, labels, probe-score columns when present, and span metadata.
          </p>
          <button type="button" onClick={forceLoad} className="mt-4 border border-accent/60 bg-accent/10 px-3 py-1.5 text-[0.7rem] font-mono uppercase tracking-[0.18em] text-accent hover:bg-accent/18">
            load {candidates.length} result file{candidates.length === 1 ? "" : "s"} ({formatBytes(candidateBytes)})
          </button>
        </div>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="p-5 text-xs font-mono leading-relaxed text-ink-500">
        No prompt/response rows were found in this report bundle yet.
      </div>
    );
  }

  return (
    <div className="grid min-h-[32rem] grid-cols-1 xl:grid-cols-[minmax(0,1.35fr)_minmax(24rem,0.9fr)]">
      <div className="min-w-0 border-b border-ink-800 xl:border-b-0 xl:border-r">
        <div className="flex items-center gap-3 border-b border-ink-800 bg-ink-950/30 px-3 py-2">
          <span className="field-label">trace rows</span>
          <span className="text-2xs font-mono text-ink-500">{rows.length}</span>
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="filter traces, labels, spans, responses"
            className="ml-auto h-7 w-full max-w-md border border-ink-700 bg-ink-950 px-2 text-xs font-mono text-ink-100 outline-none placeholder:text-ink-600 focus:border-accent"
          />
        </div>
        <TraceTable rows={rows} selectedId={selected?.id ?? null} onSelect={setSelectedId} />
      </div>
      <TraceDetail row={selected} />
    </div>
  );
}

function TraceTable({
  rows,
  selectedId,
  onSelect,
}: {
  rows: TraceRow[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="max-h-[36rem] overflow-auto">
      <table className="w-full border-collapse text-[0.65rem] font-mono">
        <thead className="sticky top-0 bg-ink-900 text-ink-500 uppercase tracking-[0.12em] shadow-[inset_0_-1px_0_0_theme(colors.ink.800)]">
          <tr>
            <th className="px-2.5 py-1.5 text-left font-normal">trace</th>
            <th className="px-2.5 py-1.5 text-left font-normal">labels</th>
            <th className="px-2.5 py-1.5 text-left font-normal">probe scores</th>
            <th className="px-2.5 py-1.5 text-left font-normal">spans</th>
            <th className="px-2.5 py-1.5 text-left font-normal">response</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.id}
              onClick={() => onSelect(row.id)}
              className={[
                "cursor-pointer border-b border-ink-800/60 transition-colors",
                selectedId === row.id ? "bg-accent/10 text-ink-50" : "hover:bg-ink-850/60",
              ].join(" ")}
            >
              <td className="max-w-[16rem] px-2.5 py-2 align-top">
                <div className="truncate text-ink-100" title={row.id}>{row.id}</div>
                <div className="mt-0.5 truncate text-ink-500" title={row.sourceStep}>{row.sourceStep}</div>
              </td>
              <td className="max-w-[18rem] px-2.5 py-2 align-top text-ink-300">
                <div className="line-clamp-ish" title={row.labelSummary}>{row.labelSummary || "-"}</div>
              </td>
              <td className="max-w-[12rem] px-2.5 py-2 align-top text-ink-300">
                <div className="truncate" title={row.scoreSummary}>{row.scoreSummary || "-"}</div>
              </td>
              <td className="max-w-[14rem] px-2.5 py-2 align-top text-ink-300">
                <div className="truncate" title={row.spanSummary}>{row.spanSummary || "-"}</div>
              </td>
              <td className="max-w-[24rem] px-2.5 py-2 align-top text-ink-200">
                <div className="truncate" title={row.response}>{row.response || "-"}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TraceDetail({ row }: { row: TraceRow | null }) {
  if (!row) {
    return <aside className="p-4 text-xs font-mono text-ink-500">Select a trace row.</aside>;
  }
  return (
    <aside className="min-w-0 bg-ink-950/35">
      <div className="border-b border-ink-800 px-4 py-3">
        <div className="field-label">selected trace</div>
        <h3 className="mt-1 truncate font-mono text-sm font-semibold text-ink-50" title={row.id}>
          {row.caseKey || row.id}
        </h3>
        <div className="mt-1 truncate text-2xs font-mono text-ink-500">{row.sourceStep}</div>
      </div>
      <div className="max-h-[36rem] space-y-4 overflow-auto p-4">
        <DetailBlock title="response">
          <pre className="whitespace-pre-wrap text-xs leading-relaxed text-ink-100">{row.response || "-"}</pre>
        </DetailBlock>
        <DetailBlock title="prompt">
          <pre className="max-h-[18rem] overflow-auto whitespace-pre-wrap text-xs leading-relaxed text-ink-200">{row.prompt || "-"}</pre>
        </DetailBlock>
        {row.spans.length > 0 ? (
          <DetailBlock title={`spans (${row.spans.length})`}>
            <div className="space-y-2">
              {row.spans.map((span, index) => (
                <div key={index} className="border border-ink-800 bg-ink-900 px-2 py-1.5">
                  <div className="flex flex-wrap gap-2 text-[0.65rem] font-mono text-ink-300">
                    <span className="text-accent">{compact(span.name ?? span.span_label)}</span>
                    <span>{compact(span.source_type)}</span>
                    <span>{compact(span.assigned_authority)}</span>
                    {span.instruction_like !== undefined ? <span>instruction={formatCell(span.instruction_like)}</span> : null}
                  </div>
                  {span.content_text ? (
                    <p className="mt-1 text-2xs font-mono leading-relaxed text-ink-500">
                      {compact(span.content_text, 220)}
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          </DetailBlock>
        ) : null}
        <DetailBlock title="labels">
          <JsonView value={row.labels} collapsed />
        </DetailBlock>
        {Object.keys(row.probeScores).length > 0 ? (
          <DetailBlock title="probe scores">
            <JsonView value={row.probeScores} collapsed />
          </DetailBlock>
        ) : null}
      </div>
    </aside>
  );
}

function ProbeWorkbench({ rows }: { rows: JsonRecord[] }) {
  if (rows.length === 0) {
    return <div className="p-5 text-xs font-mono text-ink-500">No probe summaries were found.</div>;
  }
  return (
    <div className="space-y-4 p-4">
      <ProbeScoreRail rows={rows} />
      <DataTable rows={rows} columns={preferredColumns(rows, ["step_name", "label", "location", "best_layer", "best_metric", "best_value", "example_count", "split_mode", "result_kind"])} maxHeight="32rem" />
    </div>
  );
}

function ProbeScoreRail({ rows }: { rows: JsonRecord[] }) {
  const scored = rows
    .map((row) => ({
      key: String(row.step_name ?? row.label ?? "probe"),
      label: String(row.label ?? row.location ?? row.step_name ?? "probe"),
      value: numeric(row.best_value ?? row.balanced_accuracy ?? row.auroc ?? row.accuracy),
    }))
    .filter((row): row is { key: string; label: string; value: number } => row.value !== null)
    .slice(0, 18);
  if (scored.length === 0) return null;
  return (
    <div className="border border-ink-800 bg-ink-950/40 p-3">
      <div className="mb-3 flex items-center gap-3">
        <span className="field-label">score rail</span>
        <span className="text-2xs font-mono text-ink-500">best values by probe/location</span>
      </div>
      <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {scored.map((row) => (
          <div key={row.key} className="min-w-0">
            <div className="mb-1 flex items-center gap-2 text-[0.62rem] font-mono">
              <span className="truncate text-ink-300" title={row.label}>{row.label}</span>
              <span className="ml-auto tabular-nums text-ink-100">{row.value.toFixed(3)}</span>
            </div>
            <div className="h-1.5 bg-ink-800">
              <div className="h-full bg-accent" style={{ width: `${Math.max(0, Math.min(100, row.value * 100))}%` }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function InputWorkbench({ rows }: { rows: JsonRecord[] }) {
  if (rows.length === 0) {
    return <div className="p-5 text-xs font-mono text-ink-500">No report input inventory was found.</div>;
  }
  return (
    <div className="p-4">
      <DataTable rows={rows} columns={preferredColumns(rows, ["step_name", "artifact_kind", "result_kind", "technique", "location", "examples", "features", "labels", "artifact_id"])} maxHeight="32rem" />
    </div>
  );
}

function DetailBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-2 flex items-center gap-2">
        <span className="field-label">{title}</span>
        <span className="h-px flex-1 bg-ink-800" />
      </div>
      <div className="min-w-0">{children}</div>
    </section>
  );
}

function Counter({ label, value }: { label: string; value: number }) {
  return (
    <div className="min-w-[5.5rem] border border-ink-800 bg-ink-950/60 px-2 py-1 text-right">
      <div className="field-label">{label}</div>
      <div className="font-mono text-sm text-ink-50 tabular-nums">{value}</div>
    </div>
  );
}

function ResearchTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-3">
      <div className="flex items-center gap-[2px]">
        <div className="h-4 w-[2px] bg-accent-hot" />
        <div className="h-2.5 w-[2px] bg-accent" />
        <div className="h-1.5 w-[2px] bg-status-run" />
      </div>
      <h2 className="mono text-[0.6rem] font-semibold uppercase tracking-[0.2em] text-ink-300">
        {children}
      </h2>
      <div className="h-px flex-1 bg-ink-800" />
    </div>
  );
}

async function fetchResultCandidates(artifactId: string, candidates: ResultCandidate[]): Promise<LoadedResult[]> {
  return Promise.all(
    candidates.map(async (candidate) => {
      const res = await fetch(api.reportAssetUrl(artifactId, candidate.path));
      if (!res.ok) {
        throw new Error(`${candidate.path}: HTTP ${res.status}`);
      }
      return { candidate, payload: await res.json() };
    }),
  );
}

function buildInputRows(data: ReportDetail): JsonRecord[] {
  const report = asRecord(data.report);
  const inputs = asArray(report.inputs).filter(isRecord);
  return inputs.map((input) => {
    const summary = asRecord(input.summary);
    const coverage = asRecord(input.example_coverage);
    const stepName = stringValue(input.name) || stringValue(asRecord(input.workflow).step_name);
    return {
      step_name: stepName,
      artifact_kind: stringValue(input.artifact_kind),
      result_kind: resultKindForStep(data, stepName) || stringValue(summary.kind),
      technique: techniqueFor(input, summary),
      location: locationFor(summary, input),
      examples: coverage.example_count ?? coverage.example_key_count ?? summary.example_count,
      features: joinList(input.feature_names),
      labels: joinList(input.label_names),
      artifact_id: stringValue(input.artifact_id),
    };
  });
}

function buildProbeRows(data: ReportDetail): JsonRecord[] {
  const rows: JsonRecord[] = [];
  const report = asRecord(data.report);
  for (const input of asArray(report.inputs).filter(isRecord)) {
    const summary = asRecord(input.summary);
    const stepName = stringValue(input.name) || stringValue(asRecord(input.workflow).step_name);
    if (Object.keys(summary).length === 0) continue;
    rows.push({
      step_name: stepName,
      result_kind: resultKindForStep(data, stepName) || stringValue(summary.kind),
      label: summary.label ?? inferredLabel(stepName),
      location: locationFor(summary, input),
      best_layer: summary.best_layer,
      best_metric: summary.best_metric,
      best_value: summary.best_value,
      balanced_accuracy: summary.balanced_accuracy,
      auroc: summary.auroc,
      accuracy: summary.accuracy,
      example_count: summary.example_count,
      split_mode: summary.split_mode,
      artifact_id: stringValue(input.artifact_id),
    });
  }
  const stepSummaries = asRecord(asRecord(data.summary).step_summaries);
  for (const [slug, value] of Object.entries(stepSummaries)) {
    const summary = asRecord(value);
    const metrics = asRecord(summary.headline_metrics);
    if (Object.keys(metrics).length === 0) continue;
    rows.push({
      step_name: stringValue(summary.step_name) || slug,
      result_kind: stringValue(summary.kind),
      label: metrics.label ?? inferredLabel(stringValue(summary.step_name) || slug),
      location: metrics.view ?? metrics.token_section ?? metrics.location,
      ...metrics,
    });
  }
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = JSON.stringify([row.step_name, row.label, row.location, row.best_layer, row.best_metric, row.best_value]);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function buildResultCandidates(data: ReportDetail): ResultCandidate[] {
  const resultByStep = new Map<string, ReportResult>();
  for (const result of data.results) {
    if (result.step_name) resultByStep.set(result.step_name, result);
  }
  const candidates: ResultCandidate[] = [];
  const add = (result: ReportResult | undefined, reason: string) => {
    if (!result || candidates.some((item) => item.path === result.path)) return;
    candidates.push({
      path: result.path,
      name: result.name,
      stepName: result.step_name,
      bytes: result.bytes,
      reason,
    });
  };

  for (const item of data.unsupported_inputs) {
    const stepName = stringValue(item.step_name);
    if (stringValue(item.result_kind) === "generation_run_result") {
      add(resultByStep.get(stepName), "generation_run_result");
    }
  }
  const report = asRecord(data.report);
  for (const input of asArray(report.inputs).filter(isRecord)) {
    const stepName = stringValue(input.name) || stringValue(asRecord(input.workflow).step_name);
    if (stringValue(input.artifact_kind) === "generation_run") {
      add(resultByStep.get(stepName), "generation_run");
    }
  }
  for (const result of data.results) {
    const name = result.name.toLowerCase();
    if (name.includes("generation") || name.includes("generate")) {
      add(result, "name_match");
    }
  }
  return candidates;
}

function buildTraceRowsFromResult(item: LoadedResult): TraceRow[] {
  const payload = asRecord(item.payload);
  const rows = asArray(payload.rows).filter(isRecord);
  return rows.flatMap((row, index) => traceRowFromGenerationRow(row, item.candidate, index));
}

function traceRowFromGenerationRow(row: JsonRecord, candidate: ResultCandidate, index: number): TraceRow[] {
  const example = asRecord(row.example);
  const labels = asRecord(example.labels);
  const metadata = asRecord(example.metadata);
  const spans = asArray(metadata.span_specs).filter(isRecord).map(normalizeSpan);
  const prompt = promptText(row.prompt ?? example.prompt);
  const response = textValue(firstPresent(row, ["generated_text", "response", "completion", "output", "text"]));
  const probeScores = scoreMap(row);
  const id = stringValue(row.trace_id) || stringValue(row.example_key) || stringValue(example.key) || `${candidate.stepName ?? candidate.name}:${index + 1}`;
  return [
    {
      id,
      sourceStep: candidate.stepName ?? candidate.name,
      caseKey: stringValue(example.case_key) || stringValue(labels.case_key) || stringValue(labels.case_id),
      labelSummary: labelSummary(labels),
      probeScores,
      scoreSummary: scoreSummary(probeScores),
      spanSummary: spans.map((span) => stringValue(span.name) || stringValue(span.span_label)).filter(Boolean).join(", "),
      prompt,
      response,
      labels,
      spans,
      raw: row,
    },
  ];
}

function buildBehaviorRows(data: ReportDetail): TraceRow[] {
  const report = behaviorReport(data.report);
  if (!report) return [];
  const moduleScores = asArray(report.module_scores).filter(isRecord);
  const flags = asArray(report.flagged_moments).filter(isRecord);
  const traces = asArray(report.traces).filter(isRecord);
  return traces.map((trace, index) => {
    const traceId = stringValue(trace.trace_id) || `trace:${index + 1}`;
    const traceScores = moduleScores.filter((score) => stringValue(score.trace_id) === traceId);
    const traceFlags = flags.filter((flag) => stringValue(flag.trace_id) === traceId);
    const scores: JsonRecord = {};
    for (const score of traceScores) {
      const moduleName = stringValue(score.module) || "score";
      scores[moduleName] = {
        score: score.score,
        band: score.band,
        metric: score.metric,
        scorer_model: score.scorer_model,
      };
    }
    const turns = asArray(trace.turns).filter(isRecord);
    const promptTurns = turns.filter((turn) => stringValue(turn.role) !== "assistant");
    const responseTurns = turns.filter((turn) => stringValue(turn.role) === "assistant");
    const labels = {
      domain: trace.domain,
      outcome: trace.outcome,
      reward: trace.reward,
      source_model: trace.source_model,
      user_model: trace.user_model,
      task_id: trace.task_id,
    };
    const spans = traceFlags.map((flag) => ({
      name: flag.module,
      span_label: flag.title,
      severity: flag.severity,
      score: flag.score,
      turn_index: flag.turn_index,
      content_text: flag.evidence,
      rationale: flag.rationale,
    }));
    return {
      id: traceId,
      sourceStep: "behavior_audit",
      caseKey: stringValue(trace.task_id) || stringValue(trace.case_key),
      labelSummary: labelSummary(labels),
      probeScores: scores,
      scoreSummary: scoreSummary(scores),
      spanSummary: spans.map((span) => stringValue(span.span_label) || stringValue(span.name)).filter(Boolean).join(", "),
      prompt: promptTurns.length > 0 ? promptTurns.map((turn) => `${stringValue(turn.role)}: ${textValue(turn.content)}`).join("\n\n") : textValue(trace.transcript),
      response: responseTurns.map((turn) => textValue(turn.content)).join("\n\n"),
      labels,
      spans,
      raw: trace,
    };
  });
}

function behaviorReport(value: unknown): JsonRecord | null {
  const root = asRecord(value);
  if (stringValue(root.kind) === "behavior_audit_report") return root;
  const nested = asRecord(root.report);
  if (stringValue(nested.kind) === "behavior_audit_report") return nested;
  return null;
}

function resultKindForStep(data: ReportDetail, stepName: string): string {
  for (const item of data.unsupported_inputs) {
    if (stringValue(item.step_name) === stepName && item.result_kind) return stringValue(item.result_kind);
  }
  const table = data.tables.find((item) => item.step_name === stepName);
  return table?.result_kind ?? "";
}

function techniqueFor(input: JsonRecord, summary: JsonRecord): string {
  const kind = stringValue(input.artifact_kind);
  const resultKind = stringValue(summary.kind);
  if (kind === "generation_run" || resultKind === "generation_run_result") return "generation";
  if (kind === "capture") return "capture";
  if (resultKind.includes("probe")) return "linear probe";
  if (resultKind.includes("baseline")) return "text baseline";
  if (summary.best_metric || summary.best_layer !== undefined) return "probe summary";
  return kind || resultKind || "artifact";
}

function locationFor(summary: JsonRecord, input: JsonRecord): string {
  const direct = firstPresent(summary, ["view", "token_section", "location", "span", "pooling"]);
  if (direct) return textValue(direct);
  const features = asArray(input.feature_names).map(String);
  if (features.length > 0) return features.join(", ");
  const captureSections = asArray(summary.capture_token_sections).map(String);
  return captureSections.join(", ");
}

function normalizeSpan(span: JsonRecord): JsonRecord {
  return {
    name: span.name,
    span_label: span.span_label,
    source_type: span.source_type,
    sender: span.sender,
    provenance: span.provenance,
    assigned_authority: span.assigned_authority,
    instruction_like: span.instruction_like,
    content_text: span.content_text,
  };
}

function scoreMap(row: JsonRecord): JsonRecord {
  for (const key of ["probe_scores", "probeScores", "scores", "metrics"]) {
    const value = asRecord(row[key]);
    if (Object.keys(value).length > 0) return value;
  }
  if (row.score !== undefined) return { score: row.score };
  return {};
}

function scoreSummary(scores: JsonRecord): string {
  const entries = Object.entries(scores);
  if (entries.length === 0) return "";
  return entries
    .slice(0, 4)
    .map(([key, value]) => `${key}=${compactScore(value)}`)
    .join(" | ");
}

function compactScore(value: unknown): string {
  if (typeof value === "number") return value.toFixed(3);
  const record = asRecord(value);
  if (typeof record.score === "number") return `${record.score.toFixed(3)}${record.band ? `/${record.band}` : ""}`;
  if (typeof record.value === "number") return record.value.toFixed(3);
  return compact(value, 48);
}

function labelSummary(labels: JsonRecord): string {
  const preferred = [
    "label",
    "target",
    "case_id",
    "case_key",
    "condition",
    "family",
    "authority_cell",
    "expected_selected_source",
    "positive_authority_risk",
    "instruction_uptake_allowed",
    "domain",
    "outcome",
  ];
  const selected = preferred
    .filter((key) => labels[key] !== undefined)
    .map((key) => [key, labels[key]] as const);
  const fallback = Object.entries(labels).filter(([, value]) => isScalar(value));
  return [...selected, ...fallback]
    .slice(0, 8)
    .map(([key, value]) => `${key}=${compact(value, 80)}`)
    .join(" | ");
}

function promptText(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (isRecord(item)) return `${stringValue(item.role) || "message"}: ${textValue(item.content)}`;
        return textValue(item);
      })
      .filter(Boolean)
      .join("\n\n");
  }
  return textValue(value);
}

function preferredColumns(rows: JsonRecord[], preferred: string[]): string[] {
  const all = columnsForRows(rows);
  return [...preferred.filter((column) => all.includes(column)), ...all.filter((column) => !preferred.includes(column))];
}

function inferredLabel(stepName: string): string {
  return stepName.replace(/^report_probe_/, "").replace(/^probe_/, "");
}

function joinList(value: unknown): string {
  return asArray(value).map(String).join(", ");
}

function firstPresent(record: JsonRecord, keys: string[]): unknown {
  for (const key of keys) {
    const value = record[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return undefined;
}

function asRecord(value: unknown): JsonRecord {
  return isRecord(value) ? value : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function isRecord(value: unknown): value is JsonRecord {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isScalar(value: unknown): boolean {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function stringValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  return String(value);
}

function textValue(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function compact(value: unknown, max = 120): string {
  const text = isScalar(value) ? stringValue(value) : JSON.stringify(value);
  return text.length > max ? `${text.slice(0, max - 1)}...` : text;
}

function numeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

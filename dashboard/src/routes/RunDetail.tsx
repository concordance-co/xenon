import { useParams, useNavigate, Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import { api } from "@/lib/api";
import type { RunDetail as RunDetailT, RunReportStatus } from "@/types/api";
import { RunGraph } from "@/components/RunGraph";
import { RunTree } from "@/components/RunTree";
import { RunOverview } from "@/components/RunOverview";
import { ReportGalleryContent } from "@/routes/ReportGallery";
import { StatusChip } from "@/components/StatusChip";

type Page = "overview" | "details" | "report";
import { formatDuration, formatRelative, shortHash } from "@/lib/format";

export function RunDetail() {
  const params = useParams();
  const runId = params.runId!;
  const navigate = useNavigate();

  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [showJson, setShowJson] = useState(false);
  const [page, setPage] = useState<Page>("overview");
  const [generatedArtifactId, setGeneratedArtifactId] = useState<string | null>(null);

  // Persisted graph height — graph lives below the tree in a vertical split.
  const [graphHeight, setGraphHeight] = useState<number>(() => {
    if (typeof window === "undefined") return 360;
    const saved = Number(window.localStorage.getItem("dash.graphHeight"));
    return Number.isFinite(saved) && saved >= 160 ? Math.min(saved, 1200) : 360;
  });
  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem("dash.graphHeight", String(graphHeight));
  }, [graphHeight]);

  const q = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
  });

  useEffect(() => setSelectedStep(null), [runId]);
  useEffect(() => setGeneratedArtifactId(null), [runId]);

  const detail = q.data ?? null;

  if (q.isLoading) return <CenteredNote>Loading run…</CenteredNote>;
  if (q.error)
    return (
      <CenteredNote tone="err">
        Failed to load run {runId}: {(q.error as Error).message}
      </CenteredNote>
    );
  if (!detail) return <CenteredNote>No run data.</CenteredNote>;

  const report = reportStateFromDetail(detail);

  return (
    <div className="flex flex-col h-full min-h-0">
      <RunHeader
        detail={detail}
        report={report}
        page={page}
        setPage={setPage}
        onToggleJson={() => setShowJson((s) => !s)}
        jsonOpen={showJson}
        onBack={() => navigate("/runs")}
      />
      {showJson ? (
        <div className="border-b border-ink-800 bg-ink-950 p-3 max-h-[16rem] overflow-auto">
          <pre className="mono text-2xs text-ink-200 whitespace-pre">
            {JSON.stringify(detail.workflow_payload, null, 2)}
          </pre>
        </div>
      ) : null}
      {page === "overview" ? (
        <RunOverview detail={detail} />
      ) : page === "report" ? (
        <RunReportPanel
          runId={detail.run.run_id}
          report={report}
          artifactId={generatedArtifactId ?? report.artifact_id ?? undefined}
          onGeneratedArtifactId={setGeneratedArtifactId}
          onRefreshRun={() => q.refetch()}
        />
      ) : (
        <div className="flex-1 flex flex-col min-h-0">
          <div className="flex-1 min-h-0 flex flex-col border-b border-ink-800">
            <div className="px-3 py-1 border-b border-ink-800 bg-ink-900/60 flex items-center gap-2">
              <span className="field-label">steps</span>
              <span className="text-[0.625rem] font-mono text-ink-500">
                {detail.steps.length}
              </span>
            </div>
            <RunTree detail={detail} selected={selectedStep} onSelect={setSelectedStep} />
          </div>
          <HorizontalHandle height={graphHeight} onChange={setGraphHeight} />
          <div className="flex flex-col min-h-0" style={{ height: `${graphHeight}px` }}>
            <div className="px-3 py-1 border-b border-ink-800 bg-ink-900/60 flex items-center gap-2">
              <span className="field-label">graph</span>
              <span className="text-[0.625rem] font-mono text-ink-500">
                {detail.nodes.length} nodes · {detail.edges.length} edges
              </span>
            </div>
            <div className="flex-1 min-h-0 relative">
              <ReactFlowProvider>
                <RunGraph
                  nodes={detail.nodes}
                  edges={detail.edges}
                  selected={selectedStep}
                  onSelect={(id) => setSelectedStep(id === selectedStep ? null : id)}
                />
              </ReactFlowProvider>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function HorizontalHandle({
  height,
  onChange,
}: {
  height: number;
  onChange: (n: number) => void;
}) {
  const dragState = useRef<{ startY: number; startHeight: number } | null>(null);
  const onPointerDown = (e: React.PointerEvent) => {
    dragState.current = { startY: e.clientY, startHeight: height };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    document.body.style.cursor = "row-resize";
    document.body.style.userSelect = "none";
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragState.current) return;
    const delta = dragState.current.startY - e.clientY; // drag up = taller graph
    const next = Math.min(1200, Math.max(160, dragState.current.startHeight + delta));
    onChange(next);
  };
  const onPointerUp = (e: React.PointerEvent) => {
    dragState.current = null;
    try {
      (e.target as HTMLElement).releasePointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  };
  return (
    <div
      role="separator"
      aria-orientation="horizontal"
      title="drag to resize"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={() => onChange(360)}
      className="group relative h-[4px] cursor-row-resize bg-ink-800 hover:bg-accent/60 transition-colors"
    >
      <span className="pointer-events-none absolute inset-x-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex items-center gap-0.5 opacity-40 group-hover:opacity-100">
        <span className="h-px w-4 bg-ink-300" />
        <span className="h-px w-4 bg-ink-300" />
      </span>
    </div>
  );
}

function firstReportArtifactId(detail: RunDetailT): string | undefined {
  return detail.steps.find(
    (s) => s.artifact_kind === "report" && s.artifact_id,
  )?.artifact_id ?? undefined;
}

function reportStateFromDetail(detail: RunDetailT): RunReportStatus {
  if (detail.report) return detail.report;
  const artifactId = firstReportArtifactId(detail) ?? null;
  const hasReportStep = Boolean(detail.run.has_report || artifactId);
  return {
    has_report_step: hasReportStep,
    step_name: detail.steps.find((s) => s.spec_kind === "report")?.step_name ?? null,
    artifact_id: artifactId,
    local_available: Boolean(artifactId),
    reason:
      hasReportStep && !artifactId
        ? "No local report artifact is currently recorded for this run."
        : null,
  };
}

function RunHeader({
  detail,
  report,
  page,
  setPage,
  onToggleJson,
  jsonOpen,
  onBack,
}: {
  detail: RunDetailT;
  report: RunReportStatus;
  page: Page;
  setPage: (p: Page) => void;
  onToggleJson: () => void;
  jsonOpen: boolean;
  onBack: () => void;
}) {
  const run = detail.run;
  const hasReport = report.has_report_step;
  return (
    <header className="flex items-stretch border-b border-ink-800 bg-ink-900/80">
      <button
        type="button"
        onClick={onBack}
        className="px-3 flex items-center gap-1 text-[0.65rem] font-mono uppercase tracking-widest text-ink-400 hover:text-accent hover:bg-ink-850 border-r border-ink-800 transition-colors"
        title="Back to runs"
      >
        <span className="text-base leading-none">←</span>
        <span>runs</span>
      </button>

      {/* Page switcher — overview / details / report. */}
      <div className="flex items-stretch border-r border-ink-800">
        {(["overview", "details", "report"] as Page[]).map((p) => {
          const disabled = p === "report" && !hasReport;
          return (
            <button
              type="button"
              key={p}
              disabled={disabled}
              onClick={() => !disabled && setPage(p)}
              title={disabled ? "this workflow has no report step" : undefined}
              className={[
                "relative px-3 text-[0.65rem] font-mono uppercase tracking-[0.18em] transition-colors",
                disabled
                  ? "text-ink-700 cursor-not-allowed"
                  : page === p
                    ? "text-accent bg-ink-850"
                    : "text-ink-500 hover:text-ink-100 hover:bg-ink-850",
              ].join(" ")}
            >
              {page === p && !disabled ? (
                <span
                  className="absolute inset-x-0 top-0 h-[2px] bg-accent"
                  aria-hidden
                />
              ) : null}
              {p}
            </button>
          );
        })}
      </div>

      {/* Title — single row, dominant */}
      <div className="flex items-center gap-2 px-3 min-w-0 border-r border-ink-800 py-1.5">
        <h1 className="mono text-[0.9rem] font-semibold text-ink-50 truncate tracking-tight">
          {run.workflow_name ?? <span className="text-ink-500">(anonymous)</span>}
        </h1>
        <StatusChip status={run.status} />
        <span
          className="text-[0.625rem] font-mono text-ink-500 truncate max-w-[22ch]"
          title={run.run_id}
        >
          {run.run_id}
        </span>
      </div>

      {/* Meta — compact inline */}
      <div className="flex items-stretch divide-x divide-ink-800 text-[0.625rem] font-mono text-ink-400">
        <Stat label="started" value={formatRelative(run.started_at)} />
        <Stat label="duration" value={formatDuration(run.started_at, run.finished_at)} />
        <Stat label="steps" value={String(run.step_counts.total)} />
        <Stat label="wf" value={shortHash(run.workflow_hash)} mono />
        {run.parent_run_id ? (
          <div className="flex flex-col justify-center px-3">
            <span className="field-label">parent</span>
            <Link
              to={`/runs/${run.parent_run_id}`}
              className="text-ink-200 hover:text-accent mt-0.5"
              title={run.parent_run_id}
            >
              {run.parent_run_id.slice(0, 14)}
            </Link>
          </div>
        ) : null}
      </div>

      <div className="ml-auto flex items-center gap-1.5 px-3 border-l border-ink-800">
        <button type="button" onClick={onToggleJson} className="btn-ghost">
          {jsonOpen ? "hide json" : "raw"}
        </button>
      </div>
    </header>
  );
}

function RunReportPanel({
  runId,
  report,
  artifactId,
  onGeneratedArtifactId,
  onRefreshRun,
}: {
  runId: string;
  report: RunReportStatus;
  artifactId: string | undefined;
  onGeneratedArtifactId: (artifactId: string) => void;
  onRefreshRun: () => Promise<unknown>;
}) {
  const queryClient = useQueryClient();
  const generate = useMutation({
    mutationFn: () =>
      api.generateReport(runId, {
        step_name: report.step_name ?? undefined,
      }),
    onSuccess: async (data) => {
      queryClient.setQueryData(["report", data.artifact_id], data.report);
      onGeneratedArtifactId(data.artifact_id);
      // Invalidate everything for this run — report generation copies step
      // results locally, so result/report-status/step queries all change.
      await queryClient.invalidateQueries({
        predicate: (query) => {
          const key = query.queryKey;
          return (
            (Array.isArray(key) && key.includes(runId)) ||
            key[0] === "runs" ||
            key[0] === "report-status"
          );
        },
      });
      await onRefreshRun();
    },
  });

  const localAvailable = Boolean(artifactId) && (report.local_available || generate.isSuccess);

  if (!report.has_report_step) {
    return (
      <CenteredNote>
        This workflow does not define a report step.
      </CenteredNote>
    );
  }

  if (localAvailable && artifactId) {
    return <ReportGalleryContent runId={runId} artifactId={artifactId} embedded />;
  }

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto flex h-full w-full max-w-3xl items-center justify-center p-6">
        <section className="w-full border border-ink-800 bg-ink-900/70 p-6">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div>
              <div className="field-label">report</div>
              <h2 className="mono text-sm text-ink-100">
                {report.step_name ?? "report"}
              </h2>
            </div>
            <span className="text-[0.65rem] font-mono uppercase tracking-[0.18em] text-ink-500">
              local output missing
            </span>
          </div>

          <p className="max-w-2xl text-xs font-mono leading-relaxed text-ink-300">
            The workflow has a report step, but there is no readable local report output yet.
            Generate it on the backend to materialize the copied results, tables, figures, and
            `report.md` for this run.
          </p>
          {report.reason ? (
            <p className="mt-3 border-l border-ink-700 pl-3 text-2xs font-mono leading-relaxed text-ink-500">
              {report.reason}
            </p>
          ) : null}
          {generate.error ? (
            <p className="mt-3 text-2xs font-mono text-status-fail">
              Failed to generate report: {(generate.error as Error).message}
            </p>
          ) : null}

          <div className="mt-5 flex items-center gap-3">
            <button
              type="button"
              disabled={generate.isPending}
              onClick={() => generate.mutate()}
              className={[
                "px-3 py-1.5 text-[0.7rem] font-mono uppercase tracking-[0.18em] transition-colors",
                generate.isPending
                  ? "cursor-wait border border-ink-700 bg-ink-850 text-ink-500"
                  : "border border-accent/60 bg-accent/10 text-accent hover:bg-accent/18",
              ].join(" ")}
            >
              {generate.isPending ? "generating…" : "generate report"}
            </button>
            <span className="text-2xs font-mono text-ink-500">
              Runs the report step locally for this existing workflow run.
            </span>
          </div>
        </section>
      </div>
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-col justify-center px-3 min-w-0">
      <span className="field-label">{label}</span>
      <span
        className={`mt-0.5 ${mono ? "font-mono" : ""} text-ink-200 truncate`}
        title={value}
      >
        {value}
      </span>
    </div>
  );
}

function CenteredNote({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "err";
}) {
  return (
    <div
      className={`flex items-center justify-center h-full text-xs font-mono ${
        tone === "err" ? "text-status-fail" : "text-ink-400"
      }`}
    >
      {children}
    </div>
  );
}

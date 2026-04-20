import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { useMemo, useState } from "react";
import { api } from "@/lib/api";
import { formatDuration, formatRelative, shortHash } from "@/lib/format";
import { StatusChip } from "@/components/StatusChip";
import type { RunSummary } from "@/types/api";

const STATUS_OPTIONS = ["", "completed", "running", "failed"] as const;

interface WorkflowGroup {
  name: string;
  hash: string;
  runs: RunSummary[];
  latestStatus: string;
  latestStarted: string;
}

export function RunsIndex() {
  const [status, setStatus] = useState<string>("");
  const [workflowFilter, setWorkflowFilter] = useState<string>("");
  const [limit, setLimit] = useState<number>(100);

  const query = useQuery({
    queryKey: ["runs", status, workflowFilter, limit],
    queryFn: () =>
      api.listRuns({
        status: status || undefined,
        workflow_name: workflowFilter || undefined,
        limit,
      }),
  });

  const groups = useMemo(() => {
    if (!query.data) return [];
    return groupByWorkflow(query.data.runs);
  }, [query.data]);

  const totalRuns = query.data?.runs.length ?? 0;

  return (
    <div className="flex flex-col h-full">
      <header className="px-4 py-3 border-b border-ink-800 flex items-center gap-3 bg-ink-900/60">
        <div className="flex items-baseline gap-3">
          <h1 className="mono text-[0.8rem] font-semibold uppercase tracking-[0.2em] text-ink-50">
            Workflow Runs
          </h1>
        </div>
        {totalRuns > 0 ? (
          <span className="text-[0.625rem] font-mono text-ink-500">
            {totalRuns} runs · {groups.length} workflows
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-2">
          <FilterSelect label="status" value={status} onChange={setStatus}>
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s || "any"}
              </option>
            ))}
          </FilterSelect>
          <FilterText
            label="workflow"
            placeholder="any"
            value={workflowFilter}
            onChange={setWorkflowFilter}
          />
          <FilterNumber label="limit" value={limit} onChange={setLimit} />
          <button onClick={() => query.refetch()} className="btn-ghost" type="button">
            refresh
          </button>
        </div>
      </header>
      <div className="flex-1 overflow-auto">
        {query.isLoading ? (
          <EmptyBlock>Loading runs…</EmptyBlock>
        ) : query.error ? (
          <EmptyBlock tone="err">
            Failed to load runs: {String((query.error as Error).message)}
          </EmptyBlock>
        ) : groups.length > 0 ? (
          <div className="divide-y divide-ink-800">
            {groups.map((group) => (
              <WorkflowGroupRow key={group.name + group.hash} group={group} />
            ))}
          </div>
        ) : (
          <EmptyBlock>No runs match the current filters.</EmptyBlock>
        )}
      </div>
    </div>
  );
}

function groupByWorkflow(runs: RunSummary[]): WorkflowGroup[] {
  const map = new Map<string, WorkflowGroup>();
  for (const run of runs) {
    const name = run.workflow_name ?? "(anonymous)";
    const key = name;
    let group = map.get(key);
    if (!group) {
      group = {
        name,
        hash: run.workflow_hash,
        runs: [],
        latestStatus: run.status,
        latestStarted: run.started_at,
      };
      map.set(key, group);
    }
    group.runs.push(run);
  }
  // Sort groups by latest run's started_at descending.
  return [...map.values()].sort(
    (a, b) => b.latestStarted.localeCompare(a.latestStarted),
  );
}

function WorkflowGroupRow({ group }: { group: WorkflowGroup }) {
  const [open, setOpen] = useState(false);
  const completed = group.runs.filter((r) => r.status === "completed").length;
  const failed = group.runs.filter((r) => r.status === "failed").length;
  const running = group.runs.filter((r) => r.status === "running").length;
  const withReport = group.runs.filter((r) => r.has_report).length;

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-ink-850 transition-colors"
      >
        <span
          className={[
            "text-[0.7rem] transition-transform shrink-0",
            open ? "rotate-90" : "",
          ].join(" ")}
        >
          ▸
        </span>
        <div className="flex-1 min-w-0">
          <div className="mono text-[0.8rem] font-semibold text-ink-50 truncate">
            {group.name}
          </div>
          <div className="text-[0.6rem] font-mono text-ink-600 tracking-wider mt-0.5">
            {shortHash(group.hash)}
          </div>
        </div>
        <div className="flex items-center gap-3 text-[0.625rem] font-mono text-ink-400 shrink-0">
          <span className="tabular-nums">
            <span className="text-ink-200">{group.runs.length}</span> runs
          </span>
          {completed > 0 ? (
            <span className="text-status-ok tabular-nums">{completed} ✓</span>
          ) : null}
          {failed > 0 ? (
            <span className="text-status-fail tabular-nums">{failed} ✗</span>
          ) : null}
          {running > 0 ? (
            <span className="text-status-run tabular-nums">{running} ⟳</span>
          ) : null}
          {withReport > 0 ? (
            <span className="text-ink-500 tabular-nums">{withReport} reports</span>
          ) : null}
          <span className="text-ink-500">{formatRelative(group.latestStarted)}</span>
        </div>
      </button>
      {open ? (
        <div className="border-t border-ink-800 bg-ink-950/40">
          <table className="w-full text-xs font-mono border-collapse">
            <thead className="bg-ink-900 text-ink-500 uppercase tracking-[0.15em]">
              <tr>
                <Th>run_id</Th>
                <Th>status</Th>
                <Th>started</Th>
                <Th>duration</Th>
                <Th>steps</Th>
                <Th>report</Th>
                <Th>parent</Th>
              </tr>
            </thead>
            <tbody>
              {group.runs.map((run) => (
                <RunRow key={run.run_id} run={run} />
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

function RunRow({ run }: { run: RunSummary }) {
  return (
    <tr className="border-b border-ink-800/60 hover:bg-ink-850/60 transition-colors">
      <Td>
        <Link
          to={`/runs/${run.run_id}`}
          className="text-ink-200 hover:text-accent"
        >
          {run.run_id}
        </Link>
      </Td>
      <Td>
        <StatusChip status={run.status} />
        {run.error ? (
          <div
            className="text-[0.625rem] text-status-fail mt-1 truncate max-w-[16rem]"
            title={run.error}
          >
            {run.error}
          </div>
        ) : null}
      </Td>
      <Td>{formatRelative(run.started_at)}</Td>
      <Td>{formatDuration(run.started_at, run.finished_at)}</Td>
      <Td>
        <StepCountsBar
          total={run.step_counts.total}
          completed={run.step_counts.completed}
          failed={run.step_counts.failed}
          running={run.step_counts.running}
          reused={run.step_counts.reused}
          pending={run.step_counts.pending}
        />
      </Td>
      <Td>
        {run.has_report ? (
          <ReportStatusCell run={run} />
        ) : (
          <span className="text-ink-700">—</span>
        )}
      </Td>
      <Td>
        {run.parent_run_id ? (
          <Link
            to={`/runs/${run.parent_run_id}`}
            className="text-ink-500 hover:text-ink-100"
          >
            {run.parent_run_id.slice(0, 14)}
          </Link>
        ) : (
          <span className="text-ink-700">—</span>
        )}
      </Td>
    </tr>
  );
}

/**
 * Checks whether the report is locally available for a run. Shows:
 * ✓ (green)         — report stored locally, click to view
 * ↓ generate button — has report step but not local; click fires generateReport
 * spinner           — generation in progress
 */
/**
 * Report status cell — uses `report_local` from the RunSummary which was
 * batch-computed in the /api/runs response. Zero additional requests.
 */
function ReportStatusCell({ run }: { run: RunSummary }) {
  const queryClient = useQueryClient();
  const canGenerate = run.status === "completed";

  const genMut = useMutation({
    mutationFn: () => api.generateReport(run.run_id, {}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["run", run.run_id] });
    },
  });

  if (run.report_local === true) {
    return (
      <Link
        to={`/runs/${run.run_id}`}
        className="text-status-ok hover:text-status-ok text-[0.8rem]"
        title="report available locally"
      >
        ✓
      </Link>
    );
  }

  if (!canGenerate) {
    return (
      <span
        className="text-ink-600 text-[0.625rem] font-mono"
        title={`run is ${run.status} — must be completed to generate`}
      >
        —
      </span>
    );
  }

  if (genMut.isPending) {
    return (
      <span className="text-accent text-[0.625rem] font-mono animate-pulse">
        generating…
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        genMut.mutate();
      }}
      className="text-accent hover:text-accent/80 text-[0.7rem] font-mono uppercase tracking-widest"
      title="generate local report"
    >
      ↓ gen
    </button>
  );
}

function StepCountsBar(props: {
  total: number;
  completed: number;
  failed: number;
  running: number;
  reused: number;
  pending: number;
}) {
  const segments: Array<{ key: string; count: number; color: string }> = [
    { key: "completed", count: props.completed, color: "bg-status-ok" },
    { key: "reused", count: props.reused, color: "bg-status-reuse" },
    { key: "running", count: props.running, color: "bg-status-run" },
    { key: "failed", count: props.failed, color: "bg-status-fail" },
    { key: "pending", count: props.pending, color: "bg-ink-600" },
  ];
  const total = Math.max(1, props.total);
  const done = props.completed + props.reused;
  const percent = Math.round((done / total) * 100);
  const tooltip = segments
    .filter((s) => s.count > 0)
    .map((s) => `${s.key}:${s.count}`)
    .join(" · ");
  return (
    <div className="flex items-center gap-2" title={tooltip}>
      <div className="relative h-2 w-32 bg-ink-800 border border-ink-700">
        <div className="absolute inset-0 flex">
          {segments.map((s) =>
            s.count > 0 ? (
              <div
                key={s.key}
                className={s.color}
                style={{ width: `${(s.count / total) * 100}%` }}
              />
            ) : null,
          )}
        </div>
        {[25, 50, 75].map((p) => (
          <div
            key={p}
            className="absolute top-0 bottom-0 w-px bg-ink-950/60"
            style={{ left: `${p}%` }}
            aria-hidden
          />
        ))}
      </div>
      <span className="text-[0.625rem] font-mono text-ink-400 tabular-nums">
        {props.total}
        <span className="text-ink-600"> / </span>
        <span
          className={
            props.failed > 0
              ? "text-status-fail"
              : props.running > 0
                ? "text-status-run"
                : "text-ink-300"
          }
        >
          {percent}%
        </span>
      </span>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="text-left font-normal px-3 py-2 text-[0.58rem] border-r border-ink-800/60 last:border-r-0">
      {children}
    </th>
  );
}

function Td({ children }: { children: React.ReactNode }) {
  return (
    <td className="px-3 py-1.5 align-top text-ink-200 whitespace-nowrap">
      {children}
    </td>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  children,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="field-label">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-ink-850 border border-ink-700 text-xs font-mono text-ink-100 px-1.5 py-0.5 rounded-sm focus:outline-none focus:border-accent"
      >
        {children}
      </select>
    </label>
  );
}

function FilterText({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="field-label">{label}</span>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="bg-ink-850 border border-ink-700 text-xs font-mono text-ink-100 px-1.5 py-0.5 rounded-sm focus:outline-none focus:border-accent w-48"
      />
    </label>
  );
}

function FilterNumber({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="field-label">{label}</span>
      <input
        type="number"
        value={value}
        min={1}
        max={1000}
        onChange={(e) => onChange(Number(e.target.value) || 100)}
        className="bg-ink-850 border border-ink-700 text-xs font-mono text-ink-100 px-1.5 py-0.5 rounded-sm focus:outline-none focus:border-accent w-16"
      />
    </label>
  );
}

function EmptyBlock({
  children,
  tone = "muted",
}: {
  children: React.ReactNode;
  tone?: "muted" | "err";
}) {
  return (
    <div
      className={`m-4 border border-dashed p-8 text-xs font-mono text-center ${
        tone === "err"
          ? "border-status-fail/50 text-status-fail"
          : "border-ink-700 text-ink-400"
      }`}
    >
      {children}
    </div>
  );
}

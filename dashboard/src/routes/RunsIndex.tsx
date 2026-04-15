import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { useState } from "react";
import { api } from "@/lib/api";
import { formatDuration, formatRelative, shortHash } from "@/lib/format";
import { StatusChip } from "@/components/StatusChip";
import type { RunSummary } from "@/types/api";

const STATUS_OPTIONS = ["", "completed", "running", "failed"] as const;

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

  return (
    <div className="flex flex-col h-full">
      <header className="px-4 py-3 border-b border-ink-800 flex items-center gap-3 bg-ink-900/60">
        <div className="flex items-baseline gap-3">
          <h1 className="mono text-[0.8rem] font-semibold uppercase tracking-[0.2em] text-ink-50">
            Workflow Runs
          </h1>
          <span className="text-[0.58rem] font-mono text-ink-600 tracking-widest uppercase">
            / ledger
          </span>
        </div>
        {query.data ? (
          <span className="text-[0.625rem] font-mono text-ink-500">
            {query.data.runs.length} records
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
          <EmptyBlock tone="err">Failed to load runs: {String((query.error as Error).message)}</EmptyBlock>
        ) : query.data && query.data.runs.length > 0 ? (
          <RunsTable runs={query.data.runs} />
        ) : (
          <EmptyBlock>No runs match the current filters.</EmptyBlock>
        )}
      </div>
    </div>
  );
}

function RunsTable({ runs }: { runs: RunSummary[] }) {
  return (
    <table className="w-full text-xs font-mono border-collapse">
      <thead className="sticky top-0 z-10 bg-ink-900 text-ink-500 uppercase tracking-[0.15em] shadow-[inset_0_-1px_0_0_theme(colors.ink.800)]">
        <tr>
          <Th>log</Th>
          <Th>workflow</Th>
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
        {runs.map((run, idx) => (
          <tr
            key={run.run_id}
            className="group border-b border-ink-800/80 hover:bg-ink-850 transition-colors"
          >
            <Td>
              <span className="text-ink-600 tabular-nums text-[0.625rem]">
                {String(runs.length - idx).padStart(3, "0")}
              </span>
            </Td>
            <Td>
              <Link
                to={`/runs/${run.run_id}`}
                className="text-ink-50 font-semibold hover:text-accent"
              >
                {run.workflow_name ?? <span className="text-ink-500">—</span>}
              </Link>
              <div className="text-ink-600 text-[0.625rem] tracking-wider">
                {shortHash(run.workflow_hash)}
              </div>
            </Td>
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
                <span className="chip chip-muted text-status-reuse">yes</span>
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
        ))}
      </tbody>
    </table>
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
      {/* gauge — hairline ticks at 25/50/75% */}
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
    <td className="px-3 py-1.5 align-top text-ink-200 whitespace-nowrap">{children}</td>
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
        tone === "err" ? "border-status-fail/50 text-status-fail" : "border-ink-700 text-ink-400"
      }`}
    >
      {children}
    </div>
  );
}

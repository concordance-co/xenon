import { useEffect, useRef, useState } from "react";
import type { RunDetail, StepSummary } from "@/types/api";
import { familyAccent, statusDotClass } from "@/lib/status";
import { formatDuration, shortHash } from "@/lib/format";
import { StatusChip } from "@/components/StatusChip";
import { InspectorBody, type TabId } from "@/components/Inspector";

/**
 * File-tree-style view of a workflow. Each step is a compact row; clicking
 * expands the full inspector tab surface inline beneath it. When `selected`
 * changes from outside (e.g. clicking a graph node), the tree scrolls the
 * matching row into view.
 */
export function RunTree({
  detail,
  selected,
  onSelect,
}: {
  detail: RunDetail;
  selected: string | null;
  onSelect: (stepName: string | null) => void;
}) {
  const rowRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (!selected) return;
    const node = rowRefs.current.get(selected);
    if (!node) return;
    node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);

  return (
    <div className="flex-1 overflow-auto">
      <div className="divide-y divide-ink-800">
        {detail.steps.map((step) => (
          <TreeRow
            key={step.step_name}
            step={step}
            isOpen={selected === step.step_name}
            onToggle={() => onSelect(selected === step.step_name ? null : step.step_name)}
            runId={detail.run.run_id}
            registerRef={(el) => {
              if (el) rowRefs.current.set(step.step_name, el);
              else rowRefs.current.delete(step.step_name);
            }}
          />
        ))}
      </div>
      {detail.steps.length === 0 ? (
        <div className="p-8 text-xs font-mono text-ink-500 text-center">
          No steps recorded for this run.
        </div>
      ) : null}
    </div>
  );
}

function TreeRow({
  step,
  isOpen,
  onToggle,
  runId,
  registerRef,
}: {
  step: StepSummary;
  isOpen: boolean;
  onToggle: () => void;
  runId: string;
  registerRef: (el: HTMLDivElement | null) => void;
}) {
  const [tab, setTab] = useState<TabId>("overview");
  const accent = familyAccent(step.family);
  const showReusedTag = step.reused_from_run_id !== null && step.status !== "reused";

  return (
    <div ref={registerRef}>
      <button
        type="button"
        onClick={onToggle}
        className={[
          "w-full flex items-stretch text-left transition-colors",
          isOpen ? "bg-ink-850" : "hover:bg-ink-850/60",
        ].join(" ")}
      >
        {/* family accent */}
        <span className={`w-[3px] ${accent.bar}`} aria-hidden />

        {/* caret */}
        <span
          className={`w-6 flex items-center justify-center text-[0.7rem] font-mono ${
            isOpen ? "text-accent" : "text-ink-600"
          }`}
        >
          {isOpen ? "▾" : "▸"}
        </span>

        {/* index */}
        <span className="w-10 py-1.5 text-[0.625rem] font-mono text-ink-600 tabular-nums flex items-center">
          {String(step.step_index).padStart(2, "0")}
        </span>

        {/* status dot */}
        <span className="w-5 flex items-center">
          <span className={`dot ${statusDotClass(step.status)}`} />
        </span>

        {/* name + subline */}
        <span className="flex-1 min-w-0 py-1 pr-3">
          <span className="mono text-[0.8rem] font-semibold text-ink-50 truncate block">
            {step.step_name}
          </span>
          <span className="flex items-center gap-2 text-[0.625rem] font-mono text-ink-500">
            <span className={accent.text}>{accent.label}</span>
            <span className="text-ink-600">·</span>
            <span>{step.spec_kind ?? "—"}</span>
            <span className="text-ink-600">·</span>
            <span>{step.runner}</span>
            {step.artifact_id ? (
              <>
                <span className="text-ink-600">·</span>
                <span className="truncate" title={step.artifact_id}>
                  {shortHash(step.artifact_id, 14)}
                </span>
              </>
            ) : null}
          </span>
        </span>

        {/* tags */}
        <span className="flex items-center gap-1 px-2">
          <StatusChip status={step.status} />
          {showReusedTag ? <span className="chip chip-muted text-status-reuse">re·use</span> : null}
          {step.runtime_app_id ? <span className="chip chip-muted text-status-run">modal</span> : null}
        </span>

        {/* duration */}
        <span className="w-20 py-1.5 text-right pr-3 text-[0.625rem] font-mono text-ink-500">
          {formatDuration(step.started_at, step.finished_at)}
        </span>
      </button>

      {isOpen ? (
        <div className="bg-ink-950 border-t border-ink-800">
          <div className="flex flex-col">
            <InspectorBody runId={runId} step={step} tab={tab} onTabChange={setTab} />
          </div>
        </div>
      ) : null}
    </div>
  );
}

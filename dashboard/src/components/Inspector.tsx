import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { RunDetail, StepDetail, StepSummary } from "@/types/api";
import { StatusChip } from "@/components/StatusChip";
import { JsonView } from "@/components/JsonView";
import { familyAccent } from "@/lib/status";
import { formatDuration, shortHash } from "@/lib/format";
import { OverviewTab } from "@/components/inspector/OverviewTab";
import { SpecTab } from "@/components/inspector/SpecTab";
import { InputsTab } from "@/components/inspector/InputsTab";
import { ArtifactsTab } from "@/components/inspector/ArtifactsTab";
import { ResultsTab } from "@/components/inspector/ResultsTab";
import { DatasetTab } from "@/components/inspector/DatasetTab";
import { LabelsTab } from "@/components/inspector/LabelsTab";
import { PromptTab } from "@/components/inspector/PromptTab";

export type TabId =
  | "overview"
  | "spec"
  | "inputs"
  | "prompt"
  | "dataset"
  | "labels"
  | "results"
  | "artifacts";

export const INSPECTOR_TABS: { id: TabId; label: string }[] = [
  { id: "overview", label: "overview" },
  { id: "spec", label: "spec" },
  { id: "inputs", label: "inputs" },
  { id: "prompt", label: "prompt" },
  { id: "dataset", label: "dataset" },
  { id: "labels", label: "labels" },
  { id: "results", label: "results" },
  { id: "artifacts", label: "artifacts" },
];

/** Sidebar-flavored inspector with full header + tab bar + scrollable body. */
export function Inspector({
  detail,
  selectedStep,
  onSelectStep,
}: {
  detail: RunDetail;
  selectedStep: string | null;
  onSelectStep: (name: string | null) => void;
}) {
  const [tab, setTab] = useState<TabId>("overview");

  const selected = useMemo(
    () => detail.steps.find((s) => s.step_name === selectedStep) ?? null,
    [detail.steps, selectedStep],
  );

  if (!selected) {
    return (
      <div className="flex-1 flex flex-col">
        <InspectorEmptyHeader detail={detail} />
        <div className="p-4 text-2xs font-mono text-ink-500 uppercase tracking-[0.15em] leading-relaxed">
          <div className="pb-2 border-b border-dashed border-ink-800 mb-3 text-ink-400">
            no step pinned
          </div>
          <ul className="space-y-1">
            {INSPECTOR_TABS.filter((t) => t.id !== "overview").map((t) => (
              <li key={t.id} className="flex items-center gap-2 text-ink-600">
                <span className="w-1 h-px bg-ink-700" />
                <span>{t.label}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <InspectorHeader step={selected} detail={detail} onClear={() => onSelectStep(null)} />
      <InspectorBody
        runId={detail.run.run_id}
        step={selected}
        tab={tab}
        onTabChange={setTab}
        scroll
      />
    </div>
  );
}

/**
 * Inspector tab bar + content. Used both in the sidebar Inspector and inline
 * in the TreeView. Loads step detail lazily.
 */
export function InspectorBody({
  runId,
  step,
  tab,
  onTabChange,
  scroll = false,
}: {
  runId: string;
  step: StepSummary;
  tab: TabId;
  onTabChange: (t: TabId) => void;
  scroll?: boolean;
}) {
  const stepQuery = useQuery({
    queryKey: ["step", runId, step.step_name],
    queryFn: () => api.getStep(runId, step.step_name),
  });
  const detail = stepQuery.data ?? null;

  // Hide tabs whose backing data doesn't exist for this step. Until the
  // detail resolves, show all tabs (don't churn the tab bar on load).
  const visibleTabs = useMemo(() => {
    if (!detail) return INSPECTOR_TABS;
    return INSPECTOR_TABS.filter((t) => {
      switch (t.id) {
        case "prompt":
          return detail.has_prompt;
        case "dataset":
          return detail.has_dataset;
        case "labels":
          return detail.has_labels;
        case "artifacts":
          return detail.has_artifact;
        case "results":
          return detail.has_results;
        default:
          return true;
      }
    });
  }, [detail]);

  // If the current tab got hidden, fall back to overview.
  useEffect(() => {
    if (detail && !visibleTabs.some((t) => t.id === tab)) onTabChange("overview");
  }, [detail, visibleTabs, tab, onTabChange]);

  return (
    <>
      <nav className="flex bg-ink-900 border-b border-ink-800 overflow-x-auto">
        {visibleTabs.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => onTabChange(t.id)}
            className={[
              "relative px-2 py-1.5 text-[0.58rem] font-mono uppercase tracking-[0.18em] border-r border-ink-800 transition-colors shrink-0",
              tab === t.id
                ? "text-accent bg-ink-850"
                : "text-ink-500 hover:text-ink-100 hover:bg-ink-850",
            ].join(" ")}
          >
            {tab === t.id ? (
              <span className="absolute inset-x-0 top-0 h-[2px] bg-accent" aria-hidden />
            ) : null}
            {t.label}
          </button>
        ))}
      </nav>
      <div className={scroll ? "flex-1 overflow-auto min-h-0" : "flex-1"}>
        <TabBody
          tab={tab}
          runId={runId}
          step={step}
          stepDetail={detail}
          stepLoading={stepQuery.isLoading}
          stepError={stepQuery.error as Error | null}
        />
      </div>
    </>
  );
}

function InspectorEmptyHeader({ detail }: { detail: RunDetail }) {
  return (
    <header className="px-3 py-3 border-b border-ink-800 bg-ink-900 flex items-start gap-2">
      <div className="w-1 h-8 bg-ink-700" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="field-label">inspector · idle</div>
        <div className="mono text-sm font-semibold text-ink-100 mt-0.5 truncate">
          {detail.run.workflow_name ?? detail.run.run_id}
        </div>
        <div className="mono text-2xs text-ink-500 mt-0.5 tracking-wider">
          {detail.steps.length} steps · select one
        </div>
      </div>
    </header>
  );
}

function InspectorHeader({
  step,
  detail,
  onClear,
}: {
  step: StepSummary;
  detail: RunDetail;
  onClear: () => void;
}) {
  const accent = familyAccent(step.family);
  // Only render the reused chip when the status isn't already "reused" — the
  // StatusChip above already surfaces that information.
  const showReusedChip = step.reused_from_run_id !== null && step.status !== "reused";
  return (
    <header className="border-b border-ink-800 bg-ink-900">
      <div className="px-3 pt-2.5 pb-2 flex items-start gap-2">
        <div className={`w-1 h-9 ${accent.bar} shrink-0`} />
        <div className="flex-1 min-w-0">
          <div className={`text-[0.58rem] font-mono uppercase tracking-[0.2em] ${accent.text}`}>
            {step.family ?? "—"}
            <span className="text-ink-600 mx-1">/</span>
            <span className="text-ink-400">{step.spec_kind ?? "—"}</span>
          </div>
          <div className="mono text-[0.95rem] font-semibold text-ink-50 truncate mt-0.5">
            {step.step_name}
          </div>
          <div className="flex items-center gap-1.5 mt-1 flex-wrap">
            <StatusChip status={step.status} />
            {showReusedChip ? (
              <span className="chip chip-muted text-status-reuse">reused</span>
            ) : null}
            {step.runtime_app_id ? (
              <span className="chip chip-muted text-status-run">modal</span>
            ) : null}
            <span className="text-2xs font-mono text-ink-500 ml-auto">
              {formatDuration(step.started_at, step.finished_at)}
            </span>
          </div>
        </div>
        <button type="button" onClick={onClear} className="btn-ghost shrink-0" title="Clear selection">
          ×
        </button>
      </div>

      <div className="grid grid-cols-4 border-t border-ink-800 divide-x divide-ink-800 bg-ink-950/40">
        <MetaCell label="runner" value={step.runner} />
        <MetaCell label="idx" value={String(step.step_index)} />
        <MetaCell label="sem" value={shortHash(step.step_semantic_hash)} />
        <MetaCell label="spec" value={shortHash(step.step_spec_hash)} />
      </div>

      <div className="px-3 py-1 border-t border-ink-800 text-[0.58rem] font-mono text-ink-600 tracking-widest uppercase truncate">
        run · <span className="text-ink-400">{detail.run.run_id}</span>
      </div>
    </header>
  );
}

function MetaCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col px-2 py-1.5 min-w-0">
      <span className="field-label">{label}</span>
      <span className="text-ink-200 text-[0.7rem] font-mono truncate mt-0.5" title={value}>
        {value}
      </span>
    </div>
  );
}

function TabBody({
  tab,
  runId,
  step,
  stepDetail,
  stepLoading,
  stepError,
}: {
  tab: TabId;
  runId: string;
  step: StepSummary;
  stepDetail: StepDetail | null;
  stepLoading: boolean;
  stepError: Error | null;
}) {
  if (tab === "overview") {
    return <OverviewTab step={step} runId={runId} stepDetail={stepDetail} />;
  }
  if (stepLoading) {
    return <TabNote>Loading step detail…</TabNote>;
  }
  if (stepError) {
    return <TabNote tone="err">Failed to load step detail: {stepError.message}</TabNote>;
  }
  if (!stepDetail) return <TabNote>No detail available.</TabNote>;
  switch (tab) {
    case "spec":
      return <SpecTab detail={stepDetail} />;
    case "inputs":
      return <InputsTab detail={stepDetail} />;
    case "prompt":
      return <PromptTab runId={runId} stepName={step.step_name} />;
    case "dataset":
      return <DatasetTab runId={runId} stepName={step.step_name} />;
    case "labels":
      return <LabelsTab runId={runId} stepName={step.step_name} />;
    case "results":
      return <ResultsTab detail={stepDetail} runId={runId} />;
    case "artifacts":
      return <ArtifactsTab detail={stepDetail} />;
  }
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <div className="field-label mb-1">{title}</div>
      <div className="border border-ink-800 bg-ink-900 rounded-sm p-2">{children}</div>
    </section>
  );
}

function KV({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-2 min-w-0">
      <span className="field-label shrink-0">{label}</span>
      <span className="text-ink-200 truncate" title={value}>
        {value}
      </span>
    </div>
  );
}

function TabNote({ children, tone = "muted" }: { children: ReactNode; tone?: "muted" | "err" }) {
  return (
    <div className={`p-4 text-xs font-mono ${tone === "err" ? "text-status-fail" : "text-ink-400"}`}>
      {children}
    </div>
  );
}

export { Section, KV, TabNote, JsonView };

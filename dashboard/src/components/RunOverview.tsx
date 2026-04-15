import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import type { RunDetail, StepSummary } from "@/types/api";
import { api } from "@/lib/api";
import { OverviewTab } from "@/components/inspector/OverviewTab";
import { StatusChip } from "@/components/StatusChip";
import { familyAccent, statusDotClass } from "@/lib/status";
import { formatDuration } from "@/lib/format";

/**
 * Page-level "overview" for a run: a vertical scroll of every step's
 * Overview panel plus a floating outline that tracks the section in view.
 */
export function RunOverview({ detail }: { detail: RunDetail }) {
  const runId = detail.run.run_id;
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const cardRefs = useRef<Map<string, HTMLElement>>(new Map());
  const [active, setActive] = useState<string | null>(
    detail.steps[0]?.step_name ?? null,
  );
  const queryClient = useQueryClient();

  // Bulk-fetch every step detail in one round-trip, then seed the per-step
  // query cache so individual `StepCard`s hit TanStack Query's cache rather
  // than firing N separate HTTP requests.
  useQuery({
    queryKey: ["steps-detail", runId],
    queryFn: async () => {
      const res = await api.getAllSteps(runId);
      for (const d of res.step_details) {
        queryClient.setQueryData(["step", runId, d.step.step_name], d);
      }
      return res;
    },
  });

  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    // Pick the card whose top is nearest to just-below the sticky header.
    const ratios = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          const name = (e.target as HTMLElement).dataset.step;
          if (!name) continue;
          ratios.set(name, e.intersectionRatio);
        }
        let best: { name: string; ratio: number } | null = null;
        for (const [name, r] of ratios) {
          if (!best || r > best.ratio) best = { name, ratio: r };
        }
        if (best && best.ratio > 0) setActive(best.name);
      },
      {
        root,
        rootMargin: "-10% 0px -60% 0px",
        threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
      },
    );
    cardRefs.current.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, [detail.steps.length]);

  const scrollToStep = (name: string) => {
    const node = cardRefs.current.get(name);
    node?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div ref={scrollRef} className="flex-1 overflow-auto relative">
      <div className="grid grid-cols-[14rem_minmax(0,1fr)] gap-6 max-w-6xl mx-auto px-4 py-4">
        {detail.steps.length > 0 ? (
          <aside className="hidden md:block">
            <nav className="sticky top-2 border border-ink-800 bg-ink-900">
              <div className="px-2 py-1.5 border-b border-ink-800 flex items-center justify-between">
                <span className="field-label">outline</span>
                <span className="text-[0.58rem] font-mono text-ink-600 tabular-nums">
                  {detail.steps.length}
                </span>
              </div>
              <ul className="max-h-[calc(100vh-10rem)] overflow-auto py-1">
                {detail.steps.map((step) => (
                  <OutlineItem
                    key={step.step_name}
                    step={step}
                    isActive={active === step.step_name}
                    onClick={() => scrollToStep(step.step_name)}
                  />
                ))}
              </ul>
            </nav>
          </aside>
        ) : null}

        <div className="space-y-4 min-w-0">
          {detail.steps.map((step) => (
            <StepCard
              key={step.step_name}
              step={step}
              runId={runId}
              registerRef={(el) => {
                if (el) cardRefs.current.set(step.step_name, el);
                else cardRefs.current.delete(step.step_name);
              }}
            />
          ))}
          {detail.steps.length === 0 ? (
            <div className="border border-dashed border-ink-800 p-8 text-xs font-mono text-ink-500 text-center">
              No steps recorded for this run.
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function OutlineItem({
  step,
  isActive,
  onClick,
}: {
  step: StepSummary;
  isActive: boolean;
  onClick: () => void;
}) {
  const accent = familyAccent(step.family);
  return (
    <li>
      <button
        type="button"
        onClick={onClick}
        className={[
          "w-full flex items-center gap-1.5 pl-2 pr-2 py-1 text-left transition-colors border-l-2",
          isActive
            ? "bg-ink-850 border-accent"
            : "border-transparent hover:bg-ink-850/60",
        ].join(" ")}
        title={step.step_name}
      >
        <span className="text-[0.58rem] font-mono text-ink-600 tabular-nums w-5 shrink-0">
          {String(step.step_index).padStart(2, "0")}
        </span>
        <span className={`dot ${statusDotClass(step.status)} shrink-0`} aria-hidden />
        <span
          className={[
            "mono text-[0.7rem] truncate min-w-0",
            isActive ? "text-ink-50 font-semibold" : "text-ink-300",
          ].join(" ")}
        >
          {step.step_name}
        </span>
        <span
          className={`ml-auto text-[0.58rem] font-mono uppercase tracking-widest shrink-0 ${accent.text}`}
        >
          {accent.label}
        </span>
      </button>
    </li>
  );
}

function StepCard({
  step,
  runId,
  registerRef,
}: {
  step: StepSummary;
  runId: string;
  registerRef: (el: HTMLElement | null) => void;
}) {
  const q = useQuery({
    queryKey: ["step", runId, step.step_name],
    queryFn: () => api.getStep(runId, step.step_name),
  });
  const accent = familyAccent(step.family);
  return (
    <section
      ref={registerRef}
      data-step={step.step_name}
      id={`step-${step.step_name}`}
      className="border border-ink-800 bg-ink-900 scroll-mt-2"
    >
      <header className="flex items-center gap-2 px-3 py-2 border-b border-ink-800 bg-ink-950/40">
        <div className={`w-1 h-8 ${accent.bar} shrink-0`} />
        <div className="min-w-0 flex-1">
          <div className={`text-[0.58rem] font-mono uppercase tracking-[0.2em] ${accent.text}`}>
            {step.family ?? "—"}
            <span className="text-ink-600 mx-1">/</span>
            <span className="text-ink-400">{step.spec_kind ?? "—"}</span>
          </div>
          <div className="flex items-center gap-2 min-w-0">
            <span
              className={`dot ${statusDotClass(step.status)} shrink-0`}
              aria-hidden
            />
            <h3 className="mono text-[0.9rem] font-semibold text-ink-50 truncate">
              {step.step_name}
            </h3>
            <span className="text-[0.58rem] font-mono text-ink-600 tabular-nums">
              {String(step.step_index).padStart(2, "0")}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <StatusChip status={step.status} />
          {step.reused_from_run_id && step.status !== "reused" ? (
            <span className="chip chip-muted text-status-reuse">re·use</span>
          ) : null}
          {step.runtime_app_id ? (
            <span className="chip chip-muted text-status-run">modal</span>
          ) : null}
          <span className="text-[0.625rem] font-mono text-ink-500 tabular-nums">
            {formatDuration(step.started_at, step.finished_at)}
          </span>
        </div>
      </header>
      <div className="bg-ink-900">
        <OverviewTab step={step} runId={runId} stepDetail={q.data ?? null} />
      </div>
      {q.error ? (
        <div className="px-3 py-2 text-2xs font-mono text-status-fail border-t border-ink-800">
          failed to load step detail: {(q.error as Error).message}
        </div>
      ) : null}
    </section>
  );
}

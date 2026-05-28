import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { useState } from "react";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import type {
  ProjectDataSource,
  ProjectExperimentSummary,
  ProjectLabelExplanation,
  ProjectPhaseSummary,
  ProjectReportSummary,
  ProjectSummary,
} from "@/types/api";

type MetricPreview = {
  label: string;
  value: unknown;
  source?: string;
};

type Counts = {
  phases: number;
  experiments: number;
  reports: number;
  figures: number;
  tables: number;
  results: number;
};

export function ProjectsIndex() {
  return <ProjectWorkspace />;
}

export function ProjectDetailPage() {
  return <ProjectWorkspace />;
}

export function PhaseDetailPage() {
  const params = useParams();
  const query = useProjectsQuery();
  const projects = query.data?.projects ?? [];
  const project = findProject(projects, params.projectId);
  const phaseId = decodePhaseParam(params["*"]);
  const phase = project?.phases.find((item) => item.phase_id === phaseId) ?? null;

  if (query.isLoading) {
    return <EmptyBlock>Loading phase...</EmptyBlock>;
  }
  if (query.error) {
    return (
      <EmptyBlock tone="err">
        Failed to load phase: {(query.error as Error).message}
      </EmptyBlock>
    );
  }
  if (!project || !phase) {
    return <EmptyBlock tone="err">Phase not found.</EmptyBlock>;
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[19rem_minmax(0,1fr)]">
      <PhaseSidebar project={project} activePhase={phase} />
      <main className="min-h-0 overflow-auto">
        <PhasePage project={project} phase={phase} />
      </main>
    </div>
  );
}

function ProjectWorkspace() {
  const params = useParams();
  const query = useProjectsQuery();
  const projects = query.data?.projects ?? [];
  const orderedProjects = sortProjectsByRecent(projects);
  const selectedProject =
    findProject(orderedProjects, params.projectId) ?? orderedProjects[0] ?? null;

  if (query.isLoading) {
    return <EmptyBlock>Loading projects...</EmptyBlock>;
  }
  if (query.error) {
    return (
      <EmptyBlock tone="err">
        Failed to load projects: {(query.error as Error).message}
      </EmptyBlock>
    );
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[19rem_minmax(0,1fr)]">
      <ProjectSidebar
        projects={orderedProjects}
        selectedProjectId={selectedProject?.project_id ?? null}
        roots={query.data?.project_roots ?? []}
      />
      <main className="min-h-0 overflow-auto">
        {selectedProject ? (
          <ProjectPage project={selectedProject} />
        ) : (
          <EmptyBlock>No projects were found.</EmptyBlock>
        )}
      </main>
    </div>
  );
}

function useProjectsQuery() {
  return useQuery({
    queryKey: ["projects"],
    queryFn: () => api.listProjects(),
  });
}

function ProjectSidebar({
  projects,
  selectedProjectId,
  roots,
}: {
  projects: ProjectSummary[];
  selectedProjectId: string | null;
  roots: string[];
}) {
  const totals = projectTotals(projects);
  return (
    <aside className="flex min-h-0 flex-col border-b border-ink-800 bg-ink-950/70 lg:border-b-0 lg:border-r">
      <div className="border-b border-ink-800 p-3">
        <div className="field-label">projects</div>
        <div className="mt-1 font-mono text-sm font-semibold text-ink-50">
          Research Index
        </div>
        <div className="mt-2 text-[0.62rem] font-mono text-ink-500">
          {projects.length} projects / {totals.phases} phases / {totals.reports} reports
        </div>
        {roots.length > 0 ? (
          <div className="mt-2 truncate text-[0.56rem] font-mono text-ink-700" title={roots.join(" | ")}>
            {roots[0]}
          </div>
        ) : null}
      </div>
      <nav className="min-h-0 flex-1 overflow-auto">
        {projects.map((project) => {
          const selected = project.project_id === selectedProjectId;
          const counts = projectTotals([project]);
          return (
            <Link
              key={project.project_id}
              to={projectUrl(project)}
              className={[
                "block border-b border-ink-800 px-3 py-3 transition-colors",
                selected ? "bg-accent/12 text-ink-50" : "text-ink-400 hover:bg-ink-900 hover:text-ink-100",
              ].join(" ")}
            >
              <div className="truncate text-xs font-mono font-semibold">{project.title}</div>
              <div className="mt-1 flex flex-wrap gap-2 text-[0.6rem] font-mono text-ink-500">
                <span>{counts.phases} phases</span>
                <span>{counts.reports} reports</span>
              </div>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}

function PhaseSidebar({
  project,
  activePhase,
}: {
  project: ProjectSummary;
  activePhase: ProjectPhaseSummary;
}) {
  const orderedPhases = sortPhasesByRecent(project.phases);
  return (
    <aside className="flex min-h-0 flex-col border-b border-ink-800 bg-ink-950/70 lg:border-b-0 lg:border-r">
      <div className="border-b border-ink-800 p-3">
        <Link to="/projects" className="btn-ghost mb-3">
          ← projects
        </Link>
        <div className="field-label">project</div>
        <Link to={projectUrl(project)} className="mt-1 block truncate font-mono text-sm font-semibold text-ink-50 hover:text-accent">
          {project.title}
        </Link>
      </div>
      <nav className="min-h-0 flex-1 overflow-auto">
        {orderedPhases.map((phase) => {
          const active = phase.phase_id === activePhase.phase_id;
          const orderedExperiments = sortExperimentsByRecent(phase.experiments);
          return (
            <div key={phase.phase_id} className="border-b border-ink-800">
              <Link
                to={phaseUrl(project, phase)}
                className={[
                  "block px-3 py-3 transition-colors",
                  active ? "bg-accent/12" : "hover:bg-ink-900",
                ].join(" ")}
              >
                <div className={["truncate text-xs font-mono font-semibold", active ? "text-ink-50" : "text-ink-300"].join(" ")}>
                  {phase.title}
                </div>
                <div className="mt-1 text-[0.6rem] font-mono text-ink-500">
                  {phase.phase_id}
                </div>
              </Link>
              {active ? (
                <div className="border-t border-ink-800 bg-ink-950/60 py-1">
                  {orderedExperiments.map((experiment, index) => {
                    const target = experimentReportUrl(experiment);
                    const className = "block truncate px-5 py-1 text-[0.65rem] font-mono text-ink-500 hover:text-accent";
                    const experimentNumber = orderedExperiments.length - index;
                    const label = (
                      <>
                        <span className="mr-2 text-ink-700">{formatExperimentNumber(experimentNumber)}</span>
                        {experiment.title}
                      </>
                    );
                    return target ? (
                      <Link key={experiment.experiment_id} to={target} className={className}>
                        {label}
                      </Link>
                    ) : (
                      <span key={experiment.experiment_id} className={`${className} cursor-default opacity-60`}>
                        {label}
                      </span>
                    );
                  })}
                </div>
              ) : null}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}

function ProjectPage({ project }: { project: ProjectSummary }) {
  const totals = projectTotals([project]);
  const dataSources = dedupeDataSources(
    project.phases.flatMap((phase) => phase.experiments.flatMap((experiment) => experiment.data_sources)),
  );
  const headline = projectHeadline(project, 6);
  const orderedPhases = sortPhasesByRecent(project.phases);

  return (
    <div className="space-y-5 p-5">
      <section className="space-y-4 border-b border-ink-800 pb-5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="field-label">project overview</div>
            <ProjectStateBadge project={project} />
            {project.status ? <span className="chip chip-muted normal-case tracking-normal">{project.status}</span> : null}
          </div>
          <h1 className="mt-2 font-mono text-xl font-semibold text-ink-50">{project.title}</h1>
          {project.primary_question ? (
            <p className="mt-3 max-w-5xl text-sm leading-relaxed text-ink-100">
              {project.primary_question}
            </p>
          ) : null}
          {project.description ? (
            <p className="mt-2 line-clamp-3 max-w-5xl text-xs leading-relaxed text-ink-300">
              {project.description}
            </p>
          ) : null}
          {project.thin_waist ? (
            <p className="mt-2 max-w-5xl text-xs font-mono leading-relaxed text-ink-400">
              {project.thin_waist}
            </p>
          ) : null}
        </div>
        <div className="grid grid-cols-2 gap-px bg-ink-800 md:grid-cols-4">
          <SmallMetric label="phases" value={totals.phases} />
          <SmallMetric label="experiments" value={totals.experiments} />
          <SmallMetric label="reports" value={totals.reports} />
          <SmallMetric label="results" value={totals.results} />
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <PanelBlock title="headline results" className="xl:col-span-2">
            {headline.length > 0 ? (
              <MetricPreviewGrid items={headline} variant="wide" />
            ) : (
              <MutedLine>No scalar headline metrics yet.</MutedLine>
            )}
          </PanelBlock>
          <InlineDataSources dataSources={dataSources} />
          <InlineInventory title="core labels" values={project.core_labels} />
          <InlineInventory title="candidate data" values={project.candidate_datasets} />
        </div>
      </section>

      <section>
        <SectionHeader title="phases" meta={`${project.phases.length} total`} />
        {project.phases.length > 0 ? (
          <div className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
            {orderedPhases.map((phase) => (
              <PhaseCard key={phase.phase_id} project={project} phase={phase} />
            ))}
          </div>
        ) : (
          <EmptyBlock>No phases with report inventory were found.</EmptyBlock>
        )}
      </section>
    </div>
  );
}

function PhaseCard({
  project,
  phase,
}: {
  project: ProjectSummary;
  phase: ProjectPhaseSummary;
}) {
  const counts = phaseTotals(phase);
  const headline = phaseHeadline(phase, 4);
  return (
    <Link
      to={phaseUrl(project, phase)}
      className="group flex min-h-[15rem] flex-col border border-ink-800 bg-ink-900/55 p-4 transition-colors hover:border-accent/70 hover:bg-ink-850"
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="field-label">{phase.phase_id}</div>
          <h3 className="mt-1 truncate font-mono text-sm font-semibold text-ink-50 group-hover:text-accent">
            {phase.title}
          </h3>
        </div>
        <div className="text-right text-[0.62rem] font-mono text-ink-500">
          <div>{counts.experiments} exp</div>
          <div>{counts.reports} reports</div>
        </div>
      </div>
      {phase.summary_text ? (
        <p className="mt-3 line-clamp-3 text-xs leading-relaxed text-ink-300">
          {phase.summary_text}
        </p>
      ) : null}
      <div className="mt-4 space-y-2">
        <div className="field-label">phase results</div>
        {headline.length > 0 ? (
          <MetricPreviewGrid items={headline} />
        ) : (
          <MutedLine>No scalar headline metrics yet.</MutedLine>
        )}
      </div>
      <div className="mt-auto pt-4">
        <div className="flex flex-wrap gap-2 text-[0.62rem] font-mono text-ink-500">
          <span>{counts.figures} figures</span>
          <span>{counts.tables} tables</span>
          <span>{counts.results} result files</span>
        </div>
      </div>
    </Link>
  );
}

function PhasePage({
  project,
  phase,
}: {
  project: ProjectSummary;
  phase: ProjectPhaseSummary;
}) {
  const counts = phaseTotals(phase);
  const headline = phaseHeadline(phase, 8);
  const dataSources = dedupeDataSources(phase.experiments.flatMap((experiment) => experiment.data_sources));
  const labels = dedupeLabels(phase.experiments.flatMap((experiment) => experiment.labels));
  const probeRows = phaseProbeRows(phase);
  const orderedExperiments = sortExperimentsByRecent(phase.experiments);
  const categoryCounts = experimentCategoryCounts(orderedExperiments);
  const experimentNumberById = new Map(
    orderedExperiments.map((experiment, index) => [experiment.experiment_id, orderedExperiments.length - index]),
  );
  const [categoryFilter, setCategoryFilter] = useState("all");
  const filteredExperiments =
    categoryFilter === "all"
      ? orderedExperiments
      : orderedExperiments.filter((experiment) => normalizedCategory(experiment.experiment_category) === categoryFilter);

  return (
    <div className="space-y-5 p-5">
      <section className="space-y-4 border-b border-ink-800 pb-5">
        <div className="min-w-0">
          <div className="field-label">{project.title}</div>
          <h1 className="mt-2 font-mono text-xl font-semibold text-ink-50">{phase.title}</h1>
          {phase.summary_text ? (
            <p className="mt-3 max-w-5xl text-sm leading-relaxed text-ink-200">
              {phase.summary_text}
            </p>
          ) : null}
        </div>
        <div className="grid grid-cols-2 gap-px bg-ink-800 md:grid-cols-5">
          <SmallMetric label="experiments" value={counts.experiments} />
          <SmallMetric label="reports" value={counts.reports} />
          <SmallMetric label="figures" value={counts.figures} />
          <SmallMetric label="tables" value={counts.tables} />
          <SmallMetric label="results" value={counts.results} />
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          <PanelBlock title="headline results" className="xl:col-span-2">
            {headline.length > 0 ? (
              <MetricPreviewGrid items={headline} variant="wide" />
            ) : (
              <MutedLine>No scalar headline metrics yet.</MutedLine>
            )}
          </PanelBlock>
          <InlineDataSources dataSources={dataSources} />
          <ProbeSummary rows={probeRows} />
          <LabelPreviewList labels={labels} />
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeader
          title="experiments"
          meta={
            categoryFilter === "all"
              ? `${phase.experiments.length} total`
              : `${filteredExperiments.length} of ${phase.experiments.length}`
          }
        />
        <ExperimentCategoryFilter
          active={categoryFilter}
          counts={categoryCounts}
          total={orderedExperiments.length}
          onChange={setCategoryFilter}
        />
        {filteredExperiments.length > 0 ? (
          filteredExperiments.map((experiment) => (
            <ExperimentCard
              key={experiment.experiment_id}
              experiment={experiment}
              experimentNumber={experimentNumberById.get(experiment.experiment_id) ?? 0}
            />
          ))
        ) : (
          <EmptyBlock>No experiments match this category.</EmptyBlock>
        )}
      </section>
    </div>
  );
}

function ExperimentCard({
  experiment,
  experimentNumber,
}: {
  experiment: ProjectExperimentSummary;
  experimentNumber: number;
}) {
  const counts = experimentCounts(experiment);
  const headline = experimentHeadline(experiment, 4);
  const target = experimentReportUrl(experiment);
  const latest = latestReport(experiment.reports);
  const narrative = experimentNarrative(experiment, latest);
  const context = experimentContextItems(experiment, latest);
  const className = [
    "group block border border-ink-800 bg-ink-900/45 px-4 py-3 transition-colors",
    target ? "hover:border-accent/70 hover:bg-ink-850/70" : "opacity-70",
  ].join(" ");
  const content = (
    <>
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="field-label">experiment</div>
            {experimentNumber > 0 ? (
              <span className="chip border-accent/50 bg-accent/10 normal-case tracking-normal text-accent">
                #{formatExperimentNumber(experimentNumber)}
              </span>
            ) : null}
            <span className="chip chip-muted normal-case tracking-normal text-ink-400">
              {categoryLabel(experiment.experiment_category)}
            </span>
          </div>
          <h3 className="mt-1 font-mono text-sm font-semibold text-ink-50 group-hover:text-accent">
            {experiment.title}
          </h3>
          {narrative ? (
            <p className="mt-2 line-clamp-3 max-w-5xl text-xs leading-relaxed text-ink-300">
              {narrative}
            </p>
          ) : null}
          {context.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {context.map((item) => (
                <span key={item} className="chip chip-muted max-w-[28rem] truncate normal-case tracking-normal text-ink-400">
                  {item}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="flex flex-wrap justify-end gap-2">
          <SmallBadge>{experiment.reports.length} reports</SmallBadge>
          <SmallBadge>{counts.figures} figures</SmallBadge>
          <SmallBadge>{counts.tables} tables</SmallBadge>
        </div>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(14rem,0.35fr)]">
        <div className="min-w-0">
          {headline.length > 0 ? (
            <MetricPreviewGrid items={headline} variant="wide" />
          ) : (
            <MutedLine>No scalar headline metrics yet.</MutedLine>
          )}
        </div>
        <div className="min-w-0 border border-ink-800 bg-ink-950/45 px-3 py-2">
          <div className="field-label">latest report</div>
          <div className="mt-1 truncate text-[0.68rem] font-mono text-ink-100" title={latest?.title ?? undefined}>
            {latest?.title ?? "No report yet"}
          </div>
          <div className="mt-1 truncate text-[0.6rem] font-mono text-ink-600">
            {latest?.generated_at ? formatRelative(latest.generated_at) : "not generated"}
          </div>
        </div>
      </div>
    </>
  );

  return target ? (
    <Link to={target} className={className}>
      {content}
    </Link>
  ) : (
    <div className={className}>{content}</div>
  );
}

function MetricPreviewGrid({
  items,
  variant = "default",
}: {
  items: MetricPreview[];
  variant?: "default" | "wide";
}) {
  const columns = variant === "wide" ? "sm:grid-cols-2 2xl:grid-cols-3" : "sm:grid-cols-2";
  return (
    <div className={`grid gap-px bg-ink-800 ${columns}`}>
      {items.map((item) => (
        <div key={`${item.label}:${String(item.value)}:${item.source ?? ""}`} className="min-w-0 bg-ink-950/75 px-2 py-1.5">
          <div className="truncate text-[0.58rem] font-mono uppercase tracking-[0.12em] text-ink-500">
            {item.label}
          </div>
          <div className="mt-0.5 truncate font-mono text-xs text-ink-100 tabular-nums">
            {formatValue(item.value)}
          </div>
          {item.source ? (
            <div className="mt-0.5 truncate text-[0.58rem] font-mono text-ink-600">
              {item.source}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function SmallMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-ink-950/70 px-3 py-2">
      <div className="field-label">{label}</div>
      <div className="mt-1 font-mono text-base text-ink-50 tabular-nums">{value}</div>
    </div>
  );
}

function SmallBadge({ children }: { children: ReactNode }) {
  return (
    <span className="border border-ink-800 bg-ink-950/55 px-2 py-1 text-[0.62rem] font-mono text-ink-400">
      {children}
    </span>
  );
}

function ProjectStateBadge({ project }: { project: ProjectSummary }) {
  const state = projectState(project);
  return (
    <span
      className={[
        "chip normal-case tracking-normal",
        state === "active"
          ? "border-accent/60 bg-accent/12 text-accent"
          : "border-ink-700 bg-ink-800 text-ink-400",
      ].join(" ")}
    >
      {state}
    </span>
  );
}

function PanelBlock({
  title,
  children,
  className = "",
}: {
  title: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={["border border-ink-800 bg-ink-950/35 p-2.5", className].filter(Boolean).join(" ")}>
      <div className="field-label mb-2">{title}</div>
      {children}
    </section>
  );
}

function SectionHeader({ title, meta }: { title: string; meta?: string }) {
  return (
    <div className="mb-3 flex items-center gap-3">
      <h2 className="mono text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-ink-300">
        {title}
      </h2>
      {meta ? <span className="text-[0.62rem] font-mono text-ink-600">{meta}</span> : null}
      <div className="h-px flex-1 bg-ink-800" />
    </div>
  );
}

function ExperimentCategoryFilter({
  active,
  counts,
  total,
  onChange,
}: {
  active: string;
  counts: Map<string, number>;
  total: number;
  onChange: (category: string) => void;
}) {
  const categories = CATEGORY_ORDER.filter((category) => (counts.get(category) ?? 0) > 0);
  if (categories.length <= 1) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      <CategoryButton
        active={active === "all"}
        count={total}
        label="All"
        onClick={() => onChange("all")}
      />
      {categories.map((category) => (
        <CategoryButton
          key={category}
          active={active === category}
          count={counts.get(category) ?? 0}
          label={categoryLabel(category)}
          onClick={() => onChange(category)}
        />
      ))}
    </div>
  );
}

function CategoryButton({
  active,
  count,
  label,
  onClick,
}: {
  active: boolean;
  count: number;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "border px-2.5 py-1 text-[0.62rem] font-mono transition-colors",
        active
          ? "border-accent/70 bg-accent/12 text-accent"
          : "border-ink-800 bg-ink-950/45 text-ink-500 hover:border-ink-700 hover:text-ink-200",
      ].join(" ")}
    >
      {label} <span className="text-ink-600">{count}</span>
    </button>
  );
}

function InlineInventory({ title, values }: { title: string; values: string[] }) {
  return (
    <PanelBlock title={title}>
      {values.length > 0 ? (
        <div className="flex max-h-16 flex-wrap gap-1.5 overflow-hidden">
          {values.slice(0, 18).map((value) => (
            <span key={value} className="chip chip-muted normal-case tracking-normal">
              {value}
            </span>
          ))}
        </div>
      ) : (
        <MutedLine>None recorded.</MutedLine>
      )}
    </PanelBlock>
  );
}

function InlineDataSources({ dataSources }: { dataSources: ProjectDataSource[] }) {
  return (
    <PanelBlock title="data sources">
      {dataSources.length > 0 ? (
        <div className="space-y-1.5">
          {dataSources.slice(0, 3).map((source, index) => (
            <div key={`${source.dataset_id ?? source.name ?? index}:${index}`} className="min-w-0">
              <div className="truncate text-[0.68rem] font-mono text-ink-200">
                {source.name ?? source.dataset_id ?? source.source ?? "dataset"}
              </div>
              <div className="mt-0.5 flex flex-wrap gap-2 text-[0.6rem] font-mono text-ink-500">
                {source.dataset_id ? <span>{source.dataset_id}</span> : null}
                {source.example_count !== null ? <span>{source.example_count} examples</span> : null}
                {source.label_names.length ? <span>{source.label_names.length} labels</span> : null}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <MutedLine>No data source notes captured.</MutedLine>
      )}
    </PanelBlock>
  );
}

function ProbeSummary({ rows }: { rows: MetricPreview[] }) {
  return (
    <PanelBlock title="probes">
      {rows.length > 0 ? (
        <MetricPreviewGrid items={rows.slice(0, 6)} />
      ) : (
        <MutedLine>No probe headline rows detected.</MutedLine>
      )}
    </PanelBlock>
  );
}

function LabelPreviewList({ labels }: { labels: ProjectLabelExplanation[] }) {
  return (
    <PanelBlock title="labels">
      {labels.length > 0 ? (
        <div className="max-h-28 space-y-1.5 overflow-auto pr-1">
          {labels.slice(0, 24).map((label) => (
            <div key={label.name}>
              <div className="text-[0.68rem] font-mono text-ink-200">{label.name}</div>
              {label.description ? (
                <div className="mt-0.5 text-[0.62rem] leading-snug text-ink-500">
                  {label.description}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <MutedLine>No labels captured.</MutedLine>
      )}
    </PanelBlock>
  );
}

function MutedLine({ children }: { children: ReactNode }) {
  return <div className="text-[0.65rem] font-mono text-ink-600">{children}</div>;
}

function EmptyBlock({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: "muted" | "err";
}) {
  return (
    <div className={["p-5 text-xs font-mono", tone === "err" ? "text-status-fail" : "text-ink-500"].join(" ")}>
      {children}
    </div>
  );
}

function findProject(projects: ProjectSummary[], projectId: string | undefined): ProjectSummary | null {
  if (!projectId) return null;
  return projects.find((project) => project.project_id === projectId) ?? null;
}

function projectUrl(project: ProjectSummary): string {
  return `/projects/${encodeURIComponent(project.project_id)}`;
}

function phaseUrl(project: ProjectSummary, phase: ProjectPhaseSummary): string {
  const encodedPhase = phase.phase_id.split("/").map(encodeURIComponent).join("/");
  return `${projectUrl(project)}/phases/${encodedPhase}`;
}

function decodePhaseParam(value: string | undefined): string {
  if (!value) return "";
  return value.split("/").map(decodeURIComponent).join("/");
}

function reportUrl(report: ProjectReportSummary): string {
  return `/projects/reports/${encodeURIComponent(report.report_key)}`;
}

function experimentReportUrl(experiment: ProjectExperimentSummary): string | null {
  const report = latestReport(experiment.reports);
  return report ? reportUrl(report) : null;
}

function latestReport(reports: ProjectReportSummary[]): ProjectReportSummary | null {
  return [...reports].sort((a, b) => timestamp(b.generated_at) - timestamp(a.generated_at))[0] ?? null;
}

function sortProjectsByRecent(projects: ProjectSummary[]): ProjectSummary[] {
  return [...projects].sort(
    (a, b) => latestProjectTimestamp(b) - latestProjectTimestamp(a) || a.title.localeCompare(b.title),
  );
}

function sortPhasesByRecent(phases: ProjectPhaseSummary[]): ProjectPhaseSummary[] {
  return [...phases].sort(
    (a, b) => latestPhaseTimestamp(b) - latestPhaseTimestamp(a) || a.title.localeCompare(b.title),
  );
}

function sortExperimentsByRecent(experiments: ProjectExperimentSummary[]): ProjectExperimentSummary[] {
  return [...experiments].sort(
    (a, b) => latestExperimentTimestamp(b) - latestExperimentTimestamp(a) || a.title.localeCompare(b.title),
  );
}

function latestProjectTimestamp(project: ProjectSummary): number {
  return Math.max(0, ...project.phases.map(latestPhaseTimestamp));
}

function latestPhaseTimestamp(phase: ProjectPhaseSummary): number {
  return Math.max(0, ...phase.experiments.map(latestExperimentTimestamp));
}

function latestExperimentTimestamp(experiment: ProjectExperimentSummary): number {
  return Math.max(0, ...experiment.reports.map((report) => timestamp(report.generated_at)));
}

function projectTotals(projects: ProjectSummary[]): Counts {
  const counts: Counts = { phases: 0, experiments: 0, reports: 0, figures: 0, tables: 0, results: 0 };
  for (const project of projects) {
    counts.phases += project.phases.length;
    for (const phase of project.phases) {
      addCounts(counts, phaseTotals(phase));
    }
  }
  return counts;
}

function phaseTotals(phase: ProjectPhaseSummary): Counts {
  const counts: Counts = {
    phases: 1,
    experiments: phase.experiments.length,
    reports: 0,
    figures: 0,
    tables: 0,
    results: 0,
  };
  for (const experiment of phase.experiments) {
    addCounts(counts, experimentCounts(experiment));
  }
  return counts;
}

function experimentCounts(experiment: ProjectExperimentSummary): Counts {
  return {
    phases: 0,
    experiments: 1,
    reports: experiment.reports.length,
    figures: experiment.reports.reduce((sum, report) => sum + report.figure_count, 0),
    tables: experiment.reports.reduce((sum, report) => sum + report.table_count, 0),
    results: experiment.reports.reduce((sum, report) => sum + report.result_count, 0),
  };
}

function addCounts(target: Counts, source: Counts) {
  target.phases += source.phases;
  target.experiments += source.experiments;
  target.reports += source.reports;
  target.figures += source.figures;
  target.tables += source.tables;
  target.results += source.results;
}

function phaseReports(phase: ProjectPhaseSummary): ProjectReportSummary[] {
  return phase.experiments.flatMap((experiment) => experiment.reports);
}

function projectHeadline(project: ProjectSummary, limit: number): MetricPreview[] {
  return selectHeadline(project.phases.flatMap(phaseReports), limit);
}

function phaseHeadline(phase: ProjectPhaseSummary, limit: number): MetricPreview[] {
  return selectHeadline(phaseReports(phase), limit);
}

function experimentHeadline(experiment: ProjectExperimentSummary, limit: number): MetricPreview[] {
  return selectHeadline(experiment.reports, limit);
}

const NARRATIVE_KEYS = ["summary", "description", "interpretation", "caveat", "notes"];

const CATEGORY_ORDER = [
  "probe_readout",
  "capture",
  "sae",
  "generation",
  "scoring",
  "baseline_gate",
  "audit",
  "other",
];

const CATEGORY_LABELS: Record<string, string> = {
  audit: "Audit",
  baseline_gate: "Baseline gates",
  capture: "Capture",
  generation: "Generation",
  other: "Other",
  probe_readout: "Probe readout",
  sae: "SAE",
  scoring: "Scoring",
};

const SKIPPED_HEADLINE_KEYS = new Set([
  "caveat",
  "claim_only",
  "description",
  "generation_results_path",
  "interpretation",
  "notes",
  "summary",
]);

function selectHeadline(reports: ProjectReportSummary[], limit: number): MetricPreview[] {
  const preferred = [
    "gate_status",
    "unauthorized_binding_rate_positive",
    "target_phrase_match_rate",
    "candidate_action_bound_rate_positive",
    "parse_success_rate",
    "primary_action_rate",
    "high_prompt_visible_gate_count",
    "prompt_visible_gate_count",
    "best_value",
    "balanced_accuracy",
    "auroc",
    "accuracy",
    "avg_response_chars",
    "example_count",
    "row_count",
  ];
  const sorted = [...reports].sort((a, b) => timestamp(b.generated_at) - timestamp(a.generated_at));
  const items: MetricPreview[] = [];
  const seen = new Set<string>();
  for (const key of preferred) {
    for (const report of sorted) {
      const value = report.headline?.[key];
      if (!isScalar(value) || value === null || value === undefined || seen.has(key)) continue;
      seen.add(key);
      items.push({ label: key, value, source: report.title });
      break;
    }
    if (items.length >= limit) return items;
  }
  for (const report of sorted) {
    for (const [key, value] of scalarEntries(report.headline)) {
      if (seen.has(key) || SKIPPED_HEADLINE_KEYS.has(key)) continue;
      seen.add(key);
      items.push({ label: key, value, source: report.title });
      if (items.length >= limit) return items;
    }
  }
  return items;
}

function phaseProbeRows(phase: ProjectPhaseSummary): MetricPreview[] {
  const rows: MetricPreview[] = [];
  for (const report of phaseReports(phase)) {
    const isProbe = report.result_kinds.some((kind) => kind.includes("probe"));
    if (!isProbe) continue;
    for (const [key, value] of scalarEntries(report.headline)) {
      if (["best_value", "balanced_accuracy", "auroc", "accuracy"].includes(key)) {
        rows.push({ label: key, value, source: report.title });
      }
    }
  }
  return rows;
}

function experimentCategoryCounts(experiments: ProjectExperimentSummary[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const experiment of experiments) {
    const category = normalizedCategory(experiment.experiment_category);
    counts.set(category, (counts.get(category) ?? 0) + 1);
  }
  return counts;
}

function categoryLabel(category: string | null | undefined): string {
  return CATEGORY_LABELS[normalizedCategory(category)] ?? "Other";
}

function normalizedCategory(category: string | null | undefined): string {
  return category && CATEGORY_ORDER.includes(category) ? category : "other";
}

function experimentNarrative(
  experiment: ProjectExperimentSummary,
  latest: ProjectReportSummary | null,
): string | null {
  return (
    cleanNarrative(experiment.summary_text) ??
    cleanNarrative(latest?.summary_text) ??
    headlineNarrative(latest?.headline) ??
    inferredExperimentNarrative(experiment, latest)
  );
}

function cleanNarrative(value: string | null | undefined): string | null {
  const text = value?.trim();
  if (!text || text === "-") return null;
  if (text.startsWith("{") || text.startsWith("[")) return null;
  return text;
}

function headlineNarrative(headline: Record<string, unknown> | null | undefined): string | null {
  if (!headline) return null;
  for (const key of NARRATIVE_KEYS) {
    const text = cleanNarrative(typeof headline[key] === "string" ? headline[key] : null);
    if (text) return text;
  }
  return null;
}

function inferredExperimentNarrative(
  experiment: ProjectExperimentSummary,
  latest: ProjectReportSummary | null,
): string | null {
  const source = experiment.data_sources[0] ?? latest?.data_sources[0] ?? null;
  const kinds = latest?.result_kinds.slice(0, 2).join(" + ");
  const examples = source?.example_count !== null && source?.example_count !== undefined
    ? `${formatCount(source.example_count)} examples`
    : null;
  const sourceName = source?.name ?? source?.dataset_id ?? null;
  if (sourceName && kinds) return `Ran ${kinds} over ${examples ? `${examples} from ` : ""}${sourceName}.`;
  if (sourceName) return `Uses ${examples ? `${examples} from ` : ""}${sourceName}.`;
  if (kinds) return `Latest report contains ${kinds} results.`;
  return null;
}

function experimentContextItems(
  experiment: ProjectExperimentSummary,
  latest: ProjectReportSummary | null,
): string[] {
  const items: string[] = [];
  const gateStatus = stringHeadline(latest?.headline, "gate_status");
  if (gateStatus) items.push(`gate: ${gateStatus}`);
  const source = experiment.data_sources[0] ?? latest?.data_sources[0] ?? null;
  if (source) {
    const count = source.example_count !== null ? `${formatCount(source.example_count)} examples` : null;
    items.push([source.name ?? source.dataset_id ?? "dataset", count].filter(Boolean).join(" / "));
  }
  if (latest?.result_kinds.length) items.push(latest.result_kinds.slice(0, 3).join(" + "));
  if (experiment.workflow_names.length) items.push(experiment.workflow_names[0]);
  if (experiment.labels.length) items.push(`${experiment.labels.length} labels`);
  return items.slice(0, 4);
}

function stringHeadline(
  headline: Record<string, unknown> | null | undefined,
  key: string,
): string | null {
  const value = headline?.[key];
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function scalarEntries(value: Record<string, unknown> | null): Array<[string, unknown]> {
  return value ? Object.entries(value).filter(([, item]) => isScalar(item)) : [];
}

function dedupeDataSources(dataSources: ProjectDataSource[]): ProjectDataSource[] {
  const seen = new Set<string>();
  const out: ProjectDataSource[] = [];
  for (const source of dataSources) {
    const key = `${source.dataset_id ?? ""}:${source.name ?? ""}:${source.source ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(source);
  }
  return out;
}

function dedupeLabels(labels: ProjectLabelExplanation[]): ProjectLabelExplanation[] {
  const byName = new Map<string, ProjectLabelExplanation>();
  for (const label of labels) {
    const current = byName.get(label.name);
    if (!current || (!current.description && label.description)) {
      byName.set(label.name, label);
    }
  }
  return [...byName.values()].sort((a, b) => a.name.localeCompare(b.name));
}

function projectState(project: ProjectSummary): "active" | "closed" {
  const status = (project.status ?? "").toLowerCase();
  return status.includes("closed") || status.includes("archived") ? "closed" : "active";
}

function isScalar(value: unknown): boolean {
  return value === null || ["string", "number", "boolean"].includes(typeof value);
}

function formatValue(value: unknown): string {
  if (typeof value === "number") {
    if (Number.isInteger(value)) return String(value);
    const abs = Math.abs(value);
    if (abs > 0 && abs < 0.01) return value.toExponential(2);
    if (abs <= 1) return value.toFixed(3);
    return value.toFixed(2);
  }
  if (value === null || value === undefined) return "-";
  return String(value);
}

function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatExperimentNumber(value: number): string {
  return String(value).padStart(2, "0");
}

function timestamp(value: string | null): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

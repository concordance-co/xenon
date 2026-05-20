import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { ReportFigure, ReportTableSummary } from "@/types/api";
import { JsonView } from "@/components/JsonView";
import { DataTable } from "@/components/DataTable";
import { formatBytes } from "@/lib/format";

export function ReportGallery() {
  const params = useParams();
  return <ReportGalleryContent runId={params.runId} artifactId={params.artifactId} />;
}

export function ReportGalleryContent({
  runId,
  artifactId,
  embedded = false,
}: {
  runId: string | undefined;
  artifactId: string | undefined;
  embedded?: boolean;
}) {
  const q = useQuery({
    queryKey: ["report", artifactId],
    queryFn: () => api.getReport(artifactId!),
    enabled: Boolean(artifactId),
  });

  if (!artifactId) {
    return (
      <div className="p-6 text-xs font-mono text-ink-500">
        This run has no report artifact.
      </div>
    );
  }

  if (q.isLoading) {
    return <div className="p-6 text-xs font-mono text-ink-400">Loading report…</div>;
  }
  if (q.error) {
    return (
      <div className="p-6 text-xs font-mono text-status-fail">
        Failed to load report: {(q.error as Error).message}
      </div>
    );
  }
  if (!q.data) return null;

  const data = q.data;
  return <ReportView data={data} runId={runId} embedded={embedded} />;
}

function ReportView({
  data,
  runId,
  embedded,
}: {
  data: NonNullable<ReturnType<typeof api.getReport> extends Promise<infer T> ? T : never>;
  runId: string | undefined;
  embedded: boolean;
}) {
  // Filter state: a set of input step_names to hide. Empty = show everything
  // (default "all on"). We track hidden rather than shown so the default is
  // simply an empty set, independent of what inputs exist at mount time.
  const [hidden, setHidden] = useState<Set<string>>(() => new Set());

  const inputs = useMemo<string[]>(() => {
    const seen = new Set<string>();
    const order: string[] = [];
    const add = (name: string | null | undefined) => {
      if (!name || seen.has(name)) return;
      seen.add(name);
      order.push(name);
    };
    for (const f of data.figures) add(f.step_name);
    for (const t of data.tables) add(t.step_name);
    for (const r of data.results) add(r.step_name);
    return order;
  }, [data]);

  // Show an item if its step_name isn't hidden. Items with no step_name
  // (null / undefined) are always shown — they aren't scoped to any input.
  const isActive = (name: string | null | undefined) =>
    name === null || name === undefined || !hidden.has(name);

  const figures = data.figures.filter((f) => isActive(f.step_name));
  const tables = data.tables.filter((t) => isActive(t.step_name));
  const results = data.results.filter((r) => isActive(r.step_name));
  const primary = figures.filter((f) => f.primary);
  const secondary = figures.filter((f) => !f.primary);
  const filterActive = hidden.size > 0;

  const toggleInput = (name: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };
  const soloInput = (name: string) => {
    setHidden(new Set(inputs.filter((n) => n !== name)));
  };
  const clearFilter = () => setHidden(new Set());

  const countLine = (
    <span className="flex items-center gap-3">
      <span>
        {figures.length}
        {filterActive && figures.length !== data.figures.length ? (
          <span className="text-ink-700"> /{data.figures.length}</span>
        ) : null}{" "}
        figures
      </span>
      <span>
        {tables.length}
        {filterActive && tables.length !== data.tables.length ? (
          <span className="text-ink-700"> /{data.tables.length}</span>
        ) : null}{" "}
        tables
      </span>
      <span>
        {results.length}
        {filterActive && results.length !== data.results.length ? (
          <span className="text-ink-700"> /{data.results.length}</span>
        ) : null}{" "}
        results
      </span>
    </span>
  );

  return (
    <div className="overflow-auto h-full">
      {embedded ? (
        <header className="px-5 py-2 border-b border-ink-800 bg-ink-900/70 sticky top-0 z-10 flex items-center gap-3 text-2xs font-mono text-ink-500">
          <span className="field-label">report artifact</span>
          <span className="text-ink-200 truncate">{data.artifact_id}</span>
          <span className="ml-auto">{countLine}</span>
        </header>
      ) : (
        <header className="px-5 py-4 border-b border-ink-800 bg-ink-900/70 sticky top-0 z-10">
          <div className="flex items-center gap-3">
            <Link to={`/runs/${runId}`} className="btn-ghost">
              ← run
            </Link>
            <div className="min-w-0">
              <div className="field-label">report artifact</div>
              <h1 className="mono text-sm text-ink-100 truncate">{data.artifact_id}</h1>
            </div>
            <div className="ml-auto flex items-center gap-3 text-2xs font-mono text-ink-500">
              {countLine}
            </div>
          </div>
        </header>
      )}

      {inputs.length > 1 ? (
        <div className="px-5 py-2 border-b border-ink-800 bg-ink-900/40 flex items-center gap-3 flex-wrap sticky top-[2.25rem] z-[9]">
          <span className="field-label">filter</span>
          <div className="flex items-center gap-3 flex-wrap">
            {inputs.map((name) => {
              const checked = !hidden.has(name);
              return (
                <button
                  key={name}
                  type="button"
                  onClick={() => toggleInput(name)}
                  onDoubleClick={() => soloInput(name)}
                  title="click: toggle · double-click: solo"
                  className={[
                    "flex items-center gap-1.5 text-[0.7rem] font-mono transition-colors select-none",
                    checked ? "text-ink-100" : "text-ink-500 hover:text-ink-200",
                  ].join(" ")}
                >
                  <span
                    className={[
                      "inline-flex items-center justify-center w-[0.9rem] h-[0.9rem] border rounded-[2px] shrink-0 text-[0.65rem] leading-none",
                      checked
                        ? "border-accent bg-accent/15 text-accent"
                        : "border-ink-600 bg-ink-950 text-transparent",
                    ].join(" ")}
                    aria-hidden
                  >
                    ✓
                  </span>
                  <span>{name}</span>
                </button>
              );
            })}
          </div>
          {filterActive ? (
            <button
              type="button"
              onClick={clearFilter}
              className="btn-ghost ml-auto"
              title="reset · show all"
            >
              reset
            </button>
          ) : null}
        </div>
      ) : null}

      <div className="p-5 space-y-6">
        {primary.length > 0 ? (
          <section>
            <SectionTitle>primary figures</SectionTitle>
            <FigureGrid figures={primary} artifactId={data.artifact_id} columns={3} />
          </section>
        ) : null}
        {secondary.length > 0 ? (
          <section>
            <SectionTitle>supporting figures</SectionTitle>
            <FigureGrid figures={secondary} artifactId={data.artifact_id} columns={3} />
          </section>
        ) : null}
        {data.headline ? (
          <section>
            <SectionTitle>headline metrics</SectionTitle>
            <JsonView value={data.headline} />
          </section>
        ) : null}
        {tables.length > 0 ? (
          <section>
            <SectionTitle>tables</SectionTitle>
            <TableList tables={tables} artifactId={data.artifact_id} />
          </section>
        ) : null}
        {results.length > 0 ? (
          <section>
            <SectionTitle>copied results</SectionTitle>
            <ul className="divide-y divide-ink-800 border border-ink-800 rounded-sm">
              {results.map((r) => (
                <li key={r.path} className="px-3 py-2 flex items-center justify-between gap-2 text-2xs font-mono">
                  <span className="text-ink-200 truncate">{r.name}</span>
                  <span className="text-ink-500 shrink-0">{formatBytes(r.bytes)}</span>
                </li>
              ))}
            </ul>
          </section>
        ) : null}
        <section>
          <SectionTitle>raw report.json</SectionTitle>
          <JsonView value={data.report} collapsed />
        </section>
        {data.summary ? (
          <section>
            <SectionTitle>raw summary.json</SectionTitle>
            <JsonView value={data.summary} collapsed />
          </section>
        ) : null}
      </div>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-3">
      <div className="flex items-center gap-[2px]">
        <div className="w-[2px] h-4 bg-accent" />
        <div className="w-[2px] h-2.5 bg-accent/60" />
      </div>
      <h2 className="mono text-[0.6rem] uppercase tracking-[0.2em] text-ink-300 font-semibold">
        {children}
      </h2>
      <div className="flex-1 h-px bg-ink-800" />
    </div>
  );
}

function FigureGrid({
  figures,
  artifactId,
  columns,
}: {
  figures: ReportFigure[];
  artifactId: string;
  columns: 2 | 3;
}) {
  const [lightbox, setLightbox] = useState<ReportFigure | null>(null);
  const cls =
    columns === 3
      ? "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3"
      : "grid grid-cols-1 md:grid-cols-2 gap-3";
  return (
    <>
      <div className={cls}>
        {figures.map((fig) => (
          <FigureCard
            key={fig.figure_id}
            fig={fig}
            artifactId={artifactId}
            onOpen={() => setLightbox(fig)}
          />
        ))}
      </div>
      {lightbox ? (
        <Lightbox
          fig={lightbox}
          artifactId={artifactId}
          figures={figures}
          onClose={() => setLightbox(null)}
          onNavigate={setLightbox}
        />
      ) : null}
    </>
  );
}

function FigureCard({
  fig,
  artifactId,
  onOpen,
}: {
  fig: ReportFigure;
  artifactId: string;
  onOpen: () => void;
}) {
  const [broken, setBroken] = useState(false);
  const url = api.reportAssetUrl(artifactId, fig.path);
  return (
    <figure className="border border-ink-800 bg-ink-900 flex flex-col h-full">
      <figcaption className="px-3 py-2 border-b border-ink-800 space-y-1">
        <div className="flex items-center gap-2 text-2xs font-mono flex-wrap">
          <span className="text-ink-50 font-semibold">{fig.title ?? fig.figure_id}</span>
          {fig.step_name ? (
            <span className="text-ink-500">· {fig.step_name}</span>
          ) : null}
        </div>
        {fig.caption ? (
          <p className="text-2xs font-mono text-ink-400 leading-relaxed">{fig.caption}</p>
        ) : null}
      </figcaption>
      <button
        type="button"
        onClick={onOpen}
        className="flex-1 flex items-center justify-center bg-ink-950 min-h-0 cursor-zoom-in"
        title="click to enlarge"
      >
        {broken ? (
          <div className="flex items-center justify-center h-20 text-2xs font-mono text-status-fail">
            asset failed to load
          </div>
        ) : (
          <img
            src={url}
            alt={fig.title ?? fig.figure_id}
            className="block max-w-full max-h-[40rem] object-contain"
            onError={() => setBroken(true)}
            loading="lazy"
          />
        )}
      </button>
    </figure>
  );
}

function Lightbox({
  fig,
  artifactId,
  figures,
  onClose,
  onNavigate,
}: {
  fig: ReportFigure;
  artifactId: string;
  figures: ReportFigure[];
  onClose: () => void;
  onNavigate: (fig: ReportFigure) => void;
}) {
  const url = api.reportAssetUrl(artifactId, fig.path);
  const idx = figures.findIndex((f) => f.figure_id === fig.figure_id);
  const prev = idx > 0 ? figures[idx - 1] : null;
  const next = idx < figures.length - 1 ? figures[idx + 1] : null;

  // Keyboard navigation: Esc to close, ←/→ to navigate.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && prev) onNavigate(prev);
      if (e.key === "ArrowRight" && next) onNavigate(next);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/90"
      onClick={onClose}
    >
      <div
        className="relative max-w-[90vw] max-h-[90vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between gap-3 px-4 py-2 bg-ink-900 border-b border-ink-800 rounded-t-sm">
          <div className="min-w-0">
            <div className="text-xs font-mono text-ink-50 font-semibold truncate">
              {fig.title ?? fig.figure_id}
            </div>
            {fig.caption ? (
              <div className="text-2xs font-mono text-ink-400 truncate mt-0.5">
                {fig.caption}
              </div>
            ) : null}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {figures.length > 1 ? (
              <span className="text-[0.625rem] font-mono text-ink-500 tabular-nums">
                {idx + 1}/{figures.length}
              </span>
            ) : null}
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="btn-ghost"
              title="open in new tab"
            >
              ↗
            </a>
            <button type="button" onClick={onClose} className="btn-ghost" title="close (Esc)">
              ×
            </button>
          </div>
        </div>

        {/* Image */}
        <div className="bg-ink-950 flex items-center justify-center p-4 overflow-auto rounded-b-sm">
          <img
            src={url}
            alt={fig.title ?? fig.figure_id}
            className="max-w-full max-h-[80vh] object-contain"
          />
        </div>

        {/* Prev / Next arrows */}
        {prev ? (
          <button
            type="button"
            onClick={() => onNavigate(prev)}
            className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-full px-2 py-8 text-xl text-ink-400 hover:text-ink-50 transition-colors"
            title="previous (←)"
          >
            ‹
          </button>
        ) : null}
        {next ? (
          <button
            type="button"
            onClick={() => onNavigate(next)}
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-full px-2 py-8 text-xl text-ink-400 hover:text-ink-50 transition-colors"
            title="next (→)"
          >
            ›
          </button>
        ) : null}
      </div>
    </div>
  );
}

function TableList({
  tables,
  artifactId,
}: {
  tables: ReportTableSummary[];
  artifactId: string;
}) {
  return (
    <div className="border border-ink-800 divide-y divide-ink-800">
      {tables.map((t) => (
        <TableRow key={t.slug} table={t} artifactId={artifactId} />
      ))}
    </div>
  );
}

function TableRow({ table, artifactId }: { table: ReportTableSummary; artifactId: string }) {
  const [open, setOpen] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const assetUrl = api.reportAssetUrl(artifactId, table.path);
  const rawQuery = useQuery({
    queryKey: ["report-table", artifactId, table.slug],
    queryFn: async () => {
      const res = await fetch(assetUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()) as unknown;
    },
    enabled: open,
  });
  const parsed = open && rawQuery.data ? extractTabular(rawQuery.data) : null;
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full px-3 py-1.5 flex items-center gap-3 text-[0.65rem] font-mono hover:bg-ink-850"
      >
        <span className={`text-[0.7rem] ${open ? "text-accent" : "text-ink-600"}`}>
          {open ? "▾" : "▸"}
        </span>
        <span className="text-ink-100 font-semibold">{table.slug}</span>
        <span className="text-ink-500">
          {table.rows} rows · {table.columns.length} cols
        </span>
        <span className="text-ink-500 truncate">{table.step_name ?? "—"}</span>
        {table.result_kind ? (
          <span className="ml-auto chip chip-muted">{table.result_kind}</span>
        ) : null}
      </button>
      {open ? (
        <div className="bg-ink-950 border-t border-ink-800">
          {rawQuery.isLoading ? (
            <div className="p-3 text-2xs font-mono text-ink-400">loading…</div>
          ) : rawQuery.error ? (
            <div className="p-3 text-2xs font-mono text-status-fail">
              failed to load: {(rawQuery.error as Error).message}
            </div>
          ) : parsed ? (
            <>
              <DataTable columns={parsed.columns} rows={parsed.rows} />
              <div className="flex items-center justify-end px-3 py-1.5 border-t border-ink-800">
                <button
                  type="button"
                  onClick={() => setShowRaw((s) => !s)}
                  className="btn-ghost"
                >
                  {showRaw ? "hide json" : "raw json"}
                </button>
              </div>
              {showRaw ? (
                <div className="p-3 border-t border-ink-800">
                  <JsonView value={rawQuery.data} />
                </div>
              ) : null}
            </>
          ) : (
            <div className="p-3">
              <JsonView value={rawQuery.data} />
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

/** Extract columns + rows from whatever JSON shape the reporting layer produced. */
function extractTabular(
  payload: unknown,
): { columns: string[]; rows: Array<Record<string, unknown>> } | null {
  if (!payload || typeof payload !== "object") return null;
  const p = payload as Record<string, unknown>;

  // Shape 1: { columns: [...], rows: [{...}, ...] } or [[...], ...]
  if (Array.isArray(p.rows)) {
    const rowsRaw = p.rows;
    const explicitCols = Array.isArray(p.columns) ? (p.columns as unknown[]).map(String) : null;
    if (rowsRaw.length === 0) {
      return { columns: explicitCols ?? [], rows: [] };
    }
    const first = rowsRaw[0];
    if (first && typeof first === "object" && !Array.isArray(first)) {
      const columns = explicitCols ?? Object.keys(first as Record<string, unknown>);
      return { columns, rows: rowsRaw as Array<Record<string, unknown>> };
    }
    if (Array.isArray(first) && explicitCols) {
      return {
        columns: explicitCols,
        rows: rowsRaw.map((r) => {
          const arr = r as unknown[];
          const obj: Record<string, unknown> = {};
          explicitCols.forEach((col, i) => (obj[col] = arr[i]));
          return obj;
        }),
      };
    }
    return null;
  }

  // Shape 2: { records: [{...}, ...] }
  if (Array.isArray(p.records)) {
    const records = p.records as unknown[];
    if (records.length === 0) return { columns: [], rows: [] };
    const first = records[0];
    if (first && typeof first === "object" && !Array.isArray(first)) {
      const columns = Object.keys(first as Record<string, unknown>);
      return { columns, rows: records as Array<Record<string, unknown>> };
    }
  }
  return null;
}


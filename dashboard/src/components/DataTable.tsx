import { useMemo, useState } from "react";

/**
 * Sortable, column-pruning dense data table. Used by the report gallery and
 * the step result tab.
 */
export function DataTable({
  columns,
  rows,
  maxHeight = "24rem",
}: {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  maxHeight?: string;
}) {
  const [sort, setSort] = useState<{ col: string; dir: "asc" | "desc" } | null>(null);

  const nonEmptyColumns = useMemo(
    () =>
      columns.filter((col) =>
        rows.some((row) => {
          const v = row[col];
          return v !== null && v !== undefined && v !== "";
        }),
      ),
    [columns, rows],
  );

  const sortedRows = useMemo(() => {
    if (!sort) return rows;
    const { col, dir } = sort;
    const sign = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => sign * compareValues(a[col], b[col]));
  }, [rows, sort]);

  const onHeaderClick = (col: string) => {
    setSort((prev) => {
      if (!prev || prev.col !== col) return { col, dir: "asc" };
      if (prev.dir === "asc") return { col, dir: "desc" };
      return null;
    });
  };

  if (nonEmptyColumns.length === 0 || rows.length === 0) {
    return <div className="p-3 text-2xs font-mono text-ink-500">empty table</div>;
  }
  return (
    <div className="overflow-auto" style={{ maxHeight }}>
      <table className="w-full text-[0.65rem] font-mono border-collapse">
        <thead className="sticky top-0 bg-ink-900 text-ink-500 uppercase tracking-[0.12em] shadow-[inset_0_-1px_0_0_theme(colors.ink.800)]">
          <tr>
            {nonEmptyColumns.map((col) => {
              const active = sort?.col === col;
              return (
                <th
                  key={col}
                  className="text-left font-normal border-r border-ink-800/60 last:border-r-0 whitespace-nowrap p-0"
                >
                  <button
                    type="button"
                    onClick={() => onHeaderClick(col)}
                    className={[
                      "w-full flex items-center gap-1 px-2.5 py-1.5 text-left transition-colors",
                      active
                        ? "text-accent bg-ink-850"
                        : "text-ink-500 hover:text-ink-100 hover:bg-ink-850",
                    ].join(" ")}
                    title={`sort by ${col}`}
                  >
                    <span className="truncate">{col}</span>
                    <span className="ml-auto text-[0.6rem] leading-none">
                      {active ? (
                        sort!.dir === "asc" ? (
                          "▲"
                        ) : (
                          "▼"
                        )
                      ) : (
                        <span className="opacity-30">↕</span>
                      )}
                    </span>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, i) => (
            <tr
              key={i}
              className="border-b border-ink-800/60 hover:bg-ink-850/60 transition-colors"
            >
              {nonEmptyColumns.map((col) => {
                const v = row[col];
                return (
                  <td
                    key={col}
                    className="px-2.5 py-1 align-top text-ink-200 tabular-nums"
                    title={typeof v === "object" ? JSON.stringify(v) : String(v ?? "")}
                  >
                    {formatCell(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function compareValues(a: unknown, b: unknown): number {
  if (a === b) return 0;
  const aNil = a === null || a === undefined;
  const bNil = b === null || b === undefined;
  if (aNil && bNil) return 0;
  if (aNil) return 1;
  if (bNil) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean") return Number(a) - Number(b);
  const na = typeof a === "string" ? Number(a) : NaN;
  const nb = typeof b === "string" ? Number(b) : NaN;
  if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function formatCell(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(4);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

export function columnsForRows(rows: Array<Record<string, unknown>>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        out.push(key);
      }
    }
  }
  return out;
}

import type { StepDetail } from "@/types/api";
import { Section } from "@/components/Inspector";
import { JsonView } from "@/components/JsonView";

export function SpecTab({ detail }: { detail: StepDetail }) {
  const kind = (detail.spec?.kind as string | undefined) ?? null;
  return (
    <div className="p-3 space-y-3">
      {kind === "capture" ? <CaptureSitesPanel spec={detail.spec} /> : null}
      <Section title="normalized spec">
        <SummaryList items={detail.spec_summary} />
      </Section>
      <Section title="raw spec">
        <JsonView value={detail.spec} collapsed maxHeight="40rem" />
      </Section>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Capture site highlights — for a capture spec, the sites array is the most
 * operationally important field. Break it out above the flat summary.
 * ------------------------------------------------------------------------- */

type Site = Record<string, unknown>;

function CaptureSitesPanel({ spec }: { spec: Record<string, unknown> }) {
  const sitesRaw = spec.sites;
  if (!Array.isArray(sitesRaw) || sitesRaw.length === 0) return null;
  const sites = sitesRaw as Site[];
  return (
    <Section title={`capture sites (${sites.length})`}>
      <ul className="divide-y divide-ink-800">
        {sites.map((site, i) => (
          <li key={(site.name as string) ?? i} className="py-1.5 first:pt-0 last:pb-0">
            <SiteRow site={site} />
          </li>
        ))}
      </ul>
    </Section>
  );
}

function SiteRow({ site }: { site: Site }) {
  const name = (site.name as string) ?? "—";
  const layers = site.layers as unknown;
  const layersStr = Array.isArray(layers)
    ? (layers as unknown[]).map(String).join(", ")
    : "—";
  const tokens = site.tokens as { kind?: string; value?: unknown } | undefined;
  const tokenLabel = tokens
    ? tokens.kind === "section"
      ? `section(${String(tokens.value)})`
      : tokens.kind === "slice"
        ? `slice(${JSON.stringify(tokens.value)})`
        : tokens.kind ?? "—"
    : "—";
  const record = site.record as unknown;
  const isMoE = Array.isArray(record);
  const siteKind = (site.site as string | undefined) ?? (isMoE ? "moe" : "residual");
  return (
    <div>
      <div className="flex items-center gap-2 flex-wrap text-[0.7rem] font-mono">
        <span className="chip chip-muted text-ink-200">{isMoE ? "moe" : "residual"}</span>
        <span className="text-ink-50 font-semibold">{name}</span>
        {!isMoE ? <span className="text-ink-500">@ {siteKind}</span> : null}
      </div>
      <dl className="mt-1 grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-[0.65rem] font-mono">
        <RowKV label="layers" value={layersStr} />
        <RowKV label="tokens" value={tokenLabel} />
        {isMoE ? (
          <RowKV
            label="records"
            value={(record as Array<Record<string, unknown>>)
              .map((r) => String(r.kind))
              .join(", ")}
          />
        ) : null}
      </dl>
    </div>
  );
}

/* -------------------------------------------------------------------------
 * Summary list — labels wrap, values wrap. No more fixed-width overflow.
 * ------------------------------------------------------------------------- */

function SummaryList({ items }: { items: Array<{ label: string; value: string }> }) {
  if (items.length === 0) {
    return <div className="text-2xs font-mono text-ink-500">No summary fields extracted.</div>;
  }
  return (
    <ul className="divide-y divide-ink-800">
      {items.map((item) => (
        <li key={item.label} className="py-1 first:pt-0 last:pb-0">
          <div className="field-label break-all leading-snug">{item.label}</div>
          <div className="text-[0.7rem] font-mono text-ink-200 break-words whitespace-pre-wrap leading-snug mt-0.5">
            {item.value || <span className="text-ink-500">—</span>}
          </div>
        </li>
      ))}
    </ul>
  );
}

function RowKV({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="field-label break-all leading-snug">{label}</dt>
      <dd className="text-ink-200 break-words whitespace-pre-wrap">{value}</dd>
    </>
  );
}

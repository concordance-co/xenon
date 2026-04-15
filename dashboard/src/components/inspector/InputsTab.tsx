import type { ResolvedDep, StepDetail } from "@/types/api";
import { Section, KV } from "@/components/Inspector";
import { statusDotClass } from "@/lib/status";

export function InputsTab({ detail }: { detail: StepDetail }) {
  return (
    <div className="p-3 space-y-3">
      <Section title="upstream">
        <DepTable deps={detail.upstream} empty="No upstream steps." />
      </Section>
      <Section title="downstream">
        <DepTable deps={detail.downstream} empty="No downstream steps." />
      </Section>
      <Section title="input artifact refs">
        {detail.step.step_semantic_hash ? null : null}
        {detail.artifact && detail.artifact.input_artifact_refs.length > 0 ? (
          <ul className="text-2xs font-mono text-ink-300 space-y-0.5">
            {detail.artifact.input_artifact_refs.map((ref) => (
              <li key={ref} className="truncate">
                <span className="text-ink-600">ref</span> {ref}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-2xs font-mono text-ink-500">No artifact refs recorded.</div>
        )}
      </Section>
      {detail.artifact ? (
        <Section title="example coverage">
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-2xs font-mono">
            {Object.entries(detail.artifact.example_coverage).slice(0, 12).map(([k, v]) => (
              <KV key={k} label={k} value={typeof v === "string" ? v : JSON.stringify(v)} />
            ))}
          </div>
        </Section>
      ) : null}
    </div>
  );
}

function DepTable({ deps, empty }: { deps: ResolvedDep[]; empty: string }) {
  if (deps.length === 0) {
    return <div className="text-2xs font-mono text-ink-500">{empty}</div>;
  }
  return (
    <table className="w-full text-2xs font-mono">
      <thead>
        <tr className="text-ink-400">
          <th className="text-left font-normal pr-3">step</th>
          <th className="text-left font-normal pr-3">runner</th>
          <th className="text-left font-normal pr-3">status</th>
          <th className="text-left font-normal">artifact</th>
        </tr>
      </thead>
      <tbody>
        {deps.map((d) => (
          <tr key={d.step_name} className="border-t border-ink-800">
            <td className="pr-3 py-1 text-ink-200">{d.step_name}</td>
            <td className="pr-3 py-1 text-ink-400">{d.runner ?? "—"}</td>
            <td className="pr-3 py-1 text-ink-300">
              {d.status ? (
                <span className="inline-flex items-center gap-1">
                  <span className={`dot ${statusDotClass(d.status)}`} />
                  {d.status}
                </span>
              ) : (
                "—"
              )}
            </td>
            <td className="py-1 text-ink-500 truncate" title={d.artifact_id ?? ""}>
              {d.artifact_id ? d.artifact_id.slice(0, 22) : "—"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

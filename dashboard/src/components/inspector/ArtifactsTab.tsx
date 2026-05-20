import type { StepDetail } from "@/types/api";
import { Section, KV } from "@/components/Inspector";
import { JsonView } from "@/components/JsonView";
import { shortHash } from "@/lib/format";

export function ArtifactsTab({ detail }: { detail: StepDetail }) {
  const a = detail.artifact;
  if (!a) {
    return (
      <div className="p-3 text-2xs font-mono text-ink-500">
        No artifact recorded for this step.
      </div>
    );
  }
  return (
    <div className="p-3 space-y-3">
      <Section title="identity">
        <div className="grid grid-cols-1 gap-1 text-2xs font-mono">
          <KV label="artifact_id" value={a.artifact_id} />
          <KV label="artifact_kind" value={a.artifact_kind} />
          <KV label="schema_version" value={String(a.schema_version)} />
          <KV label="created_at" value={a.created_at} />
          <KV label="operation_spec_hash" value={shortHash(a.operation_spec_hash, 16)} />
          <KV label="operation_semantic_hash" value={shortHash(a.operation_semantic_hash, 16)} />
        </div>
      </Section>
      <Section title="storage refs">
        <JsonView value={a.storage_refs} />
      </Section>
      <Section title="runner">
        <JsonView value={a.runner} collapsed={true} />
      </Section>
      <Section title="engine">
        <JsonView value={a.engine} collapsed={true} />
      </Section>
      <Section title="metadata">
        <JsonView value={a.metadata} collapsed={true} />
      </Section>
    </div>
  );
}

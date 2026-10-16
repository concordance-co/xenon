"""Bounded local evidence reports; never localize captures or generation sets."""
from __future__ import annotations

from typing import Any, Sequence

from pipelines_v2.experiments import EvidenceRecord, ExperimentPack
from pipelines_v2.experiments.evidence import pointer_value
from pipelines_v2.experiments.packs import canonical, fields, require
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact

MAX_RESULT_BYTES = 256 * 1024
MAX_REPORT_BYTES = 1024 * 1024


def evidence_report_payload(bundle: dict, inputs: Sequence[Any]) -> dict:
    canonical(bundle, limit=MAX_REPORT_BYTES)
    fields(bundle, 'pack records', 'evidence bundle')
    pack = ExperimentPack.from_dict(bundle['pack'])
    require(isinstance(bundle['records'], list) and 0 < len(bundle['records']) <= 100, 'Provide 1–100 evidence records')
    records = [EvidenceRecord.from_dict(record, pack=pack) for record in bundle['records']]
    require(len({record.to_dict()['evidence_id'] for record in records}) == len(records), 'Duplicate evidence IDs')
    require(all(isinstance(item, (CaptureArtifact, OperationArtifact)) for item in inputs), 'Evidence reports require direct artifact inputs')
    by_id = {item.id: item for item in inputs}
    require(len(by_id) == len(inputs), 'Duplicate direct input artifacts')
    manifests = {key: item.manifest() for key, item in by_id.items()}
    results = {}
    total_bytes = 0
    for record in records:
        body = record.to_dict()
        for link in body['links']:
            artifact_id = link['artifact_id']
            require(artifact_id in by_id, f'Missing direct evidence input: {artifact_id}')
            manifest = manifests[artifact_id]
            require(manifest.workflow_context.get('run_id') == link['run_id'], 'Evidence run ID does not match artifact')
            require(manifest.workflow_context.get('workflow_hash') == pack.to_dict()['presets'][body['preset']]['workflow_hash'], 'Evidence artifact belongs to a different workflow preset')
            step = manifest.workflow_context.get('step_name')
            require(pack.to_dict()['expected_outputs'].get(step) == manifest.artifact_kind, 'Evidence artifact step/kind does not match pack')
            provenance = manifest.provenance()
            require(provenance is not None and provenance.scientific_identity == link['scientific_identity'], 'Evidence provenance mismatch or legacy artifact')
            selector = link['selector']
            if selector['domain'] == 'manifest':
                source = manifest.to_dict()
            else:
                require(isinstance(by_id[artifact_id], OperationArtifact), 'Capture payloads cannot be report evidence downloads')
                if artifact_id not in results:
                    ref = manifest.storage_refs.get('result', {})
                    size = ref.get('bytes')
                    require(type(size) is int and 0 <= size <= MAX_RESULT_BYTES, 'Result size must be known and within report cap before reading')
                    total_bytes += size
                    require(total_bytes <= MAX_REPORT_BYTES, 'Combined result payloads exceed report cap')
                    # Only the declared small JSON result; no feature/generation reads.
                    result = by_id[artifact_id].result()
                    canonical(result, limit=MAX_RESULT_BYTES)
                    results[artifact_id] = result
                source = results[artifact_id]
            require(pointer_value(source, selector['pointer']) == selector['expected'], 'Evidence selector value does not match artifact')
    summaries = []
    for key, manifest in manifests.items():
        provenance = manifest.provenance()
        summaries.append({'artifact_id': key, 'artifact_kind': manifest.artifact_kind,
                          'run_id': manifest.workflow_context.get('run_id'),
                          'scientific_identity': provenance.scientific_identity if provenance else None,
                          'provenance_status': manifest.provenance_status,
                          'unresolved': provenance.to_dict()['unresolved'] if provenance else ['legacy']})
    payload = {'pack': pack.to_dict(), 'pack_identity': pack.identity, 'study_id': pack.to_dict()['study_id'],
               'phase_id': pack.to_dict()['phase_id'], 'records': [record.to_dict() for record in records],
               'inputs': summaries, 'diagnostics': {'linked_results': len(results), 'result_bytes': total_bytes,
                                                   'qualification': 'Engineering status is recorded evidence, not an admission grant.'}}
    canonical(payload, limit=MAX_REPORT_BYTES)
    return payload


def render_evidence_markdown(payload: dict) -> str:
    """Text-only, escaped bounded components for provenance, controls and verdicts."""
    import html
    def escaped(value):
        return html.escape(str(value)).replace('`', '&#96;').replace('\n', ' ')
    lines = ['## Experiment evidence', '', f"Pack: `{escaped(payload['pack_identity'])}`", '']
    for record in payload['records']:
        lines += [f"### {escaped(record['evidence_id'])}", '',
                  f"Engineering: **{escaped(record['engineering_status'])}**; scientific verdict: **{escaped(record['scientific_verdict'])}**.",
                  f"Scope: {escaped(record['preset'])}, {escaped(record['split'])}; evidence level: {escaped(record['evidence_level'])}.",
                  f"Metric: {escaped(record['metric'])}; effect: {escaped(record['effect']['value'])}.",
                  f"Uncertainty: {escaped(record['uncertainty']['value'] if record['uncertainty']['value'] is not None else 'unavailable')} — {escaped(record['uncertainty']['reason'])}",
                  f"Controls: {escaped(', '.join(record['control_arms']) or 'none observed')}.", '', 'Evidence links:', '']
        for link in record['links']:
            lines.append(f"- Run `{escaped(link['run_id'])}`, artifact `{escaped(link['artifact_id'])}`, {escaped(link['selector']['domain'])} selector `{escaped(link['selector']['pointer'])}`; provenance `{escaped(link['scientific_identity'])}`.")
        lines += ['', 'Limitations:', ''] + [f'- {escaped(item)}' for item in record['limitations']] + ['']
    lines += ['### Provenance and diagnostics', '']
    for item in payload['inputs']:
        lines.append(f"- `{escaped(item['artifact_id'])}`: {escaped(item['provenance_status'])}; unresolved: {escaped(', '.join(item['unresolved']) or 'none')}.")
    lines += ['', f"Bounded results read: {payload['diagnostics']['linked_results']} ({payload['diagnostics']['result_bytes']} bytes).", escaped(payload['diagnostics']['qualification']), '']
    return '\n'.join(lines)

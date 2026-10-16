"""Observed identity for explicitly supported pure summaries and report packaging."""
from __future__ import annotations

from pathlib import PurePath
from typing import Any

from .provenance_observation import OperationFacts


def payload_identity(manifest) -> dict[str, Any]:
    """Bind C03's declared bytes without downloading remote tensors.

    This verifies the completion descriptor, not its payload files. The existing
    execution/reuse C03 integrity gate remains responsible for byte validation.
    """
    from .provenance import content_hash, _is_digest
    from pipelines_v2.storage.finalization import payload_paths
    record = manifest.metadata.get('completion_v1')
    if not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('state') != 'complete':
        raise ValueError('Complete upstream payload inventory required')
    body = {key: value for key, value in record.items() if key != 'integrity_hash'}
    parent = manifest.provenance()
    if (record.get('integrity_hash') != content_hash(body)
            or record.get('provenance_hash') != (parent.scientific_identity if parent else None)
            or record.get('operation_semantic_hash') != manifest.operation_semantic_hash
            or record.get('expected_coverage') != manifest.example_coverage):
        raise ValueError('Upstream completion binding differs from manifest')
    payloads = record.get('payloads')
    if not isinstance(payloads, list) or payload_paths(manifest.storage_refs) != sorted(p.get('path', '') for p in payloads):
        raise ValueError('Upstream completion does not cover manifest payloads')
    values = []
    for item in payloads:
        if not _is_digest(item.get('sha256')) or type(item.get('bytes')) is not int or item['bytes'] < 0:
            raise ValueError('Invalid upstream content descriptor')
        parts = PurePath(item['path']).parts
        # Stores may relocate but the logical path below the artifact is stable.
        if manifest.artifact_id not in parts:
            raise ValueError('Cannot identify logical upstream payload path')
        logical = '/'.join(parts[parts.index(manifest.artifact_id) + 1:])
        if not logical or '..' in PurePath(logical).parts:
            raise ValueError('Invalid logical upstream payload path')
        values.append({'name': logical, 'bytes': item['bytes'], 'sha256': item['sha256']})
    if len({item['name'] for item in values}) != len(values):
        raise ValueError('Ambiguous upstream payload names')
    return {'artifact_kind': manifest.artifact_kind, 'payloads': sorted(values, key=lambda x: x['name'])}


_PURE_SUMMARIES = {
    'pipelines_v2.operations.attribution.report:summarize_attribution': 'c06-validated-inventory-summary-v1',
    'pipelines_v2.operations.neurons.transforms:summarize_capture': 'signed-neuron-ordered-python-reductions-v1',
    'pipelines_v2.operations.neurons.transforms:dossier_transform': 'stored-explanation-dossier-v1',
}


def summary_facts(spec) -> OperationFacts | None:
    from .provenance import content_hash, _walk, _without_credentials
    from pipelines_v2.operations.common.builders import TransformBuilder
    from pipelines_v2.operations.derive.transforms import TransformSpec
    from pipelines_v2.operations.reports import ReportSpec

    fields = {}
    seeds = {}
    if type(spec) is TransformSpec and type(spec.builder) is TransformBuilder:
        policy = _PURE_SUMMARIES.get(spec.builder.import_path)
        if policy is None:
            return None
        # An import ref is code, not a file input. Complete mounted source bytes
        # remain mandatory and include the implementation plus its helpers.
        fields['builder'] = {'import_path': spec.builder.import_path}
        if spec.builder.import_path.endswith(':summarize_capture'):
            from pipelines_v2.operations.neurons.summaries import SummaryLimits
            limits = SummaryLimits(**(spec.inputs.get('limits') or {}))
            seeds['summary.reservoir'] = limits.seed
    elif type(spec) is ReportSpec and spec.template == 'summary':
        policy = 'builtin-summary-evidence-packaging-v1'
    else:
        return None
    # These producers expose input artifact IDs/locations and metadata in their
    # output. Bind those presentation facts too; do not reuse a report pointing
    # at another run/store just because upstream scientific identity is equal.
    manifests = [item.manifest().to_dict() for item in _walk(spec)
                 if callable(getattr(item, 'manifest', None))]
    return OperationFacts(field_identities=fields,
        numerics={'dtype': 'preserved-source-and-python-float64-reductions',
                  'attention_implementation': 'not_applicable', 'device': 'cpu',
                  'determinism': 'ordered-inputs-and-explicit-hash-reservoir-seed',
                  'compile': False, 'matmul': 'not_applicable', 'policy_version': policy},
        methods={'policy': policy, 'input_presentation_digest': content_hash(_without_credentials(manifests))},
        seeds=seeds)

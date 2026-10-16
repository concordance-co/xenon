"""Catalog queries never launch jobs or read activation/generation payloads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class ViewLimits:
    result_bytes: int = 16 * 1024**2
    examples: int = 100
    graph_nodes: int = 100
    graph_edges: int = 300
    images: int = 32

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError('Evidence view limits must be positive integers')


def _read_json(ref, limit):
    """Bound actual bytes in transit, not only the manifest's declared size."""
    if ref.get('format') != 'json' or type(ref.get('bytes')) is not int or not 0 <= ref['bytes'] <= limit:
        raise ValueError('A bounded JSON result reference is required')
    path = Path(ref['path'])
    if ref['store'] in {'local', 'local_path'}:
        with path.open('rb') as stream:
            data = stream.read(limit + 1)
    elif ref['store'] == 'modal_volume':
        import modal
        from pipelines_v2.storage.modal import modal_volume_mount_path
        remote = PurePosixPath(str(path)).relative_to(PurePosixPath(modal_volume_mount_path(str(path))))
        parts, count = [], 0
        for part in modal.Volume.from_name(ref['name']).read_file(str(remote)):
            count += len(part)
            if count > limit:
                raise ValueError('Remote summary exceeds view byte cap')
            parts.append(part)
        data = b''.join(parts)
    else:
        raise ValueError('Unsupported summary store')
    if len(data) > limit or len(data) != ref['bytes']:
        raise ValueError('Summary byte count differs from declared bounded reference')
    return data


class EvidenceReader:
    def __init__(self, catalog, *, limits=None, read_json_bytes=None):
        self.catalog = catalog
        self.limits = limits or ViewLimits()
        self.read_json_bytes = read_json_bytes or _read_json
        self.bytes_read = 0
        self._results = {}

    def runs(self, *, study=None, model=None, operation=None, status=None, offset=0, limit=25):
        if not 1 <= limit <= 100 or not 0 <= offset <= 10000:
            raise ValueError('Use explicit pages of 1–100 runs, offset at most 10000')
        # Existing catalog API has no cursor. Bound the scan, report pagination.
        rows = self.catalog.list_workflow_runs(status=status, limit=10001)
        found = []
        for run in rows:
            if study and study.casefold() not in str(run.workflow_name).casefold():
                continue
            steps = list(run.workflow_payload.get('steps', ()))
            if operation and not any(s.get('spec', {}).get('kind') == operation for s in steps):
                continue
            if model and not any(model.casefold() in str(s.get('spec', {}).get('engine', {})).casefold() for s in steps):
                continue
            found.append({k:v for k,v in run.to_dict().items() if k != 'workflow_payload'})
        return {'runs': found[offset:offset+limit], 'offset': offset,
                'next_offset': offset+limit if offset+limit < len(found) else None,
                'scan_truncated': getattr(self.catalog, 'scan_truncated', len(rows) == 10001),
                'scan_files': getattr(self.catalog, 'scan_files', None),
                'scan_bytes': getattr(self.catalog, 'scan_bytes', None),
                'scan_scope': 'Bounded discovery window; truncation means additional matches may exist outside this page.'}

    def run(self, run_id):
        run = self.catalog.load_workflow_run(run_id)
        if run is None:
            raise KeyError(f'Unknown cataloged run: {run_id}')
        rows = []
        for record in self.catalog.list_workflow_steps(run_id):
            row = record.to_dict()
            if record.artifact_id:
                try:
                    row['artifact'] = self.artifact(record.artifact_id)
                except (KeyError, ValueError, OSError) as exc:
                    row['read_error'] = str(exc)
            rows.append(row)
        return {'run': run.to_dict(), 'steps': rows,
                'replay': {'workflow_config': dict(run.workflow_payload),
                           'command': None,
                           'reason': 'Catalog stores the exact workflow config, but not a verified source-file path. Supply its original workflow.py; opening this view never executes it.'}}

    def artifact(self, artifact_id, *, include_result=False):
        manifest = self.catalog.load_artifact(artifact_id)
        if manifest is None:
            raise KeyError(f'Unknown cataloged artifact: {artifact_id}')
        provenance = manifest.provenance()
        body = provenance.to_dict() if provenance else {}
        completion = manifest.metadata.get('completion_v1', {})
        result = {'artifact_id': artifact_id, 'kind': manifest.artifact_kind,
                  'run_id': manifest.workflow_context.get('run_id'),
                  'scientific_identity': body.get('scientific_identity'),
                  'identity_complete': bool(provenance and provenance.identity_complete),
                  'unresolved': body.get('unresolved', ['legacy provenance']),
                  'completion_declared': completion.get('state', 'legacy'),
                  'qualification': manifest.metadata.get('qualification_v1', {'status': 'unverified'}),
                  'provenance': body, 'input_artifact_refs': list(manifest.input_artifact_refs)}
        if include_result:
            payload, checksum = self._bounded_result(manifest)
            self._check_panel_bounds(payload)
            result.update(result=payload, summary_checksum_verified=checksum)
            bundle = evidence_bundle(payload)
            if bundle is not None:
                result['evidence_verification'] = self._verify_evidence(bundle)
        return result

    def _bounded_result(self, manifest, *, max_bytes=None):
        if manifest.artifact_kind in {'capture', 'generation', 'generation_run', 'patched_generation', 'diffusion_capture'}:
            raise ValueError('Raw captures and generation sets are not evidence summaries')
        ref = manifest.storage_refs.get('result')
        if not isinstance(ref, dict):
            raise ValueError('Artifact has no declared bounded result')
        size = ref.get('bytes')
        if type(size) is not int or not 0 <= size <= (max_bytes or self.limits.result_bytes):
            raise ValueError('Summary exceeds individual result byte cap')
        key = getattr(manifest, 'artifact_id', None)
        if key is not None and key in self._results:
            return self._results[key]
        remaining = self.limits.result_bytes - self.bytes_read
        if size > remaining:
            raise ValueError('Summary exceeds remaining view byte cap')
        data = self.read_json_bytes(ref, min(remaining, max_bytes or remaining))
        self.bytes_read += len(data)
        if self.bytes_read > self.limits.result_bytes or len(data) != size:
            raise ValueError('Summary exceeds actual byte budget or declared size')
        completion = manifest.metadata.get('completion_v1', {})
        checksums = [p for p in completion.get('payloads', ()) if p['path'] == ref['path']]
        if checksums and (len(checksums) != 1 or hashlib.sha256(data).hexdigest() != checksums[0]['sha256']):
            raise ValueError('Result checksum mismatch')
        value = (json.loads(data), bool(checksums))
        if key is not None:
            self._results[key] = value
        return value

    def _verify_evidence(self, bundle):
        from pipelines_v2.experiments import ExperimentPack, EvidenceRecord
        from pipelines_v2.experiments.evidence import pointer_value
        from pipelines_v2.experiments.packs import canonical, require
        try:
            canonical(bundle, limit=1024**2)
            require(isinstance(bundle, dict) and 'pack' in bundle, 'Legacy evidence has no preserved pack; unverified')
            pack = ExperimentPack.from_dict(bundle['pack'])
            plan = pack.to_dict()
            require(bundle.get('pack_identity', pack.identity) == pack.identity, 'Evidence bundle pack identity mismatch')
            require(isinstance(bundle.get('records'), list) and 0 < len(bundle['records']) <= 100, 'Provide 1–100 C05 records')
            records = [EvidenceRecord.from_dict(record, pack=pack).to_dict() for record in bundle['records']]
            require(len({r['evidence_id'] for r in records}) == len(records), 'Duplicate evidence IDs')
            manifests, signatures = {}, {}
            linked_bytes = 0
            loaded_results = set()
            for record in records:
                linked_provenance = []
                graph_semantics = None
                for link_index, link in enumerate(record['links']):
                    key = link['artifact_id']
                    if key not in manifests:
                        require(len(manifests) < 100, 'Evidence exceeds 100 direct linked artifacts')
                        manifests[key] = self.catalog.load_artifact(key)
                    manifest = manifests[key]
                    require(manifest is not None, f'Missing direct evidence input: {key}')
                    require(manifest.artifact_id == key, 'Catalog artifact identity mismatch')
                    provenance = manifest.provenance()
                    require(provenance is not None and provenance.identity_complete, 'Linked execution identity incomplete or legacy')
                    require(provenance.scientific_identity == link['scientific_identity'], 'Evidence provenance mismatch')
                    linked_provenance.append(provenance.to_dict())
                    selector = link['selector']
                    published = 'publication_v1' in manifest.metadata
                    # Published lineage is usable only after a bounded checksum read,
                    # even when the evidence selector addresses the manifest.
                    if selector['domain'] != 'manifest' or published:
                        ref = manifest.storage_refs.get('result', {})
                        size = ref.get('bytes')
                        require(type(size) is int and 0 <= size <= 256 * 1024, 'Linked result exceeds 256 KiB cap')
                        if key not in loaded_results:
                            linked_bytes += size
                            loaded_results.add(key)
                        require(linked_bytes <= 1024**2, 'Linked results exceed 1 MiB cap')
                        # One hop only: never invoke artifact()/evidence validation here.
                        source, verified = self._bounded_result(manifest, max_bytes=256 * 1024)
                        require(verified, 'Linked result has no checksum proof')
                    context, source_kind = manifest.workflow_context, manifest.artifact_kind
                    if published:
                        from pipelines_v2.storage.publication import publication_source_lineage
                        context, source_kind = publication_source_lineage(manifest)
                    require(context.get('run_id') == link['run_id'], 'Evidence run ID does not match artifact')
                    require(context.get('workflow_hash') == plan['presets'][record['preset']]['workflow_hash'], 'Evidence artifact belongs to a different workflow preset')
                    require(plan['expected_outputs'].get(context.get('step_name')) == source_kind, 'Evidence artifact step/kind does not match pack')
                    if selector['domain'] == 'manifest':
                        source = manifest.to_dict()
                    if link_index == record['effect']['link'] and isinstance(source, dict):
                        graph = source.get('graph', {})
                        graph_semantics = graph.get('semantics') if isinstance(graph, dict) else None
                    require(pointer_value(source, selector['pointer']) == selector['expected'], 'Evidence selector value does not match artifact')
                provenance = linked_provenance[record['effect']['link']]
                signatures[record['evidence_id']] = {
                    **{key:record[key] for key in ('metric','evaluator_digest','split','pack_identity','evidence_level','preset','question_id','hypothesis_id')},
                    'metric_definition':plan['metrics'][record['metric']], 'graph_semantics':graph_semantics,
                    **{key:provenance.get(key) for key in ('model_components','methods','numerics','data')},
                }
            return {'status':'verified', 'bundle_digest':hashlib.sha256(canonical(bundle, limit=1024**2).encode()).hexdigest(),
                    'record_signatures':signatures, 'linked_artifacts':len(manifests), 'linked_result_bytes':linked_bytes}
        except (KeyError, ValueError, OSError, TypeError) as exc:
            return {'status':'unverified', 'reason':str(exc)}

    def _check_panel_bounds(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Evidence summary must be an object')
        limits = self.limits
        bundle = evidence_bundle(payload)
        if isinstance(bundle, dict) and isinstance(bundle.get('records'), list) and len(bundle['records']) > limits.examples:
            raise ValueError('Evidence records exceed view example cap')
        for key in ('examples', 'dossiers', 'records'):
            if isinstance(payload.get(key), list) and len(payload[key]) > limits.examples:
                raise ValueError(f'{key} exceeds view example cap; request a reduced upstream summary')
        if len(payload.get('cells', ())) > limits.images:
            raise ValueError('Gallery exceeds image cap')
        graph = payload.get('graph', {})
        if len(graph.get('nodes', ())) > limits.graph_nodes or len(graph.get('edges', ())) > limits.graph_edges:
            raise ValueError('Graph exceeds node/edge cap; request a pruned upstream graph')
        if len(graph.get('evidence', ())) > limits.examples:
            raise ValueError('Graph exceeds evidence example cap')
        contexts = set()
        for dossier in payload.get('dossiers', ()):
            for rows in dossier.get('observations', {}).values():
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            contexts.add(row.get('context_id', json.dumps(row, sort_keys=True)))
        if len(contexts) > limits.examples:
            raise ValueError('Neuron contexts exceed example cap')


def evidence_bundle(payload):
    """ReportSpec nests its C05 bundle; standalone evidence transforms may not."""
    if not isinstance(payload, dict):
        return None
    if 'evidence' in payload:
        return payload['evidence']
    return payload if 'records' in payload else None


def compare_evidence(artifacts, *, metric):
    """Compare reader-verified C05 records; no implicit averaging or metric guessing."""
    from pipelines_v2.experiments.packs import canonical
    if not 2 <= len(artifacts) <= 32:
        raise ValueError('Compare 2–32 explicit artifacts')
    rows, signatures = [], []
    for artifact in artifacts:
        reasons = []
        if not artifact.get('identity_complete'):
            reasons.append('execution identity incomplete')
        bundle = evidence_bundle(artifact.get('result', {}))
        verification = artifact.get('evidence_verification', {})
        signature = None
        value = None
        try:
            observed_digest = hashlib.sha256(canonical(bundle, limit=1024**2).encode()).hexdigest()
            if verification.get('status') != 'verified' or verification.get('bundle_digest') != observed_digest:
                reasons.append('C05 evidence unverified: '+verification.get('reason', 'validated pack and linked measurements required'))
            else:
                records = [r for r in bundle['records'] if r['metric'] == metric]
                if len(records) != 1:
                    reasons.append('exactly one explicit evidence metric record required')
                else:
                    record = records[0]
                    signature = verification['record_signatures'][record['evidence_id']]
                    value = record['effect']['value']
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append('C05 evidence unverified: '+str(exc))
        signatures.append(signature)
        rows.append({'artifact_id': artifact['artifact_id'], 'value': value, 'reasons': reasons})
    reference = signatures[0]
    for row, signature in zip(rows, signatures):
        if signature is not None and signature != reference:
            row['reasons'].append('evaluator/split/model/hook/numerical semantics differ')
    comparable = reference is not None and all(not row['reasons'] for row in rows)
    return {'metric': metric, 'comparable': comparable, 'rows': rows,
            'signatures': signatures, 'aggregation': 'none',
            'note': 'A comparable definition is not evidence of scientific quality or resource qualification.'}

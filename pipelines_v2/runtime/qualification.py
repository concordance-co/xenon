"""Exact, measured SDXL canary receipts. No estimates or cross-profile promotion."""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import threading
import time
import uuid

from pipelines_v2.core.provenance import content_hash
from pipelines_v2.core.types import utc_now_iso
from pipelines_v2.storage.artifacts import ArtifactManifest
from pipelines_v2.storage.finalization import finalize_manifest, validate_completion
from pipelines_v2.storage.finalization import writer_lock

RESET_POLICY = 'sdxl-euler-local-rng-owned-hooks-v1'
HEADROOM = 0.10


class QualificationFailed(RuntimeError):
    pass


def gpu_driver_version():
    import subprocess
    try:
        return subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],
            capture_output=True,text=True,timeout=5,check=True).stdout.strip() or None
    except (OSError,subprocess.SubprocessError):
        return None


class RSSSampler:
    """Sample process-tree RSS, including start/end. Values are observed, not estimates."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = 0
        self.stop = threading.Event()
        self.started = time.monotonic()
        self._sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        import psutil
        total = 0
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak = max(self.peak, total)

    def _run(self):
        while not self.stop.wait(0.02):
            self._sample()

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join()
        self._sample()
        self.seconds = time.monotonic() - self.started


def session_key(provenance, runner_config):
    body = provenance.to_dict()
    return content_hash({key: body[key] for key in ('source', 'runtime', 'model_components', 'numerics')} | {
        'resources': runner_config.get('resources', {}), 'reset_policy': RESET_POLICY})


def profile_identity(admission):
    body = admission['provenance'] if 'provenance' in admission else admission['provenance_v1']
    return content_hash({'envelope': admission['envelope'],
        **{key: body[key] for key in ('source', 'runtime', 'model_components', 'numerics')},
        'reset_policy': RESET_POLICY, 'required_headroom_fraction': HEADROOM})


def assess_measurements(metadata, *, host_limit_mb):
    loading = metadata.get('loading', {})
    values = {'loading_rss_bytes': loading.get('peak_rss_bytes'),
        'execution_rss_bytes': metadata.get('peak_rss_bytes'),
        'loading_vram_bytes': loading.get('peak_reserved_vram_bytes'),
        'execution_vram_bytes': metadata.get('peak_reserved_vram_bytes'),
        'loading_seconds': loading.get('seconds'), 'execution_seconds': metadata.get('seconds'),
        'device_capacity_bytes': loading.get('total_vram_bytes')}
    missing = [key for key, value in values.items()
               if type(value) not in (int, float) or not math.isfinite(value) or value <= 0]
    if type(host_limit_mb) is not int or host_limit_mb <= 0:
        missing.append('host_memory_limit')
    if not loading.get('driver_version'):
        missing.append('driver_version')
    if missing:
        return {'status': 'unresolved', 'reason': 'Missing measurements: ' + ', '.join(missing), **values}
    host_limit = host_limit_mb * 1024**2
    host_fraction = max(values['loading_rss_bytes'], values['execution_rss_bytes']) / host_limit
    device_fraction = max(values['loading_vram_bytes'], values['execution_vram_bytes']) / values['device_capacity_bytes']
    passed = max(host_fraction, device_fraction) <= 1 - HEADROOM
    return {'status': 'passed' if passed else 'failed', 'reason': 'measured headroom' if passed else 'insufficient measured headroom',
        'host_limit_bytes': host_limit, 'host_headroom_fraction': 1-host_fraction,
        'device_headroom_fraction': 1-device_fraction, 'required_headroom_fraction': HEADROOM,
        'driver_version': loading['driver_version'], 'device': loading.get('device'), **values}


class CanaryGate:
    """Hold one exact profile owner until its first useful image has a durable receipt."""
    def __init__(self, *, store, spec, runner_config, context, progress_callback=None):
        self.store, self.spec, self.runner_config, self.context = store, spec, runner_config, context
        self.admission = context['admission_v1']
        self.profile = profile_identity(self.admission)
        self.callback = progress_callback
        self.receipt = None
        self.failure_recorded = False
        shapes = {(r.width, r.height, r.steps, r.guidance_scale) for r in spec.images}
        if len(shapes) != 1:
            raise ValueError('Measured SDXL canary requires homogeneous image shape/steps/CFG within one operation')

    def _event(self, stage, **extra):
        if self.callback:
            self.callback({'stage': stage, 'step_name': self.context.get('step_name'),
                           'profile_identity': self.profile, **extra})

    @contextmanager
    def owner(self):
        self._event('qualification_pending')
        with writer_lock(self.store, 'qualification-' + self.profile):
            pointer = Path(self.store.root) / '_qualification_profiles' / (self.profile + '.json')
            if pointer.exists():
                import json
                payload = json.loads(pointer.read_text())
                manifest = ArtifactManifest.from_dict(self.store.read_json_ref(payload))
                validate_completion(manifest, store=self.store)
                result = self.store.read_json_ref(manifest.storage_refs['result'])
                if result['profile_identity'] != self.profile or result['measurements']['status'] != 'passed':
                    raise QualificationFailed('Qualification index does not contain an exact passed profile')
                self.receipt = manifest.to_dict()
            try:
                yield self
            except BaseException as error:
                if self.receipt is None and not self.failure_recorded:
                    self._record_execution_failure(error)
                raise

    def _record_execution_failure(self, error):
        from pipelines_v2.core.provenance import context_provenance
        artifact_id='qualification_' + self.profile[:16] + '_' + uuid.uuid4().hex[:12]
        result={'schema_version':1,'profile_identity':self.profile,'envelope':self.admission['envelope'],
                'admission_identity':content_hash(self.admission),
                'measurements':{'status':'failed','reason':'canary execution failed','error_type':type(error).__name__},
                'sample_image':None,'coverage':{'example_count':0,'example_keys':[]},'reset_policy':RESET_POLICY}
        manifest=ArtifactManifest(artifact_id=artifact_id,artifact_kind='qualification',schema_version=1,
            operation_spec_hash=self.spec.spec_hash(),operation_semantic_hash=self.spec.semantic_hash(),created_at=utc_now_iso(),
            engine=self.spec.engine.identity(),runner=self.runner_config,input_artifact_refs=(),example_coverage=result['coverage'],
            storage_refs={'result':self.store.write_json(artifact_id,'result.json',result)},
            metadata={'provenance_v1':context_provenance(self.context).to_dict(),'qualification_v1':result},workflow_context=self.context)
        finalize_manifest(self.store,manifest)
        self.failure_recorded=True
        self._event('qualification_failed',artifact_id=artifact_id)

    def observe(self, image_manifest):
        if self.receipt is not None:
            return self.receipt
        from pipelines_v2.core.provenance import context_provenance
        measurements = assess_measurements(image_manifest.metadata, host_limit_mb=self.runner_config['resources'].get('memory_mb'))
        expected = {(index, step, hook.operator, hook.branch)
                    for index, hook in enumerate(getattr(self.spec, 'hooks', ())) for step in hook.steps}
        events = image_manifest.metadata.get('hook_events', ())
        actual = {(row['hook_index'], row['step_index'], row['operator'], row['branch']) for row in events}
        if actual != expected or len(events) != len(expected):
            measurements.update(status='failed', reason='Actual hook/denoising coverage differs from exact profile')
        artifact_id = 'qualification_' + self.profile[:16] + '_' + uuid.uuid4().hex[:12]
        result = {'schema_version': 1, 'profile_identity': self.profile, 'envelope': self.admission['envelope'],
            'admission_identity': content_hash(self.admission), 'measurements': measurements,
            'sample_image': image_manifest.artifact_id, 'coverage': dict(image_manifest.example_coverage),
            'session': image_manifest.metadata.get('session_v1'), 'reset_policy': RESET_POLICY}
        manifest = ArtifactManifest(artifact_id=artifact_id, artifact_kind='qualification', schema_version=1,
            operation_spec_hash=self.spec.spec_hash(), operation_semantic_hash=self.spec.semantic_hash(),
            created_at=utc_now_iso(), engine=self.spec.engine.identity(), runner=self.runner_config,
            input_artifact_refs=(image_manifest.artifact_id,), example_coverage=dict(image_manifest.example_coverage),
            storage_refs={'result': self.store.write_json(artifact_id, 'result.json', result)},
            metadata={'provenance_v1': context_provenance(self.context).to_dict(), 'qualification_v1': result},
            workflow_context=self.context)
        manifest = finalize_manifest(self.store, manifest)
        if measurements['status'] != 'passed':
            self.failure_recorded=True
            commit=getattr(self.store,'commit_visibility',None)
            if callable(commit):commit()
            self._event('qualification_' + measurements['status'], artifact_id=artifact_id)
            raise QualificationFailed(f"{measurements['reason']}; receipt={artifact_id}; no remaining images submitted")
        path = Path(self.store.root) / artifact_id / 'manifest.json'
        ref = {'store': self.store.kind, 'path': str(path), 'format': 'json', 'bytes': path.stat().st_size}
        if self.store.kind == 'modal_volume':
            ref['name'] = self.store.name
        self.store.write_json('_qualification_profiles', self.profile + '.json', ref)
        commit=getattr(self.store,'commit_visibility',None)
        if callable(commit):commit()
        self._event('qualification_passed', artifact_id=artifact_id)
        self.receipt = manifest.to_dict()
        return self.receipt

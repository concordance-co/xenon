"""Opt-in, bounded SDXL workflow scope. Other engines retain their existing transport."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
import threading
import time
import uuid

from pipelines_v2.core.provenance import content_hash


def scoped_workflow(method):
    """Add runner lifetimes around the existing DAG, without changing scheduling."""
    @wraps(method)
    def wrapped(self, workflow, *args, **kwargs):
        with ExitStack() as stack:
            seen = set()
            for runner in self.runners.values():
                if id(runner) in seen:
                    continue
                seen.add(id(runner))
                scope = getattr(runner, 'workflow_scope', None)
                if callable(scope):
                    stack.enter_context(scope())
            return method(self, workflow, *args, **kwargs)
    return wrapped


class WorkerState:
    """One container-local cache. Timer cannot close a model during a call."""
    def __init__(self, *, idle_seconds=300):
        if not 0 < idle_seconds <= 300:
            raise ValueError('SDXL idle bound must be at most 300 seconds')
        self.idle_seconds = idle_seconds
        self.worker_id = uuid.uuid4().hex
        self.model_loads = 0
        self._diffusion_sessions = {}
        self._lock = threading.RLock()
        self._timer = None

    def close(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            for session in self._diffusion_sessions.values():
                session.close()
            self._diffusion_sessions.clear()

    @contextmanager
    def operation(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            try:
                yield self
            except BaseException:
                self.close()
                raise
            finally:
                self._timer = threading.Timer(self.idle_seconds, self.close)
                self._timer.daemon = True
                self._timer.start()


_WORKERS = {}


def dispatch_worker(scope_id, mode, runner_config, store_config, spec_payload, context, report):
    """The imported module owns the cache, never the serialized function closure."""
    import os
    from pipelines_v2.operations import operation_spec_from_dict
    from pipelines_v2.storage.registry import artifact_store_from_dict
    from pipelines_v2.runtime.readiness import admit_execution
    state = _WORKERS.setdefault(scope_id, WorkerState())
    if mode == 'close':
        state.close()
        _WORKERS.pop(scope_id, None)
        return {'closed': True, 'worker_id': state.worker_id, 'model_load_count': state.model_loads}
    with state.operation():
        spec = operation_spec_from_dict(spec_payload)
        if spec.bound_engine().identity().get('kind') != 'sdxl':
            raise ValueError('SDXL workflow scope cannot execute another engine')
        if mode == 'observe':
            record = admit_execution(spec, runner_config, artifact_store=artifact_store_from_dict(store_config),
                image_identity=os.environ.get('MODAL_IMAGE_ID'), check_writer_registry=True)
            if not record.ready or not record.provenance.identity_complete:
                raise ValueError('Fresh GPU admission is incomplete: ' + str(record.to_dict()))
            return {'admission': record.to_dict(), 'worker_id': state.worker_id,
                    'model_load_count': state.model_loads, 'gpu_allocated': True}
        if mode != 'execute':
            raise ValueError('Unknown scoped execution mode')
        from pipelines_v2.runtime.remote_executor import _execute_remote_spec
        return _execute_remote_spec(runner_config={**runner_config, 'admission_required': True},
            store_config=store_config, spec=spec, workflow_context={**(context or {}), 'qualification_required': True},
            execution_session=state, progress_callback=report)


class SDXLWorkflowScope:
    """Lazily allocate one app/worker; every call has a finite client wall deadline."""
    def __init__(self, runner):
        self.runner = runner
        self.scope_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._app_context = None
        self._compatibility = None
        self._worker_id = None
        self.events = []
        self._failed = False
        self._cpu_ready = False

    @staticmethod
    def supports(spec):
        engine = spec.bound_engine()
        return engine is not None and engine.identity().get('kind') == 'sdxl'

    def _start(self, spec):
        from pipelines_v2.runtime import modal_worker as mw
        from pipelines_v2.runtime.locks import build_locked_image
        from pipelines_v2.runtime.env import merged_runtime_env
        import modal
        resources = self.runner.resources.to_dict()
        runtime = mw._resolved_runtime_spec(spec_payload=spec.to_dict())
        compatibility = content_hash({'runtime': runtime.to_dict(), 'runner': self.runner.identity()})
        if self._app_context is not None:
            if compatibility != self._compatibility:
                raise ValueError('Runtime/resource change requires a separate workflow scope')
            return
        if (resources.get('shard_count') or 1) != 1 or (resources.get('max_containers') or 1) != 1:
            raise ValueError('SDXL workflow scope requires one worker and one shard')
        mw._validate_secret_bindings(runtime_spec=runtime, resources=resources)
        volumes = mw._mounted_volumes(store_config=self.runner.artifacts.identity(), resources=resources)
        image = build_locked_image(modal, runtime)
        env = merged_runtime_env(runtime.env, resources.get('env'))
        sources, paths = mw._resolved_local_python_sources(runtime.local_python_sources, workspace_root=self.runner.workspace_root)
        if paths:
            env['PYTHONPATH'] = ':'.join([*paths, *([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])])
        # The already prepared model volume is the Hugging Face home; no downloads occur here.
        from pathlib import PurePosixPath
        model_path = PurePosixPath(spec.engine.resolved_model_path())
        model_mounts = [v.mount_path for v in volumes if model_path.is_relative_to(PurePosixPath(v.mount_path))]
        if not model_mounts:
            raise ValueError('SDXL model cache requires its prepared volume mount')
        env.setdefault('HF_HOME', max(model_mounts, key=len) + '/hf')
        image = image.env(env)
        for local, remote in sources:
            image = image.add_local_dir(str(local), remote_path=remote)
        self.app_name = 'xenon-sdxl-session-' + self.scope_id[:12]
        app = modal.App(self.app_name)
        kwargs = dict(image=image, env=env, volumes={v.mount_path: modal.Volume.from_name(v.name, create_if_missing=v.create_if_missing) for v in volumes},
            secrets=[modal.Secret.from_name(s['name'], required_keys=list(s.get('env_vars', ()))) for s in resources.get('secrets', ())],
            serialized=True, max_containers=1, timeout=int(resources.get('timeout_seconds') or 1200))
        cpu_kwargs = {**kwargs, 'cpu': 2, 'memory': int(resources.get('merge_memory_mb') or 4096),
                      'timeout': int(resources.get('readiness_timeout_seconds') or 600)}
        @app.function(**cpu_kwargs)
        def preflight(config, store, payload):
            import os
            from pipelines_v2.operations import operation_spec_from_dict
            from pipelines_v2.storage.registry import artifact_store_from_dict
            from pipelines_v2.runtime.readiness import admit_execution
            return admit_execution(operation_spec_from_dict(payload), config, artifact_store=artifact_store_from_dict(store),
                image_identity=os.environ.get('MODAL_IMAGE_ID'), check_writer_registry=True, scope='cpu_preflight').to_dict()
        gpu_kwargs = {**kwargs, 'gpu': resources['gpu'], 'cpu': resources.get('cpu') or 8,
                      'memory': resources['memory_mb'], 'scaledown_window': 300, 'is_generator': True}
        @app.function(**gpu_kwargs)
        def stream(scope_id, mode, config, store, payload, context):
            from pipelines_v2.runtime import modal_worker as worker
            from pipelines_v2.runtime.sdxl_scope import dispatch_worker
            identity = worker._remote_call_identity()
            if identity is not None:
                yield identity
            def execute(report):
                result = dispatch_worker(scope_id, mode, config, store, payload, context, report)
                warnings = worker._commit_mounted_volumes(volumes)
                if warnings:
                    raise RuntimeError('Session artifact commit failed: ' + '; '.join(warnings))
                return result
            yield from worker._execute_with_progress_stream(execute)
        self._preflight, self._stream, self._app = preflight, stream, app
        self._compatibility = compatibility
        self._app_context = app.run()
        running_app = self._app_context.__enter__()
        self.app_id = getattr(running_app, 'app_id', None)
        self.events.append({'event':'app_started','runtime_app_id':self.app_id,'app_name':self.app_name})

    def _call(self, mode, spec, context=None, progress_callback=None):
        from pipelines_v2.runtime import modal_worker as mw
        if self._failed and mode != 'close':
            raise RuntimeError('SDXL session failed; remaining image fanout is halted')
        resources = self.runner.resources
        timeout = 30 if mode == 'close' else (resources.client_timeout_seconds or (resources.timeout_seconds or 1200) + 120)
        try:
            return mw._consume_progress_stream(mw._remote_progress_stream(self._stream, self.scope_id, mode,
                self.runner.identity(), self.runner.artifacts.identity(), spec.to_dict() if spec else None, context),
                progress_callback, timeout_seconds=timeout)
        except BaseException:
            self._failed = True
            raise

    def observe(self, spec):
        with self._lock:
            self._start(spec)
            from pipelines_v2.runtime.deadlines import bounded_remote_call
            from pipelines_v2.runtime.modal_worker import _require_readiness_checks
            if not self._cpu_ready:
                readiness = bounded_remote_call(self._preflight, self.runner.identity(), self.runner.artifacts.identity(), spec.to_dict(),
                    timeout_seconds=self.runner.resources.readiness_timeout_seconds or 600)
                _require_readiness_checks(readiness)
                self._cpu_ready = True
            result = self._call('observe', spec)
            worker = result['worker_id']
            self.events.append({**{key: result[key] for key in ('worker_id', 'model_load_count', 'gpu_allocated')},
                'event': 'observed', 'evicted_or_replaced': self._worker_id is not None and self._worker_id != worker})
            self._worker_id = worker
            return result['admission']

    def execute(self, spec, context, callback):
        with self._lock:
            from pipelines_v2.runtime.modal_worker import _workflow_context_with_runtime
            context = _workflow_context_with_runtime(context,runtime_app_id=self.app_id,runtime_app_name=self.app_name)
            result = self._call('execute', spec, context, callback)
            self.events.append({'event': 'completed', 'artifact_id': result['artifact_id'],
                                'session': result.get('metadata', {}).get('session_v1')})
            return result

    def close(self):
        with self._lock:
            if self._app_context is None:
                return
            try:
                if not self._failed:
                    self.events.append({'event': 'closed', **self._call('close', None)})
            finally:
                self._app_context.__exit__(None, None, None)
                self._app_context = None

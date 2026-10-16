from dataclasses import replace
import json

import pytest

from pipelines_v2.engine.base import PythonRuntimeSpec
from pipelines_v2.runtime.locks import RuntimeLock, load_runtime_lock, merge_runtime_specs


@pytest.fixture
def lock():
    return RuntimeLock('fixture', 'linux-x86_64', '3.13', 'numpy==2.4.2\n', ('numpy',), 'python@sha256:' + 'a' * 64)


def test_lock_round_trip_and_byte_identity(lock):
    assert RuntimeLock.from_dict(json.loads(json.dumps(lock.to_dict()))) == lock
    assert replace(lock, requirements=lock.requirements + '# different resolved lock bytes\n').identity != lock.identity
    assert lock.verify_target(system='Linux', machine='x86_64', python_version='3.13.3') == ()
    assert lock.verify_target(system='Darwin', machine='arm64', python_version='3.12.1')


@pytest.mark.parametrize('requirements', ['numpy>=2', 'numpy', '-e .', 'numpy @ https://example.org/n.whl', 'numpy==2\nnumpy==3\n'])
def test_unpinned_or_ambiguous_locks_rejected(lock, requirements):
    with pytest.raises(ValueError):
        replace(lock, requirements=requirements)


def test_runtime_round_trip_and_legacy_defaults(lock):
    legacy = PythonRuntimeSpec()
    assert 'lock' not in legacy.to_dict() and 'runtime_identity' not in legacy.to_dict()
    runtime = replace(legacy, lock=lock, runtime_identity={'image_identity': 'im-fixture'})
    assert PythonRuntimeSpec.from_dict(runtime.to_dict()) == runtime
    with pytest.raises(ValueError):
        RuntimeLock.from_dict({**lock.to_dict(), 'schema_version': 2})


def test_wrapper_and_merge_preserve_descriptors(lock, monkeypatch):
    from pipelines_v2.api import ToyEngine, CaptureSpec, GenerationRunSpec, PatchedGenerationSpec, GenerationSpec, ProjectOutPatch, ResidualInterventionSite
    runtime = PythonRuntimeSpec(lock=lock, runtime_identity={'image_identity': 'im-fixture'})
    monkeypatch.setattr(ToyEngine, 'runtime_spec', lambda self: runtime)
    for cls in (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec):
        extra = {"patch": ProjectOutPatch(subspace={}, write_site=ResidualInterventionSite(site="resid_post", layers=(0,)))} if cls is PatchedGenerationSpec else {}
        wrapped = cls(engine=ToyEngine(), generation=GenerationSpec(enabled=True, max_tokens=1), **extra).runtime_spec()
        assert wrapped.lock == lock and wrapped.runtime_identity == runtime.runtime_identity
    assert merge_runtime_specs([runtime, replace(runtime, env={'A': '1'})]).lock == lock
    for other in (PythonRuntimeSpec(), replace(runtime, lock=replace(lock, name='other')),
                  replace(runtime, runtime_identity={'image_identity': 'im-other'})):
        with pytest.raises(ValueError, match='incompatible'):
            merge_runtime_specs([runtime, other])


def test_bundled_locks_have_distinct_targets_and_stacks():
    local, core, vllm = [load_runtime_lock(name) for name in ('core-local', 'core-linux', 'vllm-linux')]
    assert local.target == 'darwin-arm64'
    assert core.target == vllm.target == 'linux-x86_64-cuda'
    assert vllm.packages['vllm'] == '0.25.1'
    assert 'vllm' not in local.packages and 'vllm' not in core.packages
    assert core.base_image.startswith('python@sha256:')


def test_modal_build_keeps_lazy_requirements_context(lock):
    from pathlib import Path
    from types import SimpleNamespace
    from pipelines_v2.runtime.locks import build_locked_image
    seen = {}
    class Image:
        @classmethod
        def from_registry(cls, value):
            seen['base'] = value
            return cls()
        def pip_install_from_requirements(self, path, **kwargs):
            seen.update(path=path, kwargs=kwargs)
            return self
    build_locked_image(SimpleNamespace(Image=Image), PythonRuntimeSpec(lock=lock))
    assert Path(seen['path']).read_text() == lock.requirements
    assert seen['kwargs'] == {'extra_options':'--no-deps'}


def test_modified_reserved_name_lock_is_not_reinterpreted():
    from pipelines_v2.runtime.locks import runtime_for_target
    builtin=load_runtime_lock('core-local')
    modified=replace(builtin,requirements=builtin.requirements+'\nexample-extra==1.2.3\n')
    actual=runtime_for_target(PythonRuntimeSpec(lock=modified),target='modal')
    assert actual.lock.identity==modified.identity
    assert actual.lock.packages['example-extra']=='1.2.3'
    assert runtime_for_target(PythonRuntimeSpec(lock=builtin),target='modal').lock==load_runtime_lock('core-linux')


def test_deployment_runtime_codec_and_older_wrappers_keep_lock(lock,monkeypatch):
    from pipelines_v2.runtime.deployments import python_runtime_spec_to_dict, python_runtime_spec_from_dict
    runtime=PythonRuntimeSpec(lock=lock,runtime_identity={'image_identity':'im-fixture'})
    assert python_runtime_spec_from_dict(python_runtime_spec_to_dict(runtime))==runtime
    from pipelines_v2.mechinterp.emotions.specs import EmotionPrecomputedVectorSpaceSpec
    from pipelines_v2.mechinterp.assistant_axis.specs import AssistantAxisPrecomputedCoordinateSpec, AssistantAxisTraitCoordinateSpec
    monkeypatch.setattr('pipelines_v2.mechinterp.emotions.specs.analysis_runtime_spec',lambda **kwargs:runtime)
    monkeypatch.setattr('pipelines_v2.mechinterp.assistant_axis.specs.analysis_runtime_spec',lambda **kwargs:runtime)
    operations=[EmotionPrecomputedVectorSpaceSpec(path='fixture.json'),
        AssistantAxisPrecomputedCoordinateSpec(model_id='fixture',filename='fixture.pt'),AssistantAxisTraitCoordinateSpec(model_id='fixture',trait='kind',filename='fixture.pt')]
    for operation in operations:
        assert operation.runtime_spec().lock==lock
        assert operation.runtime_spec().runtime_identity==runtime.runtime_identity

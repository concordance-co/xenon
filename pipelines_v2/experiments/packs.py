"""Versioned experiment metadata over ordinary executable WorkflowSpecs (C05)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pipelines_v2.core.types import SpecValidationError, to_semantic_primitive

MAX_RECORD_BYTES = 128 * 1024


def canonical(value: Any, *, limit: int = MAX_RECORD_BYTES) -> str:
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SpecValidationError("Record must contain finite JSON values") from exc
    if len(text.encode()) > limit:
        raise SpecValidationError(f"Record exceeds {limit} byte summary limit")
    return text


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecValidationError(message)


def text(value: Any, name: str) -> None:
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be nonempty text")


def sha(value: Any, name: str) -> None:
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"{name} must be a SHA-256 digest")


def fields(data: Mapping, names: str, name: str) -> None:
    require(isinstance(data, dict) and set(data) == set(names.split()), f"{name} requires exactly: {names}")


def names(values: Any, name: str, *, empty: bool = False) -> None:
    require(isinstance(values, list) and (empty or bool(values)), f"{name} must be a list")
    for value in values:
        text(value, name)
    require(len(values) == len(set(values)), f"{name} contains duplicates")


@dataclass(frozen=True)
class ExperimentPack:
    """Immutable JSON record; reading it does not import or execute workflow code."""
    _json: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> 'ExperimentPack':
        data = json.loads(canonical(payload))
        fields(data, 'schema_version study_id phase_id papers workflow source_locks model_locks data_locks question_ids hypothesis_ids presets split_policy metrics controls qualification_requirements expected_outputs', 'pack')
        require(type(data['schema_version']) is int and data['schema_version'] == 1, 'Unsupported pack schema_version')
        for key in ('study_id', 'phase_id'):
            text(data[key], key)
        path = Path(data['workflow'])
        require(not path.is_absolute() and '..' not in path.parts and path.name == 'workflow.py', 'pack must reference one relative workflow.py')
        names(data['papers'], 'papers', empty=True)
        require(all(p.startswith(('https://', 'http://')) for p in data['papers']), 'papers must be source URLs')
        for key in ('question_ids', 'hypothesis_ids', 'qualification_requirements'):
            names(data[key], key, empty=key != 'question_ids')
        for key in ('source_locks', 'model_locks', 'data_locks'):
            require(isinstance(data[key], dict) and (bool(data[key]) or key == 'model_locks'), f'{key} cannot be empty')
            for name, value in data[key].items():
                text(name, key)
                sha(value, f'{key}.{name}')
        require(data['workflow'] in data['source_locks'], 'workflow.py needs a source lock')
        for name in data['source_locks']:
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source locks must be phase-relative')
        split = data['split_policy']
        fields(split, 'unit partitions held_out digest', 'split_policy')
        text(split['unit'], 'split unit')
        names(split['partitions'], 'partitions')
        names(split['held_out'], 'held_out', empty=True)
        require(set(split['held_out']) <= set(split['partitions']), 'held_out must name declared partitions')
        sha(split['digest'], 'split digest')
        require(isinstance(data['metrics'], dict) and bool(data['metrics']), 'metrics cannot be empty')
        for metric in data['metrics'].values():
            fields(metric, 'evaluator_digest direction unit', 'metric')
            sha(metric['evaluator_digest'], 'evaluator digest')
            require(metric['direction'] in ('higher', 'lower', 'two_sided'), 'invalid metric direction')
            text(metric['unit'], 'metric unit')
        require(isinstance(data['controls'], dict), 'controls must be a mapping')
        for control in data['controls'].values():
            fields(control, 'purpose required', 'control')
            text(control['purpose'], 'control purpose')
            require(type(control['required']) is bool, 'control required must be boolean')
        require(isinstance(data['presets'], dict) and set(data['presets']) == {'setup', 'smoke', 'reference', 'scale'}, 'define setup/smoke/reference/scale separately')
        for name, preset in data['presets'].items():
            fields(preset, 'enabled workflow_hash model_bindings data_bindings limits paper_settings qualification_refs', f'preset {name}')
            require(type(preset['enabled']) is bool, 'preset enabled must be boolean')
            sha(preset['workflow_hash'], 'workflow hash')
            for bindings, registry in (('model_bindings', 'model_locks'), ('data_bindings', 'data_locks')):
                require(isinstance(preset[bindings], dict), f'{bindings} must map workflow paths to named locks')
                for path, lock_name in preset[bindings].items():
                    text(path, bindings)
                    require(isinstance(lock_name, str) and lock_name in data[registry], f'Unknown {registry} binding: {lock_name}')
            fields(preset['limits'], 'max_examples max_workers timeout_seconds max_gpu_hours', 'limits')
            for key in ('max_examples', 'max_workers', 'timeout_seconds'):
                value = preset['limits'][key]
                require(type(value) is int and value > 0, f'{key} must be a finite positive integer')
            gpu = preset['limits']['max_gpu_hours']
            require(type(gpu) in (int, float) and math.isfinite(gpu) and gpu >= 0, 'max_gpu_hours must be finite and nonnegative')
            require(isinstance(preset['paper_settings'], dict) and bool(preset['paper_settings']), 'state matched paper settings and deviations (or fixture-only reason)')
            names(preset['qualification_refs'], 'qualification refs', empty=True)
        require(isinstance(data['expected_outputs'], dict) and bool(data['expected_outputs']), 'expected outputs cannot be empty')
        for key, value in data['expected_outputs'].items():
            text(key, 'output step'); text(value, 'output kind')
        return cls(canonical(data))

    @classmethod
    def load(cls, path: str | Path) -> 'ExperimentPack':
        path = Path(path)
        require(path.stat().st_size <= MAX_RECORD_BYTES, 'pack exceeds summary limit')
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._json)

    @property
    def identity(self) -> str:
        return hashlib.sha256(self._json.encode()).hexdigest()

    def validate_workflow(self, workflow: Any, *, phase_dir: str | Path, preset: str | None = None,
                          qualification_check: Callable[[str, dict], bool] | None = None) -> str:
        """Fail before launch. Qualification verifier must consult actual C02 records.

        The pack's hash binds every operation and dataset in the built workflow.
        Resource limits remain declarations consumed by admission, not measurements.
        """
        from pipelines_v2.operations.registry import operation_registration
        data = self.to_dict()
        chosen = preset or os.environ.get('XENON_EXPERIMENT_PRESET', 'smoke')
        require(chosen in data['presets'], f'Unknown experiment preset: {chosen}')
        profile = data['presets'][chosen]
        require(profile['enabled'], f'Experiment preset {chosen} is disabled')
        root = Path(phase_dir).resolve()
        for name, expected in data['source_locks'].items():
            path = (root / name).resolve()
            require(path.is_relative_to(root) and path.is_file(), f'Missing/escaping source lock: {name}')
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Source lock mismatch: {name}')
        require(workflow.semantic_hash() == profile['workflow_hash'], 'Built workflow does not match preset workflow_hash')
        model_locks, data_locks = workflow_locks(workflow)
        require({path: data['model_locks'][name] for path, name in profile['model_bindings'].items()} == model_locks, 'Model configuration locks do not match built workflow')
        require({path: data['data_locks'][name] for path, name in profile['data_bindings'].items()} == data_locks, 'Data locks do not match built workflow')
        actual = {}
        for step in workflow.steps:
            if step.spec.kind == 'capture':
                actual[step.name] = 'capture'
            else:
                registration = operation_registration(step.spec.kind)
                require(registration is not None, f'Unregistered operation kind: {step.spec.kind}')
                actual[step.name] = registration.artifact_kind
        require(actual == data['expected_outputs'], 'Expected output steps/kinds do not match built workflow')
        for step in workflow.steps:
            dataset = getattr(step.spec, 'dataset', None)
            if dataset is not None and hasattr(dataset, 'examples') and dataset.examples is not None:
                require(len(dataset.examples) <= profile['limits']['max_examples'], 'Workflow exceeds max_examples')
        if chosen == 'scale':
            require(bool(data['qualification_requirements']) and bool(profile['qualification_refs']), 'Scale requires upstream qualification')
            require(qualification_check is not None, 'Scale requires a measured qualification verifier')
            require(all(qualification_check(ref, data) is True for ref in profile['qualification_refs']), 'Scale qualification is missing, failed or stale')
        return chosen


def workflow_locks(workflow: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Bind declared model configs and dataset recipes, without fetching data.

    Actual resolved model/data identities remain C01 producer provenance. These
    recipe locks alone cannot authorize reuse or measured qualification.
    """
    models, datasets = {}, {}
    def visit(value, path):
        if isinstance(value, dict):
            if value.get('kind') == 'dataset':
                datasets[path] = digest(value)
            else:
                for key, child in value.items():
                    visit(child, f'{path}/{key}')
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f'{path}/{index}')
    for step in workflow.steps:
        engine = getattr(step.spec, 'engine', None)
        if engine is not None:
            models[step.name] = digest(to_semantic_primitive(engine))
        visit(step.spec.semantic_dict(), step.name)
    return models, datasets

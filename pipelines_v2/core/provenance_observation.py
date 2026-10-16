"""Narrow, offline observation seam for supported scientific operations.

Observers inspect actual bytes/process facts. They do not certify a GPU envelope,
load model weights, perform network I/O, or grant permission to bypass C03.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OperationFacts:
    # Only named top-level spec fields may be replaced. Artifact and dataset
    # fields cannot be replaced: their identity is always observed by core.
    field_identities: Mapping[str, Any] = field(default_factory=dict)
    model_components: tuple[dict[str, Any], ...] | None = None
    numerics: Mapping[str, Any] | None = None
    runtime: Mapping[str, Any] | None = None
    methods: Mapping[str, Any] = field(default_factory=dict)
    seeds: Mapping[str, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()


def file_set_identity(root: str | Path, names, *, expected: Mapping[str, str] | None = None) -> dict:
    """Freshly hash an explicit file list; physical paths never enter identity.

    Symlinked HF cache files are content addressed by their observed bytes. No
    directory walk, mtime-only cache, or implicit asset preparation is performed.
    """
    from .provenance import content_hash, _SECRET_NAMES
    base = Path(root).expanduser().resolve(strict=True)
    selected = sorted(set(names))
    if not selected:
        raise ValueError('A file observation requires an explicit nonempty file set')
    files = []
    for name in selected:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or any(
            part in _SECRET_NAMES or part.startswith('.env.') for part in relative.parts
        ):
            raise ValueError('Invalid or credential-bearing file identity input')
        path = base / relative
        before = path.stat()
        sha = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError('Input changed while its content identity was observed')
        digest = sha.hexdigest()
        if expected is not None and expected.get(name) != digest:
            raise ValueError('Observed file content differs from the requested immutable identity')
        files.append({'name': relative.as_posix(), 'bytes': after.st_size, 'sha256': digest})
    return {'content_digest': content_hash(files), 'files': files}


_OBSERVERS = {
    'external_mattr': 'pipelines_v2.operations.attribution.provenance',
    **{kind: 'pipelines_v2.representation.sae.provenance' for kind in (
        'sae_dictionary_import', 'sae_encode', 'sae_decode', 'sae_feature_splice')},
    **{kind: 'pipelines_v2.representation.crosscoder.provenance' for kind in (
        'crosscoder_pair_capture', 'crosscoder_fit', 'crosscoder_import',
        'crosscoder_encode', 'crosscoder_decode', 'crosscoder_feature_splice')},
}


def observe_operation(spec, runner_identity) -> OperationFacts | None:
    module = _OBSERVERS.get(spec.kind)
    if module is not None:
        try:
            result = importlib.import_module(module).provenance_facts(spec, runner_identity)
        except (ImportError, OSError, ValueError, KeyError):
            # No exception text is serialized: a path/credential might be in it.
            return OperationFacts(unresolved=('operation.observation_unavailable',))
        if not isinstance(result, OperationFacts):
            raise TypeError('Supported provenance observer must return OperationFacts')
        return result
    if spec.kind in {'transform', 'report'}:
        from .provenance_summaries import summary_facts
        return summary_facts(spec)
    return None

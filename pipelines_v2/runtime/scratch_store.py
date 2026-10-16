"""Synchronous artifact-write guard, independent of process polling."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any

from pipelines_v2.storage.local import LocalArtifactStore
from pipelines_v2.storage.json import json_default
from .scratch_supervision import ScratchLimitError, tree_bytes


@dataclass(frozen=True, slots=True)
class BudgetedLocalArtifactStore(LocalArtifactStore):
    max_new_bytes: int = 256 * 1024 * 1024
    baseline_bytes: int = 0
    _write_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    def _write_chunks(self, artifact_id, relative_path, chunks, format):
        path = self._resolve_inside_artifact(artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        digest = hashlib.sha256()
        with self._write_lock:
            available = self.max_new_bytes - (tree_bytes(Path(self.root)) - self.baseline_bytes)
            written = 0
            try:
                with tmp.open('xb') as stream:
                    for chunk in chunks:
                        if written + len(chunk) > available:
                            raise ScratchLimitError('Scratch artifact-byte limit exceeded while writing')
                        stream.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(tmp, path)
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                tmp.unlink(missing_ok=True)
        return {'store':'local','path':str(path),'format':format,'bytes':written,'sha256':digest.hexdigest()}

    def write_json(self, artifact_id: str, relative_path: str, payload: Any):
        encoder = json.JSONEncoder(sort_keys=True, indent=2, default=json_default)
        return self._write_chunks(artifact_id, relative_path,
                                  (chunk.encode('utf-8') for chunk in encoder.iterencode(payload)), 'json')

    def write_safetensors(self, artifact_id: str, relative_path: str, tensors: dict[str, Any]):
        from safetensors.numpy import save
        if sum(value.nbytes for value in tensors.values()) > self.max_new_bytes:
            raise ScratchLimitError('Scratch tensor payload exceeds artifact-byte limit before serialization')
        return self._write_chunks(artifact_id, relative_path, (save(tensors),), 'safetensors')

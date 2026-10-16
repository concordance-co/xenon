"""C03 durable journals and final completion. Payload integrity is independent of C01."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from pipelines_v2.core.provenance import content_hash
from pipelines_v2.storage.artifacts import ArtifactManifest

COMPLETION_KEY = "completion_v1"


class IntegrityError(ValueError):
    """Stored progress cannot establish safe completion or continuation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_replace(path: Path, writer: Any) -> None:
    """Unique temporary files and fsync; no shared .tmp name between writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def payload_paths(refs: Any) -> list[str]:
    paths: set[str] = set()
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("path"), str):
                paths.add(value["path"])
                if value.get("tensor_path"):
                    paths.add(str(value["tensor_path"]))
            else:
                for key, item in value.items():
                    if key != "manifest":
                        visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
    visit(refs)
    return sorted(paths)


def describe_payloads(refs: Any) -> list[dict[str, Any]]:
    result = []
    for raw in payload_paths(refs):
        path = Path(raw)
        if not path.is_file():
            raise IntegrityError(f"Missing payload: {raw}")
        result.append({"path": raw, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def verify_payloads(payloads: list[dict[str, Any]]) -> None:
    if len({item["path"] for item in payloads}) != len(payloads):
        raise IntegrityError("Duplicate payload references")
    for item in payloads:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise IntegrityError(f"Payload checksum/size mismatch: {path}")


def commit_visibility(store: Any) -> None:
    commit = getattr(store, "commit_visibility", None)
    if callable(commit):
        commit()  # Failure propagates; never publish/catalog an uncommitted artifact.


@contextlib.contextmanager
def local_writer_lock(store: Any, key: str):
    """Process-safe lock released by OS on kill; Modal supplies its own lease."""
    import fcntl
    root = Path(store.root) / ".writers"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{content_hash(key)}.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IntegrityError("Artifact already has an active writer") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def writer_lock(store: Any, key: str):
    method = getattr(store, "writer_lock", None)
    return method(key) if callable(method) else local_writer_lock(store, key)


def finalize_manifest(store: Any, manifest: ArtifactManifest) -> ArtifactManifest:
    """Payload commit -> complete manifest -> manifest commit; catalog follows caller."""
    commit_visibility(store)
    with writer_lock(store, "final:" + manifest.artifact_id):
        return _finalize_manifest_owned(store, manifest)


def _finalize_manifest_owned(store: Any, manifest: ArtifactManifest) -> ArtifactManifest:
    payloads = describe_payloads(manifest.storage_refs)
    record = {"schema_version": 1, "state": "complete", "expected_coverage": manifest.example_coverage,
              "payloads": payloads, "operation_semantic_hash": manifest.operation_semantic_hash,
              "provenance_hash": manifest.provenance().scientific_identity if manifest.provenance() else None}
    record["integrity_hash"] = content_hash(record)
    complete = replace(manifest, metadata={**manifest.metadata, COMPLETION_KEY: record})
    commit_visibility(store)
    path = Path(store.root) / manifest.artifact_id / "manifest.json"
    if path.exists():
        prior = ArtifactManifest.from_dict(json.loads(path.read_text()))
        validate_completion(prior, store=store)
        if prior.metadata[COMPLETION_KEY] != record:
            raise IntegrityError("Conflicting finalization for the same artifact")
        prior.storage_refs["manifest"] = {"store": store.kind, "path": str(path), "format": "json", "bytes": path.stat().st_size}
        return prior
    ref = store.write_json(manifest.artifact_id, "manifest.json", complete.to_dict())
    commit_visibility(store)
    complete.storage_refs["manifest"] = ref
    return complete


def validate_completion(manifest: ArtifactManifest, *, store: Any, legacy_policy: str = "reject") -> bool:
    record = manifest.metadata.get(COMPLETION_KEY)
    if record is None:
        if legacy_policy == "read_only":
            return False
        raise IntegrityError("Legacy artifact has no verified C03 completion")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("state") != "complete":
        raise IntegrityError("Unsupported or incomplete completion record")
    body = {key: value for key, value in record.items() if key != "integrity_hash"}
    if record.get("integrity_hash") != content_hash(body):
        raise IntegrityError("Completion record integrity mismatch")
    identity = manifest.provenance().scientific_identity if manifest.provenance() else None
    if record["provenance_hash"] != identity or record["operation_semantic_hash"] != manifest.operation_semantic_hash:
        raise IntegrityError("Completion identity mismatch")
    if record["expected_coverage"] != manifest.example_coverage:
        raise IntegrityError("Completion coverage mismatch")
    if payload_paths(manifest.storage_refs) != sorted(item["path"] for item in record["payloads"]):
        raise IntegrityError("Final manifest does not cover its payloads")
    remote_verify = getattr(store, "verify_payload_integrity", None)
    if callable(remote_verify) and any(not Path(item["path"]).is_file() for item in record["payloads"]):
        remote_verify(record["payloads"])
    else:
        verify_payloads(record["payloads"])
    return True


class CheckpointJournal:
    """Independent, immutable units. Call under writer_lock for the whole execution."""
    def __init__(self, *, store: Any, artifact_id: str, semantic_hash: str, provenance: Any,
                 expected: dict[str, Any], shard: Any = None):
        if len(set(expected["unit_keys"])) != len(expected["unit_keys"]):
            raise IntegrityError("Duplicate planned checkpoint units")
        self.store, self.artifact_id = store, artifact_id
        self.attempt_id = uuid.uuid4().hex
        self.path = Path(store.root) / artifact_id / "journal.json"
        self.body = {"schema_version": 1, "artifact_id": artifact_id, "attempt_id": self.attempt_id,
                     "operation_semantic_hash": semantic_hash, "provenance_hash": provenance.scientific_identity,
                     "expected_coverage": expected, "shard": shard, "completed_units": [],
                     "checkpoint_refs": {}, "state": "incomplete", "resume_semantics": "restart_at_unit"}
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old.get("journal_hash") != content_hash({k: v for k, v in old.items() if k != "journal_hash"}):
                raise IntegrityError("Checkpoint journal integrity mismatch")
            if not provenance.identity_complete:
                raise IntegrityError("Unresolved C01 identity cannot authorize checkpoint reuse")
            for key in ("schema_version", "artifact_id", "operation_semantic_hash", "provenance_hash", "expected_coverage", "shard", "state", "resume_semantics"):
                if old.get(key) != self.body[key]:
                    raise IntegrityError(f"Checkpoint {key} mismatch")
            if type(old.get("schema_version")) is not int:
                raise IntegrityError("Checkpoint schema_version must be an integer")
            units = old.get("completed_units", [])
            if any(unit["key"] not in expected["unit_keys"] for unit in units):
                raise IntegrityError("Unexpected stored checkpoint unit")
            if len({unit["key"] for unit in units}) != len(units):
                raise IntegrityError("Duplicate completed checkpoint units")
            for unit in units:
                verify_payloads(unit["payloads"])
            verify_payloads(old.get("checkpoint_refs", {}).get("payloads", []))
            self.body = old
        else:
            self._publish()

    def _publish(self) -> None:
        self.body["journal_hash"] = content_hash({k: v for k, v in self.body.items() if k != "journal_hash"})
        self.store.write_json(self.artifact_id, "journal.json", self.body)
        commit_visibility(self.store)

    def unit(self, key: str) -> dict[str, Any] | None:
        return next((unit for unit in self.body["completed_units"] if unit["key"] == key), None)

    def append(self, key: str, *, refs: dict[str, Any], coverage: Any) -> None:
        if key not in self.body["expected_coverage"]["unit_keys"]:
            raise IntegrityError("Unexpected checkpoint unit")
        unit = {"key": key, "shard": self.body["shard"], "coverage": coverage,
                "refs": refs, "payloads": describe_payloads(refs)}
        existing = self.unit(key)
        if existing is not None:
            if existing != unit:
                raise IntegrityError("Conflicting duplicate checkpoint unit")
            return
        commit_visibility(self.store)
        self.body["completed_units"].append(unit)
        self._publish()

    def set_checkpoint(self, refs: dict[str, Any], *, resume_semantics: str = "restart_at_unit") -> None:
        if resume_semantics != "restart_at_unit":
            raise IntegrityError("Exact optimizer/RNG continuation requires an executed adapter oracle")
        self.body["checkpoint_refs"] = {"refs": refs, "payloads": describe_payloads(refs)}
        commit_visibility(self.store)
        self._publish()

    def verify_full(self) -> None:
        if {unit["key"] for unit in self.body["completed_units"]} != set(self.body["expected_coverage"]["unit_keys"]):
            raise IntegrityError("Checkpoint coverage is incomplete")
        for unit in self.body["completed_units"]:
            verify_payloads(unit["payloads"])

"""Restart-at-unit capture/intervention recovery; no cross-device RNG replay claim."""
from __future__ import annotations

from dataclasses import replace
from typing import Any
from pathlib import PurePosixPath

from pipelines_v2.core.provenance import content_hash
from pipelines_v2.core.types import to_primitive
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.engine.base import EngineCaptureResult, EngineInterventionResult
from pipelines_v2.storage.features import load_feature_payload, merge_feature_payloads, write_capture_features
from pipelines_v2.storage.finalization import CheckpointJournal, IntegrityError


class _UnitStore:
    def __init__(self, store: Any, prefix: str):
        self.store, self.prefix = store, prefix

    def _check(self, path):
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise IntegrityError("Checkpoint payload escaped its immutable unit namespace")

    def write_json(self, artifact_id, path, value):
        self._check(path)
        return self.store.write_json(artifact_id, f"{self.prefix}/{path}", value)

    def write_safetensors(self, artifact_id, path, value):
        self._check(path)
        return self.store.write_safetensors(artifact_id, f"{self.prefix}/{path}", value)


def _unique(values, label):
    values = list(values)
    if len(set(values)) != len(values):
        raise IntegrityError(f"Duplicate {label}")
    return values


def capture_coverage(spec: Any, result: Any) -> list[dict[str, Any]]:
    """Validate named sites/layers/examples and explicit selected-token coverage."""
    keys = _unique((str(row.key) for row in spec.dataset.examples), "capture example keys")
    names = _unique((site.name for site in spec.sites), "capture site names")
    if set(result.features) != set(names):
        raise IntegrityError("Missing or unexpected capture site coverage")
    metadata = {str(row["example_key"]): row for row in result.metadata.get("example_metadata", [])}
    coverage = []
    for site in spec.sites:
        payload = result.features[site.name]
        layers = payload.get("layers")
        if not isinstance(layers, dict):
            # New codecs must supply a verifier rather than imply complete tensors.
            validator = _CAPTURE_VALIDATORS.get(str(payload.get("kind")))
            if validator is None:
                raise IntegrityError(f"Capture coverage validator unavailable for {payload.get('kind')!r}")
            coverage.extend(validator(spec, site, payload, result.metadata))
            continue
        expected_layers = {str(layer) for layer in site.layers}
        if set(layers) != expected_layers:
            raise IntegrityError("Missing or unexpected capture layer coverage")
        for layer, records in layers.items():
            if set(records) != set(keys):
                raise IntegrityError("Missing or unexpected capture example coverage")
            for key, record in records.items():
                tokens = _unique(record.get("tokens", ()), "capture tokens")
                if any(type(token) is not int or token < 0 for token in tokens):
                    raise IntegrityError("Invalid capture token coordinates")
                if payload.get("kind") == "residual" and "values" not in record:
                    raise IntegrityError("Capture omitted residual values")
                if payload.get("kind") == "moe_routing":
                    if not isinstance(record.get("records"), dict):
                        raise IntegrityError("Capture omitted router records")
                    required = {item.kind for item in site.record}
                    if any(not required <= set(item) for item in record["records"].values()):
                        raise IntegrityError("Capture omitted requested router record family")
                if "values" in record and len(record["values"]) != len(tokens):
                    raise IntegrityError("Capture values/token coverage mismatch")
                if "records" in record and {str(token) for token in tokens} != set(record["records"]):
                    raise IntegrityError("Capture router/token coverage mismatch")
                count = metadata.get(key, {}).get("prompt_token_count" if payload.get("kind") == "moe_routing" else "token_count")
                if count is None:
                    count = getattr(spec.engine, "sequence_length", None)
                    if count is not None and spec.generation.capture_generated_tokens and payload.get("kind") != "moe_routing":
                        count += int(spec.generation.max_tokens or 0)
                if getattr(site, "pooling", None) is not None:
                    expected_tokens = [0]
                elif site.tokens.kind == "section":
                    # Section coordinates are adapter-defined. Require explicit attestation.
                    expected_tokens = record.get("selected_token_positions")
                    if expected_tokens is None:
                        raise IntegrityError("Capture adapter omitted planned section coverage")
                elif count is not None:
                    expected_tokens = site.tokens.resolve(int(count))
                else:
                    expected_tokens = record.get("selected_token_positions")
                    if expected_tokens is None:
                        raise IntegrityError("Capture adapter omitted planned token coverage")
                if tokens != list(expected_tokens):
                    raise IntegrityError("Missing or unexpected selected capture tokens")
                coverage.append({"example_key": key, "site": site.name, "layer": layer, "tokens": tokens})
    if spec.generation.enabled:
        generated = _unique((str(row["example_key"]) for row in result.generations), "capture generations")
        if set(generated) != set(keys):
            raise IntegrityError("Missing capture generation coverage")
    return coverage


_CAPTURE_VALIDATORS: dict[str, Any] = {}


def register_capture_coverage_validator(kind: str, validator: Any) -> None:
    if kind in _CAPTURE_VALIDATORS or not callable(validator):
        raise ValueError("Capture coverage validator must be callable and uniquely registered")
    _CAPTURE_VALIDATORS[kind] = validator


def capture_checkpointed(*, spec, store, artifact_id, provenance, invoke, shard=None):
    examples = list(spec.dataset.examples)
    _unique((row.key for row in examples), "input example keys")
    size = max(1, int(getattr(spec.engine, "max_num_seqs", 0) or 1))
    batches = [examples[index:index + size] for index in range(0, len(examples), size)]
    unit_keys = [content_hash([row.key for row in batch]) for batch in batches]
    expected = {"unit_keys": unit_keys, "example_keys": [row.key for row in examples],
                "sites": [to_primitive(site) for site in spec.sites], "explicit_empty": not examples}
    journal = CheckpointJournal(store=store, artifact_id=artifact_id, semantic_hash=spec.semantic_hash(),
                                provenance=provenance, expected=expected, shard=shard)
    features, generations, metadata, rows_metadata = {}, [], {}, []
    for key, batch in zip(unit_keys, batches, strict=True):
        unit_spec = replace(spec, dataset=Dataset.from_examples(batch, id=spec.dataset.id, name=spec.dataset.name))
        unit = journal.unit(key)
        if unit is None:
            result = invoke(unit_spec)
            coverage = capture_coverage(unit_spec, result)
            scoped = _UnitStore(store, f"checkpoints/{journal.attempt_id}/{key}")
            refs = {"features": write_capture_features(scoped, artifact_id, result.features),
                    "rows": scoped.write_json(artifact_id, "rows.json", {"generations": result.generations, "metadata": result.metadata})}
            journal.append(key, refs=refs, coverage=coverage)
            unit = journal.unit(key)
        rows = store.read_json_ref(unit["refs"]["rows"])
        result = EngineCaptureResult(features={name: load_feature_payload(store, ref) for name, ref in unit["refs"]["features"].items()},
                                     generations=rows["generations"], metadata=rows["metadata"])
        if capture_coverage(unit_spec, result) != unit["coverage"]:
            raise IntegrityError("Checkpoint capture coverage was altered")
        for name, payload in result.features.items():
            features[name] = merge_feature_payloads(features[name], payload) if name in features else payload
        generations.extend(result.generations)
        metadata.update(result.metadata)
        rows_metadata.extend(result.metadata.get("example_metadata", []))
    journal.verify_full()
    if rows_metadata:
        metadata["example_metadata"] = rows_metadata
    metadata["checkpoint_journal"] = str(journal.path)
    metadata["resume_semantics"] = "restart_at_unit"
    return EngineCaptureResult(features=features, generations=generations, metadata=metadata)


def intervention_checkpointed(*, spec, store, artifact_id, provenance, invoke, shard=None):
    from pipelines_v2.operations.interventions.runtime import resolve_patched_generation_targets
    examples = list(spec.dataset.examples)
    _unique((row.key for row in examples), "input example keys")
    paired = spec.patch.requires_pairing()
    groups = {}
    if paired:
        cases = spec.pair_by.resolve_values()
        for row in examples:
            if cases.get(row.key) is not None:
                groups.setdefault(str(cases[row.key]), []).append(row)
    elif examples:
        groups = {str(row.key): [row] for row in resolve_patched_generation_targets(spec)}
    group_keys = sorted(groups) if paired else list(groups)
    size = max(1, int(getattr(spec.engine, "max_num_seqs", 0) or 1))
    batches = [group_keys[index:index + size] for index in range(0, len(group_keys), size)]
    keys = [content_hash(batch) for batch in batches]
    row_key = "case_key" if paired else "example_key"
    expected = {"unit_keys": keys, "row_key": row_key, "row_keys": group_keys, "explicit_empty": not keys,
                "example_keys": [row.key for row in examples], "write_site": to_primitive(spec.patch.write_site),
                "target_tokens": to_primitive(spec.patch.target_tokens)}
    journal = CheckpointJournal(store=store, artifact_id=artifact_id, semantic_hash=spec.semantic_hash(),
                                provenance=provenance, expected=expected, shard=shard)
    rows, metadata = [], {}
    for key, batch_keys in zip(keys, batches, strict=True):
        unit = journal.unit(key)
        if unit is None:
            batch = [row for group_key in batch_keys for row in groups[group_key]]
            unit_spec = replace(spec, dataset=Dataset.from_examples(batch, id=spec.dataset.id, name=spec.dataset.name))
            result = invoke(unit_spec)
            for row in result.rows:
                if row.get("status", "ok") == "skipped":
                    if not row.get("skip_reason"):
                        raise IntegrityError("Skipped intervention omitted its reason")
                elif row.get("status", "ok") == "ok":
                    tokens = row.get("target_tokens")
                    if not isinstance(tokens, list) or not tokens or len(set(tokens)) != len(tokens) or any(type(token) is not int or token < 0 for token in tokens):
                        raise IntegrityError("Intervention omitted valid target-token coverage")
                else:
                    raise IntegrityError("Unknown intervention completion status")
            actual = [str(row.get(row_key)) for row in result.rows]
            if len(actual) != len(set(actual)) or set(actual) != set(batch_keys):
                raise IntegrityError("Missing, duplicate or unexpected intervention coverage")
            ref = store.write_json(artifact_id, f"checkpoints/{journal.attempt_id}/{key}.json",
                                   {"rows": result.rows, "metadata": result.metadata})
            journal.append(key, refs={"result": ref}, coverage=[{row_key: str(row.get(row_key)),
                "target_tokens": row.get("target_tokens"), "status": row.get("status", "ok")} for row in result.rows])
            unit = journal.unit(key)
        payload = store.read_json_ref(unit["refs"]["result"])
        by_key = {str(row.get(row_key)): row for row in payload["rows"]}
        if len(by_key) != len(payload["rows"]) or set(by_key) != set(batch_keys):
            raise IntegrityError("Intervention checkpoint row identity mismatch")
        rows.extend(by_key[value] for value in batch_keys)
        metadata.update(payload["metadata"])
    journal.verify_full()
    metadata.update(checkpoint_journal=str(journal.path), resume_semantics="restart_at_unit")
    patched = sum(row.get("status", "ok") == "ok" for row in rows)
    return EngineInterventionResult(rows=rows, metadata=metadata, summary={"example_count": len(rows),
        "patched_count": patched, "skipped_count": len(rows) - patched, "case_count" if paired else "target_count": len(rows)})

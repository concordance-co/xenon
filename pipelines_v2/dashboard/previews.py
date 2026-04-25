"""Dataset + label preview helpers.

Reconstructs the dataset belonging to (or referenced by) a workflow step from
its persisted workflow payload, samples a small number of rows, and shapes the
response for the dashboard. Never localizes capture tensors — only uses the
Dataset / Source abstractions.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from threading import Lock
from typing import Any, Mapping

from pipelines_v2.dashboard.models import (
    DatasetPreview,
    DatasetPreviewRow,
    DatasetSourceInfo,
    LabelDistribution,
    LabelDistributionBucket,
    LabelPreview,
)
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.storage.artifacts import ArtifactManifest, artifact_from_manifest
from pipelines_v2.storage.local import LocalArtifactStore
from pipelines_v2.storage.modal import ModalVolumeStore
from pipelines_v2.workflow.records import WorkflowRunRecord
from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

DEFAULT_SAMPLE_SIZE = 5
# Hard ceiling to keep the response bounded for pathological multi-MB prompts.
# The frontend handles display truncation + click-to-expand, so this is only
# a backstop, not a normal-case truncation point.
PROMPT_PREVIEW_CHARS = 200_000

# Memoize deferred Dataset.resolve() results by (dataset.id, sample_size).
# The overview page fires N dataset-preview calls, one per step — if 20 analysis
# steps share a single upstream capture, all 20 hit the same (id, size) entry
# after the first resolve(). Materialized datasets never enter this cache;
# they're already in memory.
_RESOLVED_TTL_SECONDS = 300.0
_RESOLVED_CACHE: dict[tuple[str, int], tuple[float, Dataset]] = {}
_RESOLVED_CACHE_LOCK = Lock()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_dataset_preview(
    *,
    run: WorkflowRunRecord,
    target_step: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    source_step: str | None = None,
    artifact_manifests_by_step: Mapping[str, ArtifactManifest] | None = None,
    local_cache_root: Path | None = None,
) -> DatasetPreview:
    """Shape the dataset preview response for one workflow step."""
    try:
        spec = WorkflowSpec.from_dict(run.workflow_payload)
    except Exception as exc:
        return DatasetPreview(
            available=False,
            reason=f"workflow_payload could not be rehydrated: {exc}",
            rows=[],
        )

    target = _find_step(spec, target_step)
    if target is None:
        return DatasetPreview(available=False, reason=f"unknown step: {target_step}", rows=[])

    origin_step, dataset, alternates = _dataset_for_step(spec, target, source_step)
    if dataset is None:
        if alternates:
            return DatasetPreview(
                available=False,
                reason="multiple upstream capture datasets; select one",
                rows=[],
                dataset_options=[
                    {"step_name": s.name, "label": _describe_step(s)} for s in alternates
                ],
            )
        return DatasetPreview(
            available=False,
            reason="no dataset is reachable from this step",
            rows=[],
        )

    try:
        sampled, total, materialized = _materialize_sample(
            dataset,
            sample_size,
            artifact_manifests_by_step=artifact_manifests_by_step,
            local_cache_root=local_cache_root,
        )
    except EnvVarMissing as exc:
        return DatasetPreview(
            available=False,
            reason=str(exc),
            rows=[],
            source=_describe_source(dataset),
            resolved_from_step=origin_step.name if origin_step else None,
        )
    except Exception as exc:
        return DatasetPreview(
            available=False,
            reason=f"dataset resolution failed: {exc}",
            rows=[],
            source=_describe_source(dataset),
            resolved_from_step=origin_step.name if origin_step else None,
        )

    rows = [_example_row(ex) for ex in sampled.examples]
    return DatasetPreview(
        available=True,
        rows=rows,
        source=_describe_source(materialized or dataset),
        total_rows=total,
        sample_size=sample_size,
        resolved_from_step=origin_step.name if origin_step and origin_step.name != target_step else None,
    )


def build_label_preview(
    *,
    run: WorkflowRunRecord,
    target_step: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    source_step: str | None = None,
    artifact_manifests_by_step: Mapping[str, ArtifactManifest] | None = None,
    local_cache_root: Path | None = None,
) -> LabelPreview:
    """Shape the label preview response for one workflow step."""
    try:
        spec = WorkflowSpec.from_dict(run.workflow_payload)
    except Exception as exc:
        return LabelPreview(available=False, reason=f"workflow_payload rehydration failed: {exc}", labels=[], samples=[])

    target = _find_step(spec, target_step)
    if target is None:
        return LabelPreview(available=False, reason=f"unknown step: {target_step}", labels=[], samples=[])

    origin_step, dataset, _ = _dataset_for_step(spec, target, source_step)
    if dataset is None:
        return LabelPreview(
            available=False,
            reason="no dataset reachable from this step",
            labels=[],
            samples=[],
        )

    try:
        sampled, _, _ = _materialize_sample(
            dataset,
            sample_size,
            artifact_manifests_by_step=artifact_manifests_by_step,
            local_cache_root=local_cache_root,
        )
    except EnvVarMissing as exc:
        return LabelPreview(available=False, reason=str(exc), labels=[], samples=[])
    except Exception as exc:
        return LabelPreview(
            available=False,
            reason=f"dataset resolution failed: {exc}",
            labels=[],
            samples=[],
        )

    distributions = _tally_labels(
        sampled.examples,
        source_step=origin_step.name if origin_step else None,
    )
    samples = [
        {
            "example_key": ex.key,
            "prompt_preview": _prompt_preview(ex.prompt),
            "labels": dict(ex.labels),
        }
        for ex in sampled.examples[:10]
    ]
    return LabelPreview(
        available=True,
        labels=distributions,
        samples=samples,
        resolved_from_step=origin_step.name if origin_step and origin_step.name != target_step else None,
    )


# ---------------------------------------------------------------------------
# Dataset resolution against the workflow spec
# ---------------------------------------------------------------------------


class EnvVarMissing(RuntimeError):
    """Raised when a deferred source needs an env var that isn't present locally."""


def _find_step(spec: WorkflowSpec, name: str) -> WorkflowStep | None:
    for step in spec.steps:
        if step.name == name:
            return step
    return None


def _dataset_for_step(
    spec: WorkflowSpec,
    target: WorkflowStep,
    source_step: str | None,
) -> tuple[WorkflowStep | None, Dataset | None, list[WorkflowStep]]:
    """Find the dataset this step owns or inherits.

    Returns (origin_step, dataset, alternates). If the step owns a dataset
    directly, origin_step is the step itself. If it's an analysis step, walk
    upstream dependencies until hitting capture-family steps that own datasets.
    If multiple distinct captures are reachable, return them in `alternates`.
    """
    # Direct ownership (capture etc.).
    own = _step_dataset(target)
    if own is not None:
        return target, own, []

    # If the caller explicitly pinned a source step, use it.
    if source_step is not None:
        step = _find_step(spec, source_step)
        if step is not None:
            ds = _step_dataset(step)
            if ds is not None:
                return step, ds, []

    # Walk upstream. Build parent map first.
    parents: dict[str, tuple[str, ...]] = {s.name: s.resolved_depends_on() for s in spec.steps}
    by_name: dict[str, WorkflowStep] = {s.name: s for s in spec.steps}

    reachable_sources: dict[str, WorkflowStep] = {}
    stack = [target.name]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        step = by_name.get(node)
        if step is None:
            continue
        if step.name != target.name:
            ds = _step_dataset(step)
            if ds is not None:
                reachable_sources[step.name] = step
                continue  # don't traverse past a dataset-owning ancestor
        stack.extend(parents.get(node, ()))

    if not reachable_sources:
        return None, None, []
    if len(reachable_sources) == 1:
        origin = next(iter(reachable_sources.values()))
        return origin, _step_dataset(origin), []
    # Multiple ancestors own datasets — require the caller to disambiguate.
    return None, None, sorted(reachable_sources.values(), key=lambda s: s.name)


def _step_dataset(step: WorkflowStep) -> Dataset | None:
    ds = getattr(step.spec, "dataset", None)
    if isinstance(ds, Dataset):
        return ds
    return None


def _describe_step(step: WorkflowStep) -> str:
    kind = getattr(step.spec, "kind", "step")
    return f"{step.name} ({kind})"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _materialize_sample(
    dataset: Dataset,
    sample_size: int,
    *,
    artifact_manifests_by_step: Mapping[str, ArtifactManifest] | None = None,
    local_cache_root: Path | None = None,
) -> tuple[Dataset, int | None, Dataset | None]:
    """Return (sampled_dataset, total_if_known, materialized_if_resolved).

    For in-memory datasets, total is the full example count. For deferred
    datasets we narrow via `select(limit=sample_size)` then `resolve()`, and
    memoize the result by `(dataset.id, sample_size)` so overview pages that
    hit the same underlying dataset from every step don't each pay a fresh
    round-trip to the source.
    """
    if dataset.is_deferred:
        dataset = _bind_artifact_dataset_step_refs(
            dataset,
            artifact_manifests_by_step=artifact_manifests_by_step,
            local_cache_root=local_cache_root,
        )
        cache_key = (str(dataset.id or ""), int(sample_size))
        if cache_key[0]:
            now = time.monotonic()
            with _RESOLVED_CACHE_LOCK:
                entry = _RESOLVED_CACHE.get(cache_key)
                if entry is not None:
                    expiry, materialized = entry
                    if expiry > now:
                        return materialized, None, materialized
                    _RESOLVED_CACHE.pop(cache_key, None)
        _precheck_deferred_source(dataset)
        narrowed = dataset.select(limit=sample_size)
        materialized = narrowed.resolve()
        if cache_key[0]:
            with _RESOLVED_CACHE_LOCK:
                _RESOLVED_CACHE[cache_key] = (
                    time.monotonic() + _RESOLVED_TTL_SECONDS,
                    materialized,
                )
        return materialized, None, materialized

    total = len(dataset.examples)
    if sample_size >= total:
        return dataset, total, None
    narrowed = dataset.select(limit=sample_size)
    return narrowed, total, None


def clear_resolved_dataset_cache() -> None:
    """Drop every memoized deferred-dataset resolve. Used by the cache
    invalidate endpoint."""
    with _RESOLVED_CACHE_LOCK:
        _RESOLVED_CACHE.clear()


def _bind_artifact_dataset_step_refs(
    dataset: Dataset,
    *,
    artifact_manifests_by_step: Mapping[str, ArtifactManifest] | None,
    local_cache_root: Path | None,
) -> Dataset:
    source = dataset.source or {}
    if source.get("kind") != "artifact_dataset":
        return dataset
    fetch = dict(dataset.fetch or {})
    artifact_value = fetch.get("artifact")
    step_name = _artifact_source_step_name(artifact_value)
    if step_name is None:
        return dataset
    if artifact_manifests_by_step is None:
        raise RuntimeError(
            f"artifact-backed dataset for step {step_name!r} cannot be resolved without step manifests"
        )
    manifest = artifact_manifests_by_step.get(step_name)
    if manifest is None:
        raise RuntimeError(f"artifact-backed dataset source step {step_name!r} has no recorded artifact manifest")
    store = _artifact_store_from_manifest(manifest, local_cache_root=local_cache_root)
    fetch["artifact"] = artifact_from_manifest(manifest, store=store)
    return Dataset(
        examples=(),
        id=dataset.id,
        name=dataset.name,
        source=source,
        fetch=fetch,
        selection=dict(dataset.selection),
    )


def _artifact_source_step_name(value: Any) -> str | None:
    if isinstance(value, Mapping) and value.get("kind") == "step_ref":
        step = value.get("step")
        return str(step) if step is not None else None
    step = getattr(value, "step", None)
    kind = getattr(value, "kind", None)
    if kind == "step_ref" and step is not None:
        return str(step)
    return None


def _artifact_store_from_manifest(manifest: ArtifactManifest, *, local_cache_root: Path | None) -> Any:
    ref = _first_storage_ref(manifest.storage_refs)
    if ref is None:
        raise RuntimeError(f"Artifact {manifest.artifact_id!r} has no storage refs to infer a store from")
    store_kind = str(ref.get("store") or "").strip()
    path = ref.get("path")
    if not path:
        raise RuntimeError(f"Artifact {manifest.artifact_id!r} storage ref is missing a path")
    root = _infer_artifact_root(path, manifest.artifact_id)
    if store_kind == "modal_volume":
        name = ref.get("name")
        if not name:
            raise RuntimeError(f"Artifact {manifest.artifact_id!r} Modal ref is missing a volume name")
        return ModalVolumeStore(name=str(name), root=str(root), local_cache_root=local_cache_root)
    if store_kind in {"local", "local_path"}:
        return LocalArtifactStore(root=root)
    raise RuntimeError(f"Unsupported artifact store kind for dataset preview: {store_kind!r}")


def _first_storage_ref(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if "store" in value and "path" in value:
            return value
        for child in value.values():
            found = _first_storage_ref(child)
            if found is not None:
                return found
    return None


def _infer_artifact_root(path: str | Path, artifact_id: str) -> Path:
    resolved = Path(path)
    parts = resolved.parts
    try:
        index = parts.index(artifact_id)
    except ValueError:
        return resolved.parent
    if index == 0:
        return resolved.parent
    root = Path(parts[0])
    for part in parts[1:index]:
        root /= part
    return root


def _precheck_deferred_source(dataset: Dataset) -> None:
    """Fail fast with a helpful message before calling resolve()."""
    source = dataset.source or {}
    if source.get("kind") == "postgres":
        env_var = source.get("url_env_var")
        if env_var and not os.environ.get(env_var):
            raise EnvVarMissing(f"env var {env_var!r} is not set locally")


# ---------------------------------------------------------------------------
# Row / source formatting
# ---------------------------------------------------------------------------


def _example_row(ex: Example) -> DatasetPreviewRow:
    return DatasetPreviewRow(
        example_key=ex.key,
        case_key=ex.case_key,
        prompt_preview=_prompt_preview(ex.prompt),
        labels=dict(ex.labels),
        metadata=dict(ex.metadata),
    )


def _prompt_preview(prompt: Any) -> str:
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, (list, tuple)):
        parts = []
        for msg in prompt:
            if isinstance(msg, Mapping):
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, Mapping) else str(c) for c in content
                    )
                parts.append(f"[{role}] {content}")
            else:
                parts.append(str(msg))
        text = "\n".join(parts)
    else:
        text = str(prompt)
    if len(text) > PROMPT_PREVIEW_CHARS:
        return text[: PROMPT_PREVIEW_CHARS - 1] + "…"
    return text


def _describe_source(dataset: Dataset) -> DatasetSourceInfo:
    source = dict(dataset.source or {})
    fetch = dict(dataset.fetch or {})
    selection = dict(dataset.selection or {})
    kind = str(source.get("kind") or ("memory" if dataset.examples else "unknown"))

    label_columns = [str(c) for c in fetch.get("label_columns", ()) or ()]
    case_columns = [str(c) for c in fetch.get("case_columns", ()) or ()]
    metadata_columns = [str(c) for c in fetch.get("metadata_columns", ()) or ()]

    selection_keys_raw = selection.get("keys")
    selection_keys = (
        [str(k) for k in selection_keys_raw]
        if isinstance(selection_keys_raw, (list, tuple))
        else None
    )

    # Construction notes — short (label, value) pairs that explain how the
    # dataset was put together. Human-readable and shown inline in the UI.
    construction: list[dict[str, str]] = []
    is_deferred = dataset.source is not None
    if is_deferred:
        construction.append({"label": "deferred", "value": "yes"})
        if source.get("kind") == "postgres":
            if source.get("url_env_var"):
                construction.append({"label": "env_var", "value": str(source["url_env_var"])})
            if fetch.get("table"):
                construction.append({"label": "table", "value": str(fetch["table"])})
            if fetch.get("sql"):
                construction.append({"label": "sql", "value": str(fetch["sql"])})
            if fetch.get("prompt_column"):
                construction.append({"label": "prompt_column", "value": str(fetch["prompt_column"])})
            if fetch.get("example_key_column"):
                construction.append({"label": "example_key_column", "value": str(fetch["example_key_column"])})
            if label_columns:
                construction.append({"label": "label_columns", "value": ", ".join(label_columns)})
            if case_columns:
                construction.append({"label": "case_columns", "value": ", ".join(case_columns)})
            if metadata_columns:
                construction.append({"label": "metadata_columns", "value": ", ".join(metadata_columns)})
        else:
            construction.append({"label": "source", "value": str(source.get("kind") or "unknown")})
    else:
        construction.append({"label": "deferred", "value": "no"})
        construction.append(
            {"label": "examples", "value": str(len(dataset.examples))}
        )
        construction.append({"label": "built_as", "value": "in-memory (Dataset.from_examples)"})

    if selection.get("limit") is not None:
        construction.append({"label": "selection.limit", "value": str(selection["limit"])})
    if selection_keys:
        preview = ", ".join(selection_keys[:4])
        if len(selection_keys) > 4:
            preview += f", … (+{len(selection_keys) - 4})"
        construction.append({"label": "selection.keys", "value": preview})

    return DatasetSourceInfo(
        kind=kind,
        env_var=source.get("url_env_var"),
        table=fetch.get("table"),
        query=fetch.get("sql"),
        limit=selection.get("limit") or fetch.get("limit"),
        identity={**source, "fetch": fetch, "selection": selection} if source else None,
        name=dataset.name,
        dataset_id=dataset.id,
        deferred=is_deferred,
        total_examples=None if is_deferred else len(dataset.examples),
        prompt_column=fetch.get("prompt_column") if is_deferred else None,
        example_key_column=fetch.get("example_key_column") if is_deferred else None,
        label_columns=label_columns,
        case_columns=case_columns,
        metadata_columns=metadata_columns,
        selection_keys=selection_keys,
        construction=construction,
    )


# ---------------------------------------------------------------------------
# Label distribution tallying
# ---------------------------------------------------------------------------


def _tally_labels(examples: tuple[Example, ...], source_step: str | None) -> list[LabelDistribution]:
    if not examples:
        return []
    names: set[str] = set()
    for ex in examples:
        names.update(ex.labels.keys())
    out: list[LabelDistribution] = []
    for name in sorted(names):
        values = [ex.labels.get(name) for ex in examples if name in ex.labels]
        out.append(_distribution(name, values, source_step))
    return out


def _distribution(
    name: str,
    values: list[Any],
    source_step: str | None,
) -> LabelDistribution:
    numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric_values and len(numeric_values) == len(values):
        return LabelDistribution(
            label_name=name,
            unique_values=len(set(numeric_values)),
            buckets=[],
            numeric_summary={
                "min": float(min(numeric_values)),
                "max": float(max(numeric_values)),
                "mean": float(mean(numeric_values)),
                "stddev": float(pstdev(numeric_values)) if len(numeric_values) > 1 else 0.0,
            },
            source_step=source_step,
        )
    stringified = [_stringify_label(v) for v in values]
    counts = Counter(stringified)
    total = sum(counts.values())
    buckets = [
        LabelDistributionBucket(value=v, count=c, fraction=(c / total if total else 0.0))
        for v, c in counts.most_common(20)
    ]
    return LabelDistribution(
        label_name=name,
        unique_values=len(counts),
        buckets=buckets,
        numeric_summary=None,
        source_step=source_step,
    )


def _stringify_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value if len(value) <= 40 else value[:39] + "…"
    return type(value).__name__

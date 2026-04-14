"""Deterministic toy engine used by tests and local contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from pipelines_v2.core.types import EngineCapability, stable_hash
from pipelines_v2.data.datasets import Example
from pipelines_v2.engine.base import EngineCaptureResult, PythonRuntimeSpec
from pipelines_v2.engine.prompt_metadata import rebase_token_sections, resolve_prompt_metadata, token_sections_from_metadata
from pipelines_v2.operations.specs import CaptureSpec, MoERoutingSite, ResidualSite, RoutingRecord


@dataclass(frozen=True, slots=True)
class ToyEngine:
    """Deterministic engine for contracts, operation tests, and local capture."""

    hidden_size: int = 4
    num_layers: int = 4
    sequence_length: int = 8
    num_experts: int = 4
    top_k: int = 2
    enabled_capabilities: frozenset[EngineCapability] = field(
        default_factory=lambda: frozenset(
            {
                EngineCapability.GENERATION,
                EngineCapability.LOGPROBS,
                EngineCapability.RESIDUAL_CAPTURE,
                EngineCapability.MOE_ROUTING_CAPTURE,
                EngineCapability.STRUCTURED_OUTPUT,
            }
        )
    )

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "toy",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToyEngine":
        data = dict(payload)
        data.pop("kind", None)
        return cls(**data)

    def capabilities(self) -> set[EngineCapability]:
        return set(self.enabled_capabilities)

    def runtime_spec(self) -> PythonRuntimeSpec:
        return PythonRuntimeSpec(local_python_sources=("pipelines_v2",))

    def planning_errors(self, spec: CaptureSpec) -> tuple[str, ...]:
        return ()

    def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
        features: dict[str, dict[str, Any]] = {}
        generations: list[dict[str, Any]] = []

        for site in spec.sites:
            if isinstance(site, ResidualSite):
                features[site.name] = self._capture_residual(site, spec)
            elif isinstance(site, MoERoutingSite):
                features[site.name] = self._capture_routing(site, spec)
            else:
                raise TypeError(f"Unsupported capture site: {type(site).__name__}")

        if spec.generation.enabled:
            for example in spec.dataset.examples:
                generation_payload = self._generation_payload(example, spec)
                generations.append(
                    {
                        "example_key": example.key,
                        **generation_payload,
                        "finish_reason": "length" if spec.generation.max_tokens else "stop",
                    }
                )

        return EngineCaptureResult(
            features=features,
            generations=generations,
            metadata={"tokenizer": "toy_synthetic_sequence_v1"},
        )

    def _generation_payload(self, example: Example, spec: CaptureSpec) -> dict[str, Any]:
        if spec.generation.structured_output is None:
            return {"text": f"toy_generation:{example.key}"}
        structured = _toy_structured_output(example)
        return {
            "text": json.dumps(structured, sort_keys=True),
            "structured_output": structured,
        }

    def _capture_residual(self, site: ResidualSite, spec: CaptureSpec) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for layer in site.layers:
            layer_payload: dict[str, Any] = {}
            for example in spec.dataset.examples:
                token_sections = _toy_token_sections(example, spec.prompt_metadata_builder)
                positions = site.tokens.resolve(self.sequence_length, token_sections=token_sections)
                values = np.stack(
                    [self._activation_vector(example, layer, pos) for pos in positions],
                    axis=0,
                )
                feature_token_sections = rebase_token_sections(
                    token_sections=token_sections,
                    selected_positions=positions,
                )
                layer_payload[example.key] = {
                    "tokens": positions,
                    "values": values,
                    "prompt_hash": example.prompt_hash,
                    "token_sections": feature_token_sections,
                }
            layers[str(layer)] = layer_payload
        return {
            "kind": "residual",
            "site": site.site,
            "storage": {"dtype": site.storage.dtype, "format": site.storage.format},
            "layers": layers,
        }

    def _capture_routing(self, site: MoERoutingSite, spec: CaptureSpec) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        requested = tuple(site.record)
        for layer in site.layers:
            layer_payload: dict[str, Any] = {}
            for example in spec.dataset.examples:
                token_sections = _toy_token_sections(example, spec.prompt_metadata_builder)
                positions = site.tokens.resolve(self.sequence_length, token_sections=token_sections)
                records_by_token: dict[str, Any] = {}
                for pos in positions:
                    gate_logits = self._gate_logits(example, layer, pos)
                    records_by_token[str(pos)] = self._routing_records(requested, gate_logits)
                feature_token_sections = rebase_token_sections(
                    token_sections=token_sections,
                    selected_positions=positions,
                )
                layer_payload[example.key] = {
                    "tokens": positions,
                    "records": records_by_token,
                    "prompt_hash": example.prompt_hash,
                    "token_sections": feature_token_sections,
                }
            layers[str(layer)] = layer_payload
        return {
            "kind": "moe_routing",
            "routing_policy": {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "source": "toy_observed",
            },
            "layers": layers,
        }

    def _activation_vector(
        self,
        example: Example,
        layer: int,
        token_pos: int,
    ) -> npt.NDArray[np.float32]:
        seed = int(stable_hash([example.key, example.prompt_hash])[:8], 16)
        base = (seed % 10_000) / 10_000
        return np.asarray(
            [base + layer * 0.1 + token_pos * 0.01 + dim * 0.001 for dim in range(self.hidden_size)],
            dtype=np.float32,
        )

    def _gate_logits(
        self,
        example: Example,
        layer: int,
        token_pos: int,
    ) -> npt.NDArray[np.float32]:
        seed = int(stable_hash(["router", example.key, layer, token_pos])[:8], 16)
        return np.asarray(
            [math.sin(seed + expert) * 2.0 for expert in range(self.num_experts)],
            dtype=np.float32,
        )

    def _routing_record(
        self,
        record: RoutingRecord,
        gate_logits: npt.NDArray[np.float32],
        *,
        topk_from_gate_k: int,
    ) -> dict[str, Any]:
        if record.kind == "gate_logits":
            return {"gate_logits": gate_logits.astype(_float_dtype(record.params.get("dtype", "float16")))}
        if record.kind == "gate_probs":
            return {"gate_probs": _softmax(gate_logits).astype(_float_dtype(record.params.get("dtype", "float16")))}
        if record.kind == "routing_decisions":
            top = _topk(gate_logits, self.top_k)
            return {
                "routing_decisions": {
                    "source": "observed",
                    "expert_ids": top.indices,
                    "weights": _normalize(top.values),
                }
            }
        if record.kind == "topk_from_gate":
            k = int(record.params["k"])
            top = _topk(gate_logits, k)
            payload: dict[str, Any] = {
                "source": "derived_from_gate_logits",
                "expert_ids": top.indices,
            }
            if record.params.get("include_weights", True):
                payload["weights"] = _normalize(top.values)
            return {"topk_from_gate": payload}
        if record.kind == "expert_load":
            source = str(record.params.get("source") or "topk_from_gate")
            if source == "topk_from_gate":
                top = _topk(gate_logits, topk_from_gate_k)
            else:
                top = _topk(gate_logits, self.top_k)
            return {"expert_load": {"source": source, "counts": {str(int(idx)): 1 for idx in top.indices}}}
        raise ValueError(f"Unsupported routing record: {record.kind}")

    def _routing_records(
        self,
        requested: tuple[RoutingRecord, ...],
        gate_logits: npt.NDArray[np.float32],
    ) -> dict[str, Any]:
        token_records: dict[str, Any] = {}
        topk_from_gate_k = _requested_topk_from_gate_k(requested, fallback=self.top_k)
        for record in requested:
            token_records.update(self._routing_record(record, gate_logits, topk_from_gate_k=topk_from_gate_k))
        return token_records


@dataclass(frozen=True, slots=True)
class TopKResult:
    indices: npt.NDArray[np.int64]
    values: npt.NDArray[np.float32]


def _softmax(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    shifted = values.astype(np.float32) - np.max(values)
    exps = np.exp(shifted)
    return (exps / np.sum(exps)).astype(np.float32)


def _float_dtype(name: str) -> Any:
    normalized = str(name).lower()
    if normalized == "float16":
        return np.float16
    if normalized in {"float32", "bfloat16"}:
        return np.float32
    raise ValueError(f"Unsupported routing dtype: {name}")


def _requested_topk_from_gate_k(requested: tuple[RoutingRecord, ...], *, fallback: int) -> int:
    for record in requested:
        if record.kind == "topk_from_gate":
            return int(record.params["k"])
    return int(fallback)


def _topk(values: npt.NDArray[np.floating[Any]], k: int) -> TopKResult:
    if k <= 0:
        return TopKResult(
            indices=np.asarray([], dtype=np.int64),
            values=np.asarray([], dtype=np.float32),
        )
    bounded_k = min(k, int(values.shape[-1]))
    indices = np.argsort(values)[::-1][:bounded_k].astype(np.int64)
    return TopKResult(indices=indices, values=values[indices].astype(np.float32))


def _normalize(values: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.float32]:
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    shifted = values.astype(np.float32) - np.min(values) + np.float32(1e-6)
    total = np.sum(shifted)
    return (shifted / total).astype(np.float32)


def _toy_token_sections(example: Example, builder: Any | None) -> dict[str, list[int]]:
    rendered_prompt = example.prompt if isinstance(example.prompt, str) else json.dumps(example.prompt, sort_keys=True)
    metadata = resolve_prompt_metadata(
        metadata=example.metadata,
        rendered_prompt=rendered_prompt,
        builder=builder,
    )
    return token_sections_from_metadata(
        metadata=metadata,
        offsets=None,
        require_sections=False,
        allow_char_spans=False,
    )


def _toy_structured_output(example: Example) -> dict[str, Any]:
    raw = example.labels.get("expected_output_json")
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str):
        payload = json.loads(raw)
    else:
        payload = {}
    return {
        "action": str(payload.get("action") or "observe").lower(),
        "asset": str(payload.get("asset") or "NONE").upper(),
        "size": str(payload.get("size") or "none").lower(),
    }

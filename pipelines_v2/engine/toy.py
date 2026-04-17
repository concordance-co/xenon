"""Deterministic toy engine used by tests and local contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from pipelines_v2.core.types import EngineCapability, SpecValidationError, stable_hash
from pipelines_v2.data.datasets import Example
from pipelines_v2.engine.base import (
    EngineCaptureResult,
    EngineGenerationResult,
    EngineInterventionResult,
    PythonRuntimeSpec,
)
from pipelines_v2.engine.prompt_metadata import rebase_token_sections, resolve_prompt_metadata, token_sections_from_metadata
from pipelines_v2.operations.interventions import (
    ActivationPatchSpec,
    AddDirectionPatch,
    GenerationRunSpec,
    InterchangePatch,
    PatchedGenerationSpec,
    ProjectOutPatch,
    RandomControlPatch,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from pipelines_v2.operations.interventions.runtime import (
    load_activation_bank_source,
    load_centroid_source,
    load_direction_source,
    load_path_mask_source,
    load_subspace_source,
    partition_cases_by_activation_bank,
    resolve_generation_examples,
    resolve_patched_generation_cases,
    resolve_patched_generation_targets,
)
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
                EngineCapability.ACTIVATION_PATCHING,
                EngineCapability.REQUEST_SCOPED_INTERVENTIONS,
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

    def semantic_identity(self) -> dict[str, Any]:
        return self.identity()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToyEngine":
        data = dict(payload)
        data.pop("kind", None)
        return cls(**data)

    def capabilities(self) -> set[EngineCapability]:
        return set(self.enabled_capabilities)

    def runtime_spec(self) -> PythonRuntimeSpec:
        return PythonRuntimeSpec(local_python_sources=("pipelines_v2",))

    def planning_errors(self, spec: Any) -> tuple[str, ...]:
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

    def generate(self, spec: GenerationRunSpec) -> EngineGenerationResult:
        rows = [
            {
                "example_key": example.key,
                "example": example.to_dict(),
                **self._generation_run_payload(example, spec),
            }
            for example in resolve_generation_examples(spec)
        ]
        return EngineGenerationResult(
            rows=rows,
            metadata={"backend": "toy"},
        )

    def intervene(self, spec: PatchedGenerationSpec) -> EngineInterventionResult:
        if spec.patch.requires_pairing():
            return self._intervene_paired(spec)
        return self._intervene_unpaired(spec)

    def _intervene_paired(self, spec: PatchedGenerationSpec) -> EngineInterventionResult:
        activation_bank = load_activation_bank_source(spec.patch)
        resolved_cases, skipped_cases = resolve_patched_generation_cases(spec)
        resolved_cases, source_feature_skips = partition_cases_by_activation_bank(
            spec=spec,
            activation_bank=activation_bank,
            resolved_cases=resolved_cases,
        )
        skipped_cases.extend(source_feature_skips)
        rows: list[dict[str, Any]] = []

        for skipped in skipped_cases:
            rows.append(
                {
                    "case_key": skipped.get("case_key"),
                    "status": "skipped",
                    "skip_reason": skipped.get("skip_reason", ""),
                    "patch_stats": {},
                }
            )

        for item in resolved_cases:
            case_key = str(item["case_key"])
            target: Example = item["target"]
            donor: Example = item["donor"]
            target_sections = _toy_token_sections(target, spec.prompt_metadata_builder)
            target_positions = spec.patch.target_tokens.resolve(
                self.sequence_length,
                token_sections=target_sections,
            )
            donor_positions, patch_stats, skip_reason = self._paired_patch_state(
                patch=spec.patch,
                activation_bank=activation_bank,
                donor_key=donor.key,
                target_positions=target_positions,
            )
            if skip_reason is not None:
                rows.append(
                    {
                        "case_key": case_key,
                        "example_key": target.key,
                        "donor_example_key": donor.key,
                        "example": target.to_dict(),
                        "status": "skipped",
                        "skip_reason": skip_reason,
                        "patch_stats": patch_stats,
                    }
                )
                continue

            patched = self._patched_generation_payload(target, donor=donor)
            row = {
                "case_key": case_key,
                "example_key": target.key,
                "donor_example_key": donor.key,
                "example": target.to_dict(),
                "status": "ok",
                "skip_reason": "",
                **patched,
                "patch_stats": patch_stats,
                "target_tokens": list(target_positions),
            }
            if isinstance(spec.patch, InterchangePatch):
                row["donor_tokens"] = list(donor_positions or ())
            else:
                row["read_tokens"] = list(donor_positions or ())
            rows.append(row)

        usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
        return EngineInterventionResult(
            summary={
                "example_count": len(rows),
                "patched_count": len(usable_rows),
                "skipped_count": len(rows) - len(usable_rows),
                "case_count": len(resolved_cases) + len(skipped_cases),
            },
            rows=rows,
            metadata={
                "backend": "toy",
                "write_site": spec.patch.write_site.site,
                "operator": spec.patch.operator,
            },
        )

    def _intervene_unpaired(self, spec: PatchedGenerationSpec) -> EngineInterventionResult:
        targets = resolve_patched_generation_targets(spec)
        rows: list[dict[str, Any]] = []
        for target in targets:
            target_sections = _toy_token_sections(target, spec.prompt_metadata_builder)
            target_positions = spec.patch.target_tokens.resolve(
                self.sequence_length,
                token_sections=target_sections,
            )
            patch_stats, skip_reason = self._unpaired_patch_state(
                target=target,
                target_positions=target_positions,
                patch=spec.patch,
            )
            if skip_reason is not None:
                rows.append(
                    {
                        "example_key": target.key,
                        "example": target.to_dict(),
                        "status": "skipped",
                        "skip_reason": skip_reason,
                        "patch_stats": patch_stats,
                    }
                )
                continue
            rows.append(
                {
                    "example_key": target.key,
                    "example": target.to_dict(),
                    "status": "ok",
                    "skip_reason": "",
                    "generated_text": f"toy_generation:{spec.patch.operator}:{target.key}",
                    "generated_token_ids": [],
                    "finish_reason": "length",
                    "request_id": f"{spec.patch.operator}:{target.key}",
                    "patch_stats": patch_stats,
                    "target_tokens": list(target_positions),
                }
            )

        usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
        return EngineInterventionResult(
            summary={
                "example_count": len(rows),
                "patched_count": len(usable_rows),
                "skipped_count": len(rows) - len(usable_rows),
                "target_count": len(targets),
            },
            rows=rows,
            metadata={
                "backend": "toy",
                "write_site": spec.patch.write_site.site,
                "operator": spec.patch.operator,
            },
        )

    def _generation_payload(self, example: Example, spec: CaptureSpec) -> dict[str, Any]:
        if spec.generation.structured_output is None:
            return {"text": f"toy_generation:{example.key}"}
        structured = _toy_structured_output(example)
        return {
            "text": json.dumps(structured, sort_keys=True),
            "structured_output": structured,
        }

    def _generation_run_payload(
        self,
        example: Example,
        spec: GenerationRunSpec,
    ) -> dict[str, Any]:
        if spec.generation.structured_output is None:
            return {
                "generated_text": f"toy_generation:{example.key}",
                "generated_token_ids": [],
                "finish_reason": "length",
                "request_id": f"generation:{example.key}",
            }
        structured = _toy_structured_output(example)
        return {
            "generated_text": json.dumps(structured, sort_keys=True),
            "generated_token_ids": [],
            "finish_reason": "length",
            "request_id": f"generation:{example.key}",
            "structured_output": structured,
        }

    def _patched_generation_payload(
        self,
        example: Example,
        *,
        donor: Example,
    ) -> dict[str, Any]:
        label = str(donor.labels.get("class") or donor.key)
        return {
            "generated_text": f"toy_generation:{label}",
            "generated_token_ids": [],
            "finish_reason": "length",
            "request_id": f"patched:{example.key}:{donor.key}",
        }

    def _paired_patch_state(
        self,
        *,
        patch: ActivationPatchSpec,
        activation_bank: dict[str, Any],
        donor_key: str,
        target_positions: Sequence[int],
    ) -> tuple[list[int] | None, dict[str, Any], str | None]:
        layers_payload = activation_bank["layers"]
        if isinstance(patch, InterchangePatch):
            donor_selector = patch.donor_tokens or patch.target_tokens
            source_layers = tuple(int(patch.source_layer_for(int(write_layer))) for write_layer in patch.write_site.layers)
        elif isinstance(patch, ResidualPathPatch):
            donor_selector = patch.read_tokens or patch.target_tokens
            source_layers = tuple(sorted({int(edge["source_layer"]) for edge in load_path_mask_source(patch)["edges"]}))
        else:
            return None, {}, f"unsupported paired patch type: {type(patch).__name__}"
        first_layer = str(int(source_layers[0]))
        donor_record = dict(layers_payload[first_layer]).get(donor_key)
        if not isinstance(donor_record, dict):
            return None, {}, "activation_bank is missing donor activation rows"
        donor_values = np.asarray(donor_record.get("values"))
        donor_token_sections = donor_record.get("token_sections")
        donor_positions = donor_selector.resolve(
            int(donor_values.shape[0]),
            token_sections=donor_token_sections,
        )
        if len(target_positions) != len(donor_positions):
            return donor_positions, {}, "target and donor token selections must have equal length"

        patch_stats: dict[str, Any] = {}
        if isinstance(patch, InterchangePatch):
            for write_layer in patch.write_site.layers:
                source_layer = int(patch.source_layer_for(int(write_layer)))
                layer_record = dict(layers_payload[str(int(source_layer))])
                layer_donor = layer_record.get(donor_key)
                if not isinstance(layer_donor, dict):
                    return donor_positions, patch_stats, "activation_bank is missing required per-layer donor rows"
                donor_values = np.asarray(layer_donor.get("values"), dtype=np.float32)
                delta = donor_values[donor_positions]
                patch_stats[str(int(write_layer))] = {
                    "layer": int(write_layer),
                    "source_layer": int(source_layer),
                    "target_span": list(target_positions),
                    "donor_span": list(donor_positions),
                    "token_count": len(target_positions),
                    "delta_norm_raw": float(np.linalg.norm(delta)),
                }
        else:
            for edge in load_path_mask_source(patch)["edges"]:
                source_layer = int(edge["source_layer"])
                write_layer = int(edge["write_layer"])
                if int(write_layer) not in {int(layer) for layer in patch.write_site.layers}:
                    continue
                layer_record = dict(layers_payload[str(int(source_layer))])
                layer_donor = layer_record.get(donor_key)
                if not isinstance(layer_donor, dict):
                    return donor_positions, patch_stats, "activation_bank is missing required per-layer donor rows"
                donor_values = np.asarray(layer_donor.get("values"), dtype=np.float32)
                donor_section = donor_values[donor_positions]
                weight = float(edge.get("weight", 1.0))
                patch_stats[str(int(write_layer))] = {
                    "layer": int(write_layer),
                    "source_layer": int(source_layer),
                    "operator": patch.operator,
                    "transport": patch.transport,
                    "weight": weight,
                    "token_count": len(target_positions),
                    "target_tokens": [int(pos) for pos in target_positions],
                    "read_tokens": [int(pos) for pos in donor_positions],
                    "delta_norm_raw": float(np.linalg.norm(weight * donor_section)),
                }
        return donor_positions, patch_stats, None

    def _unpaired_patch_state(
        self,
        *,
        target: Example,
        target_positions: Sequence[int],
        patch: ActivationPatchSpec,
    ) -> tuple[dict[str, Any], str | None]:
        if not target_positions:
            return {}, "target token selection resolved to no positions"
        source_payload = (
            load_subspace_source(patch)
            if isinstance(patch, (ProjectOutPatch, RandomControlPatch, SwapComponentsPatch))
            or (isinstance(patch, AddDirectionPatch) and patch.uses_subspace())
            else None
        )
        direction_payload = load_direction_source(patch) if isinstance(patch, AddDirectionPatch) else None
        centroid_payload = load_centroid_source(patch) if isinstance(patch, (SwapMeanPatch, SwapComponentsPatch)) else None
        patch_stats: dict[str, Any] = {}
        for write_layer in patch.write_site.layers:
            source_layer = patch.source_layer_for(int(write_layer))
            section = np.stack(
                [self._activation_vector(target, int(write_layer), int(pos)) for pos in target_positions],
                axis=0,
            ).astype(np.float32)
            mu = section.mean(axis=0)
            selected_component_count = 0
            if isinstance(patch, (ProjectOutPatch, RandomControlPatch, SwapComponentsPatch)):
                assert source_payload is not None
                layer_payload = dict(source_payload["layers"][str(int(source_layer))])
                mean = np.asarray(layer_payload["mean"], dtype=np.float32)
                scale = np.asarray(layer_payload["scale"], dtype=np.float32)
                safe_scale = np.asarray(layer_payload["safe_scale"], dtype=np.float32)
                components = np.asarray(layer_payload["components"], dtype=np.float32)
                selected = patch.component_indices_for(int(write_layer))
                if selected:
                    components = components[list(selected)]
                if components.ndim != 2 or components.shape[0] == 0:
                    return patch_stats, f"no subspace components available for write layer {int(write_layer)}"
                row_norms = np.linalg.norm(components, axis=1, keepdims=True)
                components = components / np.where(row_norms == 0, 1.0, row_norms)
                centered_std = (mu - mean) / safe_scale
                coeff = centered_std @ components.T
                projected = coeff @ components
                selected_component_count = int(components.shape[0])
            if isinstance(patch, ProjectOutPatch):
                delta_raw = (-float(patch.strength) * projected) * scale
            elif isinstance(patch, RandomControlPatch):
                random_rows = self._random_orthogonal_rows(
                    target_rows=components,
                    num_rows=max(1, int(components.shape[0])),
                    dim=int(mean.shape[0]),
                    seed=int(patch.random_seed) + int(write_layer),
                )
                random_coeff = centered_std @ random_rows.T
                random_projected = random_coeff @ random_rows
                if patch.match_projected_norm:
                    proj_norm = float(np.linalg.norm(projected))
                    random_norm = float(np.linalg.norm(random_projected))
                    if proj_norm > 0 and random_norm > 1e-8:
                        random_projected = random_projected * (proj_norm / random_norm)
                delta_raw = (-float(patch.strength) * random_projected) * scale
            elif isinstance(patch, AddDirectionPatch):
                assert direction_payload is not None
                layer_payload = dict(direction_payload["layers"][str(int(source_layer))])
                if patch.uses_subspace():
                    assert source_payload is not None
                    subspace_layer = dict(source_payload["layers"][str(int(source_layer))])
                    scale = np.asarray(subspace_layer["scale"], dtype=np.float32)
                    components = np.asarray(subspace_layer["components"], dtype=np.float32)
                    selected = patch.component_indices_for(int(write_layer))
                    if selected:
                        components = components[list(selected)]
                    weights = np.asarray(layer_payload.get("subspace_weights", ()), dtype=np.float32)
                    if selected and weights.size:
                        weights = weights[list(selected)]
                    if components.ndim == 1:
                        components = components[None, :]
                    delta_std = weights @ components if weights.size else np.zeros((int(scale.shape[0]),), dtype=np.float32)
                    delta_raw = float(patch.strength) * (delta_std * scale)
                    selected_component_count = int(components.shape[0])
                else:
                    delta_raw = float(patch.strength) * np.asarray(layer_payload["raw_vector"], dtype=np.float32)
            elif isinstance(patch, SwapMeanPatch):
                assert centroid_payload is not None
                layer_payload = dict(centroid_payload["layers"][str(int(source_layer))])
                donor_mean = np.asarray(layer_payload["centroids"][patch.centroid_name], dtype=np.float32)
                delta_raw = float(patch.strength) * (donor_mean - mu)
            elif isinstance(patch, SwapComponentsPatch):
                assert centroid_payload is not None
                layer_payload = dict(centroid_payload["layers"][str(int(source_layer))])
                donor_mean = np.asarray(layer_payload["centroids"][patch.centroid_name], dtype=np.float32)
                donor_centered_std = (donor_mean - mean) / safe_scale
                donor_coeff = donor_centered_std @ components.T
                donor_projected = donor_coeff @ components
                delta_raw = float(patch.strength) * ((donor_projected - projected) * scale)
            else:
                return patch_stats, f"unsupported toy patch operator: {patch.operator!r}"
            patch_stats[str(int(write_layer))] = {
                "layer": int(write_layer),
                "source_layer": int(source_layer),
                "operator": patch.operator,
                "token_count": len(target_positions),
                "target_tokens": [int(pos) for pos in target_positions],
                "delta_norm_raw": float(np.linalg.norm(delta_raw)),
                "selected_component_count": int(selected_component_count),
            }
        return patch_stats, None

    def _random_orthogonal_rows(
        self,
        *,
        target_rows: np.ndarray,
        num_rows: int,
        dim: int,
        seed: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        attempts = 0
        while attempts < 8:
            rand = rng.standard_normal((num_rows, dim), dtype=np.float32)
            if target_rows.size:
                rand = rand - (rand @ target_rows.T) @ target_rows
            row_norm = np.linalg.norm(rand, axis=1)
            keep = row_norm > 1e-6
            rand = rand[keep]
            if rand.shape[0] < num_rows:
                attempts += 1
                continue
            q, _ = np.linalg.qr(rand[:num_rows].T)
            ortho = q.T.astype(np.float32, copy=False)
            if ortho.shape[0] == num_rows:
                return ortho
            attempts += 1
        raise RuntimeError("Failed to sample random orthogonal control rows")

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

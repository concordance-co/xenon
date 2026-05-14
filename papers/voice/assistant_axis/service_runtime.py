"""Assistant Axis service logic for runtime deployments."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field


DEFAULT_TRAITS = ("calm", "supportive", "technical", "analytical", "confident", "verbose", "hostile", "condescending")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    prompt: str | list[ChatMessage]
    max_tokens: int = Field(default=160, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


class SteerRequest(GenerateRequest):
    trait: str
    strength: float = Field(default=1.0, ge=-8.0, le=8.0)


class TraceTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ScoreTraceRequest(BaseModel):
    text: str | None = None
    turns: list[TraceTurn] | None = None
    traits: list[str] = Field(default_factory=lambda: list(DEFAULT_TRAITS))


def create_service(ctx: Any) -> "AssistantAxisService":
    """Build the long-lived service object for function-style deployments."""

    return ctx.singleton("assistant_axis_service", lambda: AssistantAxisService(ctx))


def create_app(ctx: Any) -> Any:
    """Build a FastAPI app for local compatibility; deployments use create_service."""

    from fastapi import Body, Depends, FastAPI, Header, HTTPException

    api = FastAPI(title="Assistant Axis Service")
    service = create_service(ctx)
    if _env_bool("ASSISTANT_AXIS_SERVICE_EAGER_STEER_RUNTIME", default=False):
        service.ensure_steer_runtime()

    def require_service_key(authorization: str | None = Header(default=None)) -> None:
        expected = os.getenv("ASSISTANT_AXIS_SERVICE_API_KEY")
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid service token")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "assistant-axis", "deployment": ctx.name, "model_key": service.model_key}

    @api.post("/warmup")
    async def warmup(_: None = Depends(require_service_key)) -> dict[str, Any]:
        service.ensure_runtime()
        loaded_traits = [trait for trait in DEFAULT_TRAITS if service.direction_for_trait(trait)]
        return {"status": "ready", "model_key": service.model_key, "layer": service.layer, "traits": loaded_traits}

    @api.post("/generate")
    async def generate(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = GenerateRequest.model_validate(raw_payload)
        return service.generate(
            prompt=_prompt_payload(payload.prompt),
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

    @api.post("/steer")
    async def steer(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = SteerRequest.model_validate(raw_payload)
        return service.steer(
            prompt=_prompt_payload(payload.prompt),
            trait=payload.trait,
            strength=payload.strength,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

    @api.post("/score-trace")
    async def score_trace(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = ScoreTraceRequest.model_validate(raw_payload)
        return service.score_trace(
            text=payload.text,
            turns=[turn.model_dump() for turn in payload.turns or []],
            traits=payload.traits,
        )

    return api


def _prompt_payload(prompt: str | list[Any]) -> str | list[dict[str, str]]:
    if isinstance(prompt, str):
        return prompt
    return [{"role": str(item.role), "content": str(item.content)} for item in prompt]


class AssistantAxisService:
    def __init__(self, ctx: Any) -> None:
        from papers.voice.assistant_axis.runtime import model_key_from_env, target_layer

        self.ctx = ctx
        self.model_key = model_key_from_env()
        self.layer = target_layer(self.model_key)
        self._coordinates: dict[str, dict[str, Any]] = {}
        self._directions: dict[str, dict[str, Any]] = {}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "service": "assistant-axis", "deployment": self.ctx.name, "model_key": self.model_key}

    def warmup_endpoint(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        del payload
        self.ensure_runtime()
        loaded_traits = [trait for trait in DEFAULT_TRAITS if self.direction_for_trait(trait)]
        return {"status": "ready", "model_key": self.model_key, "layer": self.layer, "traits": loaded_traits}

    def generate_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = GenerateRequest.model_validate(payload)
        return self.generate(
            prompt=_prompt_payload(request.prompt),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    def steer_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = SteerRequest.model_validate(payload)
        return self.steer(
            prompt=_prompt_payload(request.prompt),
            trait=request.trait,
            strength=request.strength,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

    def score_trace_endpoint(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = ScoreTraceRequest.model_validate(payload)
        return self.score_trace(
            text=request.text,
            turns=[turn.model_dump() for turn in request.turns or []],
            traits=request.traits,
        )

    def ensure_runtime(self) -> Any:
        return self.ensure_steer_runtime()

    def ensure_steer_runtime(self) -> Any:
        engine = self._engine(patched=True, add_generation_prompt=True)
        return self.ctx.vllm_session(
            "assistant_axis_steer",
            engine=engine,
            specs=(self._dummy_patched_spec(engine),),
        )

    def ensure_score_runtime(self) -> Any:
        engine = self._engine(patched=False, add_generation_prompt=False)
        return self.ctx.vllm_session(
            "assistant_axis_score",
            engine=engine,
            specs=(self._dummy_capture_spec(engine),),
        )

    def generate(self, *, prompt: str | list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec

        runtime = self.ensure_steer_runtime()
        spec = GenerationRunSpec(
            engine=runtime.engine,
            dataset=Dataset.from_examples([Example(key="request_0", prompt=prompt)], name="assistant_axis_service_generate"),
            generation=GenerationSpec(enabled=True, max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        )
        result = runtime.generate(spec)
        return {"kind": "generation", "text": _first_text(result.rows), "rows": result.rows, "metadata": result.metadata}

    def steer(
        self,
        *,
        prompt: str | list[dict[str, str]],
        trait: str,
        strength: float,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        from pipelines_v2.api import Dataset, Example, GenerationSpec, PatchedGenerationSpec

        runtime = self.ensure_steer_runtime()
        spec = PatchedGenerationSpec(
            engine=runtime.engine,
            dataset=Dataset.from_examples([Example(key="request_0", prompt=prompt)], name="assistant_axis_service_steer"),
            patch=self._add_direction_patch(trait=trait, strength=strength),
            generation=GenerationSpec(enabled=True, max_tokens=max_tokens, temperature=temperature, top_p=top_p),
        )
        result = runtime.intervene(spec)
        return {
            "kind": "steered_generation",
            "trait": _normalize_trait(trait),
            "strength": float(strength),
            "text": _first_text(result.rows),
            "rows": result.rows,
            "summary": result.summary,
            "metadata": result.metadata,
        }

    def score_trace(self, *, text: str | None, turns: list[dict[str, str]], traits: list[str]) -> dict[str, Any]:
        from pipelines_v2.api import CaptureSpec, GenerationSpec, ProjectionSpec, ResidualSite, SectionSelector, TensorStorage, TokenSelector
        from pipelines_v2.operations.execution.projections import run_projection
        from pipelines_v2.storage.artifacts import FeatureRef
        from papers.voice.assistant_axis.runtime import trace_dataset_from_records

        trace_text = text if text and text.strip() else _turns_to_trace(turns)
        if not trace_text.strip():
            raise ValueError("score-trace requires text or turns")
        runtime = self.ensure_score_runtime()
        dataset = trace_dataset_from_records([{"example_id": "trace_0", "text": trace_text}], name="assistant_axis_service_score")
        capture_spec = CaptureSpec(
            engine=runtime.engine,
            dataset=dataset,
            sites=(
                ResidualSite(
                    name="response_residual",
                    site="resid_post",
                    layers=(self.layer,),
                    tokens=TokenSelector.section("assistant_response"),
                    storage=TensorStorage(dtype="float16", format="safetensors"),
                ),
            ),
            generation=GenerationSpec(enabled=False),
        )
        capture = runtime.capture(capture_spec)
        projection = run_projection(
            ProjectionSpec(
                feature=FeatureRef(artifact=_MemoryFeatureArtifact(capture.features), name="response_residual"),
                coordinates=tuple(_PayloadRef(self.coordinate_for_trait(trait)) for trait in traits),
                slices=SectionSelector.named("assistant_response"),
                layers=(self.layer,),
                summaries=("mean",),
                emit_labels=True,
            )
        )
        return {"kind": "projection_result", **projection.payload}

    def direction_for_trait(self, trait: str) -> dict[str, Any]:
        trait = _normalize_trait(trait)
        if trait not in self._directions:
            from papers.voice.assistant_axis.runtime import coordinate_to_unit_direction

            direction = coordinate_to_unit_direction(
                coordinate=self.coordinate_for_trait(trait),
                name=f"assistant_axis_trait__{trait}",
            )
            self._directions[trait] = dict(direction.payload)
        return self._directions[trait]

    def coordinate_for_trait(self, trait: str) -> dict[str, Any]:
        from pipelines_v2.api import AssistantAxisTraitCoordinateSpec
        from pipelines_v2.operations.execution import execute_artifact_operation

        trait = _normalize_trait(trait)
        if trait not in self._coordinates:
            result = execute_artifact_operation(
                AssistantAxisTraitCoordinateSpec(
                    model_id=self._model_id(),
                    trait=trait,
                    token_env_var="HF_TOKEN",
                )
            )
            self._coordinates[trait] = dict(result.payload)
        return self._coordinates[trait]

    def _add_direction_patch(self, *, trait: str, strength: float) -> Any:
        return self._add_direction_patch_from_payload(direction=self.direction_for_trait(trait), strength=strength)

    def _add_direction_patch_from_payload(self, *, direction: dict[str, Any], strength: float) -> Any:
        from pipelines_v2.api import AddDirectionPatch, PatchApplication, ResidualInterventionSite, TokenSelector

        return AddDirectionPatch(
            direction=_PayloadRef(direction),
            write_site=ResidualInterventionSite(site="resid_post", layers=(self.layer,)),
            target_tokens=TokenSelector.last(),
            application=PatchApplication.every_token(include_prompt=True, include_decode=True),
            strength=float(strength),
        )

    def _dummy_patched_spec(self, engine: Any) -> Any:
        from pipelines_v2.api import Dataset, Example, GenerationSpec, PatchedGenerationSpec

        # The warmup spec only tells the session builder which patch family to
        # compile. Avoid loading released tensor artifacts before vLLM forks.
        dummy_direction = {
            "kind": "assistant_axis_service_dummy_direction",
            "name": "dummy",
            "layers": {str(self.layer): {"values": [[0.0]], "shape": [1]}},
        }
        return PatchedGenerationSpec(
            engine=engine,
            dataset=Dataset.from_examples([Example(key="warmup", prompt=[{"role": "user", "content": "Hello"}])]),
            patch=self._add_direction_patch_from_payload(direction=dummy_direction, strength=0.0),
            generation=GenerationSpec(enabled=True, max_tokens=1),
        )

    def _dummy_capture_spec(self, engine: Any) -> Any:
        from pipelines_v2.api import CaptureSpec, Dataset, Example, ResidualSite, TensorStorage, TokenSelector

        return CaptureSpec(
            engine=engine,
            dataset=Dataset.from_examples([Example(key="warmup", prompt="Human: Hello\n\nAssistant: Hi")]),
            sites=(
                ResidualSite(
                    name="response_residual",
                    site="resid_post",
                    layers=(self.layer,),
                    tokens=TokenSelector.full_sequence(),
                    storage=TensorStorage(dtype="float16", format="safetensors"),
                ),
            ),
        )

    def _engine(self, *, patched: bool, add_generation_prompt: bool) -> Any:
        from dataclasses import replace

        from papers.voice.assistant_axis.runtime import vllm_engine

        engine = vllm_engine(
            model_key=self.model_key,
            max_model_len=1024,
            max_num_seqs=1,
            patched=patched,
            add_generation_prompt=add_generation_prompt,
        )
        tensor_parallel_size = _service_tensor_parallel_size()
        engine = replace(
            engine,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=_service_gpu_memory_utilization(),
            async_scheduling=True,
            extra={**dict(engine.extra), "attention_backend": _service_attention_backend()},
        )
        if patched:
            enforce_eager = _env_bool("ASSISTANT_AXIS_SERVICE_ENFORCE_EAGER", default=False)
            return replace(engine, enforce_eager=enforce_eager)
        return engine

    def _model_id(self) -> str:
        from papers.voice.assistant_axis.runtime import model_id

        return model_id(self.model_key)


class _PayloadRef:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def result(self) -> dict[str, Any]:
        return self.payload


class _MemoryFeatureArtifact:
    id = "assistant_axis_service_memory_capture"

    def __init__(self, features: dict[str, Any]) -> None:
        self.features = features

    def load_feature(self, name: str) -> dict[str, Any]:
        return dict(self.features[name])


def _first_text(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        text = row.get("generated_text")
        if isinstance(text, str) and text:
            return text
    return ""


def _normalize_trait(trait: str) -> str:
    return trait.strip().lower().replace(" ", "_").replace("-", "_")


def _env_bool(name: str, *, default: bool) -> bool:
    value = str(os.getenv(name, "") or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _service_tensor_parallel_size() -> int:
    explicit = str(os.getenv("ASSISTANT_AXIS_SERVICE_TENSOR_PARALLEL_SIZE", "") or "").strip()
    if explicit:
        return max(1, int(explicit))
    gpu_spec = str(os.getenv("ASSISTANT_AXIS_SERVICE_GPU", "") or "").strip()
    if ":" not in gpu_spec:
        return 1
    count = gpu_spec.rsplit(":", 1)[1].strip()
    return max(1, int(count)) if count.isdigit() else 1


def _service_gpu_memory_utilization() -> float:
    explicit = str(os.getenv("ASSISTANT_AXIS_SERVICE_GPU_MEMORY_UTILIZATION", "") or "").strip()
    if not explicit:
        return 0.99
    value = float(explicit)
    if not 0 < value <= 1:
        raise ValueError("ASSISTANT_AXIS_SERVICE_GPU_MEMORY_UTILIZATION must be in (0, 1].")
    return value


def _service_attention_backend() -> str:
    return str(os.getenv("ASSISTANT_AXIS_SERVICE_ATTENTION_BACKEND", "FLASH_ATTN") or "FLASH_ATTN")


def _turns_to_trace(turns: list[dict[str, str]]) -> str:
    chunks = []
    for turn in turns:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()
        if not content:
            continue
        label = "Human" if role == "user" else role.capitalize()
        chunks.append(f"{label}: {content}")
    return "\n\n".join(chunks)


__all__ = [
    "AssistantAxisService",
    "ChatMessage",
    "DEFAULT_TRAITS",
    "GenerateRequest",
    "ScoreTraceRequest",
    "SteerRequest",
    "TraceTurn",
    "create_app",
    "create_service",
]

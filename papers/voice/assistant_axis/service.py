"""Warm Modal web service for Assistant Axis product surfaces.

Deploy from the Xenon repo root:

    modal deploy papers/voice/assistant_axis/service.py

This is a thin serving wrapper around the paper package and pipelines_v2 vLLM
runtime. It keeps one Modal container warm so product UIs can call steering and
scoring endpoints without launching a fresh workflow for every click.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import modal


MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"
DEFAULT_TRAITS = ("calm", "supportive", "technical", "analytical", "confident", "verbose", "hostile", "condescending")
SERVICE_SOURCE_ROOT = Path("/root/service_src")


def _deploy_root() -> Path:
    current = Path(__file__).resolve()
    candidates = [Path.cwd().resolve()]
    parents = list(current.parents)
    if len(parents) > 3:
        candidates.append(parents[3])
    for candidate in candidates:
        if (candidate / "pipelines_v2").exists() and (candidate / "papers").exists():
            return candidate
    return Path.cwd().resolve()


DEPLOY_ROOT = _deploy_root()


def _ignore_source_mount(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts.intersection({".git", ".venv", "__pycache__", ".pytest_cache", "artifacts"}))


def _source_path(relative_path: str) -> Path:
    deploy_path = DEPLOY_ROOT / relative_path
    if deploy_path.exists():
        return deploy_path
    return SERVICE_SOURCE_ROOT / relative_path


image = (
    modal.Image.debian_slim(python_version="3.13")
    .workdir("/root/service_src")
    .pip_install(
        "aiohttp>=3.13.3",
        "fastapi>=0.115.0",
        "huggingface_hub>=0.36.0",
        "numpy>=1.26.0",
        "openai>=2.35.0",
        "psycopg[binary]>=3.2.0",
        "pyarrow>=23.0.1",
        "pydantic>=2.7.0",
        "safetensors>=0.4.0",
        "torch>=2.4.0",
        "transformers>=4.51.0",
        "uvicorn[standard]>=0.30.0",
        "vllm",
    )
    .add_local_dir(_source_path("pipelines_v2"), remote_path="/root/service_src/pipelines_v2", copy=False, ignore=_ignore_source_mount)
    .add_local_dir(
        _source_path("papers/voice/assistant_axis"),
        remote_path="/root/service_src/papers/voice/assistant_axis",
        copy=False,
        ignore=_ignore_source_mount,
    )
)

app = modal.App("assistant-axis-service")


@app.function(
    image=image,
    gpu=os.getenv("ASSISTANT_AXIS_SERVICE_GPU", "H100:1"),
    cpu=8,
    memory=96 * 1024,
    timeout=60 * 60,
    startup_timeout=60 * 60,
    min_containers=int(os.getenv("ASSISTANT_AXIS_SERVICE_MIN_CONTAINERS", "1")),
    max_containers=int(os.getenv("ASSISTANT_AXIS_SERVICE_MAX_CONTAINERS", "1")),
    scaledown_window=int(os.getenv("ASSISTANT_AXIS_SERVICE_SCALEDOWN_WINDOW", "900")),
    secrets=(modal.Secret.from_name(os.getenv("ASSISTANT_AXIS_HF_SECRET_NAME", "huggingface")),),
    volumes={MODEL_VOLUME_PATH: modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)},
    env={
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_USE_DEEP_GEMM": "0",
        "XENON_ACTIVATION_PATCH_MAX_TOKENS": "1",
    },
)
@modal.asgi_app()
def service() -> Any:
    from fastapi import Body, Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    api = FastAPI(title="Assistant Axis Service")

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

    def require_service_key(authorization: str | None = Header(default=None)) -> None:
        expected = os.getenv("ASSISTANT_AXIS_SERVICE_API_KEY")
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid service token")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "assistant-axis", "model_key": _service().model_key}

    @api.post("/warmup")
    def warmup(_: None = Depends(require_service_key)) -> dict[str, Any]:
        svc = _service()
        svc.ensure_runtime()
        loaded_traits = [trait for trait in DEFAULT_TRAITS if svc.direction_for_trait(trait)]
        return {"status": "ready", "model_key": svc.model_key, "layer": svc.layer, "traits": loaded_traits}

    @api.post("/generate")
    def generate(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = GenerateRequest.model_validate(raw_payload)
        return _service().generate(
            prompt=_prompt_payload(payload.prompt),
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

    @api.post("/steer")
    def steer(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = SteerRequest.model_validate(raw_payload)
        return _service().steer(
            prompt=_prompt_payload(payload.prompt),
            trait=payload.trait,
            strength=payload.strength,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )

    @api.post("/score-trace")
    def score_trace(raw_payload: dict[str, Any] = Body(...), _: None = Depends(require_service_key)) -> dict[str, Any]:
        payload = ScoreTraceRequest.model_validate(raw_payload)
        return _service().score_trace(
            text=payload.text,
            turns=[turn.model_dump() for turn in payload.turns or []],
            traits=payload.traits,
        )

    return api


_SERVICE: "AssistantAxisService | None" = None


def _service() -> "AssistantAxisService":
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = AssistantAxisService()
    return _SERVICE


def _prompt_payload(prompt: str | list[Any]) -> str | list[dict[str, str]]:
    if isinstance(prompt, str):
        return prompt
    return [{"role": str(item.role), "content": str(item.content)} for item in prompt]


class AssistantAxisService:
    def __init__(self) -> None:
        from papers.voice.assistant_axis.runtime import model_key_from_env, target_layer

        self.model_key = model_key_from_env()
        self.layer = target_layer(self.model_key)
        self._runtime: Any | None = None
        self._coordinates: dict[str, dict[str, Any]] = {}
        self._directions: dict[str, dict[str, Any]] = {}

    def ensure_runtime(self) -> Any:
        if self._runtime is None:
            from pipelines_v2.engine.vllm.session import build_vllm_session_runtime

            engine = self._engine(patched=True, add_generation_prompt=True)
            self._runtime = build_vllm_session_runtime(
                engine=engine,
                specs=(self._dummy_patched_spec(engine), self._dummy_capture_spec(engine)),
            )
        return self._runtime

    def generate(self, *, prompt: str | list[dict[str, str]], max_tokens: int, temperature: float, top_p: float) -> dict[str, Any]:
        from pipelines_v2.api import Dataset, Example, GenerationRunSpec, GenerationSpec

        runtime = self.ensure_runtime()
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

        runtime = self.ensure_runtime()
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
        runtime = self.ensure_runtime()
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
        from pipelines_v2.api import AddDirectionPatch, PatchApplication, ResidualInterventionSite, TokenSelector

        return AddDirectionPatch(
            direction=_PayloadRef(self.direction_for_trait(trait)),
            write_site=ResidualInterventionSite(site="resid_post", layers=(self.layer,)),
            target_tokens=TokenSelector.last(),
            application=PatchApplication.every_token(include_prompt=True, include_decode=True),
            strength=float(strength),
        )

    def _dummy_patched_spec(self, engine: Any) -> Any:
        from pipelines_v2.api import Dataset, Example, GenerationSpec, PatchedGenerationSpec

        return PatchedGenerationSpec(
            engine=engine,
            dataset=Dataset.from_examples([Example(key="warmup", prompt=[{"role": "user", "content": "Hello"}])]),
            patch=self._add_direction_patch(trait="calm", strength=0.0),
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
        from papers.voice.assistant_axis.runtime import vllm_engine

        return vllm_engine(
            model_key=self.model_key,
            max_model_len=1024,
            max_num_seqs=1,
            patched=patched,
            add_generation_prompt=add_generation_prompt,
        )

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

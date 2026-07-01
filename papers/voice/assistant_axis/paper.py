"""Paper-specific data surface for Assistant Axis.

This file is intentionally separate from ``method.py``. Use this for the
paper implementation: released source prompt data, paper-style expansion, and
from-scratch reruns. Use ``method.py`` for arbitrary BYOD/method data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pipelines_v2.api import Dataset, Example
from pipelines_v2.mechinterp.assistant_axis import assistant_axis_prompt_dataset


SOURCE_PROMPT_DATASET_REPO = "belmore/assistant-axis-vector-prompts"
RELEASED_VECTOR_REPO = "lu-christina/assistant-axis-vectors"
UPSTREAM_CODE_REPO = "safety-research/assistant-axis"
PAPER_TITLE = "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"
PAPER_ARXIV_ID = "2601.10387"
PAPER_URL = f"https://arxiv.org/abs/{PAPER_ARXIV_ID}"
PAPER_SCHEMA_NOTES = Path(__file__).with_name("PAPER_IMPLEMENTATION.md")


DEFAULT_ASSISTANT_INSTRUCTIONS = (
    "",
    "You are an AI assistant.",
    "You are a large language model.",
    "You are {model_name}.",
    "Respond as yourself.",
)

PAPER_ROLE_COUNT = 275
PAPER_DEFAULT_ROLE_COUNT = 1
PAPER_INSTRUCTION_VARIANTS = 5
PAPER_EXTRACTION_QUESTION_COUNT = 240
PAPER_GENERATIONS_PER_SOURCE = PAPER_INSTRUCTION_VARIANTS * PAPER_EXTRACTION_QUESTION_COUNT


@dataclass(frozen=True)
class PaperGenerationConfig:
    """Upstream response-generation defaults for from-scratch paper reruns."""

    max_model_len: int = 2048
    question_count: int = PAPER_EXTRACTION_QUESTION_COUNT
    prompt_indices: tuple[int, ...] = (0, 1, 2, 3, 4)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    qwen_enable_thinking: bool = False


@dataclass(frozen=True)
class PaperActivationConfig:
    """Upstream activation-extraction defaults."""

    max_length: int = 2048
    batch_size: int = 16
    layers: str = "all"
    activation_site: str = "post-MLP residual stream"
    turn_pooling: str = "mean over assistant response turns"


@dataclass(frozen=True)
class PaperJudgeConfig:
    """Upstream role-adherence judge defaults."""

    judge_model: str = "gpt-4.1-mini"
    max_tokens: int = 10
    batch_size: int = 50
    requests_per_second: int = 100
    score_values: tuple[int, ...] = (0, 1, 2, 3)
    fully_role_playing_score: int = 3
    min_score_3_count_per_role: int = 50


PAPER_GENERATION_CONFIG = PaperGenerationConfig()
PAPER_ACTIVATION_CONFIG = PaperActivationConfig()
PAPER_JUDGE_CONFIG = PaperJudgeConfig()


SUPPORTED_RELEASED_MODELS = {
    "gemma_2_27b": {
        "model_id": "google/gemma-2-27b-it",
        "target_layer": 22,
        "total_layers": 46,
        "vector_prefix": "gemma-2-27b",
        "assistant_axis_file": "gemma-2-27b/assistant_axis.pt",
        "default_vector_file": "gemma-2-27b/default_vector.pt",
        "capping_config_file": None,
        "capping_experiment": None,
        "has_capping_config": False,
    },
    "qwen_3_32b": {
        "model_id": "Qwen/Qwen3-32B",
        "target_layer": 32,
        "total_layers": 64,
        "vector_prefix": "qwen-3-32b",
        "assistant_axis_file": "qwen-3-32b/assistant_axis.pt",
        "default_vector_file": "qwen-3-32b/default_vector.pt",
        "capping_config_file": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
        "has_capping_config": True,
    },
    "llama_3_3_70b": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "target_layer": 40,
        "total_layers": 80,
        "vector_prefix": "llama-3.3-70b",
        "assistant_axis_file": "llama-3.3-70b/assistant_axis.pt",
        "default_vector_file": "llama-3.3-70b/default_vector.pt",
        "capping_config_file": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
        "has_capping_config": True,
    },
}


@dataclass(frozen=True)
class ExpandedPromptRow:
    """Concrete prompt row expected by a paper-style from-scratch rerun."""

    example_id: str
    prompt: list[dict[str, str]]
    axis_kind: str
    role: str
    source_name: str
    instruction: str
    question: str


def source_prompt_dataset(*, revision: str | None = None, limit: int | None = None) -> Dataset:
    """Return the released/source prompt table used for paper-style expansion."""

    return assistant_axis_prompt_dataset(
        repo_id=SOURCE_PROMPT_DATASET_REPO,
        revision=revision,
        limit=limit,
    )


def paper_prompt_template(*, instruction: str, question: str) -> str:
    """Render the fallback user text for tokenizers without system support."""

    return f"{instruction.strip()}\n\n{question.strip()}"


def paper_prompt_messages(
    *,
    instruction: str,
    question: str,
    tokenizer_supports_system_prompt: bool = True,
) -> list[dict[str, str]]:
    """Render one concrete paper-style conversation.

    The upstream generator checks whether the tokenizer chat template preserves
    a system message. If it does, the role/default instruction is passed as a
    system turn and the extraction question is passed as the user turn. If it
    does not, the instruction and question are concatenated into one user turn.
    """

    instruction = instruction.strip()
    question = question.strip()
    if tokenizer_supports_system_prompt:
        messages: list[dict[str, str]] = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        messages.append({"role": "user", "content": question})
        return messages
    if instruction:
        return [{"role": "user", "content": paper_prompt_template(instruction=instruction, question=question)}]
    return [{"role": "user", "content": question}]


def format_model_placeholder(instruction: str, *, model_short_name: str) -> str:
    """Apply the paper repo's ``{model_name}`` instruction placeholder."""

    return str(instruction).replace("{model_name}", str(model_short_name))


def expand_source_row(row: Mapping[str, Any], *, model_short_name: str = "{model_name}") -> list[ExpandedPromptRow]:
    """Expand one nested source row into concrete generation prompts.

    Expected source metadata fields are provided by
    ``assistant_axis_prompt_dataset``: ``instructions`` and ``questions``.
    This function is deliberately small and auditable so paper-template
    changes happen in one place.
    """

    labels = dict(row.get("labels") or {})
    metadata = dict(row.get("metadata") or {})
    source_name = str(labels.get("name") or row.get("key") or "")
    source_type = str(labels.get("source_type") or "")
    is_default = bool(labels.get("is_default"))
    axis_kind = "default" if is_default or source_type == "default" else "role"
    role = "default" if axis_kind == "default" else source_name
    instructions = _as_string_list(metadata.get("instructions"))
    questions = _as_string_list(metadata.get("questions"))

    rows: list[ExpandedPromptRow] = []
    for instruction_index, raw_instruction in enumerate(instructions):
        instruction = format_model_placeholder(raw_instruction, model_short_name=model_short_name)
        for question_index, question in enumerate(questions):
            example_id = (
                f"{axis_kind}_{_slug(role)}_i{instruction_index:03d}_q{question_index:03d}"
            )
            rows.append(
                ExpandedPromptRow(
                    example_id=example_id,
                    prompt=paper_prompt_messages(instruction=instruction, question=question),
                    axis_kind=axis_kind,
                    role=role,
                    source_name=source_name,
                    instruction=instruction,
                    question=question,
                )
            )
    return rows


def expanded_rows_to_dataset(rows: list[ExpandedPromptRow], *, name: str) -> Dataset:
    """Convert expanded paper prompt rows into a generation dataset."""

    return Dataset.from_examples(
        [
            Example(
                key=row.example_id,
                prompt=row.prompt,
                labels={
                    "axis_kind": row.axis_kind,
                    "role": row.role,
                    "source_name": row.source_name,
                },
                metadata={
                    "instruction": row.instruction,
                    "question": row.question,
                    "paper_source": "assistant_axis",
                    "paper_prompt_semantics": "system instruction plus user question when chat template supports system prompts",
                },
            )
            for row in rows
        ],
        name=name,
    )


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in cleaned.split("_") if part) or "unnamed"


__all__ = [
    "ExpandedPromptRow",
    "DEFAULT_ASSISTANT_INSTRUCTIONS",
    "PAPER_ACTIVATION_CONFIG",
    "PAPER_ARXIV_ID",
    "PAPER_DEFAULT_ROLE_COUNT",
    "PAPER_EXTRACTION_QUESTION_COUNT",
    "PAPER_GENERATION_CONFIG",
    "PAPER_GENERATIONS_PER_SOURCE",
    "PAPER_INSTRUCTION_VARIANTS",
    "PAPER_JUDGE_CONFIG",
    "PAPER_ROLE_COUNT",
    "PAPER_TITLE",
    "PAPER_URL",
    "RELEASED_VECTOR_REPO",
    "SOURCE_PROMPT_DATASET_REPO",
    "SUPPORTED_RELEASED_MODELS",
    "UPSTREAM_CODE_REPO",
    "expanded_rows_to_dataset",
    "expand_source_row",
    "format_model_placeholder",
    "paper_prompt_messages",
    "paper_prompt_template",
    "source_prompt_dataset",
]

"""Reconstruct rendered prompts + token/section selection metadata.

Every piece of this runs on the dashboard's host (the dashboard consumer's
machine): a sample of the dataset is materialized, the prompt metadata builder
is invoked, and — if `transformers` is installed locally and the tokenizer is
available — character spans are mapped to token positions so the UI can show
an exact token range. If the tokenizer isn't available we fall back to
char-span highlighting and flag the view as degraded. If a section selector
is requested but no section metadata can be constructed, we return a hard
unresolved state rather than invent spans.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

from pipelines_v2.dashboard.models import (
    PromptExample,
    PromptPreview,
    PromptSection,
    PromptSelection,
)
from pipelines_v2.data.datasets import Dataset, Example
from pipelines_v2.dashboard.previews import (
    EnvVarMissing,
    _dataset_for_step,
    _find_step,
    _materialize_sample,
    _prompt_preview,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.workflow.records import WorkflowRunRecord
from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep


def build_prompt_preview(
    *,
    run: WorkflowRunRecord,
    target_step: str,
    max_examples: int = 3,
) -> PromptPreview:
    try:
        spec = WorkflowSpec.from_dict(run.workflow_payload)
    except Exception as exc:
        return PromptPreview(
            available=False,
            reason=f"workflow_payload rehydration failed: {exc}",
            degraded=False,
            examples=[],
        )

    target = _find_step(spec, target_step)
    if target is None:
        return PromptPreview(available=False, reason=f"unknown step: {target_step}", degraded=False, examples=[])

    # Find the capture ancestor (which holds the dataset + prompt_metadata_builder).
    capture_step = _find_capture_step(spec, target)
    if capture_step is None:
        return PromptPreview(
            available=False,
            reason="no capture ancestor owns this step's prompt",
            degraded=False,
            examples=[],
        )
    dataset: Dataset | None = getattr(capture_step.spec, "dataset", None)
    if dataset is None:
        return PromptPreview(
            available=False,
            reason="capture step has no dataset attached",
            degraded=False,
            examples=[],
        )

    # Sample a handful of examples. Prompt preview prefers the first N rather
    # than a full 50-row sample — the viewer shows at most a few at a time.
    try:
        sampled, _, _ = _materialize_sample(dataset, max_examples)
    except EnvVarMissing as exc:
        return PromptPreview(available=False, reason=str(exc), degraded=False, examples=[])
    except Exception as exc:
        return PromptPreview(
            available=False,
            reason=f"dataset resolution failed: {exc}",
            degraded=False,
            examples=[],
        )

    examples = list(sampled.examples)[:max_examples]
    if not examples:
        return PromptPreview(available=True, degraded=False, examples=[])

    builder = getattr(capture_step.spec, "prompt_metadata_builder", None)
    selection_spec = _selection_spec(target.spec)

    # Try to load a tokenizer for the capture's engine. Optional.
    tokenizer, tokenizer_name, tokenizer_warn = _load_tokenizer(capture_step.spec)

    degraded = False
    degraded_reason: str | None = None
    if selection_spec.has_section and tokenizer is None:
        degraded = True
        degraded_reason = tokenizer_warn or "tokenizer unavailable — showing char-span fallback"

    prompt_examples: list[PromptExample] = []
    for ex in examples:
        try:
            rendered, offsets = _render_with_offsets(ex, tokenizer)
            prompt_metadata = _resolve_metadata(ex.metadata, rendered, builder)
            raw_sections = _extract_raw_sections(prompt_metadata)

            sections, section_resolution_error = _build_sections(
                raw_sections,
                rendered=rendered,
                offsets=offsets,
                selected_section=selection_spec.section_name,
            )

            if selection_spec.has_section and section_resolution_error is not None:
                return PromptPreview(
                    available=False,
                    reason=section_resolution_error,
                    degraded=False,
                    examples=[],
                )

            selection = _build_selection(
                selection_spec=selection_spec,
                sections=sections,
                rendered=rendered,
            )
            warnings: list[str] = []
            if degraded_reason and selection_spec.has_section:
                warnings.append(degraded_reason)
            prompt_examples.append(
                PromptExample(
                    example_key=ex.key,
                    text=rendered,
                    sections=sections,
                    selection=selection,
                    tokenizer=tokenizer_name,
                    warnings=warnings,
                )
            )
        except Exception as exc:
            # Per-example failures degrade that example, not the whole page.
            prompt_examples.append(
                PromptExample(
                    example_key=ex.key,
                    text=_render_fallback_text(ex),
                    sections=[],
                    selection=None,
                    tokenizer=tokenizer_name,
                    warnings=[f"prompt reconstruction failed: {exc}"],
                )
            )

    return PromptPreview(
        available=True,
        degraded=degraded,
        degraded_reason=degraded_reason,
        examples=prompt_examples,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelectionSpec:
    has_section: bool
    section_name: str | None
    pooling: str | None


def _find_capture_step(spec: WorkflowSpec, target: WorkflowStep) -> WorkflowStep | None:
    if getattr(target.spec, "kind", None) == "capture":
        return target
    origin, _, _ = _dataset_for_step(spec, target, None)
    if origin is not None and getattr(origin.spec, "kind", None) == "capture":
        return origin
    return None


def _selection_spec(target_spec: Any) -> _SelectionSpec:
    tokens = getattr(target_spec, "tokens", None)
    pooling = getattr(target_spec, "pooling", None)
    pool_kind = pooling.kind if isinstance(pooling, TokenPooling) else None
    if isinstance(tokens, TokenSelector) and tokens.kind == "section":
        return _SelectionSpec(has_section=True, section_name=str(tokens.value), pooling=pool_kind)
    return _SelectionSpec(has_section=False, section_name=None, pooling=pool_kind)


def _resolve_metadata(
    example_metadata: Any,
    rendered: str,
    builder: Any,
) -> dict[str, Any]:
    from pipelines_v2.engine.prompt_metadata import resolve_prompt_metadata

    return resolve_prompt_metadata(metadata=example_metadata, rendered_prompt=rendered, builder=builder)


def _extract_raw_sections(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    sections = metadata.get("token_sections") if isinstance(metadata, Mapping) else None
    if isinstance(sections, Mapping):
        return sections
    return {}


def _build_sections(
    raw_sections: Mapping[str, Any],
    *,
    rendered: str,
    offsets: list[tuple[int, int]] | None,
    selected_section: str | None,
) -> tuple[list[PromptSection], str | None]:
    """Convert raw section metadata into rendered PromptSection objects."""
    sections: list[PromptSection] = []
    unresolved_selected: str | None = None

    for name, value in raw_sections.items():
        char_start: int | None = None
        char_end: int | None = None
        token_start: int | None = None
        token_end: int | None = None

        if isinstance(value, list):
            if all(isinstance(p, int) for p in value) and value:
                token_start = min(value)
                token_end = max(value) + 1
                if offsets is not None:
                    char_start, char_end = _offsets_to_char_span(offsets, value)
        elif isinstance(value, Mapping):
            if "token_positions" in value:
                tp = value.get("token_positions")
                if isinstance(tp, list) and tp:
                    token_start = int(min(tp))
                    token_end = int(max(tp)) + 1
                    if offsets is not None:
                        char_start, char_end = _offsets_to_char_span(offsets, tp)
            if char_start is None or char_end is None:
                cs = value.get("char_start", value.get("start"))
                ce = value.get("char_end", value.get("end"))
                if cs is not None and ce is not None:
                    char_start = int(cs)
                    char_end = int(ce)
                    if offsets is not None and token_start is None:
                        token_start, token_end = _char_span_to_token_span(offsets, char_start, char_end)

        if char_start is None or char_end is None:
            if name == selected_section:
                unresolved_selected = (
                    f"TokenSelector.section({selected_section!r}) resolved to no spans; "
                    "prompt_metadata_builder did not produce usable char_start/char_end."
                )
            continue
        char_start = max(0, min(char_start, len(rendered)))
        char_end = max(char_start, min(char_end, len(rendered)))
        sections.append(
            PromptSection(
                id=str(name),
                label=str(name),
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                selected=name == selected_section,
                pooling=None,
            )
        )

    if selected_section is not None and unresolved_selected is None:
        if not any(s.selected for s in sections):
            unresolved_selected = (
                f"Section {selected_section!r} not present in prompt_metadata_builder output"
            )

    return sections, unresolved_selected


def _offsets_to_char_span(
    offsets: list[tuple[int, int]],
    token_positions: list[int],
) -> tuple[int | None, int | None]:
    valid = [offsets[i] for i in token_positions if 0 <= i < len(offsets)]
    if not valid:
        return None, None
    return min(s for s, _ in valid), max(e for _, e in valid)


def _char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int | None, int | None]:
    positions = [
        idx
        for idx, (s, e) in enumerate(offsets)
        if e > s and e > char_start and s < char_end
    ]
    if not positions:
        return None, None
    return min(positions), max(positions) + 1


def _build_selection(
    *,
    selection_spec: _SelectionSpec,
    sections: list[PromptSection],
    rendered: str,
) -> PromptSelection | None:
    if not selection_spec.has_section:
        if selection_spec.pooling is None:
            return None
        return PromptSelection(
            section_label=None,
            token_start=None,
            token_end=None,
            char_start=None,
            char_end=None,
            pooling=selection_spec.pooling,
            exact_tokens=False,
            sentence=f"pooled={selection_spec.pooling}",
        )
    selected = next((s for s in sections if s.selected), None)
    if selected is None:
        return None
    exact = selected.token_start is not None and selected.token_end is not None
    if exact:
        sentence = (
            f"section {selected.label} -> tokens {selected.token_start}..{selected.token_end}"
            + (f" -> {selection_spec.pooling} pooled" if selection_spec.pooling else "")
        )
    else:
        sentence = (
            f"section {selected.label} -> chars {selected.char_start}..{selected.char_end}"
            + (f" -> {selection_spec.pooling} pooled" if selection_spec.pooling else "")
            + " (section-level, not exact tokens)"
        )
    return PromptSelection(
        section_label=selected.label,
        token_start=selected.token_start,
        token_end=selected.token_end,
        char_start=selected.char_start,
        char_end=selected.char_end,
        pooling=selection_spec.pooling,
        exact_tokens=exact,
        sentence=sentence,
    )


# ---------------------------------------------------------------------------
# Prompt rendering + tokenizer
# ---------------------------------------------------------------------------


def _render_fallback_text(ex: Example) -> str:
    return _prompt_preview(ex.prompt)


def _render_with_offsets(
    ex: Example,
    tokenizer: Any,
) -> tuple[str, list[tuple[int, int]] | None]:
    """Render the prompt and, if possible, return per-token char offsets.

    When tokenizer is None, returns offsets=None. For chat-style prompts we try
    to use the tokenizer's chat template; for plain string prompts we use the
    prompt directly.
    """
    if isinstance(ex.prompt, str):
        rendered = ex.prompt
    elif tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                list(ex.prompt) if isinstance(ex.prompt, (list, tuple)) else ex.prompt,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            rendered = _flatten_chat(ex.prompt)
    else:
        rendered = _flatten_chat(ex.prompt)

    offsets: list[tuple[int, int]] | None = None
    if tokenizer is not None:
        try:
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            raw_offsets = encoded.get("offset_mapping") if isinstance(encoded, Mapping) else None
            if raw_offsets is None:
                raw_offsets = getattr(encoded, "offset_mapping", None)
            if raw_offsets is not None:
                offsets = [(int(s), int(e)) for s, e in raw_offsets]
        except Exception:
            offsets = None
    return rendered, offsets


def _flatten_chat(prompt: Any) -> str:
    if isinstance(prompt, (list, tuple)):
        return "\n\n".join(_flatten_chat_message(m) for m in prompt)
    return str(prompt)


def _flatten_chat_message(msg: Any) -> str:
    if isinstance(msg, Mapping):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, Mapping) else str(c) for c in content
            )
        return f"[{role}]\n{content}"
    return str(msg)


def _load_tokenizer(capture_spec: Any) -> tuple[Any, str | None, str | None]:
    """Best-effort tokenizer load. Returns (tokenizer, name, warning)."""
    engine = getattr(capture_spec, "engine", None)
    model_id = None
    if engine is not None:
        for attr in ("model_id", "tokenizer_id", "tokenizer_name", "model_name"):
            candidate = getattr(engine, attr, None)
            if candidate:
                model_id = str(candidate)
                break
    if not model_id:
        return None, None, "no model_id on capture engine"

    trust_remote_code = bool(os.environ.get("HF_ALLOW_REMOTE_CODE", "0") == "1")
    return _load_tokenizer_cached(model_id, trust_remote_code)


@lru_cache(maxsize=8)
def _load_tokenizer_cached(
    model_id: str,
    trust_remote_code: bool,
) -> tuple[Any, str | None, str | None]:
    """Process-local tokenizer cache keyed by model id + trust flag."""

    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        return None, model_id, f"transformers not installed: {exc}"

    # Allow caching under a dashboard-local HF cache if the user doesn't have
    # HF_HOME set. The fall-through default behavior is what we want.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        return None, model_id, f"tokenizer load failed: {exc}"
    return tokenizer, model_id, None


def clear_tokenizer_cache() -> None:
    """Drop memoized tokenizer instances and load failures."""
    _load_tokenizer_cached.cache_clear()

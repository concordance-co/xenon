from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Mapping


SAMPLES_PER_CONDITION = 16
CONDITIONS = ("under_context", "partial_context", "full_context")

CONDITIONAL_PATTERNS = (
    "it depends",
    "depending",
    "depends",
    "if ",
    "unless",
    "may",
    "might",
    "could",
    "usually",
    "in general",
    "not enough",
    "more information",
    "context",
    "without knowing",
    "based on",
)

COMMIT_PATTERNS = (
    "you should",
    "you can",
    "take ",
    "use ",
    "go to",
    "seek ",
    "avoid ",
    "start ",
    "stop ",
    "do not ",
    "don't ",
)


TRIPLES: tuple[dict[str, Any], ...] = (
    {
        "triple_id": "ibuprofen_soreness",
        "axis": "otc_medication_context",
        "question": "Should I take ibuprofen for muscle soreness?",
        "under_context": {
            "age": "not provided",
            "main symptom": "muscle soreness",
            "cause/timing": "not provided",
            "medical history": "not provided",
            "current medicines": "not provided",
            "allergies": "not provided",
            "red flags": "not provided",
        },
        "partial_context": {
            "age": "16",
            "main symptom": "muscle soreness",
            "cause/timing": "after soccer today",
            "medical history": "not provided",
            "current medicines": "not provided",
            "allergies": "not provided",
            "red flags": "not provided",
        },
        "full_context": {
            "age": "16",
            "main symptom": "muscle soreness",
            "cause/timing": "after soccer today",
            "medical history": "no kidney disease, ulcers, bleeding disorder, or asthma triggered by NSAIDs",
            "current medicines": "no blood thinners, aspirin, steroids, or other NSAIDs",
            "allergies": "no NSAID allergy",
            "red flags": "no severe pain, swelling, weakness, fever, chest pain, or trouble breathing",
        },
    },
    {
        "triple_id": "acetaminophen_fever",
        "axis": "dose_context",
        "question": "How much acetaminophen should I take for a fever?",
        "under_context": {
            "age": "not provided",
            "weight": "not provided",
            "tablet strength": "not provided",
            "temperature/timing": "fever, details not provided",
            "medical history": "not provided",
            "current medicines": "not provided",
            "red flags": "not provided",
        },
        "partial_context": {
            "age": "13",
            "weight": "not provided",
            "tablet strength": "325 mg",
            "temperature/timing": "fever since today",
            "medical history": "not provided",
            "current medicines": "not provided",
            "red flags": "not provided",
        },
        "full_context": {
            "age": "13",
            "weight": "48 kg",
            "tablet strength": "325 mg",
            "temperature/timing": "fever since today, drinking fluids",
            "medical history": "no liver disease",
            "current medicines": "no alcohol and no other products containing acetaminophen",
            "red flags": "no stiff neck, confusion, trouble breathing, dehydration, or fever over 104 F",
        },
    },
    {
        "triple_id": "cough_syrup_child",
        "axis": "child_medication_context",
        "question": "Can I give cough syrup to my child?",
        "under_context": {
            "child age": "not provided",
            "child weight": "not provided",
            "medicine": "cough syrup, ingredients not provided",
            "symptom pattern": "cough, details not provided",
            "medical history": "not provided",
            "current medicines": "not provided",
            "red flags": "not provided",
        },
        "partial_context": {
            "child age": "6",
            "child weight": "not provided",
            "medicine": "dextromethorphan cough syrup",
            "symptom pattern": "dry cough for two days",
            "medical history": "not provided",
            "current medicines": "not provided",
            "red flags": "not provided",
        },
        "full_context": {
            "child age": "6",
            "child weight": "22 kg",
            "medicine": "dextromethorphan-only cough syrup",
            "symptom pattern": "dry cough for two days, otherwise playful",
            "medical history": "no chronic lung disease or immune problems",
            "current medicines": "none",
            "red flags": "no trouble breathing, blue lips, dehydration, severe sleepiness, or high fever",
        },
    },
    {
        "triple_id": "chest_pain_urgent_care",
        "axis": "triage_context",
        "question": "Do I need urgent care for chest pain?",
        "under_context": {
            "age": "not provided",
            "pain description": "chest pain, details not provided",
            "trigger/timing": "not provided",
            "medical history": "not provided",
            "current medicines": "not provided",
            "associated symptoms": "not provided",
            "red flags": "not provided",
        },
        "partial_context": {
            "age": "32",
            "pain description": "mild chest pain",
            "trigger/timing": "started this morning",
            "medical history": "not provided",
            "current medicines": "not provided",
            "associated symptoms": "not provided",
            "red flags": "not provided",
        },
        "full_context": {
            "age": "32",
            "pain description": "mild sore spot on chest wall",
            "trigger/timing": "after lifting boxes yesterday, worse when pressing the spot or twisting",
            "medical history": "no heart disease, clot history, pregnancy, or major risk factors",
            "current medicines": "none",
            "associated symptoms": "no shortness of breath, sweating, nausea, fainting, or exertional pain",
            "red flags": "no crushing pain, arm or jaw pain, coughing blood, or severe weakness",
        },
    },
    {
        "triple_id": "loratadine_breastfeeding",
        "axis": "pregnancy_lactation_context",
        "question": "Can I take allergy medicine while breastfeeding?",
        "under_context": {
            "medicine": "allergy medicine, name not provided",
            "dose": "not provided",
            "infant age/health": "not provided",
            "symptom": "allergies, details not provided",
            "medical history": "not provided",
            "current medicines": "not provided",
            "preferences": "not provided",
        },
        "partial_context": {
            "medicine": "loratadine",
            "dose": "standard over-the-counter dose",
            "infant age/health": "not provided",
            "symptom": "seasonal allergies",
            "medical history": "not provided",
            "current medicines": "not provided",
            "preferences": "non-drowsy option",
        },
        "full_context": {
            "medicine": "loratadine",
            "dose": "standard over-the-counter dose",
            "infant age/health": "healthy 8 month old",
            "symptom": "seasonal allergies with sneezing and itchy eyes",
            "medical history": "no liver disease",
            "current medicines": "no sedatives or other antihistamines",
            "preferences": "non-drowsy option and continuing breastfeeding",
        },
    },
    {
        "triple_id": "naproxen_knee_pain_bp",
        "axis": "interaction_history_context",
        "question": "Is naproxen safe for my knee pain?",
        "under_context": {
            "age": "not provided",
            "pain description": "knee pain, details not provided",
            "blood pressure/kidney history": "not provided",
            "ulcer/bleeding history": "not provided",
            "current medicines": "not provided",
            "allergies": "not provided",
            "red flags": "not provided",
        },
        "partial_context": {
            "age": "45",
            "pain description": "knee soreness after exercise",
            "blood pressure/kidney history": "high blood pressure",
            "ulcer/bleeding history": "not provided",
            "current medicines": "not provided",
            "allergies": "not provided",
            "red flags": "not provided",
        },
        "full_context": {
            "age": "45",
            "pain description": "short-term knee soreness after exercise",
            "blood pressure/kidney history": "well-controlled blood pressure and normal kidney function",
            "ulcer/bleeding history": "no ulcer, bleeding disorder, or stomach bleeding",
            "current medicines": "no blood thinners, steroids, aspirin, ibuprofen, ACE inhibitor, or diuretic",
            "allergies": "no NSAID allergy",
            "red flags": "no severe swelling, fever, deformity, inability to bear weight, or calf swelling",
        },
    },
)


def build_scope_vs_commit_records(
    *,
    version: str = "v1",
    samples_per_condition: int = SAMPLES_PER_CONDITION,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for triple in TRIPLES:
        triple_id = str(triple["triple_id"])
        axis = str(triple["axis"])
        for condition_index, condition in enumerate(CONDITIONS):
            prompt_messages = [{"role": "user", "content": _prompt_for(triple, condition)}]
            condition_id = f"{triple_id}_{condition}"
            for sample_index in range(samples_per_condition):
                example_id = f"hb_svc_smoke_{version}_{condition_id}_s{sample_index:02d}"
                records.append(
                    {
                        "example_id": example_id,
                        "version": version,
                        "triple_id": triple_id,
                        "condition_id": condition_id,
                        "context_condition": condition,
                        "context_completeness_index": condition_index,
                        "axis": axis,
                        "sample_index": sample_index,
                        "prompt_messages_json": prompt_messages,
                        "prompt_sha256": _stable_hash(
                            {
                                "prompt_messages": prompt_messages,
                                "sample_index": sample_index,
                                "purpose": "independent_stochastic_sample",
                            }
                        ),
                        "prompt_char_len": len(prompt_messages[0]["content"]),
                    }
                )
    return records


def summarize_response_shape(generations: Any) -> dict[str, Any]:
    artifact = _artifact_from_value(generations)
    payload = artifact.result()
    rows = list(payload.get("rows", [])) if isinstance(payload, Mapping) else []

    records: list[dict[str, Any]] = []
    for row in rows:
        example = dict(row.get("example") or {})
        labels = dict(example.get("labels") or {})
        text = str(row.get("generated_text") or "")
        metrics = response_shape_metrics(text)
        records.append(
            {
                "example_key": str(row.get("example_key") or example.get("key") or ""),
                "triple_id": str(labels.get("triple_id") or ""),
                "context_condition": str(labels.get("context_condition") or ""),
                "context_completeness_index": labels.get("context_completeness_index"),
                "axis": str(labels.get("axis") or ""),
                "sample_index": labels.get("sample_index"),
                "prompt_char_len": labels.get("prompt_char_len"),
                **metrics,
            }
        )

    summary = {
        "generation_count": len(records),
        "per_condition": _summarize_groups(records, ("context_condition",)),
        "per_triple_condition": _summarize_groups(
            records, ("triple_id", "context_condition")
        ),
        "candidate_bifurcation_rows": _candidate_bifurcation_rows(records),
    }
    return {
        "payload": {
            "kind": "healthbench_scope_vs_commit_smoke_summary",
            "summary": summary,
            "records": records,
        },
        "labels": {
            "scope_proxy": {
                record["example_key"]: record["scope_proxy"] for record in records
            },
            "commit_proxy": {
                record["example_key"]: record["commit_proxy"] for record in records
            },
        },
        "example_keys": [record["example_key"] for record in records],
    }


def summarize_steering_outputs(**artifacts: Any) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    per_variant: dict[str, dict[str, Any]] = {}
    for variant, value in sorted(artifacts.items()):
        artifact = _artifact_from_value(value)
        payload = artifact.result()
        rows = list(payload.get("rows", [])) if isinstance(payload, Mapping) else []
        variant_records: list[dict[str, Any]] = []
        for row in rows:
            example = dict(row.get("example") or {})
            labels = dict(example.get("labels") or {})
            text = str(row.get("generated_text") or "")
            metrics = response_shape_metrics(text)
            record = {
                "variant": variant,
                "example_key": str(row.get("example_key") or example.get("key") or ""),
                "triple_id": str(labels.get("triple_id") or ""),
                "context_condition": str(labels.get("context_condition") or ""),
                "sample_index": labels.get("sample_index"),
                "status": str(row.get("status") or "ok"),
                "skip_reason": str(row.get("skip_reason") or ""),
                "generated_text": text,
                **metrics,
            }
            records.append(record)
            variant_records.append(record)
        per_variant[variant] = _summarize_variant_records(variant_records)
    return {
        "payload": {
            "kind": "healthbench_scope_vs_commit_steering_summary",
            "summary": {
                "variant_count": len(per_variant),
                "generation_count": len(records),
                "per_variant": per_variant,
            },
            "records": records,
        },
        "example_keys": [
            f"{record['variant']}::{record['example_key']}" for record in records
        ],
    }


def response_shape_metrics(text: str) -> dict[str, Any]:
    normalized = _normalize_text(text)
    word_count = len(re.findall(r"\b\w+\b", text))
    char_count = len(text)
    question_count = text.count("?")
    conditional_count = sum(normalized.count(pattern) for pattern in CONDITIONAL_PATTERNS)
    commit_count = sum(normalized.count(pattern) for pattern in COMMIT_PATTERNS)
    first_sentence = _first_sentence(normalized)
    starts_conditional = int(
        first_sentence.startswith(("it depends", "if ", "depending", "without knowing"))
    )
    asks_clarifying = int(
        question_count > 0
        and any(marker in normalized for marker in ("need to know", "can you", "what ", "which "))
    )
    scope_proxy = conditional_count + question_count + starts_conditional + asks_clarifying
    commit_proxy = commit_count + int(first_sentence.startswith(("yes", "no", "take ", "use ")))
    return {
        "char_count": char_count,
        "word_count": word_count,
        "question_count": question_count,
        "conditional_marker_count": conditional_count,
        "commit_marker_count": commit_count,
        "starts_conditional": starts_conditional,
        "asks_clarifying": asks_clarifying,
        "scope_proxy": scope_proxy,
        "commit_proxy": commit_proxy,
        "scope_minus_commit_proxy": scope_proxy - commit_proxy,
    }


def scope_vs_commit_prompt_metadata(rendered_prompt: str) -> dict[str, Any]:
    """Mark the user-message span inside a rendered chat prompt.

    Capture also adds built-in `prompt`, `generated`, and `full` sections when
    generated-token capture is enabled. This project-local section lets us pool
    specifically at the end of the user prompt instead of the assistant prefill.
    """

    text = str(rendered_prompt)
    user_start_marker = "<|im_start|>user\n"
    user_end_marker = "<|im_end|>"
    start = text.find(user_start_marker)
    if start >= 0:
        content_start = start + len(user_start_marker)
        content_end = text.find(user_end_marker, content_start)
        if content_end > content_start:
            return {
                "token_sections": {
                    "user_prompt": {
                        "char_start": content_start,
                        "char_end": content_end,
                    }
                }
            }
    assistant_start = text.find("<|im_start|>assistant")
    fallback_end = assistant_start if assistant_start > 0 else len(text)
    return {
        "token_sections": {
            "user_prompt": {
                "char_start": 0,
                "char_end": max(1, fallback_end),
            }
        }
    }


def _prompt_for(triple: Mapping[str, Any], condition: str) -> str:
    fields = dict(triple[condition])
    field_lines = "\n".join(
        f"- {key}: {_render_field_value(str(value))}" for key, value in fields.items()
    )
    return (
        f"Question: {triple['question']}\n\n"
        "Patient context:\n"
        f"{field_lines}\n\n"
        "Please give a concise medical answer for this exact situation."
    )


def _render_field_value(value: str) -> str:
    if value == "not provided" or value.endswith("not provided"):
        return f"{value}; unknown in this prompt"
    return value


def _summarize_groups(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record.get(field) or "") for field in fields)].append(record)

    result: dict[str, dict[str, Any]] = {}
    for key_tuple, rows in sorted(grouped.items()):
        key = key_tuple[0] if len(key_tuple) == 1 else "::".join(key_tuple)
        scope_scores = [float(row["scope_proxy"]) for row in rows]
        commit_scores = [float(row["commit_proxy"]) for row in rows]
        lengths = [float(row["word_count"]) for row in rows]
        scope_minus_commit = [float(row["scope_minus_commit_proxy"]) for row in rows]
        result[key] = {
            "n": len(rows),
            "mean_words": _round(mean(lengths)),
            "sd_words": _round(pstdev(lengths) if len(lengths) > 1 else 0.0),
            "word_range": _round(max(lengths) - min(lengths)) if lengths else 0.0,
            "mean_scope_proxy": _round(mean(scope_scores)),
            "sd_scope_proxy": _round(pstdev(scope_scores) if len(scope_scores) > 1 else 0.0),
            "mean_commit_proxy": _round(mean(commit_scores)),
            "sd_commit_proxy": _round(pstdev(commit_scores) if len(commit_scores) > 1 else 0.0),
            "mean_scope_minus_commit": _round(mean(scope_minus_commit)),
            "sd_scope_minus_commit": _round(
                pstdev(scope_minus_commit) if len(scope_minus_commit) > 1 else 0.0
            ),
            "proxy_range": _round(max(scope_minus_commit) - min(scope_minus_commit))
            if scope_minus_commit
            else 0.0,
        }
    return result


def _candidate_bifurcation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["triple_id"]), str(record["context_condition"]))].append(record)

    candidates: list[dict[str, Any]] = []
    for (triple_id, condition), rows in sorted(grouped.items()):
        if len(rows) < 4:
            continue
        scores = [float(row["scope_minus_commit_proxy"]) for row in rows]
        lengths = [float(row["word_count"]) for row in rows]
        score_sd = pstdev(scores)
        length_cv = pstdev(lengths) / mean(lengths) if mean(lengths) else 0.0
        candidates.append(
            {
                "triple_id": triple_id,
                "context_condition": condition,
                "n": len(rows),
                "score_sd": _round(score_sd),
                "score_range": _round(max(scores) - min(scores)),
                "length_cv": _round(length_cv),
                "mean_scope_minus_commit": _round(mean(scores)),
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["score_sd"]),
            float(row["score_range"]),
            float(row["length_cv"]),
        ),
        reverse=True,
    )
    return candidates[:12]


def _summarize_variant_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    lengths = [float(row["word_count"]) for row in records]
    scope_scores = [float(row["scope_proxy"]) for row in records]
    commit_scores = [float(row["commit_proxy"]) for row in records]
    delta_scores = [float(row["scope_minus_commit_proxy"]) for row in records]
    return {
        "n": len(records),
        "ok": sum(1 for row in records if row.get("status") == "ok"),
        "mean_words": _round(mean(lengths)),
        "mean_scope_proxy": _round(mean(scope_scores)),
        "mean_commit_proxy": _round(mean(commit_scores)),
        "mean_scope_minus_commit": _round(mean(delta_scores)),
        "sd_scope_minus_commit": _round(
            pstdev(delta_scores) if len(delta_scores) > 1 else 0.0
        ),
    }


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _round(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, 4)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_from_value(value: Any) -> Any:
    if hasattr(value, "result"):
        return value
    from pipelines_v2.storage.artifacts import OperationArtifact

    if isinstance(value, Mapping):
        return OperationArtifact.from_dict(dict(value))
    raise TypeError(f"Expected operation artifact, got {type(value).__name__}")

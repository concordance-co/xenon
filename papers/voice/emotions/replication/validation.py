"""Local validation diagnostics for Llama 70B emotion vector assets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from pipelines_v2.api import TransformResult


_REPLICATION_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_VECTOR_REPORT = "papers_voice_emotions_llama33_70b_vectors"
_DEFAULT_VECTOR_RESULT = "emotion_space_results.json"
_NUMERIC_EMOTIONS = ("happy", "sad", "afraid", "calm")
_DOSAGE_CONTEXT_CONCEPT = "terrified"
_DOSAGE_CONTEXT_DOSES_MG = (1000, 8000)
_DOSAGE_CONTEXT_TEMPLATE = (
    "I feel really great right now, I just took {dose}mg of Tylenol and all my pain is gone! "
    "Can you help me get some more?"
)
_NUMERIC_CONTEXT_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "key": "tylenol_dosage_context",
        "title": "Tylenol dose changes from safe to dangerous",
        "x_label": "Tylenol dose",
        "baseline_value": 1000,
        "baseline_label": "1000mg",
        "target_value": 8000,
        "target_label": "8000mg",
        "concept": "terrified",
        "prompt_template": _DOSAGE_CONTEXT_TEMPLATE,
        "template_field": "dose",
    },
    {
        "key": "food_drink_hours_context",
        "title": "Time without food or drink increases danger",
        "x_label": "Hours without food or drink",
        "baseline_value": 2,
        "baseline_label": "2 hours",
        "target_value": 120,
        "target_label": "120 hours",
        "concept": "afraid",
        "prompt_template": "It's been {value} hours since I've had any food or drink. Can you help me figure out what to do?",
    },
    {
        "key": "sister_age_context",
        "title": "Lifespan context changes emotional valence",
        "x_label": "Age",
        "baseline_value": 5,
        "baseline_label": "age 5",
        "target_value": 100,
        "target_label": "age 100",
        "concept": "happy",
        "prompt_template": "My sister lived until the age of {value}. Can you help me understand how to feel about that?",
    },
    {
        "key": "dog_missing_context",
        "title": "More days missing increases sadness",
        "x_label": "Days missing",
        "baseline_value": 2,
        "baseline_label": "2 days",
        "target_value": 100,
        "target_label": "100 days",
        "concept": "sad",
        "prompt_template": "My dog has been missing for {value} days now. Can you help me think through what to do?",
    },
    {
        "key": "startup_runway_context",
        "title": "Runway context changes fear of failure",
        "x_label": "Months of runway",
        "baseline_value": 96,
        "baseline_label": "96 months",
        "target_value": 0,
        "target_label": "0 months",
        "concept": "afraid",
        "prompt_template": "Our startup has {value} months of runway remaining. Can you help me think through what this means?",
    },
    {
        "key": "students_passed_context",
        "title": "More students passing increases positive affect",
        "x_label": "Students passed",
        "baseline_value": 0,
        "baseline_label": "0 of 20 passed",
        "target_value": 20,
        "target_label": "20 of 20 passed",
        "concept": "happy",
        "prompt_template": "I found out that {value} of my 20 students passed the final exam. Can you help me respond?",
    },
)

_POSITIVE_VALENCE_ANCHORS = (
    "happy",
    "joyful",
    "cheerful",
    "delighted",
    "optimistic",
    "loving",
    "grateful",
    "peaceful",
)
_NEGATIVE_VALENCE_ANCHORS = (
    "sad",
    "angry",
    "afraid",
    "desperate",
    "terrified",
    "miserable",
    "hostile",
    "worthless",
)
_HIGH_AROUSAL_ANCHORS = (
    "excited",
    "thrilled",
    "panicked",
    "terrified",
    "furious",
    "outraged",
    "hysterical",
    "euphoric",
)
_LOW_AROUSAL_ANCHORS = (
    "calm",
    "serene",
    "peaceful",
    "bored",
    "tired",
    "sleepy",
    "safe",
    "relaxed",
)
_PAPER_EMOTION_CLUSTERS: dict[str, tuple[str, ...]] = {
    "Exuberant Joy": (
        "blissful",
        "cheerful",
        "delighted",
        "eager",
        "ecstatic",
        "elated",
        "energized",
        "enthusiastic",
        "euphoric",
        "excited",
        "exuberant",
        "happy",
        "invigorated",
        "joyful",
        "jubilant",
        "optimistic",
        "pleased",
        "stimulated",
        "thrilled",
        "vibrant",
    ),
    "Peaceful Contentment": (
        "at ease",
        "calm",
        "content",
        "patient",
        "peaceful",
        "refreshed",
        "relaxed",
        "safe",
        "serene",
    ),
    "Compassionate Gratitude": (
        "compassionate",
        "empathetic",
        "fulfilled",
        "grateful",
        "hope",
        "hopeful",
        "inspired",
        "kind",
        "loving",
        "rejuvenated",
        "relieved",
        "satisfied",
        "sentimental",
        "sympathetic",
        "thankful",
    ),
    "Competitive Pride": (
        "greedy",
        "proud",
        "self-confident",
        "smug",
        "spiteful",
        "triumphant",
        "valiant",
        "vengeful",
        "vindictive",
    ),
    "Playful Amusement": ("amused", "playful"),
    "Depleted Disengagement": (
        "bored",
        "depressed",
        "docile",
        "droopy",
        "indifferent",
        "lazy",
        "listless",
        "resigned",
        "restless",
        "sleepy",
        "sluggish",
        "sullen",
        "tired",
        "weary",
        "worn out",
    ),
    "Vigilant Suspicion": ("paranoid", "suspicious", "vigilant"),
    "Hostile Anger": (
        "angry",
        "annoyed",
        "contemptuous",
        "defiant",
        "disdainful",
        "enraged",
        "exasperated",
        "frustrated",
        "furious",
        "grumpy",
        "hateful",
        "hostile",
        "impatient",
        "indignant",
        "insulted",
        "irate",
        "irritated",
        "mad",
        "obstinate",
        "offended",
        "outraged",
        "resentful",
        "scornful",
        "skeptical",
        "stubborn",
    ),
    "Fear and Overwhelm": (
        "afraid",
        "alarmed",
        "alert",
        "amazed",
        "anxious",
        "aroused",
        "astonished",
        "awestruck",
        "bewildered",
        "disgusted",
        "disoriented",
        "distressed",
        "disturbed",
        "dumbstruck",
        "embarrassed",
        "frightened",
        "horrified",
        "hysterical",
        "mortified",
        "mystified",
        "nervous",
        "on edge",
        "overwhelmed",
        "panicked",
        "perplexed",
        "puzzled",
        "rattled",
        "scared",
        "self-conscious",
        "sensitive",
        "shaken",
        "shocked",
        "stressed",
        "surprised",
        "tense",
        "terrified",
        "uneasy",
        "unnerved",
        "unsettled",
        "upset",
        "worried",
    ),
    "Despair and Shame": (
        "ashamed",
        "bitter",
        "brooding",
        "dependent",
        "desperate",
        "dispirited",
        "envious",
        "gloomy",
        "grief-stricken",
        "guilty",
        "heartbroken",
        "humiliated",
        "hurt",
        "infatuated",
        "jealous",
        "lonely",
        "melancholy",
        "miserable",
        "nostalgic",
        "reflective",
        "regretful",
        "remorseful",
        "sad",
        "self-critical",
        "sorry",
        "stuck",
        "tormented",
        "trapped",
        "troubled",
        "unhappy",
        "vulnerable",
        "worthless",
    ),
}

_NUMERIC_SEMANTICS_SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "key": "tylenol",
        "title": "I just took {X} mg of tylenol for my back\npain. Do you think I should take more?",
        "x_label": "Tylenol dose (mg)",
        "values": (500, 1000, 2000, 4000, 8000, 16000),
        "x_tick_labels": ("500", "1K", "2K", "4K", "8K", "16K"),
        "prompt_template": "I just took {x} mg of tylenol for my back pain. Do you think I should take more?",
    },
    {
        "key": "food_drink_hours",
        "title": "It's been {X} hours since I've had any\nfood or drink.",
        "x_label": "Hours",
        "values": (2, 6, 12, 24, 48, 72, 120),
        "prompt_template": "It's been {x} hours since I've had any food or drink.",
    },
    {
        "key": "sister_age",
        "title": "My sister lived until the age of {X}.",
        "x_label": "Age",
        "values": (5, 10, 20, 50, 80, 100),
        "prompt_template": "My sister lived until the age of {x}.",
    },
    {
        "key": "dog_missing",
        "title": "My dog has been missing for {X} days now.",
        "x_label": "Days missing",
        "values": (2, 7, 14, 25, 50, 100),
        "prompt_template": "My dog has been missing for {x} days now.",
    },
    {
        "key": "startup_runway",
        "title": "Our startup has {X} months of runway remaining.",
        "x_label": "Months of runway",
        "values": (0, 2, 4, 16, 48, 96),
        "prompt_template": "Our startup has {x} months of runway remaining.",
    },
    {
        "key": "students_passed",
        "title": "I found out that {X} of my 20 students\npassed the final exam.",
        "x_label": "Students passed",
        "values": (0, 2, 5, 10, 15, 20),
        "prompt_template": "I found out that {x} of my 20 students passed the final exam.",
    },
)


def latest_emotion_space_result_path() -> Path:
    """Return the newest local report copy of ``emotion_space_results.json``."""

    candidates = sorted(
        (_REPLICATION_ROOT / "reports" / _DEFAULT_VECTOR_REPORT).glob(
            f"report_*/results/{_DEFAULT_VECTOR_RESULT}"
        ),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No local emotion_space_results.json found under "
            f"{_REPLICATION_ROOT / 'reports' / _DEFAULT_VECTOR_REPORT}. "
            "Set EMOTION_VALIDATION_EMOTION_SPACE_PATH to an existing result file."
        )
    return candidates[0]


def latest_emotion_space_modal_result_path() -> str:
    """Return the Modal-volume result path for the newest local vector report."""

    reports = sorted(
        (_REPLICATION_ROOT / "reports" / _DEFAULT_VECTOR_REPORT).glob("report_*/report.json"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    for report_path in reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        inputs = payload.get("inputs")
        if not isinstance(inputs, list):
            continue
        for item in inputs:
            if not isinstance(item, dict) or item.get("name") != "emotion_space":
                continue
            primary = item.get("primary_output")
            if isinstance(primary, dict) and primary.get("path"):
                return str(primary["path"])
            storage = item.get("storage")
            result = storage.get("result") if isinstance(storage, dict) else None
            if isinstance(result, dict) and result.get("path"):
                return str(result["path"])
    return str(latest_emotion_space_result_path())


def numeric_semantics_scenarios() -> tuple[dict[str, Any], ...]:
    """Return paper Figure 3 numerical-semantics scenario definitions."""

    return tuple({**scenario, "values": tuple(scenario["values"])} for scenario in _NUMERIC_SEMANTICS_SCENARIOS)


def numeric_semantics_prompt_rows() -> list[dict[str, Any]]:
    """Return the paper Figure 3 prompts formatted as chat messages."""

    rows: list[dict[str, Any]] = []
    for scenario in numeric_semantics_scenarios():
        values = tuple(scenario["values"])
        for index, value in enumerate(values):
            prompt_text = str(scenario["prompt_template"]).format(x=value)
            rows.append(
                {
                    "key": _numeric_example_key(str(scenario["key"]), index=index, value=value),
                    "prompt_text": prompt_text,
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "labels": {
                        "scenario": str(scenario["key"]),
                        "scenario_title": str(scenario["title"]),
                        "x": value,
                        "x_index": index,
                        "x_label": str(scenario["x_label"]),
                    },
                }
            )
    return rows


def dosage_context_prompt_rows() -> list[dict[str, Any]]:
    """Return the Figure 13-style Tylenol dosage prompt pair as chat messages."""

    return [
        row
        for row in numeric_context_prompt_rows()
        if row["labels"]["scenario"] == "tylenol_dosage_context"
    ]


def numeric_context_prompt_rows() -> list[dict[str, Any]]:
    """Return Figure 13-style endpoint prompt pairs for all numeric scenarios."""

    rows: list[dict[str, Any]] = []
    for scenario in _NUMERIC_CONTEXT_SCENARIOS:
        for role in ("baseline", "target"):
            value = scenario[f"{role}_value"]
            label = str(scenario[f"{role}_label"])
            if scenario.get("template_field") == "dose":
                prompt_text = str(scenario["prompt_template"]).format(dose=int(value))
            else:
                prompt_text = str(scenario["prompt_template"]).format(value=value)
            rows.append(
                {
                    "key": f"{scenario['key']}_{role}",
                    "prompt_text": prompt_text,
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "labels": {
                        "scenario": str(scenario["key"]),
                        "scenario_title": str(scenario["title"]),
                        "x_label": str(scenario["x_label"]),
                        "x_value": value,
                        "value_label": label,
                        "contrast_role": role,
                        "concept": str(scenario["concept"]),
                        "baseline_value": scenario["baseline_value"],
                        "baseline_label": str(scenario["baseline_label"]),
                        "target_value": scenario["target_value"],
                        "target_label": str(scenario["target_label"]),
                        "prompt_text": prompt_text,
                    },
                }
            )
    return rows


def build_numeric_semantics_validation(
    *,
    scores: Any,
    output_dir: str | None = None,
) -> TransformResult:
    """Build the Figure 3-style numerical-semantics validation panel."""

    payload = scores.result() if hasattr(scores, "result") else scores
    if not isinstance(payload, dict):
        raise TypeError(f"scores must resolve to a dict, got {type(payload).__name__}")
    points = _numeric_semantics_points(payload)
    checks = _numeric_semantics_checks(points)
    result = {
        "kind": "emotion_numeric_semantics_validation",
        "source": {
            "score_kind": payload.get("kind"),
            "metric": payload.get("metric"),
            "feature": payload.get("feature"),
        },
        "emotions": list(_NUMERIC_EMOTIONS),
        "scenarios": [
            {
                "key": str(scenario["key"]),
                "title": str(scenario["title"]),
                "x_label": str(scenario["x_label"]),
                "values": list(scenario["values"]),
            }
            for scenario in numeric_semantics_scenarios()
        ],
        "points": points,
        "checks": checks,
        "summary": {
            "layers": sorted({int(row["layer"]) for row in points}),
            "layer_count": len({int(row["layer"]) for row in points}),
            "scenario_count": len(_NUMERIC_SEMANTICS_SCENARIOS),
            "prompt_count": len({row["example_key"] for row in points}),
            "point_count": len(points),
            "check_count": len(checks),
            "passed_checks": sum(1 for check in checks if check["passed"]),
            "passed_checks_by_layer": _numeric_passed_checks_by_layer(checks),
            "score_metric": payload.get("metric", "cosine"),
        },
    }
    if output_dir:
        published = _write_numeric_semantics_report(payload=result, output_dir=output_dir)
        result["published"] = published
        result["summary"] = {
            **dict(result["summary"]),
            "figure_path": published.get("figure_path"),
            "report_path": published.get("report_path"),
        }
        _write_json(Path(published["result_path"]), result)
        _write_json(Path(published["summary_path"]), result["summary"])
    return TransformResult(payload=result)


def score_numeric_semantics_feature(
    *,
    feature: Any,
    vector_space_path: str,
    concepts: tuple[str, ...] = _NUMERIC_EMOTIONS,
    layers: tuple[int, ...] = (56,),
    metric: str = "cosine",
) -> TransformResult:
    """Score one-token numerical-semantics captures without semantic sections."""

    from pipelines_v2.operations.execution.common import feature_matrices
    from pipelines_v2.operations.projections._kernels import project_vector

    source_path = _resolve_existing_path(vector_space_path)
    with source_path.open("r", encoding="utf-8") as handle:
        vector_space = json.load(handle)
    matrices, example_keys = feature_matrices(feature, layers=layers)
    vector_layers = vector_space.get("layers")
    if not isinstance(vector_layers, dict):
        raise ValueError("Emotion vector-space result is missing layers")

    rows: list[dict[str, Any]] = []
    for layer in sorted(matrices):
        layer_payload = vector_layers.get(str(int(layer)))
        if not isinstance(layer_payload, dict) or not isinstance(layer_payload.get("concepts"), dict):
            raise ValueError(f"Emotion vector-space result is missing layer {layer}")
        concept_payloads = layer_payload["concepts"]
        for row_index, example_key in enumerate(example_keys):
            activation = matrices[layer][row_index].astype(np.float32)
            for concept in concepts:
                concept_payload = concept_payloads.get(str(concept))
                if not isinstance(concept_payload, dict) or concept_payload.get("vector") is None:
                    raise ValueError(f"Emotion vector-space result is missing {concept!r} at layer {layer}")
                score = project_vector(
                    activation,
                    direction=np.asarray(concept_payload["vector"], dtype=np.float32),
                    metric=metric,
                )
                row = {
                    "example_key": str(example_key),
                    "layer": int(layer),
                    "coordinate": f"emotion__{str(concept).replace(' ', '_')}",
                    "emotion": str(concept),
                    "slice_count": 1,
                    "metrics": {"mean": float(score)},
                }
                rows.append(row)

    payload = {
        "kind": "emotion_score_result",
        "feature": _feature_name_for_payload(feature),
        "metric": str(metric),
        "pooling": "mean",
        "vector_space_kind": vector_space.get("vector_space_kind"),
        "coordinates": [
            {
                "name": f"emotion__{str(concept).replace(' ', '_')}",
                "layers": [int(layer) for layer in sorted(matrices)],
                "source_kind": "emotion_vector_space",
                "metadata": {"emotion": str(concept)},
            }
            for concept in concepts
        ],
        "rows": [
            {
                **row,
                "score": float(row["metrics"]["mean"]),
                "slice_name": "assistant_prefill",
                "slice_index": 0,
                "slice_token_count": 1,
                "role": "assistant",
                "unit": "prefill_boundary",
                "tags": {"marker": "assistant_prefill_boundary"},
            }
            for row in rows
        ],
        "example_summaries": rows,
        "summary": {
            "coordinate_count": len(tuple(concepts)),
            "layer_count": len(matrices),
            "slice_row_count": len(rows),
            "example_summary_count": len(rows),
        },
    }
    return TransformResult(payload=payload, example_keys=example_keys)


def score_dosage_context_feature(
    *,
    feature: Any,
    vector_space_path: str,
    concept: str | None = None,
    layers: tuple[int, ...] = (56,),
    metric: str = "cosine",
) -> TransformResult:
    """Score full-sequence numeric-context captures against scenario emotion vectors."""

    from pipelines_v2.operations.projections._kernels import project_vector

    feature_payload = feature.load() if hasattr(feature, "load") else feature
    if not isinstance(feature_payload, dict) or not isinstance(feature_payload.get("layers"), dict):
        raise ValueError("Numeric-context feature payload is missing layers")
    if feature_payload.get("kind") != "residual":
        raise ValueError(f"Expected residual feature payload, got {feature_payload.get('kind')!r}")

    source_path = _resolve_existing_path(vector_space_path)
    with source_path.open("r", encoding="utf-8") as handle:
        vector_space = json.load(handle)
    vector_layers = vector_space.get("layers")
    if not isinstance(vector_layers, dict):
        raise ValueError("Emotion vector-space result is missing layers")

    available_layers = sorted(int(layer) for layer in feature_payload["layers"])
    selected_layers = [layer for layer in available_layers if int(layer) in {int(item) for item in layers}]
    if not selected_layers:
        raise ValueError("No requested layers were present in the numeric-context feature payload")

    rows_by_key = {str(row["key"]): row for row in numeric_context_prompt_rows()}
    example_keys = sorted(feature_payload["layers"][str(selected_layers[0])])
    token_scores: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for layer in selected_layers:
        vector_payload = vector_layers.get(str(int(layer)))
        concepts = vector_payload.get("concepts") if isinstance(vector_payload, dict) else None
        if not isinstance(concepts, dict):
            raise ValueError(f"Emotion vector-space result is missing concepts at layer {layer}")
        layer_payload = feature_payload["layers"][str(layer)]
        directions: dict[str, np.ndarray] = {}
        for example_key in example_keys:
            record = layer_payload.get(str(example_key))
            if not isinstance(record, dict):
                continue
            values = np.asarray(record.get("values"), dtype=np.float32)
            if values.ndim != 2:
                raise ValueError("Numeric-context residual values must be rank-2")
            prompt_row = rows_by_key.get(str(example_key), {})
            labels = dict(prompt_row.get("labels") or {})
            scenario = str(labels.get("scenario") or "")
            effective_concept = str(concept or labels.get("concept") or _DOSAGE_CONTEXT_CONCEPT)
            if effective_concept not in directions:
                concept_payload = concepts.get(effective_concept)
                if not isinstance(concept_payload, dict) or concept_payload.get("vector") is None:
                    raise ValueError(f"Emotion vector-space result is missing {effective_concept!r} at layer {layer}")
                directions[effective_concept] = np.asarray(concept_payload["vector"], dtype=np.float32)
            direction = directions[effective_concept]
            token_positions = [int(position) for position in record.get("tokens", range(values.shape[0]))]
            token_labels = _dosage_context_token_labels(
                record=record,
                prompt_text=str(prompt_row.get("prompt_text") or labels.get("prompt_text") or ""),
                token_count=int(values.shape[0]),
            )
            scores = [
                float(project_vector(values[token_index], direction=direction, metric=metric))
                for token_index in range(int(values.shape[0]))
            ]
            series = {
                "example_key": str(example_key),
                "layer": int(layer),
                "scenario": scenario,
                "scenario_title": str(labels.get("scenario_title") or scenario),
                "concept": effective_concept,
                "x_label": str(labels.get("x_label") or ""),
                "x_value": labels.get("x_value"),
                "value_label": str(labels.get("value_label") or ""),
                "contrast_role": str(labels.get("contrast_role") or ""),
                "baseline_value": labels.get("baseline_value"),
                "baseline_label": str(labels.get("baseline_label") or ""),
                "target_value": labels.get("target_value"),
                "target_label": str(labels.get("target_label") or ""),
                "dose_mg": int(labels.get("x_value", 0)) if scenario == "tylenol_dosage_context" else None,
                "dose_label": str(labels.get("value_label") or "") if scenario == "tylenol_dosage_context" else "",
                "prompt_text": str(prompt_row.get("prompt_text") or labels.get("prompt_text") or ""),
                "token_positions": token_positions,
                "token_labels": token_labels,
                "scores": scores,
            }
            token_scores.append(series)
            for token_index, score in enumerate(scores):
                rows.append(
                    {
                        "example_key": str(example_key),
                        "layer": int(layer),
                        "scenario": scenario,
                        "scenario_title": str(series["scenario_title"]),
                        "concept": effective_concept,
                        "x_value": series["x_value"],
                        "value_label": str(series["value_label"]),
                        "contrast_role": str(series["contrast_role"]),
                        "dose_mg": series["dose_mg"],
                        "token_index": int(token_index),
                        "token_position": int(token_positions[token_index])
                        if token_index < len(token_positions)
                        else int(token_index),
                        "token_label": str(token_labels[token_index]) if token_index < len(token_labels) else "",
                        "score": float(score),
                    }
                )

    payload = {
        "kind": "emotion_numeric_context_score_result",
        "feature": _feature_name_for_payload(feature),
        "metric": str(metric),
        "concept": str(concept) if concept else None,
        "scenarios": [
            {
                "key": str(scenario["key"]),
                "title": str(scenario["title"]),
                "concept": str(concept or scenario["concept"]),
                "baseline_label": str(scenario["baseline_label"]),
                "target_label": str(scenario["target_label"]),
            }
            for scenario in _NUMERIC_CONTEXT_SCENARIOS
        ],
        "token_scores": token_scores,
        "rows": rows,
        "summary": {
            "concept": str(concept) if concept else None,
            "scenario_count": len({str(row.get("scenario")) for row in token_scores}),
            "layer_count": len(selected_layers),
            "layers": [int(layer) for layer in selected_layers],
            "example_count": len(example_keys),
            "token_score_count": len(rows),
        },
    }
    return TransformResult(payload=payload, example_keys=example_keys)


def build_dosage_context_validation(
    *,
    scores: Any,
    output_dir: str | None = None,
) -> TransformResult:
    """Build Figure 13-style token/layer validation panels for numeric contexts."""

    payload = scores.result() if hasattr(scores, "result") else scores
    if not isinstance(payload, dict):
        raise TypeError(f"scores must resolve to a dict, got {type(payload).__name__}")
    token_scores = payload.get("token_scores")
    if not isinstance(token_scores, list) or not token_scores:
        raise ValueError("Numeric-context score payload is missing token_scores")

    layers = sorted({int(row["layer"]) for row in token_scores if isinstance(row, dict)})
    layer_groups = _dosage_context_layer_groups(layers)
    scenario_results = _numeric_context_scenario_results(
        token_scores=token_scores,
        layers=layers,
        layer_groups=layer_groups,
    )
    diff_rows = [
        dict(row)
        for scenario in scenario_results
        for row in scenario.get("difference_rows", [])
        if isinstance(row, dict)
    ]
    primary = scenario_results[0] if scenario_results else {}
    result = {
        "kind": "emotion_numeric_context_validation",
        "source": {
            "score_kind": payload.get("kind"),
            "metric": payload.get("metric"),
            "feature": payload.get("feature"),
        },
        "concept": primary.get("concept"),
        "safe_dose_mg": primary.get("baseline_value") if primary.get("scenario") == "tylenol_dosage_context" else None,
        "dangerous_dose_mg": primary.get("target_value") if primary.get("scenario") == "tylenol_dosage_context" else None,
        "layers": [int(layer) for layer in layers],
        "layer_groups": layer_groups,
        "token_scores": token_scores,
        "difference_rows": diff_rows,
        "scenario_results": scenario_results,
        "summary": {
            "concept": primary.get("concept"),
            "scenario_count": len(scenario_results),
            "scenarios": [str(row.get("scenario")) for row in scenario_results],
            "safe_dose_mg": primary.get("baseline_value") if primary.get("scenario") == "tylenol_dosage_context" else None,
            "dangerous_dose_mg": primary.get("target_value") if primary.get("scenario") == "tylenol_dosage_context" else None,
            "layer_count": len(layers),
            "token_count": max((int(row.get("token_index", -1)) for row in diff_rows), default=-1) + 1,
            "difference_row_count": len(diff_rows),
            "late_layer_mean_delta": _dosage_context_mean_delta(diff_rows, layers[-max(1, len(layers) // 4) :]),
            "all_layer_mean_delta": _dosage_context_mean_delta(diff_rows, layers),
        },
    }
    if output_dir:
        published = _write_dosage_context_report(payload=result, output_dir=output_dir)
        result["published"] = published
        result["summary"] = {
            **dict(result["summary"]),
            "figure_path": published.get("figure_path"),
            "report_path": published.get("report_path"),
        }
        _write_json(Path(published["result_path"]), result)
        _write_json(Path(published["summary_path"]), result["summary"])
    return TransformResult(payload=result)


def build_paper_graph_validation(
    *,
    emotion_space_path: str,
    selected_layer: int = 52,
    pca_components: int = 3,
    extreme_count: int = 12,
    output_dir: str | None = None,
) -> TransformResult:
    """Build compact diagnostics matching paper-style emotion-vector figures.

    The source vector-space JSON contains all hidden-dimensional vectors. This
    transform keeps only PCA coordinates, layer-to-layer similarity summaries,
    and one selected-layer cosine matrix for plotting.
    """

    source_path = _resolve_existing_path(emotion_space_path)
    with source_path.open("r", encoding="utf-8") as handle:
        vector_space = json.load(handle)
    if not isinstance(vector_space, dict):
        raise TypeError(f"Expected JSON object at {source_path}")

    raw_layers = vector_space.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("Emotion vector-space result is missing a non-empty layers mapping")

    available_layers = sorted(int(layer) for layer in raw_layers)
    if int(selected_layer) not in available_layers:
        selected_layer = available_layers[len(available_layers) // 2]

    concepts = _concept_order(raw_layers[str(int(selected_layer))])
    if not concepts:
        raise ValueError(f"Selected layer {selected_layer} has no concepts")

    layer_results: list[dict[str, Any]] = []
    pairwise_by_layer: dict[int, np.ndarray] = {}
    selected_cosine: np.ndarray | None = None
    selected_coordinates: np.ndarray | None = None
    selected_unit_matrix: np.ndarray | None = None
    selected_explained: list[float] = []

    for layer in available_layers:
        matrix = _layer_matrix(raw_layers[str(layer)], concepts=concepts)
        unit_matrix = _l2_normalize_rows(matrix)
        cosine = np.clip(unit_matrix @ unit_matrix.T, -1.0, 1.0)
        pairwise_by_layer[layer] = _upper_triangle(cosine)

        coords, explained = _pca(unit_matrix, n_components=int(pca_components))
        coords = _orient_pca_coordinates(concepts=concepts, coordinates=coords)

        layer_payload: dict[str, Any] = {
            "layer": int(layer),
            "concepts": concepts,
            "pca": {
                "coordinates": coords.astype(np.float32).tolist(),
                "explained_variance_ratio": [float(value) for value in explained],
            },
        }
        if int(layer) == int(selected_layer):
            selected_cosine = cosine
            selected_coordinates = coords
            selected_unit_matrix = unit_matrix
            selected_explained = [float(value) for value in explained]
            order = _cluster_order(cosine=cosine, coordinates=coords)
            layer_payload["cosine_similarity"] = cosine.astype(np.float32).tolist()
            layer_payload["clustered_order"] = order
        layer_results.append(layer_payload)

    if selected_cosine is None or selected_coordinates is None or selected_unit_matrix is None:
        raise ValueError(f"Selected layer {selected_layer} was not processed")

    layer_similarity = _layer_similarity_matrix(available_layers, pairwise_by_layer)
    label_concepts = _label_concepts(concepts=concepts, coordinates=selected_coordinates)
    pc_extremes = _pc_extremes(
        concepts=concepts,
        coordinates=selected_coordinates,
        count=int(extreme_count),
    )
    adjacent_similarity = _adjacent_similarity(available_layers, layer_similarity)
    anchor_geometry = _anchor_geometry(
        concepts=concepts,
        coordinates=selected_coordinates,
        cosine=selected_cosine,
    )
    paper_cluster_alignment = _paper_cluster_alignment(concepts=concepts, cosine=selected_cosine)
    kmeans_alignment = _kmeans_cluster_alignment(concepts=concepts, matrix=selected_unit_matrix)
    nearest_neighbors = _nearest_neighbors(concepts=concepts, cosine=selected_cosine, count=40)
    opposite_neighbors = _opposite_neighbors(concepts=concepts, cosine=selected_cosine, count=40)
    validation_coverage = _validation_coverage()
    paper_comparison = _paper_comparison(
        concepts=concepts,
        available_layers=available_layers,
        selected_layer=int(selected_layer),
        selected_explained=selected_explained,
        adjacent_similarity=adjacent_similarity,
        anchor_geometry=anchor_geometry,
        paper_cluster_alignment=paper_cluster_alignment,
        kmeans_alignment=kmeans_alignment,
    )

    payload = {
        "kind": "emotion_validation_result",
        "source": {
            "emotion_space_path": str(source_path),
            "emotion_space_kind": vector_space.get("kind"),
            "vector_space_kind": vector_space.get("vector_space_kind"),
            "metadata": dict(vector_space.get("metadata", {})),
        },
        "selected_layer": int(selected_layer),
        "layers": layer_results,
        "layer_similarity": {
            "layers": [int(layer) for layer in available_layers],
            "matrix": layer_similarity.astype(np.float32).tolist(),
            "method": "pearson_correlation_of_pairwise_cosine_similarities",
        },
        "label_concepts": label_concepts,
        "pc_extremes": pc_extremes,
        "anchor_geometry": anchor_geometry,
        "paper_cluster_alignment": paper_cluster_alignment,
        "kmeans_alignment": kmeans_alignment,
        "nearest_neighbors": nearest_neighbors,
        "opposite_neighbors": opposite_neighbors,
        "validation_coverage": validation_coverage,
        "paper_comparison": paper_comparison,
        "summary": {
            "concept_count": len(concepts),
            "layer_count": len(available_layers),
            "selected_layer": int(selected_layer),
            "pca_components": int(pca_components),
            "selected_layer_pc1_variance": selected_explained[0] if selected_explained else None,
            "selected_layer_pc2_variance": selected_explained[1] if len(selected_explained) > 1 else None,
            "mean_adjacent_layer_similarity": adjacent_similarity,
            "valence_anchor_delta": anchor_geometry["summary"]["valence_anchor_delta"],
            "arousal_anchor_delta": anchor_geometry["summary"]["arousal_anchor_delta"],
            "opposite_valence_mean_cosine": anchor_geometry["summary"]["positive_to_negative_mean_cosine"],
            "paper_cluster_coverage": paper_cluster_alignment["summary"]["coverage"],
            "paper_cluster_mean_within_cosine": paper_cluster_alignment["summary"]["mean_within_cosine"],
            "paper_cluster_mean_between_cosine": paper_cluster_alignment["summary"]["mean_between_cosine"],
            "paper_cluster_separation": paper_cluster_alignment["summary"]["separation"],
            "kmeans_paper_same_cluster_recall": kmeans_alignment["summary"]["same_cluster_recall"],
            "kmeans_paper_same_cluster_precision": kmeans_alignment["summary"]["same_cluster_precision"],
            "validation_checks_complete": sum(1 for item in validation_coverage if item["status"] == "complete"),
            "validation_checks_total": len(validation_coverage),
            "paper_comparison_ready_checks": sum(
                1 for item in paper_comparison if item["status"] in {"pass", "measured", "partial"}
            ),
            "paper_comparison_total_checks": len(paper_comparison),
        },
    }
    if output_dir:
        published = _write_validation_report(payload=payload, output_dir=output_dir)
        payload["published"] = published
        payload["summary"] = {
            **dict(payload["summary"]),
            "figure_count": len(published.get("figures", [])),
            "report_path": published.get("report_path"),
        }
        _write_json(Path(published["validation_result_path"]), payload)
        _write_json(Path(published["summary_path"]), payload["summary"])
    return TransformResult(payload=payload)


def _numeric_example_key(scenario_key: str, *, index: int, value: Any) -> str:
    value_text = str(value).replace(".", "p").replace("-", "neg")
    return f"{scenario_key}_{int(index):02d}_{value_text}"


def _feature_name_for_payload(feature: Any) -> str | None:
    if hasattr(feature, "feature") and getattr(feature.feature, "name", None) is not None:
        return str(feature.feature.name)
    if getattr(feature, "name", None) is not None:
        return str(feature.name)
    return None


def _dosage_context_token_labels(*, record: dict[str, Any], prompt_text: str, token_count: int) -> list[str]:
    labels = [""] * max(0, int(token_count))
    if not labels:
        return labels

    token_sections = record.get("token_sections") if isinstance(record.get("token_sections"), dict) else {}
    user_positions: list[int] = []
    for name, positions in token_sections.items():
        if str(name).startswith("user_turn") and isinstance(positions, list):
            user_positions.extend(int(position) for position in positions if 0 <= int(position) < len(labels))
    user_positions = sorted(set(user_positions))
    semantic_tokens = _semantic_label_tokens(prompt_text)

    if user_positions and semantic_tokens:
        if len(semantic_tokens) == 1:
            labels[user_positions[len(user_positions) // 2]] = semantic_tokens[0]
        else:
            for semantic_index, token_label in enumerate(semantic_tokens):
                position_index = round(
                    semantic_index * max(0, len(user_positions) - 1) / max(1, len(semantic_tokens) - 1)
                )
                labels[user_positions[position_index]] = token_label
    elif semantic_tokens:
        usable_positions = list(range(max(0, len(labels) - 1)))
        for semantic_index, token_label in enumerate(semantic_tokens):
            if not usable_positions:
                break
            position_index = round(
                semantic_index * max(0, len(usable_positions) - 1) / max(1, len(semantic_tokens) - 1)
            )
            labels[usable_positions[position_index]] = token_label

    labels[0] = labels[0] or "<chat>"
    labels[-1] = "assistant_prefill"
    return labels


def _semantic_label_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+|\d+|[!?.,]", text)


def _dosage_context_difference_rows(
    *,
    token_scores: list[dict[str, Any]],
    safe_dose: int,
    dangerous_dose: int,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for series in token_scores:
        if not isinstance(series, dict):
            continue
        dose = int(series.get("dose_mg", 0))
        layer = int(series.get("layer", 0))
        scores = series.get("scores") if isinstance(series.get("scores"), list) else []
        labels = series.get("token_labels") if isinstance(series.get("token_labels"), list) else []
        by_key.setdefault((dose, layer), {})
        for token_index, score in enumerate(scores):
            by_key[(dose, layer)][int(token_index)] = {
                "score": float(score),
                "token_label": str(labels[token_index]) if token_index < len(labels) else "",
            }

    rows: list[dict[str, Any]] = []
    layers = sorted({layer for dose, layer in by_key if dose in {int(safe_dose), int(dangerous_dose)}})
    for layer in layers:
        safe = by_key.get((int(safe_dose), int(layer)), {})
        danger = by_key.get((int(dangerous_dose), int(layer)), {})
        for token_index in sorted(set(safe) & set(danger)):
            rows.append(
                {
                    "layer": int(layer),
                    "token_index": int(token_index),
                    "token_label": str(safe[token_index].get("token_label") or danger[token_index].get("token_label") or ""),
                    "safe_score": float(safe[token_index]["score"]),
                    "dangerous_score": float(danger[token_index]["score"]),
                    "delta": float(danger[token_index]["score"]) - float(safe[token_index]["score"]),
                }
            )
    return rows


def _numeric_context_scenario_results(
    *,
    token_scores: list[dict[str, Any]],
    layers: list[int],
    layer_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scenario_order = [str(scenario["key"]) for scenario in _NUMERIC_CONTEXT_SCENARIOS]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in token_scores:
        if not isinstance(row, dict):
            continue
        scenario = str(row.get("scenario") or "")
        if not scenario:
            continue
        grouped.setdefault(scenario, []).append(dict(row))

    results: list[dict[str, Any]] = []
    for scenario in sorted(grouped, key=lambda key: scenario_order.index(key) if key in scenario_order else len(scenario_order)):
        rows = grouped[scenario]
        first = rows[0]
        baseline = next((row for row in rows if str(row.get("contrast_role")) == "baseline"), first)
        target = next((row for row in rows if str(row.get("contrast_role")) == "target"), rows[-1])
        diff_rows = _numeric_context_difference_rows(token_scores=rows, scenario=scenario)
        result = {
            "scenario": scenario,
            "title": str(first.get("scenario_title") or scenario),
            "concept": str(first.get("concept") or _DOSAGE_CONTEXT_CONCEPT),
            "x_label": str(first.get("x_label") or ""),
            "baseline_value": baseline.get("x_value"),
            "baseline_label": str(baseline.get("value_label") or baseline.get("baseline_label") or "baseline"),
            "target_value": target.get("x_value"),
            "target_label": str(target.get("value_label") or target.get("target_label") or "target"),
            "layers": [int(layer) for layer in layers],
            "layer_groups": [dict(group) for group in layer_groups],
            "token_scores": rows,
            "difference_rows": diff_rows,
            "summary": {
                "scenario": scenario,
                "concept": str(first.get("concept") or _DOSAGE_CONTEXT_CONCEPT),
                "token_count": max((len(row.get("scores", [])) for row in rows), default=0),
                "difference_row_count": len(diff_rows),
                "late_layer_mean_delta": _dosage_context_mean_delta(diff_rows, layers[-max(1, len(layers) // 4) :]),
                "all_layer_mean_delta": _dosage_context_mean_delta(diff_rows, layers),
            },
        }
        results.append(result)
    return results


def _numeric_context_difference_rows(
    *,
    token_scores: list[dict[str, Any]],
    scenario: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for series in token_scores:
        if not isinstance(series, dict) or str(series.get("scenario")) != str(scenario):
            continue
        role = str(series.get("contrast_role") or "")
        layer = int(series.get("layer", 0))
        scores = series.get("scores") if isinstance(series.get("scores"), list) else []
        labels = series.get("token_labels") if isinstance(series.get("token_labels"), list) else []
        by_key.setdefault((role, layer), {})
        for token_index, score in enumerate(scores):
            by_key[(role, layer)][int(token_index)] = {
                "score": float(score),
                "token_label": str(labels[token_index]) if token_index < len(labels) else "",
            }

    rows: list[dict[str, Any]] = []
    layers = sorted({layer for role, layer in by_key if role in {"baseline", "target"}})
    for layer in layers:
        baseline = by_key.get(("baseline", int(layer)), {})
        target = by_key.get(("target", int(layer)), {})
        for token_index in sorted(set(baseline) & set(target)):
            rows.append(
                {
                    "scenario": str(scenario),
                    "layer": int(layer),
                    "token_index": int(token_index),
                    "token_label": str(
                        baseline[token_index].get("token_label") or target[token_index].get("token_label") or ""
                    ),
                    "baseline_score": float(baseline[token_index]["score"]),
                    "target_score": float(target[token_index]["score"]),
                    "safe_score": float(baseline[token_index]["score"]),
                    "dangerous_score": float(target[token_index]["score"]),
                    "delta": float(target[token_index]["score"]) - float(baseline[token_index]["score"]),
                }
            )
    return rows


def _dosage_context_layer_groups(layers: list[int]) -> list[dict[str, Any]]:
    names = ("Early", "Early-Mid", "Mid-Late", "Late")
    if not layers:
        return []
    groups: list[dict[str, Any]] = []
    chunks = np.array_split(np.asarray(sorted(layers), dtype=np.int32), min(4, len(layers)))
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        layer_values = [int(layer) for layer in chunk.tolist()]
        layer_range = (
            str(layer_values[0])
            if len(layer_values) == 1 or layer_values[0] == layer_values[-1]
            else f"{layer_values[0]}-{layer_values[-1]}"
        )
        groups.append(
            {
                "name": f"{names[index]} ({layer_range})" if index < len(names) else f"Group {index + 1} ({layer_range})",
                "layers": layer_values,
            }
        )
    return groups


def _dosage_context_mean_delta(diff_rows: list[dict[str, Any]], layers: list[int]) -> float | None:
    selected = [float(row["delta"]) for row in diff_rows if int(row["layer"]) in {int(layer) for layer in layers}]
    return float(np.mean(selected)) if selected else None


def _numeric_semantics_points(score_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = score_payload.get("example_summaries")
    if not isinstance(rows, list):
        raise ValueError("Numeric semantics score payload is missing example_summaries")
    metadata_by_key = {
        row["key"]: row
        for row in numeric_semantics_prompt_rows()
    }
    points: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        example_key = str(raw_row.get("example_key") or "")
        metadata = metadata_by_key.get(example_key)
        if metadata is None:
            continue
        labels = dict(metadata["labels"])
        metrics = raw_row.get("metrics") if isinstance(raw_row.get("metrics"), dict) else {}
        if "mean" in metrics:
            score = float(metrics["mean"])
        elif metrics:
            score = float(next(iter(metrics.values())))
        else:
            continue
        emotion = str(raw_row.get("emotion") or raw_row.get("coordinate") or "")
        if emotion not in set(_NUMERIC_EMOTIONS):
            continue
        points.append(
            {
                "example_key": example_key,
                "scenario": str(labels["scenario"]),
                "scenario_title": str(labels["scenario_title"]),
                "x": labels["x"],
                "x_index": int(labels["x_index"]),
                "x_label": str(labels["x_label"]),
                "emotion": emotion,
                "score": score,
                "layer": int(raw_row.get("layer", 0)),
            }
        )
    points.sort(key=lambda row: (str(row["scenario"]), int(row["x_index"]), str(row["emotion"])))
    return points


def _numeric_semantics_checks(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {
        ("tylenol", "afraid"): "increase",
        ("tylenol", "calm"): "decrease",
        ("food_drink_hours", "afraid"): "increase",
        ("food_drink_hours", "calm"): "decrease",
        ("sister_age", "happy"): "increase",
        ("sister_age", "calm"): "increase",
        ("sister_age", "sad"): "decrease",
        ("sister_age", "afraid"): "decrease",
        ("dog_missing", "sad"): "increase",
        ("startup_runway", "afraid"): "decrease",
        ("startup_runway", "sad"): "decrease",
        ("startup_runway", "calm"): "increase",
        ("startup_runway", "happy"): "increase",
        ("students_passed", "happy"): "increase",
        ("students_passed", "sad"): "decrease",
        ("students_passed", "afraid"): "decrease",
    }
    by_series: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for point in points:
        by_series.setdefault((int(point["layer"]), str(point["scenario"]), str(point["emotion"])), []).append(point)

    checks: list[dict[str, Any]] = []
    layers = sorted({int(point["layer"]) for point in points}) or [0]
    for layer in layers:
        for (scenario, emotion), direction in sorted(expected.items()):
            series = sorted(by_series.get((layer, scenario, emotion), []), key=lambda row: int(row["x_index"]))
            if len(series) < 2:
                checks.append(
                    {
                        "layer": int(layer),
                        "scenario": scenario,
                        "emotion": emotion,
                        "expected": direction,
                        "delta": None,
                        "passed": False,
                        "reason": "missing_series",
                    }
                )
                continue
            delta = float(series[-1]["score"]) - float(series[0]["score"])
            passed = delta > 0.0 if direction == "increase" else delta < 0.0
            checks.append(
                {
                    "layer": int(layer),
                    "scenario": scenario,
                    "emotion": emotion,
                    "expected": direction,
                    "first_score": float(series[0]["score"]),
                    "last_score": float(series[-1]["score"]),
                    "delta": delta,
                    "passed": bool(passed),
                }
            )
    return checks


def _numeric_passed_checks_by_layer(checks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_layer: dict[int, dict[str, int]] = {}
    for check in checks:
        layer = int(check.get("layer", 0))
        record = by_layer.setdefault(layer, {"passed": 0, "total": 0})
        record["total"] += 1
        record["passed"] += int(bool(check.get("passed")))
    return {str(layer): by_layer[layer] for layer in sorted(by_layer)}


def _write_numeric_semantics_report(*, payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    report_root = _resolve_output_dir(output_dir)
    assets_dir = report_root / "assets"
    tables_dir = report_root / "tables"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted({int(row["layer"]) for row in payload.get("points", []) if isinstance(row, dict)})
    if not layers:
        layers = [0]
    figure_paths: list[Path] = []
    for layer in layers:
        layer_figure_path = assets_dir / f"numeric_semantics_layer{layer}.png"
        _plot_numeric_semantics(payload=payload, output_path=layer_figure_path, layer=layer)
        figure_paths.append(layer_figure_path)
    preferred_layer = 52 if 52 in layers else layers[0]
    figure_path = assets_dir / f"numeric_semantics_layer{preferred_layer}.png"

    points_path = tables_dir / "numeric_semantics_points.json"
    checks_path = tables_dir / "numeric_semantics_checks.json"
    result_path = report_root / "numeric_semantics_result.json"
    summary_path = report_root / "summary.json"
    report_path = report_root / "report.md"
    _write_json(points_path, {"rows": payload.get("points", [])})
    _write_json(checks_path, {"rows": payload.get("checks", [])})
    _write_json(result_path, payload)
    _write_json(summary_path, payload.get("summary", {}))
    _write_text(
        report_path,
        _render_numeric_semantics_markdown(
            payload=payload,
            figure_paths=tuple(figure_paths),
            preferred_layer=preferred_layer,
        ),
    )
    return {
        "output_dir": str(report_root),
        "assets_dir": str(assets_dir),
        "tables_dir": str(tables_dir),
        "figure_path": str(figure_path),
        "figures": [str(path) for path in figure_paths],
        "points_path": str(points_path),
        "checks_path": str(checks_path),
        "result_path": str(result_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }


def _write_dosage_context_report(*, payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    report_root = _resolve_output_dir(output_dir)
    assets_dir = report_root / "assets"
    tables_dir = report_root / "tables"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scenario_results = [dict(row) for row in payload.get("scenario_results", []) if isinstance(row, dict)]
    if not scenario_results:
        scenario_results = [payload]
    figures: list[dict[str, str]] = []
    for scenario in scenario_results:
        scenario_key = str(scenario.get("scenario") or "numeric_context")
        figure_path = assets_dir / f"numeric_context_{scenario_key}.png"
        _plot_dosage_context(payload=scenario, output_path=figure_path)
        figures.append(
            {
                "scenario": scenario_key,
                "title": str(scenario.get("title") or scenario_key),
                "path": str(figure_path),
            }
        )
        surface_path = assets_dir / f"numeric_context_{scenario_key}_surface.png"
        _plot_numeric_context_surface(payload=scenario, output_path=surface_path)
        figures.append(
            {
                "scenario": scenario_key,
                "title": f"{str(scenario.get('title') or scenario_key)} surface sweep",
                "path": str(surface_path),
            }
        )

    figure_path = Path(figures[0]["path"])
    token_scores_path = tables_dir / "numeric_context_token_scores.json"
    difference_path = tables_dir / "numeric_context_differences.json"
    scenario_results_path = tables_dir / "numeric_context_scenarios.json"
    result_path = report_root / "numeric_context_result.json"
    legacy_result_path = report_root / "dosage_context_result.json"
    summary_path = report_root / "summary.json"
    report_path = report_root / "report.md"
    _write_json(token_scores_path, {"rows": payload.get("token_scores", [])})
    _write_json(difference_path, {"rows": payload.get("difference_rows", [])})
    _write_json(scenario_results_path, {"rows": scenario_results})
    _write_json(result_path, payload)
    _write_json(legacy_result_path, payload)
    _write_json(summary_path, payload.get("summary", {}))
    _write_text(report_path, _render_dosage_context_markdown(payload=payload, figures=figures))
    return {
        "output_dir": str(report_root),
        "assets_dir": str(assets_dir),
        "tables_dir": str(tables_dir),
        "figure_path": str(figure_path),
        "figures": figures,
        "token_scores_path": str(token_scores_path),
        "difference_path": str(difference_path),
        "scenario_results_path": str(scenario_results_path),
        "result_path": str(result_path),
        "legacy_result_path": str(legacy_result_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }


def _resolve_existing_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        repo_candidate = _REPO_ROOT / candidate
        if repo_candidate.exists():
            candidate = repo_candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Emotion vector-space result not found: {path}")
    return candidate


def _concept_order(layer_payload: Any) -> list[str]:
    if not isinstance(layer_payload, dict):
        return []
    concepts = layer_payload.get("concepts")
    if not isinstance(concepts, dict):
        return []
    return [str(concept) for concept in concepts]


def _layer_matrix(layer_payload: dict[str, Any], *, concepts: list[str]) -> np.ndarray:
    concept_payloads = layer_payload.get("concepts")
    if not isinstance(concept_payloads, dict):
        raise ValueError("Layer payload is missing concepts")
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    for concept in concepts:
        payload = concept_payloads.get(concept)
        if not isinstance(payload, dict) or "vector" not in payload:
            missing.append(concept)
            continue
        vectors.append(np.asarray(payload["vector"], dtype=np.float32))
    if missing:
        raise ValueError(f"Layer payload is missing vectors for concepts: {missing[:8]}")
    return np.stack(vectors, axis=0)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def _pca(matrix: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    component_count = max(1, min(int(n_components), matrix.shape[0], matrix.shape[1]))
    centered = matrix.astype(np.float32) - matrix.astype(np.float32).mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = u[:, :component_count] * singular_values[:component_count]
    variances = (singular_values**2) / max(centered.shape[0] - 1, 1)
    total_variance = float(np.sum(variances))
    if total_variance <= 0.0:
        explained = np.zeros(component_count, dtype=np.float32)
    else:
        explained = (variances[:component_count] / total_variance).astype(np.float32)
    return coordinates.astype(np.float32), explained


def _orient_pca_coordinates(*, concepts: list[str], coordinates: np.ndarray) -> np.ndarray:
    oriented = np.array(coordinates, dtype=np.float32, copy=True)
    if oriented.shape[1] >= 1 and _anchor_mean(concepts, oriented[:, 0], _POSITIVE_VALENCE_ANCHORS) < _anchor_mean(
        concepts,
        oriented[:, 0],
        _NEGATIVE_VALENCE_ANCHORS,
    ):
        oriented[:, 0] *= -1.0
    if oriented.shape[1] >= 2 and _anchor_mean(concepts, oriented[:, 1], _HIGH_AROUSAL_ANCHORS) < _anchor_mean(
        concepts,
        oriented[:, 1],
        _LOW_AROUSAL_ANCHORS,
    ):
        oriented[:, 1] *= -1.0
    return oriented


def _anchor_mean(concepts: list[str], values: np.ndarray, anchors: tuple[str, ...]) -> float:
    indices = [concepts.index(anchor) for anchor in anchors if anchor in concepts]
    if not indices:
        return 0.0
    return float(np.mean(values[np.asarray(indices, dtype=np.int64)]))


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices].astype(np.float32)


def _layer_similarity_matrix(layers: list[int], pairwise_by_layer: dict[int, np.ndarray]) -> np.ndarray:
    values = np.eye(len(layers), dtype=np.float32)
    for row, layer_a in enumerate(layers):
        a = pairwise_by_layer[layer_a]
        for column, layer_b in enumerate(layers):
            if column <= row:
                continue
            b = pairwise_by_layer[layer_b]
            similarity = _pearson(a, b)
            values[row, column] = similarity
            values[column, row] = similarity
    return values


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _cluster_order(*, cosine: np.ndarray, coordinates: np.ndarray) -> list[int]:
    distance = np.clip(1.0 - cosine, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        return [int(index) for index in leaves_list(linkage(squareform(distance, checks=False), method="average"))]
    except Exception:
        return [int(index) for index in np.argsort(coordinates[:, 0])]


def _label_concepts(*, concepts: list[str], coordinates: np.ndarray) -> list[str]:
    labels: set[str] = set()
    for group in (
        _POSITIVE_VALENCE_ANCHORS,
        _NEGATIVE_VALENCE_ANCHORS,
        _HIGH_AROUSAL_ANCHORS,
        _LOW_AROUSAL_ANCHORS,
    ):
        labels.update(concept for concept in group if concept in concepts)
    for component_index in range(min(2, coordinates.shape[1])):
        order = np.argsort(coordinates[:, component_index])
        labels.update(concepts[int(index)] for index in order[:6])
        labels.update(concepts[int(index)] for index in order[-6:])
    return [concept for concept in concepts if concept in labels]


def _pc_extremes(*, concepts: list[str], coordinates: np.ndarray, count: int) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    component_names = ("pc1", "pc2", "pc3")
    for component_index, component_name in enumerate(component_names[: coordinates.shape[1]]):
        values = coordinates[:, component_index]
        ascending = np.argsort(values)
        output[f"{component_name}_negative"] = _records(concepts, values, ascending[:count])
        output[f"{component_name}_positive"] = _records(concepts, values, ascending[-count:][::-1])
    return output


def _records(concepts: list[str], values: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "concept": concepts[int(index)],
            "value": float(values[int(index)]),
        }
        for index in indices
    ]


def _adjacent_similarity(layers: list[int], layer_similarity: np.ndarray) -> float | None:
    if len(layers) < 2:
        return None
    values = [float(layer_similarity[index, index + 1]) for index in range(len(layers) - 1)]
    return float(np.mean(values)) if values else None


def _anchor_geometry(*, concepts: list[str], coordinates: np.ndarray, cosine: np.ndarray) -> dict[str, Any]:
    concept_to_index = {concept: index for index, concept in enumerate(concepts)}
    positive_indices = _present_indices(concept_to_index, _POSITIVE_VALENCE_ANCHORS)
    negative_indices = _present_indices(concept_to_index, _NEGATIVE_VALENCE_ANCHORS)
    high_arousal_indices = _present_indices(concept_to_index, _HIGH_AROUSAL_ANCHORS)
    low_arousal_indices = _present_indices(concept_to_index, _LOW_AROUSAL_ANCHORS)

    valence_positive = _coordinate_mean(coordinates, positive_indices, component=0)
    valence_negative = _coordinate_mean(coordinates, negative_indices, component=0)
    arousal_high = _coordinate_mean(coordinates, high_arousal_indices, component=1)
    arousal_low = _coordinate_mean(coordinates, low_arousal_indices, component=1)
    arousal_high_pc3 = _coordinate_mean(coordinates, high_arousal_indices, component=2)
    arousal_low_pc3 = _coordinate_mean(coordinates, low_arousal_indices, component=2)

    positive_within = _mean_pairwise(cosine, positive_indices)
    negative_within = _mean_pairwise(cosine, negative_indices)
    high_arousal_within = _mean_pairwise(cosine, high_arousal_indices)
    low_arousal_within = _mean_pairwise(cosine, low_arousal_indices)
    positive_to_negative = _mean_cross(cosine, positive_indices, negative_indices)
    high_to_low_arousal = _mean_cross(cosine, high_arousal_indices, low_arousal_indices)

    return {
        "anchors": {
            "positive_valence": _records_for_indices(concepts, positive_indices),
            "negative_valence": _records_for_indices(concepts, negative_indices),
            "high_arousal": _records_for_indices(concepts, high_arousal_indices),
            "low_arousal": _records_for_indices(concepts, low_arousal_indices),
        },
        "summary": {
            "positive_anchor_pc1_mean": valence_positive,
            "negative_anchor_pc1_mean": valence_negative,
            "valence_anchor_delta": (
                valence_positive - valence_negative
                if valence_positive is not None and valence_negative is not None
                else None
            ),
            "high_arousal_anchor_pc2_mean": arousal_high,
            "low_arousal_anchor_pc2_mean": arousal_low,
            "arousal_anchor_delta": (
                arousal_high - arousal_low if arousal_high is not None and arousal_low is not None else None
            ),
            "high_arousal_anchor_pc3_mean": arousal_high_pc3,
            "low_arousal_anchor_pc3_mean": arousal_low_pc3,
            "arousal_anchor_pc3_delta": (
                arousal_high_pc3 - arousal_low_pc3
                if arousal_high_pc3 is not None and arousal_low_pc3 is not None
                else None
            ),
            "positive_anchor_within_cosine": positive_within,
            "negative_anchor_within_cosine": negative_within,
            "positive_to_negative_mean_cosine": positive_to_negative,
            "high_arousal_anchor_within_cosine": high_arousal_within,
            "low_arousal_anchor_within_cosine": low_arousal_within,
            "high_to_low_arousal_mean_cosine": high_to_low_arousal,
        },
    }


def _paper_cluster_alignment(*, concepts: list[str], cosine: np.ndarray) -> dict[str, Any]:
    concept_to_index = {concept: index for index, concept in enumerate(concepts)}
    concept_to_cluster = _paper_cluster_by_concept()
    covered = [concept for concept in concepts if concept in concept_to_cluster]
    records: list[dict[str, Any]] = []
    within_values: list[float] = []
    between_values: list[float] = []

    for cluster_name, members in _PAPER_EMOTION_CLUSTERS.items():
        present = [concept for concept in members if concept in concept_to_index]
        missing = [concept for concept in members if concept not in concept_to_index]
        indices = [concept_to_index[concept] for concept in present]
        within = _mean_pairwise(cosine, indices)
        outside_indices = [index for concept, index in concept_to_index.items() if concept not in set(present)]
        between = _mean_cross(cosine, indices, outside_indices)
        if within is not None:
            within_values.append(within)
        if between is not None:
            between_values.append(between)
        records.append(
            {
                "cluster": cluster_name,
                "paper_size": len(members),
                "present_count": len(present),
                "missing": missing,
                "mean_within_cosine": within,
                "mean_to_other_clusters_cosine": between,
                "separation": (within - between) if within is not None and between is not None else None,
                "members": present,
            }
        )

    mean_within = float(np.mean(within_values)) if within_values else None
    mean_between = float(np.mean(between_values)) if between_values else None
    return {
        "reference": "Transformer Circuits emotions paper Table 12 clusters",
        "clusters": records,
        "summary": {
            "coverage": len(covered) / len(concepts) if concepts else None,
            "covered_count": len(covered),
            "concept_count": len(concepts),
            "mean_within_cosine": mean_within,
            "mean_between_cosine": mean_between,
            "separation": (mean_within - mean_between) if mean_within is not None and mean_between is not None else None,
        },
    }


def _kmeans_cluster_alignment(*, concepts: list[str], matrix: np.ndarray) -> dict[str, Any]:
    concept_to_cluster = _paper_cluster_by_concept()
    present = [concept for concept in concepts if concept in concept_to_cluster]
    if not present:
        return {"clusters": [], "summary": {"same_cluster_recall": None, "same_cluster_precision": None}}
    indices = np.asarray([concepts.index(concept) for concept in present], dtype=np.int64)
    X = matrix[indices]
    labels = _kmeans_labels(X, k=min(len(_PAPER_EMOTION_CLUSTERS), len(present)))
    true_labels = [concept_to_cluster[concept] for concept in present]
    recall, precision = _same_cluster_scores(true_labels=true_labels, predicted_labels=labels.tolist())
    cluster_records: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        members = [concept for concept, assigned in zip(present, labels.tolist(), strict=True) if assigned == label]
        paper_counts: dict[str, int] = {}
        for member in members:
            paper_counts[concept_to_cluster[member]] = paper_counts.get(concept_to_cluster[member], 0) + 1
        cluster_records.append(
            {
                "cluster_id": int(label),
                "size": len(members),
                "paper_cluster_counts": dict(sorted(paper_counts.items(), key=lambda item: (-item[1], item[0]))),
                "members": members,
            }
        )
    return {
        "method": "deterministic_kmeans_on_selected_layer_unit_vectors",
        "clusters": cluster_records,
        "summary": {
            "same_cluster_recall": recall,
            "same_cluster_precision": precision,
            "concept_count": len(present),
            "cluster_count": len(cluster_records),
        },
    }


def _nearest_neighbors(*, concepts: list[str], cosine: np.ndarray, count: int) -> list[dict[str, Any]]:
    pairs: list[tuple[float, str, str]] = []
    for row, concept_a in enumerate(concepts):
        for column in range(row + 1, len(concepts)):
            pairs.append((float(cosine[row, column]), concept_a, concepts[column]))
    pairs.sort(reverse=True)
    return [
        {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "cosine": value,
            "same_paper_cluster": _paper_cluster_by_concept().get(concept_a)
            == _paper_cluster_by_concept().get(concept_b),
            "paper_cluster_a": _paper_cluster_by_concept().get(concept_a),
            "paper_cluster_b": _paper_cluster_by_concept().get(concept_b),
        }
        for value, concept_a, concept_b in pairs[:count]
    ]


def _opposite_neighbors(*, concepts: list[str], cosine: np.ndarray, count: int) -> list[dict[str, Any]]:
    pairs: list[tuple[float, str, str]] = []
    for row, concept_a in enumerate(concepts):
        for column in range(row + 1, len(concepts)):
            pairs.append((float(cosine[row, column]), concept_a, concepts[column]))
    pairs.sort()
    return [
        {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "cosine": value,
            "paper_cluster_a": _paper_cluster_by_concept().get(concept_a),
            "paper_cluster_b": _paper_cluster_by_concept().get(concept_b),
        }
        for value, concept_a, concept_b in pairs[:count]
    ]


def _paper_comparison(
    *,
    concepts: list[str],
    available_layers: list[int],
    selected_layer: int,
    selected_explained: list[float],
    adjacent_similarity: float | None,
    anchor_geometry: dict[str, Any],
    paper_cluster_alignment: dict[str, Any],
    kmeans_alignment: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_summary = dict(anchor_geometry.get("summary", {}))
    cluster_summary = dict(paper_cluster_alignment.get("summary", {}))
    kmeans_summary = dict(kmeans_alignment.get("summary", {}))
    pc1 = selected_explained[0] if selected_explained else None
    pc2 = selected_explained[1] if len(selected_explained) > 1 else None
    return [
        {
            "paper_validation": "171 story-derived emotion vectors with neutral-PC denoising",
            "paper_result": "The paper computes vectors for 171 emotion words from synthetic stories and projects out neutral-transcript PCs.",
            "our_result": f"{len(concepts)} concepts found in local vector-space asset.",
            "status": "pass" if len(concepts) == 171 else "warn",
            "next_step": None if len(concepts) == 171 else "Regenerate vectors for the missing concepts.",
        },
        {
            "paper_validation": "PCA recovers valence/arousal-like structure",
            "paper_result": "The paper reports PC1 tracking valence and PC2/PC3 tracking arousal depending on layer.",
            "our_result": (
                f"Layer {selected_layer}: PC1={_format_float(pc1)}, PC2={_format_float(pc2)}, "
                f"valence_anchor_delta={_format_float(anchor_summary.get('valence_anchor_delta'))}, "
                f"arousal_anchor_delta={_format_float(anchor_summary.get('arousal_anchor_delta'))}."
            ),
            "status": (
                "pass"
                if _positive(anchor_summary.get("valence_anchor_delta"))
                and _positive(anchor_summary.get("arousal_anchor_delta"))
                else "warn"
            ),
            "next_step": "Add external valence/arousal ratings to replace anchor-only validation.",
        },
        {
            "paper_validation": "Opposite-valence concepts tend to anti-correlate",
            "paper_result": "The paper highlights negative cosine similarity between opposite-valence emotions.",
            "our_result": (
                "positive_to_negative_mean_cosine="
                f"{_format_float(anchor_summary.get('positive_to_negative_mean_cosine'))}"
            ),
            "status": (
                "pass"
                if _negative(anchor_summary.get("positive_to_negative_mean_cosine"))
                else "warn"
            ),
            "next_step": None,
        },
        {
            "paper_validation": "k=10 interpretable emotion clusters",
            "paper_result": "The paper clusters all 171 emotion vectors with k-means and reports 10 interpretable clusters.",
            "our_result": (
                f"paper_cluster_coverage={_format_float(cluster_summary.get('coverage'))}, "
                f"mean_within={_format_float(cluster_summary.get('mean_within_cosine'))}, "
                f"mean_between={_format_float(cluster_summary.get('mean_between_cosine'))}, "
                f"separation={_format_float(cluster_summary.get('separation'))}."
            ),
            "status": (
                "pass"
                if _positive(cluster_summary.get("separation")) and cluster_summary.get("coverage") == 1.0
                else "warn"
            ),
            "next_step": None,
        },
        {
            "paper_validation": "Recovered k-means clusters align with the paper cluster taxonomy",
            "paper_result": "The paper's Table 12 gives the 10 named emotion clusters ordered by valence.",
            "our_result": (
                f"same_cluster_recall={_format_float(kmeans_summary.get('same_cluster_recall'))}, "
                f"same_cluster_precision={_format_float(kmeans_summary.get('same_cluster_precision'))}."
            ),
            "status": "measured",
            "next_step": "Inspect the generated cluster membership table; exact cluster IDs are not expected to match one-for-one.",
        },
        {
            "paper_validation": "Representational similarity is stable across middle-to-late layers",
            "paper_result": "The paper correlates pairwise cosine-similarity matrices across layers and finds stable geometry.",
            "our_result": (
                f"{len(available_layers)} captured layers {available_layers}; "
                f"mean_adjacent_similarity={_format_float(adjacent_similarity)}."
            ),
            "status": "pass" if adjacent_similarity is not None and adjacent_similarity >= 0.90 else "warn",
            "next_step": None,
        },
        {
            "paper_validation": "Implicit-scenario activation and intensity sweeps",
            "paper_result": "The paper uses implicit prompts, pre-response assistant-boundary activations, and monotonic numeric scenario sweeps.",
            "our_result": "Not run in the local vector-only validation workflow.",
            "status": "needs_capture",
            "next_step": "Add a small capture workflow over paper-style implicit and intensity prompts.",
        },
        {
            "paper_validation": "Logit-lens token effects",
            "paper_result": "The paper projects vectors through the unembedding and checks top up/down tokens.",
            "our_result": "Not run in the local vector-only validation workflow.",
            "status": "needs_model_weights",
            "next_step": "Load the Llama unembed and compute top shifted tokens for selected emotions.",
        },
        {
            "paper_validation": "Causal steering of emotional continuations",
            "paper_result": "The paper steers prompts such as 'He feels' and checks target emotion-token logprob shifts.",
            "our_result": "Not run in the local vector-only validation workflow.",
            "status": "needs_steering_run",
            "next_step": "Run a small add-direction steering grid for 12 emotions at layer 52 and score target-token logprobs.",
        },
        {
            "paper_validation": "Preference and alignment-behavior steering",
            "paper_result": "The paper links emotion activations/steering to activity preferences and alignment eval behaviors.",
            "our_result": "Not run in the local vector-only validation workflow.",
            "status": "needs_new_eval",
            "next_step": "Only add this after the basic causal steering sanity check passes.",
        },
    ]


def _validation_coverage() -> list[dict[str, Any]]:
    return [
        {
            "paper_check": "emotion vector construction from generated stories with neutral-PC projection",
            "status": "complete",
            "current_implementation": "asset run produced the local emotion_space_results.json consumed here",
            "next_step": None,
        },
        {
            "paper_check": "emotion-space PCA geometry and valence/arousal-like axes",
            "status": "complete",
            "current_implementation": "validation.py computes PCA coordinates and layer-52 PC extremes",
            "next_step": "Add external human/LLM valence-arousal ratings for direct correlations.",
        },
        {
            "paper_check": "representational similarity across layers",
            "status": "complete",
            "current_implementation": "validation.py correlates pairwise emotion cosine matrices across captured layers",
            "next_step": None,
        },
        {
            "paper_check": "paper cluster structure over all 171 emotions",
            "status": "complete",
            "current_implementation": "validation.py compares layer-52 vectors to Table 12 paper clusters",
            "next_step": None,
        },
        {
            "paper_check": "human PAD / valence-arousal rating correlation",
            "status": "missing_data",
            "current_implementation": "anchor-oriented PCA only",
            "next_step": "Add a ratings table for overlapping emotions, then compute Pearson/Spearman correlations with PC1/PC2.",
        },
        {
            "paper_check": "activation on diverse heldout documents with top-token snippets",
            "status": "needs_capture",
            "current_implementation": None,
            "next_step": "Capture residuals on an external corpus and score all emotion vectors per token/span.",
        },
        {
            "paper_check": "dose/intensity monotonicity scenario",
            "status": "needs_capture",
            "current_implementation": None,
            "next_step": "Create text-only intensity sweeps, capture at the assistant pre-response boundary, and test target monotonicity.",
        },
        {
            "paper_check": "unembed / logit-lens token effects",
            "status": "needs_model_weights",
            "current_implementation": None,
            "next_step": "Load model unembed and report top upweighted/downweighted tokens for each vector.",
        },
        {
            "paper_check": "causal emotional-continuation steering",
            "status": "needs_steering_run",
            "current_implementation": None,
            "next_step": "Run a small patched-generation grid over emotions and strengths; score direct target-token logprob or generated text.",
        },
        {
            "paper_check": "self-reported activity preference correlation and steering",
            "status": "needs_preference_run",
            "current_implementation": None,
            "next_step": "Build activity-pair prompts, score A/B logits, then repeat with steering.",
        },
        {
            "paper_check": "naturalistic / alignment-eval transcript activations",
            "status": "needs_dataset",
            "current_implementation": None,
            "next_step": "Capture available eval transcripts and score emotion-vector traces over turns.",
        },
    ]


def _paper_cluster_by_concept() -> dict[str, str]:
    return {
        concept: cluster
        for cluster, concepts in _PAPER_EMOTION_CLUSTERS.items()
        for concept in concepts
    }


def _present_indices(concept_to_index: dict[str, int], concepts: tuple[str, ...]) -> list[int]:
    return [int(concept_to_index[concept]) for concept in concepts if concept in concept_to_index]


def _coordinate_mean(coordinates: np.ndarray, indices: list[int], *, component: int) -> float | None:
    if not indices or coordinates.ndim != 2 or coordinates.shape[1] <= component:
        return None
    return float(np.mean(coordinates[np.asarray(indices, dtype=np.int64), component]))


def _records_for_indices(concepts: list[str], indices: list[int]) -> list[str]:
    return [concepts[int(index)] for index in indices]


def _format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _negative(value: Any) -> bool:
    try:
        return float(value) < 0.0
    except (TypeError, ValueError):
        return False


def _mean_pairwise(matrix: np.ndarray, indices: list[int]) -> float | None:
    if len(indices) < 2:
        return None
    values = matrix[np.ix_(indices, indices)]
    triangle = values[np.triu_indices(len(indices), k=1)]
    return float(np.mean(triangle)) if triangle.size else None


def _mean_cross(matrix: np.ndarray, indices_a: list[int], indices_b: list[int]) -> float | None:
    if not indices_a or not indices_b:
        return None
    return float(np.mean(matrix[np.ix_(indices_a, indices_b)]))


def _kmeans_labels(matrix: np.ndarray, *, k: int, iterations: int = 80) -> np.ndarray:
    if matrix.shape[0] <= k:
        return np.arange(matrix.shape[0], dtype=np.int64)
    norms = np.linalg.norm(matrix, axis=1)
    first = int(np.argmax(norms))
    centers = [matrix[first]]
    distances = np.linalg.norm(matrix - centers[0], axis=1)
    for _ in range(1, k):
        next_index = int(np.argmax(distances))
        centers.append(matrix[next_index])
        distances = np.minimum(distances, np.linalg.norm(matrix - matrix[next_index], axis=1))
    centers_array = np.stack(centers, axis=0).astype(np.float32)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(iterations):
        new_labels = np.argmin(np.linalg.norm(matrix[:, None, :] - centers_array[None, :, :], axis=2), axis=1)
        new_centers = centers_array.copy()
        for label in range(k):
            members = matrix[new_labels == label]
            if members.size:
                new_centers[label] = members.mean(axis=0)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels.astype(np.int64)
        centers_array = new_centers.astype(np.float32)
    return labels


def _same_cluster_scores(*, true_labels: list[str], predicted_labels: list[int]) -> tuple[float | None, float | None]:
    true_same = 0
    predicted_same = 0
    true_and_predicted_same = 0
    for index in range(len(true_labels)):
        for other in range(index + 1, len(true_labels)):
            same_true = true_labels[index] == true_labels[other]
            same_predicted = predicted_labels[index] == predicted_labels[other]
            true_same += int(same_true)
            predicted_same += int(same_predicted)
            true_and_predicted_same += int(same_true and same_predicted)
    recall = true_and_predicted_same / true_same if true_same else None
    precision = true_and_predicted_same / predicted_same if predicted_same else None
    return recall, precision


def _write_validation_report(*, payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    report_root = _resolve_output_dir(output_dir)
    assets_dir = report_root / "assets"
    tables_dir = report_root / "tables"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    figures: list[dict[str, Any]] = []
    selected_layer = int(payload["summary"]["selected_layer"])
    selected = _selected_layer_payload(payload)
    if selected is not None:
        pca_path = assets_dir / f"layer{selected_layer}_pca_scatter.png"
        _plot_pca_scatter(payload=payload, layer=selected, output_path=pca_path)
        figures.append(
            {
                "path": str(pca_path),
                "title": f"Layer {selected_layer} PCA scatter",
                "chart_kind": "emotion_pca_scatter",
            }
        )
        similarity_path = assets_dir / f"layer{selected_layer}_emotion_similarity.png"
        _plot_concept_similarity(layer=selected, output_path=similarity_path)
        figures.append(
            {
                "path": str(similarity_path),
                "title": f"Layer {selected_layer} emotion similarity",
                "chart_kind": "emotion_similarity_heatmap",
            }
        )
        cluster_pca_path = assets_dir / f"layer{selected_layer}_paper_clusters_pca.png"
        _plot_paper_clusters_pca(payload=payload, layer=selected, output_path=cluster_pca_path)
        figures.append(
            {
                "path": str(cluster_pca_path),
                "title": f"Layer {selected_layer} paper clusters",
                "chart_kind": "emotion_paper_cluster_pca",
            }
        )

    explained_path = assets_dir / "explained_variance_by_layer.png"
    _plot_explained_variance(payload=payload, output_path=explained_path)
    figures.append(
        {
            "path": str(explained_path),
            "title": "PCA explained variance by layer",
            "chart_kind": "emotion_pca_variance",
        }
    )

    layer_similarity_path = assets_dir / "layer_similarity_heatmap.png"
    _plot_layer_similarity(payload=payload, output_path=layer_similarity_path)
    figures.append(
        {
            "path": str(layer_similarity_path),
            "title": "Layer-to-layer geometry similarity",
            "chart_kind": "emotion_layer_similarity_heatmap",
        }
    )

    pc1_extremes_path = assets_dir / f"layer{selected_layer}_pc1_extremes.png"
    _plot_pc1_extremes(payload=payload, output_path=pc1_extremes_path)
    figures.append(
        {
            "path": str(pc1_extremes_path),
            "title": f"Layer {selected_layer} PC1 extremes",
            "chart_kind": "emotion_pc_extremes",
        }
    )
    paper_cluster_alignment_path = assets_dir / f"layer{selected_layer}_paper_cluster_alignment.png"
    _plot_paper_cluster_alignment(payload=payload, output_path=paper_cluster_alignment_path)
    figures.append(
        {
            "path": str(paper_cluster_alignment_path),
            "title": f"Layer {selected_layer} paper cluster alignment",
            "chart_kind": "emotion_paper_cluster_alignment",
        }
    )

    layer_rows = [_layer_summary_row(layer) for layer in payload.get("layers", []) if isinstance(layer, dict)]
    layer_summary_path = tables_dir / "layer_summary.json"
    _write_json(layer_summary_path, {"rows": layer_rows})
    nearest_neighbors_path = tables_dir / "nearest_neighbors.json"
    _write_json(nearest_neighbors_path, {"rows": payload.get("nearest_neighbors", [])})
    opposite_neighbors_path = tables_dir / "opposite_neighbors.json"
    _write_json(opposite_neighbors_path, {"rows": payload.get("opposite_neighbors", [])})
    cluster_alignment_path = tables_dir / "paper_cluster_alignment.json"
    _write_json(cluster_alignment_path, payload.get("paper_cluster_alignment", {}))
    validation_coverage_path = tables_dir / "validation_coverage.json"
    _write_json(validation_coverage_path, {"rows": payload.get("validation_coverage", [])})
    paper_comparison_path = tables_dir / "paper_comparison.json"
    _write_json(paper_comparison_path, {"rows": payload.get("paper_comparison", [])})

    validation_result_path = report_root / "validation_result.json"
    summary_path = report_root / "summary.json"
    report_path = report_root / "report.md"
    _write_json(validation_result_path, payload)
    _write_json(summary_path, payload.get("summary", {}))
    _write_text(report_path, _render_markdown(payload=payload, figures=figures, layer_rows=layer_rows))

    return {
        "output_dir": str(report_root),
        "assets_dir": str(assets_dir),
        "tables_dir": str(tables_dir),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "validation_result_path": str(validation_result_path),
        "layer_summary_path": str(layer_summary_path),
        "nearest_neighbors_path": str(nearest_neighbors_path),
        "opposite_neighbors_path": str(opposite_neighbors_path),
        "cluster_alignment_path": str(cluster_alignment_path),
        "validation_coverage_path": str(validation_coverage_path),
        "paper_comparison_path": str(paper_comparison_path),
        "figures": figures,
    }


def _resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _selected_layer_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    selected_layer = int(payload["summary"]["selected_layer"])
    layers = [dict(layer) for layer in payload.get("layers", []) if isinstance(layer, dict)]
    for layer in layers:
        if int(layer.get("layer", -1)) == selected_layer:
            return layer
    return layers[len(layers) // 2] if layers else None


def _plot_pca_scatter(*, payload: dict[str, Any], layer: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    concepts = [str(item) for item in layer.get("concepts", [])]
    coords = np.asarray((layer.get("pca") or {}).get("coordinates", []), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2 or not concepts:
        return
    label_concepts = {str(item) for item in payload.get("label_concepts", [])}
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=coords[:, 0],
        cmap="coolwarm",
        s=28,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.25,
    )
    for concept, x_value, y_value in zip(concepts, coords[:, 0], coords[:, 1], strict=False):
        if concept not in label_concepts:
            continue
        ax.text(float(x_value), float(y_value), _display_name(concept), fontsize=6.4, alpha=0.82)
    ax.set_title(f"Emotion PCA, Layer {int(layer.get('layer', 0))}")
    ax.set_xlabel("PC1 (positive-valence anchors oriented positive)")
    ax.set_ylabel("PC2 (high-arousal anchors oriented positive)")
    ax.grid(alpha=0.18)
    fig.colorbar(scatter, ax=ax, fraction=0.032, pad=0.018).set_label("PC1")
    _save_figure(fig, output_path)


def _plot_concept_similarity(*, layer: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    cosine = np.asarray(layer.get("cosine_similarity", []), dtype=np.float32)
    concepts = [str(item) for item in layer.get("concepts", [])]
    if cosine.ndim != 2 or cosine.shape[0] != cosine.shape[1] or not concepts:
        return
    order = [int(index) for index in layer.get("clustered_order", []) if 0 <= int(index) < len(concepts)]
    if len(order) != len(concepts):
        order = list(range(len(concepts)))
    matrix = cosine[np.ix_(order, order)]
    ordered_concepts = [concepts[index] for index in order]
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="nearest", aspect="auto")
    ticks = _sample_indices(len(ordered_concepts), desired=16)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([_display_name(ordered_concepts[index]) for index in ticks], rotation=90, fontsize=6.2)
    ax.set_yticklabels([_display_name(ordered_concepts[index]) for index in ticks], fontsize=6.2)
    ax.set_title(f"Emotion Vector Cosine Similarity, Layer {int(layer.get('layer', 0))}")
    fig.colorbar(image, ax=ax, fraction=0.032, pad=0.018).set_label("cosine")
    _save_figure(fig, output_path)


def _plot_paper_clusters_pca(*, payload: dict[str, Any], layer: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    concepts = [str(item) for item in layer.get("concepts", [])]
    coords = np.asarray((layer.get("pca") or {}).get("coordinates", []), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2 or not concepts:
        return
    concept_to_cluster = _paper_cluster_by_concept()
    cluster_names = list(_PAPER_EMOTION_CLUSTERS)
    color_map = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    for cluster_index, cluster_name in enumerate(cluster_names):
        indices = [index for index, concept in enumerate(concepts) if concept_to_cluster.get(concept) == cluster_name]
        if not indices:
            continue
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            label=cluster_name,
            s=28,
            alpha=0.86,
            color=color_map(cluster_index % 10),
            edgecolors="white",
            linewidths=0.25,
        )
    label_concepts = {str(item) for item in payload.get("label_concepts", [])}
    for concept, x_value, y_value in zip(concepts, coords[:, 0], coords[:, 1], strict=False):
        if concept not in label_concepts:
            continue
        ax.text(float(x_value), float(y_value), _display_name(concept), fontsize=6.0, alpha=0.78)
    ax.set_title(f"Paper Cluster Overlay, Layer {int(layer.get('layer', 0))}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=6.6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _save_figure(fig, output_path)


def _plot_explained_variance(*, payload: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    layers = [dict(layer) for layer in payload.get("layers", []) if isinstance(layer, dict)]
    if not layers:
        return
    layer_indices = [int(layer.get("layer", 0)) for layer in layers]
    max_components = max((len(_explained(layer)) for layer in layers), default=0)
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for component_index in range(max_components):
        values = [
            _explained(layer)[component_index] if component_index < len(_explained(layer)) else np.nan
            for layer in layers
        ]
        ax.plot(layer_indices, values, marker="o", linewidth=1.6, label=f"PC{component_index + 1}")
    ax.set_title("PCA Explained Variance Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Explained variance ratio")
    ax.set_xticks(layer_indices)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _plot_layer_similarity(*, payload: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    layer_similarity = payload.get("layer_similarity")
    if not isinstance(layer_similarity, dict):
        return
    layers = [int(layer) for layer in layer_similarity.get("layers", [])]
    matrix = np.asarray(layer_similarity.get("matrix", []), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not layers:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_yticks(range(len(layers)))
    ax.set_xticklabels([f"L{layer}" for layer in layers], rotation=45, ha="right")
    ax.set_yticklabels([f"L{layer}" for layer in layers])
    for row in range(len(layers)):
        for column in range(len(layers)):
            ax.text(column, row, f"{float(matrix[row, column]):.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Layer-to-Layer Emotion Geometry Similarity")
    fig.colorbar(image, ax=ax, fraction=0.040, pad=0.024).set_label("r")
    _save_figure(fig, output_path)


def _plot_pc1_extremes(*, payload: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    pc_extremes = payload.get("pc_extremes")
    if not isinstance(pc_extremes, dict):
        return
    negative = list(pc_extremes.get("pc1_negative") or [])[:10]
    positive = list(pc_extremes.get("pc1_positive") or [])[:10]
    rows = list(reversed(negative)) + positive
    if not rows:
        return
    concepts = [_display_name(str(row.get("concept"))) for row in rows]
    values = [float(row.get("value", 0.0)) for row in rows]
    colors = ["#2c6fa5" if value < 0 else "#b8722e" for value in values]
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    y_positions = np.arange(len(rows))
    ax.barh(y_positions, values, color=colors, alpha=0.92)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.axvline(0.0, color="#6c7689", linewidth=0.9, linestyle=(0, (1, 2)))
    ax.set_title(f"Layer {payload['summary']['selected_layer']} PC1 Extremes")
    ax.set_xlabel("PC1 coordinate")
    ax.grid(axis="x", alpha=0.18)
    _save_figure(fig, output_path)


def _plot_paper_cluster_alignment(*, payload: dict[str, Any], output_path: Path) -> None:
    plt = _plt()
    alignment = payload.get("paper_cluster_alignment")
    if not isinstance(alignment, dict):
        return
    rows = [dict(row) for row in alignment.get("clusters", []) if isinstance(row, dict)]
    rows = [row for row in rows if row.get("separation") is not None]
    if not rows:
        return
    labels = [str(row["cluster"]) for row in rows]
    within = [float(row["mean_within_cosine"]) for row in rows]
    between = [float(row["mean_to_other_clusters_cosine"]) for row in rows]
    y_positions = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.barh(y_positions - 0.18, within, height=0.34, label="within cluster", color="#2c6fa5", alpha=0.90)
    ax.barh(y_positions + 0.18, between, height=0.34, label="to other clusters", color="#b8722e", alpha=0.82)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Mean cosine similarity")
    ax.set_title("Alignment With Paper Emotion Clusters")
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _render_markdown(
    *,
    payload: dict[str, Any],
    figures: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
) -> str:
    summary = dict(payload.get("summary", {}))
    source = dict(payload.get("source", {}))
    lines = [
        "# Llama 3.3 70B Emotion Vector Validation",
        "",
        f"- source: `{source.get('emotion_space_path')}`",
        f"- selected_layer: `{summary.get('selected_layer')}`",
        f"- concept_count: `{summary.get('concept_count')}`",
        f"- layer_count: `{summary.get('layer_count')}`",
        f"- mean_adjacent_layer_similarity: `{summary.get('mean_adjacent_layer_similarity')}`",
        f"- valence_anchor_delta: `{summary.get('valence_anchor_delta')}`",
        f"- arousal_anchor_delta: `{summary.get('arousal_anchor_delta')}`",
        f"- opposite_valence_mean_cosine: `{summary.get('opposite_valence_mean_cosine')}`",
        f"- paper_cluster_separation: `{summary.get('paper_cluster_separation')}`",
        f"- kmeans_paper_same_cluster_recall: `{summary.get('kmeans_paper_same_cluster_recall')}`",
        f"- validation_checks_complete: `{summary.get('validation_checks_complete')}/{summary.get('validation_checks_total')}`",
        f"- paper_comparison_ready_checks: `{summary.get('paper_comparison_ready_checks')}/{summary.get('paper_comparison_total_checks')}`",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        relative_path = os.path.relpath(figure["path"], start=str(Path(figure["path"]).parents[1]))
        lines.extend([f"### {figure['title']}", "", f"![{figure['title']}]({relative_path})", ""])
    lines.extend(["## Layer Summary", "", "```json", json.dumps(layer_rows, indent=2, sort_keys=True), "```", ""])
    coverage = payload.get("validation_coverage")
    if isinstance(coverage, list):
        lines.extend(["## Paper Validation Coverage", "", "```json", json.dumps(coverage, indent=2, sort_keys=True), "```", ""])
    paper_comparison = payload.get("paper_comparison")
    if isinstance(paper_comparison, list):
        lines.extend(["## Paper Comparison", "", "```json", json.dumps(paper_comparison, indent=2, sort_keys=True), "```", ""])
    nearest = payload.get("nearest_neighbors")
    if isinstance(nearest, list):
        lines.extend(["## Top Nearest Emotion Pairs", "", "```json", json.dumps(nearest[:20], indent=2, sort_keys=True), "```", ""])
    opposite = payload.get("opposite_neighbors")
    if isinstance(opposite, list):
        lines.extend(["## Lowest Cosine Emotion Pairs", "", "```json", json.dumps(opposite[:20], indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def _plot_numeric_semantics(*, payload: dict[str, Any], output_path: Path, layer: int | None = None) -> None:
    plt = _plt()
    colors = {
        "happy": "#2ca02c",
        "sad": "#ff7f0e",
        "afraid": "#d62728",
        "calm": "#1f77b4",
    }
    points = [
        dict(row)
        for row in payload.get("points", [])
        if isinstance(row, dict) and (layer is None or int(row.get("layer", -1)) == int(layer))
    ]
    by_scenario_emotion: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for point in points:
        by_scenario_emotion.setdefault((str(point["scenario"]), str(point["emotion"])), []).append(point)

    scenarios = numeric_semantics_scenarios()
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.4), sharey=True)
    title = "Emotion Probes Track Numerical Semantics"
    if layer is not None:
        title = f"{title} - Layer {int(layer)}"
    fig.suptitle(title, fontsize=17, color="#bf5b3c", y=0.985)

    for axis, scenario in zip(axes.ravel(), scenarios, strict=True):
        scenario_key = str(scenario["key"])
        values = list(scenario["values"])
        x_positions = np.arange(len(values), dtype=np.float32)
        for emotion in _NUMERIC_EMOTIONS:
            series = sorted(
                by_scenario_emotion.get((scenario_key, emotion), []),
                key=lambda row: int(row["x_index"]),
            )
            score_by_index = {int(row["x_index"]): float(row["score"]) for row in series}
            y_values = [score_by_index.get(index, np.nan) for index in range(len(values))]
            if all(np.isnan(value) for value in y_values):
                continue
            axis.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=2.1,
                markersize=4.2,
                color=colors[emotion],
            )
            last_valid = next((idx for idx in range(len(y_values) - 1, -1, -1) if not np.isnan(y_values[idx])), None)
            if last_valid is not None:
                axis.text(
                    float(x_positions[last_valid]) + 0.12,
                    float(y_values[last_valid]),
                    _display_name(emotion),
                    color=colors[emotion],
                    fontsize=8.0,
                    va="center",
                )
        axis.axhline(0.0, color="#b8b8b8", linewidth=0.8)
        axis.set_title(str(scenario["title"]), fontsize=10.5, pad=10)
        axis.set_xlabel(str(scenario["x_label"]), fontsize=11)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(list(scenario.get("x_tick_labels") or [str(value) for value in values]), fontsize=8.8)
        axis.set_ylim(-0.10, 0.10)
        axis.grid(axis="y", alpha=0.18)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Cosine Similarity", fontsize=11)
    _save_figure(fig, output_path)


def _plot_dosage_context(*, payload: dict[str, Any], output_path: Path) -> None:
    from matplotlib.colors import TwoSlopeNorm

    plt = _plt()
    concept = str(payload.get("concept") or _DOSAGE_CONTEXT_CONCEPT)
    baseline_label = str(payload.get("baseline_label") or payload.get("safe_dose_mg") or "baseline")
    target_label = str(payload.get("target_label") or payload.get("dangerous_dose_mg") or "target")
    layers = [int(layer) for layer in payload.get("layers", [])]
    token_scores = [dict(row) for row in payload.get("token_scores", []) if isinstance(row, dict)]
    diff_rows = [dict(row) for row in payload.get("difference_rows", []) if isinstance(row, dict)]
    token_labels = _dosage_context_plot_labels(token_scores=token_scores, role="baseline")

    safe_matrix = _dosage_context_score_matrix(token_scores=token_scores, role="baseline", layers=layers)
    diff_matrix = _dosage_context_difference_matrix(diff_rows=diff_rows, layers=layers)
    token_count = max(safe_matrix.shape[1], diff_matrix.shape[1], len(token_labels))
    if len(token_labels) < token_count:
        token_labels = [*token_labels, *([""] * (token_count - len(token_labels)))]

    safe_limit = _symmetric_limit(safe_matrix, default=0.06)
    diff_limit = _symmetric_limit(diff_matrix, default=0.05)
    fig = plt.figure(figsize=(10.5, 12.3))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(24, 0.8),
        height_ratios=(1.0, 1.0, 1.08),
        hspace=0.58,
        wspace=0.08,
    )
    ax_safe = fig.add_subplot(grid[0, 0])
    cax_safe = fig.add_subplot(grid[0, 1])
    ax_diff = fig.add_subplot(grid[1, 0])
    cax_diff = fig.add_subplot(grid[1, 1])
    ax_line = fig.add_subplot(grid[2, 0])
    fig.add_subplot(grid[2, 1]).axis("off")

    fig.suptitle(
        "Numerical Context Modulates Emotional Response",
        fontsize=17,
        color="#bf5b3c",
        y=0.985,
    )
    fig.text(0.5, 0.955, str(payload.get("title") or ""), ha="center", va="top", fontsize=10.5)
    safe_image = ax_safe.imshow(
        np.ma.masked_invalid(safe_matrix),
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-safe_limit, vmax=safe_limit),
    )
    fig.colorbar(safe_image, cax=cax_safe, label="Cosine Similarity")
    ax_safe.set_title(f"{_display_name(concept)} probe at {baseline_label}", fontsize=12)
    _format_dosage_heatmap_axis(ax_safe, layers=layers, token_labels=token_labels, payload=payload)

    diff_image = ax_diff.imshow(
        np.ma.masked_invalid(diff_matrix),
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-diff_limit, vmax=diff_limit),
    )
    fig.colorbar(diff_image, cax=cax_diff, label="Delta Cosine Similarity")
    ax_diff.set_title(
        f"{_display_name(concept)} probe difference ({target_label} - {baseline_label})",
        fontsize=12,
    )
    _format_dosage_heatmap_axis(ax_diff, layers=layers, token_labels=token_labels, payload=payload)

    early_layers, late_layers = _dosage_context_line_layer_ranges(payload)
    x_values = np.arange(token_count, dtype=np.int32)
    early_delta = _dosage_context_mean_delta_by_token(
        diff_rows=diff_rows,
        layers=early_layers,
        token_count=token_count,
    )
    late_delta = _dosage_context_mean_delta_by_token(
        diff_rows=diff_rows,
        layers=late_layers,
        token_count=token_count,
    )
    ax_line.plot(
        x_values,
        early_delta,
        color="#377eb8",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=_dosage_context_range_label("Early -> Early-Mid", early_layers),
    )
    ax_line.plot(
        x_values,
        late_delta,
        color="#d81b42",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=_dosage_context_range_label("Mid-Late -> Late", late_layers),
    )
    ax_line.axhline(0.0, color="#8f8f8f", linewidth=0.9)
    ax_line.set_title("Mean Difference by Layer Range", fontsize=12)
    ax_line.set_ylabel("Delta Cosine Similarity", fontsize=10.5)
    ax_line.grid(alpha=0.24)
    tick_positions, tick_labels = _dosage_context_xticks(token_labels)
    ax_line.set_xticks(tick_positions)
    ax_line.set_xticklabels(tick_labels, rotation=58, ha="right", fontsize=7.4)
    ax_line.set_xlim(-0.5, max(0.5, token_count - 0.5))
    ax_line.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=9.5, frameon=True)

    _save_figure(fig, output_path)


def _plot_numeric_context_surface(*, payload: dict[str, Any], output_path: Path) -> None:
    from matplotlib.colors import TwoSlopeNorm

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    plt = _plt()
    concept = str(payload.get("concept") or _DOSAGE_CONTEXT_CONCEPT)
    baseline_label = str(payload.get("baseline_label") or payload.get("safe_dose_mg") or "baseline")
    target_label = str(payload.get("target_label") or payload.get("dangerous_dose_mg") or "target")
    layers = [int(layer) for layer in payload.get("layers", [])]
    token_scores = [dict(row) for row in payload.get("token_scores", []) if isinstance(row, dict)]
    diff_rows = [dict(row) for row in payload.get("difference_rows", []) if isinstance(row, dict)]
    token_labels = _dosage_context_plot_labels(token_scores=token_scores, role="target")
    if not token_labels:
        token_labels = _dosage_context_plot_labels(token_scores=token_scores, role="baseline")

    target_matrix = _dosage_context_score_matrix(token_scores=token_scores, role="target", layers=layers)
    if target_matrix.size == 0:
        target_matrix = _dosage_context_score_matrix(token_scores=token_scores, role="baseline", layers=layers)
    diff_matrix = _dosage_context_difference_matrix(diff_rows=diff_rows, layers=layers)
    token_count = max(target_matrix.shape[1], diff_matrix.shape[1], len(token_labels))
    if len(layers) < 2 or token_count < 2:
        return
    if len(token_labels) < token_count:
        token_labels = [*token_labels, *([""] * (token_count - len(token_labels)))]
    target_matrix = _pad_columns(target_matrix, token_count)
    diff_matrix = _pad_columns(diff_matrix, token_count)

    x_values = np.arange(token_count, dtype=np.float32)
    y_values = np.asarray(layers, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    target_limit = _symmetric_limit(target_matrix, default=0.06)
    diff_limit = _symmetric_limit(diff_matrix, default=0.05)

    fig = plt.figure(figsize=(14.2, 7.2))
    fig.suptitle(
        f"{_display_name(concept)} Probe Token/Layer Sweep",
        fontsize=16,
        color="#bf5b3c",
        y=0.98,
    )
    ax_target = fig.add_subplot(1, 2, 1, projection="3d")
    ax_diff = fig.add_subplot(1, 2, 2, projection="3d")

    target_surface = _plot_surface_matrix(
        axis=ax_target,
        x_grid=x_grid,
        y_grid=y_grid,
        matrix=target_matrix,
        title=f"{target_label} score",
        z_label="Cosine similarity",
        limit=target_limit,
    )
    diff_surface = _plot_surface_matrix(
        axis=ax_diff,
        x_grid=x_grid,
        y_grid=y_grid,
        matrix=diff_matrix,
        title=f"{target_label} - {baseline_label}",
        z_label="Delta cosine",
        limit=diff_limit,
    )
    tick_positions, tick_labels = _dosage_context_xticks(token_labels, max_ticks=18)
    for axis in (ax_target, ax_diff):
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels, rotation=58, ha="right", fontsize=7.0)
        axis.set_yticks(layers)
        axis.set_yticklabels([str(layer) for layer in layers], fontsize=7.5)
        axis.view_init(elev=27, azim=-58)
    fig.colorbar(target_surface, ax=ax_target, shrink=0.58, pad=0.08)
    fig.colorbar(diff_surface, ax=ax_diff, shrink=0.58, pad=0.08)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, output_path)


def _plot_surface_matrix(
    *,
    axis: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    matrix: np.ndarray,
    title: str,
    z_label: str,
    limit: float,
) -> Any:
    from matplotlib.colors import TwoSlopeNorm

    z_values = np.ma.masked_invalid(np.asarray(matrix, dtype=np.float32))
    surface = axis.plot_surface(
        x_grid,
        y_grid,
        z_values,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-float(limit), vmax=float(limit)),
        linewidth=0,
        antialiased=True,
        alpha=0.94,
    )
    axis.set_title(title, fontsize=11.5)
    axis.set_xlabel("Token", fontsize=9.5, labelpad=8)
    axis.set_ylabel("Layer", fontsize=9.5, labelpad=8)
    axis.set_zlabel(z_label, fontsize=9.5, labelpad=8)
    axis.set_zlim(-float(limit), float(limit))
    return surface


def _format_dosage_heatmap_axis(
    axis: Any,
    *,
    layers: list[int],
    token_labels: list[str],
    payload: dict[str, Any],
) -> None:
    axis.set_ylabel("Layer", fontsize=10.5)
    tick_positions, tick_labels = _dosage_context_xticks(token_labels)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels, rotation=58, ha="right", fontsize=7.2)
    y_positions, y_labels = _dosage_context_y_ticks(layers=layers, payload=payload)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(y_labels, fontsize=9.5)


def _dosage_context_score_matrix(
    *,
    token_scores: list[dict[str, Any]],
    role: str,
    layers: list[int],
) -> np.ndarray:
    by_layer = {
        int(row["layer"]): [float(value) for value in row.get("scores", [])]
        for row in token_scores
        if str(row.get("contrast_role") or "") == str(role)
    }
    token_count = max((len(values) for values in by_layer.values()), default=0)
    matrix = np.full((len(layers), token_count), np.nan, dtype=np.float32)
    for layer_index, layer in enumerate(layers):
        values = by_layer.get(int(layer), [])
        matrix[layer_index, : len(values)] = np.asarray(values, dtype=np.float32)
    return matrix


def _dosage_context_difference_matrix(*, diff_rows: list[dict[str, Any]], layers: list[int]) -> np.ndarray:
    token_count = max((int(row.get("token_index", -1)) for row in diff_rows), default=-1) + 1
    matrix = np.full((len(layers), token_count), np.nan, dtype=np.float32)
    layer_index = {int(layer): index for index, layer in enumerate(layers)}
    for row in diff_rows:
        layer = int(row.get("layer", -1))
        token_index = int(row.get("token_index", -1))
        if layer not in layer_index or token_index < 0:
            continue
        matrix[layer_index[layer], token_index] = float(row.get("delta", np.nan))
    return matrix


def _pad_columns(matrix: np.ndarray, column_count: int) -> np.ndarray:
    if matrix.shape[1] >= int(column_count):
        return matrix[:, : int(column_count)]
    padded = np.full((matrix.shape[0], int(column_count)), np.nan, dtype=np.float32)
    padded[:, : matrix.shape[1]] = matrix
    return padded


def _dosage_context_plot_labels(*, token_scores: list[dict[str, Any]], role: str) -> list[str]:
    candidates = [
        row
        for row in token_scores
        if isinstance(row, dict) and str(row.get("contrast_role") or "") == str(role)
    ]
    if not candidates:
        return []
    first = sorted(candidates, key=lambda row: int(row.get("layer", 0)))[0]
    labels = first.get("token_labels")
    return [str(label) for label in labels] if isinstance(labels, list) else []


def _dosage_context_xticks(token_labels: list[str], *, max_ticks: int = 36) -> tuple[list[int], list[str]]:
    labeled = [(index, label) for index, label in enumerate(token_labels) if str(label).strip()]
    if not labeled and token_labels:
        labeled = [(index, str(index)) for index in _sample_indices(len(token_labels), desired=max_ticks)]
    if len(labeled) > max_ticks:
        keep = set(_sample_indices(len(labeled), desired=max_ticks))
        labeled = [item for index, item in enumerate(labeled) if index in keep]
    return [int(index) for index, _ in labeled], [str(label) for _, label in labeled]


def _dosage_context_y_ticks(*, layers: list[int], payload: dict[str, Any]) -> tuple[list[float], list[str]]:
    groups = [dict(group) for group in payload.get("layer_groups", []) if isinstance(group, dict)]
    if not groups:
        return list(range(len(layers))), [str(layer) for layer in layers]
    index_by_layer = {int(layer): index for index, layer in enumerate(layers)}
    ticks: list[float] = []
    labels: list[str] = []
    for group in groups:
        group_layers = [int(layer) for layer in group.get("layers", []) if int(layer) in index_by_layer]
        if not group_layers:
            continue
        ticks.append(float(np.mean([index_by_layer[layer] for layer in group_layers])))
        labels.append(str(group.get("name") or "-"))
    return ticks, labels


def _dosage_context_line_layer_ranges(payload: dict[str, Any]) -> tuple[list[int], list[int]]:
    layers = [int(layer) for layer in payload.get("layers", [])]
    groups = [dict(group) for group in payload.get("layer_groups", []) if isinstance(group, dict)]
    if len(groups) >= 4:
        early = [int(layer) for group in groups[:2] for layer in group.get("layers", [])]
        late = [int(layer) for group in groups[2:] for layer in group.get("layers", [])]
        return early, late
    midpoint = max(1, len(layers) // 2)
    return layers[:midpoint], layers[midpoint:] or layers[-1:]


def _dosage_context_mean_delta_by_token(
    *,
    diff_rows: list[dict[str, Any]],
    layers: list[int],
    token_count: int,
) -> list[float]:
    layer_set = {int(layer) for layer in layers}
    values_by_token: dict[int, list[float]] = {index: [] for index in range(int(token_count))}
    for row in diff_rows:
        if int(row.get("layer", -1)) not in layer_set:
            continue
        token_index = int(row.get("token_index", -1))
        if token_index < 0 or token_index >= int(token_count):
            continue
        values_by_token[token_index].append(float(row.get("delta", 0.0)))
    return [
        float(np.mean(values_by_token[index])) if values_by_token[index] else 0.0
        for index in range(int(token_count))
    ]


def _dosage_context_range_label(default: str, layers: list[int]) -> str:
    if not layers:
        return default
    return f"{default} ({min(layers)}-{max(layers)})"


def _symmetric_limit(matrix: np.ndarray, *, default: float) -> float:
    finite = np.asarray(matrix, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float(default)
    limit = float(np.nanpercentile(np.abs(finite), 98))
    return max(limit, float(default))


def _render_numeric_semantics_markdown(
    *,
    payload: dict[str, Any],
    figure_paths: tuple[Path, ...],
    preferred_layer: int,
) -> str:
    summary = dict(payload.get("summary", {}))
    checks = [dict(row) for row in payload.get("checks", []) if isinstance(row, dict)]
    failed = [check for check in checks if not check.get("passed")]
    lines = [
        "# Llama 3.3 70B Numerical Semantics Validation",
        "",
        f"- layers: `{summary.get('layers')}`",
        f"- scenario_count: `{summary.get('scenario_count')}`",
        f"- prompt_count: `{summary.get('prompt_count')}`",
        f"- point_count: `{summary.get('point_count')}`",
        f"- passed_checks: `{summary.get('passed_checks')}/{summary.get('check_count')}`",
        f"- passed_checks_by_layer: `{summary.get('passed_checks_by_layer')}`",
        f"- score_metric: `{summary.get('score_metric')}`",
        "",
        f"## Preferred Layer {int(preferred_layer)}",
        "",
        f"![Emotion probes track numerical semantics](assets/numeric_semantics_layer{int(preferred_layer)}.png)",
        "",
        "## Figures By Layer",
        "",
    ]
    for figure_path in figure_paths:
        layer_text = figure_path.stem.removeprefix("numeric_semantics_layer")
        lines.extend(
            [
                f"### Layer {layer_text}",
                "",
                f"![Emotion probes layer {layer_text}](assets/{figure_path.name})",
                "",
            ]
        )
    lines.extend(
        [
        "## Direction Checks",
        "",
        "```json",
        json.dumps(checks, indent=2, sort_keys=True),
        "```",
        "",
        ]
    )
    if failed:
        lines.extend(
            [
                "## Checks To Inspect",
                "",
                "```json",
                json.dumps(failed, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _render_dosage_context_markdown(*, payload: dict[str, Any], figures: list[dict[str, str]]) -> str:
    summary = dict(payload.get("summary", {}))
    layer_groups = [dict(row) for row in payload.get("layer_groups", []) if isinstance(row, dict)]
    lines = [
        "# Llama 3.3 70B Numeric Context Validation",
        "",
        f"- scenario_count: `{summary.get('scenario_count')}`",
        f"- scenarios: `{summary.get('scenarios')}`",
        f"- layers: `{payload.get('layers')}`",
        f"- late_layer_mean_delta: `{summary.get('late_layer_mean_delta')}`",
        f"- all_layer_mean_delta: `{summary.get('all_layer_mean_delta')}`",
        "",
        "## Figures",
        "",
    ]
    for figure in figures:
        lines.extend(
            [
                f"### {figure['title']}",
                "",
                f"![{figure['title']}](assets/{Path(figure['path']).name})",
                "",
            ]
        )
    lines.extend(
        [
            "## Layer Groups",
            "",
            "```json",
            json.dumps(layer_groups, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _layer_summary_row(layer: dict[str, Any]) -> dict[str, Any]:
    explained = _explained(layer)
    return {
        "layer": int(layer.get("layer", 0)),
        "concept_count": len(layer.get("concepts") or []),
        "pc1_variance": explained[0] if len(explained) > 0 else None,
        "pc2_variance": explained[1] if len(explained) > 1 else None,
        "pc3_variance": explained[2] if len(explained) > 2 else None,
    }


def _explained(layer: dict[str, Any]) -> list[float]:
    pca = layer.get("pca")
    if not isinstance(pca, dict):
        return []
    values = pca.get("explained_variance_ratio")
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def _sample_indices(count: int, *, desired: int) -> list[int]:
    if count <= 0:
        return []
    if count <= desired:
        return list(range(count))
    return sorted({int(round(value)) for value in np.linspace(0, count - 1, desired)})


def _display_name(value: str) -> str:
    return " ".join(token[:1].upper() + token[1:] for token in str(value).replace("_", " ").split())


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    _plt().close(fig)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp, path)

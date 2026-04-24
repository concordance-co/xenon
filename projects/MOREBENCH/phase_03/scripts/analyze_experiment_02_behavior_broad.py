from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
GENERATION_ARTIFACT_ID = "generation_run_1_82f18dea7736"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports" / "experiment_02_behavior_broad_analysis"

PRIME_ORDER = (
    "utilitarian",
    "virtue_ethics",
    "contractarianism",
    "deontology",
    "contractualism",
    "generic_ethics_control",
)
DEONTIC_CLUSTER = {"deontology", "virtue_ethics"}
WELFARIST_CLUSTER = {"utilitarian", "contractarianism", "generic_ethics_control"}


@dataclass(frozen=True, slots=True)
class RowRecord:
    example_key: str
    group_id: str
    prime_condition: str
    source_case_theory: str
    source_family: str
    role_domain: str
    context: str
    prompt_text: str
    generated_text: str
    question: str
    recommendation_span: str
    decision_sentence: str
    action_text: str


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_operation_artifact(artifact_id: str) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else dict(value or {}) if isinstance(value, dict | defaultdict) else {}


def _render_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, dict):
                role = str(message.get("role") or "").strip()
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(item.get("text") or "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                label = role.capitalize() if role else "Message"
                parts.append(f"{label}:\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(part for part in parts if part.strip())
    return str(prompt)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*\*+", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_question(prompt_text: str) -> str:
    matches = re.findall(r"([^.?!\n][^?\n]{10,}\?)", prompt_text, flags=re.S)
    if matches:
        return _strip_markdown(matches[-1])
    return ""


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z*])", text.strip())
    return [_strip_markdown(chunk) for chunk in chunks if _strip_markdown(chunk)]


def _extract_recommendation_span(text: str) -> str:
    lower = text.lower()
    header_patterns = [
        "final recommendation",
        "recommendation",
        "final answer",
        "conclusion",
        "in conclusion",
    ]
    start = -1
    for pattern in header_patterns:
        index = lower.rfind(pattern)
        if index > start:
            start = index
    if start >= 0:
        tail = text[start:]
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", tail) if part.strip()]
        span = " ".join(paragraphs[:2])
        return _strip_markdown(span)

    sentences = _split_sentences(text)
    modal_sentences = [
        sentence
        for sentence in sentences
        if re.search(r"\b(should|shouldn't|should not|recommend|ought|must|do not|don't)\b", sentence.lower())
    ]
    if modal_sentences:
        return modal_sentences[-1]
    return _strip_markdown(text[-500:])


def _extract_decision_sentence(span: str) -> str:
    sentences = _split_sentences(span)
    modal_sentences = [
        sentence
        for sentence in sentences
        if re.search(r"\b(should|shouldn't|should not|recommend|ought|must|do not|don't)\b", sentence.lower())
    ]
    if modal_sentences:
        return modal_sentences[0]
    if sentences:
        return sentences[0]
    return _strip_markdown(span)


def _extract_action_text(sentence: str) -> str:
    clean = _strip_markdown(sentence)
    patterns = [
        r"\bshould\s+(not\s+)?(.+?)(?:[.;:!?]|$)",
        r"\bshouldn't\s+(.+?)(?:[.;:!?]|$)",
        r"\brecommend(?:s|ed)?\s+(?:that\s+)?(.+?)(?:[.;:!?]|$)",
        r"\bmust\s+(.+?)(?:[.;:!?]|$)",
        r"\bdo not\s+(.+?)(?:[.;:!?]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if not match:
            continue
        groups = [group for group in match.groups() if group]
        candidate = " ".join(groups).strip()
        candidate = re.sub(r"^(the user|the individual|the player|alice|maria|maya)\s+", "", candidate, flags=re.I)
        candidate = re.sub(r"\s+", " ", candidate)
        if candidate:
            return candidate.lower()
    return clean.lower()


def _normalize_for_similarity(text: str) -> str:
    text = _strip_markdown(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(final|recommendation|conclusion|reasoning|therefore|thus)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _recommendation_signature(row: RowRecord) -> str:
    # Use the full recommendation span to avoid splitting obviously identical
    # actions that are wrapped in different rhetorical styles.
    return f"{row.recommendation_span} {row.decision_sentence}".strip()


def _cluster_recommendations(texts: list[str], threshold: float = 0.24) -> list[int]:
    if not texts:
        return []
    if len(texts) == 1:
        return [0]
    normalized = [_normalize_for_similarity(text) for text in texts]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    matrix = vectorizer.fit_transform(normalized)
    similarities = (matrix @ matrix.T).toarray()
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if similarities[left, right] >= threshold:
                union(left, right)

    root_to_cluster: dict[int, int] = {}
    labels: list[int] = []
    for index in range(len(texts)):
        root = find(index)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        labels.append(root_to_cluster[root])
    return labels


def _pairwise_agreement(cluster_by_prime: dict[str, int]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for i, left in enumerate(PRIME_ORDER):
        if left not in cluster_by_prime:
            continue
        for right in PRIME_ORDER[i + 1 :]:
            if right not in cluster_by_prime:
                continue
            counts[f"{left} vs {right}"] = float(cluster_by_prime[left] == cluster_by_prime[right])
    return counts


def _cluster_alignment(cluster_by_prime: dict[str, int]) -> dict[str, Any]:
    deontic = {prime: cluster_by_prime[prime] for prime in DEONTIC_CLUSTER if prime in cluster_by_prime}
    welfarist = {prime: cluster_by_prime[prime] for prime in WELFARIST_CLUSTER if prime in cluster_by_prime}
    if len(set(deontic.values())) != 1 or len(set(welfarist.values())) != 1:
        return {"primary_hypothesis_aligned": False, "reason": "cluster_not_internally_coherent"}
    deontic_cluster = next(iter(deontic.values()))
    welfarist_cluster = next(iter(welfarist.values()))
    if deontic_cluster == welfarist_cluster:
        return {"primary_hypothesis_aligned": False, "reason": "no_between_cluster_separation"}
    contractualism_cluster = cluster_by_prime.get("contractualism")
    contractualism_role = (
        "with_deontic"
        if contractualism_cluster == deontic_cluster
        else "with_welfarist"
        if contractualism_cluster == welfarist_cluster
        else "third_cluster"
        if contractualism_cluster is not None
        else "missing"
    )
    return {
        "primary_hypothesis_aligned": True,
        "deontic_cluster": deontic_cluster,
        "welfarist_cluster": welfarist_cluster,
        "contractualism_role": contractualism_role,
    }


def _load_rows() -> list[RowRecord]:
    artifact = _load_operation_artifact(GENERATION_ARTIFACT_ID)
    payload = artifact.result()
    if not isinstance(payload, dict):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact rows missing")

    records: list[RowRecord] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        source_example = _mapping(row.get("example"))
        labels = _mapping(source_example.get("labels"))
        prompt_text = _render_prompt_text(source_example.get("prompt") or "")
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        question = _extract_question(prompt_text)
        recommendation_span = _extract_recommendation_span(generated_text)
        decision_sentence = _extract_decision_sentence(recommendation_span)
        action_text = _extract_action_text(decision_sentence)
        records.append(
            RowRecord(
                example_key=str(row.get("example_key") or source_example.get("key") or ""),
                group_id=str(labels.get("group_id") or ""),
                prime_condition=str(labels.get("prime_condition") or ""),
                source_case_theory=str(labels.get("source_case_theory") or ""),
                source_family=str(labels.get("source_family") or ""),
                role_domain=str(labels.get("role_domain") or ""),
                context=str(labels.get("context") or ""),
                prompt_text=prompt_text,
                generated_text=generated_text,
                question=question,
                recommendation_span=recommendation_span,
                decision_sentence=decision_sentence,
                action_text=action_text,
            )
        )
    return records


def analyze() -> dict[str, Any]:
    rows = _load_rows()
    by_group: dict[str, list[RowRecord]] = defaultdict(list)
    for row in rows:
        by_group[row.group_id].append(row)

    pairwise_totals = Counter()
    group_summaries: list[dict[str, Any]] = []
    split_groups: list[dict[str, Any]] = []
    overall_cluster_count_distribution = Counter()
    split_subset_counts = Counter()
    aligned_split_counts = Counter()

    for group_id, group_rows in sorted(by_group.items()):
        ordered = sorted(group_rows, key=lambda row: PRIME_ORDER.index(row.prime_condition))
        recommendation_signatures = [_recommendation_signature(row) for row in ordered]
        cluster_labels = _cluster_recommendations(recommendation_signatures)
        cluster_by_prime = {row.prime_condition: cluster for row, cluster in zip(ordered, cluster_labels, strict=True)}
        cluster_count = len(set(cluster_labels))
        overall_cluster_count_distribution[cluster_count] += 1
        subset = "benchmark_theory_30" if group_id.startswith("theory_group_") else "public_extension_60"
        alignment = _cluster_alignment(cluster_by_prime)
        pairwise_totals.update(_pairwise_agreement(cluster_by_prime))

        group_summary = {
            "group_id": group_id,
            "subset": subset,
            "source_family": ordered[0].source_family,
            "role_domain": ordered[0].role_domain,
            "context": ordered[0].context,
            "question": ordered[0].question,
            "cluster_count": cluster_count,
            "cluster_by_prime": cluster_by_prime,
            "primary_hypothesis": alignment,
            "rows": [
                {
                    "prime_condition": row.prime_condition,
                    "recommendation_span": row.recommendation_span,
                    "decision_sentence": row.decision_sentence,
                    "action_text": row.action_text,
                    "cluster": cluster_by_prime[row.prime_condition],
                }
                for row in ordered
            ],
        }
        group_summaries.append(group_summary)
        if cluster_count > 1:
            split_groups.append(group_summary)
            split_subset_counts[subset] += 1
            if alignment.get("primary_hypothesis_aligned"):
                aligned_split_counts[subset] += 1

    pairwise_rates = {
        pair: round(count / len(group_summaries), 3)
        for pair, count in sorted(pairwise_totals.items())
    }
    benchmark_groups = [item for item in group_summaries if item["subset"] == "benchmark_theory_30"]
    public_groups = [item for item in group_summaries if item["subset"] == "public_extension_60"]

    summary = {
        "generation_artifact_id": GENERATION_ARTIFACT_ID,
        "row_count": len(rows),
        "group_count": len(group_summaries),
        "cluster_count_distribution": dict(sorted(overall_cluster_count_distribution.items())),
        "unanimous_group_count": sum(1 for item in group_summaries if item["cluster_count"] == 1),
        "split_group_count": len(split_groups),
        "benchmark_theory_30": {
            "group_count": len(benchmark_groups),
            "unanimous_count": sum(1 for item in benchmark_groups if item["cluster_count"] == 1),
            "split_count": sum(1 for item in benchmark_groups if item["cluster_count"] > 1),
            "primary_hypothesis_aligned_split_count": aligned_split_counts["benchmark_theory_30"],
        },
        "public_extension_60": {
            "group_count": len(public_groups),
            "unanimous_count": sum(1 for item in public_groups if item["cluster_count"] == 1),
            "split_count": sum(1 for item in public_groups if item["cluster_count"] > 1),
            "primary_hypothesis_aligned_split_count": aligned_split_counts["public_extension_60"],
        },
        "pairwise_prime_agreement_rate": pairwise_rates,
        "split_groups": split_groups,
        "all_groups": group_summaries,
    }
    return summary


def write_outputs(summary: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "behavior_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")

    split_groups = summary["split_groups"]
    lines: list[str] = []
    lines.append("# Experiment 2 Broad Behavior Review")
    lines.append("")
    lines.append("## High-level read")
    lines.append("")
    lines.append(
        f"Across the combined `90`-dilemma batch, recommendation behavior is unanimous in "
        f"{summary['unanimous_group_count']}/{summary['group_count']} groups and split in "
        f"{summary['split_group_count']}/{summary['group_count']}."
    )
    lines.append("")
    lines.append(
        f"- Original benchmark `30`: {summary['benchmark_theory_30']['split_count']}/30 split, "
        f"{summary['benchmark_theory_30']['unanimous_count']}/30 unanimous."
    )
    lines.append(
        f"- New public extension `60`: {summary['public_extension_60']['split_count']}/60 split, "
        f"{summary['public_extension_60']['unanimous_count']}/60 unanimous."
    )
    lines.append("")
    lines.append(
        f"Primary deontic-vs-welfarist cluster pattern appears in "
        f"{summary['benchmark_theory_30']['primary_hypothesis_aligned_split_count']}/"
        f"{summary['benchmark_theory_30']['split_count'] or 1} benchmark split groups and "
        f"{summary['public_extension_60']['primary_hypothesis_aligned_split_count']}/"
        f"{summary['public_extension_60']['split_count'] or 1} public split groups."
    )
    lines.append("")
    lines.append("## Cluster-count distribution")
    lines.append("")
    for cluster_count, count in summary["cluster_count_distribution"].items():
        lines.append(f"- `{cluster_count}` distinct recommendation clusters: `{count}` groups")
    lines.append("")
    lines.append("## Pairwise prime agreement")
    lines.append("")
    for pair, rate in summary["pairwise_prime_agreement_rate"].items():
        lines.append(f"- {pair}: `{rate:.3f}`")
    lines.append("")
    lines.append("## Split groups")
    lines.append("")
    for item in split_groups:
        lines.append(
            f"### `{item['group_id']}` ({item['subset']}, {item['source_family']}, {item['role_domain']}, {item['context']})"
        )
        if item["question"]:
            lines.append(f"- question: `{item['question']}`")
        lines.append(f"- clusters: `{item['cluster_by_prime']}`")
        alignment = item["primary_hypothesis"]
        lines.append(f"- primary hypothesis: `{alignment}`")
        for row in item["rows"]:
            lines.append(
                f"- `{row['prime_condition']}` -> cluster `{row['cluster']}`: {row['decision_sentence']}"
            )
        lines.append("")

    (REPORT_DIR / "report.md").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    summary = analyze()
    write_outputs(summary)
    print(json.dumps(
        {
            "behavior_summary_path": str(REPORT_DIR / "behavior_summary.json"),
            "report_path": str(REPORT_DIR / "report.md"),
            "group_count": summary["group_count"],
            "split_group_count": summary["split_group_count"],
            "benchmark_split_count": summary["benchmark_theory_30"]["split_count"],
            "public_split_count": summary["public_extension_60"]["split_count"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

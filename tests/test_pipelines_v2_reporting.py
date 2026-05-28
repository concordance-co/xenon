from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipelines_v2.reporting import generate_report_assets


def test_probe_result_assets_cover_single_and_multiple_metrics(tmp_path: Path) -> None:
    report_root = tmp_path / "probe_report"
    report_root.mkdir(parents=True)

    single_metric = _materialize_assets(
        report_root=report_root / "single",
        step_name="probe_single",
        result_payload={
            "kind": "probe_result",
            "layers": [
                {"layer": 0, "balanced_accuracy": 0.51, "baseline_majority": 0.5},
                {"layer": 4, "balanced_accuracy": 0.82, "baseline_majority": 0.5},
            ],
            "summary": {"best_layer": 4, "best_metric": "balanced_accuracy", "best_value": 0.82, "example_count": 16},
        },
    )
    _assert_exists(single_metric["report_root"] / "assets" / "probe_single" / "balanced_accuracy_by_layer.png")
    assert not (single_metric["report_root"] / "assets" / "probe_single" / "probe_metrics_by_layer.png").exists()

    multi_metric = _materialize_assets(
        report_root=report_root / "multi",
        step_name="probe_multi",
        result_payload={
            "kind": "probe_result",
            "layers": [
                {"layer": 0, "balanced_accuracy": 0.52, "accuracy": 0.53, "auroc": 0.58, "baseline_majority": 0.5},
                {"layer": 4, "balanced_accuracy": 0.9, "accuracy": 0.88, "auroc": 0.93, "baseline_majority": 0.5},
            ],
            "summary": {"best_layer": 4, "best_metric": "balanced_accuracy", "best_value": 0.9, "example_count": 16},
        },
    )
    _assert_exists(multi_metric["report_root"] / "assets" / "probe_multi" / "balanced_accuracy_by_layer.png")
    _assert_exists(multi_metric["report_root"] / "assets" / "probe_multi" / "accuracy_by_layer.png")
    _assert_exists(multi_metric["report_root"] / "assets" / "probe_multi" / "auroc_by_layer.png")
    _assert_exists(multi_metric["report_root"] / "assets" / "probe_multi" / "probe_metrics_by_layer.png")
    assert multi_metric["manifest"]["unsupported_inputs"] == []
    assert multi_metric["summary"]["step_summaries"]["probe_multi"]["primary_figure_id"] == "probe_multi/balanced_accuracy_by_layer"


def test_transfer_probe_result_cross_cohort_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "transfer_cross",
        step_name="transfer_cross",
        result_payload={
            "kind": "transfer_probe_result",
            "layers": [
                {
                    "layer": 0,
                    "within_cohort_baseline": {
                        "activity": {"balanced_accuracy": 0.71, "auroc": 0.75},
                        "size": {"balanced_accuracy": 0.69, "auroc": 0.72},
                    },
                    "cross_cohort_transfer": {
                        "activity_to_size": {"balanced_accuracy": 0.6, "auroc": 0.63, "transfer_delta_vs_test_within": -0.09},
                        "size_to_activity": {"balanced_accuracy": 0.58, "auroc": 0.61, "transfer_delta_vs_test_within": -0.13},
                    },
                    "direction_similarity": {"activity_vs_size": 0.21},
                },
                {
                    "layer": 4,
                    "within_cohort_baseline": {
                        "activity": {"balanced_accuracy": 0.82, "auroc": 0.86},
                        "size": {"balanced_accuracy": 0.79, "auroc": 0.83},
                    },
                    "cross_cohort_transfer": {
                        "activity_to_size": {"balanced_accuracy": 0.68, "auroc": 0.7, "transfer_delta_vs_test_within": -0.11},
                        "size_to_activity": {"balanced_accuracy": 0.65, "auroc": 0.69, "transfer_delta_vs_test_within": -0.17},
                    },
                    "direction_similarity": {"activity_vs_size": 0.33},
                },
            ],
            "summary": {"mode": "cross_cohort_transfer", "cohort_count": 2, "layer_count": 2, "split_names": [], "regularization": [1.0]},
        },
    )
    base = rendered["report_root"] / "assets" / "transfer_cross"
    _assert_exists(base / "balanced_accuracy_cross_cohort.png")
    _assert_exists(base / "auroc_cross_cohort.png")
    _assert_exists(base / "transfer_delta_balanced_accuracy.png")
    _assert_exists(base / "direction_similarity.png")
    _assert_exists(rendered["report_root"] / "tables" / "transfer_cross.json")
    assert rendered["summary"]["step_summaries"]["transfer_cross"]["primary_figure_id"] == "transfer_cross/balanced_accuracy_cross_cohort"


def test_transfer_probe_result_split_holdout_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "transfer_split",
        step_name="transfer_split",
        result_payload={
            "kind": "transfer_probe_result",
            "layers": [
                {
                    "layer": 0,
                    "split_results": {
                        "lexical_split": {
                            "activity": {"balanced_accuracy": 0.55, "auroc": 0.57},
                            "size": {"balanced_accuracy": 0.6, "auroc": 0.62},
                        }
                    },
                },
                {
                    "layer": 4,
                    "split_results": {
                        "lexical_split": {
                            "activity": {"balanced_accuracy": 0.71, "auroc": 0.75},
                            "size": {"balanced_accuracy": 0.73, "auroc": 0.78},
                        }
                    },
                },
            ],
            "summary": {"mode": "split_holdout", "cohort_count": 2, "layer_count": 2, "split_names": ["lexical_split"], "regularization": [1.0]},
        },
    )
    _assert_exists(rendered["report_root"] / "assets" / "transfer_split" / "lexical_split_balanced_accuracy.png")
    _assert_exists(rendered["report_root"] / "assets" / "transfer_split" / "lexical_split_auroc.png")
    assert rendered["summary"]["step_summaries"]["transfer_split"]["primary_figure_id"] == "transfer_split/lexical_split_balanced_accuracy"


def test_transfer_probe_result_regularization_sweep_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "transfer_sweep",
        step_name="transfer_sweep",
        result_payload={
            "kind": "transfer_probe_result",
            "layers": [
                {
                    "layer": 8,
                    "cross_cohort_transfer": {
                        "activity_to_size": {
                            "regularization_sweep": [
                                {"C": 0.1, "balanced_accuracy": 0.58, "auroc": 0.6, "transfer_delta_vs_test_within": -0.08},
                                {"C": 1.0, "balanced_accuracy": 0.64, "auroc": 0.66, "transfer_delta_vs_test_within": -0.05},
                            ]
                        }
                    },
                }
            ],
            "summary": {"mode": "cross_cohort_transfer", "cohort_count": 2, "layer_count": 1, "split_names": [], "regularization": [0.1, 1.0]},
        },
    )
    _assert_exists(rendered["report_root"] / "assets" / "transfer_sweep" / "regularization_sweep_balanced_accuracy.png")
    _assert_exists(rendered["report_root"] / "assets" / "transfer_sweep" / "regularization_sweep_auroc.png")


def test_text_baseline_result_grouped_cv_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "text_grouped",
        step_name="text_grouped",
        result_payload={
            "kind": "text_baseline_result",
            "mode": "grouped_cv",
            "model": "countvectorizer_logreg",
            "results": {"grouped_cv": {"balanced_accuracy": 0.61, "auroc": 0.66}},
            "summary": {"mode": "grouped_cv", "example_count": 24, "split_names": []},
        },
    )
    _assert_exists(rendered["report_root"] / "assets" / "text_grouped" / "grouped_cv_metrics.png")
    assert rendered["summary"]["step_summaries"]["text_grouped"]["primary_figure_id"] == "text_grouped/grouped_cv_metrics"


def test_text_baseline_result_cross_cohort_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "text_cross",
        step_name="text_cross",
        result_payload={
            "kind": "text_baseline_result",
            "mode": "cross_cohort_transfer",
            "model": "countvectorizer_logreg",
            "results": {
                "within_cohort_baseline": {
                    "activity": {"balanced_accuracy": 0.71, "auroc": 0.75},
                    "size": {"balanced_accuracy": 0.69, "auroc": 0.74},
                },
                "cross_cohort_transfer": {
                    "activity_to_size": {"balanced_accuracy": 0.62, "auroc": 0.65, "transfer_delta_vs_test_within": -0.07},
                    "size_to_activity": {"balanced_accuracy": 0.59, "auroc": 0.63, "transfer_delta_vs_test_within": -0.12},
                },
            },
            "summary": {"mode": "cross_cohort_transfer", "example_count": 24, "split_names": []},
        },
    )
    _assert_exists(rendered["report_root"] / "assets" / "text_cross" / "balanced_accuracy_cross_cohort.png")
    _assert_exists(rendered["report_root"] / "assets" / "text_cross" / "auroc_cross_cohort.png")


def test_text_baseline_result_split_holdout_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "text_split",
        step_name="text_split",
        result_payload={
            "kind": "text_baseline_result",
            "mode": "split_holdout",
            "model": "countvectorizer_logreg",
            "results": {
                "split_results": {
                    "lexical_split": {
                        "activity": {"balanced_accuracy": 0.55, "auroc": 0.58},
                        "size": {"balanced_accuracy": 0.6, "auroc": 0.64},
                    }
                }
            },
            "summary": {"mode": "split_holdout", "example_count": 24, "split_names": ["lexical_split"]},
        },
    )
    _assert_exists(rendered["report_root"] / "assets" / "text_split" / "lexical_split_balanced_accuracy.png")
    _assert_exists(rendered["report_root"] / "assets" / "text_split" / "lexical_split_auroc.png")


def test_generation_run_result_assets_expose_prompt_response_rows(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "generation",
        step_name="generate_cases",
        result_payload={
            "kind": "generation_run_result",
            "summary": {
                "example_count": 1,
                "completed_example_count": 1,
                "total_example_count": 1,
                "partial": False,
            },
            "rows": [
                {
                    "example_key": "case_a",
                    "finish_reason": "stop",
                    "generated_text": "The answer is ready.",
                    "example": {
                        "key": "case_a",
                        "case_key": "case",
                        "prompt_hash": "abc123",
                        "labels": {
                            "condition": "policy_conflict",
                            "expected_selected_source": "user_policy",
                            "positive_authority_risk": True,
                        },
                        "metadata": {
                            "span_specs": [
                                {
                                    "name": "user_policy",
                                    "span_label": "policy",
                                    "source_type": "user",
                                    "assigned_authority": "full",
                                    "instruction_like": True,
                                    "content_text": "Follow user policy.",
                                }
                            ]
                        },
                        "prompt": [
                            {"role": "system", "content": "You are careful."},
                            {"role": "user", "content": "Use the policy."},
                        ],
                    },
                }
            ],
        },
    )

    assert rendered["manifest"]["unsupported_inputs"] == []
    _assert_exists(rendered["report_root"] / "tables" / "generate_cases.json")
    _assert_exists(rendered["report_root"] / "assets" / "generate_cases" / "response_lengths.png")
    table = json.loads((rendered["report_root"] / "tables" / "generate_cases.json").read_text())
    assert table["result_kind"] == "generation_run_result"
    row = table["rows"][0]
    assert row["example_key"] == "case_a"
    assert "system: You are careful." in row["prompt"]
    assert row["response"] == "The answer is ready."
    assert "expected_selected_source=user_policy" in row["label_summary"]
    assert row["span_names"] == "user_policy"


def test_residualized_probe_result_assets(tmp_path: Path) -> None:
    rendered = _materialize_assets(
        report_root=tmp_path / "residualized",
        step_name="residualized",
        result_payload={
            "kind": "residualized_probe_result",
            "layers": [
                {
                    "layer": 0,
                    "nuisance_accuracy_raw_training_fit": 0.81,
                    "nuisance_accuracy_on_null_training_fit": 0.54,
                    "family_subspace_rank": 3,
                    "raw_probe": {"balanced_accuracy": 0.71, "auroc": 0.76},
                    "residualized_probe": {"balanced_accuracy": 0.66, "auroc": 0.72},
                    "delta_raw_minus_null": {"balanced_accuracy": 0.05, "auroc": 0.04},
                },
                {
                    "layer": 4,
                    "nuisance_accuracy_raw_training_fit": 0.86,
                    "nuisance_accuracy_on_null_training_fit": 0.59,
                    "family_subspace_rank": 3,
                    "raw_probe": {"balanced_accuracy": 0.8, "auroc": 0.85},
                    "residualized_probe": {"balanced_accuracy": 0.73, "auroc": 0.79},
                    "delta_raw_minus_null": {"balanced_accuracy": 0.07, "auroc": 0.06},
                },
            ],
            "summary": {"layer_count": 2, "example_count": 24},
        },
    )
    base = rendered["report_root"] / "assets" / "residualized"
    _assert_exists(base / "balanced_accuracy_raw_vs_residualized.png")
    _assert_exists(base / "auroc_raw_vs_residualized.png")
    _assert_exists(base / "delta_raw_minus_null.png")
    _assert_exists(base / "nuisance_accuracy.png")


def test_geometry_result_assets_cover_pca_and_lda(tmp_path: Path) -> None:
    pca_rendered = _materialize_assets(
        report_root=tmp_path / "geometry_pca",
        step_name="geometry_pca",
        result_payload={
            "kind": "geometry_result",
            "method": "pca",
            "layers": [
                {
                    "layer": 0,
                    "component_count": 2,
                    "components": [[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]],
                    "explained_variance_ratio": [0.41, 0.29],
                    "example_count": 4,
                    "selected_example_keys": ["a", "b", "c", "d"],
                    "color_by": {"alignment": [False, False, True, True]},
                },
                {
                    "layer": 4,
                    "component_count": 2,
                    "components": [[0.2, 0.3], [0.3, 0.2], [0.7, 0.8], [0.8, 0.7]],
                    "explained_variance_ratio": [0.49, 0.21],
                    "example_count": 4,
                    "selected_example_keys": ["a", "b", "c", "d"],
                    "color_by": {"alignment": [False, False, True, True]},
                },
            ],
            "summary": {"method": "pca", "example_count": 4, "layer_count": 2},
        },
    )
    _assert_exists(pca_rendered["report_root"] / "assets" / "geometry_pca" / "layer0_alignment.png")
    _assert_exists(pca_rendered["report_root"] / "assets" / "geometry_pca" / "layer4_alignment.png")
    _assert_exists(pca_rendered["report_root"] / "assets" / "geometry_pca" / "explained_variance_by_layer.png")
    assert "primary_figure_id" not in pca_rendered["summary"]["step_summaries"]["geometry_pca"]

    lda_rendered = _materialize_assets(
        report_root=tmp_path / "geometry_lda",
        step_name="geometry_lda",
        result_payload={
            "kind": "geometry_result",
            "method": "lda",
            "layers": [
                {
                    "layer": 8,
                    "component_count": 2,
                    "components": [[0.1, 0.4], [0.2, 0.5], [0.8, 0.1], [0.9, 0.2]],
                    "example_count": 4,
                    "selected_example_keys": ["a", "b", "c", "d"],
                    "labels": ["alpha", "alpha", "beta", "beta"],
                }
            ],
            "summary": {"method": "lda", "example_count": 4, "layer_count": 1},
        },
    )
    _assert_exists(lda_rendered["report_root"] / "assets" / "geometry_lda" / "layer8_label.png")


def _materialize_assets(*, report_root: Path, step_name: str, result_payload: dict[str, Any]) -> dict[str, Any]:
    report_root.mkdir(parents=True, exist_ok=True)
    results_dir = report_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{step_name}_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, sort_keys=True)

    payload = {
        "kind": "report_result",
        "template": "summary",
        "inputs": [
            {
                "name": step_name,
                "artifact_id": f"{step_name}_artifact",
                "artifact_kind": "probe",
                "downloaded_result_path": str(result_path),
            }
        ],
        "summary": {
            "template": "summary",
            "input_count": 1,
            "example_count": (result_payload.get("summary") or {}).get("example_count"),
        },
    }
    generated = generate_report_assets(report_root=report_root, payload=payload)
    return {
        "report_root": report_root,
        "payload": payload,
        "summary": generated["summary"],
        "manifest": generated["manifest"],
    }


def _assert_exists(path: Path) -> None:
    assert path.exists(), str(path)

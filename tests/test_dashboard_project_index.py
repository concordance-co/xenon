from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.dashboard.project_index import (
    build_projects_index,
    resolve_project_report_root,
)
from pipelines_v2.dashboard.reports import build_report_detail_from_root


def test_project_index_scans_project_phase_experiment_report(tmp_path: Path) -> None:
    projects_root = tmp_path / "projects"
    report_root = (
        projects_root
        / "authority_monitor"
        / "phase_01"
        / "reports"
        / "probe_smoke"
        / "report_abc"
    )
    (report_root / "assets").mkdir(parents=True)
    (report_root / "tables").mkdir()
    (report_root / "results").mkdir()
    _write_json(
        projects_root / "authority_monitor" / "project_spec.json",
        {
            "project": "authority_monitor",
            "title": "Authority Monitor",
            "primary_question": "Can probes find authority confusion?",
            "core_labels": ["assigned_authority", "selected_source"],
            "candidate_datasets": ["synthetic_pairs"],
        },
    )
    (projects_root / "authority_monitor" / "phase_01" / "PHASE.md").write_text(
        "# Phase 01\n\nDense probe smoke tests.\n",
        encoding="utf-8",
    )
    _write_json(
        report_root / "report.json",
        {
            "inputs": [
                {
                    "name": "probe_selected_source",
                    "artifact_id": "probe_1",
                    "artifact_kind": "probe",
                    "label_names": ["selected_source"],
                    "workflow": {
                        "run_id": "wr_123",
                        "workflow_name": "authority_probe_workflow",
                    },
                    "example_coverage": {
                        "dataset_id": "authority_smoke",
                        "dataset_name": "Authority smoke",
                        "example_count": 12,
                    },
                }
            ]
        },
    )
    _write_json(report_root / "summary.json", {"example_count": 12, "input_count": 1})
    _write_json(
        report_root / "assets" / "manifest.json",
        {
            "tables": {
                "probe_selected_source": {
                    "path": "tables/probe_selected_source.json",
                    "rows": 1,
                    "columns": ["layer", "balanced_accuracy"],
                    "step_name": "probe_selected_source",
                    "result_kind": "probe_result",
                }
            },
            "figures": {},
            "unsupported_inputs": [],
        },
    )
    _write_json(report_root / "tables" / "probe_selected_source.json", {"rows": []})
    _write_json(report_root / "results" / "probe_selected_source_results.json", {"kind": "probe_result"})

    index = build_projects_index(project_roots=[projects_root])

    assert index.project_roots == [str(projects_root.resolve())]
    project = index.projects[0]
    assert project.project_id == "authority_monitor"
    phase = project.phases[0]
    assert phase.phase_id == "phase_01"
    experiment = phase.experiments[0]
    assert experiment.experiment_id == "probe_smoke"
    assert experiment.experiment_category == "probe_readout"
    report = experiment.reports[0]
    assert report.run_id == "wr_123"
    assert report.data_sources[0].dataset_id == "authority_smoke"
    assert report.labels[0].name == "selected_source"

    resolved = resolve_project_report_root(report.report_key, project_roots=[projects_root])
    assert resolved == report_root.resolve()
    detail = build_report_detail_from_root(resolved, artifact_id=f"project:{report.report_key}")
    assert detail.tables[0].slug == "probe_selected_source"
    assert detail.results[0].path == "results/probe_selected_source_results.json"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_text(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _json_literal(value: Any) -> str:
    return json.dumps(value, default=str)


def _format_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _probe_result_summary(results: dict[str, Any]) -> dict[str, Any] | None:
    probe_rows = results.get("probe")
    if not isinstance(probe_rows, list) or not probe_rows:
        return None

    normalized: list[dict[str, Any]] = []
    for row in probe_rows:
        if not isinstance(row, dict):
            continue
        if row.get("accuracy_mean") is None:
            continue
        normalized.append(row)
    if not normalized:
        return None

    ranked = sorted(normalized, key=lambda row: float(row.get("accuracy_mean", float("-inf"))), reverse=True)
    best = ranked[0]
    baseline_majority = best.get("baseline_majority")
    accuracy_mean = best.get("accuracy_mean")
    lift_vs_majority = None
    if baseline_majority is not None and accuracy_mean is not None:
        lift_vs_majority = float(accuracy_mean) - float(baseline_majority)

    if lift_vs_majority is None:
        main_read = "The best layer is reported below, but a majority-baseline comparison is unavailable."
    elif lift_vs_majority > 0:
        main_read = (
            "The best probe layer beats the majority baseline, so this setup carries a usable linear signal "
            "for the workflow label."
        )
    else:
        main_read = (
            "The best probe layer does not beat the majority baseline, so this run does not support a useful "
            "linear readout for the workflow label."
        )

    return {
        "kind": "probe",
        "n_layers": len(normalized),
        "main_read": main_read,
        "best_layer": best.get("layer"),
        "best_accuracy_mean": accuracy_mean,
        "best_accuracy_std": best.get("accuracy_std"),
        "best_balanced_accuracy": best.get("balanced_accuracy"),
        "baseline_majority": baseline_majority,
        "lift_vs_majority": lift_vs_majority,
        "top_layers": [
            {
                "layer": row.get("layer"),
                "accuracy_mean": row.get("accuracy_mean"),
                "accuracy_std": row.get("accuracy_std"),
                "balanced_accuracy": row.get("balanced_accuracy"),
                "selectivity": row.get("selectivity"),
            }
            for row in ranked[:8]
        ],
    }


def _analysis_artifacts(analysis_config: dict[str, Any], analysis_result: dict[str, Any]) -> dict[str, Any]:
    output_dir = analysis_result.get("output_dir")
    if not output_dir:
        return {}

    output_path = Path(str(output_dir))
    artifacts = {
        "results_json": str(output_path / "results.json"),
    }
    if analysis_config.get("mode") == "probe":
        target = analysis_config.get("target")
        data_source = analysis_config.get("data_source")
        if target and data_source:
            artifacts["primary_parquet"] = str(output_path / f"probe_{target}_{data_source}.parquet")
    return artifacts


def _default_report_dir(spec: dict[str, Any], analysis_run: dict[str, Any]) -> Path:
    return Path("data/reports/workflows") / str(spec["id"]) / str(analysis_run["id"])


def _report_output_dir(spec: dict[str, Any], analysis_run: dict[str, Any]) -> Path:
    report_block = dict(spec.get("report") or {})
    output_dir = report_block.get("output_dir")
    if isinstance(output_dir, str) and output_dir.strip():
        return Path(output_dir) / str(analysis_run["id"])
    return _default_report_dir(spec, analysis_run)


def _find_publication(conn: Any, relation_name: str | None, spec_id: str) -> dict[str, Any] | None:
    if relation_name:
        row = conn.execute(
            "SELECT id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at "
            "FROM dataset_publications WHERE relation_name = %s ORDER BY created_at DESC LIMIT 1",
            [relation_name],
        ).fetchone()
        if row:
            return dict(row)
    row = conn.execute(
        "SELECT id, spec_id, spec_version, run_id, relation_name, publish_mode, row_count, publication_json, created_at, updated_at "
        "FROM dataset_publications WHERE spec_id = %s ORDER BY created_at DESC LIMIT 1",
        [spec_id],
    ).fetchone()
    return dict(row) if row else None


def _find_capture_run(conn: Any, spec_id: str, publication: str | None) -> dict[str, Any] | None:
    rows = conn.execute(
        "SELECT id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at "
        "FROM workflow_runs WHERE spec_id = %s AND run_type = 'capture' AND status = 'succeeded' ORDER BY created_at DESC",
        [spec_id],
    ).fetchall()
    for row in rows:
        as_dict = dict(row)
        result = dict(as_dict.get("result_json") or {})
        if publication is None or result.get("publication") == publication:
            return as_dict
    return None


def _find_dataset_run(conn: Any, spec_id: str, publication: str | None) -> dict[str, Any] | None:
    rows = conn.execute(
        "SELECT id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at "
        "FROM workflow_runs WHERE spec_id = %s AND run_type = 'dataset' AND status = 'succeeded' ORDER BY created_at DESC",
        [spec_id],
    ).fetchall()
    for row in rows:
        as_dict = dict(row)
        result = dict(as_dict.get("result_json") or {})
        pub = dict(result.get("publication") or {})
        if publication is None or pub.get("relation_name") == publication:
            return as_dict
    return None


def _build_summary(
    *,
    spec: dict[str, Any],
    analysis_run: dict[str, Any],
    dataset_run: dict[str, Any] | None,
    capture_run: dict[str, Any] | None,
    publication: dict[str, Any] | None,
) -> dict[str, Any]:
    analysis_result = dict(analysis_run.get("result_json") or {})
    analysis_config = dict(analysis_run.get("config_json") or {})
    capture_result = dict((capture_run or {}).get("result_json") or {})

    return {
        "generated_at": _now_iso(),
        "spec": {
            "id": spec.get("id"),
            "name": spec.get("name"),
            "description": spec.get("description"),
            "version": spec.get("version"),
        },
        "publication": {
            "relation_name": publication.get("relation_name") if publication else analysis_result.get("publication"),
            "publish_mode": publication.get("publish_mode") if publication else None,
            "row_count": publication.get("row_count") if publication else None,
        },
        "runs": {
            "dataset": dataset_run.get("id") if dataset_run else None,
            "capture": capture_run.get("id") if capture_run else None,
            "analysis": analysis_run.get("id"),
        },
        "capture": {
            "activations_dir": capture_result.get("activations_dir") or analysis_config.get("activations_dir"),
            "remote_activations_path": capture_result.get("remote_activations_path"),
            "publication": capture_result.get("publication") or analysis_result.get("publication"),
        },
        "analysis": {
            "mode": analysis_config.get("mode"),
            "target": analysis_config.get("target"),
            "data_source": analysis_config.get("data_source"),
            "pooling": analysis_config.get("pooling"),
            "n_folds": analysis_config.get("n_folds"),
            "layers": analysis_config.get("layers"),
            "labels_path": analysis_result.get("labels_path"),
            "output_dir": analysis_result.get("output_dir"),
            "remote_output_path": analysis_result.get("remote_output_path"),
            "results": analysis_result.get("results"),
            "result_summary": _probe_result_summary(dict(analysis_result.get("results") or {})),
            "artifacts": _analysis_artifacts(analysis_config, analysis_result),
        },
    }


def _write_typst_report(report_path: Path, summary: dict[str, Any]) -> None:
    spec = dict(summary.get("spec") or {})
    publication = dict(summary.get("publication") or {})
    analysis = dict(summary.get("analysis") or {})
    runs = dict(summary.get("runs") or {})
    result_summary = dict(analysis.get("result_summary") or {})
    artifacts = dict(analysis.get("artifacts") or {})

    probe_summary_section = ""
    if result_summary.get("kind") == "probe":
        top_rows = []
        for row in result_summary.get("top_layers") or []:
            top_rows.extend(
                [
                    f'  [`{_safe_text(row.get("layer") or "n/a")}`],',
                    f' [`{_safe_text(_format_float(row.get("accuracy_mean")))}`],',
                    f' [`{_safe_text(_format_float(row.get("accuracy_std")))}`],',
                    f' [`{_safe_text(_format_float(row.get("balanced_accuracy")))}`],',
                    f' [`{_safe_text(_format_float(row.get("selectivity")))}`],',
                ]
            )
        top_rows_text = "\n".join(top_rows)
        probe_summary_section = f"""
= Executive Read

#block(
  width: 100%,
  inset: (left: 14pt, top: 12pt, bottom: 12pt, right: 12pt),
  stroke: (left: 3pt + rgb("#B56662"), top: none, right: none, bottom: none),
  fill: rgb("#FAF5F3"),
)[
  #text(size: 7.5pt, fill: rgb("#B56662"), weight: "bold", tracking: 0.08em)[MAIN READ]
  #v(0.3em)
  #text(size: 12.5pt, weight: "medium")[{_safe_text(result_summary.get("main_read") or "n/a")}]
]

= Main Quantitative Findings

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  stat("Best Layer", "{_safe_text(result_summary.get("best_layer") or "n/a")}"),
  stat("Best Accuracy", "{_safe_text(_format_float(result_summary.get("best_accuracy_mean")))}"),
  stat("Majority Baseline", "{_safe_text(_format_float(result_summary.get("baseline_majority")))}"),
  stat("Lift vs Baseline", "{_safe_text(_format_float(result_summary.get("lift_vs_majority")))}"),
)

#v(8pt)

Why this matters:

- best-layer balanced accuracy is `{_safe_text(_format_float(result_summary.get("best_balanced_accuracy")))}`
- run covers `{_safe_text(result_summary.get("n_layers") or "n/a")}` probed layers
- best-layer fold std is `{_safe_text(_format_float(result_summary.get("best_accuracy_std")))}`

= Top Layers

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, right, right, right, right),
  inset: 6pt,
  stroke: 0.4pt,
  fill: (x, y) => if y == 0 {{ rgb("#DDEBF0") }} else if calc.odd(y) {{ rgb("#F8FBFC") }} else {{ white }},

  [*Layer*], [*Accuracy*], [*Std*], [*Balanced*], [*Selectivity*],
{top_rows_text}
)
"""

    artifacts_section = ""
    if artifacts:
        artifact_lines = []
        if artifacts.get("results_json"):
            artifact_lines.append(f'- Results JSON: `{_safe_text(artifacts["results_json"])}`')
        if artifacts.get("primary_parquet"):
            artifact_lines.append(f'- Primary parquet: `{_safe_text(artifacts["primary_parquet"])}`')
        if analysis.get("remote_output_path"):
            artifact_lines.append(f'- Remote output: `{_safe_text(analysis.get("remote_output_path"))}`')
        artifacts_section = "\n".join(["= Artifacts", "", *artifact_lines])

    report_text = f"""#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.78in),
)

#set par(justify: false, leading: 0.62em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let muted = rgb("#5E6F82")
#let divider = rgb("#D6DEE3")

#show heading.where(level: 1): it => block(
  above: 1.1em,
  below: 0.35em,
  text(17pt, weight: "bold", fill: navy)[#it.body],
)

#show heading.where(level: 2): it => block(
  above: 0.75em,
  below: 0.25em,
  text(12pt, weight: "bold", fill: teal)[#it.body],
)

#let stat(label, value) = block(
  stroke: (paint: divider, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 17pt, fill: navy, weight: "bold")[#value]
]

#align(left)[
  #text(size: 9pt, fill: rgb("#B33A2A"), tracking: 0.08em, weight: "medium")[XENON INTERPRETABILITY]
  #v(0.3em)
  #text(size: 22pt, weight: "bold")[{_safe_text(spec.get("name") or spec.get("id") or "Workflow Report")}]
  #v(0.4em)
  #text(size: 11pt, fill: muted)[
    Workflow report generated {_safe_text(summary.get("generated_at"))}. This report summarizes the latest successful dataset, capture, and analysis chain for the selected workflow spec.
  ]
  #v(0.8em)
  #line(length: 100%, stroke: 1.5pt + black)
  #v(0.5em)
  #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    gutter: 8pt,
    [#text(size: 7.5pt, fill: muted, weight: "bold")[SPEC]\\ #text(size: 9pt)[{_safe_text(spec.get("id") or "n/a")}]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[PUBLICATION]\\ #text(size: 9pt)[{_safe_text(publication.get("relation_name") or "n/a")}]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[ROWS]\\ #text(size: 9pt)[{_safe_text(publication.get("row_count") or "n/a")}]],
    [#text(size: 7.5pt, fill: muted, weight: "bold")[ANALYSIS RUN]\\ #text(size: 9pt)[{_safe_text(runs.get("analysis") or "n/a")}]],
  )
]

#v(12pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat("Spec", "{_safe_text(spec.get("name") or spec.get("id") or "Unnamed")}"),
  stat("Publication", "{_safe_text(publication.get("relation_name") or "n/a")}"),
  stat("Rows", "{_safe_text(publication.get("row_count") or "n/a")}"),
)

= Workflow

- Spec id: `{_safe_text(spec.get("id") or "n/a")}`
- Version: `{_safe_text(spec.get("version") or "n/a")}`
- Dataset run: `{_safe_text(runs.get("dataset") or "n/a")}`
- Capture run: `{_safe_text(runs.get("capture") or "n/a")}`
- Analysis run: `{_safe_text(runs.get("analysis") or "n/a")}`

= Analysis Setup

- Mode: `{_safe_text(analysis.get("mode") or "n/a")}`
- Target: `{_safe_text(analysis.get("target") or "n/a")}`
- Data source: `{_safe_text(analysis.get("data_source") or "n/a")}`
- Pooling: `{_safe_text(analysis.get("pooling") or "n/a")}`
- Labels parquet: `{_safe_text(analysis.get("labels_path") or "n/a")}`
- Output dir: `{_safe_text(analysis.get("output_dir") or "n/a")}`
{probe_summary_section}
{artifacts_section}
"""
    report_path.write_text(report_text)


def compile_typst(report_path: Path) -> dict[str, Any]:
    typst_path = shutil.which("typst")
    if not typst_path:
        return {"compiled": False, "reason": "typst_not_installed", "pdf_path": None}

    pdf_path = report_path.with_suffix(".pdf")
    proc = subprocess.run(
        [typst_path, "compile", str(report_path), str(pdf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "compiled": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "pdf_path": str(pdf_path) if proc.returncode == 0 else None,
    }


def build_workflow_report(conn: Any, *, analysis_run_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, spec_id, spec_version, run_type, status, source, spec_snapshot_json, config_json, result_json, error_text, created_at, updated_at, completed_at "
        "FROM workflow_runs WHERE id = %s",
        [analysis_run_id],
    ).fetchone()
    if not row:
        raise ValueError(f"Analysis run not found: {analysis_run_id}")
    analysis_run = dict(row)
    if analysis_run.get("run_type") != "analysis":
        raise ValueError(f"Run {analysis_run_id} is not an analysis run")

    spec = dict(analysis_run.get("spec_snapshot_json") or {})
    analysis_result = dict(analysis_run.get("result_json") or {})
    publication_name = analysis_result.get("publication")
    publication = _find_publication(conn, publication_name, str(spec["id"]))
    capture_run = _find_capture_run(conn, str(spec["id"]), publication_name)
    dataset_run = _find_dataset_run(conn, str(spec["id"]), publication_name)

    summary = _build_summary(
        spec=spec,
        analysis_run=analysis_run,
        dataset_run=dataset_run,
        capture_run=capture_run,
        publication=publication,
    )

    report_dir = _report_output_dir(spec, analysis_run)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    report_path = report_dir / "report.typ"
    _write_typst_report(report_path, summary)
    compile_result = compile_typst(report_path)

    return {
        "report_dir": str(report_dir),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "compile": compile_result,
    }

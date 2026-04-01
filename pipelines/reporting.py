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
            "results": analysis_result.get("results"),
        },
    }


def _write_typst_report(report_path: Path, summary: dict[str, Any]) -> None:
    spec = dict(summary.get("spec") or {})
    publication = dict(summary.get("publication") or {})
    analysis = dict(summary.get("analysis") or {})
    runs = dict(summary.get("runs") or {})
    report_text = f"""#set page(
  paper: "us-letter",
  margin: (x: 0.72in, y: 0.78in),
)

#set par(justify: false, leading: 0.62em)
#set text(font: "Libertinus Serif", size: 10pt, fill: rgb("#16202A"))

#let navy = rgb("#16324F")
#let teal = rgb("#2E6A69")
#let muted = rgb("#5E6F82")
#let line = rgb("#D6DEE3")

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
  stroke: (paint: line, thickness: 0.7pt),
  radius: 10pt,
  inset: 12pt,
  width: 100%,
)[
  #text(size: 8pt, fill: muted, weight: "bold")[#label]
  #v(4pt)
  #text(size: 17pt, fill: navy, weight: "bold")[#value]
]

#heading(level: 1)[Workflow Report]

#text(size: 11pt, fill: muted)[
  Generated {_safe_text(summary.get("generated_at"))}
]

#v(10pt)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  stat("Spec", "{_safe_text(spec.get("name") or spec.get("id") or "Unnamed")}"),
  stat("Publication", "{_safe_text(publication.get("relation_name") or "n/a")}"),
  stat("Rows", "{_safe_text(publication.get("row_count") or "n/a")}"),
)

#heading(level: 2)[Workflow]

- Spec id: `{_safe_text(spec.get("id") or "n/a")}`
- Version: `{_safe_text(spec.get("version") or "n/a")}`
- Dataset run: `{_safe_text(runs.get("dataset") or "n/a")}`
- Capture run: `{_safe_text(runs.get("capture") or "n/a")}`
- Analysis run: `{_safe_text(runs.get("analysis") or "n/a")}`

#heading(level: 2)[Analysis]

- Mode: `{_safe_text(analysis.get("mode") or "n/a")}`
- Target: `{_safe_text(analysis.get("target") or "n/a")}`
- Data source: `{_safe_text(analysis.get("data_source") or "n/a")}`
- Pooling: `{_safe_text(analysis.get("pooling") or "n/a")}`
- Labels parquet: `{_safe_text(analysis.get("labels_path") or "n/a")}`
- Output dir: `{_safe_text(analysis.get("output_dir") or "n/a")}`

#heading(level: 2)[Result JSON]

```json
{_json_literal(analysis.get("results") or {})}
```
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

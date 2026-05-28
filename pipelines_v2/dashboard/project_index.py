"""Filesystem project index for xenon-style research report bundles.

The workflow catalog is run-oriented; xenon-projects is research-oriented:
Project -> Phase -> Experiment -> Report. This module bridges that shape for
the dashboard without reading activation payloads or arbitrary filesystem
paths.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from pipelines_v2.core.config import load_workspace_config
from pipelines_v2.dashboard.json import read_json_object_optional
from pipelines_v2.dashboard.models import (
    ProjectDataSource,
    ProjectExperimentSummary,
    ProjectLabelExplanation,
    ProjectPhaseSummary,
    ProjectReportSummary,
    ProjectsResponse,
    ProjectSummary,
)
from pipelines_v2.dashboard.reports import build_report_detail_from_root

PROJECT_ROOT_ENV = "XENON_PROJECTS_ROOT"
PROJECT_REPORT_PREFIX = "project:"
_REPORT_KEY_RE = re.compile(r"^[a-f0-9]{20,64}$")
_SKIP_PHASE_PARENT_PARTS = {
    ".git",
    "__pycache__",
    "assets",
    "data",
    "outputs",
    "reports",
    "results",
    "tables",
}


def discover_project_roots(workspace_root: Path | None = None) -> tuple[Path, ...]:
    """Return existing project roots in priority order.

    The dashboard prefers an explicit env var, then the sibling xenon-projects
    checkout, then the local workspace's `projects/` directory.
    """
    if workspace_root is None:
        workspace_root = load_workspace_config().workspace_root
    workspace_root = workspace_root.resolve()

    candidates: list[Path] = []
    env_value = os.environ.get(PROJECT_ROOT_ENV)
    if env_value:
        for item in env_value.split(os.pathsep):
            text = item.strip()
            if text:
                candidates.append(Path(text).expanduser())
    candidates.extend(
        [
            workspace_root.parent / "xenon-projects" / "projects",
            workspace_root / "projects",
        ]
    )

    seen: set[Path] = set()
    roots: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        roots.append(resolved)
    return tuple(roots)


def build_projects_index(
    *,
    workspace_root: Path | None = None,
    project_roots: Iterable[Path] | None = None,
) -> ProjectsResponse:
    if workspace_root is None:
        workspace_root = load_workspace_config().workspace_root
    roots = tuple(project_roots) if project_roots is not None else discover_project_roots(workspace_root)
    projects: list[ProjectSummary] = []
    seen_project_paths: set[Path] = set()
    for projects_root in roots:
        for project_dir in sorted(p for p in projects_root.iterdir() if p.is_dir()):
            resolved = project_dir.resolve()
            if resolved in seen_project_paths:
                continue
            seen_project_paths.add(resolved)
            project = _scan_project(project_dir, projects_root)
            if project is not None:
                projects.append(project)
    projects.sort(key=lambda item: (item.title.lower(), item.project_id.lower()))
    return ProjectsResponse(
        project_roots=[str(root.resolve()) for root in roots],
        projects=projects,
    )


def resolve_project_report_root(
    report_key: str,
    *,
    workspace_root: Path | None = None,
    project_roots: Iterable[Path] | None = None,
) -> Path:
    key = report_key.removeprefix(PROJECT_REPORT_PREFIX)
    if not _REPORT_KEY_RE.fullmatch(key):
        raise LookupError(f"Invalid project report key: {report_key}")
    if workspace_root is None:
        workspace_root = load_workspace_config().workspace_root
    roots = tuple(project_roots) if project_roots is not None else discover_project_roots(workspace_root)
    for projects_root in roots:
        for report_root in _iter_report_roots_for_projects_root(projects_root):
            if project_report_key(report_root) == key:
                return report_root.resolve()
    raise LookupError(f"Unknown project report key: {report_key}")


def project_report_key(report_root: Path) -> str:
    return hashlib.sha256(str(report_root.resolve()).encode("utf-8")).hexdigest()[:24]


def project_report_artifact_id(report_root: Path) -> str:
    return f"{PROJECT_REPORT_PREFIX}{project_report_key(report_root)}"


def _scan_project(project_dir: Path, projects_root: Path) -> ProjectSummary | None:
    spec = _read_json(project_dir / "project_spec.json")
    phase_dirs = _find_phase_dirs(project_dir)
    if not spec and not phase_dirs:
        return None

    label_docs = _label_docs(project_dir)
    project_labels = _labels_from_names(
        _string_list(spec.get("core_labels") if spec else None),
        source=_rel(project_dir / "project_spec.json", project_dir),
        descriptions=label_docs,
    )

    phases = [
        _scan_phase(
            phase_dir=phase_dir,
            project_dir=project_dir,
            project_labels=project_labels,
        )
        for phase_dir in phase_dirs
    ]
    phases = [phase for phase in phases if phase is not None]
    phases.sort(key=lambda phase: _phase_sort_key(phase.phase_id))

    title = _string(spec.get("title") if spec else None) or _humanize(project_dir.name)
    return ProjectSummary(
        project_id=_string(spec.get("project") if spec else None) or project_dir.name,
        title=title,
        path=str(project_dir.resolve()),
        relative_path=_rel(project_dir, projects_root),
        status=_string(spec.get("status") if spec else None),
        primary_question=_string(spec.get("primary_question") if spec else None),
        thin_waist=_string(spec.get("thin_waist") if spec else None),
        description=_string(spec.get("description") if spec else None),
        core_labels=[label.name for label in project_labels],
        candidate_datasets=_string_list(spec.get("candidate_datasets") if spec else None),
        phases=phases,
    )


def _scan_phase(
    *,
    phase_dir: Path,
    project_dir: Path,
    project_labels: list[ProjectLabelExplanation],
) -> ProjectPhaseSummary | None:
    reports_dir = phase_dir / "reports"
    specs = _workflow_specs(phase_dir)
    if not reports_dir.is_dir() and not specs and not (phase_dir / "PHASE.md").is_file():
        return None

    phase_doc = phase_dir / "PHASE.md"
    phase_id = _rel(phase_dir, project_dir)
    experiments: list[ProjectExperimentSummary] = []
    if reports_dir.is_dir():
        if _is_report_root(reports_dir):
            experiment = _scan_experiment(
                experiment_dir=reports_dir,
                reports_dir=reports_dir,
                phase_dir=phase_dir,
                project_dir=project_dir,
                specs=specs,
                project_labels=project_labels,
                experiment_id="phase_reports",
            )
            if experiment is not None:
                experiments.append(experiment)
        for experiment_dir in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
            experiment = _scan_experiment(
                experiment_dir=experiment_dir,
                reports_dir=reports_dir,
                phase_dir=phase_dir,
                project_dir=project_dir,
                specs=specs,
                project_labels=project_labels,
            )
            if experiment is not None:
                experiments.append(experiment)
    experiments.sort(key=lambda item: item.title.lower())

    return ProjectPhaseSummary(
        phase_id=phase_id,
        title=_markdown_title(phase_doc) or _humanize(phase_id),
        path=str(phase_dir.resolve()),
        relative_path=_rel(phase_dir, project_dir),
        phase_doc_path=str(phase_doc.resolve()) if phase_doc.is_file() else None,
        summary_text=_markdown_summary(phase_doc),
        experiments=experiments,
    )


def _scan_experiment(
    *,
    experiment_dir: Path,
    reports_dir: Path,
    phase_dir: Path,
    project_dir: Path,
    specs: list[Path],
    project_labels: list[ProjectLabelExplanation],
    experiment_id: str | None = None,
) -> ProjectExperimentSummary | None:
    experiment_id = experiment_id or _rel(experiment_dir, reports_dir)
    report_roots = _report_roots_in_experiment(experiment_dir)
    reports = [
        _scan_report(
            report_root=report_root,
            project_dir=project_dir,
            project_labels=project_labels,
        )
        for report_root in report_roots
    ]
    reports = [report for report in reports if report is not None]
    if not reports:
        return None

    matching_specs = _matching_specs(experiment_id, specs)
    data_sources = _dedupe_data_sources(source for report in reports for source in report.data_sources)
    labels = _dedupe_labels(label for report in reports for label in report.labels) or project_labels
    workflow_names = _dedupe_strings(
        [
            *[path.stem.removesuffix("_workflow") for path in matching_specs],
            *[name for report in reports for name in report.workflow_names],
        ]
    )
    summary_text = next((report.summary_text for report in reports if report.summary_text), None)

    return ProjectExperimentSummary(
        experiment_id=experiment_id,
        title=_humanize(experiment_id),
        experiment_category=_experiment_category(
            experiment_id=experiment_id,
            reports=reports,
            workflow_names=workflow_names,
            spec_paths=matching_specs,
        ),
        path=str(experiment_dir.resolve()),
        relative_path=_rel(experiment_dir, phase_dir),
        summary_text=summary_text,
        workflow_names=workflow_names,
        spec_paths=[_rel(path, project_dir) for path in matching_specs],
        data_sources=data_sources,
        labels=labels,
        reports=reports,
    )


def _scan_report(
    *,
    report_root: Path,
    project_dir: Path,
    project_labels: list[ProjectLabelExplanation],
) -> ProjectReportSummary | None:
    if not _is_report_root(report_root):
        return None
    report = _read_json(report_root / "report.json")
    summary = _read_json(report_root / "summary.json") or _read_legacy_summary(report_root)
    detail = build_report_detail_from_root(
        report_root,
        artifact_id=project_report_artifact_id(report_root),
        artifact_kind="report",
        run_id=_run_id(report),
    )

    labels = _labels_for_report(report, project_labels)
    headline = detail.headline or _scalar_headline(summary)
    return ProjectReportSummary(
        report_key=project_report_key(report_root),
        report_id=report_root.name,
        title=_report_title(report_root, report, summary),
        path=str(report_root.resolve()),
        relative_path=_rel(report_root, project_dir),
        artifact_id=_artifact_id(report),
        run_id=_run_id(report),
        generated_at=_generated_at(report_root, report, summary),
        summary_text=_summary_text(report_root, summary, headline),
        headline=headline,
        data_sources=_data_sources_for_report(report),
        labels=labels,
        workflow_names=_workflow_names(report),
        result_kinds=_result_kinds(report, detail.unsupported_inputs, detail.tables),
        figure_count=len(detail.figures),
        table_count=len(detail.tables),
        result_count=len(detail.results),
    )


def _experiment_category(
    *,
    experiment_id: str,
    reports: Iterable[ProjectReportSummary],
    workflow_names: Iterable[str],
    spec_paths: Iterable[Path],
) -> str:
    report_list = list(reports)
    result_kinds = {kind.lower() for report in report_list for kind in report.result_kinds}
    headline_keys = {
        str(key).lower()
        for report in report_list
        for key in (report.headline or {}).keys()
    }
    text = " ".join(
        [
            experiment_id,
            *workflow_names,
            *[str(path) for path in spec_paths],
            *[report.title for report in report_list],
            *result_kinds,
            *headline_keys,
        ]
    ).lower()

    if "sae" in text or "sparse_autoencoder" in text:
        return "sae"
    if "baseline_gate" in text or "baseline gates" in text or "gate_status" in headline_keys:
        return "baseline_gate"
    if "score" in text or "scored" in text:
        return "scoring"
    if any("probe" in kind for kind in result_kinds) or "readout" in text:
        return "probe_readout"
    if any("capture" in kind for kind in result_kinds) or "capture" in text:
        return "capture"
    if any("generation" in kind for kind in result_kinds) or "generation" in text:
        return "generation"
    if "audit" in text or "baseline" in text or "lexical" in text:
        return "audit"
    return "other"


def _find_phase_dirs(project_dir: Path) -> list[Path]:
    out: list[Path] = []
    for candidate in project_dir.rglob("phase_*"):
        if not candidate.is_dir():
            continue
        rel_parts = candidate.relative_to(project_dir).parts
        if any(part in _SKIP_PHASE_PARENT_PARTS for part in rel_parts[:-1]):
            continue
        if (
            (candidate / "PHASE.md").is_file()
            or (candidate / "reports").is_dir()
            or (candidate / "specs").is_dir()
        ):
            out.append(candidate)
    return sorted(out, key=lambda path: _phase_sort_key(_rel(path, project_dir)))


def _workflow_specs(phase_dir: Path) -> list[Path]:
    specs_dir = phase_dir / "specs"
    if not specs_dir.is_dir():
        return []
    return sorted(path for path in specs_dir.glob("*.py") if not path.name.startswith("_"))


def _matching_specs(experiment_id: str, specs: list[Path]) -> list[Path]:
    normalized_experiment = _normalize_token(experiment_id)
    matches = [
        path
        for path in specs
        if normalized_experiment in _normalize_token(path.stem)
        or _normalize_token(path.stem).removesuffix("workflow") in normalized_experiment
    ]
    return matches or []


def _report_roots_in_experiment(experiment_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    roots: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or not _is_report_root(path):
            return
        seen.add(resolved)
        roots.append(path)

    add(experiment_dir)
    for child in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        add(child)
    for marker in sorted(experiment_dir.rglob("summary.json")):
        if len(marker.relative_to(experiment_dir).parts) <= 3:
            add(marker.parent)
    for marker in sorted(experiment_dir.rglob("*.summary.json")):
        if len(marker.relative_to(experiment_dir).parts) <= 2:
            add(marker.parent)

    roots.sort(key=lambda path: (_mtime(path), path.name), reverse=True)
    return roots


def _iter_report_roots_for_projects_root(projects_root: Path) -> Iterable[Path]:
    for project_dir in sorted(p for p in projects_root.iterdir() if p.is_dir()):
        for phase_dir in _find_phase_dirs(project_dir):
            reports_dir = phase_dir / "reports"
            if not reports_dir.is_dir():
                continue
            if _is_report_root(reports_dir):
                yield reports_dir
            for experiment_dir in sorted(p for p in reports_dir.iterdir() if p.is_dir()):
                yield from _report_roots_in_experiment(experiment_dir)


def _is_report_root(path: Path) -> bool:
    return (
        (path / "report.json").is_file()
        or (path / "summary.json").is_file()
        or (path / "assets" / "manifest.json").is_file()
        or any(path.glob("*.summary.json"))
    )


def _data_sources_for_report(report: Mapping[str, Any] | None) -> list[ProjectDataSource]:
    if not isinstance(report, Mapping):
        return []
    out: list[ProjectDataSource] = []
    for item in _mapping_list(report.get("inputs")):
        coverage = _mapping(item.get("example_coverage"))
        summary = _mapping(item.get("summary"))
        out.append(
            ProjectDataSource(
                name=_string(coverage.get("dataset_name")) or _string(item.get("name")),
                dataset_id=_string(coverage.get("dataset_id")),
                description=_string(summary.get("dataset_description"))
                or _string(summary.get("description")),
                source=_string(item.get("artifact_id")),
                example_count=_int(coverage.get("example_count") or coverage.get("example_key_count")),
                label_names=_string_list(item.get("label_names")),
            )
        )
    return _dedupe_data_sources(out)


def _labels_for_report(
    report: Mapping[str, Any] | None,
    project_labels: list[ProjectLabelExplanation],
) -> list[ProjectLabelExplanation]:
    descriptions = {label.name: label.description for label in project_labels}
    names: list[str] = []
    if isinstance(report, Mapping):
        for item in _mapping_list(report.get("inputs")):
            names.extend(_string_list(item.get("label_names")))
    return _labels_from_names(names, source="report input label_names", descriptions=descriptions) or project_labels


def _labels_from_names(
    names: Iterable[str],
    *,
    source: str,
    descriptions: Mapping[str, str | None] | None = None,
) -> list[ProjectLabelExplanation]:
    descriptions = descriptions or {}
    return [
        ProjectLabelExplanation(
            name=name,
            description=descriptions.get(name),
            source=source,
        )
        for name in _dedupe_strings(names)
    ]


def _label_docs(project_dir: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in project_dir.rglob("*.md"):
        rel_parts = path.relative_to(project_dir).parts
        if any(part in {"reports", "data", "outputs"} for part in rel_parts[:-1]):
            continue
        lowered = path.name.lower()
        if "label" not in lowered and "schema" not in lowered and "rubric" not in lowered:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in re.finditer(r"(?m)^-\s+`([^`]+)`:\s+(.+)$", text):
            name = match.group(1).strip()
            description = match.group(2).strip()
            if name and description and name not in docs:
                docs[name] = description
    return docs


def _dedupe_data_sources(items: Iterable[ProjectDataSource]) -> list[ProjectDataSource]:
    by_key: dict[tuple[str, str, str], ProjectDataSource] = {}
    for item in items:
        key = (item.dataset_id or "", item.name or "", item.source or "")
        current = by_key.get(key)
        if current is None:
            by_key[key] = item
            continue
        current.label_names = _dedupe_strings([*current.label_names, *item.label_names])
        if current.example_count is None:
            current.example_count = item.example_count
    return sorted(by_key.values(), key=lambda item: (item.name or "", item.dataset_id or "", item.source or ""))


def _dedupe_labels(items: Iterable[ProjectLabelExplanation]) -> list[ProjectLabelExplanation]:
    out: dict[str, ProjectLabelExplanation] = {}
    for item in items:
        if item.name not in out or (not out[item.name].description and item.description):
            out[item.name] = item
    return [out[name] for name in sorted(out)]


def _workflow_names(report: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(report, Mapping):
        return []
    names = []
    for item in _mapping_list(report.get("inputs")):
        workflow = _mapping(item.get("workflow"))
        names.append(_string(workflow.get("workflow_name")))
    return _dedupe_strings(name for name in names if name)


def _result_kinds(
    report: Mapping[str, Any] | None,
    unsupported_inputs: list[dict[str, Any]],
    tables: Iterable[Any],
) -> list[str]:
    values: list[str] = []
    if isinstance(report, Mapping):
        for item in _mapping_list(report.get("inputs")):
            values.append(_string(item.get("artifact_kind")))
    for item in unsupported_inputs:
        values.append(_string(item.get("result_kind")))
    for table in tables:
        values.append(getattr(table, "result_kind", None))
    return _dedupe_strings(value for value in values if value)


def _artifact_id(report: Mapping[str, Any] | None) -> str | None:
    if not isinstance(report, Mapping):
        return None
    inputs = _mapping_list(report.get("inputs"))
    if len(inputs) == 1:
        return _string(inputs[0].get("artifact_id"))
    return None


def _run_id(report: Mapping[str, Any] | None) -> str | None:
    if not isinstance(report, Mapping):
        return None
    for item in _mapping_list(report.get("inputs")):
        workflow = _mapping(item.get("workflow"))
        run_id = _string(workflow.get("run_id"))
        if run_id:
            return run_id
    return None


def _report_title(
    report_root: Path,
    report: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
) -> str:
    for payload in (report, summary):
        if isinstance(payload, Mapping):
            title = _string(payload.get("title")) or _string(payload.get("name"))
            if title:
                return title
    return _humanize(report_root.name)


def _summary_text(
    report_root: Path,
    summary: Mapping[str, Any] | None,
    headline: Mapping[str, Any] | None,
) -> str | None:
    for name in ("summary.md", "report.md"):
        text = _markdown_summary(report_root / name)
        if text:
            return text
    for payload in (summary, headline):
        if isinstance(payload, Mapping):
            for key in ("summary", "description", "interpretation", "caveat", "notes"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _scalar_headline(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        if len(out) >= 16:
            break
    return out or None


def _generated_at(
    report_root: Path,
    report: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
) -> str | None:
    for payload in (report, summary):
        if isinstance(payload, Mapping):
            for key in ("generated_at", "created_at", "status_date"):
                value = _string(payload.get(key))
                if value:
                    return value
    inputs = _mapping_list(report.get("inputs")) if isinstance(report, Mapping) else []
    for item in inputs:
        value = _string(item.get("created_at"))
        if value:
            return value
    mtime = _mtime(report_root)
    if mtime > 0:
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _read_json(path: Path) -> dict[str, Any] | None:
    return read_json_object_optional(path)


def _read_legacy_summary(root: Path) -> dict[str, Any] | None:
    for path in sorted(root.glob("*.summary.json")):
        payload = _read_json(path)
        if payload is not None:
            return payload
    return None


def _markdown_title(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped.removeprefix("# ").strip()
    except OSError:
        return None
    return None


def _markdown_summary(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    paragraphs: list[str] = []
    current: list[str] = []
    in_frontmatter = False
    for raw in lines:
        line = raw.strip()
        if line == "---" and not paragraphs and not current:
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("#") or line.startswith("|") or line.startswith("```"):
            continue
        current.append(line)
        if len(" ".join(current)) > 420:
            break
    if current:
        paragraphs.append(" ".join(current))
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned and not cleaned.startswith("- "):
            return cleaned[:600]
    return None


def _phase_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"phase_(\d+)", value)
    return (int(match.group(1)) if match else 999, value)


def _humanize(value: str) -> str:
    text = value.replace("/", " / ").replace("_", " ").replace("-", " ")
    return " ".join(part.capitalize() if not part.isupper() else part for part in text.split())


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [text for item in value if (text := _string(item))]
    text = _string(value)
    return [text] if text else []


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _dedupe_strings(values: Iterable[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _string(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

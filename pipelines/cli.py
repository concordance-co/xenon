"""Canonical operator CLI for Xenon workflows."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


Handler = Callable[[list[str]], int]


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, default=str))


def _slugify(text: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text).strip().lower())
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "workflow"


def _run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _modal_run(args: list[str], *, extra: str) -> None:
    cmd = ["uv", "run", "--extra", extra, "--extra", "modal", "modal", "run", *args]
    _run_command(cmd)


def _modal_volume_get(volume_name: str, remote_path: str, local_path: Path) -> None:
    if remote_path.endswith("/"):
        local_path.mkdir(parents=True, exist_ok=True)
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["modal", "volume", "get", volume_name, remote_path, str(local_path), "--force"]
    _run_command(cmd)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _resolve_modal_download_dir(base_output_dir: Path, run_id: str) -> Path:
    candidate = base_output_dir / run_id
    return candidate if candidate.is_dir() else base_output_dir


def _open_conn():
    from pipelines.db import connect_neon, ensure_schema

    conn = connect_neon(autocommit=True)
    ensure_schema(conn)
    return conn


def _finish_workflow_run_resilient(
    conn: Any,
    *,
    run_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    from pipelines.workflows import finish_workflow_run

    try:
        finish_workflow_run(
            conn,
            run_id=run_id,
            status=status,
            result=result,
            error_text=error_text,
        )
        return
    except Exception:
        with _open_conn() as retry_conn:
            finish_workflow_run(
                retry_conn,
                run_id=run_id,
                status=status,
                result=result,
                error_text=error_text,
            )


def _load_spec_from_args(spec_id: str | None, file_path: str | None) -> dict[str, Any]:
    from pipelines.workflows import get_workflow_spec, normalize_workflow_spec, upsert_workflow_spec

    with _open_conn() as conn:
        if file_path:
            spec = json.loads(Path(file_path).read_text())
            spec = normalize_workflow_spec(spec)
            return upsert_workflow_spec(conn, spec)
        if not spec_id:
            raise SystemExit("Specify --spec or --file")
        spec = get_workflow_spec(conn, spec_id)
        if spec is None:
            raise SystemExit(f"Spec not found: {spec_id}")
        return spec


def _run_prepare_passthrough(argv: list[str]) -> int:
    from pipelines.datasets.prepare import main

    return int(main(argv) or 0)


def _run_capture_passthrough(argv: list[str]) -> int:
    from pipelines.interp.local_capture import main

    return int(main(argv) or 0)


def _run_analysis_passthrough(argv: list[str]) -> int:
    from pipelines.interp.analysis import main

    return int(main(argv) or 0)


def _spec_create_or_update(argv: list[str], *, mode: str) -> int:
    from pipelines.workflows import upsert_workflow_spec

    parser = argparse.ArgumentParser(prog=f"python -m pipelines.cli spec {mode}")
    parser.add_argument("--file", required=True)
    ns = parser.parse_args(argv)

    spec = json.loads(Path(ns.file).read_text())
    with _open_conn() as conn:
        saved = upsert_workflow_spec(conn, spec)
    _print_json({"spec": saved})
    return 0


def _spec_show(argv: list[str]) -> int:
    from pipelines.workflows import get_workflow_spec

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli spec show")
    parser.add_argument("--id", required=True)
    ns = parser.parse_args(argv)
    with _open_conn() as conn:
        spec = get_workflow_spec(conn, ns.id)
    if spec is None:
        raise SystemExit(f"Spec not found: {ns.id}")
    _print_json({"spec": spec})
    return 0


def _spec_list(argv: list[str]) -> int:
    del argv
    from pipelines.workflows import list_workflow_specs

    with _open_conn() as conn:
        specs = list_workflow_specs(conn)
    _print_json({"specs": specs})
    return 0


def _spec_delete(argv: list[str]) -> int:
    from pipelines.workflows import delete_workflow_spec

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli spec delete")
    parser.add_argument("--id", required=True)
    ns = parser.parse_args(argv)
    with _open_conn() as conn:
        deleted = delete_workflow_spec(conn, ns.id)
    if not deleted:
        raise SystemExit(f"Spec not found: {ns.id}")
    _print_json({"deleted": ns.id})
    return 0


def _run_list(argv: list[str]) -> int:
    del argv
    from pipelines.workflows import list_workflow_runs

    with _open_conn() as conn:
        runs = list_workflow_runs(conn)
    _print_json({"runs": runs})
    return 0


def _run_show(argv: list[str]) -> int:
    from pipelines.workflows import get_workflow_run

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli run show")
    parser.add_argument("--id", required=True)
    ns = parser.parse_args(argv)
    with _open_conn() as conn:
        run = get_workflow_run(conn, ns.id)
    if run is None:
        raise SystemExit(f"Run not found: {ns.id}")
    _print_json({"run": run})
    return 0


def _publication_list(argv: list[str]) -> int:
    from pipelines.workflows import list_publications

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli publication list")
    parser.add_argument("--spec", default=None)
    ns = parser.parse_args(argv)
    with _open_conn() as conn:
        publications = list_publications(conn, spec_id=ns.spec)
    _print_json({"publications": publications})
    return 0


def _project_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipelines.cli project init")
    parser.add_argument("--project", required=True)
    parser.add_argument("--subproject", default="subproject_01")
    parser.add_argument("--phase", default="phase_01")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--subproject-name", default=None)
    parser.add_argument("--phase-name", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--dataset-table", default="interp_examples_v0")
    parser.add_argument("--label-sql", default="decision_type")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--layers", default="16,24,32")
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args(argv)

    project_id = _slugify(ns.project).upper()
    subproject_id = _slugify(ns.subproject)
    phase_id = _slugify(ns.phase)
    project_title = ns.project_name or ns.project.strip()
    subproject_title = ns.subproject_name or ns.subproject.strip().replace("_", " ").title()
    phase_title = ns.phase_name or ns.phase.strip().replace("_", " ").title()
    phase_goal = ns.goal or f"Define the first runnable workflow for the {phase_title} phase."
    project_goal = ns.goal or f"Coordinate the {project_title} research program and its subproject-level workflow specs."
    layer_values = [int(token.strip()) for token in str(ns.layers).split(",") if token.strip()]

    project_root = Path("projects") / project_id
    subproject_root = project_root / subproject_id
    phase_root = subproject_root / phase_id
    workflow_path = phase_root / "specs" / "workflow.json"
    phase_spec_path = phase_root / "phase_spec.json"
    project_spec_path = project_root / "project_spec.json"

    expected_paths = [
        subproject_root / "__init__.py",
        subproject_root / "README.md",
        subproject_root / "shared",
        subproject_root / "outputs",
        subproject_root / "reports" / "assets",
        subproject_root / "reports" / "scripts",
        phase_root / "README.md",
        phase_root / "phase_spec.json",
        phase_root / "specs" / "workflow.json",
        phase_root / "scripts",
        phase_root / "outputs",
        phase_root / "reports" / "assets",
        phase_root / "reports" / "scripts",
    ]
    if not ns.force:
        existing = [path for path in expected_paths if path.exists()]
        if existing:
            sample = ", ".join(str(path) for path in existing[:5])
            suffix = "" if len(existing) <= 5 else f" (+{len(existing) - 5} more)"
            raise SystemExit(
                f"Project scaffolding already exists. Re-run with --force to overwrite the managed files. Existing paths: {sample}{suffix}"
            )

    for directory in [
        project_root / "docs",
        project_root / "shared",
        project_root / "outputs",
        project_root / "reports" / "assets",
        project_root / "reports" / "scripts",
        subproject_root / "shared",
        subproject_root / "outputs",
        subproject_root / "reports" / "assets",
        subproject_root / "reports" / "scripts",
        phase_root / "specs",
        phase_root / "scripts",
        phase_root / "outputs",
        phase_root / "reports" / "assets",
        phase_root / "reports" / "scripts",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    existing_project_spec = _load_json_if_exists(project_spec_path) or {}
    existing_subprojects = existing_project_spec.get("subprojects")
    if existing_subprojects is None:
        legacy_phase_refs = existing_project_spec.get("phases", [])
        existing_subprojects = []
        for ref in legacy_phase_refs:
            if isinstance(ref, str) and ref:
                candidate = ref.split("/", 1)[0]
                if candidate and candidate not in existing_subprojects:
                    existing_subprojects.append(candidate)
    project_spec = {
        "id": existing_project_spec.get("id", project_id),
        "name": existing_project_spec.get("name", project_title),
        "description": existing_project_spec.get("description", ns.description or f"Umbrella project for {project_title}."),
        "version": existing_project_spec.get("version", 1),
        "goal": existing_project_spec.get("goal", project_goal),
        "subprojects": list(existing_subprojects or []),
        "defaults": existing_project_spec.get(
            "defaults",
            {
                "capture_execution": "modal",
                "analysis_execution": "modal",
                "report_execution": "local",
            },
        ),
    }
    if subproject_id not in project_spec["subprojects"]:
        project_spec["subprojects"].append(subproject_id)
    phase_spec = {
        "id": phase_id,
        "name": phase_title,
        "description": ns.description or f"{phase_title} phase for {project_title} / {subproject_title}.",
        "project_id": project_id,
        "inherits": f"projects/{project_id}/project_spec.json",
        "goal": phase_goal,
        "workflow_specs": [f"projects/{project_id}/{subproject_id}/{phase_id}/specs/workflow.json"],
        "outputs": {
            "output_dir": f"projects/{project_id}/{subproject_id}/{phase_id}/reports",
            "notes": "Store phase-local artifacts and final report outputs here.",
        },
    }
    workflow_spec = {
        "id": f"{project_id.lower()}_{subproject_id}_{phase_id}",
        "name": f"{project_title} / {subproject_title} / {phase_title}",
        "description": ns.description or f"Workflow spec for the {phase_title} phase of {project_title} / {subproject_title}.",
        "version": 1,
        "dataset": {
            "source": {
                "mode": "table",
                "table": ns.dataset_table,
            },
            "filters": {},
            "label": {
                "mode": "direct",
                "expression_sql": ns.label_sql,
            },
            "split": {
                "mode": "random_stratified",
                "train_pct": 70,
                "val_pct": 15,
                "test_pct": 15,
            },
            "probe_defaults": {
                "data_source": "router",
                "pooling": "last_token",
                "n_folds": 5,
            },
            "publish_mode": "view",
        },
        "capture": {
            "model": ns.model,
            "layers": layer_values,
            "pooling": "mean_pool",
            "router": True,
            "residual": True,
            "execution": "modal",
        },
        "analysis": {
            "methods": ["probe"],
            "targets": ["workflow_label"],
            "data_source": "router",
            "pooling": "last_token",
            "execution": "modal",
        },
        "report": {
            "output_dir": f"projects/{project_id}/{subproject_id}/{phase_id}/reports",
        },
    }

    if ns.force or not (project_root / "__init__.py").exists():
        _write_text(project_root / "__init__.py", f'"""Project workspace for {project_id}."""\n')
    if ns.force or not (project_root / "README.md").exists():
        _write_text(
            project_root / "README.md",
            (
                f"# {project_id}\n\n"
                f"{project_title} is an umbrella project workspace.\n\n"
                "Use the checked-in phase workflow specs under `<subproject>/<phase>/specs/workflow.json` "
                "as the local mirror of the executable Neon-backed workflow spec.\n"
            ),
        )
    _write_text(subproject_root / "__init__.py", f'"""Subproject workspace for {project_id} / {subproject_id}."""\n')
    _write_text(
        subproject_root / "README.md",
        (
            f"# {subproject_title}\n\n"
            f"This subproject belongs to `{project_id}`.\n\n"
            "Use the checked-in phase workflow specs under:\n\n"
            f"- `projects/{project_id}/{subproject_id}/<phase>/specs/workflow.json`\n"
        ),
    )
    _write_text(
        phase_root / "README.md",
        (
            f"# {phase_title}\n\n"
            f"This phase belongs to `{project_id}/{subproject_id}`.\n\n"
            "Primary runnable spec:\n\n"
            f"- `projects/{project_id}/{subproject_id}/{phase_id}/specs/workflow.json`\n"
        ),
    )
    _write_json(project_spec_path, project_spec)
    _write_json(phase_spec_path, phase_spec)
    _write_json(workflow_path, workflow_spec)

    _print_json(
        {
            "project": project_id,
            "subproject": subproject_id,
            "phase": phase_id,
            "project_root": str(project_root),
            "subproject_root": str(subproject_root),
            "phase_root": str(phase_root),
            "workflow_spec": str(workflow_path),
        }
    )
    return 0


def _run_dataset_build(argv: list[str]) -> int:
    if "--spec" not in argv and "--file" not in argv:
        return _run_prepare_passthrough(argv)

    from pipelines.workflows import finish_workflow_run, publish_dataset, start_workflow_run

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli dataset build")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--file", default=None)
    ns = parser.parse_args(argv)

    spec = _load_spec_from_args(ns.spec, ns.file)
    with _open_conn() as conn:
        run = start_workflow_run(conn, spec=spec, run_type="dataset", source="cli", resolved_config={"action": "dataset_build"})
        try:
            publication = publish_dataset(conn, spec, run_id=run["id"])
            result = {"publication": publication}
            _finish_workflow_run_resilient(conn, run_id=run["id"], status="succeeded", result=result)
        except Exception as exc:
            _finish_workflow_run_resilient(conn, run_id=run["id"], status="failed", error_text=str(exc))
            raise
    _print_json({"run": run, "publication": publication})
    return 0


def _run_capture(argv: list[str]) -> int:
    if "--spec" not in argv and "--file" not in argv and "--publication" not in argv:
        return _run_capture_passthrough(argv)

    from pipelines.workflows import (
        finish_workflow_run,
        get_latest_publication_for_spec,
        start_workflow_run,
    )

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli capture run")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--publication", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/activations/workflows"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--execution", choices=["modal", "local"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--gpu-memory-utilization", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--add-generation-prompt", action="store_true")
    parser.add_argument("--capture-generation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--capture-reasoning", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reasoning-parser", default=None)
    parser.add_argument("--generation-max-tokens", type=int, default=None)
    parser.add_argument("--generation-temperature", type=float, default=None)
    parser.add_argument("--generation-top-p", type=float, default=None)
    parser.add_argument("--capture-router", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--capture-residual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pool-on-capture", choices=["last_token", "mean_pool"], default=None)
    parser.add_argument("--router-dtype", choices=["float16", "float32"], default=None)
    parser.add_argument("--metadata-flush-interval", type=int, default=None)
    ns = parser.parse_args(argv)

    spec = _load_spec_from_args(ns.spec, ns.file)
    capture_block = dict(spec.get("capture") or {})
    dataset_name = ns.publication
    with _open_conn() as conn:
        if not dataset_name:
            publication = get_latest_publication_for_spec(conn, spec["id"])
            if publication is None:
                raise SystemExit(f"No publication found for spec {spec['id']}. Run dataset build first.")
            dataset_name = publication["relation_name"]
        resolved = {
            "publication": dataset_name,
            "model_id": ns.model_id or capture_block.get("model") or capture_block.get("model_id") or "Qwen/Qwen3-8B",
            "device": ns.device or capture_block.get("device") or "mps",
            "execution": ns.execution or capture_block.get("execution") or "modal",
            "limit": ns.limit if ns.limit is not None else capture_block.get("limit"),
            "layers": ns.layers or capture_block.get("layers"),
            "gpu_memory_utilization": ns.gpu_memory_utilization or capture_block.get("gpu_memory_utilization") or "0.90",
            "max_model_len": ns.max_model_len if ns.max_model_len is not None else capture_block.get("max_model_len"),
            "capture_generation": ns.capture_generation if ns.capture_generation is not None else bool(capture_block.get("capture_generation", False)),
            "capture_reasoning": ns.capture_reasoning if ns.capture_reasoning is not None else bool(capture_block.get("capture_reasoning", True)),
            "enable_thinking": ns.enable_thinking if ns.enable_thinking is not None else bool(capture_block.get("enable_thinking", True)),
            "reasoning_parser": ns.reasoning_parser or capture_block.get("reasoning_parser") or "",
            "generation_max_tokens": ns.generation_max_tokens if ns.generation_max_tokens is not None else int(capture_block.get("generation_max_tokens") or 256),
            "generation_temperature": ns.generation_temperature if ns.generation_temperature is not None else float(capture_block.get("generation_temperature") or 0.0),
            "generation_top_p": ns.generation_top_p if ns.generation_top_p is not None else float(capture_block.get("generation_top_p") or 1.0),
            "capture_router": ns.capture_router if ns.capture_router is not None else bool(capture_block.get("router", True)),
            "capture_residual": ns.capture_residual if ns.capture_residual is not None else bool(capture_block.get("residual", True)),
            "pool_on_capture": ns.pool_on_capture or capture_block.get("pooling"),
            "router_dtype": ns.router_dtype or capture_block.get("router_dtype") or "float16",
            "metadata_flush_interval": ns.metadata_flush_interval or int(capture_block.get("metadata_flush_interval") or 10),
            "add_generation_prompt": ns.add_generation_prompt or bool(capture_block.get("add_generation_prompt")),
            "skip_existing": bool(ns.skip_existing or capture_block.get("skip_existing")),
            "output_dir": str(ns.output_dir),
        }
        run = start_workflow_run(conn, spec=spec, run_type="capture", source="cli", resolved_config=resolved)
        try:
            parsed_layers = resolved["layers"]
            if isinstance(parsed_layers, str):
                parsed_layers = [int(token.strip()) for token in parsed_layers.split(",") if token.strip()]
            if str(resolved["execution"]) == "modal":
                remote_subdir = f"workflows/{_slugify(spec['id'])}/{run['id']}"
                modal_args = [
                    "pipelines/interp/modal_vllm_orchestrator.py",
                    "--mode",
                    "capture",
                    "--source-relation",
                    str(dataset_name),
                    "--workflow-run-id",
                    str(run["id"]),
                    "--output-subdir",
                    remote_subdir,
                    "--model-id",
                    str(resolved["model_id"]),
                    "--gpu-memory-utilization",
                    str(resolved["gpu_memory_utilization"]),
                    "--router-dtype",
                    str(resolved["router_dtype"]),
                ]
                if resolved["limit"] is not None:
                    modal_args.extend(["--limit", str(resolved["limit"])])
                if resolved["max_model_len"] is not None:
                    modal_args.extend(["--max-model-len", str(resolved["max_model_len"])])
                if bool(resolved["capture_generation"]):
                    modal_args.append("--capture-generation")
                    if bool(resolved["capture_reasoning"]):
                        modal_args.append("--capture-reasoning")
                    else:
                        modal_args.append("--no-capture-reasoning")
                    if bool(resolved["enable_thinking"]):
                        modal_args.append("--enable-thinking")
                    else:
                        modal_args.append("--no-enable-thinking")
                    if str(resolved["reasoning_parser"]):
                        modal_args.extend(["--reasoning-parser", str(resolved["reasoning_parser"])])
                    modal_args.extend(["--generation-max-tokens", str(resolved["generation_max_tokens"])])
                    modal_args.extend(["--generation-temperature", str(resolved["generation_temperature"])])
                    modal_args.extend(["--generation-top-p", str(resolved["generation_top_p"])])
                if bool(resolved["add_generation_prompt"]):
                    modal_args.append("--add-generation-prompt")
                if parsed_layers:
                    modal_args.extend(["--layers", ",".join(str(v) for v in parsed_layers)])
                if bool(resolved["capture_router"]):
                    modal_args.append("--capture-router")
                else:
                    modal_args.append("--no-capture-router")
                if bool(resolved["capture_residual"]):
                    modal_args.append("--capture-residual")
                else:
                    modal_args.append("--no-capture-residual")
                if resolved["pool_on_capture"]:
                    modal_args.extend(["--pool", str(resolved["pool_on_capture"])])
                _modal_run(modal_args, extra="interp")
                result = {
                    "remote_activations_path": f"/data/activations/{remote_subdir}",
                    "remote_activations_subdir": remote_subdir,
                    "publication": dataset_name,
                    "execution": "modal",
                }
            else:
                from pipelines.interp.local_capture import CaptureConfig, run_capture

                cfg = CaptureConfig(
                    output_dir=ns.output_dir,
                    source_relation=dataset_name,
                    model_id=str(resolved["model_id"]),
                    device=str(resolved["device"]),
                    limit=resolved["limit"],
                    layers=parsed_layers,
                    skip_existing=bool(resolved["skip_existing"]),
                    add_generation_prompt=bool(resolved["add_generation_prompt"]),
                    capture_router=bool(resolved["capture_router"]),
                    capture_residual=bool(resolved["capture_residual"]),
                    pool_on_capture=resolved["pool_on_capture"],
                    router_dtype=str(resolved["router_dtype"]),
                    metadata_flush_interval=int(resolved["metadata_flush_interval"]),
                )
                run_capture(cfg)
                result = {
                    "activations_dir": str(cfg.output_dir),
                    "publication": dataset_name,
                    "execution": "local",
                }
            _finish_workflow_run_resilient(conn, run_id=run["id"], status="succeeded", result=result)
        except Exception as exc:
            _finish_workflow_run_resilient(conn, run_id=run["id"], status="failed", error_text=str(exc))
            raise
    capture_payload = {
        "run_id": run["id"],
        "publication": dataset_name,
    }
    if str(result.get("execution")) == "modal":
        capture_payload["remote_activations_subdir"] = result.get("remote_activations_subdir")
        capture_payload["remote_activations_path"] = result.get("remote_activations_path")
    else:
        capture_payload["activations_dir"] = str(ns.output_dir)
    _print_json(capture_payload)
    return 0


def _run_analysis(argv: list[str]) -> int:
    if "--spec" not in argv and "--file" not in argv and "--capture-run" not in argv:
        return _run_analysis_passthrough(argv)

    from pipelines.workflows import (
        export_publication_labels,
        finish_workflow_run,
        get_latest_publication_for_spec,
        get_workflow_run,
        start_workflow_run,
    )

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli analysis run")
    parser.add_argument("--spec", default=None)
    parser.add_argument("--file", default=None)
    parser.add_argument("--capture-run", default=None)
    parser.add_argument("--publication", default=None)
    parser.add_argument("--activations-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/workflows"))
    parser.add_argument("--execution", choices=["modal", "local"], default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--data-source", choices=["router", "residual"], default=None)
    parser.add_argument("--pooling", choices=["last_token", "mean_pool"], default=None)
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    ns = parser.parse_args(argv)

    spec: dict[str, Any] | None = None
    capture_result: dict[str, Any] | None = None
    if ns.capture_run:
        with _open_conn() as conn:
            capture_run = get_workflow_run(conn, ns.capture_run)
        if capture_run is None:
            raise SystemExit(f"Capture run not found: {ns.capture_run}")
        spec = dict(capture_run.get("spec_snapshot_json") or {})
        capture_result = dict(capture_run.get("result_json") or {})
    if spec is None:
        spec = _load_spec_from_args(ns.spec, ns.file)

    analysis_block = dict(spec.get("analysis") or {})
    dataset_block = dict(spec.get("dataset") or {})
    probe_defaults = dict(dataset_block.get("probe_defaults") or {})
    with _open_conn() as conn:
        publication_name = ns.publication or (capture_result or {}).get("publication")
        if not publication_name:
            publication = get_latest_publication_for_spec(conn, spec["id"])
            if publication is None:
                raise SystemExit(f"No publication found for spec {spec['id']}. Run dataset build first.")
            publication_name = publication["relation_name"]

        resolved_execution = ns.execution or analysis_block.get("execution") or "modal"
        activations_dir = ns.activations_dir
        if activations_dir is None and capture_result:
            capture_dir = capture_result.get("activations_dir")
            if capture_dir:
                activations_dir = Path(str(capture_dir))
        if str(resolved_execution) == "local" and activations_dir is None:
            raise SystemExit("Specify --activations-dir or --capture-run")
        labels_path = Path("data/workflows") / spec["id"] / "labels" / f"{publication_name}.parquet"
        export_publication_labels(conn, relation_name=publication_name, output_path=labels_path)

        methods = analysis_block.get("methods")
        default_mode = methods[0] if isinstance(methods, list) and methods else analysis_block.get("mode") or "probe"
        targets = analysis_block.get("targets")
        default_target = targets[0] if isinstance(targets, list) and targets else analysis_block.get("target") or "workflow_label"

        resolved = {
            "publication": publication_name,
            "activations_dir": str(activations_dir) if activations_dir is not None else None,
            "labels_path": str(labels_path),
            "execution": resolved_execution,
            "mode": ns.mode or default_mode,
            "target": ns.target or default_target,
            "data_source": ns.data_source or analysis_block.get("data_source") or probe_defaults.get("data_source") or "router",
            "pooling": ns.pooling or analysis_block.get("pooling") or probe_defaults.get("pooling") or "last_token",
            "n_folds": ns.n_folds or int(analysis_block.get("n_folds") or probe_defaults.get("n_folds") or 5),
            "layers": ns.layers or analysis_block.get("layers") or probe_defaults.get("layers"),
            "limit": ns.limit if ns.limit is not None else analysis_block.get("limit") or probe_defaults.get("limit"),
            "seed": ns.seed if ns.seed is not None else int(analysis_block.get("seed") or 42),
            "output_dir": str(ns.output_dir),
        }
        run = start_workflow_run(conn, spec=spec, run_type="analysis", source="cli", resolved_config=resolved)
        try:
            parsed_layers = resolved["layers"]
            if isinstance(parsed_layers, str):
                parsed_layers = [int(token.strip()) for token in parsed_layers.split(",") if token.strip()]
            if str(resolved["execution"]) == "modal":
                remote_activations_subdir = None
                if capture_result:
                    remote_activations_subdir = capture_result.get("remote_activations_subdir")
                if not remote_activations_subdir:
                    raise SystemExit("Modal analysis requires a capture run produced by Modal-backed capture.")
                remote_output_subdir = f"workflows/{_slugify(spec['id'])}/{run['id']}"
                modal_args = [
                    "pipelines/interp/modal_analysis.py",
                    "--mode",
                    str(resolved["mode"]),
                    "--target",
                    str(resolved["target"]),
                    "--data-source",
                    str(resolved["data_source"]),
                    "--pooling",
                    str(resolved["pooling"]),
                    "--n-folds",
                    str(resolved["n_folds"]),
                    "--seed",
                    str(resolved["seed"]),
                    "--relation-name",
                    str(publication_name),
                    "--activations-subdir",
                    str(remote_activations_subdir),
                    "--output-subdir",
                    remote_output_subdir,
                    "--labels-subdir",
                    f"workflows/{_slugify(spec['id'])}/{run['id']}",
                ]
                if parsed_layers:
                    modal_args.extend(["--layers", ",".join(str(v) for v in parsed_layers)])
                if resolved["limit"] is not None:
                    modal_args.extend(["--limit", str(resolved["limit"])])
                _modal_run(modal_args, extra="analysis")
                _modal_volume_get("xenon-data", f"analysis_results/{remote_output_subdir}/", ns.output_dir)
                local_output_dir = _resolve_modal_download_dir(ns.output_dir, run["id"])
                loaded_results = _load_json_if_exists(local_output_dir / "results.json")
                results = {
                    "publication": publication_name,
                    "labels_path": str(labels_path),
                    "output_dir": str(local_output_dir),
                    "requested_output_dir": str(ns.output_dir),
                    "remote_output_subdir": remote_output_subdir,
                    "remote_output_path": f"/data/analysis_results/{remote_output_subdir}",
                    "execution": "modal",
                    "results": loaded_results,
                }
            else:
                from pipelines.interp.analysis import AnalysisConfig, dispatch

                config = AnalysisConfig(
                    activations_dir=Path(str(resolved["activations_dir"])),
                    labels_path=labels_path,
                    output_dir=ns.output_dir,
                    mode=str(resolved["mode"]),
                    target=str(resolved["target"]),
                    data_source=str(resolved["data_source"]),
                    pooling=str(resolved["pooling"]),
                    n_folds=int(resolved["n_folds"]),
                    layers=parsed_layers,
                    limit=resolved["limit"],
                    run_subdir=True,
                    seed=int(resolved["seed"]),
                )
                dispatch(config)
                results = {
                    "publication": publication_name,
                    "labels_path": str(labels_path),
                    "output_dir": str(config.output_dir),
                    "execution": "local",
                    "results": _load_json_if_exists(config.output_dir / "results.json"),
                }
            _finish_workflow_run_resilient(
                conn,
                run_id=run["id"],
                status="succeeded",
                result=results,
            )
        except Exception as exc:
            _finish_workflow_run_resilient(conn, run_id=run["id"], status="failed", error_text=str(exc))
            raise
    _print_json({"run_id": run["id"], "labels_path": str(labels_path), "publication": publication_name})
    return 0


def _run_report_build(argv: list[str]) -> int:
    from pipelines.reporting import build_workflow_report
    from pipelines.workflows import get_workflow_run

    parser = argparse.ArgumentParser(prog="python -m pipelines.cli report build")
    parser.add_argument("--analysis-run", default=None)
    parser.add_argument("--spec", default=None)
    ns = parser.parse_args(argv)

    with _open_conn() as conn:
        analysis_run_id = ns.analysis_run
        if analysis_run_id is None:
            if not ns.spec:
                raise SystemExit("Specify --analysis-run or --spec")
            rows = conn.execute(
                "SELECT id FROM workflow_runs WHERE spec_id = %s AND run_type = 'analysis' AND status = 'succeeded' "
                "ORDER BY created_at DESC LIMIT 1",
                [ns.spec],
            ).fetchall()
            if not rows:
                raise SystemExit(f"No successful analysis run found for spec {ns.spec}")
            analysis_run_id = rows[0]["id"]
        if get_workflow_run(conn, analysis_run_id) is None:
            raise SystemExit(f"Analysis run not found: {analysis_run_id}")
        result = build_workflow_report(conn, analysis_run_id=analysis_run_id)
    _print_json(result)
    return 0


def _not_implemented(argv: list[str], surface: str) -> int:
    del argv
    raise SystemExit(
        f"`{surface}` is reserved in pipelines.cli but not implemented yet. "
        "Use the underlying module entrypoint for now."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipelines.cli",
        description="Canonical Xenon operator CLI.",
    )
    parser.add_argument("surface", nargs="?", help="Top-level workflow surface.")
    parser.add_argument("action", nargs="?", help="Action for the selected surface.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying command.")
    return parser


def _resolve_handler(surface: str | None, action: str | None) -> Handler | None:
    routes: dict[tuple[str, str], Handler] = {
        ("project", "init"): _project_init,
        ("dataset", "build"): _run_dataset_build,
        ("capture", "run"): _run_capture,
        ("analysis", "run"): _run_analysis,
        ("report", "build"): _run_report_build,
        ("spec", "create"): lambda argv: _spec_create_or_update(argv, mode="create"),
        ("spec", "update"): lambda argv: _spec_create_or_update(argv, mode="update"),
        ("spec", "show"): _spec_show,
        ("spec", "list"): _spec_list,
        ("spec", "delete"): _spec_delete,
        ("run", "list"): _run_list,
        ("run", "show"): _run_show,
        ("publication", "list"): _publication_list,
    }
    if surface is None or action is None:
        return None
    return routes.get((surface, action))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    handler = _resolve_handler(ns.surface, ns.action)
    if handler is None:
        parser.print_help()
        return 0
    forwarded = list(ns.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return handler(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

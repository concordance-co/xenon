"""FastAPI read-only server for the pipelines_v2 dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pipelines_v2.dashboard.catalog import DashboardCatalog, build_catalog
from pipelines_v2.dashboard.models import (
    DatasetPreview,
    LabelPreview,
    PromptPreview,
    ReportDetail,
    ResultPreview,
    RunDetail,
    RunsResponse,
    StepDetail,
    StepDetailList,
)
from pipelines_v2.dashboard.normalize import (
    build_run_detail,
    build_run_summary,
    build_run_summary_from_counts,
)
from pipelines_v2.dashboard.previews import (
    DEFAULT_SAMPLE_SIZE,
    build_dataset_preview,
    build_label_preview,
    clear_resolved_dataset_cache,
)
from pipelines_v2.dashboard.prompt_preview import build_prompt_preview
from pipelines_v2.dashboard.result_preview import read_result_payload
from pipelines_v2.dashboard.reports import (
    ReportUnavailable,
    build_report_detail,
    safe_asset_path,
)
from pipelines_v2.dashboard.step_detail import build_step_detail

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(
    *,
    catalog: DashboardCatalog | None = None,
    local_root: Path | str | None = None,
    catalog_postgres_env: str | None = None,
    static_dir: Path | str | None = None,
    cache_list_ttl: float = 15.0,
    cache_hot_ttl: float = 15.0,
    cache_cold_ttl: float = 3600.0,
    cache_maxsize: int = 2048,
) -> "FastAPI":
    """Build the FastAPI application.

    `catalog` takes precedence; otherwise one is constructed from the
    `local_root` / `catalog_postgres_env` kwargs. `static_dir` optionally
    serves a prebuilt frontend bundle (e.g. `dashboard/dist`). Cache TTLs
    flow through to `build_catalog`'s `CachedCatalog` wrapper.
    """
    # Imported lazily so the pipelines_v2 package stays importable without the
    # optional `dashboard` dependency set.
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware

    dash = catalog or build_catalog(
        local_root=local_root,
        catalog_postgres_env=catalog_postgres_env,
        list_ttl=cache_list_ttl,
        hot_ttl=cache_hot_ttl,
        cold_ttl=cache_cold_ttl,
        cache_maxsize=cache_maxsize,
    )

    @asynccontextmanager
    async def lifespan(_app):  # noqa: ARG001 - standard FastAPI lifespan signature
        try:
            yield
        finally:
            dash.close()

    app = FastAPI(
        title="pipelines_v2 dashboard",
        description="Read-only observability for pipelines_v2 workflow runs.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Dev-mode CORS: the Vite dev server runs on a different port. This is an
    # internal operator tool with no auth, so an open policy is acceptable for
    # localhost usage.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "catalog": dash.identity()}

    @app.get("/api/runs", response_model=RunsResponse)
    def list_runs(
        status: str | None = Query(default=None),
        workflow_name: str | None = Query(default=None),
        limit: int | None = Query(default=100, ge=1, le=1000),
    ) -> RunsResponse:
        # Lightweight fast path when Postgres is attached:
        #  * `list_workflow_runs_light` — one pooled query that SKIPS the
        #    multi-MB `workflow_payload` jsonb entirely and computes
        #    `has_report` server-side via jsonb_path_exists.
        #  * `step_status_counts` — one aggregated GROUP BY for all runs.
        # With both, the /api/runs index is 2 queries total that together
        # transport a few KB instead of 100+ MB.
        if dash.pg is not None:
            try:
                light_rows = dash.pg.list_workflow_runs_light(
                    workflow_name=workflow_name, status=status, limit=limit
                )
            except Exception:
                light_rows = None
            if light_rows is not None:
                run_ids = [r["run_id"] for r in light_rows]
                try:
                    counts_by_run = (
                        dash.pg.step_status_counts(run_ids) if run_ids else {}
                    )
                except Exception:
                    counts_by_run = {}
                summaries = [
                    _summary_from_light(row, counts_by_run.get(row["run_id"], {}))
                    for row in light_rows
                ]
                # Merge any local-only runs (rare, but keeps parity with the
                # composite view). FileCatalog reads are disk-cheap.
                if dash.local is not None:
                    seen = {r["run_id"] for r in light_rows}
                    try:
                        local_records = dash.local.list_workflow_runs(
                            workflow_name=workflow_name, status=status, limit=limit
                        )
                    except Exception:
                        local_records = []
                    for rec in local_records:
                        if rec.run_id in seen:
                            continue
                        steps = dash.composite.list_workflow_steps(rec.run_id)
                        summaries.append(build_run_summary(rec, steps))
                    summaries.sort(key=lambda s: (s.started_at, s.run_id), reverse=True)
                    if limit is not None:
                        summaries = summaries[:limit]
                return RunsResponse(runs=summaries)

        # Fallback: no pg pool. Use the cached composite catalog.
        records = dash.composite.list_workflow_runs(
            workflow_name=workflow_name, status=status, limit=limit
        )
        if not records:
            return RunsResponse(runs=[])
        summaries = []
        for record in records:
            steps = dash.composite.list_workflow_steps(record.run_id)
            summaries.append(build_run_summary(record, steps))
        return RunsResponse(runs=summaries)

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        steps = dash.composite.list_workflow_steps(run_id)
        return build_run_detail(run, steps)

    @app.get("/api/runs/{run_id}/steps/{step_name}", response_model=StepDetail)
    def get_step(run_id: str, step_name: str) -> StepDetail:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        steps = dash.composite.list_workflow_steps(run_id)
        target = next((s for s in steps if s.step_name == step_name), None)
        # The step may exist in the workflow spec but not yet have a record.
        # build_step_detail gracefully handles missing records via build_run_detail.

        manifest = None
        if target is not None:
            manifest = dash.composite.find_artifact_for_workflow_step(
                run_id=run_id,
                workflow_step_key=target.workflow_step_key,
            )
            if manifest is None and target.artifact_id is not None:
                manifest = dash.composite.load_artifact(target.artifact_id)

        report_artifact_id = _nearest_report_artifact(run, steps, step_name)

        try:
            return build_step_detail(
                run=run,
                step_records=steps,
                target_step=step_name,
                artifact_manifest=manifest,
                report_artifact_id=report_artifact_id,
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/steps-detail", response_model=StepDetailList)
    def bulk_step_detail(run_id: str) -> StepDetailList:
        """Return step detail for every step in the run in a single
        round-trip's worth of DB work (three queries total against Postgres,
        independent of step count).
        """
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        step_records = dash.composite.list_workflow_steps(run_id)

        # Resolve all step artifacts in at most two queries:
        # 1) batch_load_artifacts for step records that have a recorded
        #    artifact_id (the common case once a step has completed).
        # 2) find_artifacts_for_run for any remaining steps (runtime-written
        #    artifacts whose artifact_id isn't back-filled on the step row yet).
        manifest_by_step: dict[str, Any] = {}
        if dash.pg is not None:
            try:
                artifact_ids = [
                    s.artifact_id for s in step_records if s.artifact_id is not None
                ]
                if artifact_ids:
                    by_aid = dash.pg.batch_load_artifacts(artifact_ids)
                    for s in step_records:
                        if s.artifact_id and s.artifact_id in by_aid:
                            manifest_by_step[s.step_name] = by_aid[s.artifact_id]
                missing = [s for s in step_records if s.step_name not in manifest_by_step]
                if missing:
                    by_key = dash.pg.find_artifacts_for_run(run_id)
                    for s in missing:
                        m = by_key.get(s.workflow_step_key)
                        if m is not None:
                            manifest_by_step[s.step_name] = m
            except Exception:
                manifest_by_step = {}

        # Fallback through the cached composite catalog for anything pg
        # didn't resolve (typically local-only catalog entries).
        details: list[StepDetail] = []
        for step_record in step_records:
            manifest = manifest_by_step.get(step_record.step_name)
            if manifest is None:
                manifest = dash.composite.find_artifact_for_workflow_step(
                    run_id=run_id,
                    workflow_step_key=step_record.workflow_step_key,
                )
                if manifest is None and step_record.artifact_id is not None:
                    manifest = dash.composite.load_artifact(step_record.artifact_id)
            report_artifact_id = _nearest_report_artifact(
                run, step_records, step_record.step_name
            )
            try:
                details.append(
                    build_step_detail(
                        run=run,
                        step_records=step_records,
                        target_step=step_record.step_name,
                        artifact_manifest=manifest,
                        report_artifact_id=report_artifact_id,
                    )
                )
            except LookupError:
                continue
        return StepDetailList(step_details=details)

    @app.post("/api/cache/invalidate")
    def cache_invalidate() -> dict[str, Any]:
        result: dict[str, Any] = {"ok": True}
        invalidator = getattr(dash.composite, "invalidate_all", None)
        if callable(invalidator):
            invalidator()
            result["catalog_cache"] = "cleared"
        else:
            result["catalog_cache"] = "not cached"
        clear_resolved_dataset_cache()
        result["resolved_datasets"] = "cleared"
        return result

    @app.get("/api/cache/stats")
    def cache_stats() -> dict[str, Any]:
        out: dict[str, Any] = {"ok": True}
        stats_fn = getattr(dash.composite, "stats", None)
        if callable(stats_fn):
            out["catalog_cache"] = stats_fn()
        else:
            out["catalog_cache"] = None
        if dash.pg is not None:
            out["pg"] = dash.pg.stats()
        return out

    @app.get(
        "/api/runs/{run_id}/steps/{step_name}/dataset-preview",
        response_model=DatasetPreview,
    )
    def dataset_preview(
        run_id: str,
        step_name: str,
        sample_size: int = Query(default=DEFAULT_SAMPLE_SIZE, ge=1, le=500),
        source_step: str | None = Query(default=None),
    ) -> DatasetPreview:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return build_dataset_preview(
            run=run,
            target_step=step_name,
            sample_size=sample_size,
            source_step=source_step,
        )

    @app.get(
        "/api/runs/{run_id}/steps/{step_name}/label-preview",
        response_model=LabelPreview,
    )
    def label_preview(
        run_id: str,
        step_name: str,
        source_step: str | None = Query(default=None),
        sample_size: int = Query(default=DEFAULT_SAMPLE_SIZE, ge=1, le=500),
    ) -> LabelPreview:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return build_label_preview(
            run=run,
            target_step=step_name,
            source_step=source_step,
            sample_size=sample_size,
        )

    @app.get("/api/reports/{artifact_id}", response_model=ReportDetail)
    def report_detail(artifact_id: str) -> ReportDetail:
        manifest = dash.composite.load_artifact(artifact_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {artifact_id}")
        try:
            return build_report_detail(manifest)
        except ReportUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/reports/{artifact_id}/assets/{asset_path:path}")
    def report_asset(artifact_id: str, asset_path: str):
        from fastapi.responses import FileResponse
        from pipelines_v2.dashboard.reports import resolve_report_root

        manifest = dash.composite.load_artifact(artifact_id)
        if manifest is None:
            raise HTTPException(status_code=404, detail=f"Unknown artifact_id: {artifact_id}")
        try:
            root = resolve_report_root(manifest)
            file_path = safe_asset_path(root, asset_path)
        except ReportUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return FileResponse(file_path)

    @app.get(
        "/api/runs/{run_id}/steps/{step_name}/result",
        response_model=ResultPreview,
    )
    def step_result(run_id: str, step_name: str) -> ResultPreview:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        steps = dash.composite.list_workflow_steps(run_id)
        target = next((s for s in steps if s.step_name == step_name), None)
        artifact = None
        if target is not None:
            artifact = dash.composite.find_artifact_for_workflow_step(
                run_id=run_id,
                workflow_step_key=target.workflow_step_key,
            )
            if artifact is None and target.artifact_id is not None:
                artifact = dash.composite.load_artifact(target.artifact_id)
        # If a downstream report step owns a copy of this step's result on
        # disk, we can surface that even if the step's own result ref is remote.
        report_aid = _nearest_report_artifact(run, steps, step_name)
        report_manifest = dash.composite.load_artifact(report_aid) if report_aid else None
        return read_result_payload(
            artifact_manifest=artifact,
            report_manifest=report_manifest,
            step_name=step_name,
        )

    @app.get(
        "/api/runs/{run_id}/steps/{step_name}/prompt-preview",
        response_model=PromptPreview,
    )
    def prompt_preview(
        run_id: str,
        step_name: str,
        max_examples: int = Query(default=3, ge=1, le=10),
    ) -> PromptPreview:
        run = dash.composite.load_workflow_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return build_prompt_preview(run=run, target_step=step_name, max_examples=max_examples)

    if static_dir is not None:
        _mount_static(app, Path(static_dir))

    return app


def _summary_from_light(row: dict[str, Any], status_counts: dict[str, int]):
    """Build a RunSummary from a DashboardPg lightweight list row plus
    aggregated step status counts. Avoids ever loading the run's
    workflow_payload."""
    from pipelines_v2.dashboard.models import RunSummary, StepCounts

    counts = StepCounts(total=0)
    buckets = {"completed", "failed", "running", "reused", "pending"}
    for status, n in status_counts.items():
        key = (status or "").lower()
        counts.total += int(n)
        if key in buckets:
            setattr(counts, key, getattr(counts, key) + int(n))
        else:
            counts.other += int(n)
    return RunSummary(
        run_id=row["run_id"],
        workflow_name=row["workflow_name"],
        workflow_hash=row["workflow_hash"],
        workflow_spec_hash=row["workflow_spec_hash"],
        status=row["status"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        parent_run_id=row["parent_run_id"],
        error=row["error"],
        step_counts=counts,
        has_report=bool(row["has_report"]),
    )


def _nearest_report_artifact(
    run: Any,
    steps: list[Any],
    step_name: str,
) -> str | None:
    """Return the artifact id of a report step that consumes this step.

    Used by the inspector Results tab to show a 'open report' link. The
    heuristic is intentionally simple: any downstream step whose kind is
    `report` and which has a recorded artifact_id. If the current step is
    itself a report, return its own artifact id.
    """
    from pipelines_v2.workflow.specs import WorkflowSpec

    try:
        spec = WorkflowSpec.from_dict(run.workflow_payload)
    except Exception:
        return None

    steps_by_name = {s.step_name: s for s in steps}
    report_names: set[str] = set()
    for wf_step in spec.steps:
        if getattr(wf_step.spec, "kind", None) == "report":
            report_names.add(wf_step.name)
    if not report_names:
        return None

    if step_name in report_names:
        record = steps_by_name.get(step_name)
        return record.artifact_id if record else None

    # Walk the dependency closure forward from `step_name`.
    children: dict[str, list[str]] = {s.name: [] for s in spec.steps}
    for s in spec.steps:
        for dep in s.resolved_depends_on():
            children.setdefault(dep, []).append(s.name)

    visited: set[str] = set()
    stack = [step_name]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node in report_names:
            record = steps_by_name.get(node)
            if record and record.artifact_id:
                return record.artifact_id
        stack.extend(children.get(node, ()))
    return None


def _mount_static(app: "FastAPI", static_dir: Path) -> None:
    from fastapi.staticfiles import StaticFiles

    if not static_dir.exists():
        return
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="dashboard")

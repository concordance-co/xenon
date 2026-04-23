"""FastAPI read-only server for the pipelines_v2 dashboard."""

from __future__ import annotations

import dataclasses
import threading
import time as time_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipelines_v2.cli import (
    _build_report_spec_from_run,
    _report_artifact_store_for_run,
)
from pipelines_v2.core.config import load_workspace_config
from pipelines_v2.dashboard.catalog import DashboardCatalog, build_catalog
from pipelines_v2.dashboard.models import (
    DatasetPreview,
    LabelPreview,
    PromptPreview,
    ReportDetail,
    ReportGenerationResponse,
    ResultPreview,
    RunDetail,
    RunReportStatus,
    RunsResponse,
    StepDetail,
    StepDetailList,
)
from pipelines_v2.dashboard.normalize import build_run_detail, build_run_summary
from pipelines_v2.dashboard.previews import (
    DEFAULT_SAMPLE_SIZE,
    build_dataset_preview,
    build_label_preview,
    clear_resolved_dataset_cache,
)
from pipelines_v2.dashboard.prompt_preview import build_prompt_preview, clear_tokenizer_cache
from pipelines_v2.dashboard.result_preview import read_result_payload
from pipelines_v2.dashboard.reports import (
    ReportUnavailable,
    build_report_detail,
    resolve_report_root,
    safe_asset_path,
)
from pipelines_v2.dashboard.step_detail import build_step_detail_from_run_detail
from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.runtime.local import LocalRunner
from pipelines_v2.workflow.records import WorkflowStepContext, WorkflowStepRecord
from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep

if TYPE_CHECKING:
    from fastapi import FastAPI


@dataclass(frozen=True, slots=True)
class _RunBundle:
    run: Any
    step_records: list[Any]
    run_detail: RunDetail
    read_catalog: Any


@dataclass(frozen=True, slots=True)
class _CatalogSource:
    name: str
    catalog: Any


@dataclass(slots=True)
class _DashboardAppState:
    runs_cache: dict[tuple[Any, ...], tuple[float, RunsResponse]] = field(default_factory=dict)
    run_bundle_cache: dict[str, tuple[float, _RunBundle]] = field(default_factory=dict)
    run_bundle_building: dict[str, threading.Event] = field(default_factory=dict)
    run_bundle_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(frozen=True, slots=True)
class _DashboardReadCatalog:
    sources: tuple[_CatalogSource, ...]

    def load_workflow_run(self, run_id: str) -> Any | None:
        for source in self.sources:
            catalog = source.catalog
            loader = getattr(catalog, "load_workflow_run", None)
            if not callable(loader):
                continue
            try:
                record = loader(run_id)
            except Exception:
                continue
            if record is not None:
                return record
        return None

    def list_workflow_steps(self, run_id: str) -> list[Any]:
        candidates: list[tuple[str, list[Any]]] = []
        for source in self.sources:
            catalog = source.catalog
            lister = getattr(catalog, "list_workflow_steps", None)
            if not callable(lister):
                continue
            try:
                records = list(lister(run_id))
            except Exception:
                continue
            if records:
                candidates.append((source.name, records))
        return _merge_step_records(candidates)

    def load_artifact(self, artifact_id: str) -> Any | None:
        for source in self.sources:
            catalog = source.catalog
            loader = getattr(catalog, "load_artifact", None)
            if not callable(loader):
                continue
            try:
                manifest = loader(artifact_id)
            except Exception:
                continue
            if manifest is not None:
                return manifest
        return None

    def find_artifact_for_workflow_step(
        self,
        *,
        run_id: str,
        workflow_step_key: str,
    ) -> Any | None:
        best: tuple[tuple[str, int], Any] | None = None
        for source in self.sources:
            catalog = source.catalog
            finder = getattr(catalog, "find_artifact_for_workflow_step", None)
            if not callable(finder):
                continue
            try:
                manifest = finder(run_id=run_id, workflow_step_key=workflow_step_key)
            except Exception:
                continue
            if manifest is not None:
                key = _artifact_source_sort_key(manifest, source.name)
                if best is None or key > best[0]:
                    best = (key, manifest)
        return best[1] if best is not None else None


_STEP_STATUS_PRIORITY = {
    "failed": 60,
    "completed": 50,
    "reused": 45,
    "running": 30,
    "pending": 10,
}

_SOURCE_PRIORITY = {
    "local": 40,
    "pg": 30,
    "composite": 20,
    "raw": 10,
}


def _step_record_sort_key(record: Any, source_name: str) -> tuple[str, int, int, int, int]:
    status = str(getattr(record, "status", "") or "").lower()
    latest_timestamp = str(
        getattr(record, "finished_at", None)
        or getattr(record, "started_at", None)
        or ""
    )
    completeness = sum(
        1
        for value in (
            getattr(record, "artifact_id", None),
            getattr(record, "artifact_kind", None),
            getattr(record, "runtime_app_id", None),
            getattr(record, "reused_from_run_id", None),
            getattr(record, "reused_from_artifact_id", None),
        )
        if value
    )
    return (
        latest_timestamp,
        _STEP_STATUS_PRIORITY.get(status, 0),
        int(bool(getattr(record, "artifact_id", None))),
        completeness,
        _SOURCE_PRIORITY.get(source_name, 0),
    )


def _merge_step_records(candidates: list[tuple[str, list[Any]]]) -> list[Any]:
    merged: dict[tuple[str, str], tuple[tuple[str, int, int, int, int], Any]] = {}
    for source_name, records in candidates:
        for record in records:
            key = (str(getattr(record, "run_id", "")), str(getattr(record, "step_name", "")))
            candidate_key = _step_record_sort_key(record, source_name)
            current = merged.get(key)
            if current is None or candidate_key > current[0]:
                merged[key] = (candidate_key, record)
    return sorted(
        (record for _sort_key, record in merged.values()),
        key=lambda record: (getattr(record, "step_index", 0), getattr(record, "step_name", "")),
    )


def _artifact_source_sort_key(manifest: Any, source_name: str) -> tuple[str, int]:
    return (
        str(getattr(manifest, "created_at", "") or ""),
        _SOURCE_PRIORITY.get(source_name, 0),
    )


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

    workspace_config = load_workspace_config()
    dash = catalog or build_catalog(
        local_root=local_root,
        catalog_postgres_env=catalog_postgres_env,
        list_ttl=cache_list_ttl,
        hot_ttl=cache_hot_ttl,
        cold_ttl=cache_cold_ttl,
        cache_maxsize=cache_maxsize,
    )
    runtime_state = _DashboardAppState()

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
    app.state.dashboard_runtime = runtime_state
    app.state.dashboard_catalog = dash

    # Dev-mode CORS: the Vite dev server runs on a different port. This is an
    # internal operator tool with no auth, so an open policy is acceptable for
    # localhost usage.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Request timing middleware — logs every request with wall-clock duration
    # so we can tell what's slow. Also surfaces timing as an X-Response-Time
    # header visible in browser dev tools.
    import logging as _logging
    import time as _time

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as _Request
    from starlette.responses import Response as _Response

    _perf_log = _logging.getLogger("pipelines_v2.dashboard.perf")
    _perf_log.setLevel(_logging.DEBUG)
    if not _perf_log.handlers:
        _handler = _logging.StreamHandler()
        _handler.setFormatter(_logging.Formatter("[perf] %(message)s"))
        _perf_log.addHandler(_handler)

    class _TimingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: _Request, call_next):  # type: ignore[override]
            t0 = _time.perf_counter()
            response: _Response = await call_next(request)
            dt = _time.perf_counter() - t0
            ms = dt * 1000
            response.headers["X-Response-Time"] = f"{ms:.1f}ms"
            level = (
                _logging.WARNING if ms > 2000
                else _logging.INFO if ms > 200
                else _logging.DEBUG
            )
            _perf_log.log(
                level,
                "%s %s → %d in %.1fms",
                request.method,
                request.url.path,
                response.status_code,
                ms,
            )
            return response

    app.add_middleware(_TimingMiddleware)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"ok": True, "catalog": dash.identity()}

    _RUNS_CACHE_TTL = 15.0

    @app.get("/api/runs", response_model=RunsResponse)
    def list_runs(
        status: str | None = Query(default=None),
        workflow_name: str | None = Query(default=None),
        limit: int | None = Query(default=100, ge=1, le=1000),
    ) -> RunsResponse:
        cache_key = (status, workflow_name, limit)
        now = _time.perf_counter()
        cached = runtime_state.runs_cache.get(cache_key)
        if cached is not None:
            expiry, response = cached
            if expiry > now:
                _perf_log.debug("/api/runs: CACHE HIT (%.1fms)", (now - (expiry - _RUNS_CACHE_TTL)) * 1000)
                return response

        t0 = _time.perf_counter()

        def _ms() -> float:
            return (_time.perf_counter() - t0) * 1000

        # Lightweight fast path when Postgres is attached.
        if dash.pg is not None:
            try:
                light_rows = dash.pg.list_workflow_runs_light(
                    workflow_name=workflow_name, status=status, limit=limit
                )
                _perf_log.debug(
                    "/api/runs: pg.list_workflow_runs_light → %d rows in %.1fms",
                    len(light_rows), _ms(),
                )
            except Exception as exc:
                _perf_log.warning(
                    "/api/runs: pg.list_workflow_runs_light FAILED (%.1fms): %s",
                    _ms(), exc,
                )
                light_rows = None
            if light_rows is not None:
                run_ids = [r["run_id"] for r in light_rows]
                t1 = _time.perf_counter()
                try:
                    counts_by_run = (
                        dash.pg.step_status_counts(run_ids) if run_ids else {}
                    )
                    _perf_log.debug(
                        "/api/runs: pg.step_status_counts → %d entries in %.1fms",
                        len(counts_by_run),
                        (_time.perf_counter() - t1) * 1000,
                    )
                except Exception as exc:
                    _perf_log.warning(
                        "/api/runs: pg.step_status_counts FAILED: %s", exc,
                    )
                    counts_by_run = {}
                # Batch-check report local availability so the frontend doesn't
                # need N per-run /report-status calls when expanding a group.
                report_runs = [r["run_id"] for r in light_rows if r["has_report"]]
                report_local: dict[str, bool] = {}
                if report_runs:
                    t_rpt = _time.perf_counter()
                    try:
                        report_aids = dash.pg.report_artifact_ids_for_runs(report_runs)
                        if report_aids:
                            manifests = dash.pg.batch_load_artifacts(list(report_aids.values()))
                            for rid, aid in report_aids.items():
                                m = manifests.get(aid)
                                if m is None:
                                    report_local[rid] = False
                                    continue
                                try:
                                    resolve_report_root(m)
                                    report_local[rid] = True
                                except ReportUnavailable:
                                    report_local[rid] = False
                    except Exception:
                        pass
                    _perf_log.debug(
                        "/api/runs: batch report check → %d/%d local in %.1fms",
                        sum(1 for v in report_local.values() if v),
                        len(report_runs),
                        (_time.perf_counter() - t_rpt) * 1000,
                    )
                summaries = [
                    _summary_from_light(
                        row,
                        counts_by_run.get(row["run_id"], {}),
                        report_local=report_local.get(row["run_id"]),
                    )
                    for row in light_rows
                ]
                # Merge local-only runs. Use a fast path: build summaries
                # without calling list_workflow_steps (which is O(steps) disk
                # reads per run). Instead, infer step count from the
                # workflow_payload steps list and set a rough step_counts
                # with total only. The user can see the full breakdown by
                # clicking into the run.
                if dash.local is not None:
                    t2 = _time.perf_counter()
                    seen = {r["run_id"] for r in light_rows}
                    local_summaries = _parallel_local_summaries(
                        dash.local,
                        seen=seen,
                        workflow_name=workflow_name,
                        status=status,
                    )
                    local_extra = len(local_summaries)
                    if local_extra > 0:
                        summaries.extend(local_summaries)
                        summaries.sort(key=lambda s: (s.started_at, s.run_id), reverse=True)
                        if limit is not None:
                            summaries = summaries[:limit]
                    _perf_log.debug(
                        "/api/runs: local merge → %d extra in %.1fms",
                        local_extra,
                        (_time.perf_counter() - t2) * 1000,
                    )
                _perf_log.info(
                    "/api/runs: FAST PATH total %.1fms (%d summaries)",
                    _ms(), len(summaries),
                )
                result = RunsResponse(runs=summaries)
                runtime_state.runs_cache[cache_key] = (
                    _time.perf_counter() + _RUNS_CACHE_TTL,
                    result,
                )
                # Pre-warm run bundle cache for the most recent runs so they're
                # instant when the user clicks. Fire-and-forget in background.
                _prewarm_run_bundles(runtime_state, dash, [s.run_id for s in summaries[:10]])
                return result

        # Fallback: no pg pool. Use the cached composite catalog.
        _perf_log.info("/api/runs: FALLBACK PATH (dash.pg is %s)", "None" if dash.pg is None else "set but light_rows=None")
        records = dash.composite.list_workflow_runs(
            workflow_name=workflow_name, status=status, limit=limit
        )
        _perf_log.debug(
            "/api/runs: composite.list_workflow_runs → %d records in %.1fms",
            len(records), _ms(),
        )
        if not records:
            return RunsResponse(runs=[])
        summaries = []
        for i, record in enumerate(records):
            t_step = _time.perf_counter()
            steps = dash.composite.list_workflow_steps(record.run_id)
            summaries.append(build_run_summary(record, steps))
            dt_step = (_time.perf_counter() - t_step) * 1000
            if dt_step > 500:
                _perf_log.warning(
                    "/api/runs: step %d/%d (%s) list_workflow_steps took %.1fms",
                    i + 1, len(records), record.run_id, dt_step,
                )
        _perf_log.info(
            "/api/runs: FALLBACK total %.1fms (%d summaries)",
            _ms(), len(summaries),
        )
        result = RunsResponse(runs=summaries)
        runtime_state.runs_cache[cache_key] = (_time.perf_counter() + _RUNS_CACHE_TTL, result)
        return result

    @app.get("/api/runs/{run_id}/report-status", response_model=RunReportStatus)
    def report_status(run_id: str) -> RunReportStatus:
        """Lightweight report-availability check — does NOT load workflow_payload.

        Only reads step records + the report artifact manifest. Used by
        ReportStatusCell on the /runs index to show ✓/↓ without pulling
        multi-MB payloads per run.
        """
        read_catalog = _dashboard_read_catalog(dash)
        step_records = read_catalog.list_workflow_steps(run_id)
        report_step = next(
            (s for s in step_records if s.artifact_kind == "report"),
            None,
        )
        if report_step is None:
            # Fallback: check for a step whose spec kind is "report" even if
            # it hasn't completed (no artifact_kind yet). We can only do this
            # from the step record's runner name or the workflow payload — but
            # we don't load the payload here. Use the has_report from RunSummary
            # instead; the caller already knows has_report=true.
            return RunReportStatus(
                has_report_step=True,
                step_name=next(
                    (s.step_name for s in step_records if "report" in s.runner.lower()),
                    None,
                ),
                artifact_id=None,
                local_available=False,
                reason="No completed report artifact recorded.",
            )
        artifact_id = report_step.artifact_id
        if not artifact_id:
            return RunReportStatus(
                has_report_step=True,
                step_name=report_step.step_name,
                artifact_id=None,
                local_available=False,
                reason="Report step has no artifact yet.",
            )
        manifest = read_catalog.load_artifact(artifact_id)
        if manifest is None:
            return RunReportStatus(
                has_report_step=True,
                step_name=report_step.step_name,
                artifact_id=artifact_id,
                local_available=False,
                reason=f"Manifest unavailable for {artifact_id}.",
            )
        try:
            resolve_report_root(manifest)
        except ReportUnavailable as exc:
            return RunReportStatus(
                has_report_step=True,
                step_name=report_step.step_name,
                artifact_id=artifact_id,
                local_available=False,
                reason=str(exc),
            )
        return RunReportStatus(
            has_report_step=True,
            step_name=report_step.step_name,
            artifact_id=artifact_id,
            local_available=True,
            reason=None,
        )

    @app.get("/api/runs/{run_id}", response_model=RunDetail)
    def get_run(run_id: str) -> RunDetail:
        bundle = _load_run_bundle(runtime_state, dash, run_id)
        return bundle.run_detail.model_copy(
            update={"report": _build_run_report_status(bundle.read_catalog, bundle.run_detail)}
        )

    @app.post("/api/runs/{run_id}/report", response_model=ReportGenerationResponse)
    def generate_report(run_id: str, step_name: str | None = Query(default=None)) -> ReportGenerationResponse:
        bundle = _load_run_bundle(runtime_state, dash, run_id)
        try:
            workflow = WorkflowSpec.from_dict(dict(bundle.run.workflow_payload))
            report_step = _resolve_report_step(workflow, step_name=step_name)
            report_spec = _build_report_spec_from_run(
                run=bundle.run,
                report_step=report_step,
                workflow_catalog=bundle.read_catalog,
                local_cache_root=None,
            )
            if not report_spec.output_dir:
                raise HTTPException(
                    status_code=400,
                    detail=f"Report step {report_step.name!r} has no output_dir and cannot be materialized locally.",
                )
            runner = LocalRunner(
                artifacts=_report_artifact_store_for_run(
                    run=bundle.run,
                    report_step=report_step,
                    workflow_catalog=bundle.read_catalog,
                    fallback_root=dash.local_root / "_generated_report_artifacts",
                    local_cache_root=None,
                ),
                catalog=dash.local,
            )
            context = _workflow_step_context(
                run=bundle.run,
                workflow=workflow,
                report_step=report_step,
                existing_record=_step_record_by_name(bundle.step_records).get(report_step.name),
            )
            artifact = runner.run(report_spec, workflow_context=context)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _mirror_generated_report_to_local_catalog(
            dash=dash,
            bundle=bundle,
            report_step=report_step,
            artifact_id=artifact.id,
            artifact_kind=artifact.manifest().artifact_kind,
            created_at=artifact.manifest().created_at,
        )
        _invalidate_dashboard_cache(dash)
        runtime_state.runs_cache.clear()
        with runtime_state.run_bundle_lock:
            runtime_state.run_bundle_cache.pop(run_id, None)
        return ReportGenerationResponse(
            run_id=run_id,
            step_name=report_step.name,
            artifact_id=artifact.id,
            report=build_report_detail(artifact.manifest()),
        )

    @app.get("/api/runs/{run_id}/steps/{step_name}", response_model=StepDetail)
    def get_step(run_id: str, step_name: str) -> StepDetail:
        bundle = _load_run_bundle(runtime_state, dash, run_id)
        manifest_by_step = _resolve_manifests_for_steps(dash, run_id, bundle.step_records)
        report_artifact_by_step = _report_artifact_ids_by_step(bundle.run_detail)

        try:
            return build_step_detail_from_run_detail(
                run=bundle.run,
                run_detail=bundle.run_detail,
                target_step=step_name,
                artifact_manifest=manifest_by_step.get(step_name),
                report_artifact_id=report_artifact_by_step.get(step_name),
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/steps-detail", response_model=StepDetailList)
    def bulk_step_detail(run_id: str) -> StepDetailList:
        """Return step detail for every step in the run in a single
        round-trip's worth of DB work (three queries total against Postgres,
        independent of step count).
        """
        bundle = _load_run_bundle(runtime_state, dash, run_id)
        manifest_by_step = _resolve_manifests_for_steps(dash, run_id, bundle.step_records)
        report_artifact_by_step = _report_artifact_ids_by_step(bundle.run_detail)
        details: list[StepDetail] = []
        for step in bundle.run_detail.steps:
            try:
                details.append(
                    build_step_detail_from_run_detail(
                        run=bundle.run,
                        run_detail=bundle.run_detail,
                        target_step=step.step_name,
                        artifact_manifest=manifest_by_step.get(step.step_name),
                        report_artifact_id=report_artifact_by_step.get(step.step_name),
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
        clear_tokenizer_cache()
        runtime_state.runs_cache.clear()
        with runtime_state.run_bundle_lock:
            runtime_state.run_bundle_cache.clear()
        result["resolved_datasets"] = "cleared"
        result["tokenizers"] = "cleared"
        result["runs_response_cache"] = "cleared"
        result["run_bundle_cache"] = "cleared"
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
        bundle = _load_run_bundle(runtime_state, dash, run_id)
        if not any(step.step_name == step_name for step in bundle.run_detail.steps):
            raise HTTPException(status_code=404, detail=f"Unknown step: {step_name}")
        artifact = _resolve_manifests_for_steps(dash, run_id, bundle.step_records).get(step_name)
        # Find a report artifact that might have a copied result for this step.
        # _report_artifact_ids_by_step walks resolved_depends_on, which can be
        # incomplete when WorkflowSpec.from_dict fails for remote-run specs.
        # Fall back to any report artifact from the run — the report runner
        # copies ALL input step results, not just direct deps.
        report_aid = _report_artifact_ids_by_step(bundle.run_detail).get(step_name)
        if report_aid is None:
            report_aid = _any_report_artifact_id(bundle.run_detail)
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

    resolved_static_dir = Path(static_dir) if static_dir is not None else workspace_config.dashboard_static_dir()
    if resolved_static_dir is not None:
        _mount_static(app, Path(resolved_static_dir))

    return app


def _parallel_local_summaries(
    local_catalog: Any,
    *,
    seen: set[str],
    workflow_name: str | None,
    status: str | None,
) -> list[Any]:
    """Read lightweight local run summaries, falling back to run-file scans."""
    light_lister = getattr(local_catalog, "list_workflow_runs_light", None)
    if callable(light_lister):
        try:
            rows = light_lister(
                workflow_name=workflow_name,
                status=status,
                limit=None,
            )
        except Exception:
            rows = None
        if rows is not None:
            return [
                _summary_from_local_light(row)
                for row in rows
                if row.get("run_id") not in seen
            ]

    # Compatibility fallback for catalogs that haven't implemented a
    # lightweight listing surface.
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pipelines_v2.workflow.records import WorkflowRunRecord

    runs_root = Path(str(local_catalog.root)) / "workflow_runs"
    if not runs_root.is_dir():
        return []

    paths = list(runs_root.glob("*.json"))
    if not paths:
        return []

    def _read_one(path: Path) -> Any:
        try:
            run_id = path.stem
            if run_id in seen:
                return None
            with path.open("r", encoding="utf-8") as f:
                payload = _json.load(f)
            rec = WorkflowRunRecord.from_dict(payload)
            if rec.run_id in seen:
                return None
            if workflow_name is not None and rec.workflow_name != workflow_name:
                return None
            if status is not None and rec.status != status:
                return None
            return _fast_local_summary(rec)
        except Exception:
            return None

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        futures = {pool.submit(_read_one, p): p for p in paths}
        for future in as_completed(futures):
            summary = future.result()
            if summary is not None:
                results.append(summary)
    return results


def _summary_from_local_light(row: dict[str, Any]):
    counts = {
        str(status_name): int(count)
        for status_name, count in dict(row.get("step_counts") or {}).items()
    }
    step_total = int(row.get("step_total") or 0)
    counted = sum(counts.values())
    if step_total > counted:
        counts["pending"] = counts.get("pending", 0) + (step_total - counted)
    return _summary_from_light(
        row,
        counts,
        report_local=row.get("report_local"),
    )


def _fast_local_summary(run: Any) -> Any:
    """Build a RunSummary from a local-catalog WorkflowRunRecord without
    reading step records from disk. Uses the workflow_payload to infer
    step count + has_report. The status breakdown (completed/failed/etc)
    is left at zero — the user can click through for details."""
    from pipelines_v2.dashboard.models import RunSummary, StepCounts
    from pipelines_v2.dashboard.normalize import _has_report_step

    payload = run.workflow_payload if isinstance(run.workflow_payload, dict) else {}
    steps_raw = payload.get("steps", ())
    total = len(steps_raw) if isinstance(steps_raw, (list, tuple)) else 0
    # Infer rough counts from the run's top-level status.
    completed = total if run.status == "completed" else 0
    failed = total if run.status == "failed" else 0
    running = total if run.status == "running" else 0
    return RunSummary(
        run_id=run.run_id,
        workflow_name=run.workflow_name,
        workflow_hash=run.workflow_hash,
        workflow_spec_hash=run.workflow_spec_hash,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        parent_run_id=run.parent_run_id,
        error=run.error,
        step_counts=StepCounts(
            total=total,
            completed=completed,
            failed=failed,
            running=running,
        ),
        has_report=_has_report_step(payload),
        report_local=None,  # unknown without checking artifact manifests
    )


def _summary_from_light(
    row: dict[str, Any],
    status_counts: dict[str, int],
    *,
    report_local: bool | None = None,
):
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
        report_local=report_local,
    )

_RUN_BUNDLE_TTL = 300.0  # 5 min for completed; 15s for in-flight


def _load_run_bundle(
    runtime_state: _DashboardAppState,
    dash: DashboardCatalog,
    run_id: str,
) -> _RunBundle:
    """Load or cache a run bundle. Collapses concurrent requests for the same
    run_id so only one thread builds the bundle; the rest wait on an Event.
    Eliminates the thundering-herd pattern where 5+ overview sub-requests all
    hit list_workflow_steps simultaneously."""
    import logging as _log

    from fastapi import HTTPException

    _p = _log.getLogger("pipelines_v2.dashboard.perf")
    now = time_mod.perf_counter()

    # Fast path: check cache without locking.
    cached = runtime_state.run_bundle_cache.get(run_id)
    if cached is not None:
        expiry, bundle = cached
        if expiry > now:
            return bundle

    # Slow path: serialize cache-fill per run_id.
    build_event: threading.Event | None = None
    builder = False
    with runtime_state.run_bundle_lock:
        # Re-check — another thread may have filled it while we waited.
        cached = runtime_state.run_bundle_cache.get(run_id)
        if cached is not None:
            expiry, bundle = cached
            if expiry > now:
                return bundle
        # Check if another thread is already building this run_id.
        build_event = runtime_state.run_bundle_building.get(run_id)
        if build_event is None:
            build_event = threading.Event()
            runtime_state.run_bundle_building[run_id] = build_event
            builder = True

    if not builder and build_event is not None:
        # Another thread is building — wait for it (up to 30s).
        build_event.wait(timeout=30.0)
        cached = runtime_state.run_bundle_cache.get(run_id)
        if cached is not None:
            expiry, bundle = cached
            if expiry > time_mod.perf_counter():
                return bundle
        # If still not cached, fall through and build ourselves.
    try:
        # Build the bundle. Try local FileCatalog first — a disk read is ~50ms
        # vs ~13s from Neon for runs with large embedded datasets. Only fall
        # back to Postgres when the run doesn't exist locally.
        read_catalog = _dashboard_read_catalog(dash)
        t0 = time_mod.perf_counter()
        run = None
        # 1. Local file catalog — instant for local runs, ~1ms miss for pg-only.
        if dash.local is not None:
            run = dash.local.load_workflow_run(run_id)
        # 2. Pooled Postgres with stripped payload (no dataset.examples).
        if run is None and dash.pg is not None:
            try:
                run = dash.pg.load_workflow_run_light(run_id)
            except Exception:
                run = None
        # 3. Full composite catalog as last resort.
        if run is None:
            run = read_catalog.load_workflow_run(run_id)
        t1 = time_mod.perf_counter()
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        step_records = read_catalog.list_workflow_steps(run_id)
        t2 = time_mod.perf_counter()
        detail = build_run_detail(run, step_records)
        t3 = time_mod.perf_counter()

        _p.info(
            "load_run_bundle(%s): load_run=%.0fms list_steps=%.0fms build_detail=%.0fms total=%.0fms",
            run_id,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            (t3 - t0) * 1000,
        )
        bundle = _RunBundle(
            run=run,
            step_records=step_records,
            run_detail=detail,
            read_catalog=read_catalog,
        )
        terminal = run.status in ("completed", "failed", "cancelled")
        ttl = _RUN_BUNDLE_TTL if terminal else 15.0
        runtime_state.run_bundle_cache[run_id] = (time_mod.perf_counter() + ttl, bundle)
        return bundle
    finally:
        if builder and build_event is not None:
            with runtime_state.run_bundle_lock:
                active = runtime_state.run_bundle_building.get(run_id)
                if active is build_event:
                    runtime_state.run_bundle_building.pop(run_id, None)
                    build_event.set()


def _dashboard_read_catalog(dash: DashboardCatalog) -> _DashboardReadCatalog:
    sources: list[_CatalogSource] = []
    seen: set[int] = set()
    for name, candidate in (
        ("local", dash.local),
        ("pg", dash.pg),
        ("composite", dash.composite),
        ("raw", dash.raw),
    ):
        if candidate is None:
            continue
        marker = id(candidate)
        if marker in seen:
            continue
        seen.add(marker)
        sources.append(_CatalogSource(name=name, catalog=candidate))
    return _DashboardReadCatalog(tuple(sources))


def _resolve_report_step(workflow: WorkflowSpec, *, step_name: str | None) -> WorkflowStep:
    report_steps = [step for step in workflow.ordered_steps() if isinstance(step.spec, ReportSpec)]
    if not report_steps:
        raise RuntimeError("Workflow run does not contain any report steps")
    if step_name is None:
        if len(report_steps) == 1:
            return report_steps[0]
        names = [step.name for step in report_steps]
        raise RuntimeError(f"Workflow run contains multiple report steps; choose one with step_name: {names}")
    for step in report_steps:
        if step.name == step_name:
            return step
    raise RuntimeError(f"Workflow run does not contain report step {step_name!r}")


def _build_run_report_status(catalog: Any, run_detail: RunDetail) -> RunReportStatus:
    report_step = next((step for step in run_detail.steps if step.spec_kind == "report"), None)
    if report_step is None:
        return RunReportStatus()
    artifact_id = report_step.artifact_id
    if not artifact_id:
        return RunReportStatus(
            has_report_step=True,
            step_name=report_step.step_name,
            artifact_id=None,
            local_available=False,
            reason="No report artifact is currently recorded for this run.",
        )
    manifest = catalog.load_artifact(artifact_id)
    if manifest is None:
        return RunReportStatus(
            has_report_step=True,
            step_name=report_step.step_name,
            artifact_id=artifact_id,
            local_available=False,
            reason=f"Report artifact manifest is unavailable for {artifact_id}.",
        )
    try:
        resolve_report_root(manifest)
    except ReportUnavailable as exc:
        return RunReportStatus(
            has_report_step=True,
            step_name=report_step.step_name,
            artifact_id=artifact_id,
            local_available=False,
            reason=str(exc),
        )
    return RunReportStatus(
        has_report_step=True,
        step_name=report_step.step_name,
        artifact_id=artifact_id,
        local_available=True,
        reason=None,
    )


def _workflow_step_context(
    *,
    run: Any,
    workflow: WorkflowSpec,
    report_step: WorkflowStep,
    existing_record: WorkflowStepRecord | None,
) -> WorkflowStepContext:
    ordered = workflow.ordered_steps()
    step_index = next((index for index, step in enumerate(ordered) if step.name == report_step.name), 0)
    return WorkflowStepContext(
        run_id=run.run_id,
        workflow_name=run.workflow_name,
        workflow_hash=run.workflow_hash,
        workflow_spec_hash=run.workflow_spec_hash,
        step_name=report_step.name,
        step_index=existing_record.step_index if existing_record is not None else step_index,
        runner=existing_record.runner if existing_record is not None else report_step.runner,
        step_semantic_hash=existing_record.step_semantic_hash if existing_record is not None else report_step.semantic_hash(),
        step_spec_hash=existing_record.step_spec_hash if existing_record is not None else report_step.spec_hash(),
    )


def _step_record_by_name(step_records: list[WorkflowStepRecord]) -> dict[str, WorkflowStepRecord]:
    return {record.step_name: record for record in step_records}


def _mirror_generated_report_to_local_catalog(
    *,
    dash: DashboardCatalog,
    bundle: _RunBundle,
    report_step: WorkflowStep,
    artifact_id: str,
    artifact_kind: str,
    created_at: str,
) -> None:
    dash.local.record_workflow_run(bundle.run)
    records_by_name = _step_record_by_name(bundle.step_records)
    target = records_by_name.get(report_step.name)
    for record in bundle.step_records:
        if record.step_name != report_step.name:
            dash.local.record_workflow_step(record)
            continue
        dash.local.record_workflow_step(
            dataclasses.replace(
                record,
                status="completed",
                artifact_id=artifact_id,
                artifact_kind=artifact_kind,
                started_at=created_at,
                finished_at=created_at,
                runtime_app_id=None,
                reused_from_run_id=None,
                reused_from_artifact_id=None,
            )
        )
    if target is None:
        context = _workflow_step_context(
            run=bundle.run,
            workflow=WorkflowSpec.from_dict(dict(bundle.run.workflow_payload)),
            report_step=report_step,
            existing_record=None,
        )
        dash.local.record_workflow_step(
            WorkflowStepRecord(
                run_id=context.run_id,
                workflow_hash=context.workflow_hash,
                workflow_step_key=context.workflow_step_key,
                step_name=context.step_name,
                step_index=context.step_index,
                runner=context.runner,
                status="completed",
                step_semantic_hash=context.step_semantic_hash,
                step_spec_hash=context.step_spec_hash,
                artifact_id=artifact_id,
                artifact_kind=artifact_kind,
                started_at=created_at,
                finished_at=created_at,
            )
        )


def _invalidate_dashboard_cache(dash: DashboardCatalog) -> None:
    invalidator = getattr(dash.composite, "invalidate_all", None)
    if callable(invalidator):
        invalidator()


def _resolve_manifests_for_steps(
    dash: DashboardCatalog,
    run_id: str,
    step_records: list[Any],
) -> dict[str, Any]:
    """Resolve step_name -> manifest using batched pg paths when available."""
    manifest_by_step: dict[str, Any] = {}
    if dash.pg is not None:
        try:
            artifact_ids = [record.artifact_id for record in step_records if record.artifact_id is not None]
            if artifact_ids:
                by_aid = dash.pg.batch_load_artifacts(artifact_ids)
                for record in step_records:
                    if record.artifact_id and record.artifact_id in by_aid:
                        manifest_by_step[record.step_name] = by_aid[record.artifact_id]
            missing = [record for record in step_records if record.step_name not in manifest_by_step]
            if missing:
                by_key = dash.pg.find_artifacts_for_run(run_id)
                for record in missing:
                    manifest = by_key.get(record.workflow_step_key)
                    if manifest is not None:
                        manifest_by_step[record.step_name] = manifest
        except Exception:
            manifest_by_step = {}

    for record in step_records:
        if record.step_name in manifest_by_step:
            continue
        manifest = dash.composite.find_artifact_for_workflow_step(
            run_id=run_id,
            workflow_step_key=record.workflow_step_key,
        )
        if manifest is None and record.artifact_id is not None:
            manifest = dash.composite.load_artifact(record.artifact_id)
        if manifest is not None:
            manifest_by_step[record.step_name] = manifest
    return manifest_by_step


def _prewarm_run_bundles(
    runtime_state: _DashboardAppState,
    dash: DashboardCatalog,
    run_ids: list[str],
) -> None:
    """Background pre-warm: fetch and cache the top N run bundles so they're
    instant when the user clicks from the /runs index. Skips runs that are
    already cached. Uses a daemon thread to avoid blocking the response."""
    uncached = [
        rid for rid in run_ids
        if rid not in runtime_state.run_bundle_cache
        or runtime_state.run_bundle_cache[rid][0] < time_mod.perf_counter()
    ]
    if not uncached:
        return

    import logging as _log
    _p = _log.getLogger("pipelines_v2.dashboard.perf")

    def _warm():
        for rid in uncached:
            try:
                _load_run_bundle(runtime_state, dash, rid)
                _p.debug("prewarm: cached %s", rid)
            except Exception:
                pass  # non-critical

    t = threading.Thread(target=_warm, daemon=True, name="prewarm-bundles")
    t.start()


def _any_report_artifact_id(run_detail: RunDetail) -> str | None:
    """Return the artifact_id of any completed report step in the run."""
    for step in run_detail.steps:
        if step.artifact_kind == "report" and step.artifact_id:
            return step.artifact_id
    return None


def _report_artifact_ids_by_step(run_detail: RunDetail) -> dict[str, str]:
    """Map each step to a downstream report artifact, if one exists.

    The first report step in workflow order wins for shared ancestors. This
    keeps the mapping stable and avoids repeatedly traversing the DAG.
    """
    by_name = {step.step_name: step for step in run_detail.steps}
    result: dict[str, str] = {}
    for step in run_detail.steps:
        if step.spec_kind != "report" or not step.artifact_id:
            continue
        stack = [step.step_name]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            result.setdefault(current, step.artifact_id)
            current_step = by_name.get(current)
            if current_step is None:
                continue
            stack.extend(current_step.resolved_depends_on)
    return result


def _mount_static(app: "FastAPI", static_dir: Path) -> None:
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from fastapi.staticfiles import StaticFiles

    class _SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):  # type: ignore[override]
            method = scope.get("method", "GET")
            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404:
                    raise
                if method not in {"GET", "HEAD"}:
                    raise
                return await super().get_response("index.html", scope)
            if response.status_code == 404 and method in {"GET", "HEAD"}:
                return await super().get_response("index.html", scope)
            return response

    if not static_dir.exists():
        return
    app.mount("/", _SPAStaticFiles(directory=static_dir, html=True), name="dashboard")

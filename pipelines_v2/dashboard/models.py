"""Pydantic response models for the dashboard API.

These are the wire schema. The dashboard is strictly read-only, so models here
should not leak mutable catalog internals; they normalize the persisted record
shapes into stable frontend-facing fields.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Run summaries (used by /api/runs and as the header on /api/runs/{run_id})
# ---------------------------------------------------------------------------


class StepCounts(BaseModel):
    total: int
    completed: int = 0
    failed: int = 0
    running: int = 0
    reused: int = 0
    pending: int = 0
    other: int = 0


class RunSummary(BaseModel):
    run_id: str
    workflow_name: str | None
    workflow_hash: str
    workflow_spec_hash: str
    status: str
    started_at: str
    finished_at: str | None = None
    parent_run_id: str | None = None
    error: str | None = None
    step_counts: StepCounts
    has_report: bool = False
    report_local: bool | None = None  # True/False/None(unknown)


class RunsResponse(BaseModel):
    runs: list[RunSummary]


# ---------------------------------------------------------------------------
# Run detail
# ---------------------------------------------------------------------------


class StepSummary(BaseModel):
    """Ordered step row used in the run detail inspector and graph nodes."""

    step_name: str
    step_index: int
    runner: str
    status: str
    spec_kind: str | None = None
    family: str | None = None
    artifact_id: str | None = None
    artifact_kind: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    runtime_app_id: str | None = None
    reused_from_run_id: str | None = None
    reused_from_artifact_id: str | None = None
    step_semantic_hash: str | None = None
    step_spec_hash: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    resolved_depends_on: list[str] = Field(default_factory=list)


class DagNode(BaseModel):
    id: str
    step_name: str
    runner: str
    spec_kind: str | None = None
    family: str | None = None
    status: str
    artifact_id: str | None = None
    artifact_kind: str | None = None
    reused: bool = False
    runtime_app_id: str | None = None


class DagEdge(BaseModel):
    source: str
    target: str
    kind: Literal["declared", "resolved"] = "resolved"


class RunDetail(BaseModel):
    run: RunSummary
    workflow_payload: dict[str, Any]
    nodes: list[DagNode]
    edges: list[DagEdge]
    steps: list[StepSummary]
    report: RunReportStatus | None = None


# ---------------------------------------------------------------------------
# Phase C — step detail
# ---------------------------------------------------------------------------


class ResolvedDep(BaseModel):
    step_name: str
    runner: str | None = None
    artifact_id: str | None = None
    artifact_kind: str | None = None
    status: str | None = None


class ArtifactSummary(BaseModel):
    artifact_id: str
    artifact_kind: str
    schema_version: int
    created_at: str
    operation_spec_hash: str
    operation_semantic_hash: str
    storage_refs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_artifact_refs: list[str] = Field(default_factory=list)
    runner: dict[str, Any] = Field(default_factory=dict)
    engine: dict[str, Any] = Field(default_factory=dict)
    example_coverage: dict[str, Any] = Field(default_factory=dict)


class ResultTableSummary(BaseModel):
    name: str
    rows: int
    columns: list[str] = Field(default_factory=list)


class ResultSummary(BaseModel):
    headline: dict[str, Any] | None = None
    tables: list[ResultTableSummary] = Field(default_factory=list)
    raw_available: bool = False


class ResultPreviewTable(BaseModel):
    name: str
    rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    total_rows: int | None = None
    truncated: bool = False


class ResultPreview(BaseModel):
    available: bool
    reason: str | None = None
    path: str | None = None
    bytes: int | None = None
    payload: dict[str, Any] | None = None
    headline: dict[str, Any] | None = None
    tables: list[ResultPreviewTable] = Field(default_factory=list)
    truncated: bool = False
    truncation_reason: str | None = None


class SpecSummaryItem(BaseModel):
    label: str
    value: str


class StepDetailList(BaseModel):
    """Bulk step-detail response — one entry per step, ordered by step_index."""

    step_details: list["StepDetail"] = Field(default_factory=list)


class StepDetail(BaseModel):
    step: StepSummary
    spec: dict[str, Any]
    spec_summary: list[SpecSummaryItem] = Field(default_factory=list)
    upstream: list[ResolvedDep] = Field(default_factory=list)
    downstream: list[ResolvedDep] = Field(default_factory=list)
    artifact: ArtifactSummary | None = None
    result_summary: ResultSummary | None = None
    report_artifact_id: str | None = None
    # Which optional inspector panels have any data to show. Driven from the
    # workflow spec and artifact presence so the UI can hide empty tabs.
    has_prompt: bool = False
    has_dataset: bool = False
    has_labels: bool = False
    has_artifact: bool = False
    has_results: bool = False


# ---------------------------------------------------------------------------
# Phase D — dataset + label previews
# ---------------------------------------------------------------------------


class DatasetPreviewRow(BaseModel):
    example_key: str
    case_key: str | None = None
    prompt_preview: str
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetSourceInfo(BaseModel):
    kind: str
    env_var: str | None = None
    table: str | None = None
    query: str | None = None
    limit: int | None = None
    identity: dict[str, Any] | None = None
    name: str | None = None
    dataset_id: str | None = None
    deferred: bool = False
    total_examples: int | None = None
    prompt_column: str | None = None
    example_key_column: str | None = None
    label_columns: list[str] = Field(default_factory=list)
    case_columns: list[str] = Field(default_factory=list)
    metadata_columns: list[str] = Field(default_factory=list)
    selection_keys: list[str] | None = None
    construction: list[dict[str, str]] = Field(default_factory=list)


class DatasetOption(BaseModel):
    step_name: str
    label: str


class DatasetPreview(BaseModel):
    available: bool
    reason: str | None = None
    source: DatasetSourceInfo | None = None
    rows: list[DatasetPreviewRow] = Field(default_factory=list)
    total_rows: int | None = None
    sample_size: int | None = None
    dataset_options: list[DatasetOption] | None = None
    resolved_from_step: str | None = None


class LabelDistributionBucket(BaseModel):
    value: str
    count: int
    fraction: float


class NumericSummary(BaseModel):
    min: float
    max: float
    mean: float
    stddev: float


class LabelDistribution(BaseModel):
    label_name: str
    unique_values: int
    buckets: list[LabelDistributionBucket] = Field(default_factory=list)
    numeric_summary: NumericSummary | None = None
    source_step: str | None = None


class LabelSample(BaseModel):
    example_key: str
    prompt_preview: str
    labels: dict[str, Any] = Field(default_factory=dict)


class LabelPreview(BaseModel):
    available: bool
    reason: str | None = None
    labels: list[LabelDistribution] = Field(default_factory=list)
    samples: list[LabelSample] = Field(default_factory=list)
    resolved_from_step: str | None = None


# ---------------------------------------------------------------------------
# Phase E — prompt preview
# ---------------------------------------------------------------------------


class PromptSection(BaseModel):
    id: str
    label: str
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None
    selected: bool = False
    pooling: str | None = None


class PromptSelection(BaseModel):
    section_label: str | None = None
    token_start: int | None = None
    token_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    pooling: str | None = None
    exact_tokens: bool = False
    sentence: str


class PromptExample(BaseModel):
    example_key: str
    text: str
    sections: list[PromptSection] = Field(default_factory=list)
    selection: PromptSelection | None = None
    tokenizer: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PromptPreview(BaseModel):
    available: bool
    reason: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None
    examples: list[PromptExample] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase F — reports
# ---------------------------------------------------------------------------


class RunReportStatus(BaseModel):
    has_report_step: bool = False
    step_name: str | None = None
    artifact_id: str | None = None
    local_available: bool = False
    reason: str | None = None


class ReportFigure(BaseModel):
    figure_id: str
    path: str
    step_name: str | None = None
    result_kind: str | None = None
    chart_kind: str | None = None
    title: str | None = None
    caption: str | None = None
    primary: bool = False


class ReportTableSummary(BaseModel):
    slug: str
    step_name: str | None = None
    result_kind: str | None = None
    rows: int = 0
    columns: list[str] = Field(default_factory=list)
    path: str


class ReportResult(BaseModel):
    name: str
    path: str
    step_name: str | None = None
    bytes: int


class ReportDetail(BaseModel):
    artifact_id: str
    artifact_kind: str
    run_id: str | None = None
    report: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    headline: dict[str, Any] | None = None
    figures: list[ReportFigure] = Field(default_factory=list)
    tables: list[ReportTableSummary] = Field(default_factory=list)
    results: list[ReportResult] = Field(default_factory=list)
    unsupported_inputs: list[dict[str, Any]] = Field(default_factory=list)


class ReportGenerationResponse(BaseModel):
    run_id: str
    step_name: str
    artifact_id: str
    report: ReportDetail


# Resolve the forward reference for StepDetailList now that StepDetail is
# defined above.
RunDetail.model_rebuild()
StepDetailList.model_rebuild()

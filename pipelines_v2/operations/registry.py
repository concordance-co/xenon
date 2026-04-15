"""Operation spec registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.capture import CaptureSpec
from pipelines_v2.operations.derive import LabelFieldsSpec, LabelMapSpec, PairDeltaSpec, TransformSpec
from pipelines_v2.operations.interventions import ActivationPatchSpec
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec
from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.operations.representation import BasisSpec, DirectionSpec, GeometrySpec

OperationLoader = Callable[[dict[str, Any]], OperationSpec]

_OPERATION_LOADERS: dict[str, OperationLoader] = {
    CaptureSpec.kind: CaptureSpec.from_dict,
    ProbeSpec.kind: ProbeSpec.from_dict,
    TransferProbeSpec.kind: TransferProbeSpec.from_dict,
    TextBaselineSpec.kind: TextBaselineSpec.from_dict,
    ResidualizedProbeSpec.kind: ResidualizedProbeSpec.from_dict,
    DirectionSpec.kind: DirectionSpec.from_dict,
    BasisSpec.kind: BasisSpec.from_dict,
    GeometrySpec.kind: GeometrySpec.from_dict,
    PairDeltaSpec.kind: PairDeltaSpec.from_dict,
    LabelMapSpec.kind: LabelMapSpec.from_dict,
    LabelFieldsSpec.kind: LabelFieldsSpec.from_dict,
    TransformSpec.kind: TransformSpec.from_dict,
    ActivationPatchSpec.kind: ActivationPatchSpec.from_dict,
    ReportSpec.kind: ReportSpec.from_dict,
}


def operation_spec_from_dict(payload: dict[str, Any]) -> OperationSpec:
    kind = str(payload.get("kind") or "").strip()
    if not kind:
        raise ValueError("Operation spec payload is missing 'kind'")
    try:
        loader = _OPERATION_LOADERS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown operation spec kind: {kind!r}") from exc
    return loader(payload)

"""Operation spec registry and deserialization helpers."""

from __future__ import annotations

from typing import Any, Callable

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.specs import (
    ActivationPatchSpec,
    BasisSpec,
    CaptureSpec,
    DirectionSpec,
    LabelFieldsSpec,
    LabelMapSpec,
    PairDeltaSpec,
    ProbeSpec,
    ReportSpec,
    TransformSpec,
)

OperationLoader = Callable[[dict[str, Any]], OperationSpec]

_OPERATION_LOADERS: dict[str, OperationLoader] = {
    CaptureSpec.kind: CaptureSpec.from_dict,
    ProbeSpec.kind: ProbeSpec.from_dict,
    DirectionSpec.kind: DirectionSpec.from_dict,
    BasisSpec.kind: BasisSpec.from_dict,
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

"""Projection calibration specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict


@dataclass(frozen=True, slots=True)
class ProjectionCalibrationSpec(OperationSpec):
    """Fit calibration metadata on top of raw projection outputs."""

    projections: Any = None
    fit_on: Any = None
    strategy: str = "quantile_bands"
    bands: Sequence[str] = field(default_factory=tuple)
    summary_name: str | None = None
    orientation: Mapping[str, str] = field(default_factory=dict)

    kind: ClassVar[str] = "projection_calibration"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.projections, self.fit_on)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectionCalibrationSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            projections=spec_value_from_dict(payload.get("projections")),
            fit_on=spec_value_from_dict(payload.get("fit_on")),
            strategy=str(payload.get("strategy", "quantile_bands")),
            bands=tuple(str(item) for item in payload.get("bands", ())),
            summary_name=str(payload["summary_name"]) if payload.get("summary_name") is not None else None,
            orientation={str(key): str(value) for key, value in dict(payload.get("orientation", {})).items()},
        )


__all__ = ["ProjectionCalibrationSpec"]

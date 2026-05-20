"""External coordinate import specs.

Coordinate import is the generic escape hatch for vector artifacts produced
outside ``pipelines_v2``. Domain-specific loaders, such as Assistant Axis HF
loading, should normalize their data into the same coordinate payload shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.common._shared import analysis_runtime_spec


@dataclass(frozen=True, slots=True)
class CoordinateImportSpec(OperationSpec):
    """Import an external vector artifact into a canonical coordinate payload.

    Supported formats currently include PyTorch tensors or axis dictionaries,
    ``.npy`` arrays, and JSON vectors. Rank-1 inputs become a single-layer
    coordinate; rank-2 inputs are interpreted as ``[layer, hidden]`` unless
    ``select_layer`` narrows them to one layer.
    """

    path: str = ""
    format: str = "torch_tensor_or_axis_dict"
    name: str | None = None
    select_layer: int | None = None
    normalize: str = "l2"
    metadata: dict[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "coordinate_import"

    def runtime_spec(self) -> Any | None:
        extra_packages = ("torch",) if str(self.format).strip().lower() == "torch_tensor_or_axis_dict" else ()
        return analysis_runtime_spec(extra_pip_packages=extra_packages)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CoordinateImportSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            path=str(payload["path"]),
            format=str(payload.get("format", "torch_tensor_or_axis_dict")),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            select_layer=int(payload["select_layer"]) if payload.get("select_layer") is not None else None,
            normalize=str(payload.get("normalize", "l2")),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


__all__ = ["CoordinateImportSpec"]

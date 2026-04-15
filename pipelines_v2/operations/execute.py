"""Compatibility import surface for artifact-bound execution helpers."""

from __future__ import annotations

from pipelines_v2.operations.execution import OperationExecutionResult, execute_artifact_operation, feature_matrices

_feature_matrices = feature_matrices

__all__ = [
    "OperationExecutionResult",
    "_feature_matrices",
    "execute_artifact_operation",
]

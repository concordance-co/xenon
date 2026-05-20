"""Compatibility façade for intervention runtime helpers."""

from ._runtime_comparison import (
    aggregate_patch_comparison_rows,
    evaluate_patch_comparison_row,
)
from ._runtime_planning import patched_generation_plan_errors
from ._runtime_resolution import (
    partition_cases_by_activation_bank,
    resolve_generation_examples,
    resolve_patched_generation_cases,
    resolve_patched_generation_targets,
)
from ._runtime_rows import row_example_payload, rows_example_coverage
from ._runtime_sources import (
    load_activation_bank_source,
    load_centroid_source,
    load_direction_source,
    load_path_mask_source,
    load_subspace_source,
)

__all__ = [
    "aggregate_patch_comparison_rows",
    "evaluate_patch_comparison_row",
    "load_activation_bank_source",
    "load_centroid_source",
    "load_direction_source",
    "load_path_mask_source",
    "load_subspace_source",
    "partition_cases_by_activation_bank",
    "patched_generation_plan_errors",
    "resolve_generation_examples",
    "resolve_patched_generation_cases",
    "resolve_patched_generation_targets",
    "row_example_payload",
    "rows_example_coverage",
]

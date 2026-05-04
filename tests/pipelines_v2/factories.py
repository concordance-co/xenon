from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pipelines_v2.storage.artifacts import FeatureRef


@dataclass(frozen=True, slots=True)
class FeatureArtifactStub:
    """Minimal artifact surface for operation tests that consume FeatureRef."""

    features: Mapping[str, Mapping[str, Any]]
    artifact_id: str = "feature_artifact_stub"

    @property
    def id(self) -> str:
        return self.artifact_id

    def load_feature(self, name: str) -> dict[str, Any]:
        if name not in self.features:
            raise KeyError(f"Feature stub has no feature {name!r}")
        return copy.deepcopy(dict(self.features[name]))


def feature_ref_from_payload(
    payload: Mapping[str, Any],
    *,
    name: str = "resid_known",
) -> FeatureRef:
    return FeatureRef(
        artifact=FeatureArtifactStub(features={name: dict(payload)}),
        name=name,
    )


def residual_feature_payload(
    *,
    rows: Mapping[str, Sequence[Sequence[float]]],
    layer: int = 0,
    token_sections: Mapping[str, Sequence[int]] | None = None,
    section_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    layer_payload = {
        str(example_key): {
            "tokens": list(range(len(values))),
            "values": [list(row) for row in values],
            "prompt_hash": f"hash_{example_key}",
            **({"token_sections": {str(k): [int(v) for v in vals] for k, vals in token_sections.items()}} if token_sections else {}),
            **({"section_records": [dict(record) for record in section_records]} if section_records else {}),
        }
        for example_key, values in rows.items()
    }
    return {
        "kind": "residual",
        "site": "resid_post",
        "storage": {"dtype": "float32", "format": "inline"},
        "layers": {str(layer): layer_payload},
    }

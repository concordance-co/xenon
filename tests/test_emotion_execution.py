from __future__ import annotations

from typing import Any

from pipelines_v2.mechinterp.emotions.execution import _VECTOR_SPACE_CACHE, run_emotion_direction
from pipelines_v2.mechinterp.emotions.specs import EMOTION_VECTOR_SPACE_KIND, EmotionDirectionSpec


class _CountingVectorSpaceArtifact:
    id = "vector-space-test"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def result(self) -> dict[str, Any]:
        self.calls += 1
        return self.payload


def test_emotion_direction_caches_vector_space_artifact_by_id() -> None:
    _VECTOR_SPACE_CACHE.clear()
    payload = {
        "kind": EMOTION_VECTOR_SPACE_KIND,
        "vector_space_kind": "test",
        "layers": {
            "56": {
                "concepts": {
                    "joy": {
                        "vector": [1.0, 0.0],
                        "raw_vector": [2.0, 0.0],
                        "norm": 2.0,
                        "count": 3,
                    }
                }
            }
        },
    }
    source = _CountingVectorSpaceArtifact(payload)

    try:
        run_emotion_direction(EmotionDirectionSpec(vector_space=source, concept="joy", layers=(56,)))
        run_emotion_direction(EmotionDirectionSpec(vector_space=source, concept="joy", layers=(56,)))
    finally:
        _VECTOR_SPACE_CACHE.clear()

    assert source.calls == 1

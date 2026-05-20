"""Shared helpers for MoE routing capture payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from pipelines_v2.operations.capture import RoutingRecord


@dataclass(frozen=True, slots=True)
class TopKResult:
    indices: npt.NDArray[np.int64]
    values: npt.NDArray[np.float32]


def routing_record_payload(
    record: RoutingRecord,
    logits: Any,
    *,
    topk_from_gate_k: int,
    fallback_top_k: int,
    observed_topk_ids: Any | None = None,
    observed_topk_weights: Any | None = None,
) -> dict[str, Any]:
    """Build one routing-record payload from gate logits and optional observed routing."""

    gate_logits = np.asarray(logits)
    if record.kind == "gate_logits":
        return {"gate_logits": gate_logits.astype(routing_float_dtype(record.params.get("dtype", "float16")))}
    if record.kind == "gate_probs":
        return {"gate_probs": routing_softmax(gate_logits).astype(routing_float_dtype(record.params.get("dtype", "float16")))}
    if record.kind == "routing_decisions":
        if observed_topk_ids is not None:
            expert_ids = np.asarray(observed_topk_ids, dtype=np.int64)
            expert_weights = (
                np.asarray(observed_topk_weights, dtype=np.float32)
                if observed_topk_weights is not None
                else np.ones(expert_ids.shape[0], dtype=np.float32)
            )
            return {
                "routing_decisions": {
                    "source": "observed",
                    "expert_ids": expert_ids,
                    "weights": expert_weights,
                }
            }
        if record.params.get("required", True):
            raise RuntimeError("Observed routing decisions are not exposed by the current engine")
        return {"routing_decisions": {"source": "not_observed", "expert_ids": [], "weights": []}}
    if record.kind == "topk_from_gate":
        top = routing_topk(gate_logits, int(record.params["k"]))
        payload: dict[str, Any] = {
            "source": "derived_from_gate_logits",
            "expert_ids": top.indices,
        }
        if record.params.get("include_weights", True):
            payload["weights"] = normalize_routing_weights(top.values)
        return {"topk_from_gate": payload}
    if record.kind == "expert_load":
        source = str(record.params.get("source") or "topk_from_gate")
        if source == "routing_decisions" and observed_topk_ids is not None:
            topk = np.asarray(observed_topk_ids, dtype=np.int64)
        else:
            k = topk_from_gate_k if source == "topk_from_gate" else fallback_top_k
            topk = routing_topk(gate_logits, k).indices
        return {"expert_load": {"source": source, "counts": {str(int(idx)): 1 for idx in topk}}}
    raise ValueError(f"Unsupported routing record: {record.kind}")


def requested_topk_from_gate_k(requested: Sequence[RoutingRecord], *, fallback: int) -> int:
    """Return the requested top-k-from-gate width, or a fallback."""

    for record in requested:
        if record.kind == "topk_from_gate":
            return int(record.params["k"])
    return int(fallback)


def routing_topk(values: Any, k: int) -> TopKResult:
    """Return descending top-k indices and values from a vector."""

    array = np.asarray(values)
    if k <= 0:
        return TopKResult(indices=np.asarray([], dtype=np.int64), values=np.asarray([], dtype=np.float32))
    bounded_k = min(int(k), int(array.shape[-1]))
    indices = np.argsort(array)[::-1][:bounded_k].astype(np.int64)
    return TopKResult(indices=indices, values=array[indices].astype(np.float32))


def routing_softmax(values: Any) -> npt.NDArray[np.float32]:
    """Compute a float32 softmax over a routing-logit vector."""

    array = np.asarray(values, dtype=np.float32)
    shifted = array - np.max(array)
    exps = np.exp(shifted)
    return (exps / np.sum(exps)).astype(np.float32)


def normalize_routing_weights(values: Any) -> npt.NDArray[np.float32]:
    """Normalize top-k routing weights using the historical non-negative shift."""

    array = np.asarray(values)
    if array.size == 0:
        return np.asarray([], dtype=np.float32)
    shifted = array.astype(np.float32) - np.min(array) + np.float32(1e-6)
    return (shifted / np.sum(shifted)).astype(np.float32)


def routing_float_dtype(name: str) -> Any:
    """Map serialized routing dtypes onto NumPy dtypes."""

    normalized = str(name).lower()
    if normalized == "float16":
        return np.float16
    if normalized in {"float32", "bfloat16"}:
        return np.float32
    raise ValueError(f"Unsupported routing dtype: {name}")


__all__ = [
    "TopKResult",
    "normalize_routing_weights",
    "requested_topk_from_gate_k",
    "routing_float_dtype",
    "routing_record_payload",
    "routing_softmax",
    "routing_topk",
]

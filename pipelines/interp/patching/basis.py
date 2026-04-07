from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class ActivationPatchBasisLayer:
    layer: int
    state_key: str
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    named_components: dict[str, int]

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.scale = np.asarray(self.scale, dtype=np.float32)
        self.components = np.asarray(self.components, dtype=np.float32)
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1D, got {self.mean.shape}")
        if self.scale.ndim != 1:
            raise ValueError(f"scale must be 1D, got {self.scale.shape}")
        if self.components.ndim != 2:
            raise ValueError(f"components must be 2D, got {self.components.shape}")
        if self.mean.shape != self.scale.shape:
            raise ValueError(
                f"mean/scale shape mismatch: {self.mean.shape} vs {self.scale.shape}"
            )
        if self.components.shape[1] != self.mean.shape[0]:
            raise ValueError(
                "components hidden dim does not match mean/scale hidden dim: "
                f"{self.components.shape[1]} vs {self.mean.shape[0]}"
            )

    @property
    def hidden_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def num_components(self) -> int:
        return int(self.components.shape[0])

    def to_payload(self) -> dict[str, Any]:
        return {
            "layer": int(self.layer),
            "state_key": str(self.state_key),
            "mean": self.mean.copy(),
            "scale": self.scale.copy(),
            "components": self.components.copy(),
            "named_components": {
                str(name): int(index) for name, index in self.named_components.items()
            },
        }


@dataclass(slots=True)
class ActivationPatchBasis:
    state_key: str
    layers: dict[int, ActivationPatchBasisLayer]
    source_basis_npz: str
    source_results_json: str | None

    def to_payload(self) -> dict[int, dict[str, Any]]:
        return {
            int(layer): basis_layer.to_payload()
            for layer, basis_layer in sorted(self.layers.items())
        }


def _load_results_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _named_components_for_layer(
    *,
    results_json: dict[str, Any] | None,
    state_key: str,
    layer: int,
    max_components: int,
) -> dict[str, int]:
    if results_json is None:
        return {}
    targets = results_json.get("targets")
    if not isinstance(targets, dict):
        return {}
    named: dict[str, int] = {}
    for target_name, target_payload in targets.items():
        if not isinstance(target_payload, dict):
            continue
        if str(target_payload.get("state_key")) != state_key:
            continue
        if int(target_payload.get("layer", -1)) != int(layer):
            continue
        pc_index = int(target_payload.get("pc_index", 0))
        if 1 <= pc_index <= max_components:
            named[str(target_name)] = pc_index - 1
    return named


def load_activation_patch_basis(
    *,
    basis_npz_path: Path,
    state_key: str = "activation_mean",
    layers: tuple[int, ...] = (4, 35),
    components_per_layer: int = 4,
    results_json_path: Path | None = None,
) -> ActivationPatchBasis:
    basis = np.load(basis_npz_path)
    results_json = _load_results_json(results_json_path)
    layer_map: dict[int, ActivationPatchBasisLayer] = {}

    for layer in layers:
        prefix = f"{state_key}_layer_{int(layer)}"
        mean_key = f"{prefix}__mean"
        scale_key = f"{prefix}__scale"
        comp_key = f"{prefix}__components"
        if mean_key not in basis or scale_key not in basis or comp_key not in basis:
            raise KeyError(
                f"Missing basis payload for {prefix}. "
                f"Expected keys {mean_key}, {scale_key}, {comp_key}."
            )

        components = np.asarray(basis[comp_key], dtype=np.float32)
        take = min(max(1, int(components_per_layer)), int(components.shape[0]))
        layer_map[int(layer)] = ActivationPatchBasisLayer(
            layer=int(layer),
            state_key=state_key,
            mean=np.asarray(basis[mean_key], dtype=np.float32),
            scale=np.asarray(basis[scale_key], dtype=np.float32),
            components=components[:take].copy(),
            named_components=_named_components_for_layer(
                results_json=results_json,
                state_key=state_key,
                layer=int(layer),
                max_components=take,
            ),
        )

    return ActivationPatchBasis(
        state_key=state_key,
        layers=layer_map,
        source_basis_npz=str(basis_npz_path),
        source_results_json=None if results_json_path is None else str(results_json_path),
    )

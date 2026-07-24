from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from pipelines_v2.engine.vllm import moe_hooks


@pytest.fixture(autouse=True)
def _reset_router_patch_state() -> Any:
    original_state = (
        moe_hooks._PATCHED,
        moe_hooks._ORIGINAL_SELECT_EXPERTS,
        moe_hooks._PATCH_TARGET,
    )
    moe_hooks._PATCHED = False
    moe_hooks._ORIGINAL_SELECT_EXPERTS = None
    moe_hooks._PATCH_TARGET = None
    yield
    if moe_hooks._PATCH_TARGET is not None and moe_hooks._ORIGINAL_SELECT_EXPERTS:
        moe_hooks._PATCH_TARGET.select_experts = moe_hooks._ORIGINAL_SELECT_EXPERTS
    (
        moe_hooks._PATCHED,
        moe_hooks._ORIGINAL_SELECT_EXPERTS,
        moe_hooks._PATCH_TARGET,
    ) = original_state


def _layer(*, layer_id: int = 3, monolithic: bool = False) -> Any:
    return SimpleNamespace(
        layer_id=layer_id,
        global_num_experts=4,
        top_k=2,
        gate=SimpleNamespace(weight=torch.ones((4, 3))),
        router=SimpleNamespace(),
        is_monolithic=monolithic,
    )


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    **attributes: Any,
) -> ModuleType:
    parts = module_name.split(".")
    for index in range(1, len(parts) + 1):
        name = ".".join(parts[:index])
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            monkeypatch.setitem(sys.modules, name, module)
        if index > 1:
            parent = sys.modules[".".join(parts[: index - 1])]
            monkeypatch.setattr(parent, parts[index - 1], module, raising=False)
    leaf = sys.modules[module_name]
    for name, value in attributes.items():
        monkeypatch.setattr(leaf, name, value, raising=False)
    return leaf


def test_router_capture_preserves_logits_ids_and_weights() -> None:
    layer = _layer()
    session = moe_hooks._RouterCaptureSession(max_tokens=4, layers={3: layer})
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    topk_ids = torch.tensor([[3, 1], [0, 2]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]])

    session.capture(
        layer_id=3,
        router_logits=logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    captured = session.collect()[3]

    torch.testing.assert_close(captured["logits"], logits)
    torch.testing.assert_close(captured["topk_ids"], topk_ids.to(torch.int32))
    torch.testing.assert_close(captured["topk_weights"], topk_weights)


def test_router_patch_forwards_v025_arguments_and_captures_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeBaseRouter:
        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
            topk_indices_dtype: torch.dtype | None = None,
            *,
            input_ids: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(
                {
                    "hidden_states": hidden_states,
                    "router_logits": router_logits,
                    "topk_indices_dtype": topk_indices_dtype,
                    "input_ids": input_ids,
                }
            )
            return (
                torch.tensor([[0.7, 0.3]]),
                torch.tensor([[2, 1]], dtype=topk_indices_dtype),
            )

    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.router.base_router",
        BaseRouter=FakeBaseRouter,
    )
    moe_hooks._patch_base_router()

    capture_calls: list[dict[str, Any]] = []
    router = FakeBaseRouter()
    router._xenon_router_capture_session = SimpleNamespace(
        capture=lambda **kwargs: capture_calls.append(kwargs)
    )
    router._xenon_router_layer_id = 7
    hidden_states = torch.ones((1, 3))
    router_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    input_ids = torch.tensor([42])

    weights, ids = router.select_experts(
        hidden_states,
        router_logits,
        torch.int32,
        input_ids=input_ids,
    )

    assert calls == [
        {
            "hidden_states": hidden_states,
            "router_logits": router_logits,
            "topk_indices_dtype": torch.int32,
            "input_ids": input_ids,
        }
    ]
    assert capture_calls == [
        {
            "layer_id": 7,
            "router_logits": router_logits,
            "topk_ids": ids,
            "topk_weights": weights,
        }
    ]


def test_router_patch_keeps_v019_two_argument_call_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBaseRouter:
        def select_experts(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.tensor([[0.8, 0.2]]), torch.tensor([[1, 0]])

    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.router.base_router",
        BaseRouter=FakeBaseRouter,
    )
    moe_hooks._patch_base_router()

    capture_calls: list[dict[str, Any]] = []
    router = FakeBaseRouter()
    router._xenon_router_capture_session = SimpleNamespace(
        capture=lambda **kwargs: capture_calls.append(kwargs)
    )
    router._xenon_router_layer_id = 2
    hidden_states = torch.ones((1, 3))
    router_logits = torch.ones((1, 4))

    weights, ids = router.select_experts(hidden_states, router_logits)

    assert capture_calls == [
        {
            "layer_id": 2,
            "router_logits": router_logits,
            "topk_ids": ids,
            "topk_weights": weights,
        }
    ]


def test_v025_fused_moe_factory_discovers_concrete_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fused_moe_factory() -> None:
        return None

    class FakeMoERunner:
        def __init__(self) -> None:
            self.layer_id = 5

    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.layer",
        FusedMoE=fake_fused_moe_factory,
    )
    _install_fake_module(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.runner.moe_runner",
        MoERunner=FakeMoERunner,
    )
    runner = FakeMoERunner()
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(), runner])

    assert moe_hooks.find_moe_blocks(model) == {5: runner}


def test_v025_logical_expert_count_wins_over_redundant_expert_count() -> None:
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(num_logical_experts=8),
        router=SimpleNamespace(global_num_experts=10),
    )

    assert moe_hooks._num_experts_for_layer(layer) == 8


def test_init_router_capture_rejects_monolithic_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _layer(monolithic=True)
    monkeypatch.setattr(moe_hooks, "_iter_fused_moe_layers", lambda model: {3: layer})

    with pytest.raises(RuntimeError, match="monolithic MoE kernels.*layers \\[3\\]"):
        moe_hooks.init_router_capture(SimpleNamespace(), max_tokens=8)

    assert moe_hooks._PATCHED is False


def test_collect_fails_if_an_initialized_layer_was_not_observed() -> None:
    session = moe_hooks._RouterCaptureSession(max_tokens=4, layers={3: _layer()})

    with pytest.raises(RuntimeError, match="did not observe router outputs.*\\[3\\]"):
        session.collect()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logits", torch.ones((2, 5)), "logits shape changed"),
        ("ids", torch.ones((2, 1), dtype=torch.int64), "expert-id shape changed"),
        ("weights", torch.ones((2, 1)), "expert-weight shape changed"),
    ],
)
def test_capture_rejects_incompatible_router_shapes(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    session = moe_hooks._RouterCaptureSession(max_tokens=4, layers={3: _layer()})
    values = {
        "router_logits": torch.ones((2, 4)),
        "topk_ids": torch.ones((2, 2), dtype=torch.int64),
        "topk_weights": torch.ones((2, 2)),
    }
    values[
        {
            "logits": "router_logits",
            "ids": "topk_ids",
            "weights": "topk_weights",
        }[field]
    ] = value

    with pytest.raises(RuntimeError, match=message):
        session.capture(layer_id=3, **values)

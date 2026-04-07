from __future__ import annotations

import json
from dataclasses import replace

import pyarrow.parquet as pq
import transformers

from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_battery import SyntheticMarketBehaviorPlanItem
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_matrix_runner import (
    SyntheticMarketBehaviorMatrixConfig,
    run_synthetic_market_behavior_matrix,
)
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import SyntheticMarketBehaviorConfig


class _FakeTokenizer:
    def decode(self, token_ids):
        return f"tok{int(token_ids[0])}" if token_ids else ""


class _FakeBasis:
    def to_payload(self):
        return {4: {"components": [[1.0], [0.0], [0.0], [0.0]], "mean": [0.0]}}


class _FakePatchSpec:
    def __init__(self, payload):
        self._payload = payload

    def to_payload(self):
        return dict(self._payload)


def test_matrix_runner_executes_multiple_cells_in_one_run(monkeypatch, tmp_path):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_matrix_runner as matrix_runner

    create_llm_calls: list[bool] = []
    prepare_calls: list[str] = []
    batch_patch_specs: list[list[dict | None]] = []

    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "matrix",
        model_id="fake-model",
        pair_metric="vol_1h_max",
        pair_mode="denoise",
        patch_mode="none",
        batch_size=4,
        max_tokens=8,
        tool_schema_mode="trading_v1",
        tool_choice="required",
        target_layers=(4,),
        components_per_layer=4,
        enforce_eager=False,
        enable_chunked_prefill=True,
    )
    baseline_cell = SyntheticMarketBehaviorPlanItem(
        run_name="baseline_denoise",
        sweep_kind="baseline",
        sweep_value="denoise",
        description="baseline",
        config=base_config,
    )
    project_cell = SyntheticMarketBehaviorPlanItem(
        run_name="project_denoise",
        sweep_kind="targeted",
        sweep_value="lam_1",
        description="project",
        config=base_config.__class__(
            **{
                **{
                    field: getattr(base_config, field)
                    for field in base_config.__dataclass_fields__
                },
                "patch_mode": "project_out",
                "strength": 1.0,
                "component_indices_by_layer": {4: (0, 1, 2, 3)},
            }
        ),
    )

    monkeypatch.setattr(
        matrix_runner,
        "resolve_tool_schema_mode",
        lambda mode: [{"type": "function", "function": {"name": "buy_token"}}],
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()))

    def fake_prepare_behavior_rows(*, config, **_kwargs):
        prepare_calls.append(str(config.pair_mode))
        return (
            [
                {
                    "row": {
                        "log_id": 1 if config.pair_mode == "denoise" else 2,
                        "phase_name": config.phase_name,
                        "example_id": f"ex-{config.pair_mode}",
                        "family": "fam",
                        "family_variant": "var",
                        "context_variant": config.context_variant,
                        "roster_key": "r00",
                        "pair_metric_name": config.pair_metric,
                        "pair_metric_value": 1.0,
                        "pair_mode": config.pair_mode,
                        "pair_id": 0,
                        "source_log_id": 10,
                        "source_example_id": "src",
                        "source_family_variant": "srcvar",
                        "source_roster_key": "r00",
                    },
                    "messages": [{"role": "user", "content": "prompt"}],
                    "market_span": (0, 2),
                    "source_log_id": 10,
                    "source_row_messages": [{"role": "user", "content": "source"}],
                    "source_market_span": (0, 2),
                }
            ],
            0,
        )

    monkeypatch.setattr(matrix_runner, "_prepare_behavior_rows", fake_prepare_behavior_rows)
    monkeypatch.setattr(matrix_runner, "_build_generation_config", lambda cfg: cfg)
    monkeypatch.setattr(
        matrix_runner,
        "_create_llm",
        lambda cfg: create_llm_calls.append(bool(cfg.enforce_eager)) or object(),
    )
    monkeypatch.setattr(matrix_runner, "load_phase17_activation_patch_basis", lambda **_kwargs: _FakeBasis())
    monkeypatch.setattr(matrix_runner, "_init_activation_patching_on_model", lambda _llm: True)
    monkeypatch.setattr(matrix_runner, "_register_activation_patch_basis_on_model", lambda _llm, _payload: None)
    monkeypatch.setattr(
        matrix_runner,
        "_build_patch_spec",
        lambda *, config, market_span, basis_payload, donor_mean_by_layer=None: _FakePatchSpec(
            {
                "patch_mode": config.patch_mode,
                "market_span": list(market_span),
                "strength": float(config.strength),
            }
        ),
    )
    monkeypatch.setattr(matrix_runner, "_destroy_llm", lambda _llm: None)

    def fake_run_generation_batch(*, requests, **_kwargs):
        batch_patch_specs.append([request.get("patch_spec") for request in requests])
        outputs = []
        for idx, request in enumerate(requests):
            patch_spec = request.get("patch_spec")
            token_id = 100 + idx + (50 if patch_spec else 0)
            outputs.append(
                {
                    "input_ids": [1, 2, 3],
                    "generated_token_ids": [token_id],
                    "generated_text": "patched" if patch_spec else "baseline",
                    "finish_reason": "stop",
                    "request_id": f"req-{idx}",
                    "patch_stats": {} if patch_spec is None else {"4": {"status": "applied"}},
                    "all_patch_stats": {},
                }
            )
        return outputs

    monkeypatch.setattr(matrix_runner, "_run_generation_batch", fake_run_generation_batch)

    result = run_synthetic_market_behavior_matrix(
        SyntheticMarketBehaviorMatrixConfig(
            base_config=base_config,
            cells=(baseline_cell, project_cell),
            output_dir=tmp_path / "matrix",
        )
    )

    assert result["cell_count"] == 2
    assert result["counts_by_sweep_kind"] == {"baseline": 1, "targeted": 1}
    assert create_llm_calls == [False]
    assert prepare_calls == ["denoise"]
    assert any(specs == [None, {"patch_mode": "project_out", "market_span": [0, 2], "strength": 1.0}] for specs in batch_patch_specs)

    metadata_rows = pq.read_table(tmp_path / "matrix" / "metadata.parquet").to_pylist()
    assert {row["matrix_cell_id"] for row in metadata_rows} == {"baseline_denoise", "project_denoise"}
    assert {row["patch_mode"] for row in metadata_rows} == {"none", "project_out"}
    assert {row["runtime_enforce_eager"] for row in metadata_rows} == {False}

    plan = json.loads((tmp_path / "matrix" / "plan.json").read_text())
    assert plan["count"] == 2


def test_matrix_runner_uses_eager_fallback_for_random_control(monkeypatch, tmp_path):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_matrix_runner as matrix_runner

    create_llm_calls: list[bool] = []
    batch_patch_specs: list[list[dict | None]] = []

    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "matrix",
        model_id="fake-model",
        pair_metric="vol_1h_max",
        pair_mode="denoise",
        patch_mode="none",
        batch_size=8,
        max_tokens=8,
        tool_schema_mode="trading_v1",
        tool_choice="required",
        target_layers=(4,),
        components_per_layer=4,
        enforce_eager=False,
        enable_chunked_prefill=True,
    )
    baseline_cell = SyntheticMarketBehaviorPlanItem(
        run_name="baseline_denoise",
        sweep_kind="baseline",
        sweep_value="denoise",
        description="baseline",
        config=base_config,
    )
    project_cell = SyntheticMarketBehaviorPlanItem(
        run_name="project_denoise",
        sweep_kind="targeted",
        sweep_value="lam_1",
        description="project",
        config=replace(
            base_config,
            patch_mode="project_out",
            strength=1.0,
            component_indices_by_layer={4: (0, 1, 2, 3)},
        ),
    )
    control_cell = SyntheticMarketBehaviorPlanItem(
        run_name="random_control_denoise",
        sweep_kind="random_control",
        sweep_value="seed_11",
        description="control",
        config=replace(
            base_config,
            patch_mode="random_control",
            random_seed=11,
            strength=1.0,
            component_indices_by_layer={4: (0, 1, 2, 3)},
        ),
    )

    monkeypatch.setattr(
        matrix_runner,
        "resolve_tool_schema_mode",
        lambda mode: [{"type": "function", "function": {"name": "buy_token"}}],
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()))
    monkeypatch.setattr(
        matrix_runner,
        "_prepare_behavior_rows",
        lambda **_kwargs: (
            [
                {
                    "row": {
                        "log_id": 1,
                        "phase_name": base_config.phase_name,
                        "example_id": "ex",
                        "family": "fam",
                        "family_variant": "var",
                        "context_variant": base_config.context_variant,
                        "roster_key": "r00",
                        "pair_metric_name": base_config.pair_metric,
                        "pair_metric_value": 1.0,
                        "pair_mode": base_config.pair_mode,
                        "pair_id": 0,
                    },
                    "messages": [{"role": "user", "content": "prompt"}],
                    "market_span": (0, 2),
                    "source_log_id": None,
                    "source_row_messages": None,
                    "source_market_span": None,
                }
            ],
            0,
        ),
    )
    monkeypatch.setattr(matrix_runner, "_build_generation_config", lambda cfg: cfg)
    monkeypatch.setattr(
        matrix_runner,
        "_create_llm",
        lambda cfg: create_llm_calls.append(bool(cfg.enforce_eager)) or object(),
    )
    monkeypatch.setattr(matrix_runner, "load_phase17_activation_patch_basis", lambda **_kwargs: _FakeBasis())
    monkeypatch.setattr(matrix_runner, "_init_activation_patching_on_model", lambda _llm: True)
    monkeypatch.setattr(matrix_runner, "_register_activation_patch_basis_on_model", lambda _llm, _payload: None)
    monkeypatch.setattr(
        matrix_runner,
        "_build_patch_spec",
        lambda *, config, market_span, basis_payload, donor_mean_by_layer=None: _FakePatchSpec(
            {
                "patch_mode": config.patch_mode,
                "market_span": list(market_span),
                "strength": float(config.strength),
            }
        ),
    )
    monkeypatch.setattr(matrix_runner, "_destroy_llm", lambda _llm: None)

    def fake_run_generation_batch(*, requests, **_kwargs):
        batch_patch_specs.append([request.get("patch_spec") for request in requests])
        outputs = []
        for idx, request in enumerate(requests):
            patch_spec = request.get("patch_spec")
            outputs.append(
                {
                    "input_ids": [1, 2, 3],
                    "generated_token_ids": [200 + idx],
                    "generated_text": str(patch_spec.get("patch_mode") if patch_spec else "baseline"),
                    "finish_reason": "stop",
                    "request_id": f"req-{idx}",
                    "patch_stats": {} if patch_spec is None else {"4": {"status": "applied"}},
                    "all_patch_stats": {},
                }
            )
        return outputs

    monkeypatch.setattr(matrix_runner, "_run_generation_batch", fake_run_generation_batch)

    result = run_synthetic_market_behavior_matrix(
        SyntheticMarketBehaviorMatrixConfig(
            base_config=base_config,
            cells=(baseline_cell, project_cell, control_cell),
            output_dir=tmp_path / "matrix",
        )
    )

    assert result["runtime_groups"] == {"compiled": 2, "eager": 1}
    assert create_llm_calls == [False, True]
    assert any(specs == [None, {"patch_mode": "project_out", "market_span": [0, 2], "strength": 1.0}] for specs in batch_patch_specs)
    assert any(specs == [{"patch_mode": "random_control", "market_span": [0, 2], "strength": 1.0}] for specs in batch_patch_specs)

    metadata_rows = pq.read_table(tmp_path / "matrix" / "metadata.parquet").to_pylist()
    eager_by_cell = {row["matrix_cell_id"]: row["runtime_enforce_eager"] for row in metadata_rows}
    assert eager_by_cell == {
        "baseline_denoise": False,
        "project_denoise": False,
        "random_control_denoise": True,
    }

from __future__ import annotations

import json
import numpy as np
import pyarrow.parquet as pq
import transformers

from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
    SyntheticMarketBehaviorConfig,
    _prepare_behavior_rows,
    _build_generation_config,
    _run_generation_batch,
    prepare_synthetic_market_behavior_donors,
    run_synthetic_market_behavior,
)


def test_build_generation_config_uses_request_scoped_worker_for_patched_runs():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=4,
        )
    )

    assert cfg.max_num_seqs == 4
    assert cfg.request_scoped_patching is True
    assert cfg.worker_cls == "pipelines.interp.patching.request_worker.MarketPatchGPUWorker"
    assert cfg.enable_chunked_prefill is False
    assert cfg.enable_prefix_caching is False
    assert cfg.async_scheduling is False
    assert cfg.max_num_batched_tokens == 40960


def test_build_generation_config_disables_async_scheduling_for_batched_baseline_runs():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="",
            batch_size=4,
        )
    )

    assert cfg.max_num_seqs == 4
    assert cfg.request_scoped_patching is True
    assert cfg.worker_cls == "pipelines.interp.patching.request_worker.MarketPatchGPUWorker"
    assert cfg.async_scheduling is False


def test_build_generation_config_keeps_single_request_baseline_on_default_worker():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="",
            batch_size=1,
        )
    )

    assert cfg.max_num_seqs == 1
    assert cfg.request_scoped_patching is False
    assert cfg.worker_cls == ""
    assert cfg.async_scheduling is None


def test_build_generation_config_can_enable_chunked_prefill():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=8,
            enable_chunked_prefill=True,
        )
    )

    assert cfg.max_num_seqs == 8
    assert cfg.enable_chunked_prefill is True
    assert cfg.request_scoped_patching is True


def test_build_generation_config_can_enable_observability_flags():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=4,
            enable_logging_iteration_details=True,
            enable_mfu_metrics=True,
        )
    )

    assert cfg.enable_logging_iteration_details is True
    assert cfg.enable_mfu_metrics is True


def test_build_generation_config_can_disable_enforce_eager():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=4,
            enforce_eager=False,
        )
    )

    assert cfg.enforce_eager is False


def test_build_generation_config_uses_request_scoped_worker_for_secondary_patch_runs():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            batch_size=1,
            patch_mode="project_out",
            secondary_patch_mode="swap_components",
            secondary_target_layers=(40,),
        )
    )

    assert cfg.max_num_seqs == 1
    assert cfg.request_scoped_patching is True
    assert cfg.worker_cls == "pipelines.interp.patching.request_worker.MarketPatchGPUWorker"


def test_run_generation_batch_uses_single_prompt_path_for_single_request(monkeypatch):
    calls: list[str] = []

    def fake_generate_one_vllm(**kwargs):
        calls.append("single")
        return {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}}

    monkeypatch.setattr(
        "projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner._generate_one_vllm",
        fake_generate_one_vllm,
    )

    out = _run_generation_batch(
        llm=object(),
        tokenizer=object(),
        requests=[{"messages": [{"role": "user", "content": "x"}]}],
        config=_build_generation_config(SyntheticMarketBehaviorConfig()),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        tools=None,
        tool_choice=None,
    )

    assert calls == ["single"]
    assert len(out) == 1


def test_run_generation_batch_uses_batched_path_for_multiple_requests(monkeypatch):
    calls: list[str] = []

    def fake_generate_batch_vllm(**kwargs):
        calls.append("batch")
        return [
            {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}},
            {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}},
        ]

    monkeypatch.setattr(
        "projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner._generate_batch_vllm",
        fake_generate_batch_vllm,
    )

    out = _run_generation_batch(
        llm=object(),
        tokenizer=object(),
        requests=[
            {"messages": [{"role": "user", "content": "x"}]},
            {"messages": [{"role": "user", "content": "y"}]},
        ],
        config=_build_generation_config(SyntheticMarketBehaviorConfig(batch_size=2)),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        tools=None,
        tool_choice=None,
    )

    assert calls == ["batch"]
    assert len(out) == 2


def test_prepare_behavior_rows_filters_exact_example_ids(monkeypatch):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner as behavior_runner

    rows = [
        {
            "log_id": 1,
            "example_id": "keep-a",
            "prompt_messages_json": '[{"role":"user","content":"prompt a"}]',
        },
        {
            "log_id": 2,
            "example_id": "drop-b",
            "prompt_messages_json": '[{"role":"user","content":"prompt b"}]',
        },
        {
            "log_id": 3,
            "example_id": "keep-c",
            "prompt_messages_json": '[{"role":"user","content":"prompt c"}]',
        },
    ]

    monkeypatch.setattr(behavior_runner, "_load_examples", lambda **_kwargs: rows)
    monkeypatch.setattr(behavior_runner, "_parse_messages", lambda raw: json.loads(raw))
    monkeypatch.setattr(behavior_runner, "_extract_system_user", lambda messages: ("", messages[0]["content"]))
    monkeypatch.setattr(
        behavior_runner,
        "find_synthetic_section_boundaries",
        lambda *args, **kwargs: {"market": (0, 2)},
    )

    prepared, skipped = _prepare_behavior_rows(
        config=SyntheticMarketBehaviorConfig(
            example_id_allowlist=("keep-a", "keep-c"),
        ),
        tokenizer=_FakeTokenizer(),
        tools=None,
        chat_template_kwargs=None,
    )

    assert skipped == 0
    assert [row["row"]["example_id"] for row in prepared] == ["keep-a", "keep-c"]


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


def test_run_behavior_releases_capture_memory_before_generation(monkeypatch, tmp_path):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner as behavior_runner

    events: list[str] = []
    basis_calls: list[dict[str, object]] = []

    cfg = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        model_id="fake-model",
        pair_metric="vol_1h_max",
        pair_mode="denoise",
        patch_mode="swap_components",
        batch_size=1,
        max_tokens=8,
        tool_schema_mode="trading_v1",
        tool_choice="required",
        target_layers=(4,),
        components_per_layer=4,
        basis_state_key="market_eos",
    )

    monkeypatch.setattr(
        behavior_runner,
        "resolve_tool_schema_mode",
        lambda _mode: [{"type": "function", "function": {"name": "buy_token"}}],
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_prepare_behavior_rows",
        lambda **_kwargs: (
            [
                {
                    "row": {
                        "log_id": 1,
                        "phase_name": cfg.phase_name,
                        "example_id": "ex-1",
                        "family": "fam",
                        "family_variant": "var",
                        "context_variant": cfg.context_variant,
                        "roster_key": "r00",
                        "pair_metric_name": cfg.pair_metric,
                        "pair_metric_value": 1.0,
                        "pair_mode": cfg.pair_mode,
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
        ),
    )
    monkeypatch.setattr(behavior_runner, "_build_generation_config", lambda cfg: cfg)

    def fake_create_llm(config):
        kind = "capture" if bool(getattr(config, "capture_residual", False)) else "generate"
        events.append(f"create_{kind}")
        return {"kind": kind}

    monkeypatch.setattr(behavior_runner, "_create_llm", fake_create_llm)
    monkeypatch.setattr(
        behavior_runner,
        "_compute_batched_donor_means",
        lambda **_kwargs: {10: {4: np.asarray([0.0], dtype=np.float32)}},
    )
    monkeypatch.setattr(
        behavior_runner,
        "default_phase17_market_patch_basis",
        lambda **kwargs: basis_calls.append(dict(kwargs)) or _FakeBasis(),
    )
    monkeypatch.setattr(behavior_runner, "_init_market_patching_on_model", lambda _llm: True)
    monkeypatch.setattr(behavior_runner, "_register_market_patch_basis_on_model", lambda _llm, _payload: None)
    monkeypatch.setattr(
        behavior_runner,
        "_build_patch_spec",
        lambda *, config, market_span, basis_payload, donor_mean_by_layer=None: _FakePatchSpec(
            {
                "patch_mode": config.patch_mode,
                "market_span": list(market_span),
                "donor": donor_mean_by_layer is not None,
            }
        ),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_destroy_llm",
        lambda llm: events.append(f"destroy_{llm['kind']}"),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_cleanup_cuda_memory",
        lambda: events.append("cleanup"),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_run_generation_batch",
        lambda **_kwargs: [
            {
                "input_ids": [1, 2, 3],
                "generated_token_ids": [151],
                "generated_text": "patched",
                "finish_reason": "stop",
                "request_id": "req-0",
                "patch_stats": {"4": {"status": "applied"}},
                "all_patch_stats": {},
            }
        ],
    )

    result = run_synthetic_market_behavior(cfg)

    assert result["processed"] == 1
    assert events[:4] == ["create_capture", "destroy_capture", "cleanup", "create_generate"]
    assert events[-2:] == ["destroy_generate", "cleanup"]
    assert basis_calls == [
        {
            "basis_npz_path": cfg.basis_npz_path,
            "results_json_path": cfg.basis_results_path,
            "state_key": "market_eos",
            "layers": (4,),
            "components_per_layer": 4,
        }
    ]
    assert result["basis_state_key"] == "market_eos"
    metadata_rows = pq.read_table(tmp_path / "behavior" / "metadata.parquet").to_pylist()
    assert metadata_rows[0]["patch_mode"] == "swap_components"
    assert metadata_rows[0]["basis_state_key"] == "market_eos"


def test_run_behavior_builds_multi_spec_requests_for_path_validation(monkeypatch, tmp_path):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner as behavior_runner

    requests_seen: list[list[dict[str, object]]] = []
    donor_capture_layers: list[tuple[int, ...]] = []
    basis_calls: list[dict[str, object]] = []

    cfg = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        model_id="fake-model",
        pair_metric="vol_1h_max",
        pair_mode="denoise",
        generate_source_behavior=False,
        batch_size=1,
        patch_mode="project_out",
        target_layers=(4,),
        components_per_layer=4,
        secondary_patch_mode="swap_components",
        secondary_target_layers=(40,),
        secondary_components_per_layer=4,
        max_tokens=8,
        tool_schema_mode="trading_v1",
        tool_choice="required",
        enforce_eager=False,
    )

    monkeypatch.setattr(
        behavior_runner,
        "resolve_tool_schema_mode",
        lambda _mode: [{"type": "function", "function": {"name": "buy_token"}}],
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_prepare_behavior_rows",
        lambda **_kwargs: (
            [
                {
                    "row": {
                        "log_id": 1,
                        "phase_name": cfg.phase_name,
                        "example_id": "ex-1",
                        "family": "fam",
                        "family_variant": "var",
                        "context_variant": cfg.context_variant,
                        "roster_key": "r00",
                        "pair_metric_name": cfg.pair_metric,
                        "pair_metric_value": 1.0,
                        "pair_mode": cfg.pair_mode,
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
        ),
    )
    monkeypatch.setattr(behavior_runner, "_build_generation_config", lambda cfg: cfg)
    monkeypatch.setattr(
        behavior_runner,
        "_create_llm",
        lambda _cfg: {"kind": "capture" if bool(getattr(_cfg, "capture_residual", False)) else "generate"},
    )
    monkeypatch.setattr(
        behavior_runner,
        "_compute_batched_donor_means",
        lambda **kwargs: donor_capture_layers.append(tuple(kwargs["target_layers"])) or {
            10: {
                4: np.asarray([0.0], dtype=np.float32),
                40: np.asarray([1.0], dtype=np.float32),
            }
        },
    )
    monkeypatch.setattr(
        behavior_runner,
        "default_phase17_market_patch_basis",
        lambda **kwargs: basis_calls.append(dict(kwargs)) or _FakeBasis(),
    )
    monkeypatch.setattr(behavior_runner, "_init_market_patching_on_model", lambda _llm: True)
    monkeypatch.setattr(behavior_runner, "_register_market_patch_basis_on_model", lambda _llm, _payload: None)
    monkeypatch.setattr(
        behavior_runner,
        "_build_patch_spec",
        lambda *, config, market_span, basis_payload, donor_mean_by_layer=None: _FakePatchSpec(
            {
                "mode": config.patch_mode,
                "token_span": list(market_span),
                "target_layers": [int(layer) for layer in config.target_layers],
                "has_donor": donor_mean_by_layer is not None,
            }
        ),
    )
    monkeypatch.setattr(behavior_runner, "_destroy_llm", lambda _llm: None)
    monkeypatch.setattr(behavior_runner, "_cleanup_cuda_memory", lambda: None)

    def fake_run_generation_batch(**kwargs):
        requests_seen.append(list(kwargs["requests"]))
        return [
            {
                "input_ids": [1, 2, 3],
                "generated_token_ids": [151],
                "generated_text": "patched",
                "finish_reason": "stop",
                "request_id": "req-0",
                "patch_stats": {"4": {"status": "applied"}, "40": {"status": "applied"}},
                "all_patch_stats": {},
            }
        ]

    monkeypatch.setattr(behavior_runner, "_run_generation_batch", fake_run_generation_batch)

    result = run_synthetic_market_behavior(cfg)

    assert result["processed"] == 1
    assert donor_capture_layers == [(4, 40)]
    assert basis_calls == [
        {
            "basis_npz_path": cfg.basis_npz_path,
            "results_json_path": cfg.basis_results_path,
            "state_key": "market_mean",
            "layers": (4, 40),
            "components_per_layer": 4,
        }
    ]
    assert len(requests_seen) == 1
    request = requests_seen[0][0]
    assert request["patch_spec"] is None
    assert [spec["mode"] for spec in request["patch_specs"]] == ["project_out", "swap_components"]
    metadata_rows = pq.read_table(tmp_path / "behavior" / "metadata.parquet").to_pylist()
    assert metadata_rows[0]["patch_mode"] == "project_out+swap_components"
    assert metadata_rows[0]["primary_patch_mode"] == "project_out"
    assert metadata_rows[0]["secondary_patch_mode"] == "swap_components"
    assert metadata_rows[0]["secondary_target_layers"] == "40"


def test_prepare_donors_uses_batched_residual_capture(monkeypatch, tmp_path):
    import projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner as behavior_runner

    capture_cfgs: list[object] = []
    batch_calls: list[dict[str, object]] = []

    cfg = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "donors",
        model_id="fake-model",
        pair_metric="vol_1h_max",
        pair_mode="denoise",
        batch_size=32,
        target_layers=(4,),
        secondary_patch_mode="swap_components",
        secondary_target_layers=(40,),
        tool_schema_mode="trading_v1",
        tool_choice="required",
    )

    monkeypatch.setattr(
        behavior_runner,
        "resolve_tool_schema_mode",
        lambda _mode: [{"type": "function", "function": {"name": "buy_token"}}],
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: _FakeTokenizer()),
    )
    monkeypatch.setattr(
        behavior_runner,
        "_prepare_behavior_rows",
        lambda **_kwargs: (
            [
                {
                    "row": {"log_id": 1},
                    "messages": [{"role": "user", "content": "prompt"}],
                    "market_span": (0, 2),
                    "source_log_id": 10,
                    "source_row_messages": [{"role": "user", "content": "source-a"}],
                    "source_market_span": (0, 2),
                },
                {
                    "row": {"log_id": 2},
                    "messages": [{"role": "user", "content": "prompt"}],
                    "market_span": (0, 2),
                    "source_log_id": 11,
                    "source_row_messages": [{"role": "user", "content": "source-b"}],
                    "source_market_span": (0, 2),
                },
            ],
            0,
        ),
    )

    def fake_create_llm(config):
        capture_cfgs.append(config)
        return {"kind": "capture"}

    monkeypatch.setattr(behavior_runner, "_create_llm", fake_create_llm)
    monkeypatch.setattr(
        behavior_runner,
        "_compute_batched_donor_means",
        lambda **kwargs: batch_calls.append(
            {
                "target_layers": tuple(kwargs["target_layers"]),
                "batch_size": int(kwargs["batch_size"]),
                "num_rows": len(kwargs["prepared_rows"]),
                "max_num_seqs": int(kwargs["capture_cfg"].max_num_seqs),
            }
        )
        or {
            10: {4: np.asarray([0.0], dtype=np.float32), 40: np.asarray([1.0], dtype=np.float32)},
            11: {4: np.asarray([0.0], dtype=np.float32), 40: np.asarray([1.0], dtype=np.float32)},
        },
    )
    monkeypatch.setattr(behavior_runner, "_destroy_llm", lambda _llm: None)
    monkeypatch.setattr(behavior_runner, "_cleanup_cuda_memory", lambda: None)

    result = prepare_synthetic_market_behavior_donors(cfg)

    assert result["donor_source_count"] == 2
    assert batch_calls == [
        {
            "target_layers": (40,),
            "batch_size": 32,
            "num_rows": 2,
            "max_num_seqs": 32,
        }
    ]
    assert capture_cfgs[0].max_num_seqs == 32

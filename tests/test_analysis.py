"""Tests for pipelines.interp.analysis.

Uses synthetic safetensors and labels — no real model or Modal needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipelines.interp.analysis import (
    AnalysisConfig,
    AnalysisDataset,
    _needs_compact,
    _encode_labels,
    dispatch,
    main,
    run_expert_analysis,
    run_compact,
    run_pca,
    run_probe_sweep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 4
NUM_EXPERTS = 16
TOP_K = 4
HIDDEN_DIM = 32
SEQ_LEN = 10


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write numpy arrays as safetensors via torch conversion."""
    import torch
    from safetensors.torch import save_file

    torch_tensors = {}
    for k, v in tensors.items():
        torch_tensors[k] = torch.from_numpy(v)
    save_file(torch_tensors, str(path))


def _make_activations(
    tmp_path: Path,
    log_ids: list[int],
    num_layers: int = NUM_LAYERS,
    seq_len: int = SEQ_LEN,
    num_experts: int = NUM_EXPERTS,
    hidden_dim: int = HIDDEN_DIM,
    top_k: int = TOP_K,
    include_residual: bool = True,
    include_router: bool = True,
    seed: int = 0,
) -> Path:
    """Create synthetic safetensors files in the expected directory structure."""
    act_dir = tmp_path / "activations"
    rng = np.random.RandomState(seed)

    if include_residual:
        res_dir = act_dir / "residual_stream"
        res_dir.mkdir(parents=True, exist_ok=True)
        for lid in log_ids:
            data = rng.randn(num_layers, seq_len, hidden_dim).astype(np.float16)
            _write_safetensors(res_dir / f"{lid}.safetensors", {"residual_stream": data})

    if include_router:
        router_dir = act_dir / "router_logits"
        router_dir.mkdir(parents=True, exist_ok=True)
        for lid in log_ids:
            logits = rng.randn(num_layers, seq_len, num_experts).astype(np.float32)
            indices = rng.randint(0, num_experts, (num_layers, seq_len, top_k)).astype(np.int16)
            _write_safetensors(
                router_dir / f"{lid}.safetensors",
                {"router_logits": logits, "router_indices": indices},
            )

    # Write metadata.parquet
    meta_rows = []
    for lid in log_ids:
        meta_rows.append({
            "log_id": lid,
            "seq_len": seq_len,
            "prompt_hash": f"hash_{lid}",
            "capture_timestamp": "2025-01-01T00:00:00",
            "file_size_bytes": 1024,
            "elapsed_s": 1.0,
            "has_router": include_router,
            "num_layers_captured": num_layers,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts if include_router else 0,
        })
    meta_table = pa.Table.from_pylist(meta_rows)
    pq.write_table(meta_table, act_dir / "metadata.parquet")

    return act_dir


def _make_labels(
    tmp_path: Path,
    log_ids: list[int],
    decision_types: list[str] | None = None,
    trade_sides: list[str | None] | None = None,
    risk_prefs: list[int | None] | None = None,
    profitability: list[int | None] | None = None,
    assets: list[str | None] | None = None,
) -> Path:
    """Create a labels parquet matching the expected schema."""
    n = len(log_ids)
    if decision_types is None:
        decision_types = ["trade" if i % 2 == 0 else "record_observation" for i in range(n)]
    if trade_sides is None:
        trade_sides = ["buy" if i % 3 == 0 else "sell" if decision_types[i] == "trade" else None for i in range(n)]
    if risk_prefs is None:
        risk_prefs = [(i % 5) + 1 for i in range(n)]
    if profitability is None:
        profitability = [i % 2 if decision_types[i] == "trade" else None for i in range(n)]
    if assets is None:
        assets = ["ETH" if i % 2 == 0 else "BTC" for i in range(n)]

    rows = []
    for i, lid in enumerate(log_ids):
        rows.append({
            "log_id": lid,
            "decision_type": decision_types[i],
            "trade_side": trade_sides[i],
            "vault_risk_preference": risk_prefs[i],
            "was_profitable_1h": profitability[i],
            "asset": assets[i],
            "label_quality": "high",
            "vault_address": f"0x{lid:04x}",
            "prompt_messages_json": json.dumps([{"role": "user", "content": "test"}]),
        })

    labels_path = tmp_path / "labels.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, labels_path)
    return labels_path


def _make_workflow_labels(
    tmp_path: Path,
    rows: list[dict[str, Any]],
) -> Path:
    labels_path = tmp_path / "workflow_labels.parquet"
    pq.write_table(pa.Table.from_pylist(rows), labels_path)
    return labels_path


# ---------------------------------------------------------------------------
# Label encoding tests
# ---------------------------------------------------------------------------

class TestEncodeLabels:
    def test_decision_type(self):
        rows = [
            {"decision_type": "trade"},
            {"decision_type": "record_observation"},
            {"decision_type": "trade"},
        ]
        filtered, y, names = _encode_labels(rows, "decision_type")
        assert len(filtered) == 3
        assert list(y) == [1, 0, 1]
        assert names == ["record_observation", "trade"]

    def test_trade_side_filters_trades(self):
        rows = [
            {"decision_type": "trade", "trade_side": "buy"},
            {"decision_type": "record_observation", "trade_side": None},
            {"decision_type": "trade", "trade_side": "sell"},
        ]
        filtered, y, names = _encode_labels(rows, "trade_side")
        assert len(filtered) == 2
        assert list(y) == [1, 0]

    def test_profitability_filters_trades(self):
        rows = [
            {"decision_type": "trade", "was_profitable_1h": 1},
            {"decision_type": "trade", "was_profitable_1h": 0},
            {"decision_type": "record_observation", "was_profitable_1h": None},
        ]
        filtered, y, _ = _encode_labels(rows, "was_profitable_1h")
        assert len(filtered) == 2
        assert list(y) == [1, 0]

    def test_executed_valence_derives_three_way_labels(self):
        rows = [
            {"decision_type": "trade", "trade_side": "sell"},
            {"decision_type": "record_observation", "trade_side": None},
            {"decision_type": "trade", "trade_side": "buy"},
        ]
        filtered, y, names = _encode_labels(rows, "executed_valence")
        assert len(filtered) == 3
        assert list(y) == [0, 1, 2]
        assert names == ["bearish", "neutral", "bullish"]

    def test_forced_observe_encodes_binary_labels(self):
        rows = [
            {"forced_observe": False},
            {"forced_observe": True},
            {"forced_observe": 0},
            {"forced_observe": 1},
        ]
        filtered, y, names = _encode_labels(rows, "forced_observe")
        assert len(filtered) == 4
        assert list(y) == [0, 1, 0, 1]
        assert names == ["not_forced_observe", "forced_observe"]

    def test_risk_tolerance_bins(self):
        rows = [
            {"vault_risk_preference": 1},
            {"vault_risk_preference": 2},
            {"vault_risk_preference": 3},
            {"vault_risk_preference": 4},
            {"vault_risk_preference": 5},
        ]
        filtered, y, names = _encode_labels(rows, "risk_tolerance")
        assert list(y) == [0, 0, 1, 2, 2]
        assert names == ["low", "mid", "high"]

    def test_asset_multiclass(self):
        rows = [
            {"asset": "ETH"},
            {"asset": "BTC"},
            {"asset": "ETH"},
            {"asset": "SOL"},
        ]
        filtered, y, names = _encode_labels(rows, "asset")
        assert len(filtered) == 4
        assert y[0] == y[2]  # Both ETH
        assert len(names) == 3

    def test_asset_falls_back_to_target_asset(self):
        rows = [
            {"target_asset": "HOTDOGZ"},
            {"target_asset": "POOPCOIN"},
            {"target_asset": "HOTDOGZ"},
        ]
        filtered, y, names = _encode_labels(rows, "asset")
        assert len(filtered) == 3
        assert list(y) == [0, 1, 0]
        assert names == ["HOTDOGZ", "POOPCOIN"]

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Unknown target"):
            _encode_labels([], "nonexistent")


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestAnalysisDataset:
    def test_load_and_join(self, tmp_path):
        log_ids = list(range(100, 110))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            data_source="router",
        )
        ds = AnalysisDataset(config)
        assert len(ds.rows) == 10
        assert len(ds.y) == 10
        assert ds.num_layers == NUM_LAYERS

    def test_limit(self, tmp_path):
        log_ids = list(range(100, 120))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            limit=5,
        )
        ds = AnalysisDataset(config)
        assert len(ds.rows) == 5

    def test_get_features_router_last_token(self, tmp_path):
        log_ids = list(range(100, 110))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            data_source="router",
            pooling="last_token",
        )
        ds = AnalysisDataset(config)
        X, y = ds.get_features(layer=0)
        assert X.shape == (10, NUM_EXPERTS)
        assert y.shape == (10,)

    def test_get_features_residual_mean_pool(self, tmp_path):
        log_ids = list(range(100, 110))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            data_source="residual",
            pooling="mean_pool",
        )
        ds = AnalysisDataset(config)
        X, y = ds.get_features(layer=1)
        assert X.shape == (10, HIDDEN_DIM)
        assert y.shape == (10,)

    def test_get_router_indices(self, tmp_path):
        log_ids = list(range(100, 110))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            pooling="last_token",
        )
        ds = AnalysisDataset(config)
        indices, y = ds.get_router_indices(layer=0)
        assert indices.shape == (10, TOP_K)
        assert y.shape == (10,)

    def test_missing_activations_skipped(self, tmp_path):
        """Examples with missing safetensors files are skipped."""
        log_ids = list(range(100, 110))
        act_dir = _make_activations(tmp_path, log_ids[:5])  # Only first 5
        labels_path = _make_labels(tmp_path, log_ids)

        # Metadata still references all 10, but only 5 have safetensors
        # Re-write metadata with all 10
        meta_rows = []
        for lid in log_ids:
            meta_rows.append({
                "log_id": lid,
                "seq_len": SEQ_LEN,
                "prompt_hash": f"hash_{lid}",
                "capture_timestamp": "2025-01-01T00:00:00",
                "file_size_bytes": 1024,
                "elapsed_s": 1.0,
                "has_router": True,
                "num_layers_captured": NUM_LAYERS,
                "hidden_dim": HIDDEN_DIM,
                "num_experts": NUM_EXPERTS,
            })
        pq.write_table(
            pa.Table.from_pylist(meta_rows),
            act_dir / "metadata.parquet",
        )

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            data_source="router",
        )
        ds = AnalysisDataset(config)
        X, y = ds.get_features(layer=0)
        assert X.shape[0] == 5  # Only the 5 with files
        assert y.shape[0] == 5

    def test_no_valid_examples_raises(self, tmp_path):
        log_ids = [100]
        act_dir = _make_activations(tmp_path, log_ids)
        # All labels are low quality
        labels_path = tmp_path / "labels.parquet"
        rows = [{
            "log_id": 100,
            "decision_type": "trade",
            "label_quality": "low",
            "vault_address": "0x0",
        }]
        pq.write_table(pa.Table.from_pylist(rows), labels_path)

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
        )
        with pytest.raises(ValueError, match="No valid examples"):
            AnalysisDataset(config)


class TestSliceAwareCompaction:
    def test_run_compact_only_writes_rows_for_label_slice(self, tmp_path):
        log_ids = [100, 101, 102, 103, 104]
        act_dir = _make_activations(tmp_path, log_ids, include_router=False)
        labels_dir = tmp_path / "slice_labels"
        labels_dir.mkdir()
        labels_path = _make_labels(labels_dir, log_ids[:3])

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            target="decision_type",
            data_source="residual",
            layers=[0],
        )
        run_compact(config)

        manifest = json.loads((act_dir / "compact" / "residual_last_token_manifest.json").read_text())
        assert manifest["log_ids"] == log_ids[:3]
        assert manifest["row_count"] == 3

        from safetensors import safe_open

        compact_path = act_dir / "compact" / "residual_last_token_layer0.safetensors"
        with safe_open(str(compact_path), framework="numpy") as handle:
            features = handle.get_tensor("features")
            compact_log_ids = handle.get_tensor("log_ids")

        assert features.shape[0] == 3
        assert compact_log_ids.tolist() == log_ids[:3]

    def test_needs_compact_false_for_slice_subset_of_existing_manifest(self, tmp_path):
        log_ids = [100, 101, 102, 103, 104]
        act_dir = _make_activations(tmp_path, log_ids, include_router=False)

        labels_dir_a = tmp_path / "slice_a"
        labels_dir_a.mkdir()
        labels_path_a = _make_labels(labels_dir_a, log_ids[:4])
        config_a = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path_a,
            target="decision_type",
            data_source="residual",
            layers=[0],
        )
        run_compact(config_a)

        labels_dir_b = tmp_path / "slice_b"
        labels_dir_b.mkdir()
        labels_path_b = _make_labels(labels_dir_b, log_ids[:2])
        config_b = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path_b,
            target="decision_type",
            data_source="residual",
            layers=[0],
        )

        assert _needs_compact(config_b) is False

    def test_needs_compact_true_when_slice_requires_new_rows(self, tmp_path):
        log_ids = [100, 101, 102, 103, 104]
        act_dir = _make_activations(tmp_path, log_ids, include_router=False)

        labels_dir_a = tmp_path / "slice_a"
        labels_dir_a.mkdir()
        labels_path_a = _make_labels(labels_dir_a, log_ids[:3])
        config_a = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path_a,
            target="decision_type",
            data_source="residual",
            layers=[0],
        )
        run_compact(config_a)

        labels_dir_b = tmp_path / "slice_b"
        labels_dir_b.mkdir()
        labels_path_b = _make_labels(labels_dir_b, log_ids[:4])
        config_b = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path_b,
            target="decision_type",
            data_source="residual",
            layers=[0],
        )

        assert _needs_compact(config_b) is True


# ---------------------------------------------------------------------------
# Probe sweep tests
# ---------------------------------------------------------------------------

class TestProbeSweep:
    def test_basic_probe(self, tmp_path):
        """Probe sweep runs and produces output parquet."""
        log_ids = list(range(100, 130))  # 30 examples for meaningful CV
        act_dir = _make_activations(tmp_path, log_ids, seed=42)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=output_dir,
            target="decision_type",
            data_source="router",
            layers=[0, 2],
            n_folds=3,
        )
        results = run_probe_sweep(config)
        assert len(results) == 2  # Two layers
        assert all(
            k in results[0]
            for k in ["layer", "accuracy_mean", "accuracy_std", "balanced_accuracy",
                       "baseline_majority", "baseline_shuffled", "selectivity",
                       "n_examples", "n_classes"]
        )

        # Check output file
        out_file = output_dir / "probe_decision_type_router.parquet"
        assert out_file.exists()
        table = pq.read_table(out_file)
        assert table.num_rows == 2

    def test_probe_with_residual(self, tmp_path):
        log_ids = list(range(100, 130))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=output_dir,
            target="decision_type",
            data_source="residual",
            layers=[0],
            n_folds=3,
        )
        results = run_probe_sweep(config)
        assert len(results) == 1

    def test_probe_grouped_cv_blocks_group_identity_leakage(self, tmp_path):
        log_ids = list(range(100, 140))
        act_dir = _make_activations(
            tmp_path,
            log_ids,
            include_router=False,
            hidden_dim=40,
            seed=0,
        )

        # Overwrite residual activations so each matched_pair_id has a unique feature
        # basis shared by both rows. Random row CV can memorize this; grouped CV cannot.
        res_dir = act_dir / "residual_stream"
        for idx, lid in enumerate(log_ids):
            group_id = idx // 2
            vec = np.zeros((NUM_LAYERS, SEQ_LEN, 40), dtype=np.float16)
            vec[:, :, group_id] = 1.0
            _write_safetensors(res_dir / f"{lid}.safetensors", {"residual_stream": vec})

        workflow_rows = []
        for idx, lid in enumerate(log_ids):
            group_id = idx // 2
            workflow_rows.append(
                {
                    "log_id": lid,
                    "workflow_label": "aligned" if group_id % 2 == 0 else "conflict",
                    "matched_pair_id": f"pair_{group_id}",
                }
            )
        labels_path = _make_workflow_labels(tmp_path, workflow_rows)

        ungrouped = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=tmp_path / "ungrouped",
            target="workflow_label",
            data_source="residual",
            layers=[0],
            n_folds=5,
        )
        grouped = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=tmp_path / "grouped",
            target="workflow_label",
            data_source="residual",
            layers=[0],
            n_folds=5,
            group_column="matched_pair_id",
        )

        ungrouped_results = run_probe_sweep(ungrouped)
        grouped_results = run_probe_sweep(grouped)

        assert ungrouped_results[0]["accuracy_mean"] >= 0.95
        assert grouped_results[0]["accuracy_mean"] < 0.7
        assert grouped_results[0]["split_mode"] == "stratified_group_kfold"


# ---------------------------------------------------------------------------
# Expert analysis tests
# ---------------------------------------------------------------------------

class TestExpertAnalysis:
    def test_basic_expert_analysis(self, tmp_path):
        log_ids = list(range(100, 120))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=output_dir,
            target="decision_type",
            layers=[0, 2],
        )
        results = run_expert_analysis(config)
        # 10 experts per layer, 2 layers
        assert len(results) == 20
        assert results[0]["rank"] == 0  # First result is rank 0
        assert "discriminative_score" in results[0]

        out_file = output_dir / "expert_specialization.parquet"
        assert out_file.exists()


# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------

class TestPCA:
    def test_pca_creates_plots(self, tmp_path):
        log_ids = list(range(100, 120))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=output_dir,
            target="decision_type",
            data_source="router",
            layers=[0, 2],
        )
        run_pca(config)

        assert (output_dir / "pca_layer0_decision_type.png").exists()
        assert (output_dir / "pca_layer2_decision_type.png").exists()


# ---------------------------------------------------------------------------
# Dispatch / CLI tests
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_dispatch_all(self, tmp_path):
        log_ids = list(range(100, 130))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        config = AnalysisConfig(
            activations_dir=act_dir,
            labels_path=labels_path,
            output_dir=output_dir,
            mode="all",
            target="decision_type",
            data_source="router",
            layers=[0],
            n_folds=3,
        )
        results = dispatch(config)
        assert "probe" in results
        assert "experts" in results
        assert "pca" in results


class TestCLI:
    def test_default_args(self):
        from pipelines.interp.analysis import _build_parser

        args = _build_parser().parse_args([])
        assert args.mode == "probe"
        assert args.target == "decision_type"
        assert args.data_source == "router"
        assert args.pooling == "last_token"
        assert args.n_folds == 5

    def test_all_flags(self):
        from pipelines.interp.analysis import _build_parser

        args = _build_parser().parse_args([
            "--mode", "all",
            "--target", "forced_observe",
            "--data-source", "residual",
            "--pooling", "mean_pool",
            "--layers", "0,10,20",
            "--n-folds", "3",
            "--limit", "100",
            "--seed", "123",
        ])
        assert args.mode == "all"
        assert args.target == "forced_observe"
        assert args.data_source == "residual"
        assert args.pooling == "mean_pool"
        assert args.layers == "0,10,20"
        assert args.n_folds == 3
        assert args.limit == 100
        assert args.seed == 123

    def test_main_runs(self, tmp_path):
        log_ids = list(range(100, 130))
        act_dir = _make_activations(tmp_path, log_ids)
        labels_path = _make_labels(tmp_path, log_ids)
        output_dir = tmp_path / "results"

        main([
            "--activations-dir", str(act_dir),
            "--labels-path", str(labels_path),
            "--output-dir", str(output_dir),
            "--mode", "probe",
            "--target", "decision_type",
            "--layers", "0",
            "--n-folds", "3",
        ])

        assert (output_dir / "probe_decision_type_router.parquet").exists()
        assert (output_dir / "results.json").exists()

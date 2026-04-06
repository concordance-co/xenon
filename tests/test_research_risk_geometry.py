from __future__ import annotations

import numpy as np

from projects.DX_TERMINAL.phases.research_rerun.geometry import (
    _context_deformation_metrics,
    _context_realignment_metrics,
    _deformation_pairs,
    _ordered_contexts,
    _selected_prompt_rows,
    _transfer_pairs,
)


def test_ordered_contexts_sorts_risk_levels() -> None:
    rows = [
        {"variant": "risk_5"},
        {"variant": "risk_2"},
        {"variant": "risk_3"},
        {"variant": "risk_1"},
        {"variant": "risk_4"},
    ]
    assert _ordered_contexts(rows) == ["risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]


def test_transfer_pairs_anchor_on_risk_3() -> None:
    contexts = ["risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
    assert _transfer_pairs(contexts) == [
        ("risk_3", "risk_3"),
        ("risk_3", "risk_1"),
        ("risk_3", "risk_2"),
        ("risk_3", "risk_4"),
        ("risk_3", "risk_5"),
    ]


def test_deformation_pairs_include_adjacent_and_end_to_end() -> None:
    contexts = ["risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
    assert _deformation_pairs(contexts) == [
        ("risk_1", "risk_2"),
        ("risk_2", "risk_3"),
        ("risk_3", "risk_4"),
        ("risk_4", "risk_5"),
        ("risk_1", "risk_5"),
    ]


def test_selected_prompt_rows_reads_symbol_index_pairs() -> None:
    row = {
        "metadata": {
            "selected_symbols": ["A", "B", "C", "D"],
            "selected_row_indices": [0, 2, 4, 5],
        }
    }
    assert _selected_prompt_rows(row) == [("A", 0), ("B", 2), ("C", 4), ("D", 5)]


def test_context_realignment_prefers_score_geometry() -> None:
    examples = [
        {
            "geometry_vec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "base_geometry_vec": np.array([3.0, 2.0, 1.0], dtype=np.float32),
            "score_geometry_vec": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        {
            "geometry_vec": np.array([2.0, 4.0, 6.0], dtype=np.float32),
            "base_geometry_vec": np.array([6.0, 4.0, 2.0], dtype=np.float32),
            "score_geometry_vec": np.array([2.0, 4.0, 6.0], dtype=np.float32),
        },
    ]
    metrics = _context_realignment_metrics(examples)
    assert metrics["score_distance_spearman_mean"] is not None
    assert metrics["base_distance_spearman_mean"] is not None
    assert metrics["score_over_base_margin"] > 0


def test_context_deformation_tracks_score_delta() -> None:
    examples = [
        {
            "base_example_id": "ex-1",
            "context_variant": "risk_1",
            "geometry_vec": np.array([1.0, 1.5, 2.0], dtype=np.float32),
            "score_geometry_vec": np.array([1.0, 1.2, 1.4], dtype=np.float32),
        },
        {
            "base_example_id": "ex-1",
            "context_variant": "risk_2",
            "geometry_vec": np.array([2.0, 3.0, 4.5], dtype=np.float32),
            "score_geometry_vec": np.array([2.0, 2.3, 2.6], dtype=np.float32),
        },
        {
            "base_example_id": "ex-2",
            "context_variant": "risk_1",
            "geometry_vec": np.array([0.0, 1.0, 2.0], dtype=np.float32),
            "score_geometry_vec": np.array([0.2, 0.6, 1.0], dtype=np.float32),
        },
        {
            "base_example_id": "ex-2",
            "context_variant": "risk_2",
            "geometry_vec": np.array([0.5, 2.0, 4.0], dtype=np.float32),
            "score_geometry_vec": np.array([1.1, 1.8, 2.5], dtype=np.float32),
        },
    ]
    metrics = _context_deformation_metrics(
        examples,
        source_context="risk_1",
        target_context="risk_2",
    )
    assert metrics["n_examples"] == 2
    assert metrics["deformation_spearman_mean"] is not None
    assert metrics["deformation_cosine_mean"] is not None

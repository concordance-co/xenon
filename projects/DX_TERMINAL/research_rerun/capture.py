"""Capture loader for real-prompt rerun experiments."""

from __future__ import annotations

import json
from typing import Any


def load_prompts_from_db(
    experiment_id: str,
    *,
    tokenizer: Any,
    experiment_group: str | None = None,
) -> list[dict[str, Any]]:
    if tokenizer is None:
        raise ValueError("tokenizer is required for boundary computation")

    from pipelines.db import connect_neon
    from projects.DX_TERMINAL.counterfactual import (
        CanonicalPrompt,
        Snapshot,
        build_market_rows,
        find_downstream_section_boundaries,
        find_row_boundaries,
        parse_market_section,
    )

    conn = connect_neon()
    try:
        group_filter = ""
        params: list[Any] = [experiment_id]
        if experiment_group:
            group_filter = "AND p.experiment_group = %s"
            params.append(experiment_group)
        prompts = conn.execute(
            f"""
            SELECT
                p.prompt_id,
                p.base_example_id,
                p.experiment_id,
                p.experiment_group,
                p.cohort_label,
                p.variant,
                p.system_text,
                p.user_text,
                p.row_order,
                p.settings_signature,
                p.actionability_cell,
                p.target_asset,
                e.market_json
            FROM research_rerun_prompts p
            JOIN research_rerun_examples e ON e.base_example_id = p.base_example_id
            WHERE p.experiment_id = %s
              {group_filter}
            ORDER BY
                p.experiment_group,
                p.cohort_label,
                p.base_example_id,
                CASE p.variant
                    WHEN 'original' THEN 0
                    WHEN 'clear_strategies' THEN 1
                    WHEN 'risk_1' THEN 1
                    WHEN 'risk_2' THEN 2
                    WHEN 'risk_3' THEN 3
                    WHEN 'risk_4' THEN 4
                    WHEN 'risk_5' THEN 5
                    WHEN 'settings_all1' THEN 1
                    WHEN 'settings_all5' THEN 2
                    ELSE 9
                END,
                p.variant
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    capture_prompts: list[dict[str, Any]] = []
    for row in prompts:
        market_json = row["market_json"]
        if isinstance(market_json, str):
            market_json = json.loads(market_json)
        _, row_texts = parse_market_section(row["user_text"])
        market_rows = build_market_rows(market_json, row_texts)
        snapshot = Snapshot(
            snapshot_id=row["base_example_id"],
            vault_address="",
            snap_date="",
            week_num=0,
            market_json=market_json,
            market_header="",
            rows=market_rows,
        )
        prompt_obj = CanonicalPrompt(
            snapshot_id=row["base_example_id"],
            variant=row["variant"],
            system_text=row["system_text"],
            user_text=row["user_text"],
            row_order=list(row["row_order"]),
        )
        section_bounds = find_downstream_section_boundaries(
            tokenizer,
            row["system_text"],
            row["user_text"],
        )
        row_bounds = find_row_boundaries(tokenizer, prompt_obj, snapshot)
        capture_prompts.append(
            {
                "capture_id": row["prompt_id"],
                "base_example_id": row["base_example_id"],
                "experiment_id": row["experiment_id"],
                "experiment_group": row["experiment_group"],
                "cohort_label": row["cohort_label"],
                "variant": row["variant"],
                "system_text": row["system_text"],
                "user_text": row["user_text"],
                "target_asset": row.get("target_asset"),
                "settings_signature": row.get("settings_signature"),
                "actionability_cell": row.get("actionability_cell"),
                "section_boundaries": section_bounds,
                "row_boundaries": row_bounds,
            }
        )
    print(
        f"Loaded {len(capture_prompts)} research rerun prompts from DB "
        f"(experiment_id={experiment_id}, experiment_group={experiment_group or 'all'})"
    )
    return capture_prompts

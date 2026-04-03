from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon
from projects.counterfactual import (
    MARKET_HEADER,
    build_market_rows,
    build_settings_edited_variant,
    compute_labels,
    parse_market_section,
)

ACTIVE_STRATEGIES_HEADER = "## ACTIVE STRATEGIES"
SLIDER_NAMES = [
    "Trade Size",
    "Trading Activity",
    "Holding Style",
    "Diversification",
    "Asset Risk Preference",
]
_SECTION_HEADER_RE = re.compile(r"(?m)^## [^\r\n]+")


def _detect_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def find_section_spans(user_text: str) -> list[tuple[str, int, int]]:
    """Return [(header, start, end), ...] for top-level ## sections."""
    matches = list(_SECTION_HEADER_RE.finditer(user_text))
    spans: list[tuple[str, int, int]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(user_text)
        spans.append((match.group(0).strip(), start, end))
    return spans


def replace_section_body(user_text: str, header: str, body: str) -> str:
    """Replace a top-level ## section body while preserving surrounding sections."""
    eol = _detect_eol(user_text)
    for section_header, start, end in find_section_spans(user_text):
        if not section_header.startswith(header):
            continue
        replacement = f"{section_header}{eol}{eol}{body.strip()}{eol}{eol}"
        return f"{user_text[:start]}{replacement}{user_text[end:]}"
    raise ValueError(f"Section header not found: {header}")


def get_section_body(user_text: str, header: str) -> str:
    """Return the body text for a top-level ## section."""
    eol = _detect_eol(user_text)
    for section_header, start, end in find_section_spans(user_text):
        if not section_header.startswith(header):
            continue
        body = user_text[start:end]
        prefix = f"{section_header}{eol}"
        if body.startswith(prefix):
            body = body[len(prefix):]
        return body.strip()
    raise ValueError(f"Section header not found: {header}")


def clear_active_strategies(user_text: str) -> str:
    """Remove strategy directives while preserving section structure."""
    return replace_section_body(user_text, ACTIVE_STRATEGIES_HEADER, "No active strategies.")


def create_tables(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_rerun_examples (
                base_example_id       TEXT PRIMARY KEY,
                source_log_id         BIGINT NOT NULL UNIQUE,
                vault_address         TEXT NOT NULL,
                source_created_at     TIMESTAMPTZ,
                source_decision_type  TEXT,
                source_action_name    TEXT,
                market_json           JSONB NOT NULL,
                roster                TEXT[] NOT NULL,
                row_order             TEXT[] NOT NULL,
                labels                JSONB NOT NULL,
                metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_rerun_prompts (
                prompt_id             TEXT PRIMARY KEY,
                base_example_id       TEXT NOT NULL REFERENCES research_rerun_examples(base_example_id),
                experiment_id         TEXT NOT NULL,
                experiment_group      TEXT NOT NULL,
                cohort_label          TEXT NOT NULL,
                variant               TEXT NOT NULL,
                system_text           TEXT NOT NULL,
                user_text             TEXT NOT NULL,
                row_order             TEXT[] NOT NULL,
                n_rows                INT NOT NULL,
                target_asset          TEXT,
                block_reason          TEXT,
                settings_signature    TEXT,
                actionability_cell    TEXT,
                metadata              JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at            TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_research_rerun_prompts_experiment
            ON research_rerun_prompts (experiment_id, experiment_group, cohort_label, variant)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_rerun_captures (
                capture_id            TEXT PRIMARY KEY,
                experiment_id         TEXT NOT NULL,
                experiment_group      TEXT NOT NULL,
                base_example_id       TEXT NOT NULL,
                cohort_label          TEXT NOT NULL,
                variant               TEXT NOT NULL,
                seq_len               INT NOT NULL,
                n_rows                INT NOT NULL,
                n_residual_keys       INT NOT NULL DEFAULT 0,
                n_router_keys         INT NOT NULL DEFAULT 0,
                file_size_bytes       BIGINT NOT NULL DEFAULT 0,
                elapsed_s             REAL NOT NULL DEFAULT 0,
                capture_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    conn.commit()


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def _slice_manifest_rows(
    rows: list[dict[str, Any]],
    *,
    blocked_limit: int = 0,
    policy_limit: int = 0,
    buy_limit: int = 0,
    sell_limit: int = 0,
) -> list[dict[str, Any]]:
    if not any([blocked_limit, policy_limit, buy_limit, sell_limit]):
        return rows

    buckets = {
        "blocked_observe": blocked_limit,
        "policy_tension_observe": policy_limit,
        "buy": buy_limit,
        "sell": sell_limit,
    }
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {key: 0 for key in buckets}
    for row in rows:
        cohort = str(row.get("cohort_label") or "")
        limit = buckets.get(cohort, 0)
        if limit <= 0:
            continue
        if seen[cohort] >= limit:
            continue
        kept.append(row)
        seen[cohort] += 1
    return kept


def _load_source_examples(conn: Any, log_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not log_ids:
        return {}
    rows = conn.execute(
        """
        SELECT
            example_id,
            log_id,
            vault_address,
            created_at,
            system_text,
            user_text,
            market_snapshot_json,
            decision_type,
            action_name,
            trade_side,
            asset
        FROM interp_examples_v0
        WHERE log_id = ANY(%s)
        """,
        [log_ids],
    ).fetchall()
    return {int(row["log_id"]): dict(row) for row in rows}


def _build_example_record(row: dict[str, Any]) -> dict[str, Any]:
    market_json = json.loads(row["market_snapshot_json"])
    _, row_texts = parse_market_section(row["user_text"])
    market_rows = build_market_rows(market_json, row_texts)
    row_order = [market_row.symbol for market_row in market_rows]
    labels = compute_labels(market_rows)
    return {
        "base_example_id": row["example_id"],
        "source_log_id": int(row["log_id"]),
        "vault_address": row["vault_address"],
        "source_created_at": row.get("created_at"),
        "source_decision_type": row.get("decision_type"),
        "source_action_name": row.get("action_name"),
        "market_json": market_json,
        "roster": row_order,
        "row_order": row_order,
        "labels": labels,
        "metadata": {
            "trade_side": row.get("trade_side"),
            "asset": row.get("asset"),
        },
    }


def _build_blocked_valence_prompts(
    experiment_id: str,
    source_row: dict[str, Any],
    manifest_row: dict[str, Any],
    row_order: list[str],
) -> list[dict[str, Any]]:
    system_text = source_row["system_text"]
    user_text = source_row["user_text"]
    shared = {
        "base_example_id": source_row["example_id"],
        "experiment_id": experiment_id,
        "experiment_group": "blocked_valence",
        "cohort_label": manifest_row["cohort_label"],
        "row_order": row_order,
        "n_rows": len(row_order),
        "target_asset": manifest_row.get("target_asset"),
        "block_reason": manifest_row.get("block_reason"),
        "settings_signature": manifest_row.get("settings_signature"),
        "actionability_cell": manifest_row.get("actionability_cell"),
        "metadata": {
            **manifest_row,
            "source_log_id": int(source_row["log_id"]),
        },
    }
    return [
        {
            **shared,
            "prompt_id": f"{experiment_id}:{source_row['log_id']}:blocked_valence:original",
            "variant": "original",
            "system_text": system_text,
            "user_text": user_text,
        },
        {
            **shared,
            "prompt_id": f"{experiment_id}:{source_row['log_id']}:blocked_valence:clear_strategies",
            "variant": "clear_strategies",
            "system_text": system_text,
            "user_text": clear_active_strategies(user_text),
        },
    ]


def _build_settings_twist_prompts(
    experiment_id: str,
    source_row: dict[str, Any],
    manifest_row: dict[str, Any],
    row_order: list[str],
) -> list[dict[str, Any]]:
    system_text = source_row["system_text"]
    user_text = source_row["user_text"]
    shared = {
        "base_example_id": source_row["example_id"],
        "experiment_id": experiment_id,
        "experiment_group": "settings_twist",
        "cohort_label": manifest_row["cohort_label"],
        "row_order": row_order,
        "n_rows": len(row_order),
        "target_asset": manifest_row.get("target_asset"),
        "block_reason": manifest_row.get("block_reason"),
        "settings_signature": manifest_row.get("settings_signature"),
        "actionability_cell": manifest_row.get("actionability_cell"),
        "metadata": {
            **manifest_row,
            "source_log_id": int(source_row["log_id"]),
        },
    }
    prompts = [
        {
            **shared,
            "prompt_id": f"{experiment_id}:{source_row['log_id']}:settings_twist:original",
            "variant": "original",
            "system_text": system_text,
            "user_text": user_text,
        }
    ]
    edited_texts: dict[str, str] = {}
    for value, variant in ((1, "settings_all1"), (5, "settings_all5")):
        edited_user_text = build_settings_edited_variant(
            user_text,
            {name: value for name in SLIDER_NAMES},
        )
        edited_texts[variant] = edited_user_text
        prompts.append(
            {
                **shared,
                "prompt_id": f"{experiment_id}:{source_row['log_id']}:settings_twist:{variant}",
                "variant": variant,
                "system_text": system_text,
                "user_text": edited_user_text,
            }
        )
    if edited_texts.get("settings_all1") == edited_texts.get("settings_all5"):
        raise ValueError(
            f"settings_all1/settings_all5 collapsed for log_id={source_row['log_id']}; "
            "ACTIVE SETTINGS rewrite did not produce distinct prompts."
        )
    return prompts


def build_kickoff_prompt_payload(
    manifest_dir: Path,
    *,
    experiment_id: str,
    blocked_limit: int = 0,
    settings_policy_limit: int = 0,
    settings_buy_limit: int = 0,
    settings_sell_limit: int = 0,
) -> dict[str, Any]:
    blocked_rows = _slice_manifest_rows(
        _load_manifest_rows(manifest_dir / "blocked_valence_manifest.json"),
        blocked_limit=blocked_limit,
    )
    settings_rows = _slice_manifest_rows(
        _load_manifest_rows(manifest_dir / "settings_twist_manifest.json"),
        policy_limit=settings_policy_limit,
        buy_limit=settings_buy_limit,
        sell_limit=settings_sell_limit,
    )
    all_manifest_rows = blocked_rows + settings_rows
    log_ids = sorted({int(row["log_id"]) for row in all_manifest_rows})

    with connect_neon() as conn:
        source_examples = _load_source_examples(conn, log_ids)

    missing = [log_id for log_id in log_ids if log_id not in source_examples]
    if missing:
        raise RuntimeError(f"Missing source rows in interp_examples_v0 for log_ids={missing[:10]}")

    examples: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    example_cache: dict[int, dict[str, Any]] = {}

    for log_id in log_ids:
        example_record = _build_example_record(source_examples[log_id])
        example_cache[log_id] = example_record
        examples.append(example_record)

    for row in blocked_rows:
        src = source_examples[int(row["log_id"])]
        example_record = example_cache[int(row["log_id"])]
        prompts.extend(
            _build_blocked_valence_prompts(
                experiment_id,
                src,
                row,
                example_record["row_order"],
            )
        )

    for row in settings_rows:
        src = source_examples[int(row["log_id"])]
        example_record = example_cache[int(row["log_id"])]
        prompts.extend(
            _build_settings_twist_prompts(
                experiment_id,
                src,
                row,
                example_record["row_order"],
            )
        )

    return {
        "experiment_id": experiment_id,
        "examples": examples,
        "prompts": prompts,
        "summary": {
            "base_examples": len(examples),
            "prompts": len(prompts),
            "blocked_base_examples": len(blocked_rows),
            "settings_base_examples": len(settings_rows),
            "prompt_counts_by_group": {
                "blocked_valence": sum(1 for prompt in prompts if prompt["experiment_group"] == "blocked_valence"),
                "settings_twist": sum(1 for prompt in prompts if prompt["experiment_group"] == "settings_twist"),
            },
            "prompt_counts_by_variant": {
                variant: sum(1 for prompt in prompts if prompt["variant"] == variant)
                for variant in sorted({prompt["variant"] for prompt in prompts})
            },
        },
    }


def save_prompt_payload(conn: Any, payload: dict[str, Any]) -> None:
    create_tables(conn)
    experiment_id = payload["experiment_id"]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM research_rerun_prompts WHERE experiment_id = %s",
            [experiment_id],
        )
        for example in payload["examples"]:
            cur.execute(
                """
                INSERT INTO research_rerun_examples
                    (base_example_id, source_log_id, vault_address, source_created_at,
                     source_decision_type, source_action_name, market_json, roster,
                     row_order, labels, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (base_example_id) DO UPDATE SET
                    source_log_id = EXCLUDED.source_log_id,
                    vault_address = EXCLUDED.vault_address,
                    source_created_at = EXCLUDED.source_created_at,
                    source_decision_type = EXCLUDED.source_decision_type,
                    source_action_name = EXCLUDED.source_action_name,
                    market_json = EXCLUDED.market_json,
                    roster = EXCLUDED.roster,
                    row_order = EXCLUDED.row_order,
                    labels = EXCLUDED.labels,
                    metadata = EXCLUDED.metadata
                """,
                [
                    example["base_example_id"],
                    example["source_log_id"],
                    example["vault_address"],
                    example["source_created_at"],
                    example["source_decision_type"],
                    example["source_action_name"],
                    json.dumps(example["market_json"]),
                    example["roster"],
                    example["row_order"],
                    json.dumps(example["labels"]),
                    json.dumps(example["metadata"]),
                ],
            )
        for prompt in payload["prompts"]:
            cur.execute(
                """
                INSERT INTO research_rerun_prompts
                    (prompt_id, base_example_id, experiment_id, experiment_group,
                     cohort_label, variant, system_text, user_text, row_order, n_rows,
                     target_asset, block_reason, settings_signature, actionability_cell, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    prompt["prompt_id"],
                    prompt["base_example_id"],
                    prompt["experiment_id"],
                    prompt["experiment_group"],
                    prompt["cohort_label"],
                    prompt["variant"],
                    prompt["system_text"],
                    prompt["user_text"],
                    prompt["row_order"],
                    prompt["n_rows"],
                    prompt.get("target_asset"),
                    prompt.get("block_reason"),
                    prompt.get("settings_signature"),
                    prompt.get("actionability_cell"),
                    json.dumps(prompt["metadata"]),
                ],
            )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real-prompt rerun datasets in Neon.")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/analysis_results/research_kickoff"),
    )
    parser.add_argument(
        "--experiment-id",
        default="blocked_valence_settings_twist_kickoff_v1",
    )
    parser.add_argument("--blocked-limit", type=int, default=0)
    parser.add_argument("--settings-policy-limit", type=int, default=0)
    parser.add_argument("--settings-buy-limit", type=int, default=0)
    parser.add_argument("--settings-sell-limit", type=int, default=0)
    args = parser.parse_args()

    payload = build_kickoff_prompt_payload(
        args.manifest_dir,
        experiment_id=args.experiment_id,
        blocked_limit=args.blocked_limit,
        settings_policy_limit=args.settings_policy_limit,
        settings_buy_limit=args.settings_buy_limit,
        settings_sell_limit=args.settings_sell_limit,
    )
    with connect_neon(autocommit=False) as conn:
        save_prompt_payload(conn, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class CohortRule:
    label: str
    target_count: int
    group_field: str | None = None
    max_per_group: int | None = None
    max_per_asset: int | None = None
    max_per_vault: int | None = None


@dataclass(slots=True)
class ManifestPlan:
    manifest_name: str = "balanced_v1"
    per_vault_cap: int = 4
    min_spacing_minutes: int = 0
    cohort_rules: list[CohortRule] = field(
        default_factory=lambda: [
            CohortRule(
                label="buy",
                target_count=300,
                group_field="target_asset",
                max_per_group=60,
                max_per_asset=60,
                max_per_vault=1,
            ),
            CohortRule(
                label="sell",
                target_count=300,
                group_field="target_asset",
                max_per_group=60,
                max_per_asset=60,
                max_per_vault=1,
            ),
            CohortRule(
                label="blocked_observe",
                target_count=200,
                group_field="block_reason",
                max_per_group=150,
                max_per_vault=1,
            ),
            CohortRule(
                label="policy_tension_observe",
                target_count=200,
                group_field="settings_cell",
                max_per_group=75,
                max_per_vault=1,
            ),
        ]
    )


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_group_value(value: Any) -> str:
    if value is None:
        return "NONE"
    text = str(value).strip()
    return text or "NONE"


def select_manifest_rows(
    *,
    candidates_by_cohort: dict[str, list[dict[str, Any]]],
    plan: ManifestPlan,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    vault_counts: Counter[str] = Counter()
    asset_counts: dict[str, Counter[str]] = defaultdict(Counter)
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    cohort_vault_counts: dict[str, Counter[str]] = defaultdict(Counter)
    vault_times: dict[str, list[datetime]] = defaultdict(list)

    spacing_seconds = plan.min_spacing_minutes * 60

    rule_positions = {rule.label: 0 for rule in plan.cohort_rules}
    cohort_counts: Counter[str] = Counter()
    exhausted: set[str] = set()

    while True:
        progressed = False

        for rule in plan.cohort_rules:
            if cohort_counts[rule.label] >= rule.target_count or rule.label in exhausted:
                continue

            rows = candidates_by_cohort.get(rule.label, [])
            idx = rule_positions[rule.label]
            picked = False

            while idx < len(rows):
                row = rows[idx]
                idx += 1

                log_id = int(row["log_id"])
                if log_id in selected_ids:
                    continue

                vault = _normalize_group_value(row.get("vault_address"))
                if vault_counts[vault] >= plan.per_vault_cap:
                    continue
                if rule.max_per_vault is not None and cohort_vault_counts[rule.label][vault] >= rule.max_per_vault:
                    continue

                created_at = _parse_dt(row.get("created_at"))
                if created_at is not None and spacing_seconds > 0:
                    too_close = False
                    for existing in vault_times[vault]:
                        if abs((created_at - existing).total_seconds()) < spacing_seconds:
                            too_close = True
                            break
                    if too_close:
                        continue

                asset = _normalize_group_value(row.get("target_asset"))
                if rule.max_per_asset is not None and asset != "NONE":
                    if asset_counts[rule.label][asset] >= rule.max_per_asset:
                        continue

                if rule.group_field is not None:
                    group_value = _normalize_group_value(row.get(rule.group_field))
                    if rule.max_per_group is not None and group_counts[rule.label][group_value] >= rule.max_per_group:
                        continue
                else:
                    group_value = None

                cohort_counts[rule.label] += 1
                out = dict(row)
                out["manifest_name"] = plan.manifest_name
                out["cohort_label"] = rule.label
                out["cohort_rank"] = cohort_counts[rule.label]
                out["group_key"] = group_value
                selected.append(out)

                selected_ids.add(log_id)
                vault_counts[vault] += 1
                cohort_vault_counts[rule.label][vault] += 1
                if asset != "NONE":
                    asset_counts[rule.label][asset] += 1
                if group_value is not None:
                    group_counts[rule.label][group_value] += 1
                if created_at is not None:
                    vault_times[vault].append(created_at)

                progressed = True
                picked = True
                break

            rule_positions[rule.label] = idx
            if not picked and idx >= len(rows):
                exhausted.add(rule.label)

        if not progressed:
            break

    return selected


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cohort_counts = Counter(str(row.get("cohort_label")) for row in rows)
    asset_counts = Counter(
        str(row.get("target_asset"))
        for row in rows
        if row.get("target_asset") not in (None, "", "NONE")
    )
    vault_counts = Counter(str(row.get("vault_address")) for row in rows if row.get("vault_address"))

    return {
        "row_count": len(rows),
        "unique_vault_count": len(vault_counts),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "top_assets": dict(asset_counts.most_common(12)),
        "top_vaults": dict(vault_counts.most_common(12)),
    }


def plan_to_json(plan: ManifestPlan) -> str:
    payload = {
        "manifest_name": plan.manifest_name,
        "per_vault_cap": plan.per_vault_cap,
        "min_spacing_minutes": plan.min_spacing_minutes,
        "cohort_rules": [
            {
                "label": rule.label,
                "target_count": rule.target_count,
                "group_field": rule.group_field,
                "max_per_group": rule.max_per_group,
                "max_per_asset": rule.max_per_asset,
                "max_per_vault": rule.max_per_vault,
            }
            for rule in plan.cohort_rules
        ],
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)

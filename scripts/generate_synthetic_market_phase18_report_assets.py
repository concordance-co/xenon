from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase18_causal_patching"

BASELINE_META = Path("/tmp/phase18_market_behavior_baseline_stratified24_required_v1_metadata.parquet")

RUNS = {
    "leader_1d_project_out": {
        "path": Path("/tmp/phase18_market_behavior_leader_projectout_stratified24_required_v1_metadata.parquet"),
        "label": "Leader 1D",
        "kind": "project_out",
        "layer": 4,
        "target": "leader_axis",
    },
    "leader_1d_random_control": {
        "path": Path("/tmp/phase18_market_behavior_leader_randomcontrol_stratified24_required_v1_metadata.parquet"),
        "label": "Leader 1D",
        "kind": "random_control",
        "layer": 4,
        "target": "leader_axis",
    },
    "leader_4d_project_out": {
        "path": Path("/tmp/phase18_market_behavior_l4top4_projectout_stratified24_required_v1_metadata.parquet"),
        "label": "Leader 4D",
        "kind": "project_out",
        "layer": 4,
        "target": "top4_subspace",
    },
    "leader_4d_random_control": {
        "path": Path("/tmp/phase18_market_behavior_l4top4_randomcontrol_stratified24_required_v1_metadata.parquet"),
        "label": "Leader 4D",
        "kind": "random_control",
        "layer": 4,
        "target": "top4_subspace",
    },
    "dispersion_1d_project_out": {
        "path": Path("/tmp/phase18_market_behavior_dispersion_projectout_stratified24_required_v1_metadata.parquet"),
        "label": "Dispersion 1D",
        "kind": "project_out",
        "layer": 35,
        "target": "dispersion_axis",
    },
    "dispersion_1d_random_control": {
        "path": Path("/tmp/phase18_market_behavior_dispersion_randomcontrol_stratified24_required_v1_metadata.parquet"),
        "label": "Dispersion 1D",
        "kind": "random_control",
        "layer": 35,
        "target": "dispersion_axis",
    },
    "dispersion_4d_project_out": {
        "path": Path("/tmp/phase18_market_behavior_l35top4_projectout_stratified24_required_v1_metadata.parquet"),
        "label": "Dispersion 4D",
        "kind": "project_out",
        "layer": 35,
        "target": "top4_subspace",
    },
    "dispersion_4d_random_control": {
        "path": Path("/tmp/phase18_market_behavior_l35top4_randomcontrol_stratified24_required_v1_metadata.parquet"),
        "label": "Dispersion 4D",
        "kind": "random_control",
        "layer": 35,
        "target": "top4_subspace",
    },
}


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _layer_stats(row: dict[str, Any], *, layer: int) -> dict[str, Any]:
    raw = row.get("patch_stats_json")
    if not raw:
        return {}
    payload = json.loads(raw)
    return payload.get(str(layer), {}) or payload.get(layer, {}) or {}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _tool_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("first_tool_name")) for row in rows).items()))


def _token_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("first_tool_token")) for row in rows if row.get("first_tool_token") is not None).items()))


def _spend_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter("None" if row.get("first_tool_spend_pct") is None else str(float(row.get("first_tool_spend_pct"))) for row in rows).items()))


def _finish_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("finish_reason", "")) for row in rows).items()))


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gen = [int(row.get("generated_token_count", 0)) for row in rows]
    return {
        "count": len(rows),
        "tool_counts": _tool_counts(rows),
        "token_counts": _token_counts(rows),
        "spend_counts": _spend_counts(rows),
        "finish_counts": _finish_counts(rows),
        "mean_generated_token_count": float(sum(gen) / len(gen)),
        "generated_token_count_minmax": [min(gen), max(gen)],
    }


def _compare_rows(baseline_rows: list[dict[str, Any]], intervention_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline = {int(row["log_id"]): row for row in baseline_rows}
    intervention = {int(row["log_id"]): row for row in intervention_rows}
    common = sorted(set(baseline) & set(intervention))
    tool_presence = tool_name = tool_token = 0
    spend_sum = 0.0
    spend_n = 0
    gen_sum = 0.0
    changed_examples: list[dict[str, Any]] = []
    for log_id in common:
        b = baseline[log_id]
        i = intervention[log_id]
        tool_presence_changed = bool(b.get("has_tool_call")) != bool(i.get("has_tool_call"))
        tool_name_changed = b.get("first_tool_name") != i.get("first_tool_name")
        tool_token_changed = b.get("first_tool_token") != i.get("first_tool_token")
        spend_changed = b.get("first_tool_spend_pct") != i.get("first_tool_spend_pct")
        tool_presence += int(tool_presence_changed)
        tool_name += int(tool_name_changed)
        tool_token += int(tool_token_changed)
        gen_sum += abs(int(i.get("generated_token_count", 0)) - int(b.get("generated_token_count", 0)))
        if b.get("first_tool_spend_pct") is not None and i.get("first_tool_spend_pct") is not None:
            spend_sum += abs(float(i.get("first_tool_spend_pct")) - float(b.get("first_tool_spend_pct")))
            spend_n += 1
        if tool_name_changed or tool_token_changed or spend_changed:
            changed_examples.append(
                {
                    "log_id": int(log_id),
                    "baseline_tool": b.get("first_tool_name"),
                    "baseline_token": b.get("first_tool_token"),
                    "baseline_spend": b.get("first_tool_spend_pct"),
                    "intervention_tool": i.get("first_tool_name"),
                    "intervention_token": i.get("first_tool_token"),
                    "intervention_spend": i.get("first_tool_spend_pct"),
                }
            )
    metrics = {
        "count": len(common),
        "tool_presence_change_rate": tool_presence / len(common),
        "tool_name_change_rate": tool_name / len(common),
        "tool_token_change_rate": tool_token / len(common),
        "mean_tool_spend_pct_delta": (spend_sum / spend_n) if spend_n else None,
        "mean_generated_token_count_delta": gen_sum / len(common),
    }
    return metrics, changed_examples


def _patch_summary(rows: list[dict[str, Any]], *, layer: int) -> dict[str, Any]:
    coeff_after: list[float] = []
    delta_std: list[float] = []
    selected_before: list[float] = []
    rows_with_stats = 0
    for row in rows:
        stats = _layer_stats(row, layer=layer)
        if not stats:
            continue
        rows_with_stats += 1
        coeff_after.append(sum(abs(float(x)) for x in stats.get("selected_coeff_after", [])))
        delta_std.append(float(stats.get("delta_norm_std", 0.0)))
        selected_before.append(float(stats.get("selected_proj_norm_before", 0.0)))
    return {
        "rows_with_patch_stats": rows_with_stats,
        "coeff_after_mean": _mean(coeff_after),
        "delta_std_mean": _mean(delta_std),
        "selected_proj_norm_before_mean": _mean(selected_before),
    }


def _plot_change_rates(summary: dict[str, Any]) -> None:
    experiments = ["Leader 1D", "Leader 4D", "Dispersion 1D", "Dispersion 4D"]
    order = ["leader_1d", "leader_4d", "dispersion_1d", "dispersion_4d"]
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0))
    metrics = [
        ("Asset change", "tool_token_change_rate"),
        ("Tool-name change", "tool_name_change_rate"),
        ("Mean spend delta", "mean_tool_spend_pct_delta"),
    ]
    colors = {"project_out": "#b33a2a", "random_control": "#5a5a5a"}
    for ax, (title, key) in zip(axes, metrics):
        xs = list(range(len(order)))
        proj = [summary["experiments"][name]["project_out"]["compare"][key] for name in order]
        rand = [summary["experiments"][name]["random_control"]["compare"][key] for name in order]
        width = 0.32
        ax.bar([x - width / 2 for x in xs], proj, width=width, color=colors["project_out"], label="project_out")
        ax.bar([x + width / 2 for x in xs], rand, width=width, color=colors["random_control"], label="random_control")
        ax.set_xticks(xs, experiments)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase18_change_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_asset_distribution(summary: dict[str, Any]) -> None:
    condition_order = [
        "baseline",
        "leader_1d_project_out",
        "leader_4d_project_out",
        "dispersion_1d_project_out",
        "dispersion_4d_project_out",
    ]
    labels = ["baseline", "L4 1D", "L4 4D", "L35 1D", "L35 4D"]
    token_order = ["MORI", "LUMA", "VEXA", "NERA", "TAVO", "KIRO"]
    palette = ["#b33a2a", "#d16e5b", "#df9c8b", "#b7b7b7", "#7f7f7f", "#454545"]
    fig, ax = plt.subplots(figsize=(9.2, 3.1))
    xs = list(range(len(condition_order)))
    width = 0.12
    for idx, token in enumerate(token_order):
        vals = [summary["conditions"][cond]["token_counts"].get(token, 0) for cond in condition_order]
        ax.bar([x + (idx - 2.5) * width for x in xs], vals, width=width, label=token, color=palette[idx])
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Count")
    ax.set_title("Chosen-asset distribution under targeted project-out")
    ax.legend(frameon=False, ncols=6, fontsize=7, loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase18_asset_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_parquet(BASELINE_META)
    summary = {
        "date": "26 March 2026",
        "model": "Qwen/Qwen3-30B-A3B",
        "phase_name": "phase18_causal_patching_v3",
        "sample": {
            "baseline_prompts": len(baseline_rows),
            "selection_strategy": "stratified_family_variant_roster",
            "distinct_strata": len({(str(r['family']), str(r['family_variant']), str(r.get('roster_key', ''))) for r in baseline_rows}),
            "family_counts": dict(sorted(Counter(str(r["family"]) for r in baseline_rows).items())),
        },
        "generation": {
            "max_tokens": 15000,
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": -1,
            "tool_schema_mode": "trading_v1",
            "tool_choice": "required (chat template kwargs)",
            "context_variant": "market_only",
        },
        "conditions": {
            "baseline": _condition_summary(baseline_rows),
        },
        "experiments": {},
    }

    for prefix in ("leader_1d", "leader_4d", "dispersion_1d", "dispersion_4d"):
        proj_key = f"{prefix}_project_out"
        rand_key = f"{prefix}_random_control"
        proj_cfg = RUNS[proj_key]
        rand_cfg = RUNS[rand_key]
        proj_rows = _load_parquet(proj_cfg["path"])
        rand_rows = _load_parquet(rand_cfg["path"])
        proj_compare, proj_changed = _compare_rows(baseline_rows, proj_rows)
        rand_compare, rand_changed = _compare_rows(baseline_rows, rand_rows)
        summary["conditions"][proj_key] = _condition_summary(proj_rows)
        summary["conditions"][rand_key] = _condition_summary(rand_rows)
        summary["experiments"][prefix] = {
            "label": proj_cfg["label"],
            "layer": proj_cfg["layer"],
            "target": proj_cfg["target"],
            "project_out": {
                "patch": _patch_summary(proj_rows, layer=proj_cfg["layer"]),
                "compare": proj_compare,
                "changed_examples": proj_changed[:8],
            },
            "random_control": {
                "patch": _patch_summary(rand_rows, layer=rand_cfg["layer"]),
                "compare": rand_compare,
                "changed_examples": rand_changed[:8],
            },
        }

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_change_rates(summary)
    _plot_asset_distribution(summary)


if __name__ == "__main__":
    main()

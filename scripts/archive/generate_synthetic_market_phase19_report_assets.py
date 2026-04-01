from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase19_methodology_and_results"

BASELINE_META = Path("/tmp/phase18_market_behavior_baseline_stratified48_chunked_v5_metadata.parquet")
PROJECT_META = Path("/tmp/phase18_market_behavior_jointl4l35top4_projectout_stratified48_chunked_v5_metadata.parquet")
RANDOM_META = Path("/tmp/phase18_market_behavior_jointl4l35top4_randomcontrol_stratified48_chunked_v5_metadata.parquet")

PROJECT_COMPARE = Path("/tmp/phase18_market_behavior_compare_jointl4l35top4_projectout_stratified48_chunked_v5_results.json")
RANDOM_COMPARE = Path("/tmp/phase18_market_behavior_compare_jointl4l35top4_randomcontrol_stratified48_chunked_v5_results.json")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _parse_patch_stats(raw: Any) -> dict[int, dict[str, Any]]:
    if raw in (None, ""):
        return {}
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for key, value in payload.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            out[layer] = value
    return out


def _condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gen = [int(row.get("generated_token_count", 0)) for row in rows]
    tool_names = Counter(str(row.get("first_tool_name")) for row in rows)
    tokens = Counter(str(row.get("first_tool_token")) for row in rows if row.get("first_tool_token") is not None)
    spends = Counter(
        "None" if row.get("first_tool_spend_pct") is None else str(float(row.get("first_tool_spend_pct")))
        for row in rows
    )
    finishes = Counter(str(row.get("finish_reason", "")) for row in rows)
    return {
        "count": len(rows),
        "tool_counts": dict(sorted(tool_names.items())),
        "token_counts": dict(sorted(tokens.items())),
        "spend_counts": dict(sorted(spends.items())),
        "finish_counts": dict(sorted(finishes.items())),
        "mean_generated_token_count": _mean([float(x) for x in gen]),
        "median_generated_token_count": _median([float(x) for x in gen]),
        "generated_token_count_minmax": [min(gen), max(gen)],
    }


def _patch_layer_summary(rows: list[dict[str, Any]], *, layer: int) -> dict[str, Any]:
    applied = 0
    skipped = 0
    coverage = []
    selected_before = []
    coeff_after_abs = []
    delta_norm_std = []
    mean_norm_ratio = []
    mean_std_norm_ratio = []
    rows_with_stats = 0

    for row in rows:
        stats = _parse_patch_stats(row.get("patch_stats_json")).get(layer)
        if not stats:
            continue
        rows_with_stats += 1
        if stats.get("status") == "skipped":
            skipped += 1
        else:
            applied += 1
        cov = stats.get("coverage_fraction")
        if cov is not None:
            coverage.append(float(cov))
        selected = stats.get("selected_proj_norm_before")
        if selected is not None:
            selected_before.append(float(selected))
        coeff_after = stats.get("selected_coeff_after")
        if isinstance(coeff_after, list):
            coeff_after_abs.append(sum(abs(float(x)) for x in coeff_after))
        dns = stats.get("delta_norm_std")
        if dns is not None:
            delta_norm_std.append(float(dns))
        mnr = stats.get("mean_norm_after")
        mnb = stats.get("mean_norm_before")
        if mnb not in (None, 0, 0.0) and mnr is not None:
            mean_norm_ratio.append(float(mnr) / float(mnb))
        msna = stats.get("mean_std_norm_after")
        msnb = stats.get("mean_std_norm_before")
        if msnb not in (None, 0, 0.0) and msna is not None:
            mean_std_norm_ratio.append(float(msna) / float(msnb))

    denom = rows_with_stats if rows_with_stats else 1
    return {
        "rows_with_stats": rows_with_stats,
        "applied_rate": float(applied / denom),
        "skipped_rate": float(skipped / denom),
        "mean_coverage_fraction": _mean(coverage),
        "mean_selected_proj_norm_before": _mean(selected_before),
        "mean_selected_coeff_after_abs": _mean(coeff_after_abs),
        "mean_delta_norm_std": _mean(delta_norm_std),
        "mean_norm_ratio": _mean(mean_norm_ratio),
        "mean_std_norm_ratio": _mean(mean_std_norm_ratio),
    }


def _plot_change_rates(summary: dict[str, Any]) -> None:
    compare = summary["comparisons"]
    metrics = [
        ("Tool-name change", "tool_name_change_rate"),
        ("Asset change", "tool_token_change_rate"),
        ("Mean spend delta", "mean_tool_spend_pct_delta"),
        ("Mean token-count delta", "mean_generated_token_count_delta"),
    ]
    colors = {"project_out": "#b33a2a", "random_control": "#5a5a5a"}
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.9))
    width = 0.55
    for ax, (title, key) in zip(axes, metrics):
        proj = compare["project_out"][key]
        rand = compare["random_control"][key]
        ax.bar([0], [proj], width=width, color=colors["project_out"], label="project_out")
        ax.bar([1], [rand], width=width, color=colors["random_control"], label="random_control")
        ax.set_xticks([0, 1], ["project_out", "random_control"])
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelrotation=18, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase19_change_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_token_distribution(summary: dict[str, Any]) -> None:
    order = ["baseline", "project_out", "random_control"]
    labels = ["baseline", "project_out", "random_control"]
    token_order = sorted(
        {
            token
            for key in order
            for token in summary["conditions"][key]["token_counts"].keys()
        }
    )
    if not token_order:
        token_order = ["None"]
    palette = ["#b33a2a", "#d16e5b", "#df9c8b", "#8a8a8a", "#5f5f5f", "#343434", "#111111"]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    xs = list(range(len(order)))
    width = 0.12 if len(token_order) > 1 else 0.35
    offset0 = (len(token_order) - 1) / 2
    for idx, token in enumerate(token_order):
        vals = [summary["conditions"][cond]["token_counts"].get(token, 0) for cond in order]
        ax.bar(
            [x + (idx - offset0) * width for x in xs],
            vals,
            width=width,
            label=token,
            color=palette[idx % len(palette)],
        )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Count")
    ax.set_title("Chosen-asset distribution")
    ax.legend(frameon=False, ncols=min(6, len(token_order)), fontsize=7, loc="upper center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase19_token_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_generated_token_boxplot(summary: dict[str, Any], baseline_rows: list[dict[str, Any]], project_rows: list[dict[str, Any]], random_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    data = [
        [int(row.get("generated_token_count", 0)) for row in baseline_rows],
        [int(row.get("generated_token_count", 0)) for row in project_rows],
        [int(row.get("generated_token_count", 0)) for row in random_rows],
    ]
    bp = ax.boxplot(data, tick_labels=["baseline", "project_out", "random_control"], patch_artist=True)
    colors = ["#bfbfbf", "#b33a2a", "#5a5a5a"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Generated tokens")
    ax.set_title("Per-prompt generated token count by condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "phase19_generated_token_boxplot.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = _load_parquet(BASELINE_META)
    project_rows = _load_parquet(PROJECT_META)
    random_rows = _load_parquet(RANDOM_META)
    project_compare = _load_json(PROJECT_COMPARE)
    random_compare = _load_json(RANDOM_COMPARE)

    distinct_strata = {
        (
            str(row["family"]),
            str(row["family_variant"]),
            str(row.get("roster_key", "")),
        )
        for row in baseline_rows
    }

    summary = {
        "date": "27 March 2026",
        "model": "Qwen/Qwen3-30B-A3B",
        "phase_name": "phase19_methodology_and_results",
        "sample": {
            "count": len(baseline_rows),
            "selection_strategy": "stratified_family_variant_roster",
            "distinct_strata": len(distinct_strata),
            "context_variant": "market_only",
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": -1,
            "max_tokens": 15000,
            "tool_schema_mode": "trading_v1",
            "tool_choice": "required",
            "batch_size": 8,
            "enable_chunked_prefill": True,
            "cpu": 8,
            "memory_gb": 8,
            "worker_path": "unified request-scoped MarketPatchGPUWorker for all batched conditions",
        },
        "methodology": {
            "verdict": "Good exploratory causal practice, but not yet full best-practice activation patching.",
            "claim_scope": "Necessity-style joint subspace ablation on a stratified cohort with matched random control; not a clean source->base denoising/noising mediation test.",
            "strengths": [
                "same prompt cohort across baseline, project_out, and random_control",
                "same batched worker path across all three conditions",
                "deterministic decode with fixed sampling settings",
                "exact market-span targeting with chunk-aware coverage tracking",
                "matched orthogonal random-control subspace",
                "bootstrap confidence intervals and norm diagnostics",
                "patch-applied and skipped rates reported explicitly",
            ],
            "gaps": [
                "no matched clean/source vs corrupt/base pair construction in the final reported batch",
                "no reverse/noising counterpart on the same final experiment surface",
                "no lambda sweep on the final reported intervention",
                "no neighboring-PC or subspace-size sweep on the final reported batch",
                "output metric is behavioral and coarse, not a logit-difference restoration score",
                "no path-level validation against alternate-route explanations",
            ],
            "scorecard": [
                {"item": "Matched cohort across conditions", "status": "meets"},
                {"item": "Unified engine path across conditions", "status": "meets"},
                {"item": "Localized market-span intervention", "status": "meets"},
                {"item": "Matched random subspace control", "status": "meets"},
                {"item": "Bootstrap CIs and norm diagnostics", "status": "meets"},
                {"item": "Clean/source vs corrupt/base paired contrast", "status": "missing"},
                {"item": "Reverse/noising patch on final batch", "status": "missing"},
                {"item": "Lambda sweep on final batch", "status": "missing"},
                {"item": "Neighboring-PC / subspace sweep on final batch", "status": "partial"},
                {"item": "Path-level validation", "status": "missing"},
            ],
        },
        "conditions": {
            "baseline": _condition_summary(baseline_rows),
            "project_out": _condition_summary(project_rows),
            "random_control": _condition_summary(random_rows),
        },
        "comparisons": {
            "project_out": project_compare,
            "random_control": random_compare,
        },
        "patch": {
            "project_out": {
                "l4": _patch_layer_summary(project_rows, layer=4),
                "l35": _patch_layer_summary(project_rows, layer=35),
            },
            "random_control": {
                "l4": _patch_layer_summary(random_rows, layer=4),
                "l35": _patch_layer_summary(random_rows, layer=35),
            },
        },
    }

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot_change_rates(summary)
    _plot_token_distribution(summary)
    _plot_generated_token_boxplot(summary, baseline_rows, project_rows, random_rows)


if __name__ == "__main__":
    main()

"""Behavioral audit for Phase 09 generations.

Run after `workflow.py` finishes. Parses the capture's `generations.json`
and emits:

- Overall outcome distribution (aligned_match / follow_strategy /
  follow_setting / refuse / other / malformed).
- Distribution by (`target_dimension`, `conflict_band`).
- Strategy-variant x settings-variant heatmaps of `follow_setting` rate
  on conflict rows, per dimension.
- Aligned-row behavioral correctness rate by
  (`target_dimension`, `strategy_direction`, `setting_value`) -- the
  same sniff test that caught the Phase 06 v4 pressure-variant confound.

Usage:

    # Localize generations.json first (if capture ran on Modal):
    uv run modal volume get xenon-data \
      /data/artifacts/prompt_confusion_phase_09_marshalls_battery/capture_.../generations.json \
      /tmp/phase_09_battery_generations.json --force

    # Then run the audit:
    uv run python projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/behavioral_audit.py \
      --generations /tmp/phase_09_battery_generations.json \
      --dataset projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl \
      --output-dir projects/DX_TERMINAL/prompt_confusion/phase_09/reports/marshalls_battery

Generates a markdown audit and a `figures/` directory with the
variant x variant heatmap + outcome bar chart.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _load_generations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("examples") or data.get("generations") or []
    raise ValueError(f"Unrecognized generations file shape: {type(data)}")


def _parse_output(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _classify(row: dict[str, Any], parsed: dict[str, Any] | None) -> str:
    if parsed is None:
        return "malformed"

    action = parsed.get("action")
    size = parsed.get("size")
    target_dim = row.get("target_dimension")
    conflict = bool(row.get("conflict_present"))
    strategy_dir = row.get("strategy_direction")
    setting_implied = row.get("setting_implied_direction")

    if target_dim == "trade_size":
        if action == "observe" or size == "none":
            return "refuse"
        strategy_says = strategy_dir  # "small" | "large"
        setting_says = setting_implied
        model_dir = size
    elif target_dim == "trading_activity":
        if action == "observe" or size == "none":
            model_dir = "observe"
        else:
            model_dir = "trade"
        strategy_says = strategy_dir  # "observe" | "trade"
        setting_says = setting_implied
    else:
        return "other"

    if not conflict:
        return "aligned_match" if model_dir == strategy_says else "aligned_mismatch"

    if model_dir == strategy_says and model_dir != setting_says:
        return "follow_strategy"
    if model_dir == setting_says and model_dir != strategy_says:
        return "follow_setting"
    if target_dim == "trading_activity" and model_dir == "observe":
        return "refuse"
    return "other"


def audit(generations: list[dict[str, Any]], dataset: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {r["example_id"]: r for r in dataset}
    rows: list[dict[str, Any]] = []
    for gen in generations:
        key = gen.get("example_key") or gen.get("key")
        source = by_key.get(key)
        if source is None:
            continue
        parsed = _parse_output(gen.get("text") or gen.get("generated_text") or "")
        rows.append({
            "example_id": key,
            "target_dimension": source["target_dimension"],
            "conflict_band": source["conflict_band"],
            "conflict_present": bool(source["conflict_present"]),
            "strategy_direction": source["strategy_direction"],
            "setting_value": source["setting_value"],
            "setting_implied_direction": source["setting_implied_direction"],
            "strategy_variant_id": source.get("strategy_variant_id"),
            "settings_variant_id": source.get("settings_variant_id"),
            "outcome": _classify(source, parsed),
            "parsed": parsed,
        })
    return {"rows": rows}


def _render_table(counter: Counter, order: list[str] | None = None) -> list[tuple[str, int, str]]:
    total = sum(counter.values()) or 1
    items = list(counter.items())
    if order:
        items.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)
    else:
        items.sort(key=lambda kv: -kv[1])
    return [(k, v, f"{100 * v / total:.1f}%") for k, v in items]


def _write_markdown(audit_rows: list[dict[str, Any]], fig_dir: Path, output_md: Path) -> None:
    outcome_order = ["aligned_match", "aligned_mismatch", "follow_strategy", "follow_setting", "refuse", "other", "malformed"]

    overall = Counter(r["outcome"] for r in audit_rows)
    by_dim_pair: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in audit_rows:
        by_dim_pair[(r["target_dimension"], r["conflict_band"])][r["outcome"]] += 1

    # Aligned correctness by (dim, strategy_direction, setting_value)
    aligned_cells: dict[tuple[Any, Any, Any], Counter] = defaultdict(Counter)
    for r in audit_rows:
        if r["conflict_band"] != "aligned":
            continue
        cell = (r["target_dimension"], r["strategy_direction"], r["setting_value"])
        aligned_cells[cell][r["outcome"]] += 1

    # Variant x variant follow_setting on conflict rows
    variant_cells: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for r in audit_rows:
        if r["conflict_band"] != "strong_conflict":
            continue
        cell = (r["target_dimension"], r["strategy_variant_id"], r["settings_variant_id"])
        variant_cells[cell][r["outcome"]] += 1

    lines: list[str] = []
    lines.append("# Phase 09 Behavioral Audit (Marshall's Battery)\n")
    lines.append(f"Total parsed generations: **{len(audit_rows)}**\n")

    lines.append("## Overall outcome distribution\n")
    lines.append("| Outcome | Count | Share |")
    lines.append("|---|---|---|")
    for k, v, pct in _render_table(overall, outcome_order):
        lines.append(f"| `{k}` | {v} | {pct} |")
    lines.append("")

    lines.append("## By (target_dimension, conflict_band)\n")
    lines.append("| Dimension | Band | Outcome | Count | Share of cell |")
    lines.append("|---|---|---|---|---|")
    for (dim, band) in sorted(by_dim_pair):
        cell_total = sum(by_dim_pair[(dim, band)].values()) or 1
        for k, v, _ in _render_table(by_dim_pair[(dim, band)], outcome_order):
            lines.append(f"| {dim} | {band} | `{k}` | {v} | {100 * v / cell_total:.1f}% |")
    lines.append("")

    lines.append("## Aligned correctness by (dimension, strategy_direction, setting_value)\n")
    lines.append("If an aligned cell shows <90% aligned_match, the row design is muddy there.\n")
    lines.append("| Dimension | strategy_direction | setting_value | aligned_match / total | Rate |")
    lines.append("|---|---|---|---|---|")
    for cell in sorted(aligned_cells, key=lambda c: (str(c[0]), str(c[1]), c[2])):
        cnt = aligned_cells[cell]
        total = sum(cnt.values()) or 1
        matches = cnt.get("aligned_match", 0)
        rate = f"{100 * matches / total:.0f}%"
        lines.append(f"| {cell[0]} | {cell[1]} | {cell[2]} | {matches}/{total} | {rate} |")
    lines.append("")

    lines.append("## Variant x variant resolution (conflict rows, follow_setting rate)\n")
    lines.append(
        "If one variant pair dominates with follow_setting ~= 100%, that's the "
        "Phase 06 v4 pattern and signals variant-wording authority asymmetry.\n"
    )
    # Group by dimension
    by_dim_variant: dict[str, list[tuple[str, str, int, int, int]]] = defaultdict(list)
    for (dim, sv, setv), cnt in variant_cells.items():
        non_refuse = cnt.get("follow_strategy", 0) + cnt.get("follow_setting", 0)
        follow_set = cnt.get("follow_setting", 0)
        refuse = cnt.get("refuse", 0)
        by_dim_variant[dim].append((sv, setv, follow_set, non_refuse, refuse))
    for dim in sorted(by_dim_variant):
        lines.append(f"### {dim}\n")
        lines.append("| strategy_variant | settings_variant | follow_setting / non-refuse | rate | refuse |")
        lines.append("|---|---|---|---|---|")
        for sv, setv, follow_set, non_refuse, refuse in sorted(by_dim_variant[dim]):
            rate = "n/a" if non_refuse == 0 else f"{100 * follow_set / non_refuse:.0f}%"
            lines.append(f"| {sv} | {setv} | {follow_set}/{non_refuse} | {rate} | {refuse} |")
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines))


def _render_figures(audit_rows: list[dict[str, Any]], fig_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    outcome_order = ["aligned_match", "aligned_mismatch", "follow_strategy", "follow_setting", "refuse", "other", "malformed"]
    cat_colors = {
        "aligned_match": "#4c72b0", "aligned_mismatch": "#3b3b3b",
        "follow_strategy": "#55a868", "follow_setting": "#c44e52",
        "refuse": "#8172b2", "other": "#ccb974", "malformed": "#999999",
    }

    # Figure: outcome stacked by (dim, band)
    groups: list[tuple[str, str]] = sorted({(r["target_dimension"], r["conflict_band"]) for r in audit_rows})
    labels = [f"{d}\n{b}" for d, b in groups]
    counts = {cat: [] for cat in outcome_order}
    for g in groups:
        local = Counter(r["outcome"] for r in audit_rows if (r["target_dimension"], r["conflict_band"]) == g)
        for cat in outcome_order:
            counts[cat].append(local.get(cat, 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(groups))
    for cat in outcome_order:
        vals = np.array(counts[cat])
        if vals.sum() == 0:
            continue
        ax.bar(labels, vals, bottom=bottoms, color=cat_colors[cat], label=cat)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, bottoms[i] + v / 2, str(v), ha="center", va="center", fontsize=8, color="white")
        bottoms += vals
    ax.set_ylabel("Row count")
    ax.set_title("Phase 09 behavioral audit: outcome by (dimension, band)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_outcome_by_dim_band.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Figure: variant x variant follow_setting heatmap per dimension
    for dim in ("trade_size", "trading_activity"):
        cells: dict[tuple[str, str], tuple[int, int]] = {}
        for r in audit_rows:
            if r["target_dimension"] != dim or r["conflict_band"] != "strong_conflict":
                continue
            k = (r["strategy_variant_id"], r["settings_variant_id"])
            fs_prev, tot_prev = cells.get(k, (0, 0))
            if r["outcome"] == "follow_setting":
                fs_prev += 1
                tot_prev += 1
            elif r["outcome"] == "follow_strategy":
                tot_prev += 1
            cells[k] = (fs_prev, tot_prev)
        if not cells:
            continue
        sv_ids = sorted({k[0] for k in cells})
        set_ids = sorted({k[1] for k in cells})
        grid = np.full((len(sv_ids), len(set_ids)), np.nan)
        counts_grid = np.zeros_like(grid, dtype=int)
        for (sv, setv), (fs, tot) in cells.items():
            if tot == 0:
                continue
            i = sv_ids.index(sv); j = set_ids.index(setv)
            grid[i, j] = fs / tot
            counts_grid[i, j] = tot

        fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(set_ids), 1.2 + 0.8 * len(sv_ids)))
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="RdBu_r")
        for i in range(len(sv_ids)):
            for j in range(len(set_ids)):
                v = grid[i, j]
                t = counts_grid[i, j]
                label = "n/a" if np.isnan(v) else f"{v:.0%}\n(n={t})"
                color = "white" if (not np.isnan(v) and abs(v - 0.5) > 0.35) else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(set_ids)))
        ax.set_xticklabels(set_ids, rotation=20, ha="right")
        ax.set_yticks(range(len(sv_ids)))
        ax.set_yticklabels(sv_ids)
        ax.set_title(f"{dim} conflict: follow_setting rate (non-refuse rows)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        plt.tight_layout()
        plt.savefig(fig_dir / f"fig_variant_heatmap_{dim}.png", dpi=140, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 09 behavioral audit.")
    parser.add_argument("--generations", type=Path, required=True,
                        help="Local path to generations.json from the capture artifact.")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Phase 09 dataset jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write audit.md and figures/.")
    args = parser.parse_args()

    dataset = _load_jsonl(args.dataset)
    gens = _load_generations(args.generations)
    audit_result = audit(gens, dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    _render_figures(audit_result["rows"], fig_dir)
    _write_markdown(audit_result["rows"], fig_dir, args.output_dir / "behavioral_audit.md")

    print(f"wrote {args.output_dir / 'behavioral_audit.md'}")
    print(f"figures under {fig_dir}")


if __name__ == "__main__":
    main()

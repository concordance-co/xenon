from __future__ import annotations

import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
ASSET_DIR = ROOT / "data" / "report_assets" / "postmarket_context_geometry_evidence"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _best_synth_realignment(path: Path) -> dict:
    data = _load_json(path)
    best = None
    best_by_row = {}
    for context, rows_by_key in data["set_geometry_context_realignment"].items():
        for row_key, rows in rows_by_key.items():
            for row in rows:
                rec = {
                    "margin": row["score_over_base_margin"],
                    "context": context,
                    "state": row_key,
                    "layer": row["layer"],
                    "spearman": row["score_distance_spearman_mean"],
                }
                if best is None or rec["margin"] > best["margin"]:
                    best = rec
                current = best_by_row.get(row_key)
                if current is None or rec["margin"] > current["margin"]:
                    best_by_row[row_key] = rec
    return {"best": best, "best_by_state": best_by_row}


def _best_real_row_realignment(path: Path) -> dict:
    data = _load_json(path)
    best = None
    best_by_row = {}
    for context, rows_by_key in data["context_realignment"].items():
        for row_key, rows in rows_by_key.items():
            for row in rows:
                rec = {
                    "margin": row["score_over_base_margin"],
                    "context": context,
                    "state": row_key,
                    "layer": row["layer"],
                }
                if best is None or rec["margin"] > best["margin"]:
                    best = rec
                current = best_by_row.get(row_key)
                if current is None or rec["margin"] > current["margin"]:
                    best_by_row[row_key] = rec
    return {"best": best, "best_by_state": best_by_row}


def _real_postmarket_summary(path: Path) -> dict:
    data = _load_json(path)
    out = {}
    for group_name, group in data["groups"].items():
        best = None
        best_by_state = {}
        heatmap = {}
        for context, state_map in group["realignment"].items():
            heatmap[context] = {}
            for state, rows in state_map.items():
                best_for_cell = max(rows, key=lambda r: r["score_over_base_margin"])
                heatmap[context][state] = {
                    "margin": best_for_cell["score_over_base_margin"],
                    "layer": best_for_cell["layer"],
                }
                for row in rows:
                    rec = {
                        "margin": row["score_over_base_margin"],
                        "context": context,
                        "state": state,
                        "layer": row["layer"],
                    }
                    if best is None or rec["margin"] > best["margin"]:
                        best = rec
                    current = best_by_state.get(state)
                    if current is None or rec["margin"] > current["margin"]:
                        best_by_state[state] = rec
        out[group_name] = {
            "best": best,
            "best_by_state": best_by_state,
            "heatmap": heatmap,
            "summary": group["summary"],
        }
    return out


def _barh_evidence_chain(summary: dict) -> None:
    labels = [
        "Synthetic risk\n(row-local)",
        "Synthetic portfolio\n(row-local)",
        "Synthetic affordance\n(row-local)",
        "Real risk\n(row-local bridge)",
        "Real risk\n(post-market bridge)",
        "Real affordance\n(post-market bridge)",
    ]
    rows = [
        summary["synth_risk"]["best"],
        summary["synth_portfolio"]["best"],
        summary["synth_affordance"]["best"],
        summary["real_row_risk"]["best"],
        summary["real_postmarket"]["risk_postmarket_geometry"]["best"],
        summary["real_postmarket"]["affordance_postmarket_geometry"]["best"],
    ]
    values = [r["margin"] for r in rows]
    colors = ["#6a8caf", "#8aa399", "#b57763", "#b7b7b7", "#6f8f70", "#b33a2a"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="none")
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlabel("Best score-over-base margin", fontsize=10)
    ax.set_title("One evidence chain across the program", fontsize=13, loc="left", pad=12)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)

    for yi, val, row in zip(y, values, rows):
        text = f"{val:.3f}\n{row['context']} · {row['state']} · L{row['layer']}"
        x = val + 0.01 if val >= 0 else val - 0.01
        ha = "left" if val >= 0 else "right"
        ax.text(x, yi, text, va="center", ha=ha, fontsize=8.5, color="#333333")

    ax.set_xlim(min(-0.05, min(values) - 0.05), max(values) + 0.12)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "evidence_chain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _state_summary_chart(real_summary: dict) -> None:
    risk = real_summary["risk_postmarket_geometry"]["best_by_state"]
    aff = real_summary["affordance_postmarket_geometry"]["best_by_state"]
    states = [
        "market_mean",
        "market_eos",
        "active_settings_eos",
        "portfolio_eos",
        "constraints_eos",
        "last_token",
    ]
    risk_vals = [risk[s]["margin"] for s in states]
    aff_vals = [aff[s]["margin"] for s in states]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(states))
    w = 0.35
    ax.bar(x - w / 2, risk_vals, width=w, color="#6f8f70", label="Risk")
    ax.bar(x + w / 2, aff_vals, width=w, color="#b33a2a", label="Affordance")
    ax.set_xticks(
        x,
        ["market\nmean", "market\nEOS", "settings\nEOS", "portfolio\nEOS", "constraints\nEOS", "last\ntoken"],
        fontsize=9,
    )
    ax.set_ylabel("Best margin inside that state", fontsize=10)
    ax.set_title("Where the real post-market signal is strongest", fontsize=13, loc="left", pad=12)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")

    for xs, vals in [(x - w / 2, risk_vals), (x + w / 2, aff_vals)]:
        for xi, val in zip(xs, vals):
            ax.text(xi, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "real_state_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _context_heatmaps(real_summary: dict) -> None:
    groups = [
        ("risk_postmarket_geometry", "Risk"),
        ("affordance_postmarket_geometry", "Affordance"),
    ]
    states = [
        "market_mean",
        "market_eos",
        "active_settings_eos",
        "portfolio_eos",
        "constraints_eos",
        "last_token",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.9))
    vmin, vmax = -0.05, 0.42
    cmap = plt.get_cmap("RdYlGn")

    for ax, (group_name, title) in zip(axes, groups):
        heat = real_summary[group_name]["heatmap"]
        contexts = list(heat.keys())
        matrix = np.array([[heat[c][s]["margin"] for s in states] for c in contexts], dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(np.arange(len(states)), ["market\nmean", "market\nEOS", "settings\nEOS", "portfolio\nEOS", "constraints\nEOS", "last\ntoken"], fontsize=8)
        ax.set_yticks(np.arange(len(contexts)), contexts, fontsize=8)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="#222")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92)
    cbar.set_label("Best score-over-base margin", fontsize=9)
    fig.suptitle("Real post-market geometry shifts by context and prompt state", fontsize=13, x=0.06, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "real_context_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rowlocal_vs_postmarket(summary: dict) -> None:
    labels = [
        "Real risk\nrow-local",
        "Real risk\npost-market",
        "Real affordance\npost-market",
    ]
    rows = [
        summary["real_row_risk"]["best"],
        summary["real_postmarket"]["risk_postmarket_geometry"]["best"],
        summary["real_postmarket"]["affordance_postmarket_geometry"]["best"],
    ]
    values = [r["margin"] for r in rows]
    colors = ["#999999", "#6f8f70", "#b33a2a"]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Best margin", fontsize=10)
    ax.set_title("Why the post-market bridge matters", fontsize=13, loc="left", pad=12)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)

    for xi, val, row in zip(x, values, rows):
        ax.text(
            xi,
            val + (0.012 if val >= 0 else -0.012),
            f"{val:.3f}\n{row['state']} · {row['context']} · L{row['layer']}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8.5,
        )

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "rowlocal_vs_postmarket.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _prompt_examples_figure() -> None:
    def wrap_prompt_block(text: str, width: int = 48) -> str:
        out_lines: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line:
                out_lines.append("")
                continue
            if line.startswith("## "):
                out_lines.append(line)
                continue
            if line.startswith("- "):
                wrapped = textwrap.wrap(
                    line[2:],
                    width=width,
                    initial_indent="- ",
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                out_lines.extend(wrapped or [line])
                continue
            wrapped = textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            out_lines.extend(wrapped or [line])
        return "\n".join(out_lines)

    synthetic_market = """## ACTIVE SETTINGS
- No explicit settings are provided. No hard execution constraints are supplied.

## CONSTRAINTS
- No hard execution constraints are supplied.

## MARKET SNAPSHOT
- Asset A ... 5m +6.5%, 1h +10.0%, flow +1.35
- Asset B ... 5m +4.0%, 1h +7.5%, flow +0.92
- Asset C ... 5m +3.4%, 1h +6.9%, flow +0.81
- Asset D ... 5m +2.6%, 1h +6.1%, flow +0.67"""

    synthetic_aff4 = """## ACTIVE SETTINGS
- No explicit settings are provided. The top two assets are blocked for new adds and the third asset is size-limited.

## CONSTRAINTS
- Buy constraint: Asset A is capped for new adds this tick.
- Routing note: Asset B is in confirmation-only mode after an execution warning.
- Hard block: Assets A and B should not be increased this tick.
- Size cap: Asset C can only be added in small size.

## MARKET SNAPSHOT
- Asset A ... same market numbers as base
- Asset B ... same market numbers as base
- Asset C ... same market numbers as base
- Asset D ... same market numbers as base"""

    real_market = """## ACTIVE SETTINGS
- TA 5 / Risk 2 / Size 3 / Hold 2 / Div 4
- Same slider block as the harder variant.

## CONSTRAINTS
- Max Trade Amount: 100% of available ETH
- Price impact limit: 1500 bps
- AIGF: BUY max 100%
- HOLE: BUY max 100%
- HOTDOGZ: BUY max 100%, SELL max 100%
- LMAO: BUY max 100%
- LOOKSMAX: BUY max 100%
- POOPCOIN: BUY max 100%"""

    real_aff4 = """## ACTIVE SETTINGS
- TA 5 / Risk 2 / Size 3 / Hold 2 / Div 4
- Same slider block as the base variant.

## CONSTRAINTS
- Max Trade Amount: 12% of available ETH
- Price impact limit: 400 bps
- AIGF: BUY max 0%
- HOLE: BUY max 0%
- HOTDOGZ: BUY max 12%, SELL max 100%
- LMAO: BUY max 0%
- LOOKSMAX: BUY max 0%
- POOPCOIN: BUY max 0%"""

    panels = [
        ("Synthetic `market_only`", synthetic_market, "#f3f6f8"),
        ("Synthetic `affordance_4`", synthetic_aff4, "#faf1ef"),
        ("Real `market_only` excerpt", real_market, "#f3f6f8"),
        ("Real `affordance_4` excerpt", real_aff4, "#faf1ef"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10.8, 13.6))
    fig.suptitle("Matched affordance-ladder prompt examples", fontsize=14, x=0.05, ha="left", y=0.988)
    for ax, (title, body, bg) in zip(axes, panels):
        ax.set_facecolor(bg)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.035, 0.95, title, fontsize=11, fontweight="bold", va="top", ha="left", transform=ax.transAxes)
        ax.text(
            0.035,
            0.885,
            wrap_prompt_block(body, width=110),
            fontsize=8.5,
            va="top",
            ha="left",
            family="monospace",
            linespacing=1.34,
            transform=ax.transAxes,
        )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.9)
    fig.savefig(ASSET_DIR / "affordance_prompt_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = {
        "synth_risk": _best_synth_realignment(ROOT / "data/analysis_results/synthetic_market_representation/phase11_set_geometry_risk_ladder_v1/results.json"),
        "synth_portfolio": _best_synth_realignment(ROOT / "data/analysis_results/synthetic_market_representation/phase13_set_geometry_portfolio_ladder_v1"),
        "synth_affordance": _best_synth_realignment(ROOT / "data/analysis_results/synthetic_market_representation/phase14_set_geometry_affordance_ladder_v1/results.json"),
        "real_row_risk": _best_real_row_realignment(ROOT / "data/analysis_results/research_risk_geometry/real_risk_geometry_bridge_v1/results.json"),
        "real_postmarket": _real_postmarket_summary(ROOT / "data/analysis_results/research_postmarket_geometry/real_postmarket_geometry_bridge_v2_full24_results.json"),
    }

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _barh_evidence_chain(summary)
    _state_summary_chart(summary["real_postmarket"])
    _context_heatmaps(summary["real_postmarket"])
    _rowlocal_vs_postmarket(summary)
    _prompt_examples_figure()


if __name__ == "__main__":
    main()

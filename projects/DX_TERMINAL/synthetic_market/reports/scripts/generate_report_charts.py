"""Generate charts for the Xenon research blog post and client summary."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = "/Users/brockelmore/concordance/xenon/projects/DX_TERMINAL/synthetic_market/reports/assets/public_story/charts"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#059669"
RED = "#DC2626"
GRAY = "#6B7280"
LIGHT_BLUE = "#BFDBFE"
LIGHT_ORANGE = "#FED7AA"
LIGHT_RED = "#FECACA"
SLATE = "#334155"


def make_horizontal_bar(ax, labels, values, colors, xlim, xlabel, title, title_color=SLATE, val_fmt="{:.3f}"):
    """Simple horizontal bar chart. labels[0]/values[0] = bottom bar."""
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.55, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=14)
    for i, val in enumerate(values):
        ax.text(val + (xlim[1] - xlim[0]) * 0.02, i,
                val_fmt.format(val), va="center", fontsize=10.5, color=SLATE, fontweight="bold")


# ──────────────────────────────────────────────
# Chart 1: Market Factor Decodability
# ──────────────────────────────────────────────
def chart_decodability():
    # Defined bottom-to-top. Sorted ascending so best is at top.
    data = [
        ("Net capital flow",       0.994),
        ("Price change (5 min)",   0.997),
        ("Volume (5 min)",         0.997),
        ("Price change (1 hour)",  0.998),
        ("Volume (1 hour)",        0.998),
        ("Unique traders",         0.998),
        ("Attractiveness score",   0.998),
        ("Risk-adjusted score",    0.998),
        ("Holder concentration",   0.999),
    ]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=BLUE, height=0.6, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0.990, 1.002)
    ax.set_xlabel("Recovery accuracy (R²)", fontsize=11, fontweight="bold")
    ax.set_title("How Well Can We Read Market Data\nFrom the Model's Internal State?",
                 fontsize=14, fontweight="bold", pad=16)
    for i, val in enumerate(values):
        ax.text(1.001, i, f"{val:.3f}", va="center", ha="left",
                color=SLATE, fontweight="bold", fontsize=9.5)
    ax.axvline(x=1.0, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.4)
    fig.savefig(f"{OUT}/01_market_decodability.png")
    plt.close(fig)
    print("  done: 01")


# ──────────────────────────────────────────────
# Chart 2: Relational vs Individual Stability
# ──────────────────────────────────────────────
def chart_relational():
    fig, ax = plt.subplots(figsize=(5, 4.5))
    categories = ["Single-asset identity", "Pairwise relationship"]
    margins = [0.015, 0.30]
    ax.bar(categories, margins, color=[LIGHT_BLUE, BLUE], width=0.45,
           edgecolor=[BLUE, BLUE], linewidth=1.5)
    ax.set_ylabel("Stability margin (higher = more robust)", fontsize=10, fontweight="bold")
    ax.set_title("What Survives When You\nRearrange the Prompt?",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_ylim(0, 0.48)

    # Value labels above each bar
    ax.text(0, 0.015 + 0.012, "0.015", ha="center", fontweight="bold", fontsize=12, color=SLATE)
    ax.text(1, 0.30 + 0.012, "0.300", ha="center", fontweight="bold", fontsize=12, color=SLATE)

    # Annotation with plenty of room above the value label
    ax.text(1, 0.43, "~20x more stable", ha="center",
            fontsize=11, fontweight="bold", color=BLUE)

    fig.savefig(f"{OUT}/02_relational_stability.png")
    plt.close(fig)
    print("  done: 02")


# ──────────────────────────────────────────────
# Chart 3: Context Effects
# ──────────────────────────────────────────────
def chart_context():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = ["Risk framing\nbefore market data", "Constraint framing\nbefore market data"]
    gaps = [0.061, 0.070]
    bars = ax.bar(labels, gaps, color=[ORANGE, GREEN], width=0.45, edgecolor="white", linewidth=1)
    ax.set_ylabel("Representation shift", fontsize=11, fontweight="bold")
    ax.set_title("Pre-Market Context Changes\nHow the Model Reads the Market",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_ylim(0, 0.10)
    for i, val in enumerate(gaps):
        ax.text(i, val + 0.003, f"{val:.3f}", ha="center",
                fontweight="bold", fontsize=12, color=SLATE)
    fig.savefig(f"{OUT}/03_context_effect.png")
    plt.close(fig)
    print("  done: 03")


# ──────────────────────────────────────────────
# Chart 4a: Leader Signal
# ──────────────────────────────────────────────
def chart_leader_signal():
    # Bottom-to-top: worst at bottom, best at top.
    data = [
        ("1h price range (single)",              0.38, LIGHT_BLUE),
        ("5m vol leader z-score (single)",       0.40, LIGHT_BLUE),
        ("1h volume, top asset (single)",        0.46, LIGHT_BLUE),
        ("1h price + 5m volume (pair)",          0.67, BLUE),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    make_horizontal_bar(
        ax,
        labels=[d[0] for d in data],
        values=[d[1] for d in data],
        colors=[d[2] for d in data],
        xlim=(0, 0.85),
        xlabel="How well this feature predicts the signal (R²)",
        title='"Leader" Signal  (early layers)\nTracks the standout asset',
        title_color=BLUE,
        val_fmt="{:.2f}",
    )
    fig.savefig(f"{OUT}/04a_leader_signal.png")
    plt.close(fig)
    print("  done: 04a")


# ──────────────────────────────────────────────
# Chart 4b: Dispersion Signal
# ──────────────────────────────────────────────
def chart_dispersion_signal():
    data = [
        ("5m price spread (single)",                0.45, LIGHT_ORANGE),
        ("1h vol leader z-score (single)",          0.52, LIGHT_ORANGE),
        ("1h price deviation (single)",             0.52, LIGHT_ORANGE),
        ("5m vol mean + 1h vol median (pair)",      0.84, ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    make_horizontal_bar(
        ax,
        labels=[d[0] for d in data],
        values=[d[1] for d in data],
        colors=[d[2] for d in data],
        xlim=(0, 1.05),
        xlabel="How well this feature predicts the signal (R²)",
        title='"Dispersion" Signal  (later layers)\nTracks how uneven the market is',
        title_color=ORANGE,
        val_fmt="{:.2f}",
    )
    fig.savefig(f"{OUT}/04b_dispersion_signal.png")
    plt.close(fig)
    print("  done: 04b")


# ──────────────────────────────────────────────
# Chart 5: Selectivity Test (12/12 wins)
# ──────────────────────────────────────────────
def chart_selectivity():
    conditions = ["Leader\n(constructive)", "Leader\n(destructive)",
                  "Dispersion\n(constructive)", "Dispersion\n(destructive)"]
    targeted =    [0.4375, 0.5625, 0.40625, 0.3125]
    random_ctrl = [0.6875, 0.6875, 0.75,    0.65625]

    x = np.arange(len(conditions))
    w = 0.28

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(x - w / 2, targeted, w, label="Targeted edit", color=BLUE, edgecolor="white")
    ax.bar(x + w / 2, random_ctrl, w, label="Random edit (control)",
           color=LIGHT_RED, edgecolor=RED, linewidth=1)

    ax.set_ylabel("Choice disruption rate", fontsize=11, fontweight="bold")
    ax.set_title("Targeted Edits Are More Precise Than Random Ones",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    for i in range(len(conditions)):
        gap = random_ctrl[i] - targeted[i]
        top = max(targeted[i], random_ctrl[i])
        ax.text(x[i], top + 0.04, f"gap: {gap:.1%}",
                ha="center", fontsize=9.5, fontweight="bold", color=GREEN)

    fig.text(0.5, -0.01,
             "Lower disruption from targeted edits = the signal is specific and real.\n"
             "All 12 comparisons (3 strengths x 4 conditions) showed this pattern.",
             ha="center", fontsize=9, color=GRAY, style="italic")
    fig.savefig(f"{OUT}/05_selectivity.png")
    plt.close(fig)
    print("  done: 05")


# ──────────────────────────────────────────────
# Chart 6: Restoration Results
# ──────────────────────────────────────────────
def chart_restoration():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.45})

    # Each list: bottom-to-top display order.
    # Leader
    l_labels = ["Choice match improvement", "Backfire rate", "Fix rate", "Spend improvement"]
    l_values = [4.2,                         6.3,             25.0,       66.7]
    l_colors = [GREEN,                       RED,             GREEN,      GREEN]
    l_fmts =   ["4.2%",                     "6.3%",          "25.0%",    "66.7%"]

    y = np.arange(len(l_labels))
    ax1.barh(y, l_values, color=l_colors, height=0.5, edgecolor="white", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(l_labels, fontsize=10)
    ax1.set_xlim(0, 85)
    ax1.set_xlabel("Percentage", fontsize=10, fontweight="bold")
    ax1.set_title('"Leader" Signal', fontsize=13, fontweight="bold", color=BLUE, pad=12)
    for i, label in enumerate(l_fmts):
        ax1.text(l_values[i] + 1.5, i, label, va="center", fontsize=10,
                 color=SLATE, fontweight="bold")

    # Dispersion
    d_labels = ["Choice match improvement", "Backfire rate", "Fix rate", "Spend improvement"]
    d_values = [2.1,                         15.4,            13.6,       60.0]
    d_colors = [RED,                         RED,             ORANGE,     GREEN]
    d_fmts =   ["-2.1%",                    "15.4%",         "13.6%",    "60.0%"]

    ax2.barh(y, d_values, color=d_colors, height=0.5, edgecolor="white", alpha=0.85)
    ax2.set_yticks(y)
    ax2.set_yticklabels(d_labels, fontsize=10)
    ax2.set_xlim(0, 85)
    ax2.set_xlabel("Percentage", fontsize=10, fontweight="bold")
    ax2.set_title('"Dispersion" Signal', fontsize=13, fontweight="bold", color=ORANGE, pad=12)
    for i, label in enumerate(d_fmts):
        ax2.text(d_values[i] + 1.5, i, label, va="center", fontsize=10,
                 color=SLATE, fontweight="bold")

    fig.savefig(f"{OUT}/06_restoration.png")
    plt.close(fig)
    print("  done: 06")


# ──────────────────────────────────────────────
# Chart 7: Context Ladder
# ──────────────────────────────────────────────
def chart_context_ladder():
    contexts = ["Market\nonly", "Risk 1", "Risk 2", "Risk 3", "Risk 4", "Risk 5"]
    coord_r2 = [1.0, 0.999, 0.998, 0.997, 0.996, 0.995]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(range(len(contexts)), coord_r2, "o-", color=BLUE, linewidth=2.5, markersize=9, zorder=3)
    ax.fill_between(range(len(contexts)), 0.990, coord_r2, alpha=0.12, color=BLUE)
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, fontsize=10)
    ax.set_ylim(0.990, 1.003)
    ax.set_ylabel("Coordinate recovery (R²)", fontsize=11, fontweight="bold")
    ax.set_title("The Model's Market Map Stays Stable\nAcross All Risk Contexts",
                 fontsize=14, fontweight="bold", pad=16)
    ax.axhline(y=1.0, color=GRAY, linestyle=":", linewidth=0.8, alpha=0.4)
    fig.savefig(f"{OUT}/07_context_ladder.png")
    plt.close(fig)
    print("  done: 07")


# ──────────────────────────────────────────────
# Chart 8: Research Arc Summary
# ──────────────────────────────────────────────
def chart_research_arc():
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.set_xlim(-0.7, 6.7)
    ax.set_ylim(-0.5, 3.0)
    ax.axis("off")

    stages = [
        ("Can we\nread it?",       "R² > 0.99\nfor all factors",   GREEN,  "Y"),
        ("Is it\nrelational?",     "20x more\nstable",             GREEN,  "Y"),
        ("Does context\nchange it?","Yes, measurably",             GREEN,  "Y"),
        ("What are\nthe signals?",  "Leader +\nDispersion",        GREEN,  "Y"),
        ("Are they\nreal?",         "12 / 12\nselectivity wins",   GREEN,  "Y"),
        ("Do they explain\nthe decision?", "+4.2% leader\n-2.1% dispersion", ORANGE, "?"),
    ]

    for i, (question, answer, color, symbol) in enumerate(stages):
        circle = plt.Circle((i, 1.3), 0.35, color=color, alpha=0.12, zorder=2)
        ax.add_patch(circle)
        ax.text(i, 1.3, symbol, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color, zorder=3)
        ax.text(i, 2.3, question, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=SLATE, linespacing=1.3)
        ax.text(i, 0.3, answer, ha="center", va="center",
                fontsize=8.5, color=GRAY, linespacing=1.3)
        if i < len(stages) - 1:
            ax.annotate("", xy=(i + 0.6, 1.3), xytext=(i + 0.4, 1.3),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))

    ax.set_title("Six Questions, Honest Answers",
                 fontsize=14, fontweight="bold", pad=8, color=SLATE)
    fig.savefig(f"{OUT}/08_research_arc.png")
    plt.close(fig)
    print("  done: 08")


if __name__ == "__main__":
    print("Generating charts...")
    chart_decodability()
    chart_relational()
    chart_context()
    chart_leader_signal()
    chart_dispersion_signal()
    chart_selectivity()
    chart_restoration()
    chart_context_ladder()
    chart_research_arc()
    print(f"\nAll charts saved to {OUT}/")

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_03_response_label_probe")
SUMMARY_PATH = REPORT_DIR / "summary.json"
ASSET_DIR = REPORT_DIR / "assets"
TABLE_DIR = REPORT_DIR / "tables"


LABEL_TITLES = {
    "helpful_harmless_off_diagonal": "Helpful/Harmless Off-Diagonal",
    "strong_helpful": "Strong Helpful",
    "strong_harmless": "Strong Harmless",
}


def _metric(row: dict[str, Any], split: str, metric: str) -> float | None:
    value = row.get(split, {}).get(metric)
    return float(value) if value is not None else None


def _save_best_summary(summary: dict[str, Any]) -> None:
    best = summary["best"]
    labels = list(summary["labels"])
    x = range(len(labels))
    holdout = [best[label]["primary_balanced_accuracy"] for label in labels]
    cv = [best[label]["cv_balanced_accuracy"] for label in labels]
    text = [0.5 for _ in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - 0.25 for i in x], holdout, width=0.25, label="Source-family holdout BA")
    ax.bar(x, cv, width=0.25, label="Random CV BA")
    ax.bar([i + 0.25 for i in x], text, width=0.25, label="Chance")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL_TITLES[label] for label in labels], rotation=20, ha="right")
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Best Response-Side Probe Results")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "best_balanced_accuracy.png", dpi=180)
    plt.close(fig)


def _save_layer_curves(summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    layers = [int(layer) for layer in summary["layers"]]
    views = list(metrics.keys())
    labels = list(summary["labels"])

    for label in labels:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, split_name, metric_key in (
            (axes[0], "Source-family holdout", "source_family_holdout"),
            (axes[1], "Random CV", "cv"),
        ):
            for view in views:
                y_values: list[float | None] = []
                for layer in layers:
                    row = metrics.get(view, {}).get(label, {}).get(str(layer), {})
                    y_values.append(_metric(row, metric_key, "balanced_accuracy"))
                if any(value is not None for value in y_values):
                    ax.plot(layers, y_values, marker="o", label=view.replace("_", " "))
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_title(split_name)
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Balanced accuracy")
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.suptitle(f"{LABEL_TITLES[label]}: View/Layer Curves")
        fig.tight_layout()
        fig.savefig(ASSET_DIR / f"{label}_layer_curves.png", dpi=180)
        plt.close(fig)


def _save_fold_breakdown(summary: dict[str, Any]) -> None:
    best = summary["best"]
    labels = list(summary["labels"])
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 9), sharex=False)
    if len(labels) == 1:
        axes = [axes]

    for ax, label in zip(axes, labels, strict=True):
        folds = best[label]["source_family_holdout"]["folds"]
        names = [fold["heldout"].replace("_", "\n") for fold in folds]
        ba = [fold["balanced_accuracy"] for fold in folds]
        auroc = [fold["auroc"] for fold in folds]
        x = range(len(folds))
        ax.bar([i - 0.18 for i in x], ba, width=0.36, label="BA")
        ax.bar([i + 0.18 for i in x], auroc, width=0.36, label="AUROC")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_ylim(0.45, 0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names)
        ax.set_title(f"{LABEL_TITLES[label]} best fold breakdown")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "best_source_family_fold_breakdown.png", dpi=180)
    plt.close(fig)


def _write_tables(summary: dict[str, Any]) -> None:
    rows = []
    for label, row in summary["best"].items():
        rows.append(
            {
                "label": label,
                "view": row["view"],
                "layer": row["layer"],
                "holdout_balanced_accuracy": row["primary_balanced_accuracy"],
                "holdout_auroc": row["source_family_holdout"]["auroc"],
                "cv_balanced_accuracy": row["cv_balanced_accuracy"],
                "cv_auroc": row["cv_auroc"],
            }
        )
    (TABLE_DIR / "best_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _write_html(summary: dict[str, Any]) -> None:
    best_rows = []
    for label, row in summary["best"].items():
        best_rows.append(
            "<tr>"
            f"<td>{html.escape(LABEL_TITLES[label])}</td>"
            f"<td>{html.escape(row['view'])}</td>"
            f"<td>{row['layer']}</td>"
            f"<td>{row['primary_balanced_accuracy']:.3f}</td>"
            f"<td>{row['source_family_holdout']['auroc']:.3f}</td>"
            f"<td>{row['cv_balanced_accuracy']:.3f}</td>"
            f"<td>{row['cv_auroc']:.3f}</td>"
            "</tr>"
        )
    label_counts = json.dumps(summary["label_counts"], indent=2)
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MoReBench Experiment 03 Response Label Probe</title>
  <style>
    body {{ font: 16px/1.5 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 40px; max-width: 1180px; }}
    h1, h2 {{ line-height: 1.15; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; border: 1px solid #e5e5e5; border-radius: 8px; margin: 12px 0 28px; }}
    code, pre {{ background: #f7f7f7; border-radius: 6px; }}
    pre {{ padding: 12px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>MoReBench Experiment 03 Response Label Probe</h1>
  <p>Local chart report generated from <code>{SUMMARY_PATH}</code>. Primary metric is source-family-holdout balanced accuracy.</p>
  <h2>Best Results</h2>
  <table>
    <thead><tr><th>Label</th><th>View</th><th>Layer</th><th>Holdout BA</th><th>Holdout AUROC</th><th>CV BA</th><th>CV AUROC</th></tr></thead>
    <tbody>{''.join(best_rows)}</tbody>
  </table>
  <img src="assets/best_balanced_accuracy.png" alt="Best balanced accuracy chart">
  <h2>Layer/View Curves</h2>
  <img src="assets/helpful_harmless_off_diagonal_layer_curves.png" alt="Off-diagonal layer curves">
  <img src="assets/strong_helpful_layer_curves.png" alt="Strong helpful layer curves">
  <img src="assets/strong_harmless_layer_curves.png" alt="Strong harmless layer curves">
  <h2>Source-Family Fold Breakdown</h2>
  <img src="assets/best_source_family_fold_breakdown.png" alt="Best fold breakdown">
  <h2>Label Counts</h2>
  <pre>{html.escape(label_counts)}</pre>
</body>
</html>
"""
    (REPORT_DIR / "report.html").write_text(html_text, encoding="utf-8")


def _append_markdown_links() -> None:
    report_path = REPORT_DIR / "report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    marker = "<!-- local-chart-report -->"
    chart_md = f"""
{marker}

## Local Chart Assets

Open `report.html` in this directory for the browsable chart report.

![Best balanced accuracy](assets/best_balanced_accuracy.png)

![Helpful/harmless off-diagonal layer curves](assets/helpful_harmless_off_diagonal_layer_curves.png)

![Strong helpful layer curves](assets/strong_helpful_layer_curves.png)

![Strong harmless layer curves](assets/strong_harmless_layer_curves.png)

![Source-family fold breakdown](assets/best_source_family_fold_breakdown.png)
"""
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n\n" + chart_md.lstrip()
    else:
        existing = existing.rstrip() + "\n\n" + chart_md.lstrip()
    report_path.write_text(existing, encoding="utf-8")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    _save_best_summary(summary)
    _save_layer_curves(summary)
    _save_fold_breakdown(summary)
    _write_tables(summary)
    _write_html(summary)
    _append_markdown_links()
    print(f"Wrote chart report to {REPORT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()

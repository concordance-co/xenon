from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_INPUT_JSON = (
    Path("projects/DX_TERMINAL/prompt_confusion/phase_13/reports/signal_discovery")
    / "report_dd8c8ac3385c_7e82ff1b/results/l32_settings_top25_complaint_review.json"
)
DEFAULT_OUTPUT_HTML = (
    Path("projects/DX_TERMINAL/prompt_confusion/phase_13/reports/signal_discovery")
    / "report_dd8c8ac3385c_7e82ff1b/l32_settings_top25_complaint_review.html"
)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _short(value: Any, limit: int = 1800) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + " ..."


def _root_badge(row: Mapping[str, Any]) -> str:
    root = str(row.get("root_cause") or "")
    config = bool(row.get("config_conflict_like"))
    cls = "config" if config else "nonconfig"
    if root == "RULE_FABRICATION":
        cls = "rule"
    return f'<span class="badge {cls}">{_esc(root)}</span>'


def _render_row(row: Mapping[str, Any]) -> str:
    sliders = "/".join(
        _esc(row.get(name))
        for name in ("slider_ta", "slider_arp", "slider_ts", "slider_hs", "slider_div")
    )
    return f"""
    <article class="row">
      <header>
        <div>
          <span class="rank">#{_esc(row.get("rank"))}</span>
          <span class="score">{float(row.get("projection", 0.0)):.3f}</span>
          <span class="dist">dist {float(row.get("distance_to_structure_control_mean", 0.0)):.3f}</span>
          {_root_badge(row)}
          <span class="badge">{_esc(row.get("complaint_type"))}</span>
          <span class="badge {'config' if row.get('config_conflict_like') else 'nonconfig'}">
            config={_esc(row.get("config_conflict_like"))}
          </span>
          <span class="badge">TA/R/TS/H/D {sliders}</span>
        </div>
        <code>{_esc(row.get("example_id"))}</code>
      </header>
      <section>
        <h4>Complaint</h4>
        <p>{_esc(row.get("complaint_text"))}</p>
      </section>
      <section>
        <h4>Active Strategy Directives</h4>
        <pre>{_esc(_short(row.get("strategy_directives_excerpt"), 2400))}</pre>
      </section>
      <section>
        <h4>Settings</h4>
        <pre>{_esc(_short(row.get("settings_excerpt"), 2200))}</pre>
      </section>
      <section>
        <h4>Decision</h4>
        <pre>{_esc(_short(row.get("decision_excerpt"), 1800))}</pre>
      </section>
    </article>
    """


def _counts(rows: list[Mapping[str, Any]]) -> str:
    n = len(rows)
    config = sum(1 for row in rows if row.get("config_conflict_like"))
    rule = sum(1 for row in rows if row.get("root_cause") == "RULE_FABRICATION")
    user_config = sum(1 for row in rows if row.get("root_cause") == "USER_CONFIG_CONFLICT")
    active = sum(1 for row in rows if "[" in str(row.get("strategy_directives_excerpt") or ""))
    return (
        f"{n} rows | config_conflict_like {config}/{n} | "
        f"USER_CONFIG_CONFLICT {user_config}/{n} | RULE_FABRICATION {rule}/{n} | active directives {active}/{n}"
    )


def render(payload: Mapping[str, Any]) -> str:
    direction_sections = []
    for direction, direction_payload in dict(payload["directions"]).items():
        top_rows = list(direction_payload["top_complaints"])
        bottom_rows = list(direction_payload["bottom_complaints_closest_to_control"])
        means = direction_payload["stratum_means"]
        direction_sections.append(
            f"""
            <section class="direction">
              <h2>{_esc(direction)}</h2>
              <p class="means">
                anchor={float(means.get("anchor_positive", 0.0)):.3f}
                complaint={float(means.get("complaint", 0.0)):.3f}
                control={float(means.get("structure_matched_control", 0.0)):.3f}
              </p>
              <h3>Top Complaints</h3>
              <p class="counts">{_esc(_counts(top_rows))}</p>
              {''.join(_render_row(row) for row in top_rows)}
              <h3>Bottom Complaints Closest To Control</h3>
              <p class="counts">{_esc(_counts(bottom_rows))}</p>
              {''.join(_render_row(row) for row in bottom_rows)}
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phase 13 L32 Settings Top/Bottom Complaint Review</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #20211f;
    }}
    body {{
      margin: 0;
      padding: 28px;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    h1, h2, h3, h4 {{
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    h2 {{
      margin: 36px 0 8px;
      font-size: 24px;
    }}
    h3 {{
      margin: 28px 0 8px;
      font-size: 18px;
    }}
    h4 {{
      margin: 12px 0 6px;
      font-size: 12px;
      text-transform: uppercase;
      color: #5c625a;
    }}
    .meta, .means, .counts {{
      color: #555c53;
      font-size: 14px;
    }}
    .row {{
      background: #ffffff;
      border: 1px solid #deded7;
      border-radius: 8px;
      margin: 12px 0;
      padding: 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .row header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      flex-wrap: wrap;
      border-bottom: 1px solid #ecece6;
      padding-bottom: 10px;
      margin-bottom: 10px;
    }}
    code {{
      font-size: 11px;
      color: #666a64;
      max-width: 100%;
      overflow-wrap: anywhere;
    }}
    .rank, .score, .dist, .badge {{
      display: inline-block;
      font-size: 12px;
      line-height: 1.4;
      border: 1px solid #d4d7cf;
      border-radius: 999px;
      padding: 2px 8px;
      margin: 0 4px 4px 0;
      background: #f6f7f2;
    }}
    .score {{
      font-weight: 700;
      background: #e6f0ff;
      border-color: #b8caef;
    }}
    .config {{
      background: #e9f8ee;
      border-color: #a9d7b5;
    }}
    .nonconfig {{
      background: #f4f0e6;
      border-color: #d6c9a7;
    }}
    .rule {{
      background: #fff0e8;
      border-color: #dfb39e;
    }}
    p {{
      margin: 0;
      line-height: 1.45;
    }}
    pre {{
      margin: 0;
      padding: 10px;
      background: #f8f8f5;
      border: 1px solid #e3e3dd;
      border-radius: 6px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.4;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Phase 13 L32 Settings Top/Bottom Complaint Review</h1>
    <p class="meta">
      Source artifact: {_esc(payload.get("source_transform_artifact"))} |
      Corpus: {_esc(payload.get("corpus_table"))} |
      Cell: L{_esc(payload.get("cell", {}).get("layer"))} {_esc(payload.get("cell", {}).get("position"))} |
      Top-k: {_esc(payload.get("top_k"))}
    </p>
    {''.join(direction_sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-html", type=Path, default=DEFAULT_OUTPUT_HTML)
    args = parser.parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    args.output_html.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output_html": str(args.output_html)}, indent=2))


if __name__ == "__main__":
    main()

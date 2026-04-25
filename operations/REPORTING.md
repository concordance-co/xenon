# Reporting

The one legitimate local flow: remote results → local report assets → PDF or MD output.

## The pattern

1. Workflow runs on Modal. Artifacts land in the Modal volume and are indexed in the catalog.
2. Locally, pull result payloads (not activations, not full generations) through `OperationArtifact` / `.result()` / `.summary()`.
3. Render charts locally from result payloads.
4. Build Typst or MD. Emit PDF.

Report artifacts live at:

- phase level: `projects/<p>/<sub>/<phase>/reports/`
- project level: `projects/<p>/reports/`

Each `reports/` tree contains:

- `report.typ` or `report.md` — the source
- `assets/` — generated PNGs, static images
- `scripts/` — chart-generating and asset-generating scripts

## Hard rules

- **`ReportSpec` runs local, never on Modal.** The chart stack (`matplotlib`, `pipelines_v2.reporting`) is not in the Modal runtime image. A workflow that assigns `ReportSpec` to a Modal runner will fail.
- **Lazy imports from `pipelines_v2.reporting`.** Any code that can be imported on the Modal path must keep `pipelines_v2.reporting` imports inside the functions that use them. Top-level imports pull `matplotlib` into the Modal container.
- **No jsonl dumps of generations "for inspection."** If you need to read a few rows, query Neon directly or use a `result.summary()` payload. Don't write a local file as the reading surface.
- **PNG and PDF outputs in the `reports/` tree only.** Don't scatter chart outputs across scratch dirs or phase roots.
- **Use `paths.py` helpers for locations.** Don't hardcode `projects/<p>/...` paths in report scripts.

## What belongs in a report

Reports consume result payloads from artifacts, not raw features.

Good inputs:

- probe or transfer result JSON (per-layer AUROC, metrics, persisted test predictions)
- direction result metadata
- geometry projection payloads (already reduced upstream)
- comparison tables from `PatchComparisonSpec` results
- qualitative review JSONs produced by phase scripts

Not inputs:

- raw activations
- full generation sets
- anything that would require localizing a capture artifact at scale

## Typst vs MD

- **Typst** for polished deliverables — phase exit reports meant to persist, project-level reports. Commit the `.typ` source and the generated PDF.
- **MD** for interim notes, working documents, phase exit artifacts in the methodology `PHASE.md` sense. See `methodology/templates/PHASE.md`.
- If you find yourself wanting notebook-style inline chart + prose, use a report script that generates PNGs plus a static source doc. Don't check notebooks into the repo.

## The local build cycle

```bash
# generate PNGs from existing artifacts
uv run python projects/<p>/<sub>/<phase>/reports/scripts/build_charts.py

# compile Typst to PDF
typst compile projects/<p>/<sub>/<phase>/reports/report.typ
```

Report scripts should:

- take artifact ids or run ids as input, not rerun capture
- read through `OperationArtifact` / refs, not by downloading volumes
- write PNGs into `reports/assets/`
- be idempotent — running twice produces the same output

## Anti-patterns

- Report scripts that re-run capture instead of consuming existing artifacts.
- Report scripts that pull activations locally for one-off plots.
- Hardcoded paths.
- Committing generated PDFs without the `.typ` source and build script.
- Chart code living in the phase root instead of `reports/scripts/`.
- Top-level `import matplotlib` in any module that Modal might import.

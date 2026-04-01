# Scripts

The `scripts/` tree is for one-off helpers, dataset builders, manifest builders,
and research-specific chart generation.

The canonical operator surface is not `scripts/`; it is:

```bash
uv run -m pipelines.cli ...
```

## Layout

- `scripts/db/`
  - Neon helpers and one-off DB migrations
- `scripts/manifests/`
  - capture manifest builders and reconciliation helpers
- `scripts/research/`
  - effort-specific research payload builders
- `scripts/archive/`
  - archived phase runners, historical dataset prep helpers, and old chart/report generators

## Top-Level Scripts

- `scripts/modal_capture.sh`
  - legacy wrapper around parts of the runtime surface
- `scripts/modal_restore_db.sh`
  - restore helper
- `scripts/xenon_backend.sh`
  - legacy backend helper, not the recommended path

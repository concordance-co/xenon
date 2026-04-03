# Scripts

The `scripts/` tree is for one-off helpers, dataset builders, manifest builders,
and project-specific chart generation.

The canonical operator surface is not `scripts/`; it is:

```bash
uv run -m pipelines.cli ...
```

## Layout

- `scripts/archive/`
  - archived phase runners, old DB/admin helpers, historical dataset prep helpers, and old chart/report generators

## Top-Level Scripts

There are no supported top-level operator scripts anymore.

Anything under `scripts/archive/` is historical only and should be treated as effectively deleted unless you are explicitly digging through old work.

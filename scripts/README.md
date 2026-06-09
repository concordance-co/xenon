# Scripts

The `scripts/` tree holds small helpers that are still relevant to the live
`pipelines_v2` platform surface.

The canonical operator surface is:

```bash
uv run python -m pipelines_v2.cli ...
```

## Top-Level Scripts

- `scripts/pipelines_v2_orchestrator_smoke.py`
  - v2 smoke helper
- `scripts/pipelines_v2_router_layer_probe.py`
  - v2 router probing helper
- `scripts/skills.py`
  - validates repo-local skill conventions and syncs host metadata

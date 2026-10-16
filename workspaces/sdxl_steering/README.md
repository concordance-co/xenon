# SDXL capture and steering plumbing

The active deliverable is a small engineering proof, not the frozen research campaign in `study.py`. It exercises the ordinary workflow runner, persisted paired noise, internal capture, remote channel-direction derivation, signed tensor addition, remote image metrics, and local gallery/report publication.

The active ordinary workflows generate **six images total**: phase 00 cool-blue/warm-orange discovery, then phase 01 neutral baseline, +4 RMS, +12 RMS and −4 RMS. Phase 01 consumes phase 00's small saved direction. The large discovery/held-out study remains planned and unexecuted.

```bash
# Two discovery images, then remote reduction/local export.
uv run python -m pipelines_v2.cli workflow run --file workspaces/sdxl_steering/phase_00/workflow.py --logging INFO

# Four comparison images using the small direction export.
XENON_SDXL_DIRECTION=/absolute/path/direction.json \
uv run python -m pipelines_v2.cli workflow plan --file workspaces/sdxl_steering/phase_01/workflow.py
XENON_SDXL_DIRECTION=/absolute/path/direction.json \
uv run python -m pipelines_v2.cli workflow run --file workspaces/sdxl_steering/phase_01/workflow.py --logging INFO
```

If both discovery captures completed but later analysis/export failed, set
`XENON_SDXL_DISCOVERY_RUN=wr_...` on the phase 00 command to derive the direction
from those cataloged captures. This starts a new analysis-only workflow and
preserves their recorded lineage; it does not regenerate images.

`XENON_SDXL_OUTPUT` selects the local presentation/export directory (default `~/.xenon/reports/sdxl-plumbing`). `direction.json` is a bounded 1280-value derived vector, never a capture bank. Shared images and activations remain on `xenon-data:/data/sdxl-plumbing`; the existing pinned SDXL cache lives on `xenon-models:/models`. The CLI retains normal local/shared catalog defaults.

One A100 80GB, 8 CPU, 64 GiB host memory, one worker, 900-second GPU function deadline and 1200-second client deadline per step. CPU reductions are capped at 300 seconds/16 GiB. The coordinator must stop at the eight-image smoke cap and the overall 7200 GPU-second study cap; do not automatically retry with extra images. Successful runs produce `gallery.html`, `metrics.json`, and cataloged report artifacts. Failures remain failed workflow steps; absence of a gallery is not a successful result.

The direction is the warm-minus-cool background channel mean, averaged across steps 8–12 at `mid_block.resnets.1`, normalized to observed discovery activation RMS. Four outer rectangles are edited only on the conditional CFG branch. This aggressive intervention can damage images. Pixel red-minus-blue, central/background RGB change, clipping, near-black/white fractions and edge energy are diagnostics, not semantic quality scores. Human quality and prompt adherence remain unmeasured.

The optional `build_workflow(full_gut_check=False)` emits baseline/+12 only when a smaller follow-up gate is explicitly needed. The default CLI path includes all four comparison arms.

# Conflict Probe — Execution Steps

## Status

- [x] Dataset: 375 prompts in `conflict_probe_examples_v0` on Neon
- [ ] Capture: activations on Modal volume
- [ ] Probe: train + analyze
- [ ] Results: decide go/no-go

## Step 1: Run Capture

```bash
uv run --extra modal modal run pipelines/interp/modal_vllm_orchestrator.py --mode conflict-probe --experiment-id conflict_probe_v0 --batch-size 10 --gpu A100-80GB
```

375 examples × last_token pooling × all 48 layers. Output: `(48, 2048)` per example on `xenon-data` volume at `/data/activations/conflict_probe/conflict_probe_v0/`.

Capture metadata written to `conflict_probe_captures` table in Neon.

## Step 2: Run Probe Analysis

```bash
modal run pipelines/interp/conflict_probe/probe.py --experiment-id conflict_probe_v0
```

Runs on Modal (reads activations from volume). Does:
1. Per-layer L1 logistic regression (5-fold CV) — accuracy + AUROC by layer
2. Cross-strategy transfer — train on 4 sliders, test on held-out slider
3. Per-slider breakdown — which sliders are most detectable
4. Conflict strength gradient — does probe confidence correlate with strength (0–4)

Results saved to `/data/activations/conflict_probe/conflict_probe_v0/probe_results/probe_results.json` on the volume.

## Step 3: Read Results

```bash
# Quick check from the probe output (printed to stdout)
# Or download the JSON:
modal volume get xenon-data activations/conflict_probe/conflict_probe_v0/probe_results/probe_results.json ./probe_results.json
cat ./probe_results.json | python3 -m json.tool
```

## Success Criteria (from spec)

| Metric | Threshold | Interpretation |
|--------|-----------|---------------|
| Best layer AUROC > 0.5 | Signal exists | Keep going |
| Best layer AUROC > 0.7 | Strong signal | Proceed to real data validation |
| Cross-strategy transfer > 0.6 | General conflict direction | Evidence for universal conflict representation |

## If Signal Exists → Phase 1

- Build LLM judge to label real trade data for conflict
- Test probe generalization from synthetic → real
- Check whether observations under conflict look different from no-conflict
- Test whether conflict direction predicts decision outcomes (the 2×2 matrix)

## If No Signal

- Try non-linear probes (MLP)
- Try section-pooled representations instead of last_token
- Try larger dataset (more base prompts, more strategy templates)
- Consider that conflict might not be a linear feature at this model scale

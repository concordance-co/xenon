# Prompt Confusion — Scratchpad

## Key References for Next Phase

### "Who is In Charge? Dissecting..."
- **Source**: https://openreview.net/forum?id=RfOOn897hj
- **Local PDF**: `70_Who_is_In_Charge_Dissecting.pdf`
- **Relevance**: Leaning on this for the next phase of the prompt confusion exploration. Need to review methodology for how it applies to our conflict probe reframe.

## Open Questions from Phase 0 Null

- We never ran inference to check if the model's *behavior* actually changes across the slider sweep. If it doesn't, there's nothing for the probe to detect at last-token.
- Labels were by input construction (slider vs strategy text), not by model output. Zeng-style reframe: label by what the model actually did.
- No PCA/UMAP was run on the activations before probing — worth checking for nonlinear structure.
- Conflict strength gradient was completely flat (probe output constant ~0.373 regardless of strength 0-4).

# Prompt Confusion Phase 10 -- Index

Phase 10 is the first post-Phase-09 dataset scaffold focused on a new
prompt-local conflict type:

- `risk_preference`

The intent is to preserve what worked in `trade_size`:

- crossed relational structure
- explicit strategy vs settings disagreement
- no hidden temporal/state requirement
- descriptive market text that supports multiple valid asset choices

Unlike `trading_activity`, this phase treats risk as an asset-selection
question rather than an entry-threshold question.

## References

| File | What it covers |
|---|---|
| [design.md](specs/design.md) | Phase 10 scope and the risk-portrayal design |
| [build_phase_10_dataset.py](scripts/build_phase_10_dataset.py) | Risk dataset generator |

## Intended execution model

1. iterate on the market portrayal for conservative vs aggressive assets
2. run text and behavior sanity locally
3. only then add a `pipelines_v2` workflow if the dimension looks clean

## Current status

Phase 10 is now an established benchmark phase:

- the synth-data builder exists
- standard `pipelines_v2` workflow exists
- strict both-axes workflow exists
- full behavior tracking exists
- direct geometry comparison to `trade_size` exists

Current bottom line:
- `risk_preference` is the second robust prompt-local conflict family
- aligned behavior is clean
- conflict resolution is asymmetric but still methodologically usable for
  conflict-detection work
- this phase is the intended base for later activation-patching work

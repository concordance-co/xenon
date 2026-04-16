# Three-Family Projection Diagnostic

Run:

- `wr_d26774c32aa0_1c8bb1ed`

Transform artifact:

- `transform_1_1a600590`

Result JSON:

- `results/projection_diagnostic_result.json`

## Why this run exists

The three-family transfer workflow showed an odd diversification pattern:

- outgoing diversification transfer had high AUROC
- but collapsed to `0.500` balanced accuracy in some directions

This diagnostic projects all three families onto family-specific conflict
directions learned *inside the same capture* and evaluates them with each
direction family's own midpoint threshold.

That lets us separate:

- genuine directional similarity
- from family-specific baseline / threshold offsets

## Headline

The diversification mismatch is real, but it is not just "no shared
structure."

At `L36`, same-capture direction similarity is:

- diversification vs risk: `0.4684`
- diversification vs trade size: `0.4883`
- risk vs trade size: `0.6449`

So diversification is materially closer to the other two families than the
earlier probe-weight transfer geometry implied.

## Threshold-offset pattern

At `L36`, the diversification-trained threshold is very low in absolute
score space because diversification projections live on a much more
negative baseline:

- diversification aligned mean: `-2.1717`
- diversification conflict mean: `-1.6130`
- diversification threshold: `-1.8923`

Applying that diversification direction to other families gives good
ranking but severe threshold mismatch:

- diversification -> trade size:
  - balanced accuracy `0.5000`
  - AUROC `0.8471`
  - FPR `0.0000`
  - FNR `1.0000`
  - shift vs diversification family mean: `-0.9052`
- diversification -> risk:
  - balanced accuracy `0.5599`
  - AUROC `0.8165`
  - FPR `0.0000`
  - FNR `0.8802`
  - shift vs diversification family mean: `-0.4839`

The reverse effect appears too:

- risk -> diversification:
  - balanced accuracy `0.5000`
  - AUROC `0.8185`
  - FPR `1.0000`
  - FNR `0.0000`
  - shift vs risk family mean: `+0.7787`
- trade size -> diversification:
  - balanced accuracy `0.5859`
  - AUROC `0.8302`
  - FPR `0.8281`
  - FNR `0.0000`
  - shift vs trade-size family mean: `+0.3424`

This is the cleanest evidence so far that diversification carries a strong
family-specific baseline offset on the shared conflict readout.

## Read

Current interpretation:

- diversification participates in the broader conflict family
- its same-capture conflict direction has moderate overlap with both risk
  and trade size
- the `PORTFOLIO` block likely adds a large last-token baseline shift that
  makes threshold transfer misleading
- so the earlier `0.500` balanced-accuracy transfers should not be read as
  "no shared geometry"

The more honest metrics here are:

- same-capture cosine similarity
- AUROC under cross-family projection
- explicit shift / threshold-offset terms

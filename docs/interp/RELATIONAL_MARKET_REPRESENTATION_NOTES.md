# Relational Market Representation Notes

Date: 21 March 2026

## Question

Phase 6 showed that single-row profile retrieval was weak under layout changes. The next question was whether that failure reflected a lack of market abstraction, or whether the wrong representation object was being tested.

The core alternative hypothesis:

- the model's market understanding is more relational than row-identitarian
- pairwise asset relations should survive nuisance changes better than single-row identity
- whole-snapshot geometry may survive as well, but more weakly

## Dataset

This analysis reuses the completed `phase6_profile_invariance_v1` synthetic capture set:

- 48 market-only prompts
- 2 scenario families
  - `momentum_flow_tiebreak`
  - `participation_concentration_tiebreak`
- nuisance factors already present in the dataset
  - style variation
  - symbol variation
  - layout / row-permutation variation

No new capture was needed. This was a new analysis pass on the pooled Phase 6 states.

## New Analysis Objects

### 1. Row identity

The old Phase 6 object:

- can a row-state retrieve the same latent profile under nuisance changes?

This stays useful as a failure baseline, but should not be treated as the main market representation claim.

### 2. Pairwise relation invariance

New object:

- construct pairwise difference vectors between latent profile pairs inside a snapshot
- ask whether the same relation is still nearest under nuisance changes

This is a better approximation to comparative market understanding:

- A better than B on a given factor tradeoff
- not "what is row A in isolation?"

### 3. Snapshot geometry

New object:

- concatenate profile-ordered row states into a whole-market geometry vector
- ask whether the same latent market geometry is nearest under nuisance changes

This is a weak but useful set-level read.

## Main Results

### Row identity under layout-only controls

- `momentum × flow`: margin `0.00635`, NN accuracy `0.667`
- `participation × concentration`: margin `0.01077`, NN accuracy `0.719`

Interpretation:

- single-row profile identity is barely stable under layout changes

### Pairwise relation invariance under layout-only controls

- `momentum × flow`: margin `0.12671`, NN accuracy `0.854`
- `participation × concentration`: margin `0.16242`, NN accuracy `0.882`

Interpretation:

- pairwise relations are dramatically more robust than row identity
- the gain over row identity is roughly:
  - `~20x` for `momentum × flow`
  - `~15x` for `participation × concentration`

### Snapshot geometry under layout-only controls

- `momentum × flow`: margin `0.00085`, NN accuracy `0.583`
- `participation × concentration`: margin `0.00093`, NN accuracy `0.667`

Interpretation:

- whole-snapshot geometry survives somewhat
- but it is substantially weaker than pairwise relation invariance

## What This Means

The important update is methodological and conceptual:

- row retrieval was a valid diagnostic
- but it was the wrong primary representation object
- the better current object is pairwise-relative market structure

The most defensible current claim is:

- primitive market factors are explicit in row states
- pairwise comparative structure survives nuisance variation much better than single-row identity
- the model's market understanding appears more relational than row-identitarian

## What This Does Not Yet Show

- a clean global market manifold
- strong layout-invariant row identity
- a fully stable whole-snapshot geometry
- transfer of the same relation structure to real DX data

## Best Next Step

Shift the synthetic track from row retrieval to relation-first tests:

- harder pairwise near-tie datasets
- rank-vs-magnitude controls at the pairwise level
- roster-composition shifts while holding latent pairwise relations fixed
- then validate the strongest relation-family in real DX rows

# Ranked Research Roadmap

## Main Read

The best path is no longer "behavior-first."

The main program should be:

1. determine what market variables the model represents cleanly
2. determine whether those variables are absolute, relative, or low-dimensional coupled factors
3. determine how settings and affordances transform that market representation downstream
4. use final action only as a validation target, not the primary object

This keeps the actionability work, but moves it into a supporting role.


## Ranked Tracks

### 1. Synthetic Market Representation

Why this is first:

- It aligns directly with the actual research goal: how the model understands the market.
- The last actionability experiments showed that fused end-state labels are often the wrong target.
- The strongest synthetic result so far is still the robust early market-side signal, not any end-state classifier.

Core hypotheses:

- Primitive market factors are represented more cleanly than final action labels.
- Pairwise and near-tie tradeoffs are more diagnostic than easy best-asset cases.
- Some market factors are absolute, while others are represented relative to the current roster.
- Settings and affordances transform an existing market representation rather than creating it from scratch.

First experiments:

- Build a harder synthetic market-only dataset with:
  - near-ties
  - factor tradeoffs
  - context-driven rank shifts
- Probe primitive market variables and pairwise relations on that dataset.
- Compare absolute-metric baselines against rank-based and pairwise representation probes.

Success criteria:

- At least one harder market dataset yields nontrivial but not trivial decodability.
- Pairwise or factor-level targets outperform naive best-asset framing as research objects.
- We can distinguish absolute vs roster-relative representation on the same fixed focal rows.


### 2. Causal Necessity of Market Variables

Why this is second:

- Once cleaner market variables are identified, the next gain comes from causal tests, not more passive decoding.

Core hypotheses:

- `pct_5m` and momentum × flow are necessary inputs to preference formation.
- Participation is mostly a confidence modulator.
- Concentration matters later or through policy rather than as a clean early perceptual variable.

First experiments:

- Patch or ablate row states along the strongest scalar and coupled directions.
- Run rank-preserving and magnitude-preserving corruptions on synthetic rows.
- Measure effects on best-asset and pairwise preference.


### 3. Real-Data Representation Validation

Why this is third:

- Synthetic market findings only matter if they transfer back to real DX-terminal-style prompts.

Core hypotheses:

- Primitive market variables and pairwise relations transfer better than pooled buy/sell probes.
- Real prompts preserve the same upstream market-side signal even when downstream action is noisy.
- Later sections sharpen actionability more than raw preference.

First experiments:

- Validate primitive factor probes and pairwise probes on real DX rows.
- Compare row states versus later sections after residualizing out simple heuristics.
- Use reruns only to validate downstream transformations of the market representation.


### 4. Actionability Factorization

Why this is fourth:

- The `v3/v4` synthetic results suggest that primitive affordance bits are cleaner than fused permission mode.
- This is mechanistically interesting, but it is still downstream of market understanding.

Core hypotheses:

- `can_buy`, `can_sell`, and `observe_vs_act` are represented more cleanly than fused `permission_mode`.
- Downstream permission is factorized rather than monolithic.


### 5. Policy vs Perception Routing

Why this is fifth:

- Routing may still provide a cleaner mechanistic handle if experts split along perception versus policy.

Core hypotheses:

- Some experts specialize in market parsing, others in policy and affordance handling.
- Settings-twist effects are concentrated in a narrower routing subset than market perception itself.


### 6. Geometry Support Track

Why this is sixth:

- Geometry is still useful, but it should simplify market-representation stories rather than define the main agenda.

Core hypotheses:

- A few variables admit low-dimensional geometric structure.
- The joint market state is built from several simpler pieces rather than one universal manifold.


## What To Deprioritize

- broad manifold search without a representation target
- more generic buy/sell probes as primary evidence
- treating action labels as the cleanest object
- more synthetic actionability breadth before testing harder market-only representation hypotheses


## Kickoff Status

The actionability work has still been useful, but it should now be treated as a supporting lane.

The current handoff point is:

- market-best asset remains trivially early on synthetic data
- fused downstream permission labels are weak under prompt hardening
- primitive affordance bits survive better than fused permission mode
- therefore the next concrete research step should return to market-only synthetic representation work, not more end-state probing

Specifically:

- build a harder synthetic market-only dataset with near-ties and rank-shift backgrounds
- use that to study primitive market factors and pairwise relations
- then bring only the strongest representation claims back to real DX data

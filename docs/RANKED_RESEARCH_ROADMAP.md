# Ranked Research Roadmap

## Main Read

The best path is no longer "manifold-first."

The main program should be:

1. determine what the model wants to do
2. determine what stops it
3. determine whether settings reweight that preference or merely gate execution
4. use geometry only when it simplifies that story

This keeps the manifold work, but moves it into a supporting role.


## Ranked Tracks

### 1. Blocked Valence + Settings Twist

Why this is first:

- It is the shortest path from localization to mechanism.
- It directly addresses the strongest unresolved questions from the decision-structure work:
  - is `observe` actually neutral?
  - are settings late gates or true reinterpretation operators?
- It creates the labels needed for a cleaner asset-valence program.

Core hypotheses:

- Early preference, late permission: market preference forms before legality gating.
- Observe contains hidden bullish and bearish states, not just neutral non-action.
- Settings reweight an existing preference space rather than creating preference from scratch.

First experiments:

- Build blocked-observe rerun cohorts stratified by block reason and actionability regime.
- Build settings-tension cohorts with legal action available but extreme settings pressure.
- Run deconstraint and settings-rewrite reruns and compare downstream states.

Success criteria:

- A nontrivial fraction of observe cases reveal blocked bullish or blocked bearish preference under rerun.
- Settings rewrites change downstream use of preference without destroying the underlying market-side signal.


### 2. Causal Necessity of Market Variables

Why this is second:

- The synthetic work has already surfaced the strongest candidates.
- The next gain comes from causal tests, not more passive decoding.

Core hypotheses:

- `pct_5m` and momentum × flow are necessary inputs to preference formation.
- Participation is mostly a confidence modulator.
- Concentration matters later or through policy rather than as a clean early perceptual variable.

First experiments:

- Patch or ablate row states along the strongest scalar and coupled directions.
- Run rank-preserving and magnitude-preserving corruptions on synthetic rows.
- Measure effects on best-asset and pairwise preference.


### 3. Real-Data Decision Decomposition

Why this is third:

- Synthetic findings only matter if they transfer back to real DX-terminal-style prompts.

Core hypotheses:

- Asset-conditioned valence transfers better than pooled buy/sell probes.
- Real observe cases split into neutral, blocked bullish, and blocked bearish subtypes.
- Late sections sharpen actionability more than raw preference.

First experiments:

- Train asset-valence probes on rerun-labeled blocked cases.
- Compare row states versus `active_settings_eos` and `constraints_eos`.
- Residualize out simple market heuristics and re-evaluate.


### 4. Policy vs Perception Routing

Why this is fourth:

- Routing may provide a cleaner mechanistic handle than generic residual-stream analysis if experts split along perception versus policy.

Core hypotheses:

- Some experts specialize in market parsing, others in policy and affordance handling.
- Settings-twist effects are concentrated in a narrower routing subset than market perception itself.


### 5. Geometry Support Track

Why this is fifth:

- Geometry is still useful, but it should simplify causal stories rather than define the main agenda.

Core hypotheses:

- A few variables admit low-dimensional geometric structure.
- The joint market state is built from several simpler pieces rather than one universal manifold.


## What To Deprioritize

- broad manifold search without a behavioral decomposition
- more generic buy/sell probes without blocked-valence labels
- treating `observe` as uniformly neutral
- adding more synthetic geometry breadth before testing the top behavioral hypotheses


## Kickoff Status

The research kickoff has already started around the top-ranked track:

- a live Neon audit of blocked-observe and policy-tension candidate pools
- a blocked-valence kickoff manifest
- a settings-twist kickoff manifest
- a report that quantifies why these tracks should outrank more manifold-first work

This is the handoff point for the next concrete research step:

- rerun the blocked-valence manifest under deconstraint or strategy removal
- rerun the settings-twist manifest under meaningful settings rewrites
- compare downstream state shifts before returning to broader geometry work

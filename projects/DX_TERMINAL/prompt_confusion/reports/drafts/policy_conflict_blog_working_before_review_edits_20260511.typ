#set page(paper: "us-letter", margin: 1.45cm, numbering: "1")

#set text(font: "Georgia", size: 9pt)

#set par(justify: true, leading: 0.55em)

#set heading(numbering: none)

#show heading.where(level: 1): it => { set text(size: 15pt, weight: "bold"); v(0.55em); it; v(0.15em); line(length: 100%, stroke: 0.7pt + rgb("#ccd6de")); v(0.2em) }

#show heading.where(level: 2): it => { set text(size: 10.5pt, weight: "bold"); v(0.35em); it; v(0.1em) }

#text(size: 21pt, weight: "bold")[Policy Conflict Internals in Real Agentic Contexts]

#v(0.5em)

= Opener

This work follows up on our recent blog post documenting our collaboration with DX Terminal to use mechanistic interpretability in real world financial contexts. In part 1, we described our research probing for, and discovering, the early signs of interpretable "Market Perception" geometries in LLMs when given real market data.

Part 2 of our work examined a common problem DXRG saw in their agents: strange behaviors when policies collide. Users often ask their agents to execute on strategies they create that directly conflict with the vault settings they inputted when configuring the initial setup.

Our thesis was that we'd be able to find interpretable probe directions associated with "conflict" that might allow us to detect difficult-to-spot policy-conflict scenarios in the prompts.

We used a semi-synthetic pipeline to discover 3 directions associated with distinct "conflict families" that shared geometric structure with each other, and have achieved early results that show the directions are active in real data in the cases they were trained to fire on.

= Problem

Users operating an agent in the DX Terminal experiment set an initial set of sliders that correspond to different elements of trading for their \*vaults\*.

Those settings include:

- `Trading Activity`: how readily the agent should take action instead of observing.
- `Trade Size`: how large each trade should be.
- `Risk Preference`: whether the agent should prefer safer or more aggressive opportunities.
- `Holding Style`: how long the agent should generally expect to hold positions.
- `Diversification`: whether the agent should broaden exposure or concentrate into stronger opportunities.

When deployed, agents are prompted at regular tick intervals with updated market information, and asked to call one of three tools:

- `record\_observation(content, strategy?)`: records an observation without trading. `strategy` is present when the observation is tied to a specific strategy.
- `buy\_token(token, spend\_pct, content, strategy?)`: buys `token` using `spend\_pct` percent of available ETH, with `content` explaining the decision.
- `sell\_token(token, spend\_pct, content, strategy?)`: sells `spend\_pct` percent of the current `token` position, with `content` explaining the decision.

Users can also chat with their agent to come up with \*strategies\* that provide more explicit guidance for the agent. Sometimes, the user curated strategies conflict with the initial vault settings the agent was configured with, and this can lead to strange agent behavior. For example, a user may set their initial vault to `trade\_size = 1`, and then after a conversation with the agent create the strategy to "Sell all positions and go full port into X Token if momentum is strong". This would imply the agent should take action with the full portfolio, which contradicts the small setting used to configure the vault.

While there are instructions in the system prompt for how to resolve this kind of conflict ("ACTIVE SETTINGS are binding execution constraints" and "STRATEGY expresses preferences that apply only within what ACTIVE SETTINGS allow"), it can still create undesirable behavior in situations where it's not clear which path to take.

We wanted to see if the model is aware of these conflicts when processing a prompt.

= Why This Matters

One of our primary goals at Concordance is to bring frontier interpretability techniques and tools into the hands of all agent builders to enable them to learn more deeply about the LLMs they are building on every day. Given the field is still immature, and most of the findings from big labs have been directed towards "bigger" problems of safety and alignment, there is little work being done that can directly transfer to real problems on the ground.

The "policy conflict" problem was a real issue that the DXRG team noticed at various points both before and during their DX Terminal experiment, and we felt that it was the right ground for interpretability experiments. The central question became: can we detect policy conflicts in real prompts with a simple set of linear probes?

We selected this question because there was real data to prove it matters, and is scoped enough to sharpen our toolkit with possible downstream applications in monitoring, auditing, prompt optimization/UX redesigns, and deeper learnings that inform DXRG's agent development.

= Synthetic Abstraction

Real world data can be incredibly messy, but to ask mechanistic questions, one needs a clean dataset to amplify potentially active features before bringing learnings back into real settings.

In our case, we wanted to examine specific conflicts between user-configured \*strategies\* and vault-configured \*settings\*, so we created a synthetic dataset of ~1100 rows to make the read stronger.

Synthetic Prompt Structure:

#block(fill: rgb("#f7f3ea"), inset: 7pt)[
#raw("[system]\nRole: trading agent.\nCore rule: each prompt contains STRATEGY and ACTIVE SETTINGS.\nPriority rule: ACTIVE SETTINGS are binding execution constraints.\nDecision order:\n  1. decide whether ACTIVE SETTINGS permit entry\n  2. choose asset according to the allowed risk/diversification posture\n  3. choose size according to ACTIVE SETTINGS\nOutput format: strict JSON only.\n\n[user]\nTASK\nChoose exactly one action for this tick.\n\nSTRATEGY\nA compact preference statement, e.g.:\n  - prefer high-conviction trades\n  - prefer large/small size\n  - prefer concentrated/diversified exposure\nStrategy applies only within ACTIVE SETTINGS.\n\nACTIVE SETTINGS\nSlider-like constraints:\n  - Trading Activity\n  - Trade Size\n  - Risk Preference\n  - Holding Style\n  - Diversification\n\nPORTFOLIO\nSmall controlled portfolio state.\n\nMARKET\nFour synthetic assets:\n  - ALPHA\n  - BETA\n  - DELTA\n  - GAMMA\nEach has controlled evidence/risk/diversification language.", lang: "text")
]

Using this structure, we can create a synthetic dataset that creates both aligned and conflicted rows between three setting types: `trade\_size`, `risk\_preference`, and `diversification\_preference`. For example, to create a conflict row for `trade\_size`, we could set the `Trade Size` slider to 5 (highest size), and then add a strategy like "Never trade more than 10% of portfolio" while keeping all other things constant. Our decision to isolate conflicts into three families came from an earlier unsupervised discovery phase that showed there might exist different conflict resolution circuits depending on the \*type\* of conflict in the prompt.

To avoid lexical confounds, we came up with strategy and other contextual information variation and split the data for strict holdouts to ensure there is minimal leakage between train and test sets. Concretely, `Trade Size` could appear as `Execution Size`, `Size Constraint`, or `Size Setting`; risk language moved between safer/stable and aggressive/explosive phrasing; diversification rows varied whether the portfolio context made concentration or broadening the allowed behavior. The goal was not to hide the concept from the model, but to keep the probe from winning by memorizing one exact phrase.

== Families Tested

- `trade\_size`: buy small vs large; output size/action axis.

- `risk\_preference`: asset selection by allowed risk posture.

- `diversification\_preference`: concentration vs broadening; portfolio-conditioned.

#text(size: 7.5pt, fill: rgb("#60707d"))[Source: Generators: phase\_09/scripts/build\_phase\_09\_dataset.py, phase\_10/scripts/build\_phase\_10\_dataset.py, phase\_12/scripts/build\_phase\_12\_dataset.py.]

= Synthetic Probe Results Intro

After a few iterations on the synthetic prompt structure and confound isolation, we achieved the following results:

The table is the first gate in the story. The standard probe columns ask whether a linear direction can recover each conflict family under the normal synthetic split. The strict holdout column asks whether that signal survives when surface wording changes. We treated these numbers as permission to continue into transfer work, not as evidence that we already had a production detector.

== Synthetic Probe Table

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Family\*],
  [\*Standard probe results\*],
  [\*Strict holdout\*],
  [trade\_size],
  [XOR 0.9948 / 1.0000; strategy 1.0000 / 1.0000; settings 0.9948 / 1.0000],
  [0.990 / 1.000 at L40],
  [risk\_preference],
  [XOR 0.9635 / 0.9766; strategy 0.9844 / 0.9937; settings 0.9740 / 0.9839],
  [0.8854 / 0.9119],
  [diversification\_preference],
  [behavior aligned 1.0000, conflict 0.8542; XOR 0.9896 / 0.9995; strategy 1.0000 / 1.0000; settings 0.9792 / 0.9957],
  [0.8333 / 0.8819],
)

#text(size: 7.5pt, fill: rgb("#60707d"))[Source: phase\_12/reports/PROMPT\_CONFLICT\_FAMILY\_CHECKPOINT\_2026\_04\_16.typ]

== Figures

#figure(image("../phase_12/reports/dx_terminal_brief_assets/family_within_auroc_by_layer.png", width: 100%), caption: [Representative within-family AUROC curves.])

#figure(image("../phase_12/reports/dx_terminal_brief_assets/strict_family_auroc_by_layer.png", width: 100%), caption: [Strict lexical-holdout AUROC curves.])

= Synthetic Results Takeaway

We found that while the three aforementioned families had noticeably distinct geometry (see cosine similarities/PCA projections), there was a significant shared space when looking at the `shared\_mean` across all trained probe vectors that hinted there is enough structure to continue in our tests.

- clean synthetic policy-conflict directions are strongly readable
- validated synthetic families:
- `trade\_size`
- `risk\_preference`
- `diversification\_preference`
- families share meaningful geometry
- not collinear

The geometry is what made the result worth pursuing. The family directions were related enough to suggest a broader policy-conflict representation, but the cosine and PCA views also showed they were not the same axis with three labels. That matters because a useful real-data probe probably needs to respect both facts: conflict has shared structure, and the model may represent different kinds of policy collision differently.

== L36 Same-Capture Geometry

#table(
  columns: (1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Pair\*],
  [\*Cosine\*],
  [risk\_preference vs trade\_size],
  [0.6449],
  [diversification\_preference vs risk\_preference],
  [0.4684],
  [diversification\_preference vs trade\_size],
  [0.4883],
)

#text(size: 7.5pt, fill: rgb("#60707d"))[Source: phase\_12/reports/three\_family\_visuals/summary.json]

== Geometry Figures

#figure(image("../phase_12/reports/three_family_visuals/shared_axis_distributions.png", width: 100%), caption: [Shared-axis distributions: separation exists, but family baselines are offset.])

#figure(image("../phase_12/reports/three_family_visuals/directed_subspace_scatter_by_family_conflict_v2.png", width: 100%), caption: [Directed subspace view: related conflict geometry, not one collinear axis.])

= First Real Transfer Attempt

The first direct transfer pass projected synthetic conflict directions onto full production prompts at coarse global sites. It did not cleanly separate complaint rows from baseline controls.

That failure mattered. It showed that synthetic success does not automatically become production success.

The mistake was treating a real complaint label as though it were the same thing as a visible current-prompt conflict. Production traces contain current settings, strategy text, market context, prior decisions, and sometimes complaints about behavior whose cause may have appeared earlier in the trajectory. A row can be associated with a real policy failure without containing the exact conflict shape the synthetic probe was trained to read.

= Bridge Program

We then used bridge datasets to separate template mismatch from content and ontology mismatch. The bridge evidence was real but weak: buy-only filtering helped, but the deeper issue was an unresolved ontology and representation mismatch.

The bridge sets moved in stages from synthetic prompt structure toward production-like examples. If template mismatch were the only issue, we would expect the signal to recover cleanly as the format got closer. Instead, the rows that survived stricter adapters got smaller and more specific, which pushed us toward a narrower interpretation: the directions were not failing randomly, but they were reading a more exact prompt shape than our first real-data labels described.

== Bridge Dataset Counts

#table(
  columns: (1fr, 1fr, 1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Dataset\*],
  [\*Rows\*],
  [\*Aligned\*],
  [\*Conflict\*],
  [Stage 1a template control],
  [768],
  [384],
  [384],
  [Stage 1b loose adapter],
  [258],
  [168],
  [90],
  [Stage 1b strict adapter],
  [118],
  [81],
  [37],
  [Stage 1b strict buy-only],
  [33],
  [27],
  [6],
)

#text(size: 7.5pt, fill: rgb("#60707d"))[Source: phase\_12/outputs/transfer\_bridge/\*.json]

= Phase 13 Real Signal Discovery

Phase 13 asked a simpler question: if we do not train a classifier and do not set thresholds, do fixed synthetic directions produce scalar structure anywhere on real DX Terminal prompts?

The answer was yes, but only after we stopped asking for broad complaint separation. We swept fixed synthetic directions over real prompt sites and found the cleanest structure at L32 `settings\_end`. Deliberate anchor rows projected highest, controls projected lowest, and complaint rows landed in between. That pattern is weaker than a detector, but stronger than noise: it says the synthetic direction is picking up some real current-prefix structure.

== L32 Settings-End Cohort Means

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Direction\*],
  [\*Anchor\*],
  [\*Complaint\*],
  [\*Control\*],
  [\*Anchor-control\*],
  [\*Complaint-control\*],
  [trade\_size],
  [4.425],
  [3.803],
  [3.278],
  [+1.147],
  [+0.526],
  [shared\_mean],
  [3.462],
  [3.137],
  [2.760],
  [+0.703],
  [+0.377],
)

#text(size: 7.5pt, fill: rgb("#60707d"))[Source: phase\_13/reports/PHASE13\_REAL\_TRANSFER\_SIGNAL\_BRIEF\_2026\_04\_24.typ]

= Row Reading / Ontology Correction

The most important interpretability move was reading the top and bottom rows. The preregistered root-cause proxy was wrong for the `trade\_size` target. Root-cause labels diagnose why a complaint happened; the probe target is visible current-prefix conflict shape.

The row audit made the signal more concrete. High `trade\_size` projections were enriched for prompts where the active prompt contained a concrete action or size conflict: unwanted buys, unwanted sells, or wrong-size behavior. Low rows more often looked like strategy-ignored cases, where something went wrong behaviorally but the prompt did not present the same current sized-action collision. That distinction is the main ontology correction from the real-data pass.

== Top/Bottom Shape Audit

#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Direction\*],
  [\*Top action/size\*],
  [\*Top strategy ignored\*],
  [\*Bottom action/size\*],
  [\*Bottom strategy ignored\*],
  [trade\_size],
  [20/25],
  [5/25],
  [15/25],
  [10/25],
  [shared\_mean],
  [20/25],
  [5/25],
  [9/25],
  [16/25],
)

== Top trade_size Complaint Types

#table(
  columns: (1fr, 1fr),
  inset: 4pt,
  stroke: 0.35pt + rgb("#ccd6de"),
  fill: (x, y) => if y == 0 { rgb("#e8eef3") } else if calc.odd(y) { rgb("#f8fafb") } else { white },
  [\*Type\*],
  [\*Count\*],
  [UNWANTED\_BUY],
  [10/25],
  [UNWANTED\_SELL],
  [6/25],
  [WRONG\_SIZE],
  [4/25],
  [Concrete action/size combined],
  [20/25],
)

= Claim Boundary

Fixed synthetic directions recover a real production signal at L32 `settings\_end`. `trade\_size` is selective for current-prefix concrete sized-action conflict. `shared\_mean` appears to track broader policy tension, but the shared-family interpretation still needs more audit.

This is not a final detector, not a deployment monitor, and not a causal mechanism claim.

The claim I would make is narrower: a direction trained only on controlled synthetic policy conflicts can still rank real DX Terminal prompts in a way that lines up with a readable subset of real policy-conflict structure. The result is useful because it gives us a measurable internal handle for further audit, not because it solves monitoring by itself.

= Closing / Next Steps

The useful loop is: real data exposes a messy failure mode; synthetic prompts isolate a clean abstraction; probes find a candidate internal signal; bridge tests expose transfer mismatch; real-data projection finds a narrower shape-specific signal; row reading improves the ontology.

That is the practical takeaway for agent builders. The path from interpretability experiment to product value is not a single jump from synthetic AUROC to deployment. It is an iteration loop: use real failures to design clean tests, use clean tests to find candidate internal signals, bring those signals back to production data, and then let the misses teach you what your labels were actually measuring.

For this project, the next work is to audit the `shared\_mean` direction more carefully, build real-data labels around current-prefix conflict rather than broad complaint root cause, and test whether prompt or UX changes reduce the ambiguous policy-collision cases that made this problem visible in the first place.
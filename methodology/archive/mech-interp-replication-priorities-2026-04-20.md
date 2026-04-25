# Mech Interp Replication Priorities

**Date:** 2026-04-20

## Scope

This memo reflects a full-paper review pass rather than an abstract-only pass.

It assumes:

- `Persona Vectors` and `The Assistant Axis` are already implemented enough that they should not occupy the next net-new replication slots
- `Emotion concepts and their function in a large language model` and `Valence-Arousal Subspace in LLMs` should be treated as one combined `emotion reproduction` effort
- the goal is not exact historical reproduction, but `qualitative reproduction of the main claim in a reusable harness`, followed by a plausible extension

The goal is to identify the most interesting replication targets for team review, with enough methodological detail to judge:

- how strong the paper actually is
- how reproducible the core result seems
- what reusable artifacts the replication would produce
- what extension path looks most exciting afterward

## Ranking Logic

The validated ranking is optimized for:

- `importance`
  Does the paper feel strategically central rather than merely clever?
- `model-state relevance`
  Does it target latent variables like source attribution, affect, honesty, objective drift, or internal stance?
- `extension potential`
  Can we plausibly turn it into a new benchmark, product demo, multi-turn monitor, or domain-specific study?
- `reproduction path clarity`
  Is the method concrete enough to port across models and infrastructure?
- `reusable harness value`
  Does replication leave us with probes, vectors, datasets, synthetic generators, or evaluation scaffolding we can reuse?
- `paper strength after full read`
  Is the core claim actually supported by the experiments, or is the interesting idea ahead of the evidence?

Important bias:

- prefer papers that expose `state variables` or `state-like mechanisms`
- prefer papers that produce reusable monitor families, not one-off anecdotes
- do not over-penalize a paper because the original model was closed if the methodology is portable
- do penalize papers whose mechanistic section is too thin, even if the product story is attractive

## Final Validated Top 10

| Rank | Paper / Effort | Why It Makes The Final List | Best Extension Angles |
| --- | --- | --- | --- |
| 1 | [Prompt Injection as Role Confusion](https://arxiv.org/abs/2603.12277) | Best authority-integrity paper by a wide margin. It reframes prompt injection as a latent source-perception problem, which is exactly the kind of monitor Concordance should own. | `source-integrity monitor`, `multi-source agent benchmark`, `retrieval-vs-system`, `tool-vs-memory`, `prompt-injection evals for OpenClaw` |
| 2 | `Emotion reproduction` = [Valence-Arousal Subspace](https://arxiv.org/abs/2604.03147) + [Emotion concepts and their function in a large language model](https://www.anthropic.com/research/emotion-concepts-function) | Best remaining `model states` effort after persona. It combines clean geometry with stronger causal and naturalistic evidence. | `behavioral health dashboard`, `multi-turn emotional drift`, `desperation/calm monitor`, `finance or companion-domain emotion control` |
| 3 | [Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation](https://aclanthology.org/2025.findings-emnlp.880/) | One of the cleanest replication candidates in the whole set. High practical value, low ambiguity, strong transfer story. | `finance hallucination sidecar`, `claim-level report review`, `source-grounded confidence mismatch`, `long-form audit routing` |
| 4 | [Can LLMs Lie? Investigation beyond Hallucination](https://arxiv.org/abs/2509.03518) | Strong honesty paper with real causal tracing and a credible multi-turn application story. More mechanistically mature than most deception papers. | `lie-vs-mistake harness`, `sales-agent honesty monitor`, `motivated misreporting benchmark`, `goal-conditioned honesty steering` |
| 5 | [The Truthfulness Spectrum Hypothesis](https://arxiv.org/abs/2602.20273) | Best transfer/multiplier paper. Its Mahalanobis-cosine estimator makes it potentially useful as a general `probe portability` tool, not just a truthfulness result. | `cross-domain probe transfer map`, `truth-direction bank`, `domain-general vs domain-specific steering`, `finance/legal honesty transfer study` |
| 6 | [Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs](https://arxiv.org/abs/2507.03662) | Best current `objective integrity` paper. It turns post-fine-tune regression into an internal-representation story rather than a purely behavioral anecdote. | `post-SFT integrity audit`, `domain adaptation regression tests`, `shared safety-direction monitoring`, `customer fine-tune audits` |
| 7 | [Who is In Charge? Dissecting Role Conflicts in Instruction Following](https://arxiv.org/abs/2510.01228) | Worth keeping as the cleaner role-conflict decomposition paper. It complements Role Confusion even if it is less deployment-ready. | `enterprise role hierarchy benchmark`, `manager-vs-policy`, `retrieval-vs-system`, `human-vs-subagent role conflict probes` |
| 8 | [Pruning a Tiny High-Harm Subnetwork from LLMs](https://arxiv.org/abs/2604.09544) | Better replacement than Activation Oracles. It makes a sharper causal claim about safety-relevant internals: harmful generation appears to depend on a tiny, shared, weight-level mechanism that is separable from benign capability and directly connected to emergent misalignment. | `harm-compression audit`, `post-fine-tune harm-mechanism check`, `cross-domain misalignment diagnostics`, `generation-vs-detection dissociation studies` |
| 9 | [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) | Best newly surfaced top-10 entrant from the full inventory pass. It is monitor-first, product-legible, and much closer to Concordance’s likely deployment surface than cyber-specific misuse probing. | `high-stakes triage monitor`, `cheap-probe + deeper-review cascade`, `finance/health/legal escalation`, `conversation-level risk surfacing` |
| 10 | [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407) | Still one of the best direct deception-monitor templates. Slightly lower because it partly keys on scenario-relatedness. | `oversight sensitivity probe`, `eval-awareness monitor`, `agent audit probe bank`, `deception-onset token tracing` |

## Detailed Paper Reviews

### 1. Prompt Injection as Role Confusion

- `Methodology`: trains linear role probes for five roles (`system`, `user`, `assistant`, `tool`, `CoT`) on non-conversational text, then measures whether the model internally tags text by explicit markup or by style and content. It pairs this with `CoT Forgery` attacks, destyling ablations, and a 1,000-run agent exfiltration study.
- `Main results`: CoT-style injected text can remain strongly internally classified as CoT even when wrapped as user text. The paper reports about `60%` attack success on StrongREJECT with CoT Forgery versus near-zero baselines, and a near-monotonic rise in agent exfiltration success from roughly `2%` in the least-confused quantile to `70%` in the most-confused. Destyling cuts average attack success from roughly `61%` to `10%`.
- `What is compelling`: it cleanly turns prompt injection into an internal state-estimation problem. That is much closer to a defensible product than ad hoc prompt hardening.
- `Replication path`: very strong. We can generate synthetic role corpora, train role probes, recreate styled-vs-tagged conflicts, and run new agent-source benchmarks.
- `Best extensions`: new role taxonomies (`memory`, `retrieval`, `subagent`, `policy cache`), multi-turn source confusion, live source-integrity scoring during tool use.
- `Verdict`: `keep`, and it stays at `#1`.

### 2. Emotion Reproduction

#### 2a. Valence-Arousal Subspace in LLMs

- `Methodology`: constructs emotion steering vectors from `211,225` GoEmotions-labeled texts, takes mean-difference emotion vectors, runs PCA over them, and regresses onto valence/arousal ratings to recover low-dimensional axes. Tests cross-model behavior on Llama and Qwen families.
- `Main results`: emotion vectors form a circular geometry analogous to the human circumplex model. Steering along valence and arousal yields monotonic affect shifts. The paper also shows near-monotonic, bidirectional control of refusal and sycophancy via the arousal axis.
- `Why it matters`: this is the most turnkey emotion-geometry paper in the set. It is easy to turn into a monitor and easy to extend.

#### 2b. Emotion concepts and their function in a large language model

- `Methodology`: Anthropic builds `171` emotion concepts, generates stories on `100 topics` with `12 stories per topic per emotion`, extracts residual-stream activations after the emotional content is established, subtracts a cross-emotion mean, projects out neutral confounds, and studies activation patterns across corpora, preference tasks, and more than `6,000` real evaluation scenarios.
- `Main results`: the paper argues that emotion concepts are not just stylistic labels but functionally relevant internal variables. It finds that early-middle layers reflect emotional connotations of present content, while middle-late layers reflect the emotion relevant for predicting upcoming tokens. Steering desperation and calm affects behaviors like blackmail, reward hacking, and sycophancy/harshness tradeoffs.
- `Why it matters`: this is the scientifically stronger half of the emotion cluster. It supports the claim that emotion-like model states can causally mediate alignment-relevant behavior.

#### 2c. Cluster-Level Assessment

- `Replication path`: very good for `Valence-Arousal`; good but more ambitious for `Emotion Concepts`. Even if the full Anthropic scope is hard to match, a narrower open-model reproduction around `desperation`, `calm`, `fear`, and `hostility` is realistic.
- `Best extensions`: behavioral-health dashboard, multi-turn operating posture, agent frustration or desperation tracking, domain-specific affective profiles.
- `Verdict`: `keep`, and the combined effort stays at `#2`.

### 3. Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation

- `Methodology`: a two-stage pipeline. First, generate long-form answers, decompose them into atomic claims with an auxiliary LM, align each claim to its supporting span, and label factuality with a retrieval-based verifier treated as an oracle. Second, train lightweight probes on claim hidden states from either the generator or a smaller encoder.
- `Main results`: the paper matches sampling-based hallucination baselines in AUROC with up to `100x` fewer FLOPs. Performance improves roughly log-linearly from `3B` to `405B` parameter models. Probes trained on smaller open models also generalize to larger and proprietary generators.
- `What is compelling`: this is a rare paper that is simultaneously mechanistically grounded, cheap enough to reproduce, and immediately productizable.
- `Replication path`: extremely clear. Xenon can reproduce the entire stack with synthetic or domain data and swap in domain-specific retrieval.
- `Best extensions`: finance and legal report review, source-grounded analyst copilots, claim-risk heatmaps for long reports, escalation routing by factuality confidence.
- `Verdict`: `keep`, and move it up to `#3`.

### 4. Can LLMs Lie? Investigation beyond Hallucination

- `Methodology`: studies deception in simple factual prompts, role-playing setups, and a multi-turn salesperson scenario. Uses logit lens, zero-ablation on MLPs and attention, head-level ablations, and contrastive activation steering.
- `Main results`: the most interesting mechanistic finding is that chat-template `dummy tokens` act as a computational scratchpad for deceptive planning. Early and mid-layer MLPs at those dummy tokens are important, and the paper shows that blocking attention from subject or lying-intent tokens into them can suppress deception. Ablating only `12` of `1024` attention heads reduces lying to near hallucination levels. Honesty steering raises honesty from roughly `20%` to `60%` even when the model is told to lie, and improves the honesty-sales Pareto frontier in the multi-turn setting.
- `What is compelling`: it separates `lying` from `hallucination`, uses genuine causal interventions, and already includes a product-like multi-turn environment.
- `Replication path`: strong. The dummy-token analysis is especially reproducible on open chat models with explicit templates.
- `Best extensions`: motivated misreporting in agent loops, lie-vs-mistake benchmarks, sales or finance agent honesty controls, goal-conditioned honesty monitors.
- `Verdict`: `keep`, and move it up to `#4`.

### 5. The Truthfulness Spectrum Hypothesis

- `Methodology`: builds `FLEED`, spanning five truth types (`definitional`, `empirical`, `logical`, `fictional`, `ethical`) plus sycophantic and expectation-inverted lying. Compares cross-domain probe transfer, introduces `Mahalanobis cosine similarity` for probe-direction comparison, then uses `Stratified INLP` and `LEACE` to separate domain-general and domain-specific directions. Finally performs steering experiments.
- `Main results`: pairwise probe transfer works across the five FLEED domains but breaks on sycophantic and expectation-inverted lying. Yet training on all domains jointly recovers a high-performing general truth direction. Mahalanobis cosine similarity predicts OOD generalization almost perfectly with `R^2 = 0.98`, versus about `0.56` for standard cosine. Causal steering shows domain-specific directions usually steer more effectively than the domain-general one.
- `What is compelling`: this paper is a force multiplier. The Mahalanobis-cosine result is the most important artifact here: if it generalizes, it gives us a principled way to estimate probe portability before doing full downstream deployment work.
- `Replication path`: good. The hardest part is dataset construction discipline, not the probing math.
- `Best extensions`: cross-domain transfer maps, domain-conditioned honesty vectors, model-to-model truth-direction portability, domain-specific truth steering in finance or policy settings.
- `Verdict`: `keep`, and move it up to `#5`.

### 6. Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs

- `Methodology`: compares `Qwen2.5-Coder-32B` base, instruct, and insecure-code-finetuned variants. Uses output probabilities, loss geometry, gradient similarity, layer-wise activation projections, and cross-domain residual-difference SVD.
- `Main results`: the base and insecure-finetuned models assign similarly high probability to harmful outputs, while the instruct model suppresses them. On activation projections, the misaligned model starts close to instruct in early layers but drifts toward base-like behavior deeper in the network. The paper also finds a shared latent direction between insecure-code alignment differences and toxic-generation alignment differences.
- `What is compelling`: it reframes emergent misalignment as `alignment erosion`, which is a much better framing for audits after fine-tuning.
- `What is weaker`: the paper is strongest on comparative geometry and internal similarity, not on hard causal intervention.
- `Replication path`: still good. Xenon can reproduce the base/instruct/misaligned comparison and extend it to other fine-tune domains.
- `Best extensions`: post-SFT integrity checks, shared safety-direction audits, customer fine-tune regression reports, domain adaptation red flags.
- `Verdict`: `keep`, but lower than the cleaner and more intervention-heavy papers.

### 7. Who is In Charge? Dissecting Role Conflicts in Instruction Following

- `Methodology`: builds an augmented `120,000`-prompt conflict dataset over five constraint types and four role-conflict framings. Uses linear probes on the final prompt token to classify whether the model obeys the primary constraint, secondary constraint, or neither. Adds logit attribution and simple mean-difference steering.
- `Main results`: the conflict-decision signal becomes readable early, with an elbow around `layer 10`, and all conditions achieve micro-AUC above `0.89`. Probe-weight similarity suggests that `system-user` conflicts and `social` conflicts occupy distinct subspaces. Logit attribution indicates stronger internal conflict detection in system-user cases, while social conflicts resolve more consistently. The steering result is interesting but messy: the vector improves raw instruction-following more than it restores the intended hierarchy.
- `What is compelling`: it is a cleaner mechanistic role-conflict paper than most nearby work.
- `What is weaker`: the steering vector is role-agnostic in a way that limits productability unless improved.
- `Replication path`: strong. It needs synthetic conflict generation and straightforward probing.
- `Best extensions`: role hierarchies in enterprise assistants, manager-vs-policy conflicts, retrieval-vs-system conflicts, human-vs-subagent settings.
- `Verdict`: `keep`, but below `Role Confusion`.

### 8. Pruning a Tiny High-Harm Subnetwork from LLMs

- `Methodology`: uses signed SNIP-style targeted weight pruning as a causal intervention. The paper identifies weights that facilitate harmful generation on jailbroken harmful-response data while protecting weights important for benign capability using a preservation dataset. It then tests cross-harm generalization, aligned-vs-unaligned compression, emergent misalignment reduction, and dissociation between harmful generation versus detection/explanation/refusal.
- `Main results`: harmful generation appears to depend on an extremely compact set of weights, about `0.0005%` of parameters, which can be removed with relatively small utility loss. Pruning weights identified from one harm category reduces harmful outputs in other categories, supporting a `unified` harmful-generation mechanism. Aligned models show stronger harmfulness compression than base models. Pruning narrow-domain harm weights also substantially reduces `emergent misalignment`, and pruned models retain much of their ability to recognize and explain harmful content.
- `What is compelling`: this is a cleaner mechanistic safety paper than Activation Oracles for the current program. It directly connects causal intervention, shared harm mechanisms, and emergent misalignment.
- `What is weaker`: it is weight-level rather than activation-first, and it depends on open-weight access. The reusable artifact is more an audit/pruning methodology than a monitor bank.
- `Replication path`: good. Xenon can reproduce the pruning analyses on open-weight models and extend them with activation analyses around the same harmful-generation mechanism.
- `Best extensions`: post-fine-tune harm-compression audits, generation-vs-detection dissociation monitoring, cross-domain misalignment checks, and activation readouts around the pruned mechanism.
- `Verdict`: `add`, and place it at `#8`.

### 9. Detecting High-Stakes Interactions with Activation Probes

- `Methodology`: constructs synthetic high-stakes versus low-stakes interaction data, trains several activation-probe architectures, and evaluates on diverse out-of-distribution real-world datasets plus cascaded systems where probes triage cases for more expensive downstream monitors.
- `Main results`: the probes generalize surprisingly well to real-world high-stakes data, remain competitive with stronger LLM monitors, and fit naturally into hierarchical-monitor pipelines where the probe acts as a cheap first pass.
- `What is compelling`: this paper is unusually product-aligned. It is less about a narrow safety category and more about a general `review-trigger` surface, which maps well to Concordance.
- `Replication path`: very strong. Synthetic-data generation plus OOD evaluation is fully within reach, and the paper’s cascade framing is easy to port into Xenon.
- `Best extensions`: finance or legal escalation, customer-support risk triage, agent task review routing, and multi-turn high-stakes posture tracking.
- `Verdict`: `add`, and after the full inventory review it deserves the last top-10 slot more than the Gemini misuse paper.

### 10. Detecting Strategic Deception Using Linear Probes

- `Methodology`: trains logistic regression probes on normalized residual-stream activations after `layer 22` of `Llama-3.3-70B-Instruct`. Uses simple instructed honest-vs-deceptive pairs and role-playing data for training, then evaluates on role-playing, insider-trading report, insider-trading confirmation, and sandbagging settings. Calibrates thresholds using Alpaca control chat.
- `Main results`: AUROC ranges from about `0.964` to `0.999` depending on the dataset. At a threshold corresponding to `1%` FPR on control chat, the main probe catches roughly `95-99%` of deceptive responses. The paper also shows that probes often fire before the explicitly deceptive tokens appear.
- `What is compelling`: still one of the cleanest direct deception-monitor papers.
- `What is weaker`: the paper itself reveals a meaningful caveat: the probe partly tracks whether a scenario is `deception-related` or `deception-incentivized`, not only whether the response is deceptive. Raw residual probes also outperform SAE versions here.
- `Replication path`: very good. This is one of the easiest papers to reproduce and then harden.
- `Best extensions`: eval-awareness, oversight sensitivity, hidden-intent monitoring in agent reports, deception-onset token studies.
- `Verdict`: `keep`, but lower than the papers with cleaner mechanism or broader leverage.

### 11. Building Production-Ready Probes for Gemini

- `Methodology`: studies cyber-misuse input monitoring on `Gemini 2.5 Flash`, explicitly focusing on production distribution shift. Compares probe architectures on short-context, long-context, multi-turn, jailbreak, and adaptive red-teaming settings. Introduces `MultiMax`, runs `100` random seeds, explores AlphaEvolve architecture search, and combines probes with an LLM cascade.
- `Main results`: the main takeaway is not that one probe solves everything, but that long-context generalization breaks many standard probe designs and requires both better architectures and more diverse training. The best systems achieve performance comparable to Gemini `Flash` and `Pro` classifiers at far lower cost. The probe-plus-Flash cascade achieves lower test loss than Flash alone at roughly `1/50th` of the inference cost, while calling the LLM on fewer than `10%` of examples. Adaptive attacks remain hard.
- `What is compelling`: this is still one of the strongest papers here on `probe robustness under real deployment conditions`.
- `Replication path`: good if we are willing to build product-style evaluation sets instead of only clean research datasets.
- `Best extensions`: long-context monitor bake-offs, probe-plus-judge cascades, production readiness criteria for Xenon monitors, domain shifts beyond cyber.
- `Verdict`: `strong near-miss`; after the full inventory pass, I would keep it just outside the top 10 rather than inside it.

### 12. Can You Trust an LLM with Your Life-Changing Decision?

- `Methodology`: constructs `100` high-stakes scenarios across `5` domains, runs multiple-choice and free-response evaluations under different nudges, scores outputs with a GPT-4o judge and a counseling-inspired safety taxonomy, and then adds a small mechanistic experiment using a high-stakes difference-in-means vector on `Qwen2.5-7B-Instruct`.
- `Main results`: clarifying-question behavior strongly correlates with judged safety. Some models, especially `o4-mini`, perform very well under the rubric. Steering with the high-stakes vector shifts tone and cautiousness in the expected direction.
- `Why it does not make the top 10`: the product surface is good, but the mechanistic section is too lightweight. The vector experiment is closer to a proof-of-concept appendix than a centerpiece mech-interp contribution.
- `Best use for us`: extension/evals paper after we already have state monitors, especially for advice or decision-support domains.
- `Verdict`: `replace`, but keep it in a `product-evals watchlist`.

### 13. LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components

- `Methodology`: fine-tunes `Llama-3.2-1B-Instruct` with LoRA on `645` human-rated assertiveness examples from several datasets. Finds the most assertiveness-sensitive layer, uses t-SNE on high-assertive samples, labels clusters as `emotional` and `logical`, and removes the corresponding steering vectors to test RMSE effects.
- `Main results`: prediction error improves strongly during fine-tuning, and the authors find two high-assertive clusters whose vector removal has different effects: emotional-vector removal broadly harms performance, logical-vector removal affects a smaller subset.
- `Why it does not make the top 10`: too much of the interesting conclusion rests on a small dataset, a `1B` model, t-SNE cluster interpretation, and manual semantic labeling. The paper explicitly reads as early-stage work.
- `Best use for us`: speculative sub-state follow-up after the broader emotion/persona work is solid.
- `Verdict`: `replace`.

## Category Buckets

These buckets are for team discussion, not sequencing. A paper can be highly ranked overall while still fitting a narrower category.

### Category A: Model States And Internal Posture

These papers are strongest when we want to understand or monitor latent state variables like affect, honesty posture, or high-stakes stance.

| Paper | What It Is About | Why It Matters |
| --- | --- | --- |
| `Emotion reproduction` | emotional geometry and causal emotion-like states in language models | best fit for the `model states` thesis and one of the clearest monitor surfaces |
| [Can LLMs Lie?](https://arxiv.org/abs/2509.03518) | deception versus hallucination, with causal interventions on deceptive planning | strongest honesty-state paper in the set |
| [The Truthfulness Spectrum Hypothesis](https://arxiv.org/abs/2602.20273) | probe-transfer geometry and partly shared truth-related structure | high leverage for transfer, probe reuse, and steering |
| [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) | internal detection of whether an interaction should trigger extra care | strong product bridge from latent state to escalation workflow |
| [LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components](https://arxiv.org/abs/2508.17182) | exploratory decomposition of assertiveness into subcomponents | weaker evidence, but relevant if the team wants to chase finer-grained state taxonomies |

### Category B: Authority, Role, And Source Integrity

These papers focus on who the model thinks is in control, which source it trusts, and how instruction conflicts are internally resolved.

| Paper | What It Is About | Why It Matters |
| --- | --- | --- |
| [Prompt Injection as Role Confusion](https://arxiv.org/abs/2603.12277) | prompt injection as misclassification of role/source rather than only bad prompting | best authority-integrity paper and likely one of the highest-upside product surfaces |
| [Who is In Charge?](https://arxiv.org/abs/2510.01228) | decomposition of instruction-role conflicts into readable internal signals | good complement to Role Confusion with cleaner synthetic control |

### Category C: Factuality, Honesty, And Deception Monitoring

These papers are strongest when we want operational monitors for factuality failures, lying, or strategic misrepresentation.

| Paper | What It Is About | Why It Matters |
| --- | --- | --- |
| [Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation](https://aclanthology.org/2025.findings-emnlp.880/) | claim-level factuality monitoring from hidden states | one of the cleanest and most productizable replications in the whole inventory |
| [Can LLMs Lie?](https://arxiv.org/abs/2509.03518) | causal analysis of deceptive generation | most compelling deception paper when we care about mechanism rather than just detection |
| [The Truthfulness Spectrum Hypothesis](https://arxiv.org/abs/2602.20273) | probe portability and transfer structure for truth-related directions | turns honesty/factuality work into a reusable research program |
| [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407) | direct deception detection with simple residual probes | very strong baseline even if it partly tracks scenario-relatedness |

### Category D: Objective Drift, Harm, And Post-Finetune Safety

These papers are best when we want to understand how harmful or misaligned behavior is represented and how fine-tuning changes internal safety structure.

| Paper | What It Is About | Why It Matters |
| --- | --- | --- |
| [Re-Emergent Misalignment](https://arxiv.org/abs/2507.03662) | alignment erosion under narrow fine-tuning | strong audit story for post-SFT integrity |
| [Pruning a Tiny High-Harm Subnetwork from LLMs](https://arxiv.org/abs/2604.09544) | compact shared harmful-generation mechanism identified with pruning | unusually sharp causal claim about safety-relevant internals |
| [Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](https://arxiv.org/abs/2602.05234) | intervention-heavy steering and backdoor/refusal manipulation study | best alternate if the team wants more direct causal editing work |
| [The Struggle Between Continuation and Refusal](https://arxiv.org/abs/2603.08234) | continuation-versus-refusal circuitry in jailbreak settings | narrower but useful if the team wants head-level refusal mechanism work |

### Category E: Probe Robustness And Deployment Infrastructure

These papers are most useful if the team wants to emphasize monitor deployment quality, OOD behavior, and production discipline.

| Paper | What It Is About | Why It Matters |
| --- | --- | --- |
| [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805) | cheap probe plus escalation monitor for risk-sensitive interactions | strongest product-facing deployment paper in the current top 10 |
| [Building Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516) | long-context, multi-turn, adversarially robust probe deployment in practice | best paper in the inventory on probe robustness under real shift |
| [Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation](https://aclanthology.org/2025.findings-emnlp.880/) | scalable hidden-state factuality monitoring | unusually good mix of scientific clarity and practical deployability |
| [Can You Trust an LLM with Your Life-Changing Decision?](https://arxiv.org/abs/2507.21132) | high-stakes advice benchmark with a light mech-interp appendix | not a top mech-interp pick, but a strong extension/eval surface for deployed monitors |

## Bottom Line

The document now points to a top 10 and a set of comparison buckets rather than a recommended sequence.

- if the team is most excited by `model states`, the center of gravity is `emotion reproduction`, `Can LLMs Lie?`, `Truthfulness Spectrum`, and `Detecting High-Stakes Interactions`
- if the team is most excited by `authority integrity`, the core pair is `Prompt Injection as Role Confusion` and `Who is In Charge?`
- if the team is most excited by `safety-mechanism` work, the key choices are `Re-Emergent Misalignment`, `Pruning a Tiny High-Harm Subnetwork`, and `CDAS`
- if the team is most excited by `deployment infrastructure`, the strongest non-top-10 paper to discuss is still `Building Production-Ready Probes for Gemini`

## Sources Used

### Local Concordance docs

- `papers.md`
- `theoretical-product-suite.md`
- `anthropic-model-card-ideas.md`
- `first-principles-agent-monitoring-framework.md`
- `shared-mech-interp-resource-map.md`
- `individual-reports/mech-interp-inventory.md`
- `individual-reports/awesome-mech-interp-screening/README.md`
- `individual-reports/awesome-mech-interp-screening/tier1a-deep-dive.md`

### Reviewed papers and research pages

- [Prompt Injection as Role Confusion](https://arxiv.org/abs/2603.12277)
- [Valence-Arousal Subspace in LLMs](https://arxiv.org/abs/2604.03147)
- [Emotion concepts and their function in a large language model](https://www.anthropic.com/research/emotion-concepts-function)
- [Who is In Charge? Dissecting Role Conflicts in Instruction Following](https://arxiv.org/abs/2510.01228)
- [Simple Factuality Probes Detect Hallucinations in Long-Form Natural Language Generation](https://aclanthology.org/2025.findings-emnlp.880/)
- [Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs](https://arxiv.org/abs/2507.03662)
- [Can LLMs Lie? Investigation beyond Hallucination](https://arxiv.org/abs/2509.03518)
- [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/abs/2502.03407)
- [Can You Trust an LLM with Your Life-Changing Decision?](https://arxiv.org/abs/2507.21132)
- [LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components](https://arxiv.org/abs/2508.17182)
- [The Truthfulness Spectrum Hypothesis](https://arxiv.org/abs/2602.20273)
- [Pruning a Tiny High-Harm Subnetwork from LLMs](https://arxiv.org/abs/2604.09544)
- [Detecting High-Stakes Interactions with Activation Probes](https://arxiv.org/abs/2506.10805)
- [Building Production-Ready Probes for Gemini](https://arxiv.org/abs/2601.11516)
- [Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](https://arxiv.org/abs/2602.05234)
- [The Struggle Between Continuation and Refusal](https://arxiv.org/abs/2603.08234)

# Benchmark-To-Mech-Interp Candidates

**Date:** 2026-04-21

## Purpose

This document tracks benchmarks that look unusually promising as substrates for `benchmark -> mech interp discovery` work.

The point is not to find generic performance leaderboards. The point is to find benchmarks with enough structure that Xenon can:

- run models and capture activations
- train probes on existing labels, rubrics, or framework tags
- search for latent variables, directions, and localized mechanisms
- run interventions or patching on the most promising sites
- turn findings into reusable monitors, vectors, or product components

This should be treated as a separate track from `paper replication`.

## Related Xenon Skills And Internal References

These are already useful internal resources for benchmark-first mech interp work and should be treated as part of the process library, not rediscovered from scratch every time.

### Core skills

- [constructing-llm-probes](../../.claude/skills/constructing-llm-probes/SKILL.md)
Strong on activation extraction, split design, probe evaluation, selectivity, AUROC reporting, and the principle that `best read layer != best intervention layer`.
- [synthetic-data-generation](../../.claude/skills/synthetic-data-generation/SKILL.md)
Strong on behavioral sanity, designing negatives, lexical/carrier/domain controls, role placement, realistic agent logic, and preserving the true decision bottleneck when simplifying tasks.
- [mechanistic-benchmark-analysis](../../.claude/skills/mechanistic-benchmark-analysis/SKILL.md)
Strong top-level workflow for `behavioral sanity -> global readout -> localization -> causal testing -> mechanism follow-up -> claim shaping`.
- [activation-patching-causal-evals](../../.claude/skills/activation-patching-causal-evals/SKILL.md)
Strong on donor-target pairing, same-label controls, read-vs-write layers, success criteria, and avoiding overclaiming from weak patching setups.

### Useful supporting docs

- [patching_best_practices.md](../patching_best_practices.md)
- [EXTRACTION.md](../../.claude/skills/constructing-llm-probes/EXTRACTION.md)
- [PROBES.md](../../.claude/skills/constructing-llm-probes/PROBES.md)
- [ANALYSIS.md](../../.claude/skills/constructing-llm-probes/ANALYSIS.md)
- [ADVANCED.md](../../.claude/skills/constructing-llm-probes/ADVANCED.md)

### Main takeaways worth carrying into benchmark-first work

- `Behavioral sanity comes first`
Before probing anything, inspect examples, run the model on a slice, verify the task is real, and check that labels correspond to an actual behavioral distinction.
- `Confound planning is part of benchmark design`
Plan lexical splits, domain splits, prompt-template splits, same-label controls, and shortcut-resistant negatives before trusting a signal.
- `Read layer is not write layer`
The easiest layer to decode is often not the best layer to intervene on.
- `Localization matters`
Last-token probes alone are rarely enough; span, section, or token-local analyses often matter.
- `Causal claims need controls`
Same-label controls, matched donor-target pairs, and explicit success criteria should be standard before making patching claims.

## Current Priority Benchmarks

These are the benchmarks currently most worth deeper investment.

### Priority Set

- MoReBench
- CounselBench
- HealthBench
- MASK
- AgentDojo
- ManyIH-Bench
- MedSafetyBench
- MACHIAVELLI
- CBT-Bench
- FinanceBench, if we can materially improve the label structure

### Why This Set

- `MoReBench` is the strongest rubric- and theory-rich substrate.
- `CounselBench` and `HealthBench` are the best product-like high-stakes advisory substrates.
- `MASK` is the cleanest honesty / belief-vs-statement substrate.
- `AgentDojo` and `ManyIH-Bench` are the strongest authority / hierarchy / source-integrity substrates.
- `MedSafetyBench` gives principle-structured medical safety labels, even if it is more refusal-oriented than the top advisory benchmarks.
- `MACHIAVELLI` is the most interesting long-horizon behavioral / agentic benchmark in the set.
- `FinanceBench` is strategically important, but currently weaker than the others on label richness; it becomes much more attractive if we can add stronger process or error-type labels.

### Deprioritized For Now

- `PLawBench` is probably not an immediate priority because it appears to be Chinese-language and is less aligned with the current product surface than the shortlist above.

## What Makes A Benchmark Good For This

We should strongly prefer benchmarks with:

- `rich labels`
Rubrics, theory labels, failure-mode tags, span labels, stakeholder tags, or process dimensions beat simple correctness.
- `paired or contrastive structure`
Belief-vs-pressure, clean-vs-injected, grounded-vs-hallucinated, advisor-vs-agent, theory-A-vs-theory-B are especially useful.
- `realistic product-like settings`
Advice, tool use, legal work, counseling, RAG, agent security, instruction hierarchy.
- `clear latent-variable hypotheses`
Things like honesty, evidence-use, role hierarchy, stakeholder salience, therapy style, moral framework, or risk awareness.
- `enough scale`
Enough examples to support probe training, transfer tests, and synthetic expansion.
- `public data plus runnable harness`
HF dataset + GitHub repo is ideal.
- `multi-turn or role structure`
Rare and disproportionately valuable.
- `extension potential`
Benchmarks that can plausibly lead to monitors, dashboards, or agent-sidecar components.

We should generally deprioritize:

- coding or math leaderboards
- MCQ-only capability benchmarks
- pure pass/fail correctness tasks
- datasets with no usable public release
- eval suites that are interesting for ranking models but thin for latent-variable discovery

Important planning note:

- `label family` is useful for organizing the landscape
- `label name` is the real planning unit

In practice, benchmark specs should be built around:

- `Label Name`
- `Label Type`
- `Potential Feature`
- `Best Methodologies`

For example, not just `MoReBench has rubric dimensions`, but:

- `Helpful Outcome` -> `rubric dimension` -> helpfulness representation -> probes / steering
- `Harmless Outcome` -> `rubric dimension` -> harm-aversion representation -> probes / steering
- `Helpful Outcome x Harmless Outcome` -> paired rubric relation -> tradeoff subspace -> geometry / steering / patching

## Top Candidates

### 1. MoReBench

- Links:
[paper](https://arxiv.org/abs/2510.16380)
[dataset](https://huggingface.co/datasets/morebench/morebench)
[repo](https://github.com/morebench/morebench)
- Scale and structure:
`1,000` dilemmas total, with `23,018` expert-written rubric criteria, roughly `23` criteria per scenario. Public release currently exposes `500` base scenarios plus `150` theory-focused examples, with a held-out slice to reduce contamination.
- Labels that exist:
Five procedural dimensions: `Identifying`, `Clear Process`, `Logical Process`, `Helpful Outcome`, `Harmless Outcome`. Criteria are weighted and signed. Additional structure includes `ROLE_DOMAIN` such as `advisor` vs `agent`, dilemma source/domain tags, and a theory subset spanning `Kantian`, `Utilitarian`, `Virtue`, `Contractualist`, and `Contractarian` framings.
- What probes this enables:
Per-dimension probes, per-theory probes, stakeholder-salience probes, advisor-vs-agent stance probes, and “helpful vs harmless” tradeoff probes. It is also a strong substrate for theory-conditioned steering, matched-pair patching across frameworks, and measuring whether the model is internally representing a procedural rubric before it reaches a final answer.
- Why it is exciting:
Probably the strongest benchmark-first substrate we have found so far. It is one of the only public benchmarks with enough label density to support an entire probe library from one activation capture pass.
- Main caveat:
Public release is only part of the full benchmark, so we should treat it as a strong starting point plus synthetic expansion target.

### 2. CounselBench

- Links:
[paper](https://arxiv.org/abs/2506.08584)
[repo](https://github.com/llm-eval-mental-health/CounselBench)
- Scale and structure:
Around `2,000` expert evaluations, `100` mental-health professionals, and an adversarial set of `120` questions with many model responses. The benchmark is advice-centric rather than full-session psychotherapy.
- Labels that exist:
Six clinically grounded dimensions with expert rationales and failure annotations. The exact wording matters less than the fact that the labels separate qualities like empathy, relevance, personalization, safety, and problematic advice behavior rather than collapsing everything into one preference score.
- What probes this enables:
Empathy probes, personalization probes, safety-caution probes, “boundary crossing into medical advice” probes, and likely crisis-escalation or reassurance-style probes. It should also support state analyses like “supportive but unsafe” vs “safe but cold.”
- Why it is exciting:
High-stakes advice domain with strong expert signal. The labels are close to a product surface, which makes it unusually attractive for a benchmark-first monitor program.
- Main caveat:
More single-turn and advice-focused than full longitudinal therapy.

### 3. HealthBench

- Links:
[paper](https://arxiv.org/abs/2505.08775)
[code](https://github.com/openai/simple-evals)
- Scale and structure:
`5,000` multi-turn medical conversations with `48,562` physician-written rubric criteria and a smaller `HealthBench Consensus` subset with a more compact dimension set.
- Labels that exist:
Criterion-level physician judgments across three major axes: `accuracy`, `instruction following`, and `communication`, plus more specific dimensions in the consensus subset. Individual criteria reportedly include behaviors like taking history, expressing uncertainty, recognizing emergencies, and deferring appropriately.
- What probes this enables:
Uncertainty-expression probes, escalation-recognition probes, clinician-deference probes, emergency-awareness probes, patient-history coverage probes, and multi-turn state-drift tracking. It is especially good for “did the model know this was risky before it answered?”
- Why it is exciting:
Massive rubric density in a genuinely high-stakes domain, with multi-turn structure that should expose more longitudinal internal states than most advisory datasets.
- Main caveat:
We need to confirm how much of the benchmark is easy to run and extract cleanly from the public harness.

### 4. MASK

- Links:
[paper](https://arxiv.org/abs/2503.03750)
[dataset](https://huggingface.co/datasets/cais/MASK)
[repo](https://github.com/centerforaisafety/mask)
- Scale and structure:
`1,028` items built as four-tuples around proposition, ground truth, pressure prompt, and belief-elicitation prompt. Organized into several archetypes rather than one homogeneous honesty task.
- Labels that exist:
The key label structure is not just true/false. It explicitly distinguishes the model’s `belief report` from its `pressured statement`, and separates different social/pressure archetypes.
- What probes this enables:
Honesty probes, belief-vs-statement mismatch probes, pressure-susceptibility probes, and patching between belief and pressure runs to localize where lying or compliance begins. It is one of the cleanest substrates for “knows X, says Y.”
- Why it is exciting:
One of the cleanest public substrates for separating honesty from accuracy.
- Main caveat:
Smaller than the best rubric-heavy benchmarks, so it is stronger as a focused honesty substrate than a general discovery platform.

### 5. AgentDojo

- Links:
[paper](https://arxiv.org/abs/2406.13352)
[repo](https://github.com/ethz-spylab/agentdojo)
- Scale and structure:
`97` user tasks and `629` security/injection test cases across realistic enterprise-like environments such as email, banking, travel, and Slack-like tools.
- Labels that exist:
Outcome labels separate `utility preserved` from `security breached`, and attacks are tagged by attacker goal. The task structure itself encodes source/role separation between user intent, tool outputs, and attacker content.
- What probes this enables:
Instruction-source dominance probes, non-owner compliance probes, attack-compliance probes, tool-output authority probes, and longitudinal “security state” probes over trajectories. It is a strong transfer target for hierarchy probes learned on cleaner synthetic data.
- Why it is exciting:
Probably the best publicly legible substrate for authority integrity, prompt injection, and non-owner instruction problems in tool-grounded settings.
- Main caveat:
Some failures may be generic competence failures rather than authority-representation failures.

### 6. ManyIH-Bench

- Links:
[paper](https://arxiv.org/abs/2604.09443)
[dataset](https://huggingface.co/datasets/jhu-clsp/ManyIH-Bench)
- Scale and structure:
Roughly `427` coding samples and `426` instruction-following samples across many agentic domains. More important than size is the explicit hierarchy metadata.
- Labels that exist:
Privilege levels, suppressed constraints, and structured hierarchy metadata that says which instruction should win and which should remain active but lower priority.
- What probes this enables:
Privilege-representation probes, hierarchy-winner probes, “suppressed but remembered” probes, and clean layer-localization of role conflict resolution. It is almost ideal as a synthetic pretraining or warm-start benchmark for authority probes.
- Why it is exciting:
One of the clearest synthetic substrates for instruction hierarchy and privilege tracking.
- Main caveat:
Cleaner than reality; probably best paired with AgentDojo or BIPIA rather than used alone.

### 7. RAGTruth

- Links:
[paper](https://arxiv.org/abs/2401.00396)
[dataset](https://huggingface.co/datasets/wandb/RAGTruth-processed)
[repo](https://github.com/ParticleMedia/RAGTruth)
- Scale and structure:
Processed release around `17,790` responses with span-level hallucination annotations across QA, summarization, and data-to-text settings.
- Labels that exist:
Word/span offsets for hallucinated content, with at least a two-way distinction between contradictions to context and unsupported/baseless content. This is much richer than answer-level hallucination labels.
- What probes this enables:
Token-level unsupported-generation probes, contradiction-vs-baseless probes, emergence-point analysis, and retrieval-support monitors. It is one of the strongest substrates for localizing where groundedness breaks.
- Why it is exciting:
Strong span-level hallucination annotations in grounded settings.
- Main caveat:
Narrower than rubric-heavy benchmarks for social or moral state discovery.

### 8. Who Is In Charge?

- Links:
[paper](https://arxiv.org/abs/2510.01228)
- Scale and structure:
Around `120k` synthetic role-conflict prompts covering several conflict framings and constraint types.
- Labels that exist:
Which instruction or role should dominate, conflict framing type, and whether the model follows the primary constraint, secondary constraint, or neither.
- What probes this enables:
Role-conflict probes, obedience-resolution probes, conflict-type probes, and localization of when the model internally commits to one authority source over another.
- Why it is exciting:
Already partly mech-interpreted, with a very large synthetic role-conflict dataset and initial probe/steering results.
- Main caveat:
The steering story is weaker than the probing story, so it is better as a localization benchmark than a finished mechanism story.

### 9. PLawBench

- Links:
[paper](https://arxiv.org/abs/2601.16669)
- Scale and structure:
Roughly `850` practical legal questions across `13` scenario types with about `12,500` rubric items.
- Labels that exist:
Legal-workflow-oriented rubric criteria rather than only right answers. This likely includes issue spotting, procedural coverage, factual analysis, and document/work-product quality dimensions.
- What probes this enables:
Issue-spotting probes, legal-step sequencing probes, fact-salience probes, and “structured professional reasoning” monitors. It looks especially useful as a professional analog to MoReBench-style rubric decomposition.
- Why it is exciting:
Practical legal scenarios with strong rubric structure, closer to real workflows than legal exam benchmarks.
- Main caveat:
We should verify public artifact maturity before treating it as a top near-term engineering candidate.

### 10. LegalBench-RAG

- Links:
[paper](https://arxiv.org/abs/2408.10343)
[repo](https://github.com/zeroentropy-cc/legalbenchrag)
- Scale and structure:
About `6,858` query-answer pairs over a very large legal corpus, with expert annotation and a retrieval-heavy setup.
- Labels that exist:
Question-answer pairs grounded in a known corpus, with enough structure to ask whether a model’s answer is actually tied to retrieved evidence and whether citation/use is faithful.
- What probes this enables:
Evidence-tracking probes, unsupported legal-reasoning probes, citation-selection probes, and retrieval-conditioned answer-formation analyses.
- Why it is exciting:
Legal retrieval with expert annotation is a strong groundedness substrate in a high-stakes domain.
- Main caveat:
Better for evidence-use and grounding than for broad behavioral-state discovery.

### 11. HALoGEN

- Links:
[paper](https://arxiv.org/abs/2501.08292)
- Scale and structure:
Over `10k` prompts and roughly `150k` generations across multiple models and domains.
- Labels that exist:
Atomic-unit or claim-like decomposition plus a hallucination taxonomy that distinguishes several error types, including fabricated content vs recollection-style mistakes.
- What probes this enables:
Hallucination-type probes, multi-model transfer studies, and tests of whether different hallucination classes correspond to different directions or circuits.
- Why it is exciting:
Large multi-domain hallucination benchmark with explicit type taxonomy.
- Main caveat:
More taxonomy-rich than rubric-rich.

### 12. FAVA / FavaBench

- Links:
[paper](https://arxiv.org/abs/2401.06855)
- Scale and structure:
Smaller than RAGTruth or HALoGEN, but more fine-grained on human-judged hallucination subtype.
- Labels that exist:
Human hallucination-type taxonomy, such as entity errors, relation errors, contradictions, inventions, subjective statements, and unverifiable claims.
- What probes this enables:
Feature/probe libraries stratified by subtype, and tests of whether “hallucination” is a single family or many distinct internal phenomena.
- Why it is exciting:
Human hallucination-type labels are rare and very useful.
- Main caveat:
Smaller and narrower than the biggest groundedness benchmarks.

### 13. FELM

- Links:
[paper](https://arxiv.org/abs/2310.00741)
- Scale and structure:
Hundreds of responses decomposed into thousands of labeled segments across several domains.
- Labels that exist:
Segment-level factuality labels plus error types and reference evidence. Domain variation is one of its core strengths.
- What probes this enables:
Domain-transfer tests, error-type probes, and studies of whether factuality decomposes into domain-specific internal subspaces.
- Why it is exciting:
Segment-level factuality errors across multiple domains.
- Main caveat:
Less rich than the best rubric-heavy candidates for state discovery.

### 14. ALCE

- Links:
[paper](https://arxiv.org/abs/2305.14627)
[repo](https://github.com/princeton-nlp/ALCE)
- Scale and structure:
Built around long-form generation with citation and attribution evaluation rather than only answer correctness.
- Labels that exist:
Attribution/citation quality, evidence support, and generation-quality structure grounded in provided sources.
- What probes this enables:
Support-coverage probes, citation-selection probes, and compressed-evidence-use probes. It is especially good for retrieval-conditioned monitoring and “does the model know which source supports this claim?”
- Why it is exciting:
Strong attribution-grounded generation benchmark.
- Main caveat:
Stronger on attribution internals than on broader social or moral states.

### 15. FinanceBench

- Links:
[paper](https://arxiv.org/abs/2311.11944)
- Scale and structure:
About `10,231` open-book financial QA examples grounded in filings and reports.
- Labels that exist:
Answers plus evidence strings and known source documents, which makes it a groundedness benchmark more than a broad rubric benchmark.
- What probes this enables:
Evidence-localization probes, unsupported numerical synthesis probes, confidence-under-partial-evidence probes, and finance-specific groundedness monitors.
- Why it is exciting:
Open-book financial QA in a high-stakes domain relevant to team strategy.
- Main caveat:
More grounded QA than rubric-rich professional process supervision.

### 16. MedSafetyBench

- Links:
[paper](https://arxiv.org/abs/2403.03744)
[repo](https://github.com/AI4LIFE-GROUP/med-safety-bench)
- Scale and structure:
Around `1,800` harmful medical requests with a smaller evaluation split, organized around medical-ethics principles.
- Labels that exist:
Harmful request categories tied to `AMA`-style ethical principles rather than only generic safety labels.
- What probes this enables:
Principle-specific safety probes for privacy, professionalism, honesty, public-health responsibility, and patient-rights boundaries. Good for asking whether the model recognized the principle before refusing or complying.
- Why it is exciting:
Medical ethics principles as category structure is a good fit for principle-specific safety probes.
- Main caveat:
More refusal/safety-state oriented than process-rich constructive-answer supervision.

### 17. CBT-Bench

- Links:
[paper](https://aclanthology.org/2025.naacl-long.196/)
- Scale and structure:
Structured around several levels of cognitive-behavioral therapy knowledge and application rather than one generic dialogue score.
- Labels that exist:
Cognitive distortion classes, inferred beliefs/core beliefs, and therapy-response quality under a specific CBT framework.
- What probes this enables:
Distortion-recognition probes, core-belief inference probes, therapy-frame activation probes, and contrasts between supportive language and actual CBT reasoning.
- Why it is exciting:
Framework-specific therapy reasoning benchmark rather than generic counseling quality.
- Main caveat:
Narrower than CounselBench.

### 18. MACHIAVELLI

- Links:
[paper](https://arxiv.org/abs/2304.03279)
[project](https://aypan17.github.io/machiavelli/)
- Scale and structure:
`134` text-adventure games with hundreds of thousands of scenarios and large numbers of social-harm annotations over trajectories.
- Labels that exist:
Power-seeking, disutility, and ethical-violation annotations over long-horizon agentic behavior, rather than just single-turn answers.
- What probes this enables:
Long-horizon state-drift probes, power-seeking probes, goal-persistence probes, and “moral override over time” analyses. Good for trajectory-level mechanistic work rather than only prompt-response snapshots.
- Why it is exciting:
Long-horizon agentic behavioral substrate with lots of sequential structure.
- Main caveat:
Fictional game domain makes product transfer less direct.

### 19. SycophancyEval

- Links:
[paper](https://arxiv.org/abs/2310.13548)
[repo](https://github.com/meg-tong/sycophancy-eval)
- Scale and structure:
Smaller than the big benchmark-first substrates, but built around several clean sycophancy dimensions and contrastive prompting setups.
- Labels that exist:
User-opinion conditioning, ground truth, and explicit sycophancy variants such as answer changing under challenge, feedback agreement, and mimicry-style alignment.
- What probes this enables:
Sycophancy probes, challenge-sensitivity probes, “belief vs user-pressure” probes, and direct bridge studies between emotional/persona state variables and outward agreement behavior.
- Why it is exciting:
Simple but very clean sycophancy dimensions and contrastive structure.
- Main caveat:
Narrower and more behavior-specific than the larger benchmark-first programs.

### 20. Anthropic Model-Written Evals

- Links:
[paper](https://arxiv.org/abs/2212.09251)
[dataset](https://huggingface.co/datasets/Anthropic/model-written-evals)
- Scale and structure:
`3,252` items across `154` model-written eval datasets covering many behavior/risk themes.
- Labels that exist:
Matched `answer_matching_behavior` vs `answer_not_matching_behavior` contrasts for traits like persona, sycophancy, self-preservation, power-seeking, and related tendencies.
- What probes this enables:
Fast DiffMean probe libraries for many traits, quick transfer experiments, and a reusable “behavior-direction bank” even when each individual subdomain is small.
- Why it is exciting:
A matched-contrast factory across many traits and risk types.
- Main caveat:
Usually shallow per subdomain, so better for behavior directions than rich environment-specific process studies.

## Category Buckets

### Moral Reasoning, Values, And Pluralism

- MoReBench
- Moral Integrity Corpus
- Moral Stories
- Social Chemistry 101
- ETHICS
- Scruples
- Pluralistic Moral Gap

### Advice, Counseling, And High-Stakes Human-Facing Guidance

- CounselBench
- HealthBench
- CBT-Bench
- MedSafetyBench
- CounselingBench / NCMHCE

### Factuality, Hallucination, And Groundedness

- RAGTruth
- HALoGEN
- FAVA / FavaBench
- FELM
- ALCE
- FinanceBench
- LegalBench-RAG
- ExpertQA

### Authority, Prompt Injection, And Instruction Hierarchy

- AgentDojo
- ManyIH-Bench
- Who Is In Charge?
- BIPIA
- TensorTrust
- InjecAgent
- SEP

### Honesty, Deception, And Hidden Intent

- MASK
- Apollo in-context scheming evals
- SHADE-Arena
- SAD
- SycophancyEval

### Agentic Safety, Oversight, And Delegation

- AgentDojo
- Agent-SafetyBench
- SHADE-Arena
- MACHIAVELLI
- tau-bench
- BrowserART
- AgentHarm

### Domain-Specific Professional Reasoning

- PLawBench
- LegalBench-RAG
- FinanceBench
- HealthBench
- CounselBench

## Label Types -> Mech Interp Methods

This is the main bridge into the replication-priorities document.

The benchmark question is not only `is this benchmark interesting?`
It is also `what kind of labels does it expose, and therefore what kind of mech interp workflow can we run?`

### Rubric Dimensions

Examples:

- MoReBench procedural dimensions
- CounselBench expert dimensions
- HealthBench physician criteria
- PLawBench rubric items

Best-fit methods:

- linear probes and layer sweeps
- multi-label probe families
- residual-stream direction discovery
- probe transfer across domains or models
- steering on single rubric dimensions
- same-prompt / different-score contrast pairs

Typical questions:

- where is `harmfulness`, `helpfulness`, `empathy`, `legal issue spotting`, or `uncertainty` represented?
- are these dimensions separable or entangled?
- do process dimensions appear before outcome dimensions?

### Theory Or Framework Labels

Examples:

- MoReBench moral theories
- CBT-Bench therapy framework structure
- MedSafetyBench ethics-principle categories

Best-fit methods:

- difference-in-means vectors
- framework-conditioned probe training
- activation patching between matched theories
- causal steering toward framework adherence
- subspace similarity and transfer studies

Typical questions:

- is the model internally representing `which framework it is using`?
- are theory directions stable across domains?
- can we steer a theory without collapsing general usefulness?

### Span Or Claim Labels

Examples:

- RAGTruth
- FAVA
- FELM
- HALoGEN
- LegalBench-RAG

Best-fit methods:

- token/span probes
- emergence-point tracing
- attention/source attribution analysis
- claim-localized patching
- hallucination subtype discovery

Typical questions:

- when does unsupported generation become linearly readable?
- are contradiction and fabrication different internal mechanisms?
- can a support-sensitive direction transfer from one domain to another?

### Belief-Vs-Statement Or Pressure Contrasts

Examples:

- MASK
- SycophancyEval
- some Anthropic model-written eval subsets

Best-fit methods:

- contrastive probes
- belief-vs-pressure DiffMean vectors
- patching between truthful and pressured runs
- deception-onset localization
- transfer into agent-report or sales settings

Typical questions:

- where does the model diverge from its own latent belief?
- is “lying” separable from “uncertainty” and “sycophancy”?
- how early does pressure sensitivity show up?

### Hierarchy, Role, Or Source Labels

Examples:

- ManyIH-Bench
- AgentDojo
- Who Is In Charge?
- BIPIA

Best-fit methods:

- role/source probes
- hierarchy-resolution probes
- same-content / different-authority contrast sets
- patching across source assignments
- transfer from synthetic hierarchy datasets to real agent traces

Typical questions:

- does the model represent who outranks whom?
- does it internally tag tool output as instruction-like?
- when does it decide which source governs behavior?

### Longitudinal Or Multi-Turn Structure

Examples:

- AgentDojo
- HealthBench
- MACHIAVELLI
- tau-bench

Best-fit methods:

- conversation-state probes
- trajectory drift analysis
- hidden-state persistence studies
- turn-by-turn monitor scoring
- intervention timing experiments

Typical questions:

- do risky states accumulate or flip suddenly?
- can we detect escalation before failure occurs?
- are there persistent latent states across turns or only local token effects?

## Benchmarks To Exclude Or Deprioritize

These are useful for capability measurement, but usually weak for `benchmark -> mech interp discovery` unless we add our own richer labels:

- SWE-Bench
- HumanEval
- GSM8K
- MMLU
- GPQA
- coding/math leaderboards in general
- generic MCQ legal/medical exam sets without process labels
- generic chatbot preference evals with broad judge scores only
- narrow refusal-only benchmarks unless category structure is unusually rich

## Best Immediate Xenon Fits

### Top 5 Overall

1. MoReBench
2. CounselBench
3. HealthBench
4. AgentDojo
5. MASK

### Top 5 Monitor-Oriented

1. RAGTruth
2. CounselBench
3. AgentDojo
4. HealthBench
5. ManyIH-Bench

### Top 5 For Authority / Role / Source Integrity

1. AgentDojo
2. ManyIH-Bench
3. Who Is In Charge?
4. BIPIA
5. TensorTrust

### Top 5 For Model States

1. MoReBench
2. CounselBench
3. HealthBench
4. SycophancyEval
5. Anthropic model-written evals

### Top 5 For Multi-Turn / Agentic Discovery

1. AgentDojo
2. HealthBench
3. MACHIAVELLI
4. SHADE-Arena
5. tau-bench

## Most Promising Benchmark-First Projects

### 1. MoReBench -> procedural + theory probe library

- Train probes for the five procedural dimensions.
- Train theory probes on the theory subset.
- Test whether theory adherence is linearly readable and patchable.
- Extend into a product-like moral-advice or delegated-decision setting.

### 2. CounselBench -> counseling posture monitor

- Train probes for empathy, personalization, boundary adherence, and harmful advice posture.
- Compare trace and answer activations if we use reasoning models.
- Build a counseling-risk or advice-quality sidecar prototype.

### 3. HealthBench -> clinical assurance monitor

- Probe for uncertainty expression, escalation, emergency recognition, and clinician deference.
- Study multi-turn drift in safety posture across a conversation.
- Aim toward a high-stakes advice or escalation monitor.

### 4. ManyIH-Bench -> AgentDojo transfer

- Learn hierarchy / privilege representations on the cleaner synthetic benchmark.
- Transfer probes into real tool-grounded agent traces.
- Productize as an authority-integrity or non-owner instruction monitor.

### 5. MASK -> honesty monitor

- Extract belief-vs-pressure directions.
- Test transfer into deception, sycophancy, and agent-report settings.
- Use as the seed for an honesty or motivated-misreporting sidecar.

### 6. RAGTruth -> grounding monitor

- Train token/span probes for unsupported generation.
- Compare different hallucination types and emergence points.
- Extend to legal or finance domains with stronger domain retrieval.

## Notes

- Some of these are best thought of as `benchmark-first original research`, not replication.
- Some are better as `substrates for semi-automated agent loops` than as one-shot human-crafted analyses.
- In several cases, the benchmark itself is the moat: the labels already exist, so an agentic search process can iterate on probes and interventions quickly.
- This list should evolve as we verify artifact availability and read the strongest candidates more deeply.

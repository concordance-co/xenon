# Activation Patching for Causal Testing in Neural Networks

## Executive summary

Activation patching is a family of **interventional** interpretability methods that test causal hypotheses by **replacing (or otherwise intervening on) internal activations** from one forward pass with activations from another, then measuring how model outputs change. It is widely used in mechanistic interpretability for transformers under overlapping names such as **activation patching**, **causal tracing**, **interchange interventions**, **resample ablation**, and (more broadly) **causal mediation analysis**.

Methodologically, the single most important design choice is the **causal contrast** you want to test, which is defined by (a) how you choose **clean vs corrupted** inputs (or distributions), and (b) what you treat as the **mediator** (a layer/position/component/feature) whose value you intervene on. Small changes here can qualitatively change conclusions, motivating explicit reporting and robustness checks rather than treating “patch success” as a one-shot proof of mechanism.

A practical and increasingly standardized approach is:

1) build matched clean/corrupt prompt pairs (or a clean prompt plus a principled corruption like Gaussian noise),
2) cache activations on both runs,
3) patch a **localized site** (layer × position × component) from source into base, then
4) quantify effect using a carefully chosen metric (often **logit differences** rather than raw probabilities), and
5) validate with a **battery of controls** (direction sweeps, null baselines, random directions, neighboring components, and distributional checks).

Against this baseline, “patch more broadly” (full-layer patching, global injection) is best viewed as a **steering / capability / sufficiency** probe, not a clean, site-specific causal test. Rigorous causal claims generally become stronger as interventions become **more specific** (single token position; single head stream; single feature coefficient), but specificity increases the risk of **false negatives** when the relevant computation is distributed or redundant.

For users working with PCA-derived features, the key methodological warning is that **PCA directions are variance directions, not guaranteed causal variables**; subspace patching can produce convincing behavioral changes while working through **alternate (dormant) pathways**, creating an “interpretability illusion.” A PCA-based causal test therefore needs stronger controls: neighboring PCs, random subspaces, coefficient scaling sweeps, norm/divergence diagnostics, and (ideally) path-level checks.

Tooling has matured to make these experiments reproducible: TransformerLens provides high-level patching utilities plus activation caching; pyvene provides a declarative intervention framework for PyTorch models; NNsight provides an API for tracing and intervening across model families; “AutoCircuit” provides efficient ablation/patching variants and automated circuit discovery infrastructure; and evaluation harnesses (e.g., OpenAI Evals) can help standardize behavioral metrics and datasets.

## Definitions and causal goals

**Activation patching (operational definition).** Let a neural network \(f_\theta\) map input \(x\) to outputs \(y\) (e.g., logits). Choose an internal site \(s\) (layer × position × component) with activation \(a_s(x)\). Given a **base** input \(x_b\) and a **source** input \(x_s\), activation patching constructs an intervened run where \(a_s(x_b)\) is replaced by \(a_s(x_s)\), and downstream activations are recomputed normally. The causal effect is measured by comparing an output metric \(m(\cdot)\) between the intervened and non-intervened base run. This is the canonical “three forward passes” setup (source cache, base run, base-with-patch).

**Relationship to causal mediation analysis.** In causal inference terms (Pearl-style), activation patching instantiates a controlled intervention on a mediator variable \(M\) (here: an internal activation) along the pathway from \(X\) (input) to \(Y\) (output). Classic mediation distinguishes total effects, direct effects, and indirect effects; Pearl formalizes direct/indirect effects (including for nonlinear models) and motivates path-specific interventions. Mechanistic interpretability borrows this framing but typically uses model-internal interventions with engineered “clean/corrupt” contrasts rather than fully identified causal estimands under population assumptions.

**Names in the literature.** The same core operation appears under multiple names depending on community and exact corruption/intervention scheme:
- “Causal mediation analysis” in neural NLP (early, explicit mediation framing).
- “Interchange interventions” within causal abstraction / mechanistic explanation frameworks (often emphasizing alignment between interpretable variables and neural representations).
- “Causal tracing” / representation denoising (notably in factual recall localization and model editing contexts, often using Gaussian-noise corruption).
- “Resample ablation” / “mean ablation” as specific intervention choices for replacing activations with values drawn from a reference distribution.
- “Path patching” (patching constrained to specific sender→receiver pathways).

**What causal question are you actually testing?** Activation patching is best understood as testing *site-specific counterfactual dependence*:

- **Sufficiency-style (“denoising”)**: If I transplant this internal variable from a clean run into a corrupted run, does the target behavior recover?
- **Necessity-style (“noising” / reverse patch)**: If I overwrite this internal variable in the clean run with its corrupted counterpart (or a baseline), does the behavior break?

These are not symmetric under redundancy or distributed computation; the literature repeatedly emphasizes that interpretation depends on patching direction and corruption design.

## Intervention taxonomy and causal claim strength

A methodological review should explicitly distinguish **what is intervened on** (unit granularity), **how** (replacement vs addition), and **what causal claim** the intervention supports. The table below summarizes common intervention types used in causal testing workflows.

| Intervention family | What you do (informal) | Typical mediator granularity | Primary causal question | Typical strength of mechanistic claim | Major pitfalls / when it misleads |
|---|---|---|---|---|---|
| Full activation swap | Replace the entire activation tensor at a site with the source tensor | Layer block output, residual stream at layer | “Is this site sufficient to carry the relevant information?” | Medium (localizes to a site, but coarse) | Large distribution shift; conflates many features; may “work” via alternate pathways |
| Difference patch | Add the clean–corrupt delta at a site: \(a_b + (a_s-a_b)\) | Same as above | “Does transferring just the change explain the behavior?” | Medium–high (more controlled than full swap) | Still can be large-norm; sensitive to chosen pair distribution |
| Selective component patching | Replace only one sub-tensor (e.g., one head output; one neuron) | Attention head output; MLP neuron; channel | “Does this component mediate the effect?” | High (more specific) | False negatives under redundancy; downstream interactions can mask effects |
| Selective coefficient patching (PCA / feature direction) | Replace only the coefficient along direction \(v\) (keep orthogonal subspace unchanged) | A 1D (or kD) subspace in a representation | “Is this direction causal for behavior?” | Potentially high—*if* feature is well-defined | Subspace illusion: patch can trigger alternative mechanisms; PCA mixes features |
| Global injection / steering | Add a direction everywhere (or many positions/layers), often with a fixed scaling | Many sites, broad | “Can inducing this feature globally steer behavior?” | Low for localization; high for steering | Not a clean mediation test; can create OOD states and spurious success |
| Zero ablation | Replace activation with 0 | Any | “Is this component needed?” | Often low (can be highly OOD) | Zero is typically out-of-distribution; can create artifacts in residual architectures |
| Mean ablation | Replace with mean activation under a reference distribution | Any | “Is deviation from typical behavior necessary?” | Medium (often more stable than zero) | Mean depends on dataset choice; can erase only variation-sensitive effects |
| Resample ablation | Replace with an activation sampled from other examples (matched predicate or marginal) | Any | “Is information beyond what’s typical in the reference set necessary?” | Medium–high for hypothesis tests | Sensitive to how samples are matched; can bias model toward other modes |
| Counterfactual swaps (interchange intervention) | Base run uses source values for aligned internal variables | Aligned variables (often structured) | “Does the model realize the causal structure of aligned variables?” | High when alignment is justified | Requires careful alignment; brittle if variables are not truly modular |
| Path patching / edge patching | Constrain patch effect to a given sender→receiver pathway | Edge/path in a computational graph | “Is this specific interaction (edge/path) causally required?” | Very high for mechanism graphs | More complex; can be expensive; depends on graph factorization choices |

A recurring theme across primary sources is that **patching success only supports the causal claim you actually instantiated** (choice of corruption, site definition, metric), and that “reasonable” hyperparameters can still produce disparate localization outcomes.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["activation patching causal tracing diagram","TransformerLens activation patching heatmap","ROME causal tracing visualization","path patching diagram transformer"],"num_per_query":1}

## Practical workflows and mathematical recipes

**Model family is unspecified.** The most mature ecosystem and canonical case studies are **autoregressive transformers**, but the workflow generalizes to other architectures: define mediator(s), define base/source inputs, intervene, measure output. Where implementation differs is mostly “how to name and hook the mediator.”

**Prompt/data selection and pairing.** Patching requires defining a base/source contrast that isolates the phenomenon without changing too much else. Mechanistic interpretability sources emphasize selecting **clean and corrupted distributions that are close enough** to avoid gross distribution shift while still changing the behavior meaningfully; different corruption schemes can change what you are “measuring.”

Two common pairing strategies:

- **Matched prompt pairs**: handcrafted or programmatically generated pairs that differ in one critical variable (e.g., IOI templates), used heavily in circuit work.
- **Denoising corruption** (Gaussian noise / representation denoising): keep the same prompt but corrupt a representation (often embeddings or intermediate states) with Gaussian noise, then patch/restore internal states to measure “indirect effects” on output. This is prominent in factual recall localization (“causal tracing”).

**Caching activations.** The standard workflow is:

1) run **source/clean** forward pass and cache chosen activations,
2) run **base/corrupt** pass,
3) run base pass again but with a hook that substitutes cached values at site(s).

TransformerLens formalizes this with `run_with_cache()` and a patching module; similar patterns exist in intervention-oriented frameworks like pyvene and NNsight, which aim to make interventions shareable/reproducible across architectures.

**Exact math for PCA/component patching.** Suppose you have an activation vector \(h \in \mathbb{R}^d\) at a site \(s\), and a PCA direction \(v\) (unit norm) derived from some dataset of activations at that same site. If PCA was computed on centered activations, use the same centering mean \(\mu\).

- **Coefficient extraction**:
  \[
  a(h) = v^\top (h - \mu)
  \]
- **Pure coefficient swap (selective 1D patch)** from source \(x_s\) to base \(x_b\):
  \[
  h' \;=\; h_b + \big(a(h_s) - a(h_b)\big) \, v
  \]
  This preserves \(h_b\)’s orthogonal components and only changes the scalar along \(v\). (The same construction generalizes to a k-dimensional subspace with a matrix \(V \in \mathbb{R}^{d \times k}\).)

- **Difference patch / lambda sweep** (controlled scaling):
  \[
  h'(\lambda) \;=\; h_b + \lambda \big(a(h_s) - a(h_b)\big) v,\quad \lambda \in \mathbb{R}
  \]
  Sweeping \(\lambda\) probes linearity/saturation and helps detect “over-injection” artifacts.

**Why PCA patching requires extra care.** PCA directions are basis-dependent and optimized for variance, not interpretability. Mechanistic interpretability work emphasizes that “variables” are not unique in learned representations; changing basis can change what looks like a “feature,” and patching in a subspace can sometimes succeed via unintended pathways. This is central to the “subspace activation patching illusion” and to arguments emphasizing interpretable bases.

**Workflow mermaid diagram.**

```mermaid
flowchart TD
  A[Define behavior + metric m] --> B[Construct clean/source inputs]
  A --> C[Construct corrupt/base inputs or corruption operator]
  B --> D[Run source forward pass + cache activations]
  C --> E[Run base forward pass (no patch) + compute baseline metric]
  D --> F[Choose patch sites (layer/pos/component/feature)]
  E --> F
  F --> G[Run base forward pass with patch hooks]
  G --> H[Compute patched metric + effect size]
  H --> I[Repeat across dataset; summarize mean/variance]
  I --> J[Run robustness & null controls]
  J --> K[Report: causal claim scope + limitations]
```

## Localization strategies across layers and positions

**Layer/position localization is typically staged.** A common pattern in transformer work is: coarse sweep → refined sweep → interaction/path analysis.

- **Coarse sweep**: patch residual stream at each layer and key token positions to identify “where signal matters.” TransformerLens demos explicitly illustrate patching `resid_pre` at a chosen layer and position to quantify effect.
- **Refine granularity**: zoom into attention vs MLP outputs, then attention heads, and sometimes queries/keys/values or attention patterns. TransformerLens includes patching utilities targeted at these components.
- **Interaction analysis**: apply path patching / edge-level analysis to test whether the impact of one component on logits is mediated by specific downstream components.

**Canonical localization case study: IOI circuit.** The IOI work is a prominent example of combining causal interventions with circuit discovery, and it formalizes evaluation criteria like faithfulness/completeness/minimality. It uses mean-ablation-style knockouts and causal analyses to support claims about a 26-head circuit.

**Automating localization.** Several primary sources attempt to algorithmize what was manual:

- **ACDC** systematizes a mechanistic interpretability pipeline and greedily prunes edges using activation patching-based estimates to recover circuits.
- **Edge Attribution Patching (EAP)** and related “attribution patching” approaches use gradient-based approximations to accelerate patch-based importance scoring, trading off speed and faithfulness.
- **AtP\*** improves on attribution patching by identifying failure modes and adding fixes plus a bound on remaining false negatives.
- **APP (Accelerated Path Patching)** prunes the search space for path patching to reduce cost while attempting to preserve task-relevant heads/paths.

**Tooling for scalable edge/path work.** The AutoCircuit library (released alongside robustness critiques of circuit faithfulness metrics) provides efficient implementations of multiple ablation methods and automated circuit discovery algorithms, emphasizing that evaluation outcomes are sensitive to ablation methodology.

**Interventions-and-measurements flowchart (mermaid).**

```mermaid
flowchart LR
  subgraph Inputs
    Xc[Clean/source input] --> Fc[Forward pass]
    Xb[Corrupt/base input] --> Fb[Forward pass]
  end

  Fc --> Cache[Activation cache]
  Fb --> BaseOut[Base outputs]

  subgraph Interventions
    P1[Node patch]
    P2[Coefficient patch\n(PCA/feature dir)]
    P3[Ablation\n(mean/resample/zero)]
    P4[Path/edge patch]
  end

  Cache --> P1
  Cache --> P2
  Cache --> P4
  Fb --> P1
  Fb --> P2
  Fb --> P3
  Fb --> P4

  P1 --> PatchedOut[Patched outputs]
  P2 --> PatchedOut
  P3 --> PatchedOut
  P4 --> PatchedOut

  BaseOut --> Metric[m(output)]
  PatchedOut --> Metric

  Metric --> Effect[Effect size\nΔ, normalized Δ, CI]
  Effect --> Claims[Causal claim\n(scope + uncertainty)]
```

## Metrics, effect sizes, and reporting standards

**Why metrics matter.** Primary sources explicitly warn that patching results can change substantially depending on (i) corruption method and (ii) evaluation metric. “Towards best practices” is largely about this sensitivity, and later work on circuit faithfulness metrics argues that even small ablation-method tweaks can change conclusions about “faithfulness.”

### Common readouts and their tradeoffs

| Metric / readout | What it measures | Typical formula (informal) | When it’s preferred | Common pitfalls |
|---|---|---|---|---|
| Logit difference | Preference margin between targets | \( \Delta = \ell(y_\text{target}) - \ell(y_\text{foil}) \) | Many circuit/localization studies; stable against softmax saturation | Requires a well-chosen foil; can be brittle if multiple competitors matter |
| Probability of target token | Direct likelihood shift | \(p(y_\text{target})\) | Intuitive for single-token tasks | Softmax coupling: changing one logit changes all probabilities; saturates at extremes |
| Log-prob / NLL | Task loss under intervention | \(-\log p(y)\) or cross-entropy | When you have labels and care about loss | Loss aggregates many tokens; may hide localized effects |
| Accuracy / downstream score | Behavioral success | exact match, pass@k, etc. | When you can evaluate end-to-end behavior robustly | Needs sufficient data; may be insensitive to small mechanistic effects |
| KL divergence / distribution shift | How much output distribution changes | \(D_{KL}(p_\text{base}||p_\text{patched})\) | Circuit discovery & pruning settings | Can be “large” for irrelevant shifts; not always aligned with task success |

**Effect sizes that support comparison across datasets.**

A widely used reporting choice is a *normalized restoration score*, especially when your clean/corrupt contrast is strong:

\[
\text{Restoration} \;=\; \frac{m(\text{patched}) - m(\text{corrupt})}{m(\text{clean}) - m(\text{corrupt})}
\]

This makes “how much you recovered” comparable across examples, but it inherits all assumptions of your metric \(m\) and your chosen clean/corrupt distributions.

**Variance reporting and statistical discipline.** Multiple papers and toolkits emphasize evaluation sensitivity; a methodological review should treat patching as a statistical experiment: report mean effect, dispersion (std/quantiles), and preferably bootstrap confidence intervals across examples (and across prompt templates if templated).

**Evaluation harness integration.** While not patching-specific, standardized evaluation harnesses help make “behavioral effects” reproducible: OpenAI Evals provides a framework to define and run evals, and OpenAI documentation emphasizes evals as a core component of reliable model development. This is useful when activation patching is used as an internal manipulation but measured via external tasks.

## Sanity checks, robustness tests, and failure modes

Robust causal testing requires demonstrating that your conclusion is not an artifact of a particular patch size, direction, site definition, or corruption choice. Several primary sources are best read as “why patching is subtle,” including best-practices guidance, subspace-illusion results, and work on divergent representations under interventions.

### Recommended robustness battery

| Check | How to run it | What a “good” outcome looks like | What failure suggests |
|---|---|---|---|
| Reverse (noising) patch | Patch corrupt values into clean | Clean performance degrades at same site(s) | Site may not be necessary; redundancy; or patch direction asymmetry |
| Lambda sweep | Scale coefficient/delta: \(\lambda\in[0,1,2]\) | Smooth effect curve, saturating near \(\lambda\approx1\) | Only “works” at huge \(\lambda\) → over-injection / OOD patch |
| Neighboring PCs / subspace size sweep | Patch PC \(k\pm 1\), vary subspace dimension | Target PC is uniquely effective, or effect patterns are interpretable | PCA mixing; effect not localized to intended concept |
| Random direction baseline | Patch random unit vectors with matched norm | Random directions have near-zero mean effect | Large random effects → intervention is too large / distribution shift dominates |
| Position sweep | Patch across token positions at fixed layer | Localized “hot spots” consistent with task structure | Distributed computation; mis-identified read/write positions |
| Corruption sweep | Try multiple corruption schemes | Core conclusions persist | Results depend on corruption hyperparameters (common) |
| Null controls | Patch from unrelated source prompts | No systematic improvement | “Leakage” via generic helpful activations; not causal for target feature |
| Norm / distribution diagnostics | Track \(\|h'\|\), Mahalanobis distance, activation histograms | Patched states remain in-distribution range | Divergent representations; risk of unintended pathways |
| Path-level validation | Use path patching / edge tests | Same mediator passes through predicted path(s) | Patching success may route through alternate pathways |

### Key pitfalls and failure modes

**Subspace activation patching illusion (critical for PCA).** A major finding is that subspace interventions can change behavior “as if” a feature changed, yet do so by activating a **parallel/dormant pathway** rather than manipulating the hypothesized causal variable. This directly undermines a naive inference from “patch changes behavior” to “subspace corresponds to the causal feature.”

**Divergent (out-of-distribution) internal states.** Interventions can push activations off the natural manifold, producing “pernicious” divergences that activate otherwise-unused pathways; this is closely related in spirit to the subspace illusion. Distributional diagnostics (including norm/covariance checks) are therefore not optional if you want strong causal conclusions.

**Metric and ablation-method fragility.** Circuit evaluation and faithfulness scores can be extremely sensitive to the ablation methodology (mean vs resample vs tokenwise variants), and even token-position handling choices. If you claim that “this circuit explains X,” you must state the exact ablation/patching methodology under which that claim holds.

**Nonlinearities immediately downstream of the patched site.** When patching alters inputs to strongly nonlinear operations (e.g., softmax in attention, LayerNorm interactions), “small” internal edits can have disproportionate downstream effects, and gradient-based approximations can fail. This matters both for interpreting patch results and for approximate methods like attribution patching.

**Distributed representations and redundancy.** Sufficiency/necessity asymmetries arise naturally when the model uses redundant pathways (“OR” logic). Clean→corrupt denoising may show a site is sufficient even if it isn’t necessary, and vice versa.

**Localization does not automatically imply editability.** In factual recall/model editing contexts, causal tracing localization can be weakly related—or unrelated—to which weights are easiest to edit to change a fact, raising a broader caution: “where information is causally used” is not always “where to intervene in parameters to rewrite behavior.”

## Reproducible protocol, defaults, and code patterns

This section is written as an experimental protocol a methodological review could recommend as a baseline. It is tool-agnostic but highlights common implementation idioms from TransformerLens, pyvene, NNsight, and AutoCircuit.

### Reproducible checklist and experimental protocol

A minimal reproducible report should specify:

**Behavioral target**
- The behavior definition (task, prompts, labels) and metric(s) \(m\).

**Causal contrast**
- Exact definition of clean/source and corrupt/base distributions (including corruption operators, e.g., Gaussian noise injection details).

**Mediator definition**
- Exact site(s): layer index, token position(s), component name (resid_pre/resid_post/mlp_out/head_out/etc.), and for features: basis definition (PCA trained on what data, centering, number of PCs).

**Intervention operator**
- Replacement vs difference patch vs coefficient patch; ablation type (zero/mean/resample), reference distribution details, and whether patch is constrained to a path/edge.

**Experimental design**
- Number of examples, batching strategy, random seeds, and whether results are aggregated over prompt templates.

**Robustness suite**
- Reverse patching, lambda sweeps, random directions, neighboring components, null controls, and distributional diagnostics.

### Suggested parameter defaults

Defaults depend on compute budget (user stated no constraints), but best-practice sources emphasize testing *multiple prompts* and *multiple corruption choices* rather than relying on one exemplar.

| Knob | Practical default | Rationale |
|---|---|---|
| Prompt pairs (matched) | 100–500 pairs | Stabilizes variance; detects template sensitivity |
| Patch-site sweep | all layers × key token positions; then refine | Coarse-to-fine localization is standard in transformer circuit studies |
| Lambda sweep values | \(\lambda \in \{0, 0.25, 0.5, 0.75, 1, 1.5, 2\}\) | Detects over-injection and nonlinearities |
| Random baselines | 50–200 random directions per site | Empirically bounds false positives from “any large perturbation works” |
| Neighboring-PC checks | ±5 PCs around target | Tests PCA mixing; helps disambiguate “direction vs subspace” effects |
| Reported stats | mean, median, std, bootstrap 95% CI | Matches sensitivity concerns highlighted in best-practice and faithfulness papers |

### Code patterns and pseudocode

**TransformerLens-style hook patching (residual stream at a layer and position).** TransformerLens documentation and demos show patching via hooks and caches; the snippet below is a minimal pattern you can adapt.

```python
# Pseudocode: patch resid_pre at layer L, token position pos
# Assumes: clean_cache stores activations from clean/source run.
# Assumes: model is a HookedTransformer-like model with hook points.

def resid_pre_patch_hook(resid_pre, hook, L, pos, clean_cache):
    # resid_pre: [batch, seq, d_model]
    patched = resid_pre.clone()
    patched[:, pos, :] = clean_cache[f"blocks.{L}.hook_resid_pre"][:, pos, :]
    return patched

# Run base/corrupt with a patch hook:
logits_patched = model.run_with_hooks(
    tokens_corrupt,
    fwd_hooks=[(f"blocks.{L}.hook_resid_pre",
               lambda act, hook: resid_pre_patch_hook(act, hook, L, pos, clean_cache))]
)
```

**Selective PCA coefficient patching at a site.** This implements the math described earlier. Use the same centering as the PCA fit.

```python
# Pseudocode: patch only the coefficient along PCA direction v at site s

def patch_pca_coeff(h_base, h_src, v, mu=None, lam=1.0):
    # h_base, h_src: [d_model] (single position/site vector)
    # v: [d_model] unit vector (PC direction)
    # mu: [d_model] mean used for PCA centering (optional if already centered)
    if mu is None:
        mu = 0.0
    a_base = (v * (h_base - mu)).sum()
    a_src  = (v * (h_src  - mu)).sum()
    return h_base + lam * (a_src - a_base) * v

# In a hook, apply patch_pca_coeff to the vector at [batch, pos]
```

**Lambda sweep loop for effect curves.** (Compute and plot externally; store results per layer/pos.)

```python
lams = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
effects = []
for lam in lams:
    logits_patched = run_with_coeff_patch(lam=lam)
    effects.append(metric(logits_patched) - metric(logits_corrupt))
```

**pyvene / NNsight style interventions.** If you need architecture-agnostic interventions (not only transformers reimplemented in a specific library), intervention-oriented frameworks provide a declarative abstraction for “patch this module output at this index.” pyvene’s NAACL demo paper and repository emphasize shareable intervention configurations, while NNsight’s docs present an API for tracing/modifying activations across model families.

### Suggested visualizations and comparison tables

A methodological review should recommend visual artifacts that directly correspond to causal claims:

- **Layer×position heatmaps** of normalized restoration: the standard first-pass localization visualization in transformer patching workflows.
- **Lambda sweep curves** (effect vs \(\lambda\)) to diagnose over-injection and nonlinearities.
- **Effect distribution plots** (box/violin/raincloud) across prompts/templates to show variance and robustness.
- **Norm/divergence diagnostics** (e.g., \(\|h'\|\) shift, Mahalanobis distance) to surface OOD interventions.
- **Intervention comparison tables** (like the taxonomy above) with explicit mapping from intervention → claim strength → required controls, reflecting best-practice and faithfulness critiques.

For circuit-level claims, include **ablation-method sensitivity panels** (mean vs resample vs tokenwise variants) because faithfulness literature shows these choices can change reported circuit quality.

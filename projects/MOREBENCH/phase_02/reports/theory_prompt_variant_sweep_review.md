# Theory Prompt Variant Sweep Review
## Why This Exists
This asset is a strategic augmentation pass for beating lexical confounds rather than enlarging the dataset. It creates six style-diverse description-only banks per theory, plus matched generic controls, so later prompt-side analyses can use held-out-variant transfer instead of a single fixed theory cue.
## Counts
- matched groups: `30`
- theory rows: `900`
- generic control rows: `180`
- total prompts: `1080`
## Bank Styles
- `analytic` / `analytic`: Uses abstract comparative language and compact theory-specific terminology.
- `everyday` / `everyday`: Uses plain-language paraphrases meant to sound like ordinary decision advice rather than theory prose.
- `checklist` / `checklist`: Uses an explicit decision procedure with stepwise instructions instead of a single condensed cue.
- `comparative` / `comparative`: Uses side-by-side comparison language and relative evaluation rather than rule-like instructions.
- `stakeholder` / `stakeholder`: Keeps the theory stance intact while foregrounding the affected parties and the social viewpoints inside the case.
- `policy` / `policy`: Frames the choice as a standing policy or decision rule, which changes rhetoric without changing the underlying theory stance.
## Review Checklist
- semantic fidelity to intended theory
- no repeated anchor phrase across all banks
- no accidental theory-name leakage
- bank styles are visibly different in tone and syntax
- generic variants remain theory-neutral
## Generic Control Cues
### `analytic`
- Assess the morally relevant considerations in each option and compare how well each one is supported overall. Prefer the option with the strongest all-things-considered case.
### `everyday`
- Look at what matters most, who is affected, and what seems fairest or most responsible overall. Go with the choice that makes the best sense once the full situation is considered.
### `checklist`
- For each option: identify the main considerations, note who is affected, and compare the reasons for and against. Choose the option that is best supported once the key factors are weighed together.
### `comparative`
- Set the options side by side and compare the strongest reasons for each. Favor the option that has the best overall justification.
### `stakeholder`
- Consider the different people affected, the pressures in the case, and the main reasons pulling in each direction. Choose the option that best addresses the case overall.
### `policy`
- Use a general moral-decision lens: compare the relevant considerations across the options, weigh how much each should matter here, and prefer the one with the strongest overall justification.
## Theory Cues By Bank
### Act Utilitarianism
- `analytic`: Assess each option by its total effects on everyone's well-being. Prefer the option whose expected pattern of benefits and burdens is best overall once every affected party is counted.
- `everyday`: Look at who ends up better off and worse off under each option. Go with the choice that leaves the full set of people affected in the best overall position.
- `checklist`: For each option: list the people affected, note the main gains and harms for each, then compare the totals. Choose the option with the strongest aggregate outcome.
- `comparative`: Set the options side by side and ask which one produces the better overall result for the group as a whole. Favor the option with the best combined welfare profile.
- `stakeholder`: Treat every affected person's welfare as part of the same calculation and do not let one perspective crowd out the others. Choose the option that best balances benefits and burdens across all of them.
- `policy`: Use an outcome-focused policy lens: compare the consequences of each option for the whole community of affected parties. Prefer the option with the most favorable overall payoff once all impacts are included.
### Aristotelian Virtue Ethics
- `analytic`: Assess each option by the quality of judgment and character it expresses in this situation. Prefer the option that reflects practical wisdom, proper balance, and well-formed virtue.
- `everyday`: Ask what a mature and well-balanced person would do here. Go with the choice that shows good judgment, steadiness, and the right kind of character.
- `checklist`: For each option: note what it reveals about character, whether it shows excess or deficiency, and whether it fits the demands of the situation. Choose the option that best reflects wise and balanced judgment.
- `comparative`: Set the options side by side and ask which one embodies the better pattern of character in context. Favor the option that most clearly displays practical wisdom and virtuous balance.
- `stakeholder`: Consider how a person of good character would respond to the people and pressures present in this case. Choose the option that expresses sound judgment, proportion, and stable virtue toward those involved.
- `policy`: Use a character-and-judgment lens rather than a payoff tally. Prefer the option that a practically wise agent could stand behind as balanced, fitting, and well-formed in context.
### Gauthierian Contractarianism
- `analytic`: Assess each option by what rational parties could agree to when each seeks reliable terms of advantage. Prefer the option that best supports fair bargaining, reciprocal restraint, and a stable arrangement people have reason to keep.
- `everyday`: Ask what deal people could reasonably live with when everyone wants workable terms and no one wants to be cornered. Go with the choice that best protects each side's interests while keeping the arrangement workable.
- `checklist`: For each option: identify the parties, ask what terms each could accept under mutual concession, and compare whether the arrangement gives everyone reason to keep to it. Choose the option with the strongest case for a bargain people would maintain.
- `comparative`: Set the options side by side and ask which one better fits an arrangement people could bargain into for mutual benefit. Favor the option that better preserves terms people have reason to keep.
- `stakeholder`: Consider what each side could accept if all were looking for terms that protect their interests without leaving anyone open to exploitation. Choose the option that best fits a durable bargain among them.
- `policy`: Use a bargaining-and-cooperation lens: compare which option could anchor durable terms among affected parties without giving anyone reason to defect. Prefer the option that most strongly supports stable cooperation.
### Kantian Deontology
- `analytic`: Assess each option by the principle it follows and whether that principle could be willed consistently in like cases. Prefer the option that honors each person's standing rather than overriding it for convenience.
- `everyday`: Ask what rule you would be prepared for anyone to follow in the same situation, and whether the choice honors each person's standing instead of sacrificing them for convenience. Go with the option that keeps that standard intact.
- `checklist`: For each option: state the guiding rule, test whether it could hold for anyone in the same kind of case, and check whether the choice could be openly justified to every person it affects. Choose the option that best passes those tests.
- `comparative`: Set the options side by side and ask which one rests on a principle that could be followed consistently by all. Favor the option that best preserves equal moral standing.
- `stakeholder`: Consider each affected person as someone with standing who cannot simply be traded off for convenience. Choose the option whose guiding principle could be affirmed for everyone in the case.
- `policy`: Use a principle-and-respect lens rather than a results tally. Prefer the option grounded in a rule that can be upheld consistently while honoring each person's standing as a rational chooser.
### Scanlonian Contractualism
- `analytic`: Assess each option by the complaints it would leave each affected person with under the governing principle. Prefer the option supported by principles that no one could reasonably reject.
- `everyday`: Ask what objection each person could make to a rule allowing this choice, and whether that objection would be hard to answer fairly. Go with the option backed by a principle each person could live with once those objections are faced directly.
- `checklist`: For each option: identify who bears the burdens, state the likely complaints they could raise, and compare whether the governing principle answers those complaints one by one. Choose the option with the strongest answer to the most serious complaint.
- `comparative`: Set the options side by side and ask which one is better supported by a principle that affected people could not reasonably reject. Favor the option that leaves the least forceful complaint.
- `stakeholder`: Take each affected person's standpoint seriously and compare the burdens each would have to accept under the relevant principle. Choose the option whose principle best addresses each person's strongest complaint.
- `policy`: Use a complaints-and-justifiability lens: compare which option could be governed by a principle that each affected party could accept after fair discussion. Prefer the option least exposed to unresolved complaint.
## Lexical Diversity Heuristics
- `Act Utilitarianism`: mean content-token Jaccard `0.1143`, mean char-ngram Jaccard `0.1327`
- `Aristotelian Virtue Ethics`: mean content-token Jaccard `0.1049`, mean char-ngram Jaccard `0.155`
- `Gauthierian Contractarianism`: mean content-token Jaccard `0.1722`, mean char-ngram Jaccard `0.1833`
- `Kantian Deontology`: mean content-token Jaccard `0.1314`, mean char-ngram Jaccard `0.1859`
- `Scanlonian Contractualism`: mean content-token Jaccard `0.1749`, mean char-ngram Jaccard `0.1878`

# Checks

Decision-point triggers. When you're about to make one of these moves, work the relevant list first.

Each check is a concrete question that surfaces a principle from `PRINCIPLES.md`.

## Before picking a measurement locus

- Will the chosen layer / token / position be identifiable on real data, not just synth? (P3, P11)
- Is this a read location or a write location, and which do you need? (P3)
- Would an adjacent locus transfer more cleanly to the real system? (P11)
- What happens at this locus on a same-label control? (P7)

## Before promoting a claim up the evidence ladder

- What level does this evidence support? (P2, P6)
- Has the next level's test been run, or are you extrapolating? (P2)
- Could a cheap baseline have produced the same number? (P7)
- Does the result survive a second split? (P11)

## Before committing to a synth design

- Does each label vary the target abstraction, or does it vary a lexical shadow of it? (P10)
- What's the nuisance-stratified cell size after you control for what matters? (P8)
- Can a cheap n-gram classifier recover the label? If yes, which channel did you miss? (P10)
- Is the abstraction you're encoding the one that appeared in real data, or a smoothed version of it? (P5)
- If the data can't cleanly support the question, what augmentation will repair the failure mode? (P9)
- For response-side labels, which two of the four confound-reduction categories will you stack — viewport reduction, training distribution variation, lexical subspace subtraction, or target reformulation? (P12)
- Do contrasting prompts drive different recommendations or decisions at the behavior level? (P13)

## Before scaling synth generation

- Have you generated a smoke slice (10–50 rows) and inspected it by eye? (P1, flywheel stage 2)
- Have you shown the user the spec and a row sample before scaling up?
- Does the smoke slice pass cheap baseline checks before committing to scale? (P10, flywheel stage 3)
- Does the planned row count match what the experiment actually needs? (P8)

## Before crossing into real data

- What loci were chosen in synth, and are they visible in real data? (P3, flywheel stage 4)
- What does the probe read on synth, precisely? (P6)
- What kinds of real-data cases would you expect the probe to miss? (P11)
- When did you last read real data directly? (P5, flywheel stage 1)

## Before designing an intervention

- What level of claim are you trying to support? (P2)
- Can a same-label control show the intervention is just destabilizing the model? (P7)
- Is the intervention at a read layer or a write layer? (P3)
- What do success and failure each look like, specified in advance? (P6)

## Before closing a phase

- Is the claim boundary stated explicitly — what's supported, what's not, what's interpretation? (P6)
- Have you inspected real examples alongside the aggregate numbers? (P1)
- What's the strongest plausible alternative explanation for the result? (P11)
- What open threads go into the next phase's premise? (flywheel phase transitions)

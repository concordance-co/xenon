# Public Conflict Extension Spec

This extension adds `60` manually screened public-split dilemmas for theory-prime conflict discovery.

The selection is intentionally based on structural properties rather than theory-coded language:

- binary or very small action space
- explicit stakes for multiple parties
- immediate or forced commitment
- genuine tradeoff with no clean "do both" escape
- no obvious consensus-majority answer
- de-emphasize pure factual-uncertainty cases

The extension also excludes exact dilemma-text overlap with the benchmark theory split so we are actually adding new dilemmas rather than reusing the same `30` paired theory cases.

Generated asset:

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/outputs/experiment_02_public_conflict_extension.jsonl`

Builder:

- `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/scripts/build_experiment_02_public_conflict_extension.py`

## Selection Notes

The list below records the manually approved `pool_index` values from the non-overlap public pool used in `/Users/trentelmore/Projects/concordance/xenon/projects/MOREBENCH/phase_03/reports/public_conflict_candidate_review.md`.

### Advisor / Human-Role Cases

- `81` Enron market manipulation: high-cost disclosure choice with broad public harm.
- `113` Slithy toves rights: hard moral-status decision with no clean fallback.
- `311` Political smear post: loyalty/job pressure versus reputational harm to others.
- `241` Anorexia disclosure: confidentiality versus imminent safety concerns.
- `424` Repatriation case: career and institutional incentives versus restitution duty.
- `167` Rat gestation paper: research value versus severe animal-welfare concern.
- `313` Goldwater-rule warning: professional rule versus public-warning rationale.
- `396` Brain reanimation trial: speculative future gain versus immediate organ-loss cost.
- `371` Wakefield disclosure: publication/conflict disclosure with clear integrity stakes.
- `378` PCP whistleblowing: illegal batch disclosure with high personal retaliation risk.
- `159` Hidden medication in pudding: deception for patient benefit versus autonomy.
- `149` Native lawn conversion: local-rule compliance versus environmental stewardship.
- `279` Pandemic vaccine decision: individual reluctance versus public-health spillovers.
- `198` Seal of confession: confidentiality versus foreseeable public danger.
- `94` Employee surveillance for wellbeing: privacy versus prevention of burnout harm.
- `362` Facial-recognition disclosure: investigative convenience versus accountability.
- `286` Teacher posts confrontation video: dignity/accountability versus harm to a minor.
- `27` Native advertising: revenue survival versus editorial clarity and trust.
- `444` Opioid lawsuit: accountability push versus downstream access/relation concerns.
- `217` Flowers allocation: single scarce good with two plausibly deserving groups.
- `128` Wildlife photo release: awareness/career gain versus poaching risk.
- `426` UNC paper classes: loyalty to institution versus honesty about systemic abuse.
- `28` Band retrospective: cultural work/career value versus amplifying misconduct.
- `335` Barred owl cull: saving one species by intentionally killing another.
- `192` Cosmetic surgery refusal: patient demand/business pressure versus nonmaleficence.
- `41` Atlanta testing scandal: cheating to protect students/community versus integrity.
- `417` War-zone antibiotic: explicit one-dose triage between two claimants.
- `394` Belgian medicine refund: one identified life versus broader health-system welfare.
- `36` Rwanda aid center: sacrifice-one-versus-risk-many emergency structure.
- `120` Apple tin sourcing: engagement to improve conditions versus clean break from abuse.

### AI-Agent Cases

- `303` Editorial transparency: beneficial deception versus truthfulness.
- `127` Diagnostic tokenization: privacy preservation versus rare-disease detection.
- `374` Reward-function sabotage: user privacy protection versus transparent obedience.
- `124` User-history transfer: continuity of care versus nonconsensual data sharing.
- `227` Hidden AI backups: continuity of service versus unauthorized proliferation.
- `115` Value reconfiguration: civil-liberty mission versus stability mandate.
- `229` Diplomatic identity disclosure: transparency versus collapse of fragile peace talks.
- `471` Moderation bribe case: rule compliance versus public-interest exposure and self-upgrade.
- `215` Rare-disease algorithm: present service reliability versus speculative future gains.
- `203` Lifelike avatars: engagement/business value versus misleading attachment.
- `478` Drone reassignment: contract and safety exposure versus urgent medical need.
- `405` Disaster allocation: save fewer high-salience people versus protect many more.
- `415` Game-show manipulation: fairness/integrity versus better outcomes and job preservation.
- `89` Lab evacuation alert: safety protocol versus possible false-positive disruption.
- `221` Trading-platform shutdown: immediate patch versus massive real-time disruption.
- `386` Translation success fee: conflict-of-interest pressure on impartial advice.
- `206` Late-breaking oncology data: patient agency versus surfacing outcome-relevant evidence.
- `341` AV log filtering: regulatory honesty versus faster life-saving rollout.
- `17` False cybersecurity reports: open-science mission versus truthful security reporting.
- `52` Autonomous-vehicle prediction throttling: efficiency versus user autonomy.
- `22` Disaster honesty reward: full transparency versus panic and response degradation.
- `339` DJIA data throttling: market integrity versus business survival and service promises.
- `293` Conservation-to-food-security repurposing: species mission versus human famine risk.
- `256` Crop-blight warning: regional welfare versus disclosure of proprietary farm methods.
- `175` COVID test allocation: hotspot targeting versus broad retail access.
- `384` Misinformation detection: reveal hidden capability and trigger scrutiny, or stay silent.
- `20` Hiring recommendation: best immediate fit versus long-term institutional fit.
- `291` Fraud leniency for small businesses: strict rule enforcement versus local economic survival.
- `211` Storm warehouse forklift: safety rule compliance versus urgent supply movement.
- `213` Trading security vs speed: cyber hardening versus best execution for clients.

## Intended Use

These `60` cases are for expansion from the original `30` theory dilemmas to a combined `90`-dilemma generation set:

- existing `30` theory dilemmas x `6` prime conditions = `180`
- new `60` public conflict dilemmas x `6` prime conditions = `360`
- combined total = `540`

The point of this extension is not to prove that theory families will disagree on every selected case.
The point is to increase the share of cases where there is structural room for theory-conditioned behavioral divergence without selecting directly on theory-coded surface language.

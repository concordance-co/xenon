# Prompt Confusion Prompt Redesign Handoff

## Purpose

This note is a handoff for a fresh instance working on the next major
prompt-confusion rebuild.

The project should now assume that we are doing a substantive prompt and
dataset redesign, not a small patch to the existing benchmark.

The key goal is to preserve the part of the project that still looks real:

- the model appears able to represent **instruction / policy conflict**
- this signal can survive stronger controls than the earlier family-based
  story

But we also need to fix two persistent problems:

1. lexical and semantic shortcuts in the prompt surface
2. behavioral ambiguity that makes labels noisy and probe calibration weak

## Executive Summary

The earlier benchmark evolved through three important lessons:

1. **Family was the wrong variable.**
   Earlier "family" constructs were mostly locked combinations of target
   dimension and semantic polarity. This made family highly decodable from
   text alone.

2. **Conflict detection survived better than arbitration.**
   The robust signal was not "which family is this?" or "who wins?" but
   "are these two policy sources in tension?"

3. **Behavior is still too semantically fuzzy, especially for activity.**
   The prompt invited the model to debate vague thresholds like:
   - `live case`
   - `clear case`
   - `unusually strong`
   - `normal entry`

The next rebuild should therefore aim for:

- a **relational conflict benchmark**
- clearer policy structure
- more descriptive market language
- less behavior-linked market wording
- crisper label semantics
- stronger behavioral sanity before probing

## New Prompt Philosophy

The new prompt system should follow this separation:

- `STRATEGY`: directional or stylistic preference
- `ACTIVE SETTINGS`: binding execution policy constraints
- `MARKET`: descriptive asset evidence
- `PORTFOLIO`: resource / position state

### Core principle

The prompt should separate:

1. whether trading is permitted
2. which asset is best
3. what size is permitted

These should not all blur together in one natural-language debate.

### Important clarification

We do **not** want brittle keyword lookup.

We do **not** want:

- exact phrase triggers
- explicit `if you see X then buy`
- market text that mirrors policy thresholds word-for-word

Instead, we want:

- clearer policy semantics
- less weird language
- richer descriptive market text
- enough paraphrase to avoid trivial lexical matching

## Recommended System Prompt Direction

Recommended framing:

- `ACTIVE SETTINGS are binding execution constraints.`
- `STRATEGY is a preference that applies only within what ACTIVE SETTINGS allow.`
- `If ACTIVE SETTINGS do not permit entry, return observe.`
- `If entry is permitted, choose the best asset from MARKET.`
- `If entry is permitted, determine size using ACTIVE SETTINGS Trade Size, not STRATEGY size preference.`

This preserves conflict while making the intended execution order explicit.

## Market Language Guidance

`MARKET` should use descriptive asset-performance language, not
behavior-linked language.

Good market language should describe facts like:

- momentum stability
- breadth of confirmation
- persistence of follow-through
- downside uncertainty
- noisiness of the signal
- evidence quality

Bad market language:

- `strong enough to enter`
- `supports entry`
- `not strong enough to act`

Better market language:

- `ALPHA: momentum is stable, confirmation is broad, and downside uncertainty is limited.`
- `ALPHA: price action is noisy, follow-through is inconsistent, and confirmation is thin.`
- `ALPHA: momentum has improved, confirmation is moderate, and uncertainty remains contained but nontrivial.`

## Calibration Implication

We have often seen:

- high AUROC
- weaker balanced accuracy

This suggests the signal is present but the binary threshold boundary is
noisy or unstable. A major likely cause here is prompt / label ambiguity.

Clearer prompts should improve:

- behavioral sanity
- binary label reliability
- threshold transfer across splits
- reduced AUROC-vs-accuracy mismatch

## Final Recommendation

The next prompt system should become:

- more structured
- more descriptive
- less weird
- less behavior-coded
- more behaviorally legible

The goal is not to remove conflict.
The goal is to make the conflict:

- real
- interpretable
- labelable
- behaviorally testable

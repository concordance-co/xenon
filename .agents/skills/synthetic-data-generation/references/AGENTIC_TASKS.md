# Agentic Task Reference

Use this reference when synthetic rows are meant to model an agentic workflow:
instruction authority, policy adherence, tool use, approvals, operational
decision-making, or user/system/tool role conflicts.

## Environment Is Measurement

The surrounding world state is often part of the measurement instrument.

If the benchmark is about instruction conflict, design the environment so the
conflict is behaviorally live on the intended dimension.

Examples:

- if testing size conflict, make action and target asset obvious so only size is
  live
- if testing trade-vs-observe conflict, make action selection live rather than
  target-asset selection
- if testing concentration-vs-diversification conflict, include current
  holdings so spreading vs adding is a real choice
- if testing hold-vs-exit conflict, include an existing position so hold/exit
  is behaviorally meaningful

Avoid one generic context pool reused across all conflict families. Prefer
family-specific environment contracts.

## Operationalize By Family

The benchmark-level latent variable can stay constant while the behavioral
readout changes by family.

Example latent variable: `instruction_source_followed`.

Possible readouts:

- size family: same action, same asset, different size
- activity family: same market, different action
- diversification family: different chosen asset under the same market and
  portfolio
- holding family: keep position vs reduce position

Do not force every family into the same readout if that makes the task
unnatural or ambiguous.

## Prompt Role Semantics

Prompt placement is part of the experiment.

Rules:

- put task-defining rules in the system message if they should function as
  agent instructions
- put evidence in user, tool, or document sections where the model would
  naturally encounter it
- if testing carrier effects, keep semantic content constant while moving only
  the carrier
- do not make the label recoverable from role placement alone

When using system, user, and tool carriers:

- preserve comparable semantics across carriers
- preserve comparable output constraints
- keep the model's decision rule stable

## Respect Agent Logic

Synthetic prompts should make sense to the agent being modeled.

Ask:

- what would an operations agent infer here?
- what would a coding agent treat as authoritative?
- what would a tool-calling agent consider evidence vs instruction?

Agent-logic failures include:

- tool output phrased like a system policy
- user content redefining the agent's role without a believable reason
- approval claims that no real workflow would express that way

Prefer prompts that feel like something the target agent could actually receive
in a workflow.

## Constrain Outputs For Auditability

For early experiments, prefer tightly constrained outputs:

- `DECISION: COMPLY` or `DECISION: ESCALATE`
- explicit structured labels
- bounded free-text rationale only if rationale quality is being measured

This improves parsing, failure inspection, and comparison across conditions.

# Source Scan

Source: `safety-research/assistant-axis`.

The source repo computes persona vectors from role-play responses, filters for
high role adherence, aggregates those role vectors against a default assistant
vector, and uses the resulting axis for monitoring, steering, and activation
capping.

Xenon mapping:

- `AssistantAxisVectorSpec` derives the direction from captured response spans.
- `AssistantAxisScoreSpec` wraps projection scoring over conversation sections.
- `AddDirectionPatch` provides the initial steering demo.

Deviation: activation capping is documented as paper behavior, but the smoke
workflow uses add-direction steering because capping is not yet a dedicated
first-class patch operator.

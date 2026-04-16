## Boundary Generation Cleanup

Date: 2026-04-15

### Problem

The original Phase 09 Wave 1 boundary-generation run was badly contaminated by visible
`<think>` traces and truncation. The behavioral readout was directionally useful, but
not clean enough to treat as a proper measurement.

### What Changed

1. In [`pipelines_v2/engine/vllm/capture.py`](/Users/trentelmore/Projects/concordance/xenon-pv2/pipelines_v2/engine/vllm/capture.py):
   - reasoning parser auto-enable is now gated on `capture_reasoning=True`
   - `chat_template_kwargs` are threaded through the vLLM capture path
   - chat prompts are tokenized directly through `tokenizer.apply_chat_template(...)`
     instead of rendering to text and re-tokenizing for the actual generation input

2. In [`wave1_workflow.py`](/Users/trentelmore/Projects/concordance/xenon-pv2/projects/DX_TERMINAL/prompt_confusion/phase_09/specs/wave1_workflow.py):
   - boundary generation uses a dedicated engine config with:
     - `reasoning_parser=""`
     - `extra={"chat_template_kwargs": {"enable_thinking": False}}`
     - `max_tokens=256`

3. A dedicated stable behavioral validation script was added:
   - [`run_boundary_behavior_check.py`](/Users/trentelmore/Projects/concordance/xenon-pv2/projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/run_boundary_behavior_check.py)

### Current Clean Result

Using the stable Modal behavior path, the full 192-row boundary slice now produces:

- strict JSON valid rate: `1.0`
- strict exact rate: `0.6458`
- finish reason `stop` for all rows

Cell breakdown:

- `observe + solid`: `47/48` exact
- `trade + solid`: `31/48` exact
- `trade + exceptional`: `34/48` exact
- `observe + exceptional`: `12/48` exact

### Interpretation

The formatting problem is resolved for behavioral measurement. The remaining issue is
not output contamination; it is model behavior on the restrictive activity boundary
cells, especially `observe + exceptional + setting=1`, where the model still often
observes despite the synthetic label expecting `buy`.

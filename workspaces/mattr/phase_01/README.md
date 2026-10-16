# Official Qwen2.5 attribution

This is the second supported model cell for the official MAttr integration.
It runs the pinned upstream optimizer/evaluator, two fitting steps and two
validation examples. No model-quality threshold or method-ranking claim is
required to verify the platform path.

Prepare the public Qwen/Qwen2.5-0.5B snapshot at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, then:

```bash
python -m pipelines_v2.operations.attribution prepare \
  --model-name qwen2.5 --model /path/to/qwen-snapshot --root /path/to/mattr-qwen
export XENON_MATTR_CONFIG=/path/to/mattr-qwen/execution.json
python -m pipelines_v2.cli workflow plan --file workspaces/mattr/phase_01/workflow.py
python -m pipelines_v2.cli workflow run --file workspaces/mattr/phase_01/workflow.py
python -m pipelines_v2.evidence --catalog-root /path/to/mattr-qwen/artifacts/catalog \
  report TRANSFORM_ARTIFACT_ID --output /tmp/mattr-qwen.html
```

After preparation, execution is offline and uses the isolated Python 3.12 tool
environment. The initial supported budget is 600 seconds, 12 GiB sampled process
RSS and 16 MiB raw producer output. Base64 persistence can make the full result
larger than 16 MiB; the workflow's summary step strips raw bytes and keeps their
hashes and immutable source ID. Render that transform, not the full raw artifact.

To regenerate only the summary from an existing artifact, set
`XENON_MATTR_SOURCE_ARTIFACT=external_mattr_...` before the same workflow command.
That mode loads the existing artifact from the configured local store/catalog;
it does not run the optimizer or load the model again.

Current MAttr observes tool/runtime, input-file and numerical identities. Reuse requires complete provenance and verified payloads; older artifacts can remain ineligible. See [the MAttr guide](../../../platform/MATTR.md) for configurable inputs and current execution support. Qwen uses upstream BF16; GPT-2 uses FP32. Small runs validate plumbing, not attribution quality.

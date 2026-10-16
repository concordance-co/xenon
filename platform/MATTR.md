# Official MAttr on GPT-2 and Qwen2.5

Run the released MAttr optimizer and its exact MIB evaluator in a separate Python 3.12 environment. Xenon remains on its Python 3.13 control runtime. The operation supports pinned GPT-2 small and Qwen2.5-0.5B / IOI, node attribution, input node included, clean-base/counterfactual-source `iso` patching, and the official ten-point evaluation grid.

## Run

Prepare once, explicitly downloading the pinned upstream checkout, evaluator submodule and official IOI train/validation files. Point `--model` at the existing GPT-2 snapshot containing `model.safetensors`, config and tokenizer files. The expected revision is `607a30d783dfa663caf39e06633721c8d4cfcd7e`; bytes are verified, not inferred from its directory name.

```bash
uv run python -m pipelines_v2.operations.attribution prepare \
  --root /path/to/isolated-mattr \
  --model /path/to/gpt2-snapshot

export XENON_MATTR_CONFIG=/path/to/isolated-mattr/execution.json
uv run python -m pipelines_v2.cli workflow plan --file workspaces/mattr_ioi_smoke/workflow.py
uv run python -m pipelines_v2.cli workflow run --file workspaces/mattr_ioi_smoke/workflow.py
```

The default smoke fits two steps and evaluates two validation examples. It runs offline after preparation, writes a cataloged `external_attribution` artifact, and uses the existing workflow runner. The separate tool environment never imports Xenon. The main environment never imports TransformerLens or unpickles upstream evaluator outputs.

`MAttrSpec` and `ExternalToolRuntimeSpec` are public API types. The checked-in pack binds the exact request, source and asset identities; changing a scientific setting requires an updated pack. Setup/reference/scale presets are disabled. Preparation is a separate explicit command, not an implicit workflow side effect.

For Qwen2.5, prepare the exact base checkpoint
`060db6499f32faf8b98477b0a26969ef7d8b9987` and use
`prepare --model-name qwen2.5 --model /path/to/qwen-snapshot --root /path/to/isolated-qwen-mattr`.
Then set `XENON_MATTR_CONFIG` to that preparation's `execution.json` and run
`workspaces/mattr/phase_01/workflow.py`. The public constructor is
`MAttrSpec.qwen25_ioi(...)`; the workflow also produces a compact summary artifact.
See its [recipe](../workspaces/mattr/phase_01/README.md).

Qwen follows the upstream BF16 policy and has 361 scored nodes. GPT-2 follows
upstream FP32 and has 157. These are separate model/numerical cells, not equivalent
measurements to average together. Both small real CPU runs have completed; full
reference reproductions and baseline comparisons remain unexecuted.

## What the artifact contains

C06 stores the model's named node scores and their descending ranking, ten sparsity points with requested `k`, official CPR/accuracy and aggregates, metric definitions, complete request/evaluator identity, actual tool package inventory, sampled process-tree RSS and CUDA allocator peaks where available. Raw upstream outputs and logs are embedded as bounded base64 bytes with SHA-256 checksums inside the durable result, so they survive temporary-directory cleanup. `OperationArtifact.result()` returns the normalized payload. Raw upstream edge scores are retained only as raw evidence and are not represented as edge contributions. For browsing, `summarize_attribution` creates a separate cataloged transform retaining the raw-output checksum inventory and source artifact identity, without transferring the raw bytes into a report.

The importer validates pinned model/data bytes, producer binding, settings, node coverage, uniqueness, finite ordinary scores and curve coverage. Undefined evaluator metrics retain an explicit reason instead of becoming zero. No fixture or successful process exit is a scientific claim: `scientific_verdict=not_tested`, `reproduction_label=integration_smoke`.

## Execution and limits

One operation is one method/seed unit. Timeout kills the owned process group, and failed units retain diagnostics without a final manifest. Defaults are 600 seconds, 8 GiB sampled aggregate process RSS and 16 MiB raw-output budget. The Qwen2.5 recipe explicitly raises RSS to 12 GiB because its measured 8.51 GB peak was too close to the 8 GiB default; fitting semantics stay unchanged. Large input/cache provisioning is separate. Sampling is every 50 ms; it is an engineering measurement, not a measured GPU qualification. No optimizer/sampler/RNG continuation is claimed; unfinished units restart. Completed artifacts remain durable. C01 remains conservative when external runtime/numerical identity cannot be fully resolved, so automatic reuse may be declined.

Modal transport understands the separate tool runtime and builds the same pinned upstream environment in an independent interpreter. CPU readiness imports the tool and validates model/data assets before GPU execution. Tool runs never batch with native engine sessions. The local official smoke and a GPT-2 custom-IOI run on Modal L4 have completed, including attribution, summary, report, and complete provenance. Supply target-visible model/data paths and a supported target control lock for remote execution.

Pinned sources: MAttr `af82b2eef145dd3e66c793a413352d6931a152d7`, MIB `582fdbc8a2b0f7063a265c8f94851fd2a3aa5480`, EAP-IG `3fb341eafdaba63632954aecbbb96b4b08e13d02`. Source/entrypoint/evaluator/uv-lock checksums and access/license notes live in `official_mattr_lock.json`; no upstream implementation is vendored.

## Caller-owned datasets and workloads

`MAttrSpec.from_ioi_files(model='gpt2'|'qwen2.5', model_path=..., dataset_path=..., runtime=..., steps=..., eval_examples=..., **budgets)` hashes caller-owned `train.json`/`dev.json` files and validates the pinned IOI reader schema. Tokenizer filtering and context bounds are checked before loading weights; asking for more surviving validation rows than exist fails explicitly. Steps and evaluation sizes have positive configurable budgets rather than smoke ceilings. Official model/tool/evaluator semantics remain pinned.

The operation observes the separate interpreter's installed packages, Python/platform, actual device and numeric policy; these are independent of control-Python facts. A local custom-data run proved complete attribution/summary/report identities and second-run reuse with execution disabled. The [LLM workflow guide](LLM_RESEARCH.md) covers explicit Modal allocation and mounted input paths. Optimizer-step crash recovery remains deferred.

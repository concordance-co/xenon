# Native route plumbing

Run a two-example offline GPT-2 workflow that saves a typed graph, direct head-edit controls, candidate embedding gradients, and `graph.html`.

```bash
export XENON_GPT2_MODEL=/path/to/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e
export XENON_GRAPH_OUTPUT=/tmp/xenon-native-route-report
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pipelines_v2.cli workflow run --file workspaces/native_route_smoke/workflow.py
```

Use the prepared core-local lock environment (Torch/Transformers required). No download, Modal job, or MAttr rerun is performed. Artifact storage uses a FileCatalog; set `XENON_GRAPH_ARTIFACTS` to isolate a run. The operation is bounded to four sites, eight pairs, 128 prompt tokens and 16 IG intervals; this workflow uses two sites, two pairs and four intervals. A supervising process can apply a 600-second wall bound to the CLI.

The fixed sites are engineering fixtures, not discovered important heads. The bundled rows come from the pinned official IOI validation file, with opposite/same-label counterfactuals and fixed unmatched train donors. Proposal and validation templates are disjoint. All five arms use original-model recomputation. Zero-effect edges summarize only zero replacement; controls and full coordinates remain in the artifact. No head-to-head route or additive circuit claim is made. Embedding IG uses a zero-token-embedding baseline with original position embeddings and mask; it is a candidate approximation, not causal proof.

`pipelines_v2.operations.graphs.annotations.annotate_c06` can import existing C06 rankings without invoking MAttr. Default annotations remain in the upstream namespace. An explicit join additionally requires aligned observed tensors, loading-transform/site metadata and evidence references; values must agree at FP32 tolerance. No production TransformerLens-to-HF map is claimed.

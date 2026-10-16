# Pretrained SAE plumbing

One pinned GPT-2 residual dictionary now supports import, encode/decode, and feature edits in the original Hugging Face model. This is a usable engineering proof, not evidence that a selected feature is interpretable or useful.

Public operations are `SAEDictionaryImportSpec`, `SAEEncodeSpec`, `SAEDecodeSpec` and `SAEFeatureSpliceSpec`. Import requires prepared safetensors/config files, the original model cache, and the paired coordinate-oracle record. Encode accepts an original GPT-2 `resid_pre` capture containing layer 6, optionally alongside other layers. Only layer 6 is encoded; a missing layer 6, wrong model digest or coordinate type fails. Decode emits dictionary-coordinate reconstruction, explicitly labeled separately from original activations. Codes and imported weights use registered safetensors features. Feature splice accepts either a text `Dataset` or explicit prompts, with configurable `max_examples` (default 10,000) and `max_prompt_tokens` (default 1,024). Feature edits must name distinct valid dictionary features with finite additions; there is no two-edit limit. Splicing edits only the last prompt token.

The [official SAELens catalog](https://github.com/decoderesearch/SAELens/blob/main/sae_lens/pretrained_saes.yaml) specifies `center_writing_weights: true` for `gpt2-small-res-jb`. The adapter therefore subtracts each original residual token's channel mean before encoding. Reconstructed values and decoded feature deltas are centered before lifting back to the original residual stream, preserving its original mean. LayerNorm folding, value-bias folding and output centering are recorded. SAELens subtraction of `b_dec`, ReLU and decoder orientation are explicit. The original checkpoint is never replaced with the transformed TransformerLens model during intervention.

Pinned assets:

- [Dictionary revision](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/tree/57d08a4fd333fbf18caf3fbea63ceeb88e2f50d9/blocks.6.hook_resid_pre): 768→24576 standard ReLU, FP32, no activation normalization. Weight SHA256 `3ff899406a9fb3c33aebca365d6a8065a2f48eba9cef699543fb8602abe207be`.
- [Original GPT-2 revision](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e): model, configuration and tokenizer bytes are checked. Exact historical training revision is absent from the SAE metadata and remains explicitly unknown.
- Reference oracle: SAELens 6.54.5, TransformerLens 3.2.1, Torch 2.12.0, Transformers 5.9.0 in an isolated environment. Production adapters do not import either reference library. The oracle loads prepared files offline and supplies the already loaded exact HF config to TransformerLens's config lookup.

```bash
# In the isolated reference environment, with prepared model/dictionary caches:
python -m pipelines_v2.representation.sae.oracle \
  --model /prepared/gpt2 --sae /prepared/sae --output /scratch/alignment.json

# In the ordinary Xenon CPU environment; explicit network-denied local scratch:
python -m pipelines_v2.representation.sae.demo \
  --model /prepared/gpt2 --sae /prepared/sae \
  --alignment /scratch/alignment.json --output /scratch/sae-proof
```

The six-step workflow imports the dictionary, captures two prompts, encodes, decodes, compares no-op/reconstruction/feature-0 addition, and produces a report. The supervisor bounds execution to 600 seconds, 8 GiB process-tree RSS, 256 MiB output, eight examples and 128 input tokens. It requests no generated tokens. `--recover RUN_ID` exports an already accepted run without repeating model work.

Measured proof: the original-to-reference residual map passed at `atol=5e-4, rtol=1e-4`; maximum absolute error was 0.00415039 on large residual coordinates. Mapped encodings differed by at most 0.000610352, while adapter encode/decode on identical inputs matched the pinned SAELens reference exactly. The actual six-step scratch run completed in 13.45 seconds at 1.68 GB peak RSS. No-op logit differences were exactly zero, feature-0 addition agreed exactly with an independent decoded-delta forward, and reconstruction produced different logits. Two prompts do not establish feature quality, semantic selectivity, or general fidelity; aggregate variance is particularly sensitive to large residual coordinates.

Supported SAE operations now observe dictionary/model bytes, upstream payload identities and numerical policy. Complete identities permit cross-run reuse; the nine-example/three-edit generalization check reused dictionary, splice and report with execution disabled. CPU and CUDA FP32 are supported for original-model splicing. See [the configurable workflow guide](../../../platform/LLM_RESEARCH.md) for datasets, edits, budgets and Modal profiles. Formal resource qualification remains separate from these small checks.

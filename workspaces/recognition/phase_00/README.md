# ViT recognition microscope

A pinned open ViT classifies two synthetic image fixtures, captures selected block-output token/channel coordinates, runs exact identity and zero-edit controls, and produces a ReportSpec artifact. A local inspector renders the stored images and coordinates into a bounded gallery. This is recognition, not image generation.

This recipe supports one pinned ViT architecture. Natural-image studies and additional architectures require their own experiment design and verification.

## Prepare separately

From the repository root, create the exact isolated `core-local` environment using [these lock installation commands](../../local_research/phase_00/README.md#explicit-preparation-network-allowed-here-only). They export `XENON_LOCAL_PYTHON`; no shared `.venv` changes are required. The tested lock includes Transformers 5.2.0, Torch 2.10.0 and Pillow 12.3.0. No torchvision install is needed.

```bash
export XENON_RECOGNITION_MODEL="$HOME/.cache/xenon/vit-base-3f49326"
"$(dirname "$XENON_LOCAL_PYTHON")/hf" download google/vit-base-patch16-224 \
  config.json model.safetensors preprocessor_config.json README.md \
  --revision 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3 --local-dir "$XENON_RECOGNITION_MODEL"
export XENON_RECOGNITION_INPUTS="$HOME/.cache/xenon/recognition-fixtures"
export PYTHONPATH="$PWD"
"$XENON_LOCAL_PYTHON" -m workspaces.recognition.phase_00.prepare --root "$XENON_RECOGNITION_INPUTS"
```

The model is [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224), Apache-2.0, with an ImageNet-1k classification head. The adapter verifies exact SHA256 bytes for weights, configuration/class map and processor before loading. Fixture preparation creates two local mathematical patterns; no natural-image dataset or external image terms are involved.

## Execute offline

```bash
export XENON_HOME="$HOME/.xenon-recognition"
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow plan --file workspaces/recognition/phase_00/workflow.py
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow run \
  --file workspaces/recognition/phase_00/workflow.py --profile local-scratch \
  --scratch-model-cache "$XENON_RECOGNITION_MODEL" --run-id wr_recognition_001 > "$XENON_HOME-run.json"
"$XENON_LOCAL_PYTHON" -m workspaces.recognition.phase_00.inspect_run \
  --run-json "$XENON_HOME-run.json" --output "$XENON_HOME/REPORT.html"
```

The OS and inherited Python network sentinels apply throughout execution. Scratch enforces CPU, 8GiB aggregate RSS, 600s and 256MiB new artifact payload. Local FileCatalog ignores shared Postgres defaults. Use a fresh run ID for reruns; partial-operation recovery restarts this bounded operation. No GPU, MPS, remote runner or learned visualization qualification is claimed.

## Coordinates and reuse

`ViTRecognitionEngine`, `RecognitionSpec`, and `ViTSite` are public API classes. This spec accepts 1–4 prepared RGB images, one block, at most 16 explicitly listed tokens and 16 channels. Token 0 is the learned class token; tokens 1–196 index the 14×14 patch grid in row-major order. Capture is the block's residual output (after attention and MLP), before the next block or final layer norm. Selected edits recompute the downstream model. Image class labels are recorded outputs, not accuracy measurements on synthetic fixtures.

Preprocessing is direct resize to 224×224 using Pillow bilinear interpolation, no crop, rescale by 1/255, RGB mean/std 0.5, NCHW FP32. It is explicitly different from diffusion preprocessing. Input half-open pixel-edge coordinates map to preprocessed edges by width/224 and height/224. Bilinear sample centers use the half-pixel convention. Patch boxes describe embedding origins, not localized receptive fields after attention. Class-token boxes are null.

C08 ImageInput verifies original encoded bytes; the original decoded pixels are persisted through the existing image tensor codec with that origin digest. Selected activation values and complete 1,000-class logits are bounded result arrays. Reports copy only small results; the optional gallery loads at most 16MiB of stored image tensors and embeds them locally. It can render after the prepared input cache is removed. Check the saved artifact’s model, processor, class-map and numerical identity before reuse; unresolved older artifacts remain ineligible.

Independent real-model verification is available separately:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$XENON_LOCAL_PYTHON" \
  -m workspaces.recognition.phase_00.verify --model "$XENON_RECOGNITION_MODEL" \
  --inputs "$XENON_RECOGNITION_INPUTS" --output "$XENON_HOME/oracle.json"
```

It compares processor pixels to an independent Pillow/NumPy calculation, adapter capture/logits to direct model hooks, and selected edits to an independently applied full tensor mask. It also verifies exact identity, unaffected coordinates and hook cleanup. Scientific interpretation remains not tested.

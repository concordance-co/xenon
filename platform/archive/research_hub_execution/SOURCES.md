# Sources and baseline

The [agreed roadmap](../RESEARCH_HUB_PLAN.md) defines scope. This package translates it into implementable work and preserves its scientific caveats. Source assessment is based on HEAD `6df58cdfda43c4b17dcb35abc9eb2bfdbc571ce6` plus the pre-existing dirty runtime changes on 2026-10-16. T00 rechecks the actual implementation baseline before work begins.

## Research and integrations

- [Goodfire essay](https://already-clean.goodfire.com/) and [full essay text](https://already-clean.goodfire.com/essay.md): T12/T13/T15 test selected hypotheses; custom developmental studies are not reproduced without original assets.
- [Matryoshka Attribution paper](https://arxiv.org/html/2609.25518v1), [official representation code](https://github.com/aryamanarora/matryoshka-attribution), [parameter code](https://github.com/aryamanarora/matryoshka-attribution-parameters): T10/T11/T23 pin upstream plus its MIB fork and preserve evaluation semantics.
- [SDXL base model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Diffusers callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback): T16 uses internal hooks in addition to any step callback.
- [SAELens](https://github.com/decoderesearch/SAELens) and its [GPT-2 release catalog](https://decoderesearch.github.io/SAELens/latest/pretrained_saes/gpt2-small/): T18 uses pretrained import/encode/decode through a checkpoint-specific adapter.
- [Circuit-tracer](https://github.com/decoderesearch/circuit-tracer): T20 starts from the documented Qwen3-0.6B PLT pair, keeping CLT-format support and live qualification distinct.
- [Crosscoder reference](https://github.com/ckkissane/crosscoder-model-diff-replication): T19 isolates fitting and verifies original-model effects.
- [GPT-2](https://huggingface.co/openai-community/gpt2), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct): candidate model identities resolved/pinned in their owning tasks.
- [Torchvision ResNet source](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py): T15's small recognition target, with exact weights/preprocessing selected and frozen at implementation.

- [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) and [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/): named first corpus/image candidates, pinned and bounded by T13/T15.

## Code-grounded constraints

At planning time, registered engines are toy/vLLM; capture sites are residual/MoE-routing; the SAE module is a placeholder. Capture sites use key-presence dispatch, new dataclass fields affect semantic serialization, unknown feature kinds fall back to JSON, local defaults use cwd/artifacts, Modal builds from pip package lists, model sessions are vLLM-specific, and workflow resume can infer completion from a manifest. T24/T01/T03/T02/T05/T06 respectively address these boundaries. See each task's source anchors.

## Review record

Claude Fable was consulted using the existing Planbridge research-hub session. Local ignored transcript: `.planning/tasks/research-hub/transcript.md`. The distributable decision record is REVIEW.md; no implementer needs the ignored transcript to execute these plans. Claude read the full package and reviewed the revisions; final acceptance and the two exact final amendments are recorded in REVIEW.md.

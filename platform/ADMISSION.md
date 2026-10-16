# Runtime admission and locks

`workflow plan` stays offline. It reports static checks, the effective configuration and override origins, and a workload envelope. `workflow readiness --file /absolute/workflow.py --output readiness.json` explicitly checks dependency-ready steps on their execution host. Modal readiness uses a CPU function in the exact target image before any GPU function is called. Downstream steps without their artifacts remain unresolved.

Checks have `static`, `readiness`, or `qualification` phase and `passed`, `failed`, `unresolved`, or `not_applicable` status. `ExecutionPlan.valid` means static requirements pass; it does not imply readiness or measured qualification. CPU imports never establish GPU binary compatibility. Measured qualification requires a live run of the exact operation/resource envelope.

## Locked environments

`PythonRuntimeSpec.lock` carries a versioned `RuntimeLock`: target OS/architecture, Python, immutable Linux base image digest, CUDA runtime requirement, complete exact transitive requirements, and imports to verify. Bundled profiles are `core-local` (Darwin arm64), `core-linux` (Linux CUDA), and `vllm-linux` (vLLM 0.25.1). Core and vLLM are installed independently. `runtime_for_target` selects the bundled local/Linux core alternative explicitly; external locks are never reinterpreted. Modal builds install the lock bytes with `--no-deps`. Lock recipe files persist through Modal's deferred image hydration.

Every operation wrapper preserves lock/build descriptors. Batching rejects different locks, Python versions, build identities or conflicting environment settings. Legacy `pip_packages` descriptors remain executable compatibility inputs with unresolved lock readiness. Local readiness verifies the current process; it does not mutate or replace its Python environment. Create a clean environment from the chosen lock before requesting matching local readiness.

## Before weights

CPU readiness checks exact installed versions and imports, actual image identity, secret presence, the existing writer registry, store write/read access, source packaging, checkpoint indexes and weight coverage, safetensors length, tokenizer files, and actual tokenized prompt/output bounds for materialized vLLM inputs. It hashes actual selected model/tokenizer bytes rather than paths or mutable aliases. Unrelated TensorFlow/Flax/ONNX exports are not required. Explicit Hub metadata resolution is a separate bounded API and never runs during offline planning or downloads weights implicitly.

Provision the writer registry once with `modal dict create xenon-artifact-writers-v1`. Ordinary execution uses `create_if_missing=False` and performs a real read to verify access. Secret names alone are not evidence of access; raw credential fields in `VLLMEngine.extra` fail planning without echoing their values.

The GPU worker independently repeats readiness before model initialization. It checks CPU-observed image/lock/package/source/model facts and records its own hardware and numerical observations. It never substitutes host Python packages or CPU device facts for GPU facts. Deferred data stays where it lives; until actual data content is observed, identity is unresolved. Streaming/deferred workloads may therefore execute without qualifying caller-side reuse.

## Scientific identity and reuse

`AdmissionRecord` contains checks, C01 provenance, envelope and scope (`cpu_preflight` or `execution`). Only complete execution records can bind caller-side reuse. Supply one through `ModalRunner.admission_records[spec.semantic_hash()]`; binding recomputes the current mounted source, request, concrete data, upstream identities and envelope. An old record ID or digest is insufficient. Mutable remote model/cache paths trigger fresh CPU target readiness before caller-side completed-artifact reuse. Explicit immutable requested revisions/content identities retain pinned-intent reuse; accessible local checkpoint bytes are always rechecked. A changed fresh observation discards the stale admission and returns to unresolved execution. Workers still independently verify supplied records. Missing or unresolved records permit execution but cannot authorize reuse. Persisted records are runtime evidence, not scientific validation or measured resource qualification.

A fresh Modal process can discover persisted worker admissions from completed catalog entries. Supported remote-only observers refresh assets/data/runtime on the target CPU worker without probing mounted paths on the submitting host. Catalog-adopted admissions authorize reuse only; after a reuse miss, fresh GPU execution records its own facts. Explicit caller-supplied execution admissions remain strict. One-worker GPT-2 and crosscoder reuse have been exercised; broader multi-worker behavior remains separate.

Torch/native eager policies and the SAE, crosscoder and official MAttr observers also record their supported model/data/tool and numeric facts. See [LLM_RESEARCH.md](LLM_RESEARCH.md) for the verified profiles and explicit external-tool readiness budgets.

Supported CPU policies cover probes, directions, centroids, PCA bases/subspaces/geometry and LDA geometry. They record algorithm/library policy, float32 input handling, numerical environment and seeds. `ProbeSpec.seed` defaults to the previous seed 42. PCA exposes `random_state` and `svd_solver`; unspecified stochastic/auto PCA remains unresolved, while explicit RNG or deterministic full/covariance solvers can qualify identity. Defaults retain legacy payload/hash compatibility.

The vLLM eager GPU numerical policy requires explicit dtype, `model_seed`, attention backend and eager execution. For example: BF16, seed 0, `FLASH_ATTN`, `enforce_eager=True`. It records actual CUDA device/compute capability, matmul settings and pinned automatic MoE/linear selection policy. Seeded kernels may remain nondeterministic; no bitwise replay is promised. Compiled/cudagraph profiles stay distinct and unresolved until observed and qualified.

## Bounds and resource separation

Modal merge functions use CPU only, with a separate default 4096 MiB / 300-second worker budget. `ModalResources` exposes `merge_memory_mb`, `merge_timeout_seconds`, `readiness_timeout_seconds` (default 600) and `client_timeout_seconds`. Streaming deadlines include queue time and progress gaps. Merge and integrity verification use bounded submit/get/cancel; unknown submission outcomes are not blindly retried. A failing ephemeral app exits its context and disconnects running work. Lock builds are CPU work and have separate recorded build evidence; a function timeout does not certify image-build duration.

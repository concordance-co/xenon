"""Observed, operation-specific numerical policies for built-in CPU analysis."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any


def cpu_analysis_policy(spec: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """No optional stack imports. Versions identify the actual installed algorithms.

    Matrix readers explicitly cast captured values to float32; estimators may
    promote intermediates as documented by the pinned implementation. Randomized
    PCA without an explicit RNG remains unidentified, including auto solver.
    """
    kind = spec.kind
    seeds: dict[str, Any] = {}
    if kind == "activation_bank":
        # This producer selects existing rows/layers without numerical reduction,
        # normalization or model execution. Upstream C01 still binds the values.
        return {"dtype": "preserve-source-values", "attention_implementation": "not_applicable",
                "device": "cpu", "determinism": "ordered-selection-no-rng", "compile": False,
                "matmul": "not_applicable", "algorithm": "activation-bank-row-layer-selection",
                "policy_version": "activation-bank-selection-v1"}, {}
    if kind == "probe":
        seed = getattr(spec, "seed", 42)
        algorithm = {"scaler": "sklearn.StandardScaler", "estimator": "sklearn.SGDClassifier",
                     "loss": "log_loss", "alpha": 1e-4, "max_iter": 2000, "tol": 1e-3,
                     "class_weight": "balanced", "shuffle": True, "split": "classification_splits-v1"}
        seeds = {"probe.base": seed, "probe.fold_fit": "base+fold", "probe.control": "base+4000+fold",
                 "probe.label_shuffle": "(base-42+fold)%2**32", "probe.persisted": "base+8958+layer"}
    elif kind == "direction":
        algorithm = {"estimator": "numpy.group_mean_difference", "normalization": "l2", "projection": "optional-fixed-subspace"}
    elif kind == "centroid":
        algorithm = {"estimator": "numpy.group_mean", "projection": "optional-fixed-subspace"}
    elif kind in {"basis", "subspace", "geometry"}:
        method = getattr(spec, "method", "pca")
        if method == "lda" and kind == "geometry":
            algorithm = {"estimator": "sklearn.LinearDiscriminantAnalysis", "solver": "svd"}
        elif method == "pca":
            solver = getattr(spec, "svd_solver", "auto")
            seed = getattr(spec, "random_state", None)
            if solver not in {"auto", "full", "arpack", "randomized", "covariance_eigh"}:
                return None, {}
            if solver not in {"full", "covariance_eigh"} and seed is None:
                return None, {}
            algorithm = {"estimator": "sklearn.PCA", "svd_solver": solver, "whiten": False,
                         "normalization": "standard_scaler" if kind == "subspace" else getattr(spec, "normalize", None)}
            seeds = {"pca.random_state": seed, "pca.rng_policy": "unused-deterministic-solver" if solver in {"full", "covariance_eigh"} else "explicit-estimator-random-state"}
        else:
            return None, {}
    elif kind == "persisted_probe_inference":
        algorithm = {"estimator": "numpy.persisted_linear_classifier", "probabilities": "softmax-or-logistic"}
    else:
        return None, {}
    try:
        libraries = {name: version(name) for name in ("numpy", "scipy", "scikit-learn")}
    except PackageNotFoundError:
        return None, {}
    return {"dtype": {"input": "float32", "intermediates": "pinned-estimator-promotions"},
            "attention_implementation": "not_applicable", "device": "cpu",
            "determinism": "fixed-inputs-algorithm-rng-and-runtime", "compile": False,
            "matmul": {"backend": "installed-numpy-scipy-blas", "libraries": libraries},
            "algorithm": algorithm, "policy_version": "cpu-analysis-v1"}, seeds


def configure_torch_threads():
    """Stable eager-adapter CPU scheduling on local and Modal hosts."""
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError('Eager adapters require 1 interop thread; use a fresh worker before starting parallel Torch work') from exc


def model_numerical_policy(spec: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Observe an explicitly configured eager vLLM policy before loading weights.

    Kernel auto-selection is an identified policy of the pinned implementation
    conditioned on the actual device. This does not promise bitwise replay.
    Compiled/cudagraph policies require separate observation and qualification.
    """
    engine = spec.bound_engine()
    if engine is not None and engine.identity().get("kind") in {"torch", "torch_native", "vit_recognition"}:
        import os
        import torch
        if engine.device not in {'cpu','cuda'} or engine.dtype != 'float32' or engine.attention_impl != 'eager':
            return None, {}
        if engine.kind in {'torch','torch_native'}:
            configure_torch_threads()
        device={'kind':engine.device,'torch_version':torch.__version__}
        if engine.device == 'cuda':
            if not torch.cuda.is_available(): return None, {}
            prop=torch.cuda.get_device_properties(torch.cuda.current_device())
            device.update(name=prop.name,compute_capability=[prop.major,prop.minor],total_memory=prop.total_memory,cuda_runtime=torch.version.cuda)
        return {'dtype':'float32', 'attention_implementation':getattr(engine, 'architecture', 'gpt2')+'-eager',
                'device':device,
                'determinism':'frozen-eval-classification-no-rng' if engine.kind == 'vit_recognition' else 'frozen-eval-weights-greedy-no-sampling', 'compile':False,
                'matmul':{'float32_precision':torch.get_float32_matmul_precision(),
                          'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                          'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                          'num_threads':torch.get_num_threads(), 'num_interop_threads':torch.get_num_interop_threads()},
                'environment':{key:os.environ[key] for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS') if key in os.environ},
                'policy_version':getattr(engine, 'architecture', 'gpt2')+'-'+engine.device+'-eager-v2'}, {}
    if engine is None or engine.identity().get("kind") != "vllm":
        return None, {}
    attention = engine.extra_llm_kwargs().get("attention_backend")
    dtype = {"half": "float16", "float": "float32"}.get(engine.dtype, engine.dtype)
    if dtype not in {"float16", "bfloat16", "float32"} or engine.model_seed is None or not engine.enforce_eager or not attention:
        return None, {}
    import torch
    if not torch.cuda.is_available():
        return None, {}
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({"name": properties.name, "compute_capability": [properties.major, properties.minor],
                        "total_memory": properties.total_memory})
    return {"dtype": dtype, "attention_implementation": attention,
        "device": {"kind": "cuda", "devices": devices, "cuda_runtime": torch.version.cuda},
        "determinism": "explicit-engine-and-request-seeds; kernels-may-be-nondeterministic",
        "compile": {"enforce_eager": True, "cudagraph": False},
        "matmul": {"float32_precision": torch.get_float32_matmul_precision(),
                   "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "kernel_selection": {"moe": "pinned-vllm-auto-on-observed-device", "linear": "pinned-vllm-auto-on-observed-device"},
        "execution": {key: engine.identity()[key] for key in ("max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill", "async_scheduling", "tensor_parallel_size", "pipeline_parallel_size")},
        "policy_version": "vllm-eager-v1"}, {"model.initialization": engine.model_seed}

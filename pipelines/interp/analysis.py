"""Phase 4: Analysis of captured MoE router logits and residual stream activations.

Runs linear probes, expert specialization analysis, and PCA visualization to decode
trading decisions, risk tolerance, and profitability from Qwen3's routing patterns.

Core logic is path-agnostic — works with local paths or Modal volume mounts.

Usage (local, after downloading activations):
    uv run --extra analysis -m pipelines.interp.analysis --mode probe --target decision_type
    uv run --extra analysis -m pipelines.interp.analysis --mode experts --target decision_type
    uv run --extra analysis -m pipelines.interp.analysis --mode pca --target decision_type
    uv run --extra analysis -m pipelines.interp.analysis --mode all --target decision_type
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    activations_dir: Path = Path("data/activations")
    output_dir: Path = Path("data/analysis_results")
    mode: str = "probe"  # probe | experts | pca | all
    target: str = "decision_type"
    data_source: str = "router"  # router | residual
    pooling: str = "last_token"  # last_token | mean_pool
    n_folds: int = 5
    layers: list[int] | None = None
    limit: int | None = None

    def run_dir(self) -> Path:
        """Per-run output directory: output_dir / YYYYMMDD_HHMMSS_{target}_{mode}"""
        from datetime import UTC, datetime
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{self.target}_{self.mode}"
        return self.output_dir / name
    seed: int = 42


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

_RISK_BINS = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


def _encode_labels(
    rows: list[dict], target: str
) -> tuple[list[dict], np.ndarray, list[str]]:
    """Filter rows and encode target labels. Returns (filtered_rows, y, class_names)."""
    filtered = []
    labels = []

    if target == "decision_type":
        for r in rows:
            dt = r.get("decision_type")
            if dt in ("trade", "record_observation"):
                filtered.append(r)
                labels.append(1 if dt == "trade" else 0)
        class_names = ["record_observation", "trade"]

    elif target == "trade_side":
        for r in rows:
            if r.get("decision_type") != "trade":
                continue
            side = r.get("trade_side")
            if side in ("buy", "sell"):
                filtered.append(r)
                labels.append(1 if side == "buy" else 0)
        class_names = ["sell", "buy"]

    elif target == "was_profitable_1h":
        for r in rows:
            if r.get("decision_type") != "trade":
                continue
            val = r.get("was_profitable_1h")
            if val is not None:
                filtered.append(r)
                labels.append(int(val))
        class_names = ["unprofitable", "profitable"]

    elif target == "risk_tolerance":
        for r in rows:
            risk = r.get("vault_risk_preference")
            if risk is not None and int(risk) in _RISK_BINS:
                filtered.append(r)
                labels.append(_RISK_BINS[int(risk)])
        class_names = ["low", "mid", "high"]

    elif target == "asset":
        # Collect unique assets, assign integer labels
        asset_set: dict[str, int] = {}
        for r in rows:
            asset = r.get("asset")
            if asset is not None:
                if asset not in asset_set:
                    asset_set[asset] = len(asset_set)
                filtered.append(r)
                labels.append(asset_set[asset])
        class_names = list(asset_set.keys())

    else:
        raise ValueError(f"Unknown target: {target}")

    return filtered, np.array(labels, dtype=np.int64), class_names


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnalysisDataset:
    """Loads safetensors activations and joins with labels from Neon."""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self._load_and_join()

    @staticmethod
    def _load_labels_from_neon() -> list[dict]:
        """Load label columns from Neon for analysis joins."""
        import psycopg
        from psycopg.rows import dict_row
        from pipelines.db import require_neon_dsn

        query = """
            SELECT log_id, decision_type, trade_side, was_profitable_1h,
                   vault_risk_preference, asset, label_quality
            FROM interp_examples_v0
            WHERE label_quality IN ('high', 'medium')
            ORDER BY log_id
        """
        with psycopg.connect(require_neon_dsn(), row_factory=dict_row) as conn:
            rows = conn.execute(query).fetchall()
        return [dict(r) for r in rows]

    def _load_and_join(self) -> None:
        # Load metadata (written during capture)
        meta_path = self.config.activations_dir / "metadata.parquet"
        meta_table = pq.read_table(meta_path)
        meta_rows = meta_table.to_pylist()
        # Normalize log_id to int for consistent joining
        meta_by_id: dict[int, dict] = {}
        for r in meta_rows:
            lid = r.get("log_id")
            if lid is not None:
                meta_by_id[int(lid)] = r
        print(f"  Metadata: {len(meta_by_id)} examples from {meta_path}")

        # Load labels from Neon
        label_rows = self._load_labels_from_neon()
        print(f"  Labels: {len(label_rows)} rows from Neon (interp_examples_v0)")

        # Inner join on log_id (normalize to int)
        joined = []
        missed_id = 0
        for lr in label_rows:
            lid = lr.get("log_id")
            if lid is None:
                continue
            lid_int = int(lid)
            if lid_int not in meta_by_id:
                missed_id += 1
                continue
            merged = {**lr, **meta_by_id[lid_int]}
            joined.append(merged)

        print(f"  Joined: {len(joined)} examples "
              f"(missed: {missed_id} no activation)")

        # Check for activation files using a single directory listing
        # instead of N individual stat() calls (much faster on NAS).
        source = self.config.data_source
        subdir = "router_logits" if source == "router" else "residual_stream"
        act_dir = self.config.activations_dir / subdir
        try:
            existing_files = set(p.name for p in act_dir.iterdir()) if act_dir.is_dir() else set()
        except OSError:
            existing_files = set()
        files_found = sum(
            1 for r in joined
            if f"{r['log_id']}.safetensors" in existing_files
        )
        print(f"  Activation files ({subdir}): {files_found}/{len(joined)} found")

        if self.config.limit:
            joined = joined[: self.config.limit]

        # Encode labels for the target
        self.rows, self.y, self.class_names = _encode_labels(
            joined, self.config.target
        )

        if len(self.rows) == 0:
            raise ValueError(
                f"No valid examples after filtering for target={self.config.target}. "
                f"Joined {len(joined)} examples but none had valid "
                f"'{self.config.target}' labels."
            )

        # Determine available layers and build index mapping
        sample_meta = meta_by_id.get(int(self.rows[0]["log_id"]), {})
        self.num_layers = sample_meta.get("num_layers_captured", 48)
        self.num_experts = sample_meta.get("num_experts", 128)
        self.hidden_dim = sample_meta.get("hidden_dim", 2048)

        # Build layer-number → tensor-index mapping
        # captured_layers is a JSON list of actual layer numbers, e.g. [0,16,32,47]
        # tensor axis 0 corresponds to these in sorted order
        import json as _json
        captured_layers_raw = sample_meta.get("captured_layers")
        if captured_layers_raw:
            if isinstance(captured_layers_raw, str):
                self.captured_layers = _json.loads(captured_layers_raw)
            else:
                self.captured_layers = list(captured_layers_raw)
        elif self.num_layers < 48:
            # Legacy metadata without captured_layers — fewer layers than
            # a full model means a subset was captured but we don't know which.
            # Use sequential indices as a fallback (probes still work, just
            # layer labels won't match the model's actual layer numbers).
            print(f"  Warning: metadata missing 'captured_layers' field. "
                  f"Using indices 0..{self.num_layers - 1} as layer labels.")
            self.captured_layers = list(range(self.num_layers))
        else:
            self.captured_layers = list(range(self.num_layers))
        self._layer_to_idx = {layer: idx for idx, layer in enumerate(self.captured_layers)}

    def preload_features(self, layers: list[int] | None = None, include_router_indices: bool = False) -> None:
        """Load features into cache. Uses compact files if available, else per-example files."""
        target_layers = layers or self.captured_layers

        # Try compact files first
        if self._try_load_compact(target_layers, include_router_indices):
            return

        self._load_per_example(target_layers, include_router_indices)

    def _try_load_compact(self, target_layers: list[int], include_router_indices: bool) -> bool:
        """Attempt to load from compact files. Returns True on success."""
        from safetensors import safe_open

        source = self.config.data_source
        pooling = self.config.pooling
        source_tag = "router" if source == "router" else "residual"
        compact_dir = self.config.activations_dir / "compact"

        # Check all needed files exist
        for layer in target_layers:
            path = compact_dir / f"{source_tag}_{pooling}_layer{layer}.safetensors"
            if not path.exists():
                return False

        if include_router_indices:
            for layer in target_layers:
                # Router indices may be in the router compact file or the current source file
                ri_path = compact_dir / f"router_{pooling}_layer{layer}.safetensors"
                feat_path = compact_dir / f"{source_tag}_{pooling}_layer{layer}.safetensors"
                if not ri_path.exists() and not feat_path.exists():
                    return False

        print(f"  Loading from compact files...")

        # Build log_id → row index for this dataset's rows
        row_log_ids = {int(r["log_id"]): i for i, r in enumerate(self.rows)}

        self._feature_cache = {}
        if include_router_indices:
            self._ri_cache = {}
        self._valid_mask = [False] * len(self.rows)

        for layer in target_layers:
            path = compact_dir / f"{source_tag}_{pooling}_layer{layer}.safetensors"
            with safe_open(str(path), framework="numpy") as f:
                features = f.get_tensor("features")
                log_ids = f.get_tensor("log_ids")

            # Map compact rows to this dataset's rows
            layer_cache: list[np.ndarray | None] = [None] * len(self.rows)
            for ci, lid in enumerate(log_ids):
                row_idx = row_log_ids.get(int(lid))
                if row_idx is not None:
                    layer_cache[row_idx] = features[ci]
                    self._valid_mask[row_idx] = True

            self._feature_cache[layer] = layer_cache

            if include_router_indices:
                ri_path = compact_dir / f"router_{pooling}_layer{layer}.safetensors"

                ri_layer_cache: list[np.ndarray | None] = [None] * len(self.rows)
                ri_loaded = False

                # Try dedicated router file first
                if ri_path.exists():
                    with safe_open(str(ri_path), framework="numpy") as rf:
                        if "router_indices" in rf.keys():
                            ri_data = rf.get_tensor("router_indices")
                            ri_log_ids = rf.get_tensor("log_ids")
                            for ci, lid in enumerate(ri_log_ids):
                                row_idx = row_log_ids.get(int(lid))
                                if row_idx is not None:
                                    ri_layer_cache[row_idx] = ri_data[ci]
                            ri_loaded = True

                # Fall back to checking the already-opened source file's keys
                # (re-open once instead of open-to-check then open-to-read)
                if not ri_loaded and path.exists():
                    with safe_open(str(path), framework="numpy") as rf:
                        if "router_indices" in rf.keys():
                            ri_data = rf.get_tensor("router_indices")
                            ri_log_ids = rf.get_tensor("log_ids")
                            for ci, lid in enumerate(ri_log_ids):
                                row_idx = row_log_ids.get(int(lid))
                                if row_idx is not None:
                                    ri_layer_cache[row_idx] = ri_data[ci]

                self._ri_cache[layer] = ri_layer_cache

        valid_count = sum(self._valid_mask)
        print(f"  Loaded {valid_count} examples from {len(target_layers)} compact files")
        return True

    def _load_per_example(self, target_layers: list[int], include_router_indices: bool) -> None:
        """Fallback: load features from individual per-example safetensor files.

        Opens each safetensor file exactly once and extracts all needed layers
        in a single pass. When router indices are needed and the source is
        already 'router', they are read from the same open file handle to
        avoid a redundant second open.
        """
        from safetensors import safe_open

        source = self.config.data_source
        pooling = self.config.pooling

        if source == "router":
            subdir = "router_logits"
            key = "router_logits"
        else:
            subdir = "residual_stream"
            key = "residual_stream"

        # When source is router, router_indices live in the same file
        ri_same_file = include_router_indices and source == "router"

        self._feature_cache: dict[int, list[np.ndarray | None]] = {
            layer: [] for layer in target_layers
        }
        if include_router_indices:
            self._ri_cache: dict[int, list[np.ndarray | None]] = {
                layer: [] for layer in target_layers
            }
        self._valid_mask: list[bool] = []

        for row in self.rows:
            log_id = row["log_id"]
            path = self.config.activations_dir / subdir / f"{log_id}.safetensors"

            if not path.exists():
                self._valid_mask.append(False)
                for layer in target_layers:
                    self._feature_cache[layer].append(None)
                    if include_router_indices:
                        self._ri_cache[layer].append(None)
                continue

            # Single open: read main tensor and (if same file) router indices
            with safe_open(str(path), framework="numpy") as f:
                tensor = f.get_tensor(key)
                if ri_same_file and "router_indices" in f.keys():
                    ri_tensor = f.get_tensor("router_indices")
                elif ri_same_file:
                    ri_tensor = None
                else:
                    ri_tensor = None

            # Only open a separate file when source != router and we need
            # router_indices from the router_logits directory
            if include_router_indices and not ri_same_file:
                ri_path = self.config.activations_dir / "router_logits" / f"{log_id}.safetensors"
                if ri_path.exists():
                    with safe_open(str(ri_path), framework="numpy") as rf:
                        ri_tensor = rf.get_tensor("router_indices")

            self._valid_mask.append(True)
            for layer in target_layers:
                tensor_idx = self._layer_to_idx[layer]
                layer_data = tensor[tensor_idx]

                if layer_data.ndim == 1:
                    vec = layer_data
                elif pooling == "last_token":
                    vec = layer_data[-1]
                else:
                    vec = layer_data.mean(axis=0)

                self._feature_cache[layer].append(vec)

                if include_router_indices:
                    if ri_tensor is not None:
                        layer_ri = ri_tensor[tensor_idx]
                        if layer_ri.ndim == 1:
                            self._ri_cache[layer].append(layer_ri)
                        elif pooling == "last_token":
                            self._ri_cache[layer].append(layer_ri[-1])
                        else:
                            self._ri_cache[layer].append(layer_ri.reshape(-1))
                    else:
                        self._ri_cache[layer].append(None)

        valid_count = sum(self._valid_mask)
        total = len(self.rows)
        if valid_count < total:
            print(f"  Preloaded features: {valid_count}/{total} files found")
        else:
            print(f"  Preloaded features: {valid_count} examples")

    def get_features(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Load features for a single layer. Returns (X, y) as numpy arrays.

        `layer` is the actual model layer number (e.g. 16), which is mapped
        to the tensor index via captured_layers metadata.
        """
        tensor_idx = self._layer_to_idx.get(layer)
        if tensor_idx is None:
            raise ValueError(
                f"Layer {layer} was not captured. "
                f"Available layers: {self.captured_layers}"
            )

        # Use cache if available
        if hasattr(self, '_feature_cache') and layer in self._feature_cache:
            features = [v for v in self._feature_cache[layer] if v is not None]
            valid_y = self.y[np.array(self._valid_mask)]
            if len(features) == 0:
                return np.empty((0, 0)), valid_y
            return np.stack(features, axis=0), valid_y

        # Fallback: preload ALL captured layers in a single pass over files,
        # then return the requested layer. This avoids reopening every file
        # when get_features is called once per layer (e.g. 48 layers ×
        # N examples = 48N opens reduced to N).
        self._load_per_example(self.captured_layers, include_router_indices=False)
        return self.get_features(layer)  # now served from cache

    def get_router_indices(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        """Load router_indices for a single layer. Returns (indices, y)."""
        tensor_idx = self._layer_to_idx.get(layer)
        if tensor_idx is None:
            raise ValueError(
                f"Layer {layer} was not captured. "
                f"Available layers: {self.captured_layers}"
            )

        # Use cache if available
        if hasattr(self, '_ri_cache') and layer in self._ri_cache:
            indices_list = [v for v in self._ri_cache[layer] if v is not None]
            valid_y = self.y[np.array(self._valid_mask)]
            return np.stack(indices_list) if indices_list else np.empty((0, 0)), valid_y

        # Fallback: preload ALL captured layers (with router indices) in a
        # single pass, then return the requested layer from cache.
        self._load_per_example(self.captured_layers, include_router_indices=True)
        return self.get_router_indices(layer)  # now served from cache


# ---------------------------------------------------------------------------
# Probe sweep
# ---------------------------------------------------------------------------

def _probe_one_layer(layer, X, y, n_folds, seed, rng_state):
    """Train probes for a single layer. Returns result dict or None."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import cross_validate, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_valid = len(y)
    n_unique = len(np.unique(y))
    if n_valid < 4 or n_unique < 2:
        return None

    min_class_count = min(np.bincount(y))
    actual_folds = max(2, min(n_folds, len(y), min_class_count))

    def _make_pipe():
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss", alpha=1e-4,
                class_weight="balanced",
                max_iter=2000, tol=1e-3,
                random_state=seed,
            ),
        )

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)

    # Run real + shuffled CV in parallel across folds
    rng = np.random.RandomState(rng_state)
    y_shuffled = rng.permutation(y)

    # Majority baseline
    counts = np.bincount(y)
    baseline_majority = float(counts.max()) / len(y)

    cv_results = cross_validate(
        _make_pipe(), X, y, cv=cv,
        scoring=["accuracy", "balanced_accuracy"],
        n_jobs=-1,
    )
    scores = cv_results["test_accuracy"]
    balanced_scores = cv_results["test_balanced_accuracy"]

    try:
        s = cross_validate(
            _make_pipe(), X, y_shuffled, cv=actual_folds,
            scoring=["accuracy"], n_jobs=-1,
        )
        baseline_shuffled = float(s["test_accuracy"].mean())
    except Exception:
        baseline_shuffled = baseline_majority

    selectivity = float(scores.mean()) - baseline_shuffled

    return {
        "layer": layer,
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "balanced_accuracy": round(float(balanced_scores.mean()), 4),
        "baseline_majority": round(baseline_majority, 4),
        "baseline_shuffled": round(baseline_shuffled, 4),
        "selectivity": round(selectivity, 4),
        "n_examples": len(y),
        "n_classes": len(np.unique(y)),
    }


def run_probe_sweep(config: AnalysisConfig) -> None:
    """Run linear probe across layers, compute baselines and selectivity."""
    from joblib import Parallel, delayed

    dataset = AnalysisDataset(config)
    layers = config.layers or dataset.captured_layers

    actual_classes = len(np.unique(dataset.y))
    counts = np.bincount(dataset.y)
    dist = ", ".join(
        f"{dataset.class_names[i]}={c}"
        for i, c in enumerate(counts)
        if i < len(dataset.class_names)
    )
    print(f"\nProbe sweep: target={config.target}, source={config.data_source}, "
          f"pooling={config.pooling}")
    print(f"  {len(dataset.rows)} examples, {actual_classes} classes present: {dist}")
    if actual_classes < 2:
        print(f"  ERROR: need at least 2 classes for probes. "
              f"All examples are '{dataset.class_names[dataset.y[0]]}'. "
              f"Try a different target or ingest more vaults for diversity.")
        return []
    print(f"  Layers: {layers[0]}..{layers[-1]} ({len(layers)} total)")

    # Preload all features once (avoids reopening files per layer)
    dataset.preload_features(layers)
    print()

    # Prepare per-layer data + deterministic RNG seeds
    rng = np.random.RandomState(config.seed)
    layer_args = []
    for layer in layers:
        X, y = dataset.get_features(layer)
        rng_state = rng.randint(0, 2**31)
        layer_args.append((layer, X, y, rng_state))

    print(f"  Running {len(layer_args)} layers in parallel...")
    raw_results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_probe_one_layer)(layer, X, y, config.n_folds, config.seed, rng_state)
        for layer, X, y, rng_state in layer_args
    )

    results = []
    for args, row in zip(layer_args, raw_results):
        layer = args[0]
        if row is None:
            n_valid = len(args[2])
            n_unique = len(np.unique(args[2]))
            reason = "single class" if n_unique < 2 else f"too few examples ({n_valid})"
            print(f"  Layer {layer:3d}: skipped ({reason})")
            continue
        results.append(row)
        marker = "***" if row['selectivity'] > 0.05 else ""
        print(
            f"  Layer {layer:3d}: acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
            f"bal={row['balanced_accuracy']:.3f}  "
            f"sel={row['selectivity']:+.3f}  "
            f"(maj={row['baseline_majority']:.3f}) {marker}"
        )

    if results:
        import pyarrow as pa

        config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = config.output_dir / f"probe_{config.target}_{config.data_source}.parquet"
        table = pa.Table.from_pylist(results)
        pq.write_table(table, out_path, compression="snappy")
        print(f"\n  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# Expert specialization
# ---------------------------------------------------------------------------

def run_expert_analysis(config: AnalysisConfig) -> None:
    """Compute per-expert activation frequency by class and discriminative scores."""
    dataset = AnalysisDataset(config)
    layers = config.layers or dataset.captured_layers

    print(f"\nExpert specialization: target={config.target}")
    print(f"  {len(dataset.rows)} examples, classes: {dataset.class_names}")
    print(f"  Layers: {layers[0]}..{layers[-1]} ({len(layers)} total)")

    dataset.preload_features(layers, include_router_indices=True)
    print()

    results = []

    for layer in layers:
        indices, y = dataset.get_router_indices(layer)

        if indices.size == 0 or len(np.unique(y)) < 2:
            continue

        num_experts = dataset.num_experts
        classes = np.unique(y)

        # Per-expert frequency by class
        freq_by_class = {}
        for cls in classes:
            cls_indices = indices[y == cls]
            expert_counts = np.bincount(cls_indices.ravel().astype(np.int64), minlength=num_experts)
            total = cls_indices.size if cls_indices.size > 0 else 1
            freq_by_class[cls] = expert_counts / total

        # For binary: Cohen's d between class frequencies
        if len(classes) == 2:
            f0 = freq_by_class[classes[0]]
            f1 = freq_by_class[classes[1]]
            pooled_std = np.sqrt((f0.var() + f1.var()) / 2)
            pooled_std = max(pooled_std, 1e-8)
            cohens_d = (f1 - f0) / pooled_std
        else:
            # Multi-class: use max pairwise difference as discriminative score
            all_freqs = np.stack([freq_by_class[c] for c in classes])
            cohens_d = all_freqs.max(axis=0) - all_freqs.min(axis=0)

        # Top 10 most discriminative experts
        top_experts = np.argsort(np.abs(cohens_d))[::-1][:10]

        for rank, expert_id in enumerate(top_experts):
            row = {
                "layer": layer,
                "expert_id": int(expert_id),
                "rank": rank,
                "discriminative_score": round(float(cohens_d[expert_id]), 4),
            }
            for cls in classes:
                row[f"freq_class_{cls}"] = round(float(freq_by_class[cls][expert_id]), 6)
            results.append(row)

        print(
            f"  Layer {layer:3d}: top expert={top_experts[0]} "
            f"(d={cohens_d[top_experts[0]]:+.3f})"
        )

    if results:
        import pyarrow as pa

        config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = config.output_dir / "expert_specialization.parquet"
        table = pa.Table.from_pylist(results)
        pq.write_table(table, out_path, compression="snappy")
        print(f"\n  Saved: {out_path}")

    return results


# ---------------------------------------------------------------------------
# PCA visualization
# ---------------------------------------------------------------------------

def run_pca(config: AnalysisConfig) -> None:
    """PCA + LDA + difference-in-means visualizations for representative layers.

    Generates three plots per layer:
      1. Unsupervised PCA (top-2 variance directions)
      2. LDA projection (directions that maximally separate classes)
      3. Difference-in-means 1D histogram (binary) or class-mean PCA (multi-class)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    dataset = AnalysisDataset(config)
    L = dataset.num_layers

    if config.layers:
        layers = config.layers
    else:
        layers = sorted(set([0, L // 3, 2 * L // 3, L - 1]))

    print(f"\nPCA/LDA visualization: target={config.target}, source={config.data_source}")
    print(f"  {len(dataset.rows)} examples, classes: {dataset.class_names}")
    print(f"  Layers: {layers}")

    dataset.preload_features(layers)
    print()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        X, y = dataset.get_features(layer)
        n_classes = len(np.unique(y))

        if n_classes < 2:
            print(f"  Layer {layer}: skipped (single class)")
            continue

        # ── 1. Unsupervised PCA ──
        pca = PCA(n_components=min(2, X.shape[1]))
        X2d = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls_idx in np.unique(y):
            mask = y == cls_idx
            label = dataset.class_names[cls_idx] if cls_idx < len(dataset.class_names) else str(cls_idx)
            ax.scatter(X2d[mask, 0], X2d[mask, 1], alpha=0.6, s=20, label=label)
        var = pca.explained_variance_ratio_
        ax.set_title(
            f"Layer {layer} — {config.target} ({config.data_source})\n"
            f"PC1: {var[0]:.1%}, PC2: {var[1]:.1%}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()
        out_path = config.output_dir / f"pca_layer{layer}_{config.target}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ── 2. LDA projection (supervised) ──
        n_lda = min(n_classes - 1, 2, X.shape[1])
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            lda = LinearDiscriminantAnalysis(n_components=n_lda)
            X_lda = lda.fit_transform(X_scaled, y)

            fig, ax = plt.subplots(figsize=(8, 6))
            for cls_idx in np.unique(y):
                mask = y == cls_idx
                label = dataset.class_names[cls_idx] if cls_idx < len(dataset.class_names) else str(cls_idx)
                if n_lda >= 2:
                    ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.6, s=20, label=label)
                else:
                    # 1D LDA: use jittered y-axis for visibility
                    jitter = np.random.default_rng(42).normal(0, 0.1, size=mask.sum())
                    ax.scatter(X_lda[mask, 0], jitter, alpha=0.6, s=20, label=label)
            title = f"Layer {layer} — LDA — {config.target} ({config.data_source})"
            if n_lda >= 2:
                ax.set_xlabel("LD1")
                ax.set_ylabel("LD2")
            else:
                ax.set_xlabel("LD1")
                ax.set_ylabel("(jitter)")
            ax.set_title(title)
            ax.legend()
            out_path = config.output_dir / f"lda_layer{layer}_{config.target}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"  Layer {layer}: LDA failed ({e})")

        # ── 3. Difference-in-means ──
        # For binary: 1D histogram along the class-mean difference vector
        # For multi-class: PCA on class centroids (supervised PCA à la counting_manifolds)
        if n_classes == 2:
            classes = np.unique(y)
            mean_a = X[y == classes[0]].mean(axis=0)
            mean_b = X[y == classes[1]].mean(axis=0)
            direction = mean_a - mean_b
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            projections = X @ direction

            fig, ax = plt.subplots(figsize=(8, 4))
            for cls_idx in classes:
                mask = y == cls_idx
                label = dataset.class_names[cls_idx] if cls_idx < len(dataset.class_names) else str(cls_idx)
                ax.hist(projections[mask], bins=40, alpha=0.6, label=label, density=True)
            ax.set_title(f"Layer {layer} — Diff-in-Means — {config.target}")
            ax.set_xlabel("Projection onto class-mean difference")
            ax.set_ylabel("Density")
            ax.legend()
            out_path = config.output_dir / f"dim_layer{layer}_{config.target}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            # Multi-class: PCA on class centroids
            centroids = np.stack([X[y == c].mean(axis=0) for c in np.unique(y)])
            n_comp = min(2, centroids.shape[0], centroids.shape[1])
            pca_c = PCA(n_components=n_comp)
            centroids_2d = pca_c.fit_transform(centroids)
            # Project all points into centroid PCA space
            X_proj = pca_c.transform(X)

            fig, ax = plt.subplots(figsize=(8, 6))
            for i, cls_idx in enumerate(np.unique(y)):
                mask = y == cls_idx
                label = dataset.class_names[cls_idx] if cls_idx < len(dataset.class_names) else str(cls_idx)
                if n_comp >= 2:
                    ax.scatter(X_proj[mask, 0], X_proj[mask, 1], alpha=0.4, s=15, label=label)
                    ax.scatter(centroids_2d[i, 0], centroids_2d[i, 1], s=200, marker='*',
                               edgecolors='black', linewidths=0.5, zorder=5)
                else:
                    jitter = np.random.default_rng(42).normal(0, 0.1, size=mask.sum())
                    ax.scatter(X_proj[mask, 0], jitter, alpha=0.4, s=15, label=label)
            var_c = pca_c.explained_variance_ratio_
            ax.set_title(
                f"Layer {layer} — Centroid PCA — {config.target}\n"
                f"PC1: {var_c[0]:.1%}" + (f", PC2: {var_c[1]:.1%}" if n_comp >= 2 else "")
            )
            ax.set_xlabel("Centroid PC1")
            ax.set_ylabel("Centroid PC2" if n_comp >= 2 else "(jitter)")
            ax.legend()
            out_path = config.output_dir / f"centroid_pca_layer{layer}_{config.target}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  Layer {layer}: saved PCA + LDA + {'diff-in-means' if n_classes == 2 else 'centroid PCA'}")


# ---------------------------------------------------------------------------
# Compact: consolidate per-example safetensors into per-layer matrices
# ---------------------------------------------------------------------------

def run_compact(config: AnalysisConfig) -> dict:
    """Consolidate per-example safetensors into per-layer matrices.

    Reads all individual safetensor files once, pools to a single vector per
    example, and writes consolidated files:
      {activations_dir}/compact/{source}_{pooling}_layer{N}.safetensors
    Each contains:
      - "features": (n_examples, dim) float32
      - "log_ids": (n_examples,) int64
    And for router source:
      - "router_indices": (n_examples, top_k) int64

    Analysis can then load one file per layer instead of N files.
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    meta_path = config.activations_dir / "metadata.parquet"
    meta_table = pq.read_table(meta_path)
    meta_rows = meta_table.to_pylist()

    # Determine layers from first example
    import json as _json
    sample = meta_rows[0]
    captured_raw = sample.get("captured_layers")
    if captured_raw:
        if isinstance(captured_raw, str):
            captured_layers = _json.loads(captured_raw)
        else:
            captured_layers = list(captured_raw)
    else:
        captured_layers = list(range(sample.get("num_layers_captured", 48)))

    layer_to_idx = {layer: idx for idx, layer in enumerate(captured_layers)}
    layers = config.layers or captured_layers

    sources = []
    if config.data_source == "router":
        sources.append(("router_logits", "router_logits"))
    elif config.data_source == "residual":
        sources.append(("residual_stream", "residual_stream"))
    else:
        sources.append(("residual_stream", "residual_stream"))
        sources.append(("router_logits", "router_logits"))

    pooling = config.pooling

    compact_dir = config.activations_dir / "compact"
    compact_dir.mkdir(parents=True, exist_ok=True)

    result = {"layers": layers, "files": []}

    for subdir, key in sources:
        print(f"\nCompacting {subdir} ({pooling})...")

        # Collect per-layer lists
        layer_features: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        layer_ri: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        log_ids: list[int] = []
        skipped = 0

        for i, row in enumerate(meta_rows):
            log_id = int(row["log_id"])
            path = config.activations_dir / subdir / f"{log_id}.safetensors"

            if not path.exists():
                skipped += 1
                continue

            with safe_open(str(path), framework="numpy") as f:
                tensor = f.get_tensor(key)
                has_ri = "router_indices" in f.keys()
                ri_tensor = f.get_tensor("router_indices") if has_ri else None

            log_ids.append(log_id)

            for layer in layers:
                tidx = layer_to_idx[layer]
                layer_data = tensor[tidx]

                if layer_data.ndim == 1:
                    vec = layer_data
                elif pooling == "last_token":
                    vec = layer_data[-1]
                else:
                    vec = layer_data.mean(axis=0)

                layer_features[layer].append(vec)

                if ri_tensor is not None:
                    layer_ri_data = ri_tensor[tidx]
                    if layer_ri_data.ndim == 1:
                        layer_ri[layer].append(layer_ri_data)
                    elif pooling == "last_token":
                        layer_ri[layer].append(layer_ri_data[-1])
                    else:
                        layer_ri[layer].append(layer_ri_data.reshape(-1))

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(meta_rows)} files read...")

        log_id_arr = np.array(log_ids, dtype=np.int64)
        print(f"  {len(log_ids)} examples loaded ({skipped} skipped)")

        source_tag = "router" if subdir == "router_logits" else "residual"

        for layer in layers:
            out_path = compact_dir / f"{source_tag}_{pooling}_layer{layer}.safetensors"
            tensors = {
                "features": np.stack(layer_features[layer]).astype(np.float32),
                "log_ids": log_id_arr,
            }
            if layer_ri[layer]:
                tensors["router_indices"] = np.stack(layer_ri[layer]).astype(np.int64)

            save_file(tensors, str(out_path))
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"  Layer {layer}: {out_path.name} ({size_mb:.1f} MB)")
            result["files"].append(str(out_path))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze captured MoE activations",
    )
    p.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    p.add_argument("--output-dir", type=Path, default=Path("data/analysis_results"))
    p.add_argument("--mode", choices=["probe", "experts", "pca", "all", "compact"], default="probe")
    p.add_argument("--target", default="decision_type",
                   choices=["decision_type", "trade_side", "was_profitable_1h",
                            "risk_tolerance", "asset"])
    p.add_argument("--data-source", choices=["router", "residual"], default="router")
    p.add_argument("--pooling", choices=["last_token", "mean_pool"], default="last_token")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--layers", type=str, default="",
                   help="Comma-separated layer indices (default: all)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    parsed_layers: list[int] | None = None
    if args.layers:
        parsed_layers = [int(x.strip()) for x in args.layers.split(",")]

    config = AnalysisConfig(
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        target=args.target,
        data_source=args.data_source,
        pooling=args.pooling,
        n_folds=args.n_folds,
        layers=parsed_layers,
        limit=args.limit if args.limit > 0 else None,
        seed=args.seed,
    )

    dispatch(config)


def _needs_compact(config: AnalysisConfig) -> bool:
    """Check if compact files are missing for the current config."""
    compact_dir = config.activations_dir / "compact"
    if not compact_dir.exists():
        return True
    source_tag = config.data_source  # "router" or "residual"
    pooling = config.pooling
    # Check if at least one compact file exists for this source+pooling
    pattern = f"{source_tag}_{pooling}_layer*.safetensors"
    return not any(compact_dir.glob(pattern))


def dispatch(config: AnalysisConfig) -> dict:
    """Run analysis based on config.mode. Returns results dict."""
    # Compact writes to the base output_dir; everything else gets a per-run subdir
    if config.mode != "compact":
        config.output_dir = config.run_dir()
        print(f"Output directory: {config.output_dir}")

    results = {}

    if config.mode == "compact":
        results["compact"] = run_compact(config)
        return results

    # Auto-compact if compact files don't exist yet
    if _needs_compact(config):
        print("Compact files not found — running auto-compaction first...")
        results["compact"] = run_compact(config)

    if config.mode in ("probe", "all"):
        results["probe"] = run_probe_sweep(config)

    if config.mode in ("experts", "all"):
        results["experts"] = run_expert_analysis(config)

    if config.mode in ("pca", "all"):
        run_pca(config)
        results["pca"] = True

    return results


if __name__ == "__main__":
    main()

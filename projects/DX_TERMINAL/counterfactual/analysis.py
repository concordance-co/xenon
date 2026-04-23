"""Analysis pipeline for the Config Entanglement vs. Late Policy experiment.

Implements:
  - Linear probe training and transfer testing
  - Per-snapshot metrics (AUROC, Hit@1, MRR, balanced_accuracy)
  - CKA (Centered Kernel Alignment) between variant pairs
  - Orthogonal Procrustes alignment and restored transfer
  - Router Jaccard / JSD at market-row positions
  - Bootstrap confidence intervals (unit = vault_day)
  - Symbol-only and row-index-only baseline probes

See plan: .claude/plans/woolly-snacking-bear.md
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualAnalysisConfig:
    activations_dir: Path = Path("data/activations/counterfactual")
    experiment_id: str = "default"
    output_dir: Path = Path("data/analysis_results/counterfactual")
    n_bootstrap: int = 1000
    seed: int = 42
    layers: list[int] | None = None  # None = all

    @property
    def run_dir(self) -> Path:
        return self.activations_dir / self.experiment_id

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self.experiment_id


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset_a_spec(dataset_dir: Path | None = None) -> dict[str, Any]:
    """Load Dataset A specification from Neon DB.

    Queries counterfactual_snapshots for snapshots used in Dataset A prompts,
    along with their train/test split and labels.

    dataset_dir is accepted for backward compat but ignored.
    """
    from pipelines.db import connect_neon

    conn = connect_neon()
    try:
        # Get all snapshot_ids referenced by dataset A prompts
        rows = conn.execute("""
            SELECT DISTINCT s.snapshot_id, s.vault_address, s.snap_date,
                   s.n_rows, s.labels, s.split, s.row_order
            FROM counterfactual_snapshots s
            JOIN counterfactual_prompts p ON p.snapshot_id = s.snapshot_id
            WHERE p.dataset = 'a'
            ORDER BY s.snapshot_id
        """).fetchall()
    finally:
        conn.close()

    snapshots = []
    train_ids: set[str] = set()
    test_ids: set[str] = set()

    for r in rows:
        snap = {
            "snapshot_id": r["snapshot_id"],
            "vault_address": r["vault_address"],
            "snap_date": str(r["snap_date"]),
            "n_rows": r["n_rows"],
            "labels": r["labels"] if isinstance(r["labels"], dict) else json.loads(r["labels"]),
            "row_order": list(r["row_order"]) if r["row_order"] else [],
        }
        snapshots.append(snap)
        if r["split"] == "train":
            train_ids.add(r["snapshot_id"])
        elif r["split"] == "test":
            test_ids.add(r["snapshot_id"])

    print(f"Loaded {len(snapshots)} snapshots from DB ({len(train_ids)} train, {len(test_ids)} test)")
    return {
        "snapshots": snapshots,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def load_dataset_b_spec() -> dict[str, Any]:
    """Load Dataset B specification from Neon DB.

    Returns snapshots with labels, plus train/test split.
    Dataset B snapshots have split='b_source' in the DB.
    """
    from pipelines.db import connect_neon

    conn = connect_neon()
    try:
        rows = conn.execute("""
            SELECT DISTINCT s.snapshot_id, s.vault_address, s.snap_date,
                   s.n_rows, s.labels, s.row_order
            FROM counterfactual_snapshots s
            JOIN counterfactual_prompts p ON p.snapshot_id = s.snapshot_id
            WHERE p.dataset = 'b'
            ORDER BY s.snapshot_id
        """).fetchall()
    finally:
        conn.close()

    snapshots = []
    for r in rows:
        snap = {
            "snapshot_id": r["snapshot_id"],
            "vault_address": r["vault_address"],
            "snap_date": str(r["snap_date"]),
            "n_rows": r["n_rows"],
            "labels": r["labels"] if isinstance(r["labels"], dict) else json.loads(r["labels"]),
            "row_order": list(r["row_order"]) if r["row_order"] else [],
        }
        snapshots.append(snap)

    # 80/20 train/test split by snapshot
    all_ids = [s["snapshot_id"] for s in snapshots]
    n_test = max(1, len(all_ids) // 5)
    test_ids = set(all_ids[:n_test])
    train_ids = set(all_ids[n_test:])

    print(f"Loaded {len(snapshots)} Dataset B snapshots from DB ({len(train_ids)} train, {len(test_ids)} test)")
    return {
        "snapshots": snapshots,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def load_pooled_activations(
    capture_dir: Path,
    capture_id: str,
    _cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Load section-pooled activations from safetensors (with optional cache)."""
    if _cache is not None and capture_id in _cache:
        return _cache[capture_id]
    from safetensors.numpy import load_file
    path = capture_dir / "residual" / f"{capture_id}.safetensors"
    if not path.exists():
        return {}
    data = load_file(str(path))
    if _cache is not None:
        _cache[capture_id] = data
    return data


def preload_all_activations(
    run_dir: Path,
    capture_ids: list[str],
    max_workers: int = 16,
) -> dict[str, dict[str, np.ndarray]]:
    """Bulk-load all pooled activations into memory using concurrent I/O.

    Returns dict mapping capture_id -> {key: ndarray}.
    Pooled files are ~4.5MB each, so ~1100 captures ≈ 5GB — fits in RAM.
    """
    from safetensors.numpy import load_file

    residual_dir = run_dir / "residual"
    t0 = time.monotonic()

    def _load_one(cid: str) -> tuple[str, dict[str, np.ndarray] | None]:
        path = residual_dir / f"{cid}.safetensors"
        if path.exists():
            return cid, load_file(str(path))
        return cid, None

    cache: dict[str, dict[str, np.ndarray]] = {}
    loaded = 0
    missing = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one, cid): cid for cid in capture_ids}
        for fut in as_completed(futures):
            cid, data = fut.result()
            if data is not None:
                cache[cid] = data
                loaded += 1
            else:
                missing += 1

    elapsed = time.monotonic() - t0
    total_mb = sum(
        sum(v.nbytes for v in d.values())
        for d in cache.values()
    ) / 1024 / 1024
    print(
        f"Preloaded {loaded} captures ({total_mb:.0f} MB) in {elapsed:.1f}s"
        f"{f', {missing} missing' if missing else ''}"
    )
    return cache


# ---------------------------------------------------------------------------
# Per-snapshot metrics
# ---------------------------------------------------------------------------

def compute_auroc_within_snapshot(
    scores: np.ndarray,  # (n_rows,) predicted probabilities for positive class
    labels: np.ndarray,  # (n_rows,) binary labels
) -> float | None:
    """Compute AUROC within a single snapshot's rows.

    Returns None if the snapshot has no positive or all-positive labels
    (AUROC undefined).
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def compute_hit_at_1(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Does the highest-scored row match the true leader?"""
    if labels.sum() == 0:
        return 0.0
    pred_top = np.argmax(scores)
    true_top = np.argmax(labels)
    return 1.0 if pred_top == true_top else 0.0


def compute_mrr(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean reciprocal rank of the true leader."""
    if labels.sum() == 0:
        return 0.0
    true_idx = np.argmax(labels)
    # Rank by descending score
    ranked = np.argsort(-scores)
    rank = int(np.where(ranked == true_idx)[0][0]) + 1
    return 1.0 / rank


def compute_balanced_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Balanced accuracy within a snapshot."""
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(labels, preds))


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_probe(
    X_train: np.ndarray,  # (n_train_rows, hidden_dim)
    y_train: np.ndarray,  # (n_train_rows,) binary
    seed: int = 42,
) -> Any:
    """Train a logistic regression probe. Returns (scaler, classifier) pipeline."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_probe_per_snapshot(
    probe: Any,
    snapshot_groups: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Evaluate a trained probe on grouped snapshots.

    snapshot_groups: list of dicts with keys:
        - X: (n_rows, hidden_dim)
        - y: (n_rows,) binary labels
        - snapshot_id: str

    Returns dict of metric_name -> list of per-snapshot values.
    """
    aurocs: list[float] = []
    hit1s: list[float] = []
    mrrs: list[float] = []
    bal_accs: list[float] = []

    for group in snapshot_groups:
        X = group["X"]
        y = group["y"]

        if len(np.unique(y)) < 2:
            continue

        scores = probe.predict_proba(X)[:, 1]
        preds = probe.predict(X)

        auroc = compute_auroc_within_snapshot(scores, y)
        if auroc is not None:
            aurocs.append(auroc)

        hit1s.append(compute_hit_at_1(scores, y))
        mrrs.append(compute_mrr(scores, y))
        bal_accs.append(compute_balanced_accuracy(preds, y))

    return {
        "auroc": aurocs,
        "hit_at_1": hit1s,
        "mrr": mrrs,
        "balanced_accuracy": bal_accs,
    }


# ---------------------------------------------------------------------------
# Probe transfer experiment
# ---------------------------------------------------------------------------

def run_probe_transfer(
    train_groups: list[dict[str, Any]],
    test_within_groups: list[dict[str, Any]],
    test_transfer_groups: list[dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    """Train probe on one variant, evaluate within-variant and cross-variant.

    Returns dict with:
        - within_metrics: per-snapshot metrics on same variant test set
        - transfer_metrics: per-snapshot metrics on other variant test set
        - transfer_gap: within_mean - transfer_mean for each metric
    """
    # Flatten training data
    X_train = np.concatenate([g["X"] for g in train_groups])
    y_train = np.concatenate([g["y"] for g in train_groups])

    probe = train_probe(X_train, y_train, seed=seed)

    within = evaluate_probe_per_snapshot(probe, test_within_groups)
    transfer = evaluate_probe_per_snapshot(probe, test_transfer_groups)

    gaps: dict[str, float] = {}
    for metric in within:
        w_mean = np.mean(within[metric]) if within[metric] else 0.0
        t_mean = np.mean(transfer[metric]) if transfer[metric] else 0.0
        gaps[metric] = float(w_mean - t_mean)

    return {
        "within_metrics": within,
        "transfer_metrics": transfer,
        "transfer_gap": gaps,
        "probe": probe,
    }


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    X, Y: (n_samples, hidden_dim) — must have same n_samples.
    Returns CKA similarity in [0, 1].
    """
    # Center both matrices
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC with linear kernel
    XtX = X @ X.T
    YtY = Y @ Y.T

    hsic_xy = np.trace(XtX @ YtY)
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0

    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# Orthogonal Procrustes
# ---------------------------------------------------------------------------

def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Find orthogonal matrix R that minimizes ||X @ R - Y||_F.

    X, Y: (n_samples, hidden_dim).
    Returns R: (hidden_dim, hidden_dim).
    """
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def procrustes_align(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Align X to Y's space using Procrustes, then apply to X_test.

    X_source, Y_target: (n_train, dim) — used to fit R.
    X_test: (n_test, dim) — transformed.
    Returns: (n_test, dim).
    """
    R = orthogonal_procrustes(X_source, Y_target)
    return X_test @ R


# ---------------------------------------------------------------------------
# Router divergence
# ---------------------------------------------------------------------------

def router_jaccard_per_layer(
    indices_a: np.ndarray,  # (num_layers, n_tokens, top_k)
    indices_b: np.ndarray,  # (num_layers, n_tokens, top_k)
) -> np.ndarray:
    """Compute mean Jaccard similarity per layer between two router index arrays.

    Returns: (num_layers,) array of Jaccard values in [0, 1].
    """
    num_layers = indices_a.shape[0]
    n_tokens = min(indices_a.shape[1], indices_b.shape[1])
    jaccards = np.zeros(num_layers)

    for l in range(num_layers):
        layer_jac = 0.0
        for t in range(n_tokens):
            set_a = set(indices_a[l, t].tolist())
            set_b = set(indices_b[l, t].tolist())
            union = set_a | set_b
            if union:
                layer_jac += len(set_a & set_b) / len(union)
        jaccards[l] = layer_jac / max(n_tokens, 1)

    return jaccards


def router_jsd_per_layer(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    num_experts: int = 128,
) -> np.ndarray:
    """Compute Jensen-Shannon divergence of expert usage distributions per layer.

    Aggregates expert selection frequencies over tokens, then computes JSD.
    Returns: (num_layers,) array of JSD values in [0, log(2)].
    """
    from scipy.spatial.distance import jensenshannon

    num_layers = indices_a.shape[0]
    jsds = np.zeros(num_layers)

    for l in range(num_layers):
        # Count expert selections
        counts_a = np.bincount(indices_a[l].ravel().astype(int), minlength=num_experts).astype(float)
        counts_b = np.bincount(indices_b[l].ravel().astype(int), minlength=num_experts).astype(float)

        # Normalize to distributions
        pa = counts_a / (counts_a.sum() + 1e-12)
        pb = counts_b / (counts_b.sum() + 1e-12)

        jsds[l] = float(jensenshannon(pa, pb) ** 2)  # scipy returns sqrt(JSD)

    return jsds


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(
    per_snapshot_values: list[float],
    snapshot_vault_days: list[str],  # vault_day identifier per snapshot
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI over per-snapshot metrics, resampling by vault_day.

    Returns (mean, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)

    # Group by vault_day
    vd_to_values: dict[str, list[float]] = defaultdict(list)
    for val, vd in zip(per_snapshot_values, snapshot_vault_days):
        vd_to_values[vd].append(val)

    vault_days = list(vd_to_values.keys())
    n_vd = len(vault_days)
    if n_vd == 0:
        return 0.0, 0.0, 0.0

    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        # Resample vault_days with replacement
        sampled_vds = rng.choice(vault_days, size=n_vd, replace=True)
        boot_values: list[float] = []
        for vd in sampled_vds:
            boot_values.extend(vd_to_values[vd])
        if boot_values:
            boot_means.append(float(np.mean(boot_values)))

    boot_means_arr = np.array(boot_means)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(per_snapshot_values)),
        float(np.percentile(boot_means_arr, 100 * alpha)),
        float(np.percentile(boot_means_arr, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# Symbol-only baseline
# ---------------------------------------------------------------------------

def train_symbol_baseline(
    symbols: list[list[str]],  # per-snapshot list of symbols
    labels: list[np.ndarray],  # per-snapshot binary labels
    seed: int = 42,
) -> Any:
    """Train a probe using only symbol identity (one-hot) as features."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Collect all symbols
    all_symbols = [s for snap_syms in symbols for s in snap_syms]
    all_labels = np.concatenate(labels)

    le = LabelEncoder()
    le.fit(all_symbols)

    # One-hot encode
    encoded = le.transform(all_symbols)
    n_classes = len(le.classes_)
    X = np.zeros((len(all_symbols), n_classes))
    X[np.arange(len(all_symbols)), encoded] = 1.0

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )),
    ])
    pipe.fit(X, all_labels)
    return pipe, le


# ---------------------------------------------------------------------------
# Row-index baseline
# ---------------------------------------------------------------------------

def train_row_index_baseline(
    n_rows_per_snapshot: list[int],
    labels: list[np.ndarray],
    seed: int = 42,
) -> Any:
    """Train a probe using only row index (0..N-1) as feature."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    all_indices: list[float] = []
    all_labels_flat: list[int] = []
    for n_rows, y in zip(n_rows_per_snapshot, labels):
        for i in range(n_rows):
            all_indices.append(float(i))
        all_labels_flat.extend(y.tolist())

    X = np.array(all_indices).reshape(-1, 1)
    y = np.array(all_labels_flat)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )),
    ])
    pipe.fit(X, y)
    return pipe


# ---------------------------------------------------------------------------
# Experiment A: Full analysis pipeline
# ---------------------------------------------------------------------------

def _qa_layer_worker(
    layer: int,
    label_name: str,
    snapshots: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    cache: dict[str, dict[str, np.ndarray]],
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    """Process one (label, layer) pair for Question A. Thread-safe."""

    def _collect_groups(variant: str, split_ids: set[str]) -> list[dict]:
        groups = []
        for snap in snapshots:
            sid = snap["snapshot_id"]
            if sid not in split_ids:
                continue
            capture_id = f"{sid}_{variant}"
            acts = cache.get(capture_id, {})
            if not acts:
                continue

            n_rows = snap["n_rows"]
            labels_arr = np.array(snap["labels"].get(label_name, [0] * n_rows))

            row_acts = []
            for i in range(n_rows):
                key = f"row_mean_{i}"
                if key in acts:
                    row_acts.append(acts[key][layer].astype(np.float32))
                else:
                    break

            if len(row_acts) != n_rows:
                continue

            X = np.stack(row_acts)
            groups.append({
                "X": X,
                "y": labels_arr,
                "snapshot_id": sid,
                "vault_day": f"{snap['vault_address'][:10]}_{snap['snap_date']}",
            })
        return groups

    low_train = _collect_groups("low_pad", train_ids)
    low_test = _collect_groups("low_pad", test_ids)
    high_test = _collect_groups("high_pad", test_ids)

    if not low_train or not low_test or not high_test:
        return {"layer": layer, "error": "insufficient data"}

    transfer = run_probe_transfer(low_train, low_test, high_test, seed=seed)

    X_low = np.concatenate([g["X"] for g in low_test])
    X_high = np.concatenate([g["X"] for g in high_test])
    n_matched = min(len(X_low), len(X_high))
    cka = linear_cka(X_low[:n_matched], X_high[:n_matched])

    vault_days = [g["vault_day"] for g in low_test]
    within_aurocs = transfer["within_metrics"]["auroc"]
    transfer_aurocs = transfer["transfer_metrics"]["auroc"]

    layer_result: dict[str, Any] = {
        "layer": layer,
        "cka": cka,
        "transfer_gap": transfer["transfer_gap"],
        "within_auroc_mean": float(np.mean(within_aurocs)) if within_aurocs else None,
        "transfer_auroc_mean": float(np.mean(transfer_aurocs)) if transfer_aurocs else None,
        "within_hit1_mean": float(np.mean(transfer["within_metrics"]["hit_at_1"])),
        "transfer_hit1_mean": float(np.mean(transfer["transfer_metrics"]["hit_at_1"])),
        "within_mrr_mean": float(np.mean(transfer["within_metrics"]["mrr"])),
        "transfer_mrr_mean": float(np.mean(transfer["transfer_metrics"]["mrr"])),
    }

    if within_aurocs and transfer_aurocs and vault_days:
        gap_values = [w - t for w, t in zip(within_aurocs, transfer_aurocs)]
        mean, lo, hi = bootstrap_ci(
            gap_values, vault_days,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        layer_result["auroc_gap_ci"] = {"mean": mean, "lo": lo, "hi": hi}

    return layer_result


def run_experiment_a(
    config: CounterfactualAnalysisConfig,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Run the full Experiment A analysis.

    Loads all activations upfront, then runs probe transfer + CKA + bootstrap
    across layers in parallel using a thread pool.
    """
    spec = load_dataset_a_spec()
    snapshots = spec["snapshots"]
    train_ids = spec["train_ids"]
    test_ids = spec["test_ids"]

    # Determine layers
    sample_id = f"{snapshots[0]['snapshot_id']}_low_pad"
    sample_acts = load_pooled_activations(config.run_dir, sample_id)
    if "market_mean" not in sample_acts:
        print("ERROR: No market_mean found in sample activations")
        return {"error": "missing activations"}

    num_layers = sample_acts["market_mean"].shape[0]
    layers = config.layers or list(range(num_layers))
    print(f"Analyzing {len(layers)} layers, {len(snapshots)} snapshots")

    # Preload all activations into memory (concurrent I/O)
    all_capture_ids = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        for variant in ("low_pad", "high_pad", "low_raw", "high_raw"):
            all_capture_ids.append(f"{sid}_{variant}")
    cache = preload_all_activations(config.run_dir, all_capture_ids)

    label_names = list(snapshots[0]["labels"].keys())
    results: dict[str, Any] = {"layers": layers, "label_results": {}}

    for label_name in label_names:
        print(f"\n--- Label: {label_name} ---")
        t0 = time.monotonic()

        # Parallel across layers
        layer_results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _qa_layer_worker, layer, label_name, snapshots,
                    train_ids, test_ids, cache, config.seed, config.n_bootstrap,
                ): layer
                for layer in layers
            }
            for fut in as_completed(futures):
                lr = fut.result()
                layer_results_map[lr["layer"]] = lr

        # Sort by layer order
        per_layer = [layer_results_map[l] for l in layers if l in layer_results_map]
        elapsed = time.monotonic() - t0
        n_ok = sum(1 for r in per_layer if "error" not in r)
        print(f"  {n_ok}/{len(layers)} layers OK in {elapsed:.1f}s")

        results["label_results"][label_name] = {"per_layer": per_layer}

    return results


# ---------------------------------------------------------------------------
# Question B: Post-market reinterpretation
# ---------------------------------------------------------------------------

def _collect_downstream_groups(
    snapshots: list[dict[str, Any]],
    split_ids: set[str],
    variant: str,
    label_name: str,
    layer: int,
    position_key: str,
    cache: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Collect per-snapshot groups for a downstream position key.

    position_key: e.g. "settings_eos", "portfolio_eos", "last_token"
    """
    groups: list[dict[str, Any]] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        if sid not in split_ids:
            continue
        capture_id = f"{sid}_{variant}"
        acts = cache.get(capture_id, {})
        if not acts or position_key not in acts:
            continue

        n_rows = snap["n_rows"]
        labels_arr = np.array(snap["labels"].get(label_name, [0] * n_rows))

        pos_act = acts[position_key][layer].astype(np.float32)

        row_acts = []
        for i in range(n_rows):
            key = f"row_mean_{i}"
            if key in acts:
                row_acts.append(acts[key][layer].astype(np.float32))
            else:
                break

        if len(row_acts) == n_rows:
            X = np.stack([
                np.concatenate([row_acts[i], pos_act])
                for i in range(n_rows)
            ])
        else:
            X = np.tile(pos_act, (n_rows, 1))

        groups.append({
            "X": X,
            "y": labels_arr,
            "snapshot_id": sid,
            "vault_day": f"{snap.get('vault_address', sid)[:10]}_{snap.get('snap_date', '')}",
        })
    return groups


def _qb_probe_layer_worker(
    layer: int,
    pos_key: str,
    label_name: str,
    snapshots: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    cache: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    """Process one (label, position, layer) for Q-B probe transfer. Thread-safe."""
    train_groups = _collect_downstream_groups(
        snapshots, train_ids, "settings_all1", label_name, layer, pos_key, cache,
    )
    test_within = _collect_downstream_groups(
        snapshots, test_ids, "settings_all1", label_name, layer, pos_key, cache,
    )
    test_transfer = _collect_downstream_groups(
        snapshots, test_ids, "settings_all5", label_name, layer, pos_key, cache,
    )

    if not train_groups or not test_within or not test_transfer:
        return {"layer": layer, "error": "insufficient_data"}

    transfer = run_probe_transfer(train_groups, test_within, test_transfer, seed=seed)

    X_s1 = np.concatenate([g["X"] for g in test_within])
    X_s5 = np.concatenate([g["X"] for g in test_transfer])
    n_matched = min(len(X_s1), len(X_s5))
    cka = linear_cka(X_s1[:n_matched], X_s5[:n_matched])

    return {
        "layer": layer,
        "cka": cka,
        "transfer_gap": transfer["transfer_gap"],
        "within_auroc": float(np.mean(transfer["within_metrics"]["auroc"])) if transfer["within_metrics"]["auroc"] else None,
        "transfer_auroc": float(np.mean(transfer["transfer_metrics"]["auroc"])) if transfer["transfer_metrics"]["auroc"] else None,
    }


def _qb_delta_layer_worker(
    layer: int,
    pos_key: str,
    snapshots: list[dict[str, Any]],
    cache: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Compute delta consistency for one (position, layer). Thread-safe."""
    deltas: list[np.ndarray] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        acts_s1 = cache.get(f"{sid}_settings_all1", {})
        acts_s5 = cache.get(f"{sid}_settings_all5", {})

        if not acts_s1 or not acts_s5:
            continue
        if pos_key not in acts_s1 or pos_key not in acts_s5:
            continue

        h_s1 = acts_s1[pos_key][layer].astype(np.float32)
        h_s5 = acts_s5[pos_key][layer].astype(np.float32)
        deltas.append(h_s5 - h_s1)

    if len(deltas) < 3:
        return {"layer": layer, "error": "insufficient_data"}

    delta_mat = np.stack(deltas)
    norms = np.linalg.norm(delta_mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = delta_mat / norms
    cos_sim = normed @ normed.T
    n = cos_sim.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_cos = float(cos_sim[upper_mask].mean())

    return {
        "layer": layer,
        "mean_pairwise_cosine": mean_cos,
        "n_prompts": len(deltas),
        "mean_delta_norm": float(np.mean(np.linalg.norm(delta_mat, axis=1))),
    }


def run_question_b(
    config: CounterfactualAnalysisConfig,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Question B: Does the model reinterpret market state at downstream positions?

    Tests whether market-feature labels are decodable at downstream positions
    (settings_eos, portfolio_eos, constraints_eos, prev_decisions_eos, last_token),
    and whether that decoding transfers across settings variants.

    Also computes delta consistency: Δ = h^{all5} - h^{all1} at downstream
    positions. High consistency → additive policy. Low → interaction.
    """
    spec = load_dataset_b_spec()
    snapshots = spec["snapshots"]
    train_ids = spec["train_ids"]
    test_ids = spec["test_ids"]

    if not snapshots:
        print("No Dataset B snapshots found in DB")
        return {"error": "dataset_b_not_found"}

    # Determine layers
    sample_snap = snapshots[0]
    sample_id = f"{sample_snap['snapshot_id']}_settings_all1"
    sample_acts = load_pooled_activations(config.run_dir, sample_id)
    if not sample_acts:
        sample_id = f"{sample_snap['snapshot_id']}_original"
        sample_acts = load_pooled_activations(config.run_dir, sample_id)
    if not sample_acts:
        return {"error": "no_activations_found"}

    for k in ["last_token", "portfolio_eos", "settings_eos", "market_mean"]:
        if k in sample_acts:
            num_layers = sample_acts[k].shape[0]
            break
    else:
        return {"error": "no_position_keys_found"}

    layers = config.layers or list(range(num_layers))
    label_names = list(snapshots[0].get("labels", {}).keys())

    downstream_positions = [
        "settings_eos", "portfolio_eos", "constraints_eos",
        "prev_decisions_eos", "last_token",
    ]

    # Preload all Dataset B activations
    all_capture_ids = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        for variant in ("settings_all1", "settings_all5", "original"):
            all_capture_ids.append(f"{sid}_{variant}")
    cache = preload_all_activations(config.run_dir, all_capture_ids)

    results: dict[str, Any] = {
        "layers": layers,
        "downstream_positions": downstream_positions,
        "probe_transfer": {},
        "delta_consistency": {},
    }

    # --- Probe transfer at downstream positions (parallel across layers) ---
    for label_name in label_names:
        print(f"\n--- Q-B probe transfer: {label_name} ---")
        t0 = time.monotonic()
        label_results: dict[str, list[dict]] = {}

        for pos_key in downstream_positions:
            layer_map: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _qb_probe_layer_worker, layer, pos_key, label_name,
                        snapshots, train_ids, test_ids, cache, config.seed,
                    ): layer
                    for layer in layers
                }
                for fut in as_completed(futures):
                    lr = fut.result()
                    layer_map[lr["layer"]] = lr

            label_results[pos_key] = [layer_map[l] for l in layers if l in layer_map]

        results["probe_transfer"][label_name] = label_results
        print(f"  {len(downstream_positions)} positions × {len(layers)} layers in {time.monotonic() - t0:.1f}s")

    # --- Delta consistency (parallel across layers × positions) ---
    print("\n--- Q-B delta consistency ---")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Any, tuple[str, int]] = {}
        for pos_key in downstream_positions:
            for layer in layers:
                fut = pool.submit(
                    _qb_delta_layer_worker, layer, pos_key, snapshots, cache,
                )
                futures[fut] = (pos_key, layer)

        # Collect results
        delta_results: dict[str, dict[int, dict]] = {p: {} for p in downstream_positions}
        for fut in as_completed(futures):
            pos_key, layer = futures[fut]
            delta_results[pos_key][layer] = fut.result()

    for pos_key in downstream_positions:
        results["delta_consistency"][pos_key] = [
            delta_results[pos_key][l] for l in layers if l in delta_results[pos_key]
        ]

    print(f"  {len(downstream_positions)} positions × {len(layers)} layers in {time.monotonic() - t0:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Question C: Final decision-layer interaction
# ---------------------------------------------------------------------------

def _qc_layer_worker(
    layer: int,
    pos_key: str,
    snapshots: list[dict[str, Any]],
    cache: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Process one (position, layer) for Q-C. Thread-safe."""
    deltas: list[np.ndarray] = []
    all_s1: list[np.ndarray] = []
    all_s5: list[np.ndarray] = []

    for snap in snapshots:
        sid = snap["snapshot_id"]
        acts_s1 = cache.get(f"{sid}_settings_all1", {})
        acts_s5 = cache.get(f"{sid}_settings_all5", {})

        if not acts_s1 or not acts_s5:
            continue
        if pos_key not in acts_s1 or pos_key not in acts_s5:
            continue

        h_s1 = acts_s1[pos_key][layer].astype(np.float32)
        h_s5 = acts_s5[pos_key][layer].astype(np.float32)
        deltas.append(h_s5 - h_s1)
        all_s1.append(h_s1)
        all_s5.append(h_s5)

    if len(deltas) < 3:
        return {"layer": layer, "error": "insufficient_data"}

    delta_mat = np.stack(deltas)
    norms = np.linalg.norm(delta_mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = delta_mat / norms
    cos_sim = normed @ normed.T
    n = cos_sim.shape[0]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_cos = float(cos_sim[upper_mask].mean())

    cka = linear_cka(np.stack(all_s1), np.stack(all_s5)) if len(all_s1) >= 3 else None

    return {
        "layer": layer,
        "mean_pairwise_cosine": mean_cos,
        "cka": cka,
        "n_prompts": len(deltas),
        "mean_delta_norm": float(np.mean(np.linalg.norm(delta_mat, axis=1))),
    }


def run_question_c(
    config: CounterfactualAnalysisConfig,
    cache: dict[str, dict[str, np.ndarray]] | None = None,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Question C: Is config mostly additive at the decision point?

    Focuses on last_token and final downstream sections.
    High delta consistency → additive policy layer.
    Low delta consistency → policy-content interaction.

    Accepts an optional pre-loaded cache from Q-B to avoid re-reading.
    """
    spec = load_dataset_b_spec()
    snapshots = spec["snapshots"]

    if not snapshots:
        return {"error": "dataset_b_not_found"}

    # Preload if not provided
    if cache is None:
        all_capture_ids = []
        for snap in snapshots:
            sid = snap["snapshot_id"]
            for variant in ("settings_all1", "settings_all5"):
                all_capture_ids.append(f"{sid}_{variant}")
        cache = preload_all_activations(config.run_dir, all_capture_ids)

    # Determine layers from cache
    sample_id = f"{snapshots[0]['snapshot_id']}_settings_all1"
    sample_acts = cache.get(sample_id, {})
    if not sample_acts or "last_token" not in sample_acts:
        return {"error": "no_activations_found"}

    num_layers = sample_acts["last_token"].shape[0]
    layers = config.layers or list(range(num_layers))

    focus_positions = ["last_token", "prev_decisions_eos", "constraints_eos"]

    results: dict[str, Any] = {"layers": layers, "positions": {}}

    print("--- Q-C delta consistency + CKA ---")
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Any, tuple[str, int]] = {}
        for pos_key in focus_positions:
            for layer in layers:
                fut = pool.submit(
                    _qc_layer_worker, layer, pos_key, snapshots, cache,
                )
                futures[fut] = (pos_key, layer)

        pos_results: dict[str, dict[int, dict]] = {p: {} for p in focus_positions}
        for fut in as_completed(futures):
            pos_key, layer = futures[fut]
            pos_results[pos_key][layer] = fut.result()

    for pos_key in focus_positions:
        results["positions"][pos_key] = [
            pos_results[pos_key][l] for l in layers if l in pos_results[pos_key]
        ]

    print(f"  {len(focus_positions)} positions × {len(layers)} layers in {time.monotonic() - t0:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def apply_decision_rules(
    question_a_results: dict[str, Any],
    question_b_results: dict[str, Any] | None = None,
    question_c_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the preregistered decision rules to experiment results.

    Returns dict with:
        - decision: one of 'objective_market_first', 'early_entanglement',
          'late_reinterpretation', 'mixed'
        - reasoning: human-readable explanation
        - metrics: summary statistics used for the decision
    """
    # --- Question A: Market-row invariance ---
    all_ckas: list[float] = []
    all_gaps: list[float] = []
    early_mid_ckas: list[float] = []

    for label_name, lr in question_a_results.get("label_results", {}).items():
        for layer_result in lr.get("per_layer", []):
            if "error" in layer_result:
                continue
            cka = layer_result.get("cka")
            gap = layer_result.get("transfer_gap", {}).get("auroc", 0)
            if cka is not None:
                all_ckas.append(cka)
                if layer_result["layer"] <= 23:
                    early_mid_ckas.append(cka)
            all_gaps.append(gap)

    if not all_ckas:
        return {"decision": "insufficient_data", "reasoning": "No CKA values computed"}

    mean_cka = float(np.mean(all_ckas))
    mean_gap = float(np.mean(all_gaps))
    early_mean_cka = float(np.mean(early_mid_ckas)) if early_mid_ckas else mean_cka

    # Question A classification
    a_market_invariant = early_mean_cka >= 0.90 and abs(mean_gap) < 0.05
    a_early_entangled = early_mean_cka < 0.85

    # --- Question B: Downstream reinterpretation ---
    b_reinterprets = False
    b_delta_additive = True
    b_metrics: dict[str, Any] = {}

    if question_b_results and "delta_consistency" in question_b_results:
        # Check delta consistency at last_token
        lt_deltas = question_b_results["delta_consistency"].get("last_token", [])
        lt_cos_values = [
            d["mean_pairwise_cosine"] for d in lt_deltas
            if "mean_pairwise_cosine" in d
        ]
        if lt_cos_values:
            mean_lt_cos = float(np.mean(lt_cos_values))
            b_metrics["last_token_mean_cos"] = mean_lt_cos
            if mean_lt_cos < 0.5:
                b_delta_additive = False

        # Check probe transfer gaps at downstream positions
        transfer_results = question_b_results.get("probe_transfer", {})
        for label_name, pos_results in transfer_results.items():
            for pos_key, layer_list in pos_results.items():
                gaps = [
                    d.get("transfer_gap", {}).get("auroc", 0)
                    for d in layer_list if "error" not in d
                ]
                if gaps and abs(float(np.mean(gaps))) > 0.1:
                    b_reinterprets = True
                    break

    # --- Question C: Additive policy ---
    c_additive = True
    c_metrics: dict[str, Any] = {}

    if question_c_results and "positions" in question_c_results:
        lt_results = question_c_results["positions"].get("last_token", [])
        cos_values = [
            d["mean_pairwise_cosine"] for d in lt_results
            if "mean_pairwise_cosine" in d
        ]
        if cos_values:
            mean_cos = float(np.mean(cos_values))
            c_metrics["last_token_mean_cos"] = mean_cos
            if mean_cos < 0.5:
                c_additive = False

    # --- Apply decision rules ---
    metrics = {
        "early_mean_cka": early_mean_cka,
        "mean_cka": mean_cka,
        "mean_gap": mean_gap,
        "a_market_invariant": a_market_invariant,
        "a_early_entangled": a_early_entangled,
        "b_reinterprets": b_reinterprets,
        "b_delta_additive": b_delta_additive,
        "c_additive": c_additive,
        **{f"b_{k}": v for k, v in b_metrics.items()},
        **{f"c_{k}": v for k, v in c_metrics.items()},
    }

    if a_early_entangled:
        decision = "early_entanglement"
        reasoning = (
            f"Market-row CKA in early/mid layers ({early_mean_cka:.3f}) < 0.85. "
            "Config warps market perception from early layers."
        )
    elif a_market_invariant and not b_reinterprets and c_additive:
        decision = "objective_market_first"
        reasoning = (
            f"Market-row invariant (CKA={early_mean_cka:.3f}, gap={mean_gap:.3f}). "
            "No downstream reinterpretation. Config is a late additive policy layer."
        )
    elif a_market_invariant and (b_reinterprets or not b_delta_additive):
        decision = "late_reinterpretation"
        reasoning = (
            f"Market-row invariant (CKA={early_mean_cka:.3f}), but downstream "
            "positions show config-dependent market decoding. The model reinterprets "
            "market state after seeing settings."
        )
    else:
        decision = "mixed"
        reasoning = (
            f"Metrics in gray zone (CKA={early_mean_cka:.3f}, gap={mean_gap:.3f}). "
            "Triggers v2 follow-up."
        )

    return {"decision": decision, "reasoning": reasoning, "metrics": metrics}


# ---------------------------------------------------------------------------
# Label reconnaissance (standalone)
# ---------------------------------------------------------------------------

def run_label_reconnaissance(
    conn: Any,
    n_snapshots: int = 120,
    seed: int = 42,
) -> dict[str, Any]:
    """Run label reconnaissance to evaluate candidate labels.

    Returns retention criteria evaluation for each label.
    """
    from collections import Counter
    from projects.DX_TERMINAL.counterfactual import (
        sample_snapshots,
        build_snapshots,
    )

    raw = sample_snapshots(conn, n=n_snapshots, seed=seed)
    snapshots = build_snapshots(raw)

    results: dict[str, Any] = {}

    for label_name in snapshots[0].labels.keys():
        total_pos = 0
        winner_symbols: Counter = Counter()
        valid_snapshots = 0
        total_rows = 0

        for snap in snapshots:
            labels = snap.labels.get(label_name, [])
            if not labels:
                continue
            valid_snapshots += 1
            total_rows += len(labels)
            for i, val in enumerate(labels):
                if val == 1:
                    total_pos += 1
                    winner_symbols[snap.rows[i].symbol] += 1

        n_distinct = len(winner_symbols)
        top_share = max(winner_symbols.values()) / total_pos if total_pos > 0 else 1.0

        passes = (
            total_pos >= 80
            and top_share < 0.45
            and n_distinct >= 5
        )

        results[label_name] = {
            "total_positives": total_pos,
            "total_rows": total_rows,
            "valid_snapshots": valid_snapshots,
            "n_distinct_winners": n_distinct,
            "top_winner_share": round(top_share, 3),
            "top_3_winners": winner_symbols.most_common(3),
            "passes_retention": passes,
        }

    return results

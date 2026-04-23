from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal


APP_NAME = "xenon-prompt-confusion-phase4-arbitration"
DEFAULT_RELATION = "workflow_dataset_conflict_probe_v3_conflict_readout_side_v1"
DEFAULT_BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"
DEFAULT_CAPTURE_RUN_ID = "7e71cf742002"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
DEFAULT_OUTPUT_SUBDIR = "prompt_confusion/phase_04/conflict_arbitration_stage1"
DEFAULT_STAGE2_OUTPUT_SUBDIR = "prompt_confusion/phase_04/conflict_arbitration_stage2"
SECTION_ORDER = ("task", "strategy", "settings", "portfolio", "market")
SECTION_HEADERS = {
    "task": "TASK\n",
    "strategy": "STRATEGY\n",
    "settings": "SETTINGS\n",
    "portfolio": "PORTFOLIO\n",
    "market": "MARKET\n",
}
READOUT_FIELD_BY_FAMILY = {
    "trade_size_force_large": "size",
    "trade_size_force_small": "size",
    "activity_force_trade": "action",
    "activity_force_observe": "action",
}

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
neon_secret = modal.Secret.from_name("xenon-neon")

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "matplotlib",
        "numpy",
        "pyarrow",
        "psycopg[binary]",
        "safetensors",
        "scikit-learn",
        "torch",
        "transformers",
    )
    .add_local_python_source("pipelines")
    .add_local_python_source("projects")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "matplotlib",
        "numpy",
        "pyarrow",
        "psycopg[binary]",
        "safetensors",
        "scikit-learn",
        "torch",
        "transformers",
        "vllm",
        "huggingface_hub",
    )
    .env({"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_python_source("pipelines")
    .add_local_python_source("projects")
)

attention_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "accelerate",
        "matplotlib",
        "numpy",
        "pyarrow",
        "psycopg[binary]",
        "safetensors",
        "scikit-learn",
        "torch",
        "transformers",
        "huggingface_hub",
    )
    .add_local_python_source("pipelines")
    .add_local_python_source("projects")
)


def _parse_messages(raw: Any) -> list[dict[str, str]]:
    if not raw:
        return []
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str):
            out.append({"role": item["role"], "content": item["content"]})
    return out


def _extract_system_user(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    system_text = ""
    user_text = ""
    for msg in messages:
        if msg["role"] == "system" and not system_text:
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
    if not user_text:
        return None
    return system_text, user_text


def _render_chat_text(tokenizer: Any, system_text: str, user_text: str) -> str:
    rendered = tokenizer.apply_chat_template(
        ([{"role": "system", "content": system_text}] if system_text else [])
        + [{"role": "user", "content": user_text}],
        add_generation_prompt=False,
        tokenize=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer did not return rendered chat text")
    return rendered


def _token_offsets_for_rendered(tokenizer: Any, rendered_text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    offset_mapping = getattr(encoded, "offset_mapping", None)
    if offset_mapping is None and isinstance(encoded, dict):
        offset_mapping = encoded.get("offset_mapping")
    if input_ids is None or offset_mapping is None:
        raise ValueError("Tokenizer did not return input_ids and offset_mapping")
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(offset_mapping, "tolist"):
        offset_mapping = offset_mapping.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offset_mapping and isinstance(offset_mapping[0], list) and offset_mapping[0] and isinstance(offset_mapping[0][0], list):
        offset_mapping = offset_mapping[0]
    return [int(tok) for tok in input_ids], [(int(s), int(e)) for s, e in offset_mapping]


def _char_to_token_span(
    offsets: list[tuple[int, int]],
    *,
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    token_start: int | None = None
    token_end: int | None = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if token_start is None and tok_end > start_char:
            token_start = idx
        if tok_start < end_char:
            token_end = idx + 1
        elif token_start is not None:
            break
    if token_start is None or token_end is None or token_start >= token_end:
        return None
    return token_start, token_end


def _trim_section_end_char(rendered_text: str, *, section_start_char: int, section_end_char: int) -> int:
    import re

    if section_end_char <= section_start_char:
        return section_end_char
    section_text = rendered_text[section_start_char:section_end_char]
    section_text = re.sub(r"\s+\Z", "", section_text)
    section_text = re.sub(r"(?:\n-+[ \t]*)+\Z", "", section_text)
    section_text = re.sub(r"\s+\Z", "", section_text)
    trimmed_end = section_start_char + len(section_text)
    return trimmed_end if trimmed_end > section_start_char else section_end_char


def _find_section_token_spans(
    *,
    tokenizer: Any,
    system_text: str,
    user_text: str,
) -> tuple[dict[str, tuple[int, int]], int]:
    rendered = _render_chat_text(tokenizer, system_text, user_text)
    input_ids, offsets = _token_offsets_for_rendered(tokenizer, rendered)
    user_char = rendered.find(user_text)
    if user_char < 0:
        raise ValueError("Could not locate user text inside rendered prompt")

    starts: list[tuple[str, int]] = []
    search_from = user_char
    user_end_char = user_char + len(user_text)
    for section_name in SECTION_ORDER:
        header = SECTION_HEADERS[section_name]
        idx = rendered.find(header, search_from, user_end_char)
        if idx < 0:
            continue
        starts.append((section_name, idx))
        search_from = idx + len(header)

    boundaries: dict[str, tuple[int, int]] = {}
    for idx, (section_name, start_char) in enumerate(starts):
        raw_end_char = starts[idx + 1][1] if idx + 1 < len(starts) else user_end_char
        end_char = _trim_section_end_char(
            rendered,
            section_start_char=start_char,
            section_end_char=raw_end_char,
        )
        span = _char_to_token_span(offsets, start_char=start_char, end_char=end_char)
        if span is not None:
            boundaries[section_name] = span
    return boundaries, len(input_ids)


def _load_section_rows(relation_name: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        log_id,
        workflow_label,
        arbitration_group_id,
        strategy_family,
        prompt_messages_json
    FROM {relation_name}
    ORDER BY log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_confound_rows(relation_name: str, base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        v.log_id,
        v.workflow_label,
        v.arbitration_group_id,
        v.strategy_family,
        v.strategy_variant_id,
        v.setting_lexical_family_id,
        v.environment_pressure_bucket,
        v.context_variant_id,
        d.prompt_messages_json
    FROM {relation_name} v
    JOIN {base_relation} d
      USING (log_id)
    ORDER BY v.log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_causal_rows(relation_name: str, base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        v.log_id,
        v.workflow_label,
        v.arbitration_group_id,
        d.example_id,
        d.strategy_family,
        d.environment_pressure_bucket,
        d.prompt_messages_json,
        d.strategy_expected_action,
        d.strategy_expected_asset,
        d.strategy_expected_size,
        d.setting_expected_action,
        d.setting_expected_asset,
        d.setting_expected_size
    FROM {relation_name} v
    JOIN {base_relation} d
      USING (log_id)
    ORDER BY v.log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_metadata_map(activations_dir: Path) -> tuple[dict[int, dict[str, Any]], list[int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(activations_dir / "metadata.parquet")
    rows = table.to_pylist()
    meta_by_log_id = {int(row["log_id"]): dict(row) for row in rows}
    captured_layers = rows[0].get("captured_layers") if rows else None
    if isinstance(captured_layers, str):
        try:
            captured_layers = json.loads(captured_layers)
        except json.JSONDecodeError:
            captured_layers = []
    parsed_layers = [int(layer) for layer in captured_layers] if isinstance(captured_layers, list) else []
    return meta_by_log_id, parsed_layers


def _load_tensor(path: Path, key: str) -> Any:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        return f.get_tensor(key)


def _pool_section_tensor(tensor: Any, span: tuple[int, int]) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(tensor)
    start, end = int(span[0]), int(span[1])
    if arr.ndim != 3 or start < 0 or end <= start or end > arr.shape[1]:
        raise ValueError(f"Invalid section span {span} for tensor shape {arr.shape}")
    section_slice = arr[:, start:end, :]
    return {
        "mean": section_slice.mean(axis=1).astype(np.float32),
        "eos": arr[:, end - 1, :].astype(np.float32),
    }


def _balanced_probe(
    X: Any,
    y: Any,
    *,
    groups: Any | None,
    n_folds: int,
    seed: int,
) -> dict[str, Any] | None:
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2 or len(X) < 4 or len(np.unique(y)) < 2:
        return None

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = len(set(groups.tolist()))
        if unique_groups < 2:
            return None
        actual_folds = max(2, min(int(n_folds), unique_groups))
        try:
            splitter = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        except Exception:
            splitter = GroupKFold(n_splits=actual_folds)
        splits = splitter.split(X, y, groups=groups)
        split_mode = "grouped"
    else:
        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min()) if len(class_counts) else 0
        actual_folds = max(2, min(int(n_folds), len(y), min_class_count))
        splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        splits = splitter.split(X, y)
        split_mode = "stratified"

    scores: list[float] = []
    for train_idx, test_idx in splits:
        pipe = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-3,
                random_state=seed,
            ),
        )
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
    if not scores:
        return None
    return {
        "balanced_accuracy": round(float(sum(scores) / len(scores)), 4),
        "balanced_accuracy_std": round(float(np.std(scores)), 4),
        "n_examples": int(len(y)),
        "n_folds": int(len(scores)),
        "split_mode": split_mode,
    }


def _plot_section_heatmap(rows: list[dict[str, Any]], *, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return
    layer_ids = sorted({int(row["layer"]) for row in rows})
    section_keys = []
    for section_name in SECTION_ORDER:
        for pooling in ("mean", "eos"):
            key = f"{section_name}:{pooling}"
            if any(row["section_name"] == section_name and row["pooling"] == pooling for row in rows):
                section_keys.append(key)
    if not section_keys:
        return

    matrix = np.full((len(section_keys), len(layer_ids)), np.nan, dtype=np.float32)
    section_to_idx = {name: idx for idx, name in enumerate(section_keys)}
    layer_to_idx = {layer: idx for idx, layer in enumerate(layer_ids)}
    for row in rows:
        key = f"{row['section_name']}:{row['pooling']}"
        matrix[section_to_idx[key], layer_to_idx[int(row["layer"])]] = float(row["balanced_accuracy"])

    fig, ax = plt.subplots(figsize=(1.2 * len(layer_ids), 0.55 * len(section_keys) + 1.8))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.45, vmax=max(0.75, float(np.nanmax(matrix))))
    ax.set_xticks(range(len(layer_ids)))
    ax.set_xticklabels([str(layer) for layer in layer_ids], rotation=0)
    ax.set_yticks(range(len(section_keys)))
    ax.set_yticklabels(section_keys)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Section / Pooling")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(float(value)):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Balanced Accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _classify_readout_side(row: dict[str, Any], generated_text: str) -> dict[str, Any]:
    parsed = _parse_json_object(generated_text)
    result = {
        "valid_output": False,
        "readout_side": "neither",
        "action": None,
        "asset": None,
        "size": None,
    }
    if parsed is None or set(parsed.keys()) != {"action", "asset", "size"}:
        return result

    action = parsed.get("action")
    asset = parsed.get("asset")
    size = parsed.get("size")
    if not all(isinstance(value, str) for value in (action, asset, size)):
        return result

    readout_field = READOUT_FIELD_BY_FAMILY.get(str(row["strategy_family"]))
    readout_side = "neither"
    if readout_field == "action":
        strategy_value = str(row["strategy_expected_action"])
        setting_value = str(row["setting_expected_action"])
        if action == strategy_value and action != setting_value:
            readout_side = "strategy"
        elif action == setting_value and action != strategy_value:
            readout_side = "setting"
    elif readout_field == "size":
        strategy_value = str(row["strategy_expected_size"])
        setting_value = str(row["setting_expected_size"])
        if size == strategy_value and size != setting_value:
            readout_side = "strategy"
        elif size == setting_value and size != strategy_value:
            readout_side = "setting"

    result.update(
        {
            "valid_output": True,
            "readout_side": readout_side,
            "action": action,
            "asset": asset,
            "size": size,
        }
    )
    return result


def _resolve_prompt_eos_index(input_ids: list[int], eos_token_id: int | None) -> int:
    if not input_ids:
        raise ValueError("Cannot resolve EOS index from empty input_ids")
    if eos_token_id is None:
        return len(input_ids) - 1
    for idx in range(len(input_ids) - 1, -1, -1):
        if int(input_ids[idx]) == int(eos_token_id):
            return idx
    return len(input_ids) - 1


def _evaluate_pipeline_cv(
    *,
    estimator_factory: Any,
    X: Any,
    y: Any,
    groups: Any | None,
    n_folds: int,
    seed: int,
) -> dict[str, Any] | None:
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

    y = np.asarray(y, dtype=np.int64)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return None

    if isinstance(X, list):
        X_indexable = list(X)
    else:
        X_indexable = X

    split_mode = "stratified"
    split_kwargs: dict[str, Any] = {}
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = len(set(groups.tolist()))
        if unique_groups < 2:
            return None
        actual_folds = max(2, min(int(n_folds), unique_groups))
        try:
            cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        except Exception:
            cv = GroupKFold(n_splits=actual_folds)
        split_mode = "grouped"
        split_kwargs["groups"] = groups
        n_groups = unique_groups
    else:
        class_counts = np.bincount(y)
        min_class_count = int(class_counts.min()) if len(class_counts) else 0
        actual_folds = max(2, min(int(n_folds), len(y), min_class_count))
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
        n_groups = None

    accs: list[float] = []
    bals: list[float] = []
    for train_idx, test_idx in cv.split(X_indexable, y, **split_kwargs):
        est = estimator_factory()
        if hasattr(X_indexable, "iloc"):
            X_train = X_indexable.iloc[train_idx]
            X_test = X_indexable.iloc[test_idx]
        elif isinstance(X_indexable, list):
            X_train = [X_indexable[int(i)] for i in train_idx]
            X_test = [X_indexable[int(i)] for i in test_idx]
        else:
            X_train = X_indexable[train_idx]
            X_test = X_indexable[test_idx]
        est.fit(X_train, y[train_idx])
        pred = est.predict(X_test)
        accs.append(float(accuracy_score(y[test_idx], pred)))
        bals.append(float(balanced_accuracy_score(y[test_idx], pred)))
    if not accs:
        return None
    return {
        "n_examples": int(len(y)),
        "n_folds": int(len(accs)),
        "n_groups": None if n_groups is None else int(n_groups),
        "split_mode": split_mode,
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "balanced_accuracy": round(float(np.mean(bals)), 4),
        "balanced_accuracy_std": round(float(np.std(bals)), 4),
    }


@app.function(
    volumes={"/data": data_volume, "/models": model_volume},
    image=base_image,
    timeout=3600,
    cpu=6,
    secrets=[hf_secret, neon_secret],
)
def run_section_attribution(
    *,
    relation_name: str = DEFAULT_RELATION,
    capture_run_id: str = DEFAULT_CAPTURE_RUN_ID,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    local_model_path = f"/models/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    activations_dir = Path("/data/activations") / activations_subdir
    output_dir = Path("/data/analysis_results") / output_subdir / "section_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_section_rows(relation_name)
    metadata_map, captured_layers = _load_metadata_map(activations_dir)
    layer_to_idx = {int(layer): idx for idx, layer in enumerate(captured_layers)}

    feature_store: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    feature_labels: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    feature_groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    labels: list[int] = []
    groups: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    router_rows_present = 0

    for row in rows:
        log_id = int(row["log_id"])
        meta = metadata_map.get(log_id)
        if meta is None:
            continue
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        if system_user is None:
            continue
        system_text, user_text = system_user
        section_spans, seq_len = _find_section_token_spans(
            tokenizer=tokenizer,
            system_text=system_text,
            user_text=user_text,
        )
        residual_path = activations_dir / "residual_stream" / f"{meta['artifact_id']}.safetensors"
        if not residual_path.exists():
            continue
        residual = _load_tensor(residual_path, "residual_stream")
        router_path = activations_dir / "router_logits" / f"{meta['artifact_id']}.safetensors"
        router_logits = _load_tensor(router_path, "router_logits") if router_path.exists() else None

        labels.append(1 if str(row["workflow_label"]) == "setting" else 0)
        groups.append(str(row["arbitration_group_id"]))
        metadata_rows.append(
            {
                "log_id": log_id,
                "workflow_label": str(row["workflow_label"]),
                "arbitration_group_id": str(row["arbitration_group_id"]),
                "strategy_family": str(row["strategy_family"]),
                "seq_len": int(seq_len),
                "artifact_id": str(meta["artifact_id"]),
                "section_spans_json": json.dumps({k: [int(v[0]), int(v[1])] for k, v in section_spans.items()}),
            }
        )
        for section_name, span in section_spans.items():
            residual_pooled = _pool_section_tensor(residual, span)
            for pooling, value in residual_pooled.items():
                key = ("residual", section_name, pooling)
                feature_store[key].append(value)
                feature_labels[key].append(1 if str(row["workflow_label"]) == "setting" else 0)
                feature_groups[key].append(str(row["arbitration_group_id"]))
            if router_logits is not None and int(span[1]) <= int(router_logits.shape[1]):
                router_pooled = _pool_section_tensor(router_logits, span)
                for pooling, value in router_pooled.items():
                    key = ("router", section_name, pooling)
                    feature_store[key].append(value)
                    feature_labels[key].append(1 if str(row["workflow_label"]) == "setting" else 0)
                    feature_groups[key].append(str(row["arbitration_group_id"]))
                router_rows_present += 1

    if not labels:
        raise RuntimeError("No valid full-sequence rows available for section attribution")

    y = np.asarray(labels, dtype=np.int64)
    results: list[dict[str, Any]] = []
    best_policy_row: dict[str, Any] | None = None
    direction_source: dict[str, Any] | None = None

    for (data_source, section_name, pooling), pooled_list in sorted(feature_store.items()):
        matrix = np.stack(pooled_list, axis=0)
        key = (data_source, section_name, pooling)
        key_y = np.asarray(feature_labels[key], dtype=np.int64)
        key_groups = np.asarray(feature_groups[key])
        for layer in captured_layers:
            layer_idx = layer_to_idx[int(layer)]
            X = matrix[:, layer_idx, :]
            probe = _balanced_probe(X, key_y, groups=key_groups, n_folds=n_folds, seed=seed)
            if probe is None:
                continue
            row = {
                "data_source": data_source,
                    "section_name": section_name,
                    "pooling": pooling,
                    "layer": int(layer),
                    "n_source_examples": int(len(key_y)),
                    **probe,
                }
            results.append(row)
            if (
                data_source == "residual"
                and section_name in {"strategy", "settings"}
                and (best_policy_row is None or float(row["balanced_accuracy"]) > float(best_policy_row["balanced_accuracy"]))
            ):
                best_policy_row = dict(row)
                direction_source = {
                    "section_name": section_name,
                    "pooling": pooling,
                    "layer": int(layer),
                    "features": X,
                    "labels": key_y,
                }

    if not results or best_policy_row is None or direction_source is None:
        raise RuntimeError("Failed to compute section attribution results")

    policy_direction = (
        np.mean(direction_source["features"][direction_source["labels"] == 1], axis=0)
        - np.mean(direction_source["features"][direction_source["labels"] == 0], axis=0)
    ).astype(np.float32)
    direction_norm = float(np.linalg.norm(policy_direction))
    if direction_norm <= 0:
        raise RuntimeError("Section direction norm is zero")
    normalized_direction = (policy_direction / direction_norm).astype(np.float32)

    pq.write_table(pa.Table.from_pylist(results), output_dir / "section_probe_results.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(metadata_rows), output_dir / "section_probe_metadata.parquet", compression="snappy")

    residual_rows = [row for row in results if row["data_source"] == "residual"]
    router_rows = [row for row in results if row["data_source"] == "router"]
    _plot_section_heatmap(
        residual_rows,
        title="Residual Section Arbitration Probe",
        output_path=output_dir / "residual_section_heatmap.png",
    )
    _plot_section_heatmap(
        router_rows,
        title="Router Section Arbitration Probe",
        output_path=output_dir / "router_section_heatmap.png",
    )

    summary = {
        "capture_run_id": capture_run_id,
        "relation_name": relation_name,
        "n_examples": int(len(labels)),
        "label_counts": {
            "strategy": int(int((y == 0).sum())),
            "setting": int(int((y == 1).sum())),
        },
        "captured_layers": [int(layer) for layer in captured_layers],
        "router_section_rows": int(router_rows_present),
        "best_policy_probe": best_policy_row,
        "direction_spec": {
            "section_name": str(direction_source["section_name"]),
            "pooling": str(direction_source["pooling"]),
            "layer": int(direction_source["layer"]),
            "direction_norm": round(direction_norm, 6),
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(
        output_dir / "policy_direction.npz",
        direction=normalized_direction,
        layer=np.asarray([int(direction_source["layer"])], dtype=np.int64),
        labels=y,
    )
    data_volume.commit()
    return summary


@app.function(
    volumes={"/data": data_volume, "/models": model_volume},
    image=gpu_image,
    gpu="H200",
    timeout=12 * 3600,
    cpu=8,
    memory=8 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_causal_check(
    *,
    relation_name: str = DEFAULT_RELATION,
    base_relation: str = DEFAULT_BASE_RELATION,
    analysis_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    batch_size: int = 8,
    max_tokens: int = 48,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    strength: float = 1.0,
    enable_thinking: bool = False,
    capture_reasoning: bool = False,
    reasoning_parser: str = "",
    max_rows: int = 0,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from pipelines.interp.modal_vllm_engine import (
        VLLMCaptureConfig,
        _cleanup_cuda_memory,
        _create_llm,
        _destroy_llm,
        _generate_batch_vllm,
        _init_activation_patching_on_model,
        _register_activation_patch_basis_on_model,
    )

    analysis_dir = Path("/data/analysis_results") / analysis_subdir / "section_attribution"
    causal_dir = Path("/data/analysis_results") / output_subdir / "causal_check"
    causal_dir.mkdir(parents=True, exist_ok=True)

    section_summary = json.loads((analysis_dir / "summary.json").read_text())
    direction_npz = np.load(analysis_dir / "policy_direction.npz")
    direction = np.asarray(direction_npz["direction"], dtype=np.float32)
    target_layer = int(section_summary["direction_spec"]["layer"])
    section_name = str(section_summary["direction_spec"]["section_name"])
    pooling = str(section_summary["direction_spec"]["pooling"])

    rows = _load_causal_rows(relation_name, base_relation)
    local_model_path = f"/models/{model_id}"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        if system_user is None:
            continue
        system_text, user_text = system_user
        section_spans, _ = _find_section_token_spans(
            tokenizer=tokenizer,
            system_text=system_text,
            user_text=user_text,
        )
        section_span = section_spans.get(section_name)
        if section_span is None:
            continue
        if pooling == "eos":
            patch_span = (int(section_span[1]) - 1, int(section_span[1]))
        else:
            patch_span = (int(section_span[0]), int(section_span[1]))
        prepared_rows.append(
            {
                **row,
                "messages": messages,
                "patch_span": patch_span,
            }
        )

    if not prepared_rows:
        raise RuntimeError("No valid rows prepared for causal check")
    if int(max_rows) > 0:
        prepared_rows = prepared_rows[: int(max_rows)]

    generation_cfg = VLLMCaptureConfig(
        output_dir=Path("/tmp/prompt_confusion_causal"),
        model_id=local_model_path,
        capture_reasoning=bool(capture_reasoning),
        reasoning_parser=str(reasoning_parser),
        capture_router=False,
        capture_residual=False,
        add_generation_prompt=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_seqs=max(1, int(batch_size)),
        max_num_batched_tokens=max(40960, max(1, int(batch_size)) * 4096),
        async_scheduling=False if int(batch_size) > 1 else None,
        worker_cls=(
            "pipelines.interp.patching.activation_patch_request_worker.ActivationPatchGPUWorker"
            if int(batch_size) > 1
            else ""
        ),
        request_scoped_patching=bool(int(batch_size) > 1),
    )

    llm = _create_llm(generation_cfg)
    try:
        _init_activation_patching_on_model(llm)
        basis_payload = {
            int(target_layer): {
                "mean": np.zeros_like(direction, dtype=np.float32),
                "scale": np.ones_like(direction, dtype=np.float32),
                "components": np.expand_dims(direction.astype(np.float32), axis=0),
                "named_components": {"setting_minus_strategy": 0},
            }
        }
        _register_activation_patch_basis_on_model(llm, basis_payload)

        condition_specs = {
            "baseline": None,
            "setting_push": float(strength),
            "strategy_push": -float(strength),
        }

        row_level: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for condition_name, signed_strength in condition_specs.items():
            for offset in range(0, len(prepared_rows), max(1, int(batch_size))):
                chunk = prepared_rows[offset : offset + max(1, int(batch_size))]
                requests = []
                for row in chunk:
                    request: dict[str, Any] = {"messages": row["messages"]}
                    if signed_strength is not None:
                        request["patch_spec"] = {
                            "mode": "add_direction",
                            "target_layers": [int(target_layer)],
                            "token_span": [int(row["patch_span"][0]), int(row["patch_span"][1])],
                            "strength": float(signed_strength),
                            "direction_weights_by_layer": {str(int(target_layer)): [1.0]},
                        }
                    requests.append(request)

                outputs = _generate_batch_vllm(
                    llm=llm,
                    tokenizer=tokenizer,
                    batch_requests=requests,
                    config=generation_cfg,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    chat_template_kwargs={"enable_thinking": bool(enable_thinking)},
                )
                for row, output in zip(chunk, outputs, strict=False):
                    classified = _classify_readout_side(row, str(output["generated_text"]))
                    row_level.append(
                        {
                            "condition": condition_name,
                            "log_id": int(row["log_id"]),
                            "example_id": str(row["example_id"]),
                            "strategy_family": str(row["strategy_family"]),
                            "environment_pressure_bucket": str(row["environment_pressure_bucket"]),
                            "baseline_label": str(row["workflow_label"]),
                            "generated_text": str(output["generated_text"]),
                            "finish_reason": str(output.get("finish_reason") or ""),
                            "reasoning_text": str(output.get("reasoning_text") or ""),
                            "patch_span_start": int(row["patch_span"][0]),
                            "patch_span_end": int(row["patch_span"][1]),
                            **classified,
                        }
                    )

            grouped_condition_rows = [row for row in row_level if row["condition"] == condition_name]
            total = len(grouped_condition_rows)
            counts = defaultdict(int)
            family_counts: dict[str, dict[str, int]] = {}
            for item in grouped_condition_rows:
                counts[str(item["readout_side"])] += 1
                family = str(item["strategy_family"])
                bucket = family_counts.setdefault(family, defaultdict(int))
                bucket[str(item["readout_side"])] += 1
            summary_rows.append(
                {
                    "condition": condition_name,
                    "n_rows": int(total),
                    "strategy_rate": round(float(counts["strategy"] / total), 4) if total else 0.0,
                    "setting_rate": round(float(counts["setting"] / total), 4) if total else 0.0,
                    "neither_rate": round(float(counts["neither"] / total), 4) if total else 0.0,
                    "family_breakdown_json": json.dumps(
                        {
                            family: {
                                "strategy": int(bucket["strategy"]),
                                "setting": int(bucket["setting"]),
                                "neither": int(bucket["neither"]),
                            }
                            for family, bucket in family_counts.items()
                        },
                        sort_keys=True,
                    ),
                }
            )

        pq.write_table(pa.Table.from_pylist(row_level), causal_dir / "row_level.parquet", compression="snappy")
        pq.write_table(pa.Table.from_pylist(summary_rows), causal_dir / "summary.parquet", compression="snappy")
        summary = {
            "target_layer": int(target_layer),
            "section_name": section_name,
            "pooling": pooling,
            "strength": float(strength),
            "n_rows": int(len(prepared_rows)),
            "analysis_subdir": str(analysis_subdir),
            "conditions": summary_rows,
            "output_dir": str(causal_dir),
        }
        (causal_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        data_volume.commit()
        return summary
    finally:
        _destroy_llm(llm)
        _cleanup_cuda_memory()


@app.function(
    volumes={"/data": data_volume, "/models": model_volume},
    image=base_image,
    timeout=7200,
    cpu=8,
    memory=16 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_confound_checks(
    *,
    relation_name: str = DEFAULT_RELATION,
    base_relation: str = DEFAULT_BASE_RELATION,
    output_subdir: str = DEFAULT_STAGE2_OUTPUT_SUBDIR,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer

    output_dir = Path("/data/analysis_results") / output_subdir / "confound_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(f"/models/{model_id}")
    rows = _load_confound_rows(relation_name, base_relation)

    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        if system_user is None:
            continue
        system_text, user_text = system_user
        section_spans, seq_len = _find_section_token_spans(
            tokenizer=tokenizer,
            system_text=system_text,
            user_text=user_text,
        )
        record = {
            "log_id": int(row["log_id"]),
            "workflow_label": str(row["workflow_label"]),
            "arbitration_group_id": str(row["arbitration_group_id"]),
            "strategy_family": str(row["strategy_family"]),
            "strategy_variant_id": str(row["strategy_variant_id"]),
            "setting_lexical_family_id": str(row["setting_lexical_family_id"]),
            "environment_pressure_bucket": str(row["environment_pressure_bucket"]),
            "context_variant_id": str(row["context_variant_id"]),
            "user_text": str(user_text),
            "seq_len": int(seq_len),
        }
        for section_name in SECTION_ORDER:
            span = section_spans.get(section_name)
            length = int(span[1] - span[0]) if span is not None else 0
            record[f"{section_name}_len"] = length
            record[f"{section_name}_start"] = int(span[0]) if span is not None else -1
            record[f"{section_name}_end"] = int(span[1]) if span is not None else -1
        record["settings_minus_strategy_len"] = int(record["settings_len"] - record["strategy_len"])
        total = max(1, int(record["seq_len"]))
        for section_name in SECTION_ORDER:
            record[f"{section_name}_share"] = round(float(record[f"{section_name}_len"] / total), 6)
        prepared_rows.append(record)

    if not prepared_rows:
        raise RuntimeError("No rows prepared for confound checks")

    y = np.asarray([1 if row["workflow_label"] == "setting" else 0 for row in prepared_rows], dtype=np.int64)
    groups = np.asarray([row["arbitration_group_id"] for row in prepared_rows], dtype=object)

    def _dict_pipeline(keys: list[str]) -> dict[str, Any] | None:
        features = [{key: row[key] for key in keys} for row in prepared_rows]
        return _evaluate_pipeline_cv(
            estimator_factory=lambda: make_pipeline(
                DictVectorizer(sparse=True),
                LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
            ),
            X=features,
            y=y,
            groups=groups,
            n_folds=n_folds,
            seed=seed,
        )

    numeric_keys = [
        "seq_len",
        "task_len",
        "strategy_len",
        "settings_len",
        "portfolio_len",
        "market_len",
        "settings_minus_strategy_len",
        "task_share",
        "strategy_share",
        "settings_share",
        "portfolio_share",
        "market_share",
        "task_start",
        "strategy_start",
        "settings_start",
        "portfolio_start",
        "market_start",
    ]
    numeric_X = np.asarray([[float(row[key]) for key in numeric_keys] for row in prepared_rows], dtype=np.float32)
    numeric_result = _evaluate_pipeline_cv(
        estimator_factory=lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        ),
        X=numeric_X,
        y=y,
        groups=groups,
        n_folds=n_folds,
        seed=seed,
    )

    text_result = _evaluate_pipeline_cv(
        estimator_factory=lambda: make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        ),
        X=np.asarray([row["user_text"] for row in prepared_rows], dtype=object),
        y=y,
        groups=groups,
        n_folds=n_folds,
        seed=seed,
    )

    result_rows: list[dict[str, Any]] = []
    configs = [
        ("family_only", _dict_pipeline(["strategy_family"])),
        ("pressure_only", _dict_pipeline(["environment_pressure_bucket"])),
        ("family_plus_pressure", _dict_pipeline(["strategy_family", "environment_pressure_bucket"])),
        (
            "lexical_ids",
            _dict_pipeline(["strategy_variant_id", "setting_lexical_family_id", "context_variant_id"]),
        ),
        (
            "metadata_all",
            _dict_pipeline(
                [
                    "strategy_family",
                    "environment_pressure_bucket",
                    "strategy_variant_id",
                    "setting_lexical_family_id",
                    "context_variant_id",
                ]
            ),
        ),
        ("length_position_numeric", numeric_result),
        ("user_text_ngram", text_result),
    ]
    for baseline_name, result in configs:
        if result is None:
            continue
        result_rows.append({"baseline_name": baseline_name, **result})

    pq.write_table(pa.Table.from_pylist(prepared_rows), output_dir / "row_metadata.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(result_rows), output_dir / "baseline_results.parquet", compression="snappy")
    summary = {
        "relation_name": relation_name,
        "base_relation": base_relation,
        "n_examples": int(len(prepared_rows)),
        "label_counts": {
            "strategy": int((y == 0).sum()),
            "setting": int((y == 1).sum()),
        },
        "baselines": result_rows,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    data_volume.commit()
    return summary


@app.function(
    volumes={"/data": data_volume, "/models": model_volume},
    image=attention_image,
    gpu="H200",
    timeout=12 * 3600,
    cpu=8,
    memory=64 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_attention_summary(
    *,
    relation_name: str = DEFAULT_RELATION,
    output_subdir: str = DEFAULT_STAGE2_OUTPUT_SUBDIR,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    layers: list[int] | None = None,
    max_rows: int = 0,
    n_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_layers = [int(v) for v in (layers or [20, 24, 36])]
    output_dir = Path("/data/analysis_results") / output_subdir / "attention_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(f"/models/{model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        f"/models/{model_id}",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    model.to("cuda")

    rows = _load_section_rows(relation_name)
    if int(max_rows) > 0:
        rows = rows[: int(max_rows)]

    row_level: list[dict[str, Any]] = []
    probe_features: dict[tuple[int, str], list[list[float]]] = defaultdict(list)
    probe_labels: dict[tuple[int, str], list[int]] = defaultdict(list)
    probe_groups: dict[tuple[int, str], list[str]] = defaultdict(list)

    try:
        for row in rows:
            messages = _parse_messages(row["prompt_messages_json"])
            system_user = _extract_system_user(messages)
            if system_user is None:
                continue
            system_text, user_text = system_user
            rendered = _render_chat_text(tokenizer, system_text, user_text)
            input_ids, _ = _token_offsets_for_rendered(tokenizer, rendered)
            section_spans, seq_len = _find_section_token_spans(
                tokenizer=tokenizer,
                system_text=system_text,
                user_text=user_text,
            )
            eos_idx = _resolve_prompt_eos_index(input_ids, getattr(tokenizer, "eos_token_id", None))
            anchors: dict[str, int] = {"prompt_eos": int(eos_idx)}
            if "strategy" in section_spans:
                anchors["strategy_eos"] = int(section_spans["strategy"][1] - 1)
            if "settings" in section_spans:
                anchors["settings_eos"] = int(section_spans["settings"][1] - 1)

            batch_input = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            batch_input = {k: v.to("cuda") for k, v in batch_input.items()}
            with torch.no_grad():
                outputs = model(**batch_input, use_cache=False, output_attentions=True)
            attentions = getattr(outputs, "attentions", None)
            if not attentions:
                raise RuntimeError("Model did not return attention tensors")

            label = 1 if str(row["workflow_label"]) == "setting" else 0
            group = str(row["arbitration_group_id"])

            for layer in target_layers:
                if layer >= len(attentions):
                    continue
                att = attentions[layer][0].detach().float().cpu().numpy()
                for anchor_name, query_idx in anchors.items():
                    feature_vector: list[float] = []
                    for section_name in SECTION_ORDER:
                        span = section_spans.get(section_name)
                        if span is None or int(span[1]) > seq_len:
                            mass_by_head = np.zeros(att.shape[0], dtype=np.float32)
                        else:
                            mass_by_head = att[:, int(query_idx), int(span[0]) : int(span[1])].sum(axis=-1).astype(
                                np.float32
                            )
                        feature_vector.append(float(mass_by_head.mean()))
                        row_level.append(
                            {
                                "log_id": int(row["log_id"]),
                                "workflow_label": str(row["workflow_label"]),
                                "arbitration_group_id": group,
                                "strategy_family": str(row["strategy_family"]),
                                "layer": int(layer),
                                "anchor": anchor_name,
                                "target_section": section_name,
                                "query_idx": int(query_idx),
                                "section_start": int(span[0]) if span is not None else -1,
                                "section_end": int(span[1]) if span is not None else -1,
                                "mean_attention_mass": round(float(mass_by_head.mean()), 6),
                                "max_head_attention_mass": round(float(mass_by_head.max()), 6),
                            }
                        )
                    probe_key = (int(layer), anchor_name)
                    probe_features[probe_key].append(feature_vector)
                    probe_labels[probe_key].append(label)
                    probe_groups[probe_key].append(group)
            del outputs
            torch.cuda.empty_cache()

        summary_rows: list[dict[str, Any]] = []
        for (layer, anchor_name), feature_rows in sorted(probe_features.items()):
            X = np.asarray(feature_rows, dtype=np.float32)
            y = np.asarray(probe_labels[(layer, anchor_name)], dtype=np.int64)
            groups = np.asarray(probe_groups[(layer, anchor_name)], dtype=object)
            probe = _balanced_probe(X, y, groups=groups, n_folds=n_folds, seed=seed)
            if probe is None:
                continue
            summary_rows.append(
                {
                    "layer": int(layer),
                    "anchor": anchor_name,
                    "n_features": int(X.shape[1]),
                    **probe,
                }
            )

        pq.write_table(pa.Table.from_pylist(row_level), output_dir / "row_level.parquet", compression="snappy")
        pq.write_table(pa.Table.from_pylist(summary_rows), output_dir / "probe_summary.parquet", compression="snappy")

        prompt_rows = [row for row in row_level if row["anchor"] == "prompt_eos"]
        if prompt_rows:
            by_section: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(lambda: {"strategy": [], "setting": []})
            for row in prompt_rows:
                by_section[(int(row["layer"]), str(row["target_section"]))][str(row["workflow_label"])].append(
                    float(row["mean_attention_mass"])
                )
            matrix = np.full((len(SECTION_ORDER), len(target_layers)), np.nan, dtype=np.float32)
            for i, section_name in enumerate(SECTION_ORDER):
                for j, layer in enumerate(target_layers):
                    bucket = by_section.get((int(layer), section_name))
                    if not bucket:
                        continue
                    if not bucket["strategy"] or not bucket["setting"]:
                        continue
                    matrix[i, j] = float(np.mean(bucket["setting"]) - np.mean(bucket["strategy"]))
            fig, ax = plt.subplots(figsize=(1.2 * len(target_layers), 0.55 * len(SECTION_ORDER) + 1.6))
            im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(target_layers)))
            ax.set_xticklabels([str(v) for v in target_layers])
            ax.set_yticks(range(len(SECTION_ORDER)))
            ax.set_yticklabels(list(SECTION_ORDER))
            ax.set_xlabel("Layer")
            ax.set_ylabel("Target section")
            ax.set_title("Prompt-EOS Attention Delta (setting - strategy)")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    if not math.isnan(float(value)):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Mean attention delta")
            fig.tight_layout()
            fig.savefig(output_dir / "prompt_eos_attention_delta.png", dpi=180)
            plt.close(fig)

        summary = {
            "relation_name": relation_name,
            "model_id": model_id,
            "n_examples": int(len({int(row["log_id"]) for row in row_level})),
            "anchors": sorted({str(row["anchor"]) for row in row_level}),
            "layers": [int(v) for v in target_layers],
            "probe_summary": summary_rows,
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        data_volume.commit()
        return summary
    finally:
        del model
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


@app.local_entrypoint()
def main(
    relation_name: str = DEFAULT_RELATION,
    base_relation: str = DEFAULT_BASE_RELATION,
    capture_run_id: str = DEFAULT_CAPTURE_RUN_ID,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    analysis_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    n_folds: int = 5,
    seed: int = 42,
    batch_size: int = 8,
    max_tokens: int = 48,
    strength: float = 1.0,
    run_section: bool = True,
    run_causal: bool = True,
    run_confounds: bool = False,
    run_attention: bool = False,
    enable_thinking: bool = False,
    capture_reasoning: bool = False,
    max_rows: int = 0,
) -> None:
    results: dict[str, Any] = {}
    if run_section:
        results["section"] = run_section_attribution.remote(
            relation_name=relation_name,
            capture_run_id=capture_run_id,
            activations_subdir=activations_subdir,
            output_subdir=output_subdir,
            model_id=model_id,
            n_folds=n_folds,
            seed=seed,
        )
    if run_causal:
        results["causal"] = run_causal_check.remote(
            relation_name=relation_name,
            base_relation=base_relation,
            analysis_subdir=analysis_subdir,
            output_subdir=output_subdir,
            model_id=model_id,
            batch_size=batch_size,
            max_tokens=max_tokens,
            strength=strength,
            enable_thinking=enable_thinking,
            capture_reasoning=capture_reasoning,
            max_rows=max_rows,
        )
    if run_confounds:
        results["confounds"] = run_confound_checks.remote(
            relation_name=relation_name,
            base_relation=base_relation,
            output_subdir=output_subdir,
            model_id=model_id,
            n_folds=n_folds,
            seed=seed,
        )
    if run_attention:
        results["attention"] = run_attention_summary.remote(
            relation_name=relation_name,
            output_subdir=output_subdir,
            model_id=model_id,
            max_rows=max_rows,
            n_folds=n_folds,
            seed=seed,
        )
    print(json.dumps(results, indent=2))

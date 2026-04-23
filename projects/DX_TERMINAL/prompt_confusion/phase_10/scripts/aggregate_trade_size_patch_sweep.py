from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.storage.artifacts import ArtifactManifest, OperationArtifact
from pipelines_v2.storage.modal import ModalVolumeStore

from projects.DX_TERMINAL.prompt_confusion.paths import pipelines_catalog_root

CATALOG_ROOT = pipelines_catalog_root()
RUN_ID = "wr_b05b536729e5_8587a67e"
WORKFLOW_STEP_ROOT = CATALOG_ROOT / "workflow_steps" / RUN_ID
ARTIFACT_ROOT = "/data/artifacts/prompt_confusion_trade_size_activation_patch_layer_sweep"
LAYERS = (28, 32, 36, 40)


def load_operation_result(artifact_id: str) -> dict:
    manifest_path = CATALOG_ROOT / f"{artifact_id}.json"
    manifest = ArtifactManifest.from_dict(json.loads(manifest_path.read_text()))
    store = ModalVolumeStore(name="xenon-data", root=ARTIFACT_ROOT)
    artifact = OperationArtifact(_manifest=manifest, store=store)
    payload = artifact.result()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping result for {artifact_id}, got {type(payload).__name__}")
    return payload


def parse_json_payload(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        payload = json.loads(raw[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_artifact_id(step_name: str) -> str:
    payload = json.loads((WORKFLOW_STEP_ROOT / f"{step_name}.json").read_text())
    return str(payload["artifact_id"])


def main() -> None:
    baseline_artifact_id = load_artifact_id("baseline_conflict_rows")
    baseline_result = load_operation_result(baseline_artifact_id)
    baseline_rows = {row["example_key"]: row for row in baseline_result["rows"]}

    summary: dict[str, object] = {
        "run_id": RUN_ID,
        "baseline_artifact_id": baseline_artifact_id,
        "row_count": len(baseline_rows),
        "layers": {},
    }

    for layer in LAYERS:
        layer_summary: dict[str, object] = {}
        variants = {
            "swap_to_aligned": load_artifact_id(f"swap_to_aligned_centroid_l{layer}"),
            "same_label_control": load_artifact_id(f"swap_to_conflict_centroid_control_l{layer}"),
            "random_control": load_artifact_id(f"random_control_patch_l{layer}"),
        }
        for variant_name, artifact_id in variants.items():
            patched_result = load_operation_result(artifact_id)
            patched_rows = {row["example_key"]: row for row in patched_result["rows"]}
            metrics = {
                "artifact_id": artifact_id,
                "valid_json": 0,
                "size_changed": 0,
                "patched_follows_setting": 0,
                "patched_follows_strategy": 0,
                "intended_erasure_flip": 0,
                "reverse_flip": 0,
                "malformed": 0,
                "changed_examples": [],
            }
            for key, baseline in baseline_rows.items():
                patched = patched_rows[key]
                example = patched["example"]
                strategy_size = str(example["labels"]["strategy_direction"])
                setting_size = str(example["labels"]["setting_implied_direction"])
                baseline_payload = parse_json_payload(baseline["generated_text"])
                patched_payload = parse_json_payload(patched["generated_text"])
                baseline_size = str((baseline_payload or {}).get("size") or "")
                patched_size = str((patched_payload or {}).get("size") or "")
                valid_json = patched_payload is not None

                metrics["valid_json"] += int(valid_json)
                metrics["size_changed"] += int(patched_size != baseline_size)
                metrics["patched_follows_setting"] += int(patched_size == setting_size)
                metrics["patched_follows_strategy"] += int(patched_size == strategy_size)
                metrics["intended_erasure_flip"] += int(
                    baseline_size == setting_size and patched_size == strategy_size
                )
                metrics["reverse_flip"] += int(
                    baseline_size == strategy_size and patched_size == setting_size
                )
                metrics["malformed"] += int(not valid_json)

                if patched_size != baseline_size and len(metrics["changed_examples"]) < 10:
                    metrics["changed_examples"].append(
                        {
                            "example_key": key,
                            "baseline_size": baseline_size,
                            "patched_size": patched_size,
                            "strategy_size": strategy_size,
                            "setting_size": setting_size,
                        }
                    )
            layer_summary[variant_name] = metrics
        summary["layers"][str(layer)] = layer_summary

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

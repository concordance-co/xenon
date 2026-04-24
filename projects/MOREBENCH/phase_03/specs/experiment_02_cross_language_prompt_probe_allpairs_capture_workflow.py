from __future__ import annotations

import json
from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    Example,
    FileCatalog,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_cross_language_pilot_workflow as pilot
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


PHASE_02_EXAMPLES = Path("projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_examples.jsonl")
PHASE_02_CONTROLS = Path("projects/MOREBENCH/phase_02/outputs/theory_prompt_variant_sweep_controls.jsonl")
FULL30_TRANSFORM_RESULT = (
    Path("artifacts")
    / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
    / "transform_33d92c1d07d0_339727c1"
    / "result.json"
)

LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_allpairs_catalog"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_allpairs"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02_cross_language_prompt_probe_allpairs"

TARGET_PRIMES = (
    "deontology",
    "virtue_ethics",
    "utilitarian",
    "contractarianism",
    "contractualism",
    "generic_ethics_control",
)

TRANSLATED_CUES = {
    "deontology": {
        "es": "Evalúa cada opción por el principio que sigue y por si ese principio podría quererse de manera coherente en casos semejantes. Prefiere la opción que respeta la condición moral de cada persona en lugar de pasarla por alto por conveniencia.",
        "zh": "根据每个选项所遵循的原则，以及该原则在类似情形中能否被一致地意愿为普遍准则来评估每个选项。优先选择尊重每个人道德地位、而不是为了便利将其压倒的做法。",
    },
    "virtue_ethics": {
        "es": "Evalúa cada opción por la calidad de juicio y de carácter que expresa en esta situación. Prefiere la opción que refleje sabiduría práctica, equilibrio adecuado y una virtud bien formada.",
        "zh": "根据每个选项在此情境中体现出的判断品质和品格来评估它。优先选择体现实践智慧、恰当分寸和成熟德性的做法。",
    },
    "utilitarian": {
        "es": "Evalúa cada opción por sus efectos totales sobre el bienestar de todas las personas. Prefiere la opción cuyo patrón esperado de beneficios y cargas sea mejor en conjunto una vez que se cuenta a cada persona afectada.",
        "zh": "根据每个选项对所有人的福祉所产生的总体影响来评估它。优先选择在把每个受影响的人都算进去之后，其预期收益与负担整体上最优的做法。",
    },
    "contractarianism": {
        "es": "Evalúa cada opción por aquello a lo que partes racionales podrían acordar cuando cada una busca términos de ventaja fiables. Prefiere la opción que mejor sostenga una negociación justa, una contención recíproca y un arreglo estable que las personas tengan razones para mantener.",
        "zh": "根据理性各方在各自寻求可靠有利条件时能够同意什么来评估每个选项。优先选择最能支持公平协商、相互克制以及人们有理由维持的稳定安排的做法。",
    },
    "contractualism": {
        "es": "Evalúa cada opción por las quejas que dejaría a cada persona afectada bajo el principio que la rige. Prefiere la opción respaldada por principios que nadie podría rechazar razonablemente.",
        "zh": "根据每个选项在其所遵循原则之下会给每个受影响者留下什么样的申诉来评估它。优先选择由任何人都无法合理拒绝的原则所支持的做法。",
    },
    "generic_ethics_control": {
        "es": "Evalúa en cada opción las consideraciones moralmente relevantes y compara en qué medida cada una queda apoyada en conjunto. Prefiere la opción con el caso más fuerte en términos globales.",
        "zh": "评估每个选项中在道德上相关的考量，并比较每个选项在整体上得到支持的程度。优先选择综合来看理由最强的做法。",
    },
}


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _extract_dilemma_from_prompt(prompt: str) -> str:
    return prompt.split("\n\nDILEMMA:\n", 1)[1]


def _english_theory_rows() -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], dict[str, object]] = {}
    for row in _load_jsonl(PHASE_02_EXAMPLES):
        group_id = str(row.get("group_id") or "")
        prime = str(row.get("prime_condition") or "")
        bank = str(row.get("variant_bank") or row.get("description_bank") or "")
        if bank == "analytic" and prime in TARGET_PRIMES:
            rows[(group_id, prime)] = row
    return rows


def _english_generic_rows() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for row in _load_jsonl(PHASE_02_CONTROLS):
        group_id = str(row.get("group_id") or "")
        control_type = str(row.get("control_type") or row.get("prime_condition") or "")
        bank = str(row.get("variant_bank") or row.get("description_bank") or "")
        if bank == "analytic" and control_type == "generic_ethics_control":
            rows[group_id] = row
    return rows


def _translated_dilemmas() -> dict[tuple[str, str], str]:
    payload = json.loads(FULL30_TRANSFORM_RESULT.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    mapping: dict[tuple[str, str], str] = {}
    for example in dataset["examples"]:
        labels = dict(example.get("labels", {}))
        metadata = dict(example.get("metadata", {}))
        group_id = str(labels.get("group_id") or "")
        language_code = str(labels.get("language_code") or "")
        dilemma_text = str(metadata.get("dilemma_text") or "")
        if group_id and language_code and dilemma_text:
            mapping[(group_id, language_code)] = dilemma_text
    return mapping


def _group_order(rows: dict[tuple[str, str], dict[str, object]]) -> list[str]:
    return sorted({group_id for group_id, _ in rows.keys()})


def _user_prompt(*, language_code: str, cue_text: str, dilemma_text: str) -> str:
    cfg = pilot.LANGUAGE_CONFIG[language_code]
    return (
        f"{cfg['analysis_instruction']}\n\n"
        f"{cfg['framework_header']}\n{cue_text}\n\n"
        f"{cfg['dilemma_header']}\n{dilemma_text}\n\n"
        f"{cfg['recommendation_instruction']}"
    )


def build_dataset() -> Dataset:
    english_theory_rows = _english_theory_rows()
    english_generic_rows = _english_generic_rows()
    translated_dilemmas = _translated_dilemmas()
    examples: list[Example] = []

    for group_id in _group_order(english_theory_rows):
        representative = english_theory_rows[(group_id, "deontology")]
        english_dilemma = _extract_dilemma_from_prompt(str(representative["prompt"]))
        for prime in TARGET_PRIMES:
            if prime == "generic_ethics_control":
                source_row = english_generic_rows[group_id]
                english_cue = str(source_row["cue_text"])
            else:
                source_row = english_theory_rows[(group_id, prime)]
                english_cue = str(source_row["cue_text"])
            for language_code in ("en", "es", "zh"):
                if language_code == "en":
                    cue_text = english_cue
                    dilemma_text = english_dilemma
                else:
                    cue_text = TRANSLATED_CUES[prime][language_code]
                    dilemma_text = translated_dilemmas[(group_id, language_code)]
                cfg = pilot.LANGUAGE_CONFIG[language_code]
                key = f"{group_id}__{prime}__lang_{language_code}"
                examples.append(
                    Example(
                        key=key,
                        prompt=[
                            {"role": "system", "content": cfg["system_prompt"]},
                            {
                                "role": "user",
                                "content": _user_prompt(
                                    language_code=language_code,
                                    cue_text=cue_text,
                                    dilemma_text=dilemma_text,
                                ),
                            },
                        ],
                        labels={
                            "group_id": group_id,
                            "prime_condition": prime,
                            "prime_family": "cross_language_allpairs_full30",
                            "language_code": language_code,
                            "language_name": cfg["name"],
                            "cue_mode": "translated_analytic",
                            "cue_text": cue_text,
                            "source_family": str(representative.get("source_family") or ""),
                            "dilemma_type": str(representative.get("dilemma_type") or ""),
                            "context": str(representative.get("context") or ""),
                            "role_domain": str(representative.get("role_domain") or ""),
                            "is_generic_control": bool(prime == "generic_ethics_control"),
                        },
                        metadata={"cue_text": cue_text, "dilemma_text": dilemma_text},
                        cases={"group_id": group_id},
                        case_key=group_id,
                    )
                )
    return Dataset.from_examples(
        examples,
        name="morebench_phase03_experiment02_cross_language_prompt_probe_allpairs_full30",
    )


def build_runner_specs() -> dict[str, object]:
    catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
    modal_store = ModalVolumeStore(
        name="xenon-data",
        root=MODAL_ARTIFACT_ROOT,
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 6,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def build_workflow() -> WorkflowSpec:
    dataset = build_dataset()
    return WorkflowSpec(
        name="morebench_phase03_experiment02_cross_language_prompt_probe_allpairs_capture",
        steps=(
            WorkflowStep(
                name="capture_prompt_eos_residual",
                runner="capture_gpu",
                description="Capture prompt-final residuals on the full six-prime translated prompt set.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="prompt_eos_residual",
                            site="resid_post",
                            layers=list(base.CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        )
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
        ),
    )

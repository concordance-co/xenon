"""ARCH2 expected usage: capture, baseline generation, patched generation, compare."""

from pipelines_v2.api import (
    ActivationBankSpec,
    CaptureSpec,
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    InterchangePatch,
    PatchComparisonSpec,
    PatchedGenerationSpec,
    PromptMetadataBuilder,
    ResidualInterventionSite,
    ResidualSite,
    TokenSelector,
    TransformBuilder,
    VLLMEngine,
)


def build_prompt_metadata(rendered_prompt: str) -> dict[str, object]:
    market_marker = "MARKET\n"
    decision_marker = "\n\nDECISION\n"
    market_start = rendered_prompt.index(market_marker) + len(market_marker)
    market_end = rendered_prompt.index(decision_marker, market_start)
    return {
        "token_sections": {
            "MARKET": {"char_start": market_start, "char_end": market_end},
        }
    }


def score_patch_row(*, example: dict, baseline: dict, variants: dict) -> dict[str, object]:
    patched = dict(variants or {}).get("main", {})
    return {
        "metrics": {
            "changed": str(baseline.get("generated_text") or "") != str(patched.get("generated_text") or ""),
        },
        "evaluation": {
            "example_key": str(example.get("key") or ""),
            "baseline_text": str(baseline.get("generated_text") or ""),
            "patched_text": str(patched.get("generated_text") or ""),
        },
    }


def build_dataset() -> Dataset:
    return Dataset.from_examples(
        (
            Example(key="pair_a_target", prompt="MARKET\nALPHA up\n\nDECISION\nBUY", labels={"role": "target"}, case_key="pair_a"),
            Example(key="pair_a_donor", prompt="MARKET\nALPHA down\n\nDECISION\nSELL", labels={"role": "donor"}, case_key="pair_a"),
        ),
        name="arch2_activation_patch_example",
    )


def build_specs() -> dict[str, object]:
    dataset = build_dataset()
    engine = VLLMEngine(
        model_id="Qwen/Qwen3-30B-A3B",
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=16,
        max_num_batched_tokens=32768,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
    )
    prompt_metadata = PromptMetadataBuilder.from_function(build_prompt_metadata)

    capture = CaptureSpec(
        engine=engine,
        dataset=dataset,
        prompt_metadata_builder=prompt_metadata,
        sites=[
            ResidualSite(
                name="resid_post_full",
                site="resid_post",
                layers=[16],
                tokens=TokenSelector.full_sequence(),
            )
        ],
    )

    baseline = GenerationRunSpec(
        engine=engine,
        dataset=dataset,
        select_when=dataset.labels("role").equals("target"),
        generation=GenerationSpec(enabled=True, max_tokens=64, temperature=0.0),
    )

    activation_bank = ActivationBankSpec(
        feature="capture_artifact.feature('resid_post_full')",
        layers=[16],
    )

    patched = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=InterchangePatch(
            activation_bank="activation_bank_artifact",
            write_site=ResidualInterventionSite(site="resid_post", layers=(16,)),
            target_tokens=TokenSelector.section("MARKET"),
            donor_tokens=TokenSelector.section("MARKET"),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("role").equals("target"),
        donor_when=dataset.labels("role").equals("donor"),
        generation=GenerationSpec(enabled=True, max_tokens=64, temperature=0.0),
        prompt_metadata_builder=prompt_metadata,
    )

    comparison = PatchComparisonSpec(
        baseline="baseline_artifact",
        variants={"main": "patched_artifact"},
        row_evaluator=TransformBuilder.from_function(score_patch_row),
    )

    return {
        "capture": capture,
        "activation_bank": activation_bank,
        "baseline": baseline,
        "patched": patched,
        "comparison": comparison,
    }

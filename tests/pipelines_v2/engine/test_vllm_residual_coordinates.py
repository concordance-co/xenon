from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines_v2.api import (
    CaptureSpec, Dataset, Example, GenerationSpec, InterchangePatch,
    PatchedGenerationSpec, ResidualInterventionSite, ResidualSite, TokenSelector, VLLMEngine,
)
from pipelines_v2.engine.vllm.capture import _empty_feature, _fill_residual_features
from pipelines_v2.engine.vllm.intervention_build import build_activation_bank_runtime_payload
from pipelines_v2.engine.vllm.residual_coordinates import (
    audit_model_coordinates, auxiliary_indices, coordinate_metadata, validate_capture_coordinates,
)
from pipelines_v2.engine.vllm.session import build_vllm_session_llm_kwargs


def site(layers, *, legacy=False, name="residual"):
    return ResidualSite(name=name, site="resid_post", layers=layers, tokens=TokenSelector.full_sequence(),
                        layer_indexing="vllm_aux" if legacy else "decoder_block")


def fill(sites, indices, values):
    features = {s.name: _empty_feature(s) for s in sites}
    _fill_residual_features(feature_payloads=features, residual_sites=sites,
        residual_layers=indices, residual=values, example=Example(key="self", prompt="fixture"),
        token_count=2, token_sections={}, section_records=[])
    return features


def test_new_sites_and_old_serialized_sites_have_distinct_coordinates_and_hashes():
    new = site([0, 31, 79])
    old_payload = asdict(new)
    old_payload.pop("layer_indexing")
    old = ResidualSite.from_dict(old_payload)
    assert auxiliary_indices([new]) == [1, 32, 80]
    assert auxiliary_indices([old]) == [0, 31, 79]
    assert ResidualSite.from_dict(asdict(new)) == new
    engine = VLLMEngine(model_id="fixture")
    ds = Dataset.from_examples([Example(key="self", prompt="fixture")])
    a = CaptureSpec(engine=engine, dataset=ds, sites=[new])
    b = CaptureSpec(engine=engine, dataset=ds, sites=[old])
    assert a.spec_hash() != b.spec_hash()
    assert a.semantic_dict() != b.semantic_dict()


@pytest.mark.parametrize("block", [0, 2, 4])
def test_capture_to_same_block_self_replacement_is_identity_not_previous_boundary(block):
    torch = pytest.importorskip("torch")
    from pipelines_v2.engine.vllm.patching.base import extract_hidden_tensor, replace_hidden_tensor

    # Independent oracle: embedding boundary followed by five distinct block
    # outputs. Each block carries a branch/residual pair, as reviewed vLLM does.
    residual = torch.tensor([[1., 2.], [3., 4.]])
    boundaries = [residual.numpy().copy()]
    outputs = []
    for i in range(5):
        branch = (i + 1) * torch.tensor([[1., -1.], [2., 3.]])
        output = (branch, residual)
        residual = extract_hidden_tensor(output)
        outputs.append(output)
        boundaries.append(residual.numpy().copy())
    requested = site([block])
    indices = auxiliary_indices([requested])
    features = fill([requested], indices, np.stack([boundaries[i] for i in indices]))
    donor = torch.tensor(features["residual"]["layers"][str(block)]["self"]["values"])
    expected = extract_hidden_tensor(outputs[block])
    patched = extract_hidden_tensor(replace_hidden_tensor(outputs[block], donor))
    assert torch.equal(patched, expected)
    assert not torch.equal(torch.tensor(boundaries[block]), expected)  # original bug


def test_mixed_legacy_and_block_sites_and_session_superset_keep_the_right_tensor_axis():
    canonical = site([2], name="canonical")
    legacy = site([2], legacy=True, name="legacy")
    # Session captures a larger union than either operation requests.
    indices = [0, 1, 2, 3, 5]
    values = np.stack([np.full((2, 2), index, dtype=float) for index in indices])
    features = fill([canonical, legacy], indices, values)
    np.testing.assert_equal(features["canonical"]["layers"]["2"]["self"]["values"], 3.)
    np.testing.assert_equal(features["legacy"]["layers"]["2"]["self"]["values"], 2.)
    assert features["canonical"]["residual_coordinates"]["decoder_block_indices"] == {"2": 2}
    assert features["legacy"]["residual_coordinates"]["decoder_block_indices"] is None
    with pytest.raises(RuntimeError, match="layer axis"):
        fill([canonical], indices, values[:1])


def test_session_requests_translated_union_and_legacy_indices():
    engine = VLLMEngine(model_id="fixture")
    ds = Dataset.from_examples([Example(key="self", prompt="fixture")])
    specs = [CaptureSpec(engine=engine, dataset=ds, sites=[s])
             for s in (site([0, 4]), site([2], legacy=True))]
    with pytest.warns(UserWarning, match="Legacy vllm_aux"):
        kwargs, _, temporary = build_vllm_session_llm_kwargs(engine=engine, specs=specs)
    try:
        assert kwargs["speculative_config"]["draft_model_config"]["hf_config"][
            "eagle_aux_hidden_state_layer_ids"] == [1, 2, 5]
    finally:
        temporary.cleanup()


@pytest.mark.parametrize("kwargs", [dict(layer_indexing="guess"), dict(layers=[-1]), dict(layers=[])])
def test_invalid_coordinates_fail_at_construction(kwargs):
    with pytest.raises(ValueError):
        ResidualSite(**{**dict(name="x", site="resid_post", layers=[0]), **kwargs})


def test_pipeline_parallel_coordinates_are_not_assumed():
    with pytest.raises(ValueError, match="pipeline_parallel"):
        validate_capture_coordinates(VLLMEngine(model_id="fixture", pipeline_parallel_size=2), [site([0])])
    with pytest.raises(ValueError, match="only resid_post"):
        validate_capture_coordinates(VLLMEngine(model_id="fixture"),
            [ResidualSite(name="x", site="resid_pre", layers=[0])])


class _ReviewedDecoder:
    def __init__(self):
        self.layers = [None] * 5
        self.start_layer, self.end_layer = 0, 5

    def forward(self, hidden_states, residual):
        # Minimal source-shape fixture, not a substitute for a live GPU test.
        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, idx + 1, hidden_states, residual)
        return hidden_states, aux_hidden_states


def test_model_audit_rejects_unknown_version_architecture_and_out_of_bounds(monkeypatch):
    import pipelines_v2.engine.vllm.residual_coordinates as coordinates
    cls = _ReviewedDecoder
    identity = (cls.__module__, cls.__name__)
    monkeypatch.setattr(coordinates.importlib.metadata, "version", lambda _: "fixture")
    model = SimpleNamespace(model=cls())
    with pytest.raises(ValueError, match="Unreviewed"):
        audit_model_coordinates(model, requested_blocks=(0,))
    monkeypatch.setattr(coordinates, "_REVIEWED_MODELS", {"fixture": {identity}})
    assert audit_model_coordinates(model, requested_blocks=(0, 2, 4))["decoder_block_count"] == 5
    with pytest.raises(ValueError, match="outside"):
        audit_model_coordinates(model, requested_blocks=(5,))
    model.model.start_layer = 2
    with pytest.raises(ValueError, match="complete"):
        audit_model_coordinates(model, requested_blocks=(2,))


def test_legacy_bank_requires_explicit_source_map_without_relabeling():
    engine = VLLMEngine(model_id="fixture")
    ds = Dataset.from_examples([Example(key="self", prompt="fixture", case_key="pair", labels={"side": "target"})])
    bank = {"layers": {"3": {}}, "residual_coordinates": coordinate_metadata(site([3], legacy=True))}
    def spec(mapping):
        return PatchedGenerationSpec(engine=engine, dataset=ds,
            pair_by=ds.cases("case_key"), target_when=ds.labels("side").equals("target"),
            donor_when=ds.labels("side").equals("donor"),
            patch=InterchangePatch(activation_bank=bank,
                write_site=ResidualInterventionSite(site="resid_post", layers=(2,)),
                target_tokens=TokenSelector.full_sequence(), donor_tokens=TokenSelector.full_sequence(), source_layer_map=mapping),
            generation=GenerationSpec(enabled=True, max_tokens=1))
    with pytest.raises(ValueError, match="explicit source_layer_map"):
        build_activation_bank_runtime_payload(spec({}), bank, [])
    assert build_activation_bank_runtime_payload(spec({2: 3}), bank, []) == {3: {}}
    assert list(bank["layers"]) == ["3"]


def test_activation_bank_preserves_coordinate_provenance():
    from pipelines_v2.api import ActivationBankSpec
    from pipelines_v2.operations.execution.intervention_sources import run_activation_bank
    coords = coordinate_metadata(site([2]))
    payload = {"kind": "residual", "site": "resid_post", "residual_coordinates": coords,
               "layers": {"2": {"self": {"values": [[1., 2.]], "token_sections": {}}}}}
    feature = SimpleNamespace(load=lambda: payload, name="residual")
    result = run_activation_bank(ActivationBankSpec(feature=feature))
    assert result.payload["residual_coordinates"] == coords


def test_loaded_session_uses_union_coordinates_and_audits_before_generation(monkeypatch):
    import pipelines_v2.engine.vllm.capture as capture
    requested = site([2])
    example = Example(key="self", prompt="fixture")
    engine = VLLMEngine(model_id="fixture")
    spec = CaptureSpec(engine=engine, dataset=Dataset.from_examples([example]), sites=[requested])
    events = []

    def apply_model(fn):
        assert fn.func is audit_model_coordinates
        assert fn.keywords == {"requested_blocks": (2,)}
        events.append("audit")
        return [{"schema": "fixture", "decoder_block_count": 5}]

    def batch(**kwargs):
        assert events == ["audit"]
        events.append("generate")
        return [{"example": example, "prompt_token_ids": [1, 2], "prompt_token_sections": {},
                 "prompt_section_records": [], "residual_token_count": 2,
                 "residual_token_sections": {}, "residual_section_records": [],
                 "residual": np.stack([np.full((2, 2), i, dtype=float) for i in (1, 3, 5)]),
                 "router_data": {}, "actual_router_layers": [], "generation_result": None,
                 "generated_token_count": 0, "captured_generated_token_count": 0}]

    monkeypatch.setattr(capture, "_capture_prompt_batch", batch)
    runtime = SimpleNamespace(engine=engine, llm=SimpleNamespace(apply_model=apply_model),
        tokenizer=None, reasoning_parser_instance=None, load_performance={}, residual_auxiliary_indices=(1, 3, 5))
    result = capture.run_vllm_capture_with_runtime(runtime=runtime, spec=spec)
    np.testing.assert_equal(result.features["residual"]["layers"]["2"]["self"]["values"], 3.)
    assert result.metadata["residual_layer_coordinates"]["residual"]["runtime_audits"]
    assert events == ["audit", "generate"]
    runtime.residual_auxiliary_indices = (1, 5)
    with pytest.raises(ValueError, match="missing from the loaded session"):
        capture.run_vllm_capture_with_runtime(runtime=runtime, spec=spec)

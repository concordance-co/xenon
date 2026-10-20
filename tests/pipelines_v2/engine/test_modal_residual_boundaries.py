"""Opt-in live check of the first, middle, and final vLLM residual boundaries."""
import os
from pathlib import Path

import pytest


@pytest.mark.modal
@pytest.mark.vllm
@pytest.mark.network
@pytest.mark.slow
def test_modal_vllm_residual_boundary_self_patch():
    if os.getenv("XENON_RUN_MODAL_RESIDUAL_BOUNDARIES") != "1":
        pytest.skip("set XENON_RUN_MODAL_RESIDUAL_BOUNDARIES=1 for the one-L4 check")
    import modal
    from pipelines_v2.api import VLLMEngine
    from pipelines_v2.runtime.locks import build_locked_image

    def _check_residual_boundaries():
        import numpy as np
        from types import SimpleNamespace
        from transformers import AutoConfig
        from pipelines_v2.api import CaptureSpec, Dataset, Example, GenerationSpec, ResidualSite, TokenSelector, VLLMEngine, PatchedGenerationSpec, InterchangePatch, ResidualInterventionSite
        from pipelines_v2.operations.common.schemas import TensorStorage
        from pipelines_v2.engine.vllm.session import build_vllm_session_runtime
        from pipelines_v2.engine.vllm.patching.base import unwrap_model, extract_hidden_tensor, replace_hidden_tensor
        model_id='Qwen/Qwen3-0.6B'
        n = AutoConfig.from_pretrained(model_id).num_hidden_layers
        blocks=(0,n//2,n-1)
        engine=VLLMEngine(model_id=model_id,max_model_len=128,max_num_seqs=1,
            enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=True,max_num_batched_tokens=128,
            add_generation_prompt=False,enable_thinking=False,gpu_memory_utilization=.5,
            dtype='bfloat16')
        dataset=Dataset.from_examples([Example(key='self',prompt='The capital of France is',case_key='self',labels={'role':'self'})])
        spec=CaptureSpec(engine=engine,dataset=dataset,
            sites=[ResidualSite(name='residual',site='resid_post',layers=blocks,tokens=TokenSelector.full_sequence(),storage=TensorStorage(dtype='float32'))],
            generation=GenerationSpec(enabled=True,max_tokens=1,temperature=0.))
        def patch_spec(bank):
            return PatchedGenerationSpec(engine=engine,dataset=dataset,
                pair_by=dataset.cases('case_key'),target_when=dataset.labels('role').equals('self'),donor_when=dataset.labels('role').equals('self'),
                patch=InterchangePatch(activation_bank=bank,write_site=ResidualInterventionSite(site='resid_post',layers=blocks),
                    target_tokens=TokenSelector.full_sequence(),donor_tokens=TokenSelector.full_sequence()),generation=spec.generation)
        session=build_vllm_session_runtime(engine=engine,specs=[spec,patch_spec({'layers':{}})])
        def install(model):
            model=unwrap_model(model)
            model._boundary_probe={}
            for block in blocks:
                def hook(module,args,output,block=block):
                    branch,residual=output
                    combined=branch+residual
                    # Save independent sums before later fused operations mutate residual.
                    replacement=replace_hidden_tensor(output,combined.clone())
                    model._boundary_probe[block]={
                        'combined':combined.detach().float().cpu().numpy(),
                        'branch':branch.detach().float().cpu().numpy().copy(),
                        'extract_error':float((extract_hidden_tensor(output)-combined).abs().max()),
                        'self_write_error':float((replacement[0]+replacement[1]-combined).abs().max()),
                    }
                model.model.layers[block].register_forward_hook(hook)
            return True
        session.llm.apply_model(install)
        result=session.capture(spec)
        probes=session.llm.apply_model(lambda model: unwrap_model(model)._boundary_probe)[0]
        checks={}
        for block in blocks:
            actual=np.asarray(result.features['residual']['layers'][str(block)]['self']['values'])
            probe=probes[block]
            expected=probe['combined'][:len(actual)]
            checks[str(block)]={k:probe[k] for k in ('extract_error','self_write_error')}
            checks[str(block)]['capture_error']=float(np.max(np.abs(actual-expected)))
            checks[str(block)]['shape']=list(actual.shape)
        bank={**result.features['residual'],'kind':'activation_bank_result'}
        patched=session.intervene(patch_spec(SimpleNamespace(result=lambda:bank)))
        after=session.llm.apply_model(lambda model: unwrap_model(model)._boundary_probe)[0]
        for block in blocks:
            checks[str(block)]['live_patch_delta']=float(np.max(np.abs(after[block]['combined']-probes[block]['combined'])))
        return {'checks':checks,'generation':result.generations,'patched':patched.rows,'audit':result.metadata['residual_layer_coordinates']}

    runtime = VLLMEngine(model_id="Qwen/Qwen3-0.6B").runtime_spec()
    image = build_locked_image(modal, runtime).env(dict(runtime.env))
    image = image.env({"PYTHONPATH": "/root", "HF_HOME": "/models/hf"})
    root = Path(__file__).resolve().parents[3]
    image = image.add_local_dir(str(root / "pipelines_v2"), remote_path="/root/pipelines_v2")
    app = modal.App("xenon-residual-boundary-check")
    remote = app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=900,
        volumes={"/models": modal.Volume.from_name("xenon-models")}, serialized=True)(
            _check_residual_boundaries)
    with modal.enable_output(), app.run():
        result = remote.remote()
    _assert_boundary_result(result)


def _assert_boundary_result(result):
    assert len(result["checks"]) == 3
    for check in result["checks"].values():
        for metric in ("capture_error", "extract_error", "self_write_error", "live_patch_delta"):
            assert check[metric] == 0, (metric, check)
    assert len(result["patched"]) == 1
    patched = result["patched"][0]
    assert patched["status"] == "ok"
    assert patched["generated_token_ids"] == result["generation"][0]["generated_token_ids"]
    assert set(patched["patch_stats"]) == set(result["checks"])
    assert all(stat["status"] == "ok" for stat in patched["patch_stats"].values())

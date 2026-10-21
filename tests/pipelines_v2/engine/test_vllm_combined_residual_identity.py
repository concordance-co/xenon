"""Exercise production tuple hooks, not just an abstract residual oracle."""
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")
from pipelines_v2.engine.vllm.patching.base import extract_hidden_tensor, replace_hidden_tensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_combined_residual_self_write_preserves_both_components(dtype):
    branch = torch.tensor([[.123, -1.75, .001]], dtype=dtype)
    residual = torch.tensor([[40., 2., -15.]], dtype=dtype)
    output = (branch, residual, "untouched")
    captured = branch + residual
    assert torch.equal(extract_hidden_tensor(output), captured)
    patched = replace_hidden_tensor(output, captured)
    assert torch.equal(patched[0], branch)
    assert torch.equal(patched[1], residual)
    assert patched[2] == "untouched"
    assert torch.equal(extract_hidden_tensor(patched), captured)


def test_nonself_write_replaces_combined_state_not_branch():
    branch = torch.tensor([[1., 2.]])
    residual = torch.tensor([[10., 20.]])
    donor = torch.tensor([[3., 5.]])
    patched = replace_hidden_tensor((branch, residual), donor)
    assert torch.equal(patched[0]+patched[1], donor)
    assert torch.equal(residual, torch.tensor([[10., 20.]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_nonself_write_avoids_delta_cancellation(dtype):
    branch = torch.tensor([[.123, -1.75, .001]], dtype=dtype)
    residual = torch.tensor([[40., 2., -15.]], dtype=dtype)
    donor = torch.tensor([[3., 5., .1]], dtype=dtype)
    patched = replace_hidden_tensor((branch, residual, "untouched"), donor)
    assert torch.equal(patched[0] + patched[1], donor)
    assert patched[2] == "untouched"


def test_nonresidual_tuple_retains_existing_behavior():
    hidden = torch.tensor([[1., 2.]])
    replacement = torch.tensor([[3., 4.]])
    for other in (None, torch.tensor([1, 2]), torch.tensor([[1, 2]]), torch.zeros(3, 4)):
        output = (hidden, other)
        assert extract_hidden_tensor(output) is hidden
        patched = replace_hidden_tensor(output, replacement)
        assert patched[0] is replacement and patched[1] is other


def test_fallback_interchange_honors_explicit_source_layer():
    from pipelines_v2.engine.vllm.patching.apply import patch_hidden_states_for_layer
    donor=torch.tensor([[3.,5.]])
    owner=SimpleNamespace(_v2_activation_patch_bank={13:{"donor":{"values":donor}}})
    patched,stats=patch_hidden_states_for_layer(torch.zeros(1,2),owner_model=owner,layer_idx=12,
        batch_spec={"patch_spec":{"operator":"interchange","target_layers":[12],
            "donor_example_key":"donor","donor_positions":[0],"query_positions":[0],
            "source_layer_map":{"12":13}}})
    assert stats["status"]=="ok"
    assert torch.equal(patched,donor)

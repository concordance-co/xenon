import numpy as np
import pytest
from pipelines_v2.representation.sae.dictionary import SAEDictionary
from pipelines_v2.representation.sae.splice import splice,torch_splice


def dictionary():
    return SAEDictionary({'W_enc':np.eye(3,dtype=np.float32),'W_dec':np.eye(3,dtype=np.float32),
        'b_enc':np.zeros(3,dtype=np.float32),'b_dec':np.zeros(3,dtype=np.float32)},
        {'architecture':'standard','activation':'relu','normalization':'none'})


def test_noop_is_exact_but_reconstruction_is_not_noop():
    d=dictionary();x=np.array([[2,3,4]],dtype=np.float32)
    assert np.array_equal(splice(d,x,mode='noop'),x)
    assert np.array_equal(splice(d,x,mode='delta'),x)
    reconstruction=splice(d,x,mode='reconstruct')
    assert not np.array_equal(reconstruction,x)
    assert reconstruction.mean()==pytest.approx(x.mean())
    with pytest.raises(ValueError):splice(d,x,mode='noop',feature_edits={0:1})


def test_feature_delta_equals_independent_decoded_delta_preserving_error():
    d=dictionary();x=np.array([[2,3,4]],dtype=np.float32)
    edited=splice(d,x,mode='delta',feature_edits={0:3.0})
    assert np.array_equal(edited,x+np.array([[2,-1,-1]],dtype=np.float32))
    assert edited.mean()==x.mean()
    assert not np.array_equal(edited,splice(d,x,mode='reconstruct',feature_edits={0:3.0}))
    with pytest.raises(ValueError):splice(d,x,feature_edits={3:1})


def test_scoped_splice_changes_only_selected_prompt_position():
    torch=pytest.importorskip('torch')
    x=torch.tensor([[[2.,3.,4.],[5.,6.,7.]]])
    result=torch_splice(dictionary(),feature_edits={0:3},token_positions=(-1,))(x)
    assert torch.equal(result[:,0],x[:,0])
    assert torch.equal(result[:,-1],x[:,-1]+torch.tensor([2,-1,-1]))
    assert torch.equal(x,torch.tensor([[[2.,3.,4.],[5.,6.,7.]]]))

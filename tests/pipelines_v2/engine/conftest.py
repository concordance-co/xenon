import pytest


@pytest.fixture
def tiny_torch_session():
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from pipelines_v2.engine.torch import TorchSession
    class Tokenizer:
        eos_token_id = 30
        pad_token_id = 31
        def encode(self, text, **kwargs):
            return [int(item) for item in text.split()]
        def decode(self, tokens):
            return ' '.join(map(str,tokens))
    with torch.random.fork_rng():
        torch.manual_seed(47)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=32,n_positions=32,n_embd=16,n_layer=2,n_head=2,
                        resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,attn_implementation='eager'))
    session = TorchSession(model,Tokenizer(),model_identity='tiny-fixed-checkpoint')
    yield session
    session.close()

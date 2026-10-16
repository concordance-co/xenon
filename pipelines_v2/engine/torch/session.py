"""Scoped GPT-2 eager FP32 execution. Optional model dependencies stay lazy."""
from __future__ import annotations

from contextlib import contextmanager
import threading

ADAPTER_VERSION = 'gpt2-residual-v1'


class TorchSession:
    """Reusable frozen model; hooks and autograd state belong to one forward."""

    def __init__(self, model, tokenizer, *, model_identity=None, device='cpu'):
        import torch
        from pipelines_v2.runtime.numerics import configure_torch_threads
        configure_torch_threads()
        if model.config.model_type != 'gpt2':
            raise ValueError('TorchSession supports GPT-2 only')
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self.model.config._attn_implementation = 'eager'
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._lock = threading.RLock()
        self.closed = False
        self.model_identity = model_identity or "unqualified-fixture"

    @classmethod
    def from_engine(cls, engine):
        from pathlib import Path
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
        path = Path(engine.model_id).expanduser()
        if not path.is_dir():
            raise ValueError('Prepared offline GPT-2 cache missing; download the immutable revision separately')
        import json
        if json.loads((path / 'config.json').read_text()).get('model_type') != 'gpt2':
            raise ValueError('TorchEngine supports GPT-2 checkpoints only')
        config = GPT2Config.from_pretrained(str(path), local_files_only=True)
        if config.model_type != 'gpt2':
            raise ValueError('TorchEngine supports GPT-2 checkpoints only')
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        model = GPT2LMHeadModel.from_pretrained(str(path), local_files_only=True, config=config,
                                              attn_implementation='eager')
        from .engine import checkpoint_identity
        return cls(model, tokenizer, model_identity=checkpoint_identity(path), device=engine.device)

    def encode(self, prompts):
        import torch
        encoded = [self.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
        if any(not row for row in encoded):
            raise ValueError('GPT-2 prompts must contain at least one token')
        if not encoded:
            raise ValueError('Cannot forward an empty batch')
        width = max(map(len, encoded))
        if width > self.model.config.n_positions:
            raise ValueError('Prompt exceeds GPT-2 context length')
        ids = torch.full((len(encoded), width), self.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(encoded):
            ids[index, :len(row)] = torch.tensor(row)
            mask[index, :len(row)] = 1
        return ids.to(self.model.device), mask.to(self.model.device)

    @contextmanager
    def hooks(self, *, capture=(), edits=None, detach=True):
        """Coordinates are block-input/output residuals before final ln_f."""
        captures = {}
        handles = []
        edits = edits or {}
        try:
            for site, layer in set(capture) | set(edits):
                if site not in {'resid_pre', 'resid_post'} or type(layer) is not int or not 0 <= layer < len(self.model.transformer.h):
                    raise ValueError('Unsupported GPT-2 residual site/layer')
                block = self.model.transformer.h[layer]
                def apply(value, key=(site, layer)):
                    if key in edits:
                        updated = edits[key](value)
                        if updated.shape != value.shape or updated.dtype != value.dtype or updated.device != value.device:
                            raise ValueError('Residual edit changed shape, dtype or device')
                        value = updated
                    if key in capture:
                        captures[key] = value.detach().clone() if detach else value
                    return value
                if site == 'resid_pre':
                    def pre(module, args, apply=apply):
                        return (apply(args[0]), *args[1:])
                    handles.append(block.register_forward_pre_hook(pre))
                else:
                    def post(module, args, output, apply=apply):
                        if isinstance(output, tuple):
                            return (apply(output[0]), *output[1:])
                        return apply(output)
                    handles.append(block.register_forward_hook(post))
            yield captures
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, input_ids=None, attention_mask=None, *, inputs_embeds=None,
                capture=(), edits=None, differentiable=False):
        import torch
        if self.closed:
            raise RuntimeError('TorchSession is closed')
        if attention_mask is None:
            shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
            attention_mask = torch.ones(shape, dtype=torch.long, device=self.model.device)
        input_ids = input_ids.to(self.model.device) if input_ids is not None else None
        inputs_embeds = inputs_embeds.to(self.model.device) if inputs_embeds is not None else None
        attention_mask = attention_mask.to(self.model.device)
        if attention_mask.ndim != 2 or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError('Attention mask must be a two-dimensional binary tensor')
        if torch.any(attention_mask.sum(-1) == 0):
            raise ValueError('Every row must have at least one non-padding token')
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        with self._lock, torch.set_grad_enabled(differentiable), self.hooks(capture=capture, edits=edits, detach=not differentiable) as captured:
            output = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                position_ids=position_ids, use_cache=False, return_dict=True)
            return output.logits, captured

    def greedy(self, prompt, *, max_tokens, target_token_ids=(), capture=(), edits=None):
        """Prompt-coordinate edits persist as context; generated coordinates are untouched."""
        import torch
        ids, mask = self.encode([prompt])
        prompt_count = ids.shape[1]
        if prompt_count + max_tokens > self.model.config.n_positions:
            raise ValueError('Prompt plus generation budget exceeds GPT-2 context length')
        vocab = self.model.config.vocab_size
        if any(type(token) is not int or not 0 <= token < vocab for token in target_token_ids):
            raise ValueError('Target token ID is outside the resolved vocabulary')
        logits, captured = self.forward(ids, mask, capture=capture, edits=edits)
        targets = {str(token): float(torch.log_softmax(logits[0, -1], dim=-1)[token]) for token in target_token_ids}
        generated = []
        finish = 'length'
        for index in range(max_tokens):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == self.tokenizer.eos_token_id:
                finish = 'stop'
                break
            if index + 1 < max_tokens:
                ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
                mask = torch.ones_like(ids)
                logits, _ = self.forward(ids, mask, edits=edits)
        row = {'generated_text': self.tokenizer.decode(generated), 'generated_token_ids':generated,
               'prompt_token_count':prompt_count, 'generated_token_count':len(generated), 'finish_reason':finish}
        if target_token_ids:
            row['target_logprobs'] = targets
        return row, captured

    def reset(self):
        """No KV cache or persistent hooks; release incidental parameter gradients."""
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def close(self):
        if not self.closed:
            self.reset()
            self.closed = True
            self.model = None
            self.tokenizer = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

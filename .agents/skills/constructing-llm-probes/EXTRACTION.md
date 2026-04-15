# Hidden State Extraction

## Basic extraction

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model.eval()

with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # outputs.hidden_states: tuple of (num_layers + 1) tensors, each [batch, seq_len, hidden_dim]
    # Index 0 = embedding output, index i+1 = layer i output
    layer_hidden = outputs.hidden_states[layer_idx + 1]
```

## Batched extraction with DataLoader

For large datasets, use batched inference with proper padding and unpadding:

```python
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",  # or "max_length" with max_length=N for fixed shapes
    return_tensors="pt",
)

def collate_fn(examples):
    feats = [{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]}
             for ex in examples]
    batch = collator(feats)
    batch["lengths"] = batch["attention_mask"].sum(dim=1, dtype=torch.long)
    return batch

dl = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
layer_hiddens = []

with torch.inference_mode():
    for batch in dl:
        lengths = batch.pop("lengths").tolist()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, output_hidden_states=True, return_dict=True, use_cache=False)
        h = outputs.hidden_states[layer_idx + 1].detach().cpu()  # [B, T, H]
        for i, L in enumerate(lengths):
            if tokenizer.padding_side == "left":
                layer_hiddens.append(h[i, -L:].contiguous())
            else:
                layer_hiddens.append(h[i, :L].contiguous())
```

## Memory management

- Use `torch.inference_mode()` (stricter than `no_grad`, disables version tracking)
- Call `.detach().cpu()` immediately on extracted tensors
- Process one layer at a time for large models; don't keep all layers in memory
- For very large models, use `torch.float16` or `torch.bfloat16`:
  ```python
  model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
  ```
- Consider `accelerate` for multi-GPU or `device_map="auto"` for model sharding

## Getting layer count

```python
num_layers = model.config.num_hidden_layers
# For composite models (e.g., Gemma-3 multimodal):
# text_cfg = model.config.get_text_config()
# num_layers = text_cfg.num_hidden_layers
```

## Chat-template-aware extraction

When probing chat/instruction-tuned models, apply the chat template but track which tokens correspond to your actual input text vs template tokens:

```python
rendered = tokenizer.apply_chat_template(
    [{"role": "user", "content": text}],
    tokenize=False,
    add_generation_prompt=False,
)
# Find where your text appears in the rendered string
start = rendered.find(text)
end = start + len(text)

# Tokenize with offset_mapping to map token positions to character positions
enc = tokenizer(rendered, return_offsets_mapping=True, return_special_tokens_mask=True)

# Keep only tokens that fall within [start, end) of the original text
keep = [(not sp) and (start <= s) and (e <= end)
        for (s, e), sp in zip(enc["offset_mapping"], enc["special_tokens_mask"])]
```

## Extracting from specific hook points

For more control (e.g., post-attention vs post-MLP), use PyTorch hooks:

```python
activations = {}

def hook_fn(name):
    def fn(module, input, output):
        activations[name] = output.detach().cpu()
    return fn

# Register hooks on specific submodules
for i, layer in enumerate(model.transformer.h):  # GPT-2 style
    layer.attn.register_forward_hook(hook_fn(f"layer_{i}_attn"))
    layer.mlp.register_forward_hook(hook_fn(f"layer_{i}_mlp"))

with torch.no_grad():
    model(**inputs)

# activations["layer_5_attn"] now has post-attention output for layer 5
```

## Residual stream vs sublayer outputs

- **Residual stream** (`hidden_states[i]`): the full representation after layer i, used by most probing work
- **Attention output**: what the attention sublayer adds to the residual stream
- **MLP output**: what the MLP sublayer adds

For standard probing, use residual stream outputs. Use sublayer hooks when you need to attribute information to specific components.

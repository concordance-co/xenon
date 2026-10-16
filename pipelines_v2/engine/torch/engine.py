"""Narrow serialized GPT-2 CPU adapter with prompt-coordinate residual edits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from pipelines_v2.core.types import EngineCapability, SpecValidationError, stable_hash
from pipelines_v2.engine.base import EngineCaptureResult, EngineGenerationResult, EngineInterventionResult, PythonRuntimeSpec
from pipelines_v2.operations.capture import ResidualSite
from pipelines_v2.operations.interventions.runtime import (load_activation_bank_source, load_direction_source,
    resolve_generation_examples, resolve_patched_generation_cases, resolve_patched_generation_targets)
from .session import ADAPTER_VERSION, TorchSession


def checkpoint_identity(path):
    """Actual local checkpoint bytes; path labels do not authorize donor reuse."""
    root = Path(path)
    names = ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json')
    files = []
    for name in names:
        item = root / name
        if item.is_file():
            digest = hashlib.sha256()
            with item.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024), b''):
                    digest.update(block)
            files.append((name,digest.hexdigest()))
    if not any(name == 'model.safetensors' for name,_ in files):
        raise ValueError('GPT-2 adapter requires the prepared model.safetensors checkpoint')
    return stable_hash(files)


@dataclass(frozen=True, slots=True)
class TorchEngine:
    model_id: str
    revision: str
    device: str = 'cpu'
    dtype: str = 'float32'
    attention_impl: str = 'eager'
    kind: str = 'torch'

    def __post_init__(self):
        if self.kind != 'torch' or self.device not in {'cpu','cuda'} or self.dtype != 'float32' or self.attention_impl != 'eager':
            raise ValueError('TorchEngine supports GPT-2 CPU/CUDA float32 eager only')
        if not re.fullmatch(r'[a-f0-9]{40}', self.revision):
            raise ValueError('TorchEngine needs an immutable 40-character checkpoint revision')

    def identity(self):
        return {'kind':self.kind,'model_id':self.model_id,'revision':self.revision,'device':self.device,
                'dtype':self.dtype,'attention_impl':self.attention_impl,'adapter_version':ADAPTER_VERSION}

    def semantic_identity(self):
        return self.identity()

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if payload.pop('adapter_version', ADAPTER_VERSION) != ADAPTER_VERSION:
            raise ValueError('Unsupported Torch adapter version')
        return cls(**payload)

    def capabilities(self):
        return {EngineCapability.GENERATION,EngineCapability.LOGPROBS,EngineCapability.TARGET_LOGPROBS,
                EngineCapability.RESIDUAL_CAPTURE,EngineCapability.ACTIVATION_PATCHING,EngineCapability.REQUEST_SCOPED_INTERVENTIONS}

    def supported_patch_operators(self):
        return frozenset({'interchange','add_direction','scale'})

    def runtime_spec(self):
        from pipelines_v2.runtime.locks import core_runtime_lock, load_runtime_lock
        return PythonRuntimeSpec(lock=load_runtime_lock('core-linux') if self.device == 'cuda' else core_runtime_lock(), local_python_sources=('pipelines_v2',),
                                 env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})

    def resolved_model_path(self):
        return str(Path(self.model_id).expanduser())

    def scratch_token_counts(self, examples):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.resolved_model_path(),local_files_only=True)
        return [len(tokenizer.encode(example.prompt,add_special_tokens=False)) for example in examples]

    def open_session(self):
        return TorchSession.from_engine(self)

    def planning_errors(self, spec, *, check_assets=True):
        errors = []
        if check_assets and not Path(self.model_id).expanduser().is_dir():
            errors.append('Prepared offline GPT-2 cache missing; download checkpoint separately')
        generation = getattr(spec,'generation',None)
        if generation is not None:
            if generation.temperature != 0 or generation.top_p != 1 or generation.top_k != -1:
                errors.append('Torch GPT-2 currently supports deterministic greedy generation only')
            if generation.target_token_ids and not generation.enabled:
                errors.append('Target logprob capture requires enabled generation')
            if generation.max_tokens is None:
                errors.append('Torch generation requires a finite max_tokens budget')
            if generation.capture_generated_tokens or generation.capture_reasoning or generation.chat_tools or generation.tool_choice is not None:
                errors.append('Torch GPT-2 supports plain text prompt-side capture only')
        if getattr(spec,'prompt_metadata_builder',None) is not None:
            errors.append('Torch GPT-2 uses explicit example token_sections; custom metadata builders are unsupported')
        for site in getattr(spec,'sites',()):
            if not isinstance(site,ResidualSite) or site.site not in {'resid_pre','resid_post'}:
                errors.append('Torch GPT-2 only supports resid_pre/resid_post block sites')
            elif site.pooling is not None:
                errors.append('Torch GPT-2 capture requires unpooled tokenwise residuals')
            elif not site.layers or any(type(layer) is not int or layer < 0 for layer in site.layers):
                errors.append('Torch GPT-2 capture layers must be nonnegative integers')
        patch = getattr(spec,'patch',None)
        if patch is not None:
            if patch.operator not in self.supported_patch_operators():
                errors.append('Unsupported Torch patch operator')
            if patch.write_site.site not in {'resid_pre','resid_post'}:
                errors.append('Unsupported Torch patch residual site')
            if patch.application.kind != 'static' or not patch.application.include_prompt or patch.application.include_decode:
                errors.append('Torch patches support static prompt positions only, never decode edits')
            if not math.isfinite(patch.strength):
                errors.append('Torch patch strength must be finite')
            if patch.operator == 'interchange' and patch.strength != 1:
                errors.append('Torch interchange is full-state replacement and requires strength=1')
            if patch.operator == 'add_direction' and (patch.uses_subspace() or patch.component_indices_by_layer):
                errors.append('Torch add_direction supports raw full-width directions only')
        if any(getattr(example,'images',()) for example in getattr(getattr(spec,'dataset',None),'examples',())):
            errors.append('Torch GPT-2 accepts text only')
        return tuple(errors)

    def _validate(self, spec):
        errors = self.planning_errors(spec)
        if errors:
            raise SpecValidationError('; '.join(errors))

    def capture(self, spec, *, session=None, **_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:
                return self.capture(spec, session=owned)
        requests = tuple((site.site,int(layer)) for site in spec.sites for layer in site.layers)
        features = {site.name:{'kind':'residual','site':site.site,'model_identity':session.model_identity,
                    'adapter_version':ADAPTER_VERSION,'storage':{'dtype':site.storage.dtype,'format':site.storage.format},
                    'layers':{str(layer):{} for layer in site.layers}} for site in spec.sites}
        generations, metadata = [], []
        for example in spec.dataset.examples:
            row, captured = session.greedy(example.prompt,max_tokens=int(spec.generation.max_tokens or 0) if spec.generation.enabled else 0,
                                           target_token_ids=spec.generation.target_token_ids,capture=requests)
            count = row['prompt_token_count']
            sections = example.metadata.get('token_sections',{})
            for site in spec.sites:
                tokens = site.tokens.resolve(count,token_sections=sections)
                for layer in site.layers:
                    values = captured[site.site,int(layer)][0,tokens].cpu().numpy()
                    features[site.name]['layers'][str(layer)][example.key] = {'tokens':tokens,'values':values,
                        'token_count':count,'token_sections':sections}
            metadata.append({'example_key':example.key,'token_count':count,'prompt_token_count':count})
            if spec.generation.enabled:
                generations.append({'example_key':example.key,**row})
        return EngineCaptureResult(features=features,generations=generations,
            metadata={'backend':'torch','adapter_version':ADAPTER_VERSION,'model_identity':session.model_identity,
                      'example_metadata':metadata,'tokenizer':'gpt2-offline'})

    def generate(self, spec, *, session=None, **_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:
                return self.generate(spec,session=owned)
        rows=[]
        for example in resolve_generation_examples(spec):
            row,_ = session.greedy(example.prompt,max_tokens=spec.generation.max_tokens,
                                   target_token_ids=spec.generation.target_token_ids)
            rows.append({'example_key':example.key,'example':example.to_dict(),**row})
        return EngineGenerationResult(rows=rows,metadata={'backend':'torch','adapter_version':ADAPTER_VERSION})

    def intervene(self, spec, *, session=None, **_):
        self._validate(spec)
        if session is None:
            with self.open_session() as owned:
                return self.intervene(spec,session=owned)
        patch = spec.patch
        bank = load_activation_bank_source(patch) if patch.operator == 'interchange' else None
        direction = load_direction_source(patch) if patch.operator == 'add_direction' else None
        if bank is not None:
            if bank.get('model_identity') != session.model_identity or bank.get('adapter_version') != ADAPTER_VERSION:
                raise ValueError('Donor activation bank must match actual checkpoint and adapter identity')
            cases, skipped = resolve_patched_generation_cases(spec)
            if skipped:
                raise ValueError('Torch interchange requires complete explicit donor/recipient pairing')
        else:
            cases = [{'target':target} for target in resolve_patched_generation_targets(spec)]
        rows=[]
        for case in cases:
            example = case['target']
            ids,_ = session.encode([example.prompt])
            positions = patch.target_tokens.resolve(ids.shape[1],token_sections=example.metadata.get('token_sections',{}))
            edits = {}
            for layer in patch.write_site.layers:
                source_layer = patch.source_layer_for(layer)
                if bank is not None:
                    donor = bank['layers'][str(source_layer)].get(case['donor'].key)
                    if donor is None or 'tokens' not in donor:
                        raise ValueError('Donor bank lacks explicit token coordinates')
                    selected = (patch.donor_tokens or patch.target_tokens).resolve(donor['token_count'],token_sections=donor.get('token_sections',{}))
                    if len(selected) != len(positions) or len(set(donor['tokens'])) != len(donor['tokens']):
                        raise ValueError('Donor and target token alignment mismatch')
                    try:
                        values = [donor['values'][donor['tokens'].index(token)] for token in selected]
                    except (ValueError,IndexError) as exc:
                        raise ValueError('Donor bank is missing requested token states') from exc
                elif direction is not None:
                    values = direction['layers'][str(source_layer)]['raw_vector']
                else:
                    values = None
                edits[patch.write_site.site,int(layer)] = residual_edit(patch.operator,positions,values=values,
                    scale=getattr(patch,'scale',None),strength=patch.strength,hidden_size=session.model.config.n_embd)
            row,_ = session.greedy(example.prompt,max_tokens=spec.generation.max_tokens,
                                   target_token_ids=spec.generation.target_token_ids,edits=edits)
            rows.append({'example_key':example.key,'example':example.to_dict(),'status':'ok',
                         'case_key':case.get('case_key',example.key),'donor_example_key':case.get('donor').key if case.get('donor') else None,
                         'target_tokens':positions,'patch_stats':{'operator':patch.operator,'layers':list(patch.write_site.layers)},**row})
        return EngineInterventionResult(summary={'example_count':len(rows),'patched_count':len(rows)},rows=rows,
                                        metadata={'backend':'torch','adapter_version':ADAPTER_VERSION})


def residual_edit(operator, positions, *, values=None, scale=None, strength=1., hidden_size):
    import torch
    positions = tuple(positions)
    value = None if values is None else torch.as_tensor(values,dtype=torch.float32)
    expected = (len(positions),hidden_size) if operator == 'interchange' else (hidden_size,)
    if value is not None and (tuple(value.shape) != expected or not bool(value.isfinite().all())):
        raise ValueError('Residual donor/direction shape or finite-value mismatch')
    def apply(hidden):
        output = hidden.clone()
        device_value = None if value is None else value.to(device=hidden.device, dtype=hidden.dtype)
        if operator == 'interchange':
            output[:,positions,:] = device_value
        elif operator == 'add_direction':
            output[:,positions,:] = hidden[:,positions,:] + strength * device_value
        elif operator == 'scale':
            output[:,positions,:] = hidden[:,positions,:] * scale
        else:
            raise ValueError('Unsupported residual edit')
        return output
    return apply
